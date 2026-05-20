"""阶段三：SADA 训练与评估管理器。"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from threadpoolctl import threadpool_limits
from models import CLBert
from utils.build_ml import build_batch_whole_word_token_spans
from utils.cluster_refinement import ClusterRefiner
from utils.contrastive import PrototypeContrastiveLoss
from utils.memory import PrototypeManager
from utils.output_paths import OutputPaths
from utils.parallel import (
    DistributedWeightedSampler,
    barrier,
    gather_features_labels,
    gather_tensor,
    get_model_backbone,
    is_main_process,
    rank0_broadcast_object,
    unwrap_parallel_model,
    wrap_model,
)
from utils.tools import clustering_score, hungray_aligment, set_seed, view_generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SADAModelManager:
    """管理双视图对比学习、原型刷新、评估与结果保存。"""

    def __init__(self, args, data, pretrained_model=None, student_model=None):
        """初始化阶段三训练所需模型、增强器与状态。"""
        set_seed(args.seed)
        if getattr(args, "distributed", False):
            self.device = args.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.model = CLBert(
            args.bert_model,
            device=self.device,
            num_labels=self.num_labels,
            feat_dim=args.feat_dim,
        )
        self.model = wrap_model(self.model, self.device, distributed=getattr(args, "distributed", False))

        if not args.disable_pretrain:
            self.pretrained_model = pretrained_model
            self.load_pretrained_model()

        self.num_train_optimization_steps = int(
            len(data.train_semi_dataset) / args.train_batch_size
        ) * args.num_train_epochs
        self.student_model = student_model
        self.optimizer, self.scheduler = self.get_optimizer(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.bracket_left_id = self.tokenizer.convert_tokens_to_ids("[")
        self.bracket_right_id = self.tokenizer.convert_tokens_to_ids("]")
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        self.output_paths = OutputPaths(args)
        # 传递已知类别标签列表给 ClusterRefiner，用于构建 LLM 提示词
        # epoch 会在 refresh_unknown_structure 时更新
        self.cluster_refiner = ClusterRefiner(args, known_label_list=data.known_label_list, epoch=-1)
        self.latest_cluster_packets = []
        self.train_dataloader = self.build_train_dataloader(args, data)

        # 追踪训练过程中的最佳评估结果
        self.best_eval_results = None
        # evaluation() 会在运行中更新该字段；先初始化避免首轮访问时报 AttributeError
        self.test_results = None


    def build_train_dataloader(self, args, data):
        """构建兼顾已标注与未标注样本的训练加载器。"""
        all_labels = data.train_semi_dataset.tensors[3].numpy()
        labeled_mask = all_labels != -1
        unlabeled_mask = all_labels == -1
        n_labeled = labeled_mask.sum()

        if n_labeled == 0:
            return data.train_semi_train_dataloader

        known_labels = all_labels[labeled_mask]
        class_counts = np.bincount(known_labels, minlength=data.n_known_cls) + 1e-6
        class_weights = n_labeled / (len(class_counts) * class_counts)
        weights = np.zeros_like(all_labels, dtype=np.float32)
        weights[labeled_mask] = class_weights[all_labels[labeled_mask]]
        weights[unlabeled_mask] = 1.0
        weights_tensor = torch.FloatTensor(weights)

        if getattr(args, "distributed", False):
            sampler = DistributedWeightedSampler(weights_tensor, replacement=True, seed=args.seed)
        else:
            sampler = WeightedRandomSampler(weights_tensor.tolist(), num_samples=len(weights), replacement=True)

        return DataLoader(
            data.train_semi_dataset,
            batch_size=args.train_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def get_global_features_labels(self, dataloader, model, args):
        """获取全局特征与标签（所有 rank 均参与，结果各自一致）。

        分布式模式下 get_features_labels 内部已调用 gather_features_labels，
        每个 rank 执行后本地即持有全量数据，无需额外 broadcast。
        """
        features, labels = self.get_features_labels(dataloader, model, args)
        return features, labels

    def build_adjacency_mask(self, targets, pseudo_labels=None, pseudo_conf=None,
                             data_inds=None, assignment_targets=None, assignment_confidence=None):
        """根据标签构建对比学习邻接掩码，包含已标注样本的硬正例与无标签样本的软正例。

        软正例来源：
        1. pseudo_labels >= 0 的无标签样本视为已知类伪标签，与同类样本（含有标签）建立正连接；
        2. assignment_targets >= 0 的无标签样本与同 cluster 其他样本建立正连接。

        当前实验设置下，无标签连接统一二值化：只要连边成立，权重即为 1。
        历史的置信度加权逻辑已保留为注释（见函数内部 NOTE），便于快速回滚。

        Args:
            targets:               当前 batch 真实标签 [batch_size]，-1 表示无标签。
            pseudo_labels:         无标签样本已知类路由结果 [batch_size]，-1 表示未路由。
            pseudo_conf:           对应置信度 [batch_size]，0 表示无效。
            data_inds:             当前 batch 样本在 train_semi_dataset 中的全局索引（已不使用，保留兼容）。
            assignment_targets:    无标签样本 unknown 路由结果 [batch_size]，全局原型索引，-1 表示未路由。
            assignment_confidence: 对应置信度 [batch_size]，0 表示无效。
        Returns:
            adjacency: [batch_size, batch_size] 的软邻接矩阵，值域 [0, 1]。
        """
        batch_size = targets.size(0)
        adjacency = torch.eye(batch_size, device=self.device)

        # ── 1. 有标签样本：硬正例（原逻辑不变） ────────────────────────
        labeled_mask = targets >= 0
        if labeled_mask.any():
            labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).squeeze(-1).to(self.device)
            labeled_targets = targets[labeled_indices]
            same_class = torch.eq(labeled_targets.unsqueeze(0), labeled_targets.unsqueeze(1)).float()
            adjacency[labeled_indices.unsqueeze(1), labeled_indices.unsqueeze(0)] = same_class
            adjacency.fill_diagonal_(1.0)

        # ── 2. pseudo_labels（已知类路由）软正例 ────────────────────────
        # 策略：将 pseudo 样本与同类（真实有标签 OR 同 pseudo 类）样本建立软正例。
        if pseudo_labels is not None and pseudo_conf is not None:
            pseudo_mask = pseudo_labels >= 0          # [batch_size]，bool
            if pseudo_mask.any():
                # 索引必须在正确 device 上
                pseudo_indices = torch.nonzero(pseudo_mask, as_tuple=False).squeeze(-1).to(self.device)
                p_labels = pseudo_labels[pseudo_indices]    # [n_pseudo]
                p_confs  = pseudo_conf[pseudo_indices]      # [n_pseudo], 已在 self.device 上

                # pseudo 样本之间：同类建立二值连接（值=1）
                same_pseudo = torch.eq(p_labels.unsqueeze(0), p_labels.unsqueeze(1)).float()
                # NOTE:
                # 按当前实验需求，adjacency 对无标签样本改为二值连接：
                # 只要满足同类关系（有值）就置为 1，不再使用置信度连续权重。
                # 保留旧逻辑（注释）以便后续快速恢复：
                # conf_weight = torch.outer(p_confs, p_confs)
                # soft_pp = same_pseudo * conf_weight  # [n_pseudo, n_pseudo]
                soft_pp = same_pseudo
                adjacency[
                    pseudo_indices.unsqueeze(1),
                    pseudo_indices.unsqueeze(0),
                ] = torch.maximum(
                    adjacency[pseudo_indices.unsqueeze(1), pseudo_indices.unsqueeze(0)],
                    soft_pp,
                )

                # pseudo 样本 与 有标签样本：同类建立二值连接（值=1）
                if labeled_mask.any():
                    labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).squeeze(-1).to(self.device)
                    labeled_targets = targets[labeled_indices]
                    # 比较 pseudo 类与有标签类是否相同：[n_pseudo, n_labeled]
                    same_pl = torch.eq(p_labels.unsqueeze(1), labeled_targets.unsqueeze(0)).float()
                    # NOTE:
                    # 当前改为二值连接：只要同类即为 1。
                    # 保留旧逻辑（注释）：
                    # conf_pl = p_confs.unsqueeze(1) * same_pl  # [n_pseudo, n_labeled]
                    conf_pl = same_pl
                    # 写入 pseudo→labeled 方向
                    adjacency[
                        pseudo_indices.unsqueeze(1),
                        labeled_indices.unsqueeze(0),
                    ] = torch.maximum(
                        adjacency[pseudo_indices.unsqueeze(1), labeled_indices.unsqueeze(0)],
                        conf_pl,
                    )
                    # 写入 labeled→pseudo 方向（对称）
                    adjacency[
                        labeled_indices.unsqueeze(1),
                        pseudo_indices.unsqueeze(0),
                    ] = torch.maximum(
                        adjacency[labeled_indices.unsqueeze(1), pseudo_indices.unsqueeze(0)],
                        conf_pl.T,
                    )

        # ── 3. assignment_targets（未知类路由）软正例 ───────────────────
        # 使用当前 batch 实时路由结果，不再依赖跨 batch 的全局缓存。
        # 策略：同 unknown cluster 的无标签样本互为二值正例（值=1）。
        if assignment_targets is not None and assignment_confidence is not None:
            unlabeled_mask = targets < 0
            # 已被 pseudo 路由的样本不参与 unknown 软正例（避免双重计数）
            if pseudo_labels is not None:
                unlabeled_mask = unlabeled_mask & (pseudo_labels < 0)
            if unlabeled_mask.any():
                ul_indices = torch.nonzero(unlabeled_mask, as_tuple=False).squeeze(-1).to(self.device)
                cluster_ids  = assignment_targets[ul_indices]     # 全局原型索引，-1 表示未路由
                cluster_conf = assignment_confidence[ul_indices]

                assigned_mask = cluster_ids >= 0
                if assigned_mask.sum() >= 2:
                    a_local = ul_indices[assigned_mask]
                    a_clust = cluster_ids[assigned_mask]
                    a_conf  = cluster_conf[assigned_mask]

                    same_cluster = torch.eq(
                        a_clust.unsqueeze(0), a_clust.unsqueeze(1)
                    ).float()
                    # NOTE:
                    # 按当前实验需求，unknown 路由的软正例也改为二值连接。
                    # 保留旧逻辑（注释）以便回退：
                    # conf_weight = torch.outer(a_conf, a_conf)
                    # soft_uc = same_cluster * conf_weight
                    soft_uc = same_cluster

                    adjacency[
                        a_local.unsqueeze(1),
                        a_local.unsqueeze(0),
                    ] = torch.maximum(
                        adjacency[a_local.unsqueeze(1), a_local.unsqueeze(0)],
                        soft_uc,
                    )

        adjacency.fill_diagonal_(1.0)
        return adjacency

    def _build_routing_stats(self, unlabeled_total, gate_known, gate_unknown, best_known_sim,
                             known_conf, best_unknown_sim, unknown_conf):
        """构建路由可观测性统计。"""
        known_passed = int(gate_known.sum().item()) if gate_known.numel() > 0 else 0
        unknown_passed = int(gate_unknown.sum().item()) if gate_unknown.numel() > 0 else 0
        passed_total = known_passed + unknown_passed

        return {
            "mode": "hard_fixed",
            "unlabeled_total": int(unlabeled_total),
            "passed_total": passed_total,
            "known_passed": known_passed,
            "unknown_passed": unknown_passed,
            "mean_known_sim": float(best_known_sim.mean().item()) if best_known_sim.numel() > 0 else 0.0,
            "mean_known_conf": float(known_conf.mean().item()) if known_conf.numel() > 0 else 0.0,
            "mean_unknown_sim": float(best_unknown_sim.mean().item()) if best_unknown_sim.numel() > 0 else 0.0,
            "mean_unknown_conf": float(unknown_conf.mean().item()) if unknown_conf.numel() > 0 else 0.0,
            "routed_known_sim": float(best_known_sim[gate_known].mean().item()) if gate_known.any() else 0.0,
            "routed_known_conf": float(known_conf[gate_known].mean().item()) if gate_known.any() else 0.0,
            "routed_unknown_sim": float(best_unknown_sim[gate_unknown].mean().item()) if gate_unknown.any() else 0.0,
            "routed_unknown_conf": float(unknown_conf[gate_unknown].mean().item()) if gate_unknown.any() else 0.0,
        }

    def route_unlabeled_targets(self, features, targets, data_inds, proto_manager, data, args, epoch=0):
        """为无标签样本执行 known/unknown 独立门控与竞争路由。"""
        device = features.device
        batch_size = targets.size(0)
        pseudo_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        pseudo_conf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        assignment_targets = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        assignment_confidence = torch.zeros(batch_size, dtype=torch.float32, device=device)

        labeled_mask = targets >= 0
        unlabeled_mask = ~labeled_mask

        default_stats = {
            "mode": "hard_fixed",
            "unlabeled_total": int(unlabeled_mask.sum().item()),
            "passed_total": 0,
            "known_passed": 0,
            "unknown_passed": 0,
            "mean_known_sim": 0.0,
            "mean_known_conf": 0.0,
            "mean_unknown_sim": 0.0,
            "mean_unknown_conf": 0.0,
            "routed_known_sim": 0.0,
            "routed_known_conf": 0.0,
            "routed_unknown_sim": 0.0,
            "routed_unknown_conf": 0.0,
        }

        # 无 unlabeled 样本或无已知类时，直接返回空路由
        if unlabeled_mask.sum() == 0 or data.n_known_cls <= 0:
            return pseudo_labels, pseudo_conf, assignment_targets, assignment_confidence, default_stats

        all_prototypes = proto_manager.get_prototypes()  # [num_classes, feat_dim]
        n_known = data.n_known_cls
        num_all = all_prototypes.size(0)

        # 已知原型必须已初始化才参与路由
        if not proto_manager.initialized[:n_known].any():
            return pseudo_labels, pseudo_conf, assignment_targets, assignment_confidence, default_stats

        unlabeled_features = features[unlabeled_mask]  # [n_ul, feat_dim]
        sim_matrix = torch.matmul(unlabeled_features, all_prototypes.T)  # [n_ul, num_classes]

        # --- 1. known 分支：子空间内独立计算 sim 与 conf ---
        known_sim = sim_matrix[:, :n_known]
        best_known_sim, best_known_idx = torch.max(known_sim, dim=1)
        if n_known >= 2:
            known_second_sim = torch.topk(known_sim, k=2, dim=1).values[:, 1]
        else:
            known_second_sim = torch.zeros_like(best_known_sim)
        known_conf = best_known_sim - known_second_sim

        # --- 2. unknown 分支：置信度 = 跨空间 Gap (Unknown - Known) ---
        n_unknown_protos = num_all - n_known
        if n_unknown_protos <= 0:
            return pseudo_labels, pseudo_conf, assignment_targets, assignment_confidence, default_stats

        unknown_sim = sim_matrix[:, n_known:]
        n_unknown = unknown_sim.size(1)

        # 初始化或重置频率追踪器
        if getattr(self, "cluster_freq", None) is None or self.cluster_freq.size(0) != n_unknown:
            self.cluster_freq = torch.ones(n_unknown, device=features.device) / n_unknown

        # 频率惩罚仅基于历史频率追踪器 self.cluster_freq（取消当前轮频率参与）。
        effective_freq = self.cluster_freq

        # 1. 施加垄断惩罚，计算调整后的相似度
        margin_scale = getattr(args, "margin_scale", 2.5)
        margin = margin_scale * (1.0 / n_unknown)
        penalty_weight = getattr(args, "balance_penalty", 0.4)
        penalty = penalty_weight * F.relu(effective_freq - margin)
        adjusted_unknown_sim = unknown_sim - penalty

        # 2. 在【惩罚后的空间】寻找最优簇（强制均贫富）
        best_adjusted_sim, best_unknown_local_idx = torch.max(adjusted_unknown_sim, dim=1)
        best_unknown_idx = best_unknown_local_idx + n_known
        raw_best_unknown_sim, raw_best_unknown_local_idx = torch.max(unknown_sim, dim=1)
        kicked_from_dominant = best_unknown_local_idx != raw_best_unknown_local_idx

        # 3. 提取【原始空间】的相似度（用于真实特征门控）
        best_unknown_sim = unknown_sim.gather(1, best_unknown_local_idx.unsqueeze(1)).squeeze(1)
        max_original_unknown_sim = raw_best_unknown_sim.clone()
        if kicked_from_dominant.any():
            # 若样本被大簇“踢出”，则后续比较不再考虑该大簇。
            masked_original_sim = unknown_sim.clone()
            row_idx = torch.nonzero(kicked_from_dominant, as_tuple=False).squeeze(-1)
            col_idx = raw_best_unknown_local_idx[kicked_from_dominant]
            masked_original_sim[row_idx, col_idx] = -float('inf')
            max_after_drop = torch.max(masked_original_sim, dim=1).values
            max_original_unknown_sim = torch.where(
            kicked_from_dominant,
                max_after_drop,
                max_original_unknown_sim,
            )

        # 4. 计算内部 Gap：【关键修复B】必须在【惩罚后的空间】计算，确保改判样本的 Conf 为正！
        if n_unknown >= 2:
            masked_adj_sim = adjusted_unknown_sim.clone()
            masked_adj_sim.scatter_(1, best_unknown_local_idx.unsqueeze(1), -float('inf'))
            if kicked_from_dominant.any():
                row_idx = torch.nonzero(kicked_from_dominant, as_tuple=False).squeeze(-1)
                col_idx = raw_best_unknown_local_idx[kicked_from_dominant]
                masked_adj_sim[row_idx, col_idx] = -float('inf')
            adjusted_second_sim = torch.max(masked_adj_sim, dim=1).values
            adjusted_second_sim = torch.where(
                torch.isfinite(adjusted_second_sim),
                adjusted_second_sim,
                best_adjusted_sim,
            )
            unknown_conf = best_adjusted_sim - adjusted_second_sim
        else:
            unknown_conf = torch.zeros_like(best_adjusted_sim)

        # --- 3. 独立门控 ---
        pass_known = (
            (best_known_sim >= args.known_sim_thresh)
            & (known_conf >= args.known_conf_thresh)
        )
        refine_start = int(getattr(args, "refine_start_epoch", 10))
        if epoch < refine_start:
            pass_unknown = (best_unknown_sim >= args.unknown_sim_thresh)
        else:
            pass_unknown = (
                (best_unknown_sim >= args.unknown_sim_thresh)
                & (unknown_conf >= args.unknown_conf_thresh)
            )

        # --- 4. 竞争规则（compete_bias）---
        unknown_wins = (max_original_unknown_sim + args.compete_bias) >= best_known_sim
        gate_known = pass_known & (~pass_unknown | ~unknown_wins)
        gate_unknown = pass_unknown & (~pass_known | unknown_wins)

        gate_combined = gate_known | gate_unknown
        routing_stats = self._build_routing_stats(
            unlabeled_total=unlabeled_features.size(0),
            gate_known=gate_known,
            gate_unknown=gate_unknown,
            best_known_sim=best_known_sim,
            known_conf=known_conf,
            best_unknown_sim=best_unknown_sim,
            unknown_conf=unknown_conf,
        )

        if gate_combined.any():
            unlabeled_indices = torch.nonzero(unlabeled_mask, as_tuple=False).squeeze(-1)

            # 已知类路由
            if gate_known.any():
                passed_known = unlabeled_indices[gate_known]
                passed_known_conf = known_conf[gate_known].float().clamp_min(1e-6)
                pseudo_labels[passed_known] = best_known_idx[gate_known]
                pseudo_conf[passed_known] = passed_known_conf

            # 未知类路由：存全局原型索引
            if gate_unknown.any():
                passed_unknown = unlabeled_indices[gate_unknown]
                passed_unknown_conf = unknown_conf[gate_unknown].float().clamp_min(1e-6)
                assignment_targets[passed_unknown] = best_unknown_idx[gate_unknown]
                assignment_confidence[passed_unknown] = passed_unknown_conf

        return pseudo_labels, pseudo_conf, assignment_targets, assignment_confidence, routing_stats


    def build_input_view(self, args, batch, epoch, visualize=False):
        """根据配置生成单个训练视图。"""
        input_ids, attention_mask, token_type_ids, _, special_tokens_mask = batch[:5]

        # 两套 tensor：
        #   aug_*  — 仅用于 no_grad 的 teacher/student 推理，detach+clone 完全隔离 autograd
        #   view_* — 用于返回给 grad-enabled 的 self.model(view) forward
        # 注意：torch.no_grad() 不隔离 version counter，若两个阶段共享同一 tensor 对象
        # 则 teacher/student 内部的 inplace 操作（如 HF 内部的 position_ids 等）会导致
        # backward 时出现 "modified by an inplace operation" 错误。
        aug_input_ids        = input_ids.detach().clone()
        aug_attention_mask   = attention_mask.detach().clone()
        aug_token_type_ids   = token_type_ids.detach().clone()
        aug_special_tokens_mask = special_tokens_mask.detach().clone()

        view_input_ids       = input_ids.clone()
        view_attention_mask  = attention_mask.clone()
        view_token_type_ids  = token_type_ids.clone()

        if args.view_strategy == "rtr":
            return {
                "input_ids": self.generator.random_token_replace(aug_input_ids.cpu()).to(self.device),
                "attention_mask": view_attention_mask,
                "token_type_ids": view_token_type_ids,
            }

        if args.view_strategy == "shuffle":
            return {
                "input_ids": self.generator.shuffle_tokens(aug_input_ids.cpu()).to(self.device),
                "attention_mask": view_attention_mask,
                "token_type_ids": view_token_type_ids,
            }

        if args.view_strategy == "none":
            return {
                "input_ids": view_input_ids,
                "attention_mask": view_attention_mask,
                "token_type_ids": view_token_type_ids,
            }

        if args.view_strategy != "SADA":
            raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")

        if not self.student_model:
            raise ValueError("Student model not found!")
        if self.mask_token_id is None:
            raise ValueError("Tokenizer does not define a [MASK] token for whole-word masking.")

        student_model = self.student_model
        # aug_* 系列完全 detach，确保 teacher/student 推理不会污染 view_* 系列的 autograd graph
        word_spans = build_batch_whole_word_token_spans(
            aug_input_ids.cpu(),
            aug_special_tokens_mask.cpu(),
            self.tokenizer,
        )
        with torch.no_grad():
            backbone = get_model_backbone(self.model)
            teacher_outputs = backbone(
                input_ids=aug_input_ids,
                attention_mask=aug_attention_mask,
                output_hidden_states=True,
            )
            teacher_hidden = teacher_outputs.hidden_states[-1].detach()
            word_probs, word_valid_mask, _ = student_model(
                teacher_hidden,
                attention_mask=aug_attention_mask,
                special_tokens_mask=aug_special_tokens_mask,
                temperature=0.1,
                noise_ratio=args.noise_ratio,
                word_spans=word_spans,
                return_word_probs=True,
                apply_inference_gate=True,
            )
            word_probs = word_probs.squeeze(-1) * word_valid_mask.squeeze(-1)
            unique_vals = word_probs.unique()
            is_binary = torch.all((unique_vals >= 0) & (unique_vals <= 1)) and unique_vals.numel() <= 2
            if is_binary:
                sampled_word_mask = (word_probs > 0.5).long()
            else:
                sampled_word_mask = torch.bernoulli(word_probs).long()

        # 在 view_input_ids（training forward 专用 clone）上做 masking，不触碰 aug_* 系列
        masked_input_ids = view_input_ids
        for batch_idx, spans in enumerate(word_spans):
            for word_idx, span in enumerate(spans):
                if word_idx >= sampled_word_mask.size(1):
                    continue
                if sampled_word_mask[batch_idx, word_idx].item() == 0:
                    continue
                masked_input_ids[batch_idx, span] = self.mask_token_id

        if visualize:
            print(f"\n[Student Augmentation Visualization - Epoch {epoch + 1}]")
            input_id_sample = aug_input_ids[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_id_sample)
            print(f"{'Word':<20} | {'Noise Prob':<12} | {'Action'}")
            print("-" * 45)
            for word_idx, span in enumerate(word_spans[0]):
                word = self.tokenizer.convert_tokens_to_string([tokens[pos] for pos in span])
                score = word_probs[0, word_idx].item() if word_idx < word_probs.size(1) else 0.0
                action = "MASK" if sampled_word_mask[0, word_idx].item() == 1 else "KEEP"
                print(f"{word:<20} | {score:.4f}       | {action}")
            print("-" * 45 + "\n")

        return {
            "input_ids": masked_input_ids,
            "attention_mask": view_attention_mask,
            "token_type_ids": view_token_type_ids,
        }

    def build_dual_views(self, args, batch, epoch, visualize=False):
        """为同一样本生成两份增强视图。

        SADA 模式下：
          view_one — student model 噪声遮蔽（语义感知掩码）
          view_two — 原始输入（不做任何增强），作为干净锚点

        非 SADA 模式：两个视图均由 build_input_view 生成（行为与原来相同）。
        """
        view_one = self.build_input_view(args, batch, epoch, visualize=visualize)

        if args.view_strategy == "SADA":
            # view_two 保持原始 token，作为干净锚点与 view_one 形成对比
            view_two = {
                "input_ids": batch[0].clone(),
                "attention_mask": batch[1].clone(),
                "token_type_ids": batch[2].clone(),
            }
        else:
            view_two = self.build_input_view(args, batch, epoch, visualize=False)

        return view_one, view_two

    def extract_unlabeled_features(self, data, args):
        """提取未标注样本的特征、索引与样本对象列表。

        只在 rank 0 独占块内调用，因此必须用 unwrap_parallel_model 绕过 DDP wrapper，
        避免触发需要所有 rank 参与的 all_reduce collective 操作导致死锁。
        """
        # 用裸模型（非 DDP wrapper）推理，避免 DDP all_reduce 要求所有 rank 同时调用
        raw_model = unwrap_parallel_model(self.model)
        raw_model.eval()
        features = []
        indices = []
        examples = []
        known_cls_num = data.n_known_cls

        with torch.no_grad():
            for batch in tqdm(data.train_semi_dataloader, desc="Refreshing unknown structure"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _, batch_indices = batch[:6]
                unlabeled_mask = label_ids == -1
                if unlabeled_mask.sum() == 0:
                    continue
                X = {
                    "input_ids": input_ids[unlabeled_mask],
                    "attention_mask": input_mask[unlabeled_mask],
                    "token_type_ids": segment_ids[unlabeled_mask],
                }
                batch_features = raw_model(X)["features"]
                features.append(batch_features.detach().cpu())
                selected_indices = batch_indices[unlabeled_mask].detach().cpu()
                indices.append(selected_indices)
                for raw_idx in selected_indices.tolist():
                    dataset_idx = raw_idx - len(data.train_labeled_examples) if known_cls_num > 0 else raw_idx
                    examples.append(data.train_unlabeled_examples[dataset_idx])

        if not features:
            return None, None, None

        feature_tensor = torch.cat(features, dim=0).contiguous()  # 保留在 CPU，避免 77k×H GPU 峰值
        index_tensor = torch.cat(indices, dim=0).long()
        return feature_tensor, index_tensor, examples

    def initialize_unknown_centroids(self, args, unlabeled_features, n_unknown, proto_manager, data):
        """初始化未知类聚类中心。"""
        if n_unknown <= 0:
            return None, None

        # 统一使用 lloyd 算法，避免 elkan 在高维稀疏空间下触发 MKL/OMP native crash
        features_np = unlabeled_features.float().contiguous().numpy()  # 已在 CPU

        # 关闭 unknown refresh 的 proto-init，统一回退到 kmeans++ + n_init=10，
        # 避免旧原型偏置通过 n_init=1 被持续放大导致单簇塌缩。
        km = KMeans(n_clusters=n_unknown, n_init=10,
                    random_state=args.seed, algorithm="lloyd")
        
        with threadpool_limits(limits=1):
            assignments = km.fit_predict(features_np)
        # 结果留在 CPU；调用方（refresh_unknown_structure）只需广播小体积的 centroids
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        centroids = nn.functional.normalize(centroids, dim=1)
        return torch.tensor(assignments, dtype=torch.long), centroids


    def refresh_unknown_structure(self, args, data, proto_manager, epoch):
        """刷新未知类聚类结构与原型。"""
        # 以 proto_manager 当前维护的 unknown 区间长度为准：
        # LLM merge/split 后 proto_manager.num_classes 已被 set_unknown_prototypes 更新，
        # 这里直接用它减去已知类数，得到当前真实的 unknown 簇数。
        n_unknown = proto_manager.num_classes - data.n_known_cls
        if n_unknown <= 0:
            return

        # epoch=-1 是训练前的初始化 refresh，LLM 决策无意义，get_decisions 内部会跳过。

        # 更新 epoch 到 cluster_refiner，用于文件名区分
        self.cluster_refiner.epoch = epoch

        payload = None
        if not getattr(args, "distributed", False) or is_main_process():
            unlabeled_features, data_indices, examples = self.extract_unlabeled_features(data, args)
            if unlabeled_features is not None and data_indices is not None:
                # 过滤掉最接近已知类原型的 15% unlabeled 样本，降低 known 污染 unknown 刷新。
                # 过滤依据：每个样本与已知原型的最大余弦相似度（best_known_sim）。
                if data.n_known_cls > 0 and unlabeled_features.size(0) > 0:
                    known_protos = proto_manager.get_prototypes()[:data.n_known_cls].detach().to(unlabeled_features.device)
                    if known_protos.numel() > 0:
                        feats_norm = F.normalize(unlabeled_features.float(), dim=1)
                        protos_norm = F.normalize(known_protos.float(), dim=1)
                        known_sim = torch.matmul(feats_norm, protos_norm.T)
                        best_known_sim = known_sim.max(dim=1).values

                        total_ul = unlabeled_features.size(0)
                        drop_count = int(total_ul * 0.15)
                        # 保证过滤后至少还能覆盖 unknown 簇数，避免 KMeans n_samples < n_clusters。
                        max_droppable = max(total_ul - n_unknown, 0)
                        drop_count = min(drop_count, max_droppable)

                        if drop_count > 0:
                            keep_count = total_ul - drop_count
                            # 保留 best_known_sim 较低的样本（更不像 known）
                            keep_idx = torch.topk(best_known_sim, k=keep_count, largest=False).indices
                            keep_idx = keep_idx.sort().values

                            unlabeled_features = unlabeled_features[keep_idx]
                            data_indices = data_indices[keep_idx]
                            if examples is not None:
                                keep_list = keep_idx.tolist()
                                examples = [examples[i] for i in keep_list]

                assignments, centroids = self.initialize_unknown_centroids(
                    args, unlabeled_features, n_unknown, proto_manager, data
                )
                if assignments is not None and centroids is not None:
                    if epoch == -1:
                        # 训练前的初始化 refresh：特征空间质量最好，KMeans 原型直接可用。
                        # 跳过 cluster_refiner.refine()（含 packet 构建、写磁盘、LLM 调用），
                        # 避免在训练尚未开始时做无意义的 LLM 决策，同时节省 I/O 开销。
                        # 正式训练中的 refine（epoch >= 0）才需要 LLM 参与。
                        refined_assignments = assignments
                        refined_centroids   = centroids
                        packets    = None
                        summaries  = None
                    else:
                        refined_assignments, refined_centroids, packets, summaries = self.cluster_refiner.refine(
                            unlabeled_features,
                            examples,
                            assignments,
                            centroids,
                        )
                    payload = {
                        # 只广播必要的小体积数据：centroids（n_unknown × H）+ packets/summaries
                        # unlabeled_features（77k×H）、data_indices、refined_assignments
                        # 在 broadcast 之后均未被使用，去掉可大幅降低 broadcast_object_list 内存压力
                        "refined_centroids": refined_centroids.cpu(),
                        "packets": packets,
                        "summaries": summaries,
                    }

        payload = rank0_broadcast_object(payload)
        if payload is None:
            return

        # 可视化在 broadcast 同步之后执行：所有 rank 都参与 DDP forward，rank 0 负责保存图片
        self.visualize_Tsne(data, args, epoch=epoch)

        refined_centroids = payload["refined_centroids"].to(self.device)
        packets = payload["packets"]
        summaries = payload["summaries"]

        # 在调用 set_unknown_prototypes 之前记录旧的 unknown 簇数，
        # 用于判断 LLM 决策是否改变了簇数量（merge/split）。
        new_n_unknown = refined_centroids.size(0)

        proto_manager.set_unknown_prototypes(
            refined_centroids,
            data.n_known_cls,
            summaries=summaries,
            source=f"refresh_epoch_{epoch + 1}",
        )
        self.latest_cluster_packets = packets
        # 结构更新完成后同步所有 rank，避免后续批次读到不一致状态。
        barrier()


    def evaluation(self, args, data, save_results=True, plot_cm=True, proto_manager=None):
        """执行聚类评估并按需保存结果。

        使用 projection head 输出的对比特征（feat_dim 维）进行聚类，与训练损失作用的空间一致。
        所有 rank 都参与特征提取（DDP gather），只有 rank 0 执行 KMeans/打印/保存。
        """
        # 所有 rank 参与（内部含 gather_features_labels）
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args, use_cls=False)

        # 以下操作只在 rank 0 执行
        if not getattr(args, "distributed", False) or is_main_process():
            feats_test_np = feats_test.cpu().numpy()

            n_clusters_eval = proto_manager.num_classes if proto_manager is not None else self.num_labels
            km_kwargs = {
                "n_clusters": n_clusters_eval,
                "random_state": args.seed,
            }

            proto_init_applied = False
            if args.use_proto_init_kmeans and proto_manager is not None:
                proto_centers = proto_manager.get_prototypes().detach().cpu().numpy()
                if proto_centers.shape == (n_clusters_eval, feats_test_np.shape[1]):
                    km_kwargs.update({"init": proto_centers, "n_init": 1})
                    proto_init_applied = True
                else:
                    print(
                        "[Warning] Prototype init skipped in final KMeans: "
                        f"expected {(n_clusters_eval, feats_test_np.shape[1])}, got {proto_centers.shape}."
                    )

            if not proto_init_applied:
                km_kwargs.update({"n_init": 10})

            with threadpool_limits(limits=1):
                km = KMeans(**km_kwargs).fit(feats_test_np)
            init_mode = "prototype-init" if proto_init_applied else "kmeans++"
            print(
                f"KMeans clustering on projection features "
                f"(n_clusters={n_clusters_eval}, init={init_mode})"
            )

            y_pred = km.labels_
            y_true = labels.cpu().numpy()
            results = clustering_score(y_true, y_pred)
            print("results", results)

            self.test_results = results

            if plot_cm:
                ind, _ = hungray_aligment(y_true, y_pred)
                if ind is None:
                    raise ValueError("Hungarian alignment failed to produce an index mapping.")
                map_ = {i[0]: i[1] for i in ind}
                mapped_pred = [map_.get(idx, idx) for idx in list(np.asarray(y_pred, dtype=np.int64))]
                y_pred = np.asarray(mapped_pred, dtype=np.int64)
                cm = confusion_matrix(y_true, y_pred)
                print("confusion matrix", cm)

            if save_results:
                self.save_results(args)

    def build_refine_weight_schedule(self, args):
        """构建阶段三的权重调度配置。

        目标行为：
        - pseudo loss 从 refine_start_epoch 开始启用（无热身）
        - 最后一次 LLM refine 跳过不调用
        """
        num_epochs = int(args.num_train_epochs)
        weight_start_epoch = int(args.refine_start_epoch)
        refine_every = max(1, int(args.refine_every))

        refine_epochs = [
            e for e in range(num_epochs)
            if (e + 1) >= args.refine_start_epoch
            and ((e + 1 - args.refine_start_epoch) % refine_every == 0)
        ]

        last_refine_epoch = refine_epochs[-1] if refine_epochs else -1

        return {
            "num_epochs": num_epochs,
            "weight_start_epoch": weight_start_epoch,
            "refine_every": refine_every,
            "last_refine_epoch": last_refine_epoch,
        }



    def train(self, args, data):
        """执行阶段三的对比学习与原型更新训练。"""
        criterion = unwrap_parallel_model(self.model).loss_cl

        proto_manager = PrototypeManager(
            num_classes=data.num_labels,
            feat_dim=args.feat_dim,
            momentum=args.proto_momentum,
            device=str(self.device),
        )
        proto_criterion = PrototypeContrastiveLoss(temperature=args.temp)
        scaler = torch.cuda.amp.GradScaler()

        schedule = self.build_refine_weight_schedule(args)
        num_epochs = schedule["num_epochs"]
        loss_cl_weight = args.loss_cl_weight

        # proto_w：前 3 个 epoch 线性热身 0 → target，之后固定在 target
        proto_warmup_epochs = 3

        weight_start_epoch = schedule["weight_start_epoch"]
        refine_every = schedule["refine_every"]
        last_refine_epoch = schedule["last_refine_epoch"]

        if data.n_known_cls > 0:
            print("Initializing prototypes from labeled data...")
            with torch.no_grad():
                labeled_feats, labeled_labels = self.get_global_features_labels(
                    data.train_labeled_full_dataloader, self.model, args
                )
            proto_manager.init_from_labeled(labeled_feats, labeled_labels, data.known_label_list)

        self.refresh_unknown_structure(args, data, proto_manager, epoch=-1)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            data.set_epoch(epoch)
            sampler = getattr(self.train_dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            # 同步所有 rank，确保没有死锁
            barrier()

            self.model.train()
            if self.student_model:
                self.student_model.eval()

            tr_loss = 0.0
            tr_loss_cl = 0.0
            tr_loss_proto = 0.0
            tr_loss_pseudo = 0.0
            nb_tr_steps = 0
            # proto_w：前 3 epoch 线性热身，之后固定在目标值
            proto_weight = min(epoch / proto_warmup_epochs, 1.0) * args.proto_loss_weight

            # pseudo_w：refine_start_epoch 开始线性 ramp up，10 epoch 到 1.0
            if (epoch + 1) < weight_start_epoch:
                pseudo_weight = 0.0
            else:
                pseudo_weight = min((epoch + 1 - weight_start_epoch) / 10.0, 1.0)

            epoch_features = []
            epoch_labels = []
            epoch_pseudo_labels = []
            epoch_pseudo_conf = []
            epoch_assignment_targets = []
            epoch_unified_labels = []
            epoch_unified_conf = []
            routing_stats_buffer = []

            for batch in tqdm(self.train_dataloader, desc="Iteration"):
                batch = tuple(t.to(self.device) for t in batch)
                targets = batch[3]
                data_inds = batch[5].detach().cpu().long()
                visualize_now = nb_tr_steps == 0 and epoch < 5
                view_one, view_two = self.build_dual_views(args, batch, epoch, visualize=visualize_now)

                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        output_one = self.model(view_one)
                        output_two = self.model(view_two)
                        features = output_one["features"]
                        pseudo_labels, pseudo_conf, assignment_targets, assignment_confidence, routing_stats = self.route_unlabeled_targets(
                            features,
                            targets,
                            data_inds,
                            proto_manager,
                            data,
                            args,
                            epoch=epoch,
                        )
                        adjacency = self.build_adjacency_mask(
                            targets,
                            pseudo_labels=pseudo_labels,
                            pseudo_conf=pseudo_conf,
                            assignment_targets=assignment_targets,
                            assignment_confidence=assignment_confidence,
                        )
                        f_pos = torch.stack([output_one["features"], output_two["features"]], dim=1)

                        if epoch < weight_start_epoch:
                            labeled_mask_cl = (targets >= 0)
                            unlabeled_mask_cl = ~labeled_mask_cl

                            # ① labeled 子集：SupCon（有类结构的有监督对比）
                            # unlabeled 不插入 labeled 的 adjacency，避免 SimCLR 与 SupCon 梯度对撕。
                            if labeled_mask_cl.any():
                                f_pos_l = f_pos[labeled_mask_cl]
                                adj_l   = adjacency[labeled_mask_cl][:, labeled_mask_cl]
                                loss_cl_sup = criterion(f_pos_l, mask=adj_l,
                                                        temperature=args.temp,
                                                        base_temperature=args.temp)
                            else:
                                loss_cl_sup = features.sum() * 0.0

                            # ② unlabeled（含 unknown）子集：同实例双视图 SimCLR
                            # mask=None → eye(N) → 正例只是自己的两个视图，不与 labeled 构成交叉正/负例。
                            # 给 unknown 特征提供最小梯度信号，防止其在 warmup 期随 BERT 漂离。
                            warmup_ucl_weight = float(getattr(args, "warmup_ucl_weight", 0.0))
                            if warmup_ucl_weight > 0 and unlabeled_mask_cl.any():
                                f_pos_u = f_pos[unlabeled_mask_cl]
                                loss_cl_u = criterion(f_pos_u,
                                                      temperature=args.temp,
                                                      base_temperature=args.temp)
                            else:
                                loss_cl_u = features.sum() * 0.0

                            loss_cl = loss_cl_sup + warmup_ucl_weight * loss_cl_u

                        else:
                            # post-warmup：只有 labeled + 已路由 unlabeled 参与 loss_cl。
                            # 全量 unlabeled（含未路由的）进入 adjacency 会引入大量噪声正例，
                            # 因为它们的 assignment_targets 还是 -1，adjacency 连边全靠猜。
                            labeled_mask_cl   = (targets >= 0)
                            routed_unknown_mask = (assignment_targets >= 0) & (targets < 0)
                            cl_mask = labeled_mask_cl | routed_unknown_mask

                            if cl_mask.any():
                                f_pos_cl = f_pos[cl_mask]
                                adj_cl   = adjacency[cl_mask][:, cl_mask]
                                loss_cl  = criterion(f_pos_cl, mask=adj_cl,
                                                     temperature=args.temp,
                                                     base_temperature=args.temp)
                            else:
                                loss_cl = features.sum() * 0.0

                        loss_proto = proto_criterion(features, proto_manager.get_prototypes(), targets)

                        # 统一路由标签（用于后续原型 EMA 更新统计）
                        unified_labels = pseudo_labels.clone()
                        unified_conf   = pseudo_conf.clone()
                        unknown_routed = (pseudo_labels < 0) & (assignment_targets >= 0)
                        unified_labels[unknown_routed] = assignment_targets[unknown_routed]
                        unified_conf[unknown_routed]   = assignment_confidence[unknown_routed]
                        routing_stats_buffer.append(routing_stats)

                        all_protos = proto_manager.get_prototypes()

                        # Known-routed loss（分母=全体原型）
                        loss_pseudo_known = proto_criterion(
                            features,
                            all_protos,
                            pseudo_labels,
                            confidences=pseudo_conf,
                        )

                        # Unknown-routed loss（分母=全体原型）
                        global_assignment = assignment_targets.clone()
                        valid_unknown = assignment_targets >= data.n_known_cls
                        global_assignment[~valid_unknown] = -1
                        loss_pseudo_unknown = proto_criterion(
                            features,
                            all_protos,
                            global_assignment,
                            confidences=assignment_confidence,
                        )

                        loss_pseudo = (
                            args.pseudo_weight_known * loss_pseudo_known
                            + args.pseudo_weight_unknown * loss_pseudo_unknown
                        )

                        loss = (
                            loss_cl_weight * loss_cl
                            + proto_weight * loss_proto
                            + pseudo_weight * loss_pseudo
                        )
                        tr_loss += loss.item()
                        tr_loss_cl += loss_cl.item()
                        tr_loss_proto += loss_proto.item()
                        tr_loss_pseudo += loss_pseudo.item()

                    if not loss.requires_grad:
                        raise RuntimeError(
                            "Stage3 loss has no grad_fn before backward. "
                            f"epoch={epoch}, step={nb_tr_steps}, "
                            f"loss_cl.requires_grad={getattr(loss_cl, 'requires_grad', None)}, "
                            f"loss_proto.requires_grad={getattr(loss_proto, 'requires_grad', None)}, "
                            f"loss_pseudo.requires_grad={getattr(loss_pseudo, 'requires_grad', None)}, "
                            f"proto_weight={proto_weight}, pseudo_weight={pseudo_weight}"
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # ── warmup 期每 batch 用 labeled 样本 EMA 更新 known 原型 ──────────────
                # 目的：让原型向量跟上 BERT 参数的变化节奏，防止原型与特征空间脱节。
                # 背景：warmup 期 pass_rate=0%，epoch 末的 update_known_from_pseudo 无样本
                #       可用，known 原型实际冻结。与此同时 BERT 被 loss_cl/loss_proto 持续
                #       重塑，导致 unlabeled 与原型的 cos sim 单调下跌（0.82→0.25），
                #       进而使得 refine_start_epoch 时 KMeans 在极差特征上建立 unknown 原型。
                # 实现：每 batch backward 后，用当前 batch 里 labeled 样本的最新特征对
                #       known 原型做 momentum EMA，无需跨 rank 通信（各 rank 本地更新，
                #       BERT 参数由 DDP allreduce 保证一致，原型的微小 rank 间差异可接受）。
                if epoch < weight_start_epoch and data.n_known_cls > 0:
                    with torch.no_grad():
                        labeled_mask_proto = (targets >= 0) & (targets < data.n_known_cls)
                        if labeled_mask_proto.any():
                            lf = F.normalize(features[labeled_mask_proto].detach(), dim=1)
                            lt = targets[labeled_mask_proto]
                            for cls_idx in range(data.n_known_cls):
                                cls_mask = (lt == cls_idx)
                                if not cls_mask.any():
                                    continue
                                batch_mean = F.normalize(lf[cls_mask].mean(dim=0), dim=0)
                                updated = (
                                    proto_manager.momentum * proto_manager.prototypes[cls_idx]
                                    + (1.0 - proto_manager.momentum) * batch_mean
                                )
                                proto_manager.prototypes[cls_idx] = F.normalize(updated, dim=0)

                with torch.no_grad():
                    epoch_features.append(features.detach())
                    epoch_labels.append(targets.detach())
                    epoch_pseudo_labels.append(pseudo_labels.detach())
                    epoch_pseudo_conf.append(pseudo_conf.detach())
                    epoch_assignment_targets.append(assignment_targets.detach())
                    epoch_unified_labels.append(unified_labels.detach())
                    epoch_unified_conf.append(unified_conf.detach())
                nb_tr_steps += 1

            all_feats = torch.cat(epoch_features, dim=0)
            all_labels = torch.cat(epoch_labels, dim=0)
            all_pseudo_labels = torch.cat(epoch_pseudo_labels, dim=0)
            all_pseudo_conf = torch.cat(epoch_pseudo_conf, dim=0)
            all_assignments = torch.cat(epoch_assignment_targets, dim=0)
            all_unified_labels = torch.cat(epoch_unified_labels, dim=0)
            all_unified_conf = torch.cat(epoch_unified_conf, dim=0)
            all_feats, all_labels = gather_features_labels(all_feats, all_labels)
            all_pseudo_labels = gather_tensor(all_pseudo_labels)
            all_pseudo_conf = gather_tensor(all_pseudo_conf)
            all_assignments = gather_tensor(all_assignments)
            all_unified_labels = gather_tensor(all_unified_labels)
            all_unified_conf = gather_tensor(all_unified_conf)

            proto_manager.update_known_from_pseudo(all_feats, all_unified_labels, all_unified_conf, data.n_known_cls)
            proto_manager.update_unknown_from_assignments(
                all_feats,
                all_unified_labels,
                all_unified_conf,
                data.n_known_cls,
            )

            stats_count = max(len(routing_stats_buffer), 1)
            routing_mode = routing_stats_buffer[0]["mode"] if routing_stats_buffer else "hard_fixed"
            total_ul = sum(s["unlabeled_total"] for s in routing_stats_buffer)
            total_passed = sum(s["passed_total"] for s in routing_stats_buffer)
            total_known_passed = sum(s["known_passed"] for s in routing_stats_buffer)
            total_unknown_passed = sum(s["unknown_passed"] for s in routing_stats_buffer)
            pass_rate = (total_passed / max(total_ul, 1)) * 100.0
            known_route_rate = (total_known_passed / max(total_passed, 1)) * 100.0
            unknown_route_rate = (total_unknown_passed / max(total_passed, 1)) * 100.0

            known_mask = (all_unified_labels >= 0) & (all_unified_labels < data.n_known_cls)
            unknown_mask = all_unified_labels >= data.n_known_cls

            known_hist = torch.bincount(
                all_unified_labels[known_mask].long().cpu(),
                minlength=data.n_known_cls,
            ) if known_mask.any() else torch.zeros(data.n_known_cls, dtype=torch.long)
            unknown_bins = max(proto_manager.num_classes - data.n_known_cls, 0)
            unknown_hist = torch.bincount(
                (all_unified_labels[unknown_mask] - data.n_known_cls).long().cpu(),
                minlength=unknown_bins,
            ) if unknown_mask.any() and unknown_bins > 0 else torch.zeros(unknown_bins, dtype=torch.long)

            avg_known_sim = sum(s["mean_known_sim"] for s in routing_stats_buffer) / stats_count
            avg_known_conf = sum(s["mean_known_conf"] for s in routing_stats_buffer) / stats_count
            avg_unknown_sim = sum(s["mean_unknown_sim"] for s in routing_stats_buffer) / stats_count
            avg_unknown_conf = sum(s["mean_unknown_conf"] for s in routing_stats_buffer) / stats_count
            routed_known_sim = (
                sum(s["routed_known_sim"] * s["known_passed"] for s in routing_stats_buffer)
                / max(total_known_passed, 1)
            )
            routed_known_conf = (
                sum(s["routed_known_conf"] * s["known_passed"] for s in routing_stats_buffer)
                / max(total_known_passed, 1)
            )
            routed_unknown_sim = (
                sum(s["routed_unknown_sim"] * s["unknown_passed"] for s in routing_stats_buffer)
                / max(total_unknown_passed, 1)
            )
            routed_unknown_conf = (
                sum(s["routed_unknown_conf"] * s["unknown_passed"] for s in routing_stats_buffer)
                / max(total_unknown_passed, 1)
            )

            avg_loss = tr_loss / max(nb_tr_steps, 1)
            avg_loss_cl = tr_loss_cl / max(nb_tr_steps, 1)
            avg_loss_proto = tr_loss_proto / max(nb_tr_steps, 1)
            avg_loss_pseudo = tr_loss_pseudo / max(nb_tr_steps, 1)
            print(
                f"train_loss: {avg_loss:.4f}, "
                f"loss_cl: {avg_loss_cl:.4f}, "
                f"loss_proto: {avg_loss_proto:.4f}, "
                f"loss_pseudo: {avg_loss_pseudo:.4f}, "
                f"loss_cl_w: {loss_cl_weight:.2f}, "
                f"proto_w: {proto_weight:.3f}, "
                f"pseudo_w: {pseudo_weight:.3f}"
            )
            print(
                f"pseudo_obs: mode={routing_mode}, "
                f"known_th(sim/conf)=({args.known_sim_thresh:.3f}/{args.known_conf_thresh:.3f}), "
                f"unknown_th(sim/conf)=({args.unknown_sim_thresh:.3f}/{args.unknown_conf_thresh:.3f}), "
                f"compete_bias={args.compete_bias:.3f}, "
                f"best_known(sim/conf)=({avg_known_sim:.3f}/{avg_known_conf:.3f}), "
                f"best_unknown(sim/conf)=({avg_unknown_sim:.3f}/{avg_unknown_conf:.3f}), "
                f"pass_rate={pass_rate:.2f}% ({total_passed}/{total_ul}), "
                f"known_route={known_route_rate:.2f}%, unknown_route={unknown_route_rate:.2f}%, "
                f"known_mean(sim/conf)=({routed_known_sim:.3f}/{routed_known_conf:.3f}), "
                f"unknown_mean(sim/conf)=({routed_unknown_sim:.3f}/{routed_unknown_conf:.3f})"
            )
            print(f"pseudo_known_hist: {known_hist.tolist()}")
            if unknown_bins > 0:
                print(f"pseudo_unknown_hist: {unknown_hist.tolist()}")

            # 打印 unlabeled 样本与 known / unknown 原型的 sim 分布，用于诊断路由偏差
            if is_main_process() and unknown_bins > 0:
                with torch.no_grad():
                    ul_mask_diag = (all_labels < 0)
                    if ul_mask_diag.any():
                        _protos = proto_manager.get_prototypes()
                        _sim = torch.matmul(all_feats[ul_mask_diag], _protos.T)
                        _sim_known   = _sim[:, :data.n_known_cls]
                        _sim_unknown = _sim[:, data.n_known_cls:]
                        _best_known   = _sim_known.max(dim=1).values
                        _best_unknown = _sim_unknown.max(dim=1).values
                        print(
                            f"sim_diag: best_known={_best_known.mean():.3f}"
                            f"(p50={_best_known.median():.3f},"
                            f"p80={torch.quantile(_best_known.float(),0.8):.3f})"
                            f" | best_unknown={_best_unknown.mean():.3f}"
                            f"(p50={_best_unknown.median():.3f},"
                            f"p80={torch.quantile(_best_unknown.float(),0.8):.3f})"
                        )

            # 路由阈值采用硬赋值：known/unknown 各自的 sim/conf 阈值 + compete_bias。

            # ── epoch 结束后用实际路由结果更新 cluster_freq ────────────────
            # cluster_freq 用于 route_unlabeled_targets 里的垄断惩罚。
            # 此前该值从不更新（永远是初始均匀分布），导致惩罚对真实垄断完全失明。
            # 现在：每个 epoch 结束后，用本轮 unknown 路由的实际频率做 EMA 更新。
            unknown_bins = max(proto_manager.num_classes - data.n_known_cls, 0)
            if unknown_bins > 0:
                unknown_routed_mask = all_unified_labels >= data.n_known_cls
                if unknown_routed_mask.any():
                    actual_counts = torch.bincount(
                        (all_unified_labels[unknown_routed_mask] - data.n_known_cls).long().cpu().clamp(0, unknown_bins - 1),
                        minlength=unknown_bins,
                    ).float()
                    actual_freq = actual_counts / actual_counts.sum().clamp_min(1.0)
                    actual_freq = actual_freq.to(self.device)
                    if (
                        getattr(self, "cluster_freq", None) is not None
                        and self.cluster_freq.size(0) == unknown_bins
                    ):
                        # EMA 平滑：避免单 epoch 噪声引起惩罚剧烈震荡
                        self.cluster_freq = 0.7 * self.cluster_freq + 0.3 * actual_freq
                    else:
                        self.cluster_freq = actual_freq

            should_refresh = (
                epoch + 1 >= args.refine_start_epoch
                and ((epoch + 1 - args.refine_start_epoch) % args.refine_every == 0)
            )
            if should_refresh:
                self.refresh_unknown_structure(args, data, proto_manager, epoch)
                # KMeans 重新索引了簇编号，但分布形状仍有意义：
                # 排序保留（丢弃索引对应关系），防止 post-refine 首轮惩罚真空。
                if self.cluster_freq is not None:
                    self.cluster_freq, _ = self.cluster_freq.sort(descending=True)

            # 每个 epoch 结束都做一次 evaluation，方便观察训练曲线
            print(f"\n[Eval @ epoch {epoch + 1}]")
            self.evaluation(
                args, data,
                save_results=False,
                plot_cm=False,
                proto_manager=proto_manager,
            )
            # 追踪最佳 epoch（以 ACC 为准）
            if self.test_results is not None:
                current_score = self.test_results.get('ACC', 0) * 0.5 + self.test_results.get('NMI', 0) * 0.5
                best_score = (self.best_eval_results or {}).get('ACC', 0) * 0.5 + (self.best_eval_results or {}).get('NMI', 0) * 0.5
                if self.best_eval_results is None or current_score > best_score:
                    self.best_eval_results = dict(self.test_results)

        return proto_manager

    def get_optimizer(self, args):
        """构建阶段三训练的优化器与调度器。"""
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )
        return optimizer, scheduler

    def load_pretrained_model(self):
        """加载预训练模型骨干参数到当前模型。"""
        pretrained_dict = get_model_backbone(self.pretrained_model).state_dict()
        get_model_backbone(self.model).load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args, use_cls=False):
        """提取给定数据加载器的特征与标签。

        Args:
            use_cls: True 时返回 BERT 最后一层 CLS embedding（768 维），
                     False 时返回 projection head 输出的对比特征（feat_dim 维）。
                     聚类评估与可视化应使用 CLS embedding，原型更新应使用 projection 特征。
        """
        model.eval()
        backbone = get_model_backbone(model)
        if use_cls:
            # 兼容不同 backbone 结构：
            # - BertForMaskedLM: backbone.config.hidden_size
            # - 包含内层 backbone 的封装: backbone.backbone.config.hidden_size
            if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
                feat_dim = backbone.config.hidden_size
            elif hasattr(backbone, "backbone") and hasattr(backbone.backbone, "config"):
                feat_dim = backbone.backbone.config.hidden_size
            else:
                raise AttributeError("Unable to resolve hidden_size from backbone in get_features_labels")
        else:
            feat_dim = args.feat_dim
        total_features = torch.empty((0, feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                output = model(X, output_hidden_states=use_cls)
                feature = output["hidden_states"] if use_cls else output["features"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        if getattr(args, "distributed", False):
            total_features, total_labels = gather_features_labels(total_features, total_labels)

        return total_features, total_labels

    def evaluate_cluster_stability(self, args, data, n_runs=5):
        """评估多次聚类结果的稳定性。"""
        from sklearn.metrics import normalized_mutual_info_score

        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats_test = feats_test.cpu().numpy()

        all_preds = []
        for seed in range(n_runs):
            km = KMeans(n_clusters=self.num_labels, n_init=10, random_state=seed).fit(feats_test)
            all_preds.append(km.labels_)

        stability_scores = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                nmi = normalized_mutual_info_score(all_preds[i], all_preds[j])
                stability_scores.append(nmi)

        mean_stability = np.mean(stability_scores)
        std_stability = np.std(stability_scores)
        print(f"Cluster Stability NMI: {mean_stability:.4f} ± {std_stability:.4f}")
        return mean_stability, std_stability

    def save_results(self, args):
        """将测试结果追加保存到结果文件。"""
        if getattr(args, "distributed", False) and not is_main_process():
            return
        self.output_paths.results_dir()

        var = [
            args.internal_dataset,
            args.known_cls_ratio,
            args.labeled_ratio,
            args.view_strategy,
            args.input_strategy,
            args.with_speaker,
            args.noise_ratio,
            args.noise_ratio,
            args.seed,
        ]
        names = [
            "dataset",
            "known_cls_ratio",
            "labeled_ratio",
            "view_strategy",
            "input_strategy",
            "with_speaker",
            "teacher_noise_ratio",
            "student_noise_ratio",
            "seed",
        ]

        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        results_path = self.output_paths.results_csv_path()

        if not os.path.exists(results_path):
            df1 = pd.DataFrame([values], columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            if not isinstance(df1, pd.DataFrame):
                raise ValueError("results.csv could not be loaded as a DataFrame")
            df1.loc[len(df1)] = results
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)
        print("test_results", data_diagram)

    def visualize_Tsne(self, data, args, n_samples=5000, epoch=None):
        """生成并保存测试表示的 UMAP 可视化，同时输出两张图：

        1. CLS embedding（BERT 最后一层 [CLS]，768 维）——展示 backbone 语义空间的全局结构。
        2. Projection head 输出（feat_dim 维）——展示 loss_cl / loss_proto 直接作用的判别空间。

        UMAP 参数调整为 n_neighbors=15, min_dist=0.05，更适合展示紧凑的局部聚类结构。
        散点图同时展示两层信息：
          - 颜色：class label（意图类别）
          - 形状：speaker 角色（T=教师三角形, S=学生圆形, 未知=正方形）

        Args:
            epoch: 当前 epoch 编号，不为 None 时文件名中包含 epoch 标记。
        """
        # 所有 rank 参与 DDP forward 推理（gather 需要全员），计算完特征后汇聚到 rank 0
        # 只有 rank 0 执行 UMAP 计算和图片保存（numba 不支持多进程并发编译）
        is_rank_0 = not getattr(args, "distributed", False) or is_main_process()

        epoch_tag = f" (epoch {epoch})" if epoch is not None else ""
        print(f"Visualizing embeddings{epoch_tag}...")
        self.model.eval()
        dataloader = data.test_dataloader

        cls_reps = []
        proj_reps = []
        labels = []
        speakers = []
        count = 0

        # test_dataloader 使用 SequentialSampler，与 test_examples 严格顺序对齐
        example_iter = iter(data.test_examples)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings for visualization"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch[:4]
                X = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "token_type_ids": segment_ids,
                }
                # 同时提取 CLS embedding 和 projection head 输出
                outputs = self.model(X, output_hidden_states=True)
                cls_emb  = outputs["hidden_states"].detach().cpu().numpy()   # (B, 768)
                proj_emb = outputs["features"].detach().cpu().numpy()        # (B, feat_dim)
                batch_labels = label_ids.detach().cpu().numpy()

                # 按 batch 大小依次取对应 example 的 speaker 字段
                batch_size = input_ids.size(0)
                for _ in range(batch_size):
                    try:
                        ex = next(example_iter)
                        spk = getattr(ex, "speaker", "") or ""
                        speakers.append(spk.lower())
                    except StopIteration:
                        speakers.append("")

                cls_reps.append(cls_emb)
                proj_reps.append(proj_emb)
                labels.append(batch_labels)

                count += batch_size
                if count >= n_samples:
                    break

        labels_arr = np.concatenate(labels, axis=0)
        cls_arr    = np.concatenate(cls_reps,  axis=0)
        proj_arr   = np.concatenate(proj_reps, axis=0)
        speakers   = speakers[:len(labels_arr)]

        # 所有 rank 的 DDP forward 已完成，在 UMAP 前同步一次
        # 防止非 rank 0 提前走到下一个 collective 而 rank 0 还在跑 UMAP
        barrier()

        # 只有 rank 0 执行 UMAP 计算和图片保存（numba 不支持多进程并发 JIT 编译）
        if is_rank_0:
            reducer_kwargs = dict(n_neighbors=15, min_dist=0.05, random_state=args.seed)

            # setup_for_distributed 把 builtins.print 替换成了自定义函数，
            # numba JIT 编译时无法识别非原生 print，临时还原再恢复。
            import builtins
            _patched_print = builtins.print
            _orig_print = getattr(builtins, "_original_print", builtins.print)
            builtins.print = _orig_print
            try:
                for space, embeddings in [("cls", cls_arr), ("proj", proj_arr)]:
                    print(f"Collected {len(labels_arr)} samples. Running UMAP ({space})...")
                    t0 = time.time()
                    reducer = umap.UMAP(**reducer_kwargs)
                    embedding_2d = reducer.fit_transform(embeddings)
                    print(f"UMAP ({space}) finished in {time.time() - t0:.2f}s")

                    space_label = "CLS embedding" if space == "cls" else "Projection Head"
                    title = (
                        f"UMAP [{space_label}] – {args.internal_dataset} "
                        f"(Method: {args.method}){epoch_tag}"
                    )
                    fig = self.plot_embedding(embedding_2d, labels_arr, title, speakers=speakers)

                    if args.save_results_path:
                        save_path = self.output_paths.umap_png_path(epoch=epoch, space=space)
                        plt.savefig(save_path)
                        print(f"Visualization saved to {save_path}")

                    plt.close(fig)
            finally:
                # 无论成功还是失败，都恢复被替换的分布式 print
                builtins.print = _patched_print

        # UMAP 完成后再次同步，确保所有 rank 步调一致后再继续训练
        barrier()

    @staticmethod
    def plot_embedding(data, label, title, speakers=None):
        """绘制二维嵌入散点图。

        Args:
            data: UMAP 降维后的 (N, 2) 坐标数组。
            label: 每个样本的 class label（整数数组）。
            title: 图标题。
            speakers: 每个样本的说话者字符串列表（'t'=教师, 's'=学生, 其他=未知）。
                      不为 None 时用形状区分角色：教师=三角形, 学生=圆形, 未知=正方形。
        """
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(111)
        unique_labels = np.unique(label)
        n_classes = len(unique_labels)
        cmap = plt.cm.get_cmap("tab20" if n_classes <= 20 else "jet", n_classes)

        # speaker → matplotlib marker
        _SPEAKER_MARKER = {"t": "^", "s": "o"}   # 教师=三角形, 学生=圆形
        _SPEAKER_LABEL  = {"t": "T(teacher)", "s": "S(student)"}
        _DEFAULT_MARKER = "s"                     # 未知=正方形

        speakers_arr = np.array(speakers) if speakers is not None else None

        for i, lbl in enumerate(unique_labels):
            class_mask = label == lbl
            color = cmap(i / n_classes)

            if speakers_arr is not None:
                # 按 speaker 细分，分别画不同形状
                drawn_any = False
                for spk_key, marker in _SPEAKER_MARKER.items():
                    spk_mask = class_mask & (speakers_arr == spk_key)
                    if spk_mask.sum() == 0:
                        continue
                    plt.scatter(
                        data[spk_mask, 0],
                        data[spk_mask, 1],
                        c=[color],
                        marker=marker,
                        label=str(lbl) if not drawn_any else "_nolegend_",
                        s=12,
                        alpha=0.7,
                    )
                    drawn_any = True
                # 未知 speaker
                other_mask = class_mask & ~np.isin(speakers_arr, list(_SPEAKER_MARKER.keys()))
                if other_mask.sum() > 0:
                    plt.scatter(
                        data[other_mask, 0],
                        data[other_mask, 1],
                        c=[color],
                        marker=_DEFAULT_MARKER,
                        label=str(lbl) if not drawn_any else "_nolegend_",
                        s=12,
                        alpha=0.7,
                    )
            else:
                plt.scatter(
                    data[class_mask, 0],
                    data[class_mask, 1],
                    c=[color],
                    label=str(lbl),
                    s=10,
                    alpha=0.7,
                )

        # 图例：class label（颜色）+ speaker 角色说明（形状）
        plt.legend(loc="best", fontsize=7, markerscale=2)
        if speakers_arr is not None:
            # 在右下角加一个形状图例说明
            from matplotlib.lines import Line2D
            shape_legend = [
                Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=6, label="Teacher (T)"),
                Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="Student (S)"),
            ]
            ax = plt.gca()
            leg1 = ax.get_legend()
            ax.add_artist(leg1)
            ax.legend(handles=shape_legend, loc="lower right", fontsize=7, title="Speaker")

        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig

    def generate_intent_profiles(self, args, data, proto_manager, top_k=5):
        """为各意图原型生成代表性话语画像。所有 rank 参与特征提取，rank 0 负责计算和返回结果。"""
        # 所有 rank 都参与（内部含 gather_features_labels）
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)

        if getattr(args, "distributed", False) and not is_main_process():
            return {}
        all_texts = []
        for batch in data.test_dataloader:
            indices = batch[-1].numpy()
            for idx in indices:
                example = data.test_examples[idx]
                all_texts.append(example.text_a)

        prototypes = proto_manager.get_prototypes()
        profiles = {}
        for class_idx in range(data.num_labels):
            proto = prototypes[class_idx].unsqueeze(0)
            sims = torch.cosine_similarity(feats_test, proto.expand_as(feats_test), dim=1)
            top_k_indices = sims.topk(min(top_k, len(sims))).indices
            label_name = (
                args.id2label.get(class_idx, f"Class_{class_idx}")
                if class_idx < data.n_known_cls
                else f"NewIntent_{class_idx - data.n_known_cls}"
            )
            profiles[label_name] = {
                "representative_utterances": [
                    all_texts[i] for i in top_k_indices.cpu().numpy() if i < len(all_texts)
                ],
                "is_new_intent": class_idx >= data.n_known_cls,
                "summary": proto_manager.cluster_summary[class_idx],
            }
            print(f"\n{'=' * 50}")
            print(f"Intent: {label_name} ({'NEW' if class_idx >= data.n_known_cls else 'KNOWN'})")
            print("Representative utterances:")
            for utt in profiles[label_name]["representative_utterances"]:
                print(f"  - {utt}")
            if profiles[label_name]["summary"]:
                print(f"Summary: {profiles[label_name]['summary']}")

        output_path = self.output_paths.intent_profiles_json_path()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(profiles, handle, indent=2, ensure_ascii=False)
        print(f"\nIntent profiles saved to {output_path}")
        return profiles
