"""阶段二：教师掩码蒸馏与学生噪声建模管理器。"""

import heapq
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler

from models import CLBert, NoiseGenerator
from utils.parallel import barrier, is_main_process, unwrap_parallel_model, wrap_model
from utils.build_ml import seed_everything, get_whole_word_attributions
from utils.output_paths import OutputPaths
from utils.tools import TeacherWrapper, set_seed
from utils.sequence_classification import TripletSequenceClassificationExplainer, SequenceClassificationExplainer

class NoiseManager:
    """管理蒸馏阶段的教师掩码缓存与学生模型训练。"""

    def __init__(self, args, data):
        """初始化教师模型、学生模型与蒸馏配置。"""
        set_seed(args.seed)
        self.args = args
        if getattr(args, "distributed", False):
            self.device = args.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.n_known_cls = data.n_known_cls

        # 教师模型只负责对已知类样本计算 Integrated Gradients 归因，
        # 因此分类头只需覆盖已知类，避免未知类的随机权重污染归因信号。
        teacher_num_labels = self.n_known_cls if self.n_known_cls > 0 else self.num_labels
        self.teacher_model = CLBert(args.bert_model, device=self.device, num_labels=teacher_num_labels)
        hidden_size = self.teacher_model.backbone.config.hidden_size
        # dropout_prob 是网络正则化参数，固定为 0.1；
        # noise_ratio 是噪声词选取比例，由 args 在推理阶段控制，两者语义不同。
        self.student_model = NoiseGenerator(
            hidden_size=hidden_size,
            device=self.device,
            dropout_prob=0.1,
            use_transformer=args.use_transformer_student,
            num_heads=args.student_num_heads,
            num_layers=args.student_num_layers
        ).to(self.device)

        self.teacher_model = wrap_model(self.teacher_model, self.device, distributed=getattr(args, "distributed", False))
        self.student_model = wrap_model(self.student_model, self.device, distributed=getattr(args, "distributed", False))

        # NoiseGenerator 从零初始化，需要比 BERT 微调大得多的学习率（lr_distill，默认 1e-3）。
        # args.lr 是为 Stage3 BERT 微调设计的（1e-5），对小网络会导致收敛极慢。
        distill_lr = getattr(args, 'lr_distill', 1e-3)
        self.optimizer = AdamW([
            {'params': self.student_model.parameters(), 'lr': distill_lr}
        ], lr=distill_lr)
        
        num_training_steps = len(data.train_distillation_dataloader) * args.num_distillate_epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.output_paths = OutputPaths(args)
        self.viz_count = 0

    def cache_teacher_masks(self, data, args, noise_ratio, with_abs, with_pos, is_continuous):
        """
        预先缓存教师模型生成的掩码，避免蒸馏阶段重复计算。

        只对已知类的有标注样本生成掩码：教师分类头仅覆盖已知类，
        对未知类样本的归因信号无意义，不应纳入蒸馏训练。
        """
        cache_path = self.output_paths.mask_cache_path(noise_ratio, with_pos)

        if os.path.exists(cache_path):
            # 避免在 DDP 下广播超大 Python 对象导致 NCCL BROADCAST 超时：
            # 统一走“各 rank 本地加载缓存文件”。
            barrier()
            return torch.load(cache_path, map_location="cpu")
        self.output_paths.results_dir()
        cached_masks = {}
        # 只遍历有标注的已知类样本（train_labeled_full_dataloader），
        # 而非含未标注样本的 train_distillation_full_dataloader。
        dataloader = data.train_labeled_full_dataloader
        self.teacher_model.eval()
        if is_main_process():
            for batch in tqdm(dataloader, desc="Caching Masks"):
                batch_data = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'special_tokens_mask': batch[4],
                    'text': self.tokenizer.batch_decode(batch[0], skip_special_tokens=True)
                }
                indices = batch[5]
                with torch.no_grad():
                    masks = self.generate_teacher_mask(
                        batch=batch_data,
                        noise_ratio=noise_ratio,
                        with_abs=with_abs,
                        with_pos=with_pos,
                        is_continuous=is_continuous
                    )
                for i, idx in enumerate(indices):
                    cached_masks[idx.item()] = masks[i]
            torch.save(cached_masks, cache_path)
        # 仅 rank0 写盘；其余 rank 等待后从同一路径读取，避免大对象广播。
        barrier()
        return torch.load(cache_path, map_location="cpu")

    def generate_teacher_mask(self, batch, noise_ratio, with_abs=False, with_pos=False, is_continuous=False):
        """
        根据教师解释分数生成掩码。

        参数说明：
        - with_abs：是否对归因分数取绝对值。
        - with_pos：是否仅保留特定词性。
        - is_continuous：是否使用连续 span 掩码。
        """
        self.teacher_model.eval()
        attention_mask = batch["attention_mask"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        special_tokens_mask = batch["special_tokens_mask"].to(self.device)

        if is_continuous:
            if with_pos and not getattr(self, "_warned_with_pos_disabled", False):
                print(
                    "[Warning] --with_pos is incompatible with --is_continuous (span masking). "
                    "with_pos has been automatically disabled. "
                    "To use POS filtering, set --is_continuous to False."
                )
                self._warned_with_pos_disabled = True
            with_pos = False  # span masking 不支持 POS 过滤，强制关闭

        if self.args.input_strategy == "CONTEXT":
            triplet_explainer = TripletSequenceClassificationExplainer(
                model=TeacherWrapper(unwrap_parallel_model(self.teacher_model)),
                tokenizer=self.tokenizer
            )
            sequence_explainer = None
        else:
            triplet_explainer = None
            sequence_explainer = SequenceClassificationExplainer(
                model=TeacherWrapper(unwrap_parallel_model(self.teacher_model)),
                tokenizer=self.tokenizer
            )
        batch_size, seq_len = input_ids.shape
        teacher_masks = []
        sep_token_id = self.tokenizer.sep_token_id
        for i in range(batch_size):
            torch.cuda.empty_cache()
            with torch.enable_grad():
                curr_input_ids = input_ids[i]
                if self.args.input_strategy == "CONTEXT":
                    sep_indices = (curr_input_ids == sep_token_id).nonzero(as_tuple=True)[0]
                    prev_ids = curr_input_ids[1: sep_indices[0]]
                    curr_ids = curr_input_ids[sep_indices[0] + 1: sep_indices[1]]
                    next_ids = curr_input_ids[sep_indices[1] + 1: sep_indices[2]]

                    text_prev = self.tokenizer.decode(prev_ids, skip_special_tokens=True)
                    text_curr = self.tokenizer.decode(curr_ids, skip_special_tokens=True)
                    text_next = self.tokenizer.decode(next_ids, skip_special_tokens=True)

                    if triplet_explainer is None:
                        raise ValueError("Triplet explainer is not initialized.")
                    explainer_call = getattr(triplet_explainer, "__call__")
                    word_attributions = np.array(explainer_call(
                        text_prev=text_prev,
                        text_curr=text_curr,
                        text_next=text_next
                    ))
                else:
                    sentence = self.tokenizer.decode(curr_input_ids, skip_special_tokens=True)
                    if sequence_explainer is None:
                        raise ValueError("Sequence explainer is not initialized.")
                    explainer_call = getattr(sequence_explainer, "__call__")
                    word_attributions = np.array(explainer_call(text=sentence))

                if with_abs:
                    word_attributions[:, 1] = [abs(eval(attribution[1])) for attribution in word_attributions]

                word_attributions = get_whole_word_attributions(word_attributions, with_pos)
                self.visualize_teacher_attribution(triplet_explainer, sequence_explainer)
                if word_attributions.ndim < 2 or len(word_attributions) == 0:
                    teacher_masks.append({
                        'word_spans': [],
                        'word_noise_labels': torch.zeros(0, dtype=torch.float32)
                    })
                    continue

                top_N = math.ceil(len(word_attributions) * noise_ratio)
                top_N = min(max(top_N, 0), len(word_attributions))

                if top_N == 0:
                    top_index = []
                elif not is_continuous:
                    tmp = zip(range(len(word_attributions)), word_attributions[:, 1].astype(float))
                    top_index = heapq.nsmallest(top_N, tmp, key=lambda x: x[1])
                    top_index = [top[0] for top in top_index]
                else:
                    max_attr = np.sum(word_attributions[:, 1][0:top_N])
                    max_idx = 0
                    for begin_idx in range(1, len(word_attributions) - top_N, 1):
                        if np.sum(word_attributions[:, 1][begin_idx:begin_idx + top_N]) < max_attr:
                            max_attr = np.sum(word_attributions[:, 1][begin_idx:begin_idx + top_N])
                            max_idx = begin_idx

                    top_index = list(range(max_idx, max_idx + top_N))
                selected_word_indices = set(top_index)
                word_spans = []
                word_noise_labels = []
                sample_special_mask = special_tokens_mask[i] if special_tokens_mask is not None else None
                for word_idx, attribution in enumerate(word_attributions):
                    token_pos_list = []
                    for pos in attribution[2]:
                        pos = int(pos)
                        if pos >= seq_len:
                            continue
                        if sample_special_mask is not None and sample_special_mask[pos].item() == 1:
                            continue
                        token_pos_list.append(pos)
                    if not token_pos_list:
                        continue
                    word_spans.append(token_pos_list)
                    word_noise_labels.append(1.0 if word_idx in selected_word_indices else 0.0)

                teacher_masks.append({
                    'word_spans': word_spans,
                    'word_noise_labels': torch.tensor(word_noise_labels, dtype=torch.float32)
                })

        return teacher_masks

    def visualize_teacher_attribution(self, triplet_explainer, sequence_explainer):
        """保存教师解释器生成的 HTML 可视化结果。"""
        if not self.args.save_model_path or self.viz_count >= 10:
            return

        save_path = self.output_paths.attribution_viz_file(self.viz_count)
        try:
            active_explainer = triplet_explainer if triplet_explainer is not None else sequence_explainer
            if active_explainer is None:
                raise ValueError("No explainer available for visualization.")
            active_explainer.visualize(html_filepath=save_path)
        except Exception as e:
            print(f"Visualization failed for sample {self.viz_count}: {e}")

        self.viz_count += 1

    @staticmethod
    def build_padded_word_targets(teacher_targets, device):
        """将教师词级标签补齐为统一张量形状。"""
        max_words = max((len(target['word_spans']) for target in teacher_targets), default=0)
        if max_words == 0:
            max_words = 1

        word_labels = torch.zeros(len(teacher_targets), max_words, 1, device=device)
        word_valid_mask = torch.zeros(len(teacher_targets), max_words, 1, device=device)
        for batch_idx, target in enumerate(teacher_targets):
            labels = target['word_noise_labels']
            if labels.numel() == 0:
                continue
            labels = labels.to(device)
            word_labels[batch_idx, :labels.size(0), 0] = labels
            word_valid_mask[batch_idx, :labels.size(0), 0] = 1.0
        return word_labels, word_valid_mask


    def Mask_BERT_with_ratio(self, args, data):
        """按教师掩码监督训练学生噪声模型。"""
        num_epochs = args.num_distillate_epochs
        random_states = getattr(args, 'random_states', [args.seed])

        cached_masks = self.cache_teacher_masks(
            data, args, args.noise_ratio, args.with_abs, args.with_pos, args.is_continuous
        )
        if cached_masks is None:
            raise ValueError("Cached teacher masks are not available.")
        cached_masks = dict(cached_masks)

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            padding=True,
            truncation=True,
            model_max_length=args.internal_max_seq_length
        )

        for random_state in random_states:
            seed_everything(random_state)
            min_loss = 99999
            best_student_state = None
            wait_patient = 20
            wait = 0
            student_net = unwrap_parallel_model(self.student_model)
            # Temperature annealing：从 1.0 线性退火到 0.1（与推理时一致），
            # 确保训练末期输出分布与推理对齐，消除 train/inference 分布偏移。
            temp_start = 1.0
            temp_end = 0.1
            for epoch in range(num_epochs):
                # 线性退火：epoch 0 → temp_start，epoch num_epochs-1 → temp_end
                if num_epochs > 1:
                    current_temperature = temp_start + (temp_end - temp_start) * epoch / (num_epochs - 1)
                else:
                    current_temperature = temp_end
                data.set_epoch(epoch)
                self.teacher_model.eval()   # 教师只提供 embedding，不参与训练，保持 eval 模式关闭 dropout
                self.student_model.train()

                tr_loss = 0
                nb_tr_steps = 0

                # 只在已知类有标注样本上训练，与 cache_teacher_masks 保持一致。
                train_dataloader = data.train_labeled_dataloader
                sampler = getattr(train_dataloader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

                for raw_batch in pbar:
                    if isinstance(raw_batch, dict):
                        batch = {key: value for key, value in raw_batch.items()}
                    else:
                        batch = {
                            'input_ids': raw_batch[0],
                            'attention_mask': raw_batch[1],
                            'token_type_ids': raw_batch[2],
                            'labels': raw_batch[3],
                            'special_tokens_mask': raw_batch[4],
                            'indices': raw_batch[5]
                        }
                    batch['text'] = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    special_tokens_mask = batch["special_tokens_mask"].to(self.device)
                    indices = batch['indices']
                    teacher_target_list = []
                    for idx in indices:
                        mask_tensor = cached_masks.get(idx.item())
                        if mask_tensor is None:
                            raise ValueError(f"Missing cached mask for index {idx.item()}")
                        teacher_target_list.append(mask_tensor)

                    batch_word_spans = [target['word_spans'] for target in teacher_target_list]
                    teacher_word_labels, teacher_word_valid_mask = self.build_padded_word_targets(
                        teacher_target_list,
                        self.device,
                    )

                    teacher_obj = unwrap_parallel_model(self.teacher_model)
                    backbone = teacher_obj.backbone
                    with torch.no_grad():
                        # 方案一：使用教师最后一层隐藏状态作为学生输入。
                        # 相比底层 word_embeddings，最后一层已包含完整上下文语义，
                        # 学生更容易从中识别出归因分数低（不重要）的词。
                        teacher_outputs = backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
                        # hidden_states[-1]: shape (batch, seq_len, hidden_size)
                        teacher_hidden = teacher_outputs.hidden_states[-1].detach()

                    student_word_probs, student_word_valid_mask, student_word_probs_loss = self.student_model(
                        teacher_hidden,
                        attention_mask=attention_mask,
                        special_tokens_mask=special_tokens_mask,
                        temperature=current_temperature,
                        noise_ratio=args.noise_ratio,
                        word_spans=batch_word_spans,
                        return_word_probs=True,
                        apply_inference_gate=False,
                    )


                    valid_mask = teacher_word_valid_mask * student_word_valid_mask
                    if valid_mask.sum() == 0:
                        continue

                    # pos_weight 补偿正负类不均衡：噪声词（正类）占比 noise_ratio，干净词占 1-noise_ratio
                    pos_weight = torch.tensor(
                        [(1.0 - args.noise_ratio) / max(args.noise_ratio, 1e-6)],
                        device=self.device
                    )
                    # 数值稳健性：CUDA BCE 内核要求 input ∈ [0,1]。
                    # 极端情况下（上游 NaN/Inf 传播）会触发 device-side assert。
                    # 这里先做清洗和截断，再计算损失，避免整卡崩溃。
                    student_word_probs_loss = torch.nan_to_num(
                        student_word_probs_loss,
                        nan=0.5,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    teacher_word_labels = torch.nan_to_num(
                        teacher_word_labels,
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)

                    sample_weight = pos_weight.expand_as(student_word_probs_loss) * teacher_word_labels + (1.0 - teacher_word_labels)
                    sample_weight = torch.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)

                    loss_distill = (F.binary_cross_entropy(
                        student_word_probs_loss,
                        teacher_word_labels,
                        weight=sample_weight,
                        reduction='none'
                    ) * valid_mask).sum() / valid_mask.sum()
                    loss = loss_distill
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    pbar.set_postfix({'loss': tr_loss / nb_tr_steps})
                epoch_avg_loss = tr_loss / max(nb_tr_steps, 1)
                if is_main_process():
                    print(f"Epoch {epoch + 1} Loss: {epoch_avg_loss:.5f} (Best: {min_loss:.5f}, Temp: {current_temperature:.3f})")

                if epoch_avg_loss < min_loss:
                    if is_main_process():
                        print(f" Loss decreased ({min_loss:.5f} -> {epoch_avg_loss:.5f}). Saving model...")
                    min_loss = epoch_avg_loss
                    best_student_state = {k: v.cpu().clone() for k, v in student_net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

                # 早停：每张卡独立判断即可。loss 走势在 DDP 各卡间一致，
                # 即使个别卡提前退出，后续 runner.py 的 barrier() 会同步。
                stop_now = bool(wait >= wait_patient)
                if stop_now:
                    if is_main_process():
                        print(f"Early stopping at epoch {epoch + 1} (wait={wait}, patience={wait_patient}).")
                    break

            if best_student_state is not None:
                student_net.load_state_dict(best_student_state)
            elif is_main_process():
                print("[Warning] No best model found (Training might have failed).")
