import numpy as np
import torch
import torch.nn.functional as F

class PrototypeManager:
    """
    维护每个类别的原型向量，支持动量更新。
    已知类原型由标注数据初始化，未知类原型由聚类中心初始化。
    """
    def __init__(self, num_classes, feat_dim, momentum=0.9, device='cpu'):
        """初始化类别原型及其元信息。"""
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.device = device
        # 原型矩阵，L2归一化存储
        self.prototypes = torch.zeros(num_classes, feat_dim).to(device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool).to(device)
        self.prototype_confidence = torch.zeros(num_classes, dtype=torch.float32).to(device)
        self.prototype_source = ["uninitialized" for _ in range(num_classes)]
        self.cluster_summary = [None for _ in range(num_classes)]
        self.cluster_version = torch.zeros(num_classes, dtype=torch.long).to(device)

    def init_from_labeled(self, features, labels, known_label_list):
        """用有标签数据初始化已知类原型"""
        for label_idx in range(len(known_label_list)):
            mask = (labels == label_idx)
            if mask.sum() > 0:
                class_feat = features[mask].mean(dim=0)
                self.prototypes[label_idx] = F.normalize(
                    class_feat, dim=0)
                self.initialized[label_idx] = True
                self.prototype_confidence[label_idx] = 1.0
                self.prototype_source[label_idx] = "labeled_init"

    def init_unknown_from_kmeans(self, km_centers, known_cls_num):
        """用k-means中心初始化未知类原型"""
        num_unknown = self.num_classes - known_cls_num
        for i in range(num_unknown):
            proto_idx = known_cls_num + i
            center = torch.FloatTensor(km_centers[i]).to(self.device)
            self.prototypes[proto_idx] = F.normalize(center, dim=0)
            self.initialized[proto_idx] = True
            self.prototype_confidence[proto_idx] = 0.5
            self.prototype_source[proto_idx] = "kmeans_init"

    def set_unknown_prototypes(self, prototypes, known_cls_num,
                               confidences=None, summaries=None,
                               source="llm_refined"):
        """设置未知类原型及其附属信息，支持动态簇数量变化。

        当 LLM 的 merge/split 决策导致未知簇数量改变时，自动调整
        prototypes 矩阵、initialized、confidence 等状态向量的 unknown 区间。
        """
        new_unknown_count = prototypes.size(0)
        new_total = known_cls_num + new_unknown_count

        # 如果簇数量发生了变化，重建相关状态向量
        if new_total != self.num_classes:
            old_known_protos = self.prototypes[:known_cls_num].clone()
            old_known_init = self.initialized[:known_cls_num].clone()
            old_known_conf = self.prototype_confidence[:known_cls_num].clone()
            old_known_source = self.prototype_source[:known_cls_num]
            old_known_summary = self.cluster_summary[:known_cls_num]
            old_known_version = self.cluster_version[:known_cls_num].clone()

            self.num_classes = new_total
            self.prototypes = torch.zeros(new_total, self.feat_dim).to(self.device)
            self.initialized = torch.zeros(new_total, dtype=torch.bool).to(self.device)
            self.prototype_confidence = torch.zeros(new_total, dtype=torch.float32).to(self.device)
            self.cluster_version = torch.zeros(new_total, dtype=torch.long).to(self.device)
            self.prototype_source = ["uninitialized"] * new_total
            self.cluster_summary = [None] * new_total

            # 恢复已知类部分
            self.prototypes[:known_cls_num] = old_known_protos
            self.initialized[:known_cls_num] = old_known_init
            self.prototype_confidence[:known_cls_num] = old_known_conf
            self.prototype_source[:known_cls_num] = old_known_source
            self.cluster_summary[:known_cls_num] = old_known_summary
            self.cluster_version[:known_cls_num] = old_known_version

        normalized = F.normalize(prototypes.to(self.device), dim=1)
        self.prototypes[known_cls_num:] = normalized
        self.initialized[known_cls_num:] = True
        self.cluster_version[known_cls_num:] += 1
        self.prototype_source[known_cls_num:] = [source] * new_unknown_count

        if confidences is None:
            self.prototype_confidence[known_cls_num:] = 1.0
        else:
            self.prototype_confidence[known_cls_num:] = confidences.to(self.device)

        if summaries is not None:
            for offset, summary in enumerate(summaries):
                if known_cls_num + offset < new_total:
                    self.cluster_summary[known_cls_num + offset] = summary

    def update_unknown_from_assignments(self, features, assignments, confidences, known_cls_num):
        """依据伪分配结果更新未知类原型。"""
        with torch.no_grad():
            unknown_mask = (assignments >= known_cls_num) & (confidences > 0)
            if unknown_mask.sum() == 0:
                return

            assigned_features = features[unknown_mask]
            assigned_labels = assignments[unknown_mask]
            assigned_conf = confidences[unknown_mask].to(features.device)
            for label_idx in range(known_cls_num, self.num_classes):
                mask = assigned_labels == label_idx
                if mask.sum() == 0:
                    continue
                weights = assigned_conf[mask].unsqueeze(1)
                weight_sum = weights.sum().clamp_min(1e-6)
                batch_mean = F.normalize((assigned_features[mask] * weights).sum(dim=0) / weight_sum, dim=0)
                if self.initialized[label_idx]:
                    updated = self.momentum * self.prototypes[label_idx] + (1 - self.momentum) * batch_mean
                    self.prototypes[label_idx] = F.normalize(updated, dim=0)
                else:
                    self.prototypes[label_idx] = batch_mean
                    self.initialized[label_idx] = True
                self.prototype_source[label_idx] = "assignment_update"
                avg_conf = float(assigned_conf[mask].mean().item())
                self.prototype_confidence[label_idx] = max(float(self.prototype_confidence[label_idx].item()), avg_conf)

    def update(self, features, labels):
        """动量更新原型，避免剧烈波动"""
        with torch.no_grad():
            for label_idx in range(self.num_classes):
                mask = (labels == label_idx)
                if mask.sum() > 0:
                    batch_mean = F.normalize(
                        features[mask].mean(dim=0), dim=0)
                    if self.initialized[label_idx]:
                        self.prototypes[label_idx] = (
                            self.momentum * self.prototypes[label_idx] +
                            (1 - self.momentum) * batch_mean
                        )
                    else:
                        self.prototypes[label_idx] = batch_mean
                        self.initialized[label_idx] = True
                    # 保持L2归一化
                    self.prototypes[label_idx] = F.normalize(
                        self.prototypes[label_idx], dim=0)
                    self.prototype_source[label_idx] = "momentum_update"
                    self.prototype_confidence[label_idx] = max(float(self.prototype_confidence[label_idx].item()), 0.5)

    def update_known_from_pseudo(self, features, labels, confidences, known_cls_num):
        """使用高置信度伪已知标签更新已知类原型。"""
        with torch.no_grad():
            valid_mask = (labels >= 0) & (labels < known_cls_num) & (confidences > 0)
            if valid_mask.sum() == 0:
                return

            valid_features = features[valid_mask]
            valid_labels = labels[valid_mask]
            valid_confidences = confidences[valid_mask].to(features.device)

            for label_idx in range(known_cls_num):
                mask = valid_labels == label_idx
                if mask.sum() == 0:
                    continue
                weights = valid_confidences[mask].unsqueeze(1)
                weight_sum = weights.sum().clamp_min(1e-6)
                batch_mean = F.normalize((valid_features[mask] * weights).sum(dim=0) / weight_sum, dim=0)
                if self.initialized[label_idx]:
                    updated = self.momentum * self.prototypes[label_idx] + (1 - self.momentum) * batch_mean
                    self.prototypes[label_idx] = F.normalize(updated, dim=0)
                else:
                    self.prototypes[label_idx] = batch_mean
                    self.initialized[label_idx] = True
                avg_conf = float(valid_confidences[mask].mean().item())
                self.prototype_source[label_idx] = "pseudo_update"
                self.prototype_confidence[label_idx] = max(float(self.prototype_confidence[label_idx].item()), avg_conf)

    def get_prototypes(self):
        """返回当前全部类别原型。"""
        return self.prototypes
    
class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        """初始化记忆银行缓存。"""
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        """执行带权重的 KNN 检索预测。"""
        # 执行带权近邻检索。
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        """执行标准 KNN 检索预测。"""
        # 执行标准近邻检索。
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        """挖掘每个样本的最近邻索引。"""
        # 为每个样本挖掘 top-k 最近邻。
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        # faiss-gpu 使用 StandardGpuResources，faiss-cpu 不含该接口，需安全回退。
        if torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except AttributeError:
                pass  # faiss-cpu 安装时不含 GPU 接口，静默回退到 CPU 索引。
        index.add(features)
        distances, indices = index.search(features, topk + 1)  # 检索结果中包含样本自身。
        
        # 评估近邻标签一致性。
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0)  # 评估时排除样本自身。
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        """重置记忆银行写入指针。"""
        self.ptr = 0 
        
    def update(self, features, targets):
        """向记忆银行写入一批特征与标签。超出容量时截断并打印警告。"""
        b = features.size(0)
        remaining = self.n - self.ptr
        if b > remaining:
            print(
                f"[MemoryBank] Warning: batch size {b} exceeds remaining capacity "
                f"{remaining}. Truncating to fit."
            )
            b = remaining
        if b <= 0:
            return
        self.features[self.ptr:self.ptr + b].copy_(features.detach()[:b])
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach()[:b])
        self.ptr += b

    def to(self, device):
        """将记忆银行迁移到指定设备。"""
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        """将记忆银行迁移到 CPU。"""
        self.to('cpu')

    def cuda(self):
        """将记忆银行迁移到默认 CUDA 设备。"""
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, device=None):
    """使用数据加载器填充记忆银行。"""
    model.eval()
    memory_bank.reset()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, batch in enumerate(loader):

            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            feature = model(X, output_hidden_states=True)["hidden_states"]

            memory_bank.update(feature, label_ids)
            if i % 100 == 0:
                print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
