import torch
import torch.nn as nn

class PrototypeContrastiveLoss(nn.Module):
    """
    原型对比损失：以类原型作为正例锚点，天然消除类别数量不平衡的影响。
    每个类在分母中只贡献一项，与样本数量无关。
    """
    def __init__(self, temperature=0.07):
        """初始化原型对比损失温度参数。"""
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels, confidences=None):
        """
        参数说明：
            features: 样本嵌入 [bsz, feat_dim]，已做 L2 归一化。
            prototypes: 类别原型 [num_classes, feat_dim]，已做 L2 归一化。
            labels: 样本标签 [bsz]，-1 表示无标签样本。
        返回：
            标量损失值。
        """
        device = features.device
        
        # 只计算有标签样本的原型对比损失
        labeled_mask = (labels >= 0)
        if labeled_mask.sum() == 0:
            # 返回挂在当前 batch 计算图上的零，避免 DDP/AMP 下出现 no-grad backward。
            return features.sum() * 0.0
        
        labeled_features = features[labeled_mask]  
        labeled_labels = labels[labeled_mask]       
        
        # 计算每个样本与所有原型的相似度
        # 结果张量形状为 [有标签样本数, 类别数]。
        sim_matrix = torch.matmul(labeled_features, 
                                   prototypes.T) / self.temperature
        
        # 数值稳定性
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # 正例：与自己所属类的原型
        pos_sim = sim_matrix[
            torch.arange(len(labeled_labels), device=device),
            labeled_labels
        ]
        
        # 分母：与所有原型的相似度之和
        log_denom = torch.log(torch.exp(sim_matrix).sum(dim=1))
        
        loss_vector = -(pos_sim - log_denom)
        if confidences is not None:
            valid_confidences = confidences[labeled_mask].to(device)
            normalizer = valid_confidences.sum().clamp_min(1e-6)
            return (loss_vector * valid_confidences).sum() / normalizer

        return loss_vector.mean()


class UnknownPrototypeAssignmentLoss(nn.Module):
    """基于簇归属的未知类原型分配损失。"""

    def __init__(self, temperature=0.07):
        """初始化未知类原型分配损失。"""
        super(UnknownPrototypeAssignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, assignments, confidences=None):
        """计算未知类样本到原型的分配损失。"""
        device = features.device
        valid_mask = assignments >= 0
        if valid_mask.sum() == 0:
            # 返回挂在当前 batch 计算图上的零，避免 DDP/AMP 下出现 no-grad backward。
            return features.sum() * 0.0

        valid_features = features[valid_mask]
        valid_assignments = assignments[valid_mask]
        sim_matrix = torch.matmul(valid_features, prototypes.T) / self.temperature
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        pos_sim = sim_matrix[
            torch.arange(len(valid_assignments), device=device),
            valid_assignments
        ]
        log_denom = torch.log(torch.exp(sim_matrix).sum(dim=1))
        loss_vector = -(pos_sim - log_denom)

        if confidences is not None:
            valid_confidences = confidences[valid_mask].to(device)
            normalizer = valid_confidences.sum().clamp_min(1e-6)
            return (loss_vector * valid_confidences).sum() / normalizer

        return loss_vector.mean()
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        """初始化监督对比损失配置。"""
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """当 `labels` 与 `mask` 都为空时，退化为 SimCLR 风格的无监督对比损失。

        参数说明：
            features: 形状为 [bsz, n_views, ...] 的隐藏表示。
            labels: 形状为 [bsz] 的真实标签。
            mask: 形状为 [bsz, bsz] 的对比掩码，若样本 j 与样本 i 同类则为 1。
        返回：
            标量损失值。
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            if mask is None:
                raise ValueError("对比学习掩码不能为空。")
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算样本间相似度分数。
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # 为提升数值稳定性，减去每行最大值。
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 扩展掩码以匹配多视图展开后的维度。
        mask = mask.repeat(anchor_count, contrast_count)
        # 屏蔽样本与自身的对比项。
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算对数概率。
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算正样本对的平均对数似然。
        # 某些极端 batch（例如某些 anchor 没有任何正例）会导致 mask.sum(1)=0，
        # 若直接相除会产生 NaN。这里显式过滤无正例 anchor。
        pos_count = mask.sum(1)
        valid_anchor_mask = pos_count > 0
        if not valid_anchor_mask.any():
            # 返回挂在当前图上的零，保证 backward 在 DDP/AMP 下稳定。
            return contrast_feature.sum() * 0.0

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchor_mask] / pos_count[valid_anchor_mask]

        # 计算最终损失（仅在有效 anchor 上归约）。
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
