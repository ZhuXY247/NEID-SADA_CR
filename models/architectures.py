import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from utils.contrastive import SupConLoss


"""项目核心模型结构定义。"""

class NoiseGenerator(nn.Module):
    """根据教师信号生成词级噪声概率，并映射到 token 级执行。"""

    def __init__(self, hidden_size, device=None, dropout_prob=0.1,
                 use_transformer=True, num_heads=4, num_layers=2) -> None:
        """初始化词级噪声生成网络。"""
        super(NoiseGenerator, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.use_transformer = use_transformer

        # 输入投影层：对教师隐藏状态做自适应变换，初始化为恒等变换。
        # 让学生可以从教师的表示空间中学习对噪声检测任务有利的特征方向，
        # 而不是被动接受固定的教师表示。
        self.input_proj = nn.Linear(hidden_size, hidden_size)

        if use_transformer:
            # 使用 TransformerEncoder 保留序列建模能力。
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout_prob,
                batch_first=True  # 保持 (batch, seq, feat) 维度顺序。
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, 
                                                  num_layers=num_layers,
                                                  enable_nested_tensor=False)
            self.output_proj = nn.Linear(hidden_size, 1)
        else:
            # 用于消融实验的简单 MLP 分支。
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
        self.reset_parameters()


    def reset_parameters(self):
        """
        初始化策略：
        - input_proj：恒等初始化，保证训练初期行为与无投影层完全一致，
          随训练进行逐渐学习对噪声检测有利的特征变换。
        - output_proj / MLP：Xavier 初始化。
        """
        # 恒等初始化：weight=I，bias=0，不破坏教师隐藏状态的原始信号
        nn.init.eye_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        if self.use_transformer:
            nn.init.xavier_normal_(self.output_proj.weight)
            nn.init.constant_(self.output_proj.bias, 0.0)
        else:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def gumbel_sigmoid_ste(self, logits, temperature=0.1):
        """Gumbel-Sigmoid 采样，使用 STE 将梯度桥接回干净概率。

        前向：用 Gumbel 噪声做随机采样（保持探索性）。
        反向：梯度通过干净的 sigmoid(logits) 传导（避免噪声污染梯度方向）。

        相比原实现，BCE 损失始终使用 sigmoid(logits) 计算，
        Gumbel 采样只影响前向输出分布，不污染损失目标。
        """
        probs_clean = torch.sigmoid(logits)
        if self.training:
            eps = 1e-10
            u1 = torch.rand_like(logits)
            noise = torch.log(u1 + eps) - torch.log(1 - u1 + eps)
            probs_noisy = torch.sigmoid((logits + noise) / temperature)
            # STE：前向用含噪声的采样值，反向梯度走干净概率路径
            probs_ste = probs_clean + (probs_noisy - probs_clean).detach()
            return probs_clean, probs_ste
        else:
            return probs_clean, probs_clean

    def aggregate_word_probs(self, token_probs, word_spans):
        """将 token 级概率聚合为词级概率。"""
        batch_size = token_probs.size(0)
        max_words = max((len(spans) for spans in word_spans), default=0)
        if max_words == 0:
            max_words = 1

        word_probs = torch.zeros(batch_size, max_words, device=token_probs.device)
        word_valid_mask = torch.zeros(batch_size, max_words, device=token_probs.device)
        for batch_idx, spans in enumerate(word_spans):
            for word_idx, span in enumerate(spans):
                if not span:
                    continue
                span_indices = torch.tensor(span, device=token_probs.device, dtype=torch.long)
                word_probs[batch_idx, word_idx] = token_probs[batch_idx, span_indices].mean()
                word_valid_mask[batch_idx, word_idx] = 1.0
        return word_probs.unsqueeze(-1), word_valid_mask.unsqueeze(-1)

    def forward(self, hidden_states, attention_mask=None, special_tokens_mask=None, temperature=0.1,
                noise_ratio=0.5, word_spans=None, return_word_probs=False,
                apply_inference_gate=True):
        """根据隐藏表示生成噪声概率输出。

        训练时返回两路概率：
          - probs_for_loss (clean): sigmoid(logits)，用于 BCE 损失，不含 Gumbel 噪声。
          - probs_for_forward (ste): Gumbel 采样值，STE 梯度桥回 clean，用于前向 token 操作。
        推理时两路相同（均为 sigmoid(logits)）。

        Args:
            apply_inference_gate: 推理时是否应用阈值门控。
                                  True: 输出二值 mask（确定性）
                                  False: 输出原始概率（用于 Bernoulli 采样）
        """
        if attention_mask is None:
            raise ValueError("NoiseGenerator 前向计算必须提供 attention_mask。")

        if self.use_transformer:
            if special_tokens_mask is not None:
                key_padding_mask = (attention_mask <= 0) | (special_tokens_mask > 0)
            else:
                key_padding_mask = attention_mask <= 0
            # 先经过输入投影层做自适应特征变换，再送入 Transformer encoder
            projected = self.input_proj(hidden_states)
            encoded = self.encoder(projected, src_key_padding_mask=key_padding_mask)
            logits = self.output_proj(encoded)
        else:
            logits = self.net(self.input_proj(hidden_states))

        # probs_clean: 干净概率，用于 BCE 损失（无 Gumbel 噪声污染）
        # probs_ste:   STE 采样概率，用于前向 token mask 操作
        probs_clean, probs_ste = self.gumbel_sigmoid_ste(logits, temperature=temperature)

        # 统一用 probs_ste 做前向操作（推理时与 probs_clean 相同）
        probs = probs_ste.squeeze(-1) if probs_ste.dim() == 3 else probs_ste
        probs_clean = probs_clean.squeeze(-1) if probs_clean.dim() == 3 else probs_clean

        valid_token_mask = attention_mask.to(probs.device)
        if special_tokens_mask is not None:
            valid_token_mask = valid_token_mask * (1 - special_tokens_mask.to(probs.device))
        probs = probs * valid_token_mask
        probs_clean = probs_clean * valid_token_mask

        # 推理时的阈值门控：选择噪声比例最高的词作为 mask
        if not self.training and apply_inference_gate:
            valid_len = valid_token_mask.sum(dim=1)
            k_sample = (valid_len.float() * noise_ratio).ceil().long()
            k_sample = torch.clamp(k_sample, min=1)
            _, sorted_indices = torch.sort(probs, dim=1, descending=True)
            # 取前 k_sample 个词为噪声词，生成二值 mask
            mask = torch.zeros_like(probs)
            for b in range(probs.size(0)):
                k = k_sample[b].item()
                mask[b, sorted_indices[b, :k]] = 1.0
            mask = mask * valid_token_mask
            probs = mask
            probs_clean = mask  # 推理时两路一致

        if return_word_probs:
            if word_spans is None:
                raise ValueError("return_word_probs=True 时必须提供 word_spans。")
            # 返回 (word_probs_for_forward, word_valid_mask, word_probs_for_loss)
            word_probs_fwd, word_valid = self.aggregate_word_probs(probs, word_spans)
            word_probs_loss, _ = self.aggregate_word_probs(probs_clean, word_spans)
            return word_probs_fwd, word_valid, word_probs_loss

        probs = probs.unsqueeze(-1)
        return probs

    def save_model(self, save_path):
        """保存当前噪声生成器参数。"""
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        """从文件加载噪声生成器参数。"""
        state_dict = torch.load(save_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)

class BertForModel(nn.Module):
    """用于预训练阶段的 BERT 分类模型。"""

    def __init__(self, model_name, num_labels, device=None, loss_weights=None):
        """初始化预训练阶段使用的分类模型。"""
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.loss_weights = loss_weights
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """前向过程直接返回未经过 softmax 的 logits。"""
        outputs = self.backbone(**X, output_hidden_states=True)
        # 提取最后一层的 [CLS] 表示作为分类输入。
        CLSEmbedding = outputs.hidden_states[-1][:, 0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        """计算掩码语言模型损失。"""
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        """计算分类交叉熵损失。"""
        if self.loss_weights is not None:
            loss = nn.CrossEntropyLoss(weight=self.loss_weights)
        else:
            loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def save_backbone(self, save_path):
        """保存骨干模型参数。"""
        self.backbone.save_pretrained(save_path)
    
    def save_model(self, save_path):
        """保存完整模型参数。"""
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        """加载完整模型参数。"""
        state_dict = torch.load(save_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)


class CLBert(nn.Module):
    """用于对比学习阶段的 BERT 编码模型。"""

    def __init__(self,model_name, device, feat_dim=128, num_labels=2):
        """初始化对比学习阶段的编码模型。"""
        super(CLBert, self).__init__()
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, feat_dim)
        )
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels).to(self.device)
        self.backbone.to(self.device)
        self.head.to(device)

    def forward(self, X, output_hidden_states=False, output_attentions=False, output_logits=False):
        """前向过程直接返回未经过 softmax 的 logits。"""

        # BERT 内部的 position_ids 是 register_buffer，两次 forward 共用同一对象。
        # 若两次 forward 在同一 autograd graph 内（如双视图训练），第二次 forward
        # 内部对 position_ids buffer 的 inplace 操作会破坏第一次 forward 保存的引用，
        # 触发 "EmbeddingBackward0: modified by an inplace operation" 错误。
        # 修复：显式构造独立的 position_ids tensor 传入，绕过 BERT 对内部 buffer 的共享引用。
        seq_len = X["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=X["input_ids"].device).unsqueeze(0)
        X = dict(X, position_ids=position_ids)

        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features, "logits": self.classifier(cls_embed)}
        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=None):
        """计算对比学习损失。"""
        # base_temperature 默认与 temperature 保持一致，确保缩放系数恒为 1.0；
        # 若显式传入则尊重调用方意图。
        if base_temperature is None:
            base_temperature = temperature
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output

    def save_backbone(self, save_path):
        """保存编码骨干参数。"""
        self.backbone.save_pretrained(save_path)

    def save_model(self, save_path):
        """保存完整对比学习模型参数。"""
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        """加载完整对比学习模型参数。"""
        state_dict = torch.load(save_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
