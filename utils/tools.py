import os
import csv
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment

def set_seed(seed: int) -> None:
    """设置项目全量随机种子，覆盖 Python / NumPy / PyTorch / CUDA。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def hungray_aligment(y_true, y_pred):
    """使用匈牙利算法对齐聚类标签。"""
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    """计算聚类结果的对齐准确率。"""
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    """汇总常用聚类评估指标。"""
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}


class TeacherWrapper(nn.Module):
    def __init__(self, cl_bert_model):
        """将项目模型包装为解释器兼容接口。"""
        super().__init__()
        self.model = cl_bert_model
        backbone = cl_bert_model.backbone

        prefix = getattr(backbone, "base_model_prefix", "bert")
        self.base_model_prefix = prefix

        if hasattr(backbone, prefix):
            setattr(self, prefix, getattr(backbone, prefix))
        else:
            setattr(self, prefix, backbone)

    @property
    def device(self):
        """返回底层模型所在设备。"""
        return self.model.device

    @property
    def config(self):
        """返回补齐标签映射后的模型配置。"""
        cfg = self.model.backbone.config
        cfg.id2label = {i: f"LABEL_{i}" for i in range(self.model.num_labels)}
        cfg.label2id = {v: k for k, v in cfg.id2label.items()}
        return cfg

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
        """将解释器输入转换为项目模型输出。"""
        # 1. 构造 CLBert 需要的参数字典 X
        X = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None:
            X["token_type_ids"] = token_type_ids
        if position_ids is not None:
            X["position_ids"] = position_ids
        outputs = self.model(X, output_attentions=kwargs.get('output_attentions', False))
        return SequenceClassifierOutput(
            loss=None,
            logits=outputs['logits'],
            hidden_states=outputs.get('hidden_states'),
            attentions=outputs.get('attentions'),
        )

    def get_input_embeddings(self):
        """获取底层模型的输入嵌入层。"""
        return self.model.backbone.get_input_embeddings()

def mask_tokens(inputs, tokenizer,
    special_tokens_mask=None, mlm_probability=0.15):
        """
        为掩码语言模型任务构造输入与标签：80% 使用 MASK，10% 使用随机词，10% 保持原词。
        注意：函数内部会对 inputs 做 inplace 修改，调用前请确保传入 clone 后的副本。
        """
        inputs = inputs.clone()   # 防止调用方原 tensor 被 inplace 污染
        labels = inputs.clone()
        # 按设定概率为每个序列采样需要参与 MLM 的词元。
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 仅在被遮蔽位置上计算 MLM 损失。

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels

class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        """初始化训练视图生成器。"""
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids, mlm_probability=0.25):
        """对输入执行随机词替换增强。mlm_probability 控制替换比例。"""
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=mlm_probability)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        """对句内普通词元执行随机打乱增强。"""
        ids = ids.clone()   # 防止 unbind 返回的 view 的 inplace 操作污染原 tensor
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = inp.clone()   # 用 clone 而非 deepcopy，保持 tensor 语义一致
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            new_ids[sent_tokens_inds] = inp[shuffled_inds]   # 写到 new_ids，不改原 inp
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos
