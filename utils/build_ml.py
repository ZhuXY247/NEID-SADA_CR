import os
import random
import itertools

import nltk
import numpy as np

# 自动确保 NLTK POS tagger 资源存在，避免运行时 LookupError。
# averaged_perceptron_tagger_eng 是 NLTK 3.8+ 的新名称，旧版本用 averaged_perceptron_tagger。
def _ensure_nltk_resources():
    for resource in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(f"taggers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass  # 离线环境下跳过，运行时若真正需要 with_pos 才会报错

_ensure_nltk_resources()
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler

from utils.adamW import AdamW
from utils.tools import set_seed


def set_args(new_args):
    """设置全局参数并初始化依赖对象。"""
    global args
    args = new_args
    init()


def init():
    """初始化全局分词器、设备与长度配置。"""
    global tokenizer, device, max_length
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        padding=True,
        truncation=True,
        model_max_length=args.internal_max_seq_length,
    )
    device = args.device
    max_length = args.internal_max_seq_length


def seed_everything(seed_value: int) -> None:
    """统一设置各类随机种子（委托给 utils.tools.set_seed）。"""
    set_seed(seed_value)


def tokenize_function(example):
    """对单条样本文本执行分词。"""
    return tokenizer(example["text"], truncation=True,max_length = args.internal_max_seq_length)

def collate_function(data):
    """将样本列表整理为批量张量。"""

    text = [d['text'] for d in data]
    labels = [d['labels'] for d in data]
    input_ids = [d['input_ids'] for d in data]
    token_type_ids = [d['token_type_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]

    
    
    max_len = max([len(input_id) for input_id in input_ids])

    # 使用 tokenizer 的 pad_token_id 进行填充，避免硬编码 BERT 的 [SEP] id(102)。
    pad_id = tokenizer.pad_token_id if tokenizer is not None else 0
    input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id, [pad_id] * (max_len - len(input_id))]) for input_id in input_ids]
    token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type, [0] * (max_len - len(token_type))]) for token_type in token_type_ids]
    attention_mask = [mask if len(mask) == max_len else np.concatenate([mask, [0] * (max_len - len(mask))]) for mask in attention_mask]

    if 'sample_type' in data[0].keys():
        sample_type = [d['sample_type'] for d in data]

        batch = {'input_ids':torch.LongTensor(input_ids),
                'token_type_ids':torch.LongTensor(token_type_ids),
                'attention_mask':torch.LongTensor(attention_mask),
                'sample_type':torch.LongTensor(sample_type),
                'labels':torch.LongTensor(labels),
                'text':text
                }
    else:
        batch = {'input_ids':torch.LongTensor(input_ids),
            'token_type_ids':torch.LongTensor(token_type_ids),
            'attention_mask':torch.LongTensor(attention_mask),
            'labels':torch.LongTensor(labels),
            'text':text
            }

    return batch

def constrative_loss(feature,label,novel_label = None):
    """计算样本级对比损失。"""
    pairwised_consine = torch.cosine_similarity(feature.double().unsqueeze(1),feature.unsqueeze(0),dim = -1)
    pairwised_consine = torch.exp(pairwised_consine)
    pairwised_consine = torch.triu(pairwised_consine,diagonal = 1)

    denominator = torch.sum(pairwised_consine)

    numerator = pairwised_consine[0,0] 
    max_label = torch.max(label)
    for l in range(max_label + 1):
        l_idx = np.array(list(range(0,len(label))))[label.detach().cpu().numpy() == l]
        if len(l_idx) > 1: 
            pair_idx = list(itertools.combinations(l_idx, 2))
            for idx in pair_idx:
                numerator = numerator + pairwised_consine[idx]

    loss = - torch.log( (numerator + 1e-4) / (denominator + 1e-4) ) / len(label)
    
    return loss

def build_optimizer(model,lr,n_epoch,data_loader = None,with_scheduler = False,betas = (0.9,0.999),with_bc=True):
    """构建优化器并按需附加调度器。"""
    optimizer = AdamW(model.parameters(), lr=lr,betas=betas,with_bc = with_bc)

    if with_scheduler:
        if data_loader is None:
            raise ValueError("启用学习率调度器时必须提供 data_loader。")
        # 初始化学习率调度器。
        num_training_steps = len(data_loader) * n_epoch 

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps= int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return optimizer,lr_scheduler
    return optimizer,None

def train_one_epoch_without_mask(model,dataloader,optimizer,device,lr_scheduler=None):
    """执行一轮不带掩码的训练。"""
    progress_bar = tqdm(range( len(dataloader)))

    model.train()

    for batch in dataloader:
        text = batch.pop("text")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    return model

def test_without_mask(model,dataloader, device, cls = None, return_samples = False):
    """在不带掩码设置下评估模型表现。"""
    acc = 0
    model.eval()
    y_true = []
    y_pred = []
    text_list = []
    for batch in tqdm(dataloader):
        
        text = batch.pop("text")
        text_list.extend(text)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        if cls is not None:
            tmp = logits[:,cls]
            logits = torch.ones_like(logits) * (-1e5)
            logits[:,cls] = tmp
            
        predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        if len(y_true) == 0:
            y_true = labels
            y_pred = predictions
        else:
            y_true = np.concatenate([y_true,labels])
            y_pred = np.concatenate([y_pred,predictions])

    if len(y_true) >0:
        acc = accuracy_score(y_true, y_pred)

    if return_samples:
        text_samples = pd.DataFrame({'text':text_list,'predictions':y_pred,'labels':y_true})
        return text_samples,acc
    
    return acc

def get_whole_word_attributions(word_attributions, with_pos = False):
    """将子词级归因合并为整词级归因。

    额外处理方括号说话者标记（[s]/[t]/[S]/[T]）：
    BERT tokenizer 将其切为三个独立 token（'['、's'/'t'、']'）。
    合并时仅取中间字母位的归因值作为该标记的归因，与
    build_whole_word_token_spans 的处理逻辑保持一致。
    """
    sentence = []
    whole_word_attributions = []
    word_idxs = []
    i = -1
    while True:
        whole_word_attribution = []
        word_idx = []
        i = i + 1
        if i >= len(word_attributions):
            break

        token = word_attributions[i][0]

        # 检测说话者方括号标记：[ + s/t + ] → 合并为一个 word，仅取中间字母的归因
        if token == '[' and i + 2 < len(word_attributions):
            middle_token = word_attributions[i + 1][0]
            end_token    = word_attributions[i + 2][0]
            if end_token == ']' and middle_token.lower() in {'s', 't'}:
                whole_word = '[' + middle_token + ']'
                # 仅使用中间字母的归因值（与 build_whole_word_token_spans 一致）
                middle_att = word_attributions[i + 1][1].astype(float)
                sentence.append(whole_word)
                whole_word_attributions.append(np.mean(middle_att))
                word_idxs.append([i + 1])   # 仅记录中间字母位的 index
                i += 2
                continue

        whole_word = token
        whole_word_attribution.append(word_attributions[i][1].astype(float))
        word_idx.append(i)

        while i + 1 < len(word_attributions) and word_attributions[i + 1][0].startswith('##'):
            i = i + 1
            whole_word = whole_word + word_attributions[i][0][2:]
            whole_word_attribution.append(word_attributions[i][1].astype(float))
            word_idx.append(i)

        sentence.append(whole_word)   
        
        whole_word_attributions.append(np.mean(whole_word_attribution))
        word_idxs.append(word_idx)
    
    if word_idxs:
        assert word_idxs[-1][-1] < len(word_attributions), print('word_attributions error')
    
    result = []
    core_sentence = sentence[1:-1]
    core_attributions = whole_word_attributions[1:-1]
    core_word_idxs = word_idxs[1:-1]

    if with_pos:
        # NLTK 在极短输入或异常分词返回下，可能给出空列表或结构不规整对象，
        # 不能假设可直接做 [:, 1] 二维切片。
        pos_pairs = nltk.pos_tag(core_sentence) if core_sentence else []
        allowed_pos = {
            'JJ', 'JJR', 'JJS',
            'NN', 'NNS', 'NNP', 'NNPS',
            'RB', 'RBR', 'RBS',
            'UH',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WRB',
        }
        for whole_word, att, pos_pair, word_idx in zip(core_sentence, core_attributions, pos_pairs, core_word_idxs):
            pos = pos_pair[1] if isinstance(pos_pair, (list, tuple)) and len(pos_pair) > 1 else None
            if pos in allowed_pos:
                result.append([whole_word, att, word_idx])
    else:
        for whole_word, att, word_idx in zip(core_sentence, core_attributions, core_word_idxs):  # 排除 [CLS] 与 [SEP]
            result.append([whole_word, att, word_idx])
    result = np.array(result,dtype=object)
    
    return result


def build_whole_word_token_spans(input_ids, special_tokens_mask, tokenizer):
    """根据 WordPiece 切分结果构造整词 token span。

    额外处理方括号角色标记（如 [t]、[s]、[T]、[S]）：
    BERT tokenizer 将其切成三个独立 token（'['、't'/'s'、']'）。
    对这类标记仅保留中间字母位作为 span，避免把 '[' 与 ']'
    纳入词级噪声概率计算。
    """
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(special_tokens_mask, "tolist"):
        special_tokens_mask = special_tokens_mask.tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    spans = []
    current_span = []
    in_bracket = False  # 是否正在积累方括号标记内的 token
    skip_until = -1

    for pos, (token, is_special) in enumerate(zip(tokens, special_tokens_mask)):
        if pos <= skip_until:
            continue

        if is_special:
            if current_span:
                spans.append(current_span)
                current_span = []
            in_bracket = False
            continue

        # 方括号说话者标记：[ + s/t + ] 仅保留中间字母位
        if token == '[' and not in_bracket:
            next_pos = pos + 1
            end_pos = pos + 2
            if end_pos < len(tokens):
                middle_token = tokens[next_pos]
                end_token = tokens[end_pos]
                if end_token == ']' and middle_token.lower() in {'s', 't'} and not special_tokens_mask[next_pos]:
                    if current_span:
                        spans.append(current_span)
                    spans.append([next_pos])
                    skip_until = end_pos
                    in_bracket = False
                    current_span = []
                    continue

            # 开始积累一个方括号标记
            if current_span:
                spans.append(current_span)
            current_span = [pos]
            in_bracket = True
            continue

        if in_bracket:
            # 继续把内容（'t'/'s'/...）和右括号（']'）合并进当前 span
            current_span.append(pos)
            if token == ']':
                # 方括号标记结束，提交这个整体 span
                spans.append(current_span)
                current_span = []
                in_bracket = False
            continue

        # 普通 WordPiece 续词（'##xxx'）
        if token.startswith('##') and current_span:
            current_span.append(pos)
        else:
            if current_span:
                spans.append(current_span)
            current_span = [pos]

    if current_span:
        spans.append(current_span)
    return spans


def build_batch_whole_word_token_spans(input_ids_batch, special_tokens_mask_batch, tokenizer):
    """为 batch 构造整词 token span。"""
    batch_spans = []
    for input_ids, special_tokens_mask in zip(input_ids_batch, special_tokens_mask_batch):
        batch_spans.append(build_whole_word_token_spans(input_ids, special_tokens_mask, tokenizer))
    return batch_spans

    
