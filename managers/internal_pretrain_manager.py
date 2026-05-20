"""阶段一：内部多任务预训练管理器。"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from utils.parallel import is_main_process, unwrap_parallel_model, wrap_model
from utils.tools import mask_tokens, set_seed
from models import BertForModel


class InternalPretrainModelManager:
    """管理内部标注数据上的多任务预训练。"""
    def __init__(self, args, data):
        """初始化内部预训练模型与训练状态。"""
        set_seed(args.seed)
        if getattr(args, "distributed", False):
            self.device = args.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = getattr(data, 'class_weights', None)
        if weights is not None:
            weights = weights.to(self.device)
        self.model = BertForModel(
            args.bert_model,
            num_labels=data.n_known_cls,
            device=self.device,
            loss_weights=weights
        )
        self.model = wrap_model(self.model, self.device,
                                distributed=getattr(args, "distributed", False))
        if self.model is None:
            raise ValueError("Internal pretrain model initialization failed.")
        
        self.num_train_optimization_steps = len(data.train_labeled_dataloader) * args.num_pretrain_epochs

        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.best_eval_score = 0
        
    def eval(self, args, data):
        """在验证集上计算分类准确率。"""
        model = self.model
        if model is None:
            raise ValueError("内部预训练模型不可用，无法执行评估。")
        model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)
        
        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                logits = model(X)["logits"]
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
        
        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(float(accuracy_score(y_true, y_pred)) * 100, 2)

        return acc
        
    def train(self, args, data):
        """执行内部监督分类与 MLM 联合预训练。"""
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        wait = 0
        best_model = None
        mlm_iter = iter(data.train_semi_train_dataloader)
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            data.set_epoch(epoch)
            model = self.model
            if model is None:
                raise ValueError("内部预训练模型不可用，无法继续训练。")
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(data.train_labeled_dataloader, desc="Iteration")):
                # 1. 加载监督分类与 MLM 所需批数据。
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch[:4]
                X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                try:
                    batch = next(mlm_iter)
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids = batch[:3]
                except StopIteration:
                    mlm_iter = iter(data.train_semi_train_dataloader)
                    batch = next(mlm_iter)
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids = batch[:3]
                X_mlm = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}

                # 2. 构造 MLM 掩码输入。
                mask_ids, mask_lb = mask_tokens(X_mlm['input_ids'].cpu(), tokenizer)
                X_mlm["input_ids"] = mask_ids.to(self.device)

                # 3. 计算分类损失与 MLM 损失，并更新参数。
                # 两路 forward 均通过 base_model（unwrapped）调用，避免触发
                # DDP 的 forward hook 计数。DDP 只通过参数的 .grad_fn hook 在
                # backward 时做一次 allreduce，每个参数仅被标记一次 ready。
                with torch.set_grad_enabled(True):
                    base_model = unwrap_parallel_model(model)
                    if base_model is None:
                        raise ValueError("Internal pretrain base model is unavailable.")
                    logits = base_model(X)["logits"]
                    loss_src = base_model.loss_ce(logits, label_ids)
                    loss_mlm = base_model.mlmForward(X_mlm, mask_lb.to(self.device))
                    lossTOT = loss_src + loss_mlm
                    lossTOT.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    tr_loss += lossTOT.item()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            if is_main_process():
                print('train_loss',loss)
            
            eval_score = self.eval(args, data)
            if is_main_process():
                print('score', eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model
        
    def get_optimizer(self, args):
        """构建内部预训练的优化器与调度器。"""
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        if self.model is None:
            raise ValueError("Internal pretrain model is unavailable for optimizer setup.")
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
        
    def get_features_labels(self, dataloader, model, args):
        """提取表示向量及其对应标签。"""
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels
