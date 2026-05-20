"""阶段一：外部数据预训练管理器。"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.parallel import is_main_process, unwrap_parallel_model, wrap_model
from utils.tools import set_seed
from models import BertForModel


class ExternalPretrainModelManager:
    """管理外部数据上的监督预训练过程。"""

    def __init__(self, args, external_train_loader, external_eval_loader, external_num_labels):
        """初始化外部预训练所需的模型与优化器。"""
        set_seed(args.seed)
        if getattr(args, "distributed", False):
            self.device = args.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = external_train_loader
        self.eval_dataloader = external_eval_loader
        self.num_labels = external_num_labels

        self.model = BertForModel(args.bert_model, num_labels=self.num_labels, device=self.device)
        self.model = wrap_model(self.model, self.device, distributed=getattr(args, "distributed", False))
        if self.model is None:
            raise ValueError("External pretrain model initialization failed.")

        self.num_train_optimization_steps = int(
            len(self.train_dataloader.dataset) / args.pretrain_batch_size) * args.num_pretrain_epochs

        self.optimizer, self.scheduler = self.get_optimizer(args)

        self.best_eval_score = 0

    def eval(self, args):
        """在外部验证集上评估当前模型。"""
        model = self.model
        if model is None:
            raise ValueError("外部预训练模型不可用，无法执行评估。")
        model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        for batch in tqdm(self.eval_dataloader, desc="External Eval Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                logits = model(X)["logits"]
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(float(accuracy_score(y_true, y_pred)) * 100, 2)

        return acc

    def train(self, args):
        """执行外部数据上的监督预训练。"""
        wait = 0
        best_model = None

        for epoch in trange(int(args.num_pretrain_epochs), desc="External Pre-train Epoch"):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            model = self.model
            if model is None:
                raise ValueError("外部预训练模型不可用，无法继续训练。")
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="External Train Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch[:4]
                X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                with torch.set_grad_enabled(True):
                    logits = model(X)["logits"]
                    base_model = unwrap_parallel_model(model)
                    if base_model is None:
                        raise ValueError("External pretrain base model is unavailable.")
                    loss_src = base_model.loss_ce(logits, label_ids)
                    lossTOT = loss_src

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
                print('external_pretrain_loss', loss)

            eval_score = self.eval(args)
            if is_main_process():
                print('external_pretrain_score', eval_score)

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
        """构建外部预训练使用的优化器与调度器。"""
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        if self.model is None:
            raise ValueError("External pretrain model is unavailable for optimizer setup.")
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
