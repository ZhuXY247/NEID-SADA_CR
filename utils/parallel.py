import builtins
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


def is_distributed():
    """判断当前是否已初始化分布式环境。"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """获取当前进程的分布式 rank。"""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size():
    """获取当前分布式总进程数。"""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process():
    """判断当前进程是否为主进程。"""
    return get_rank() == 0


def barrier():
    """在分布式环境下执行同步屏障。"""
    if is_distributed():
        dist.barrier()


def setup_for_distributed(is_main):
    """按主进程身份重写打印行为，同时保留原始 print 供 numba 等 JIT 框架使用。"""
    import builtins
    original_print = builtins.print

    # 把原始 print 存到 builtins 模块属性上，供 numba 调用前临时恢复
    builtins._original_print = original_print

    def distributed_print(*args, **kwargs):
        """按主进程权限控制打印输出。"""
        force = kwargs.pop("force", False)
        if is_main or force:
            original_print(*args, **kwargs)

    builtins.print = distributed_print


def init_distributed_mode(args):
    """根据环境变量初始化分布式训练。"""
    args.distributed = False
    args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    args.rank = int(os.environ.get("RANK", getattr(args, "rank", 0)))
    args.world_size = int(os.environ.get("WORLD_SIZE", getattr(args, "world_size", 1)))

    if args.world_size > 1:
        if not torch.cuda.is_available():
            raise ValueError("Distributed training requires CUDA in this project.")

        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        # 显式设置超时为 7200s（2小时），覆盖 LLM 在线请求（最长 llm_timeout=1200s × 3次重试）
        # 默认 1800s 在 rank 0 阻塞于 LLM HTTP 请求时会被 c10d rendezvous 心跳判定失联
        import datetime
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=7200),
        )
        dist.barrier()
        setup_for_distributed(args.rank == 0)


def cleanup_distributed():
    """清理分布式进程组。"""
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_parallel_model(model):
    """返回 DataParallel 或 DDP 包装模型内部的原始模块。"""
    if hasattr(model, "module"):
        return model.module
    return model


def get_model_backbone(model):
    """获取模型的骨干网络对象。"""
    raw_model = unwrap_parallel_model(model)
    # 兼容两种输入：
    # 1) 封装模型（如 CLBert），其 backbone 位于 raw_model.backbone
    # 2) 原生 backbone（如 BertForMaskedLM），对象本身即骨干
    if hasattr(raw_model, "backbone"):
        return raw_model.backbone
    return raw_model


def load_wrapped_model(model, save_path):
    """加载包装模型的参数。"""
    unwrap_parallel_model(model).load_model(save_path)


def save_wrapped_model(model, save_path):
    """保存包装模型的完整参数。"""
    unwrap_parallel_model(model).save_model(save_path)


def save_wrapped_backbone(model, save_path):
    """保存包装模型的骨干参数。"""
    unwrap_parallel_model(model).save_backbone(save_path)


def get_wrapped_state_dict(model):
    """获取包装模型的状态字典。"""
    return unwrap_parallel_model(model).state_dict()


def wrap_model(model, device, distributed=False):
    """按运行模式包装模型为并行版本。"""
    if distributed:
        return DistributedDataParallel(model,
                                    device_ids=[device.index], 
                                    output_device=device.index,
                                    find_unused_parameters=True)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def rank0_broadcast_object(obj):
    """将主进程对象广播到所有进程。"""
    object_list = [obj if is_main_process() else None]
    if is_distributed():
        dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


def _gather_tensor_lengths(tensor):
    """收集各进程张量的首维长度。"""
    local_length = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
    gathered_lengths = [torch.zeros_like(local_length) for _ in range(get_world_size())]
    dist.all_gather(gathered_lengths, local_length)
    return [item.item() for item in gathered_lengths]


def gather_tensor(tensor):
    """聚合所有进程上的张量。"""
    if not is_distributed():
        return tensor

    lengths = _gather_tensor_lengths(tensor)
    max_length = max(lengths)
    if max_length == 0:
        shape = (0,) + tuple(tensor.shape[1:])
        return tensor.new_empty(shape)

    if tensor.size(0) < max_length:
        pad_shape = (max_length - tensor.size(0),) + tuple(tensor.shape[1:])
        padding = tensor.new_zeros(pad_shape)
        padded_tensor = torch.cat([tensor, padding], dim=0)
    else:
        padded_tensor = tensor

    gathered = [torch.zeros_like(padded_tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded_tensor)

    chunks = []
    for gathered_tensor, length in zip(gathered, lengths):
        chunks.append(gathered_tensor[:length])
    return torch.cat(chunks, dim=0)


def gather_features_labels(features, labels):
    """聚合所有进程上的特征与标签。"""
    if not is_distributed():
        return features, labels
    return gather_tensor(features), gather_tensor(labels)


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_samples=None, replacement=True, seed=0):
        """初始化分布式加权采样器。"""
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
        self.seed = seed
        self.rank = get_rank()
        self.num_replicas = get_world_size()
        self.num_samples = num_samples or int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        """返回当前轮次的采样索引迭代器。"""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator
        )
        indices = sampled[self.rank:self.total_size:self.num_replicas].tolist()
        return iter(indices)

    def __len__(self):
        """返回当前进程负责的样本数。"""
        return self.num_samples

    def set_epoch(self, epoch):
        """设置当前采样轮次。"""
        self.epoch = epoch
