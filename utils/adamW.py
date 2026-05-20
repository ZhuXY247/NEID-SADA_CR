import math
import torch
from torch.optim.optimizer import Optimizer

class AdamW(Optimizer):
    """实现 Adam 优化算法。

    该算法最早发表于《Adam: A Method for Stochastic Optimization》。

    参数说明：
        params (iterable): 待优化参数，或参数组字典。
        lr (float, optional): 学习率，默认 1e-3。
        betas (Tuple[float, float], optional): 一阶矩与二阶矩滑动平均系数。
        eps (float, optional): 提升数值稳定性的分母常数项。
        weight_decay (float, optional): 权重衰减系数。
        amsgrad (boolean, optional): 是否启用 AMSGrad 变体。

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2,with_bc = True, amsgrad=False):
        """初始化 AdamW 优化器配置。"""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.with_bc = with_bc
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        """恢复优化器状态并补齐默认字段。"""
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """执行一次参数更新。

        参数说明：
            closure (callable, optional): 可选闭包，用于重新计算损失。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # 初始化优化器状态。
                if len(state) == 0:
                    state['step'] = 0
                    # 梯度一阶矩的指数滑动平均。
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 梯度平方二阶矩的指数滑动平均。
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # 记录历史最大二阶矩，供 AMSGrad 使用。
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                max_exp_avg_sq = None
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # 更新一阶矩与二阶矩的滑动平均。
                exp_avg.mul_(beta1).add_(1 - beta1, grad) 
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    if max_exp_avg_sq is None:
                        raise ValueError("AMSGrad 模式下缺少 max_exp_avg_sq 状态。")
                    # 更新截至当前的最大二阶矩。
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # 使用最大二阶矩进行归一化。
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                if self.with_bc:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(-group['lr'],  torch.mul(p.data, group['weight_decay']))
                    p.data.addcdiv_(-group['lr'] * step_size, exp_avg, denom)
                else:
                    p.data.add_(-group['lr'],  torch.mul(p.data, group['weight_decay']))
                    p.data.addcdiv_(-group['lr'], exp_avg, denom)
        return loss
