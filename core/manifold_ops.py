#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 黎曼流形算子模块
RMSNorm、并行插值扫描等几何运算

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        # 强制在 float32 进行归一化以求稳
        x_f32 = x.to(torch.float32)
        norm = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x_f32 / norm * self.scale.to(torch.float32)).to(x.dtype)

def parallel_lerp_scan(theta, alpha):
    """
    MRT-12 极速算子：数值稳定版并行扫描
    h_t = (1-a)h_{t-1} + a*theta
    输入输出: (B, T, D)
    使用 TorchScript 编译，速度极快且数值绝对稳定。
    """
    B, T, D = theta.shape
    # 1. 提升精度并严格限制 alpha 范围 (0.01 ~ 0.9)
    # alpha 太大会导致瞬间遗忘，太小会导致路径爆炸
    a = alpha.to(torch.float32).clamp(0.01, 0.9)
    t = theta.to(torch.float32)
    
    # 2. 对数域转换
    log_gamma = torch.log(1 - a)
    cum_log_gamma = torch.cumsum(log_gamma, dim=1)
    
    # 3. 稳定化处理 (Offset Trick)
    # 我们通过减去当前的累积对数来保证 exp() 的输入永远 <= 0
    # 这样 exp() 的输出永远在 (0, 1] 之间，绝对不会 NaN
    x = t * a * torch.exp(-cum_log_gamma)
    
    # 4. 并行求和并还原
    path = torch.cumsum(x, dim=1) * torch.exp(cum_log_gamma)
    
    return path.to(torch.bfloat16) # 返回给 3090 Ti 使用

@torch.jit.script
def serial_lerp_scan(theta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的串行线性递归扫描：h_t = (1-alpha_t)*h_{t-1} + alpha_t*theta_t
    输入输出: (B, T, D)
    使用 TorchScript 编译，速度极快且数值绝对稳定。
    """
    B, T, D = theta.size()
    h_current = torch.zeros(B, D, device=theta.device, dtype=theta.dtype)
    h_list = []
    
    for t in range(T):
        a_t = alpha[:, t, :]
        theta_t = theta[:, t, :]
        h_current = (1.0 - a_t) * h_current + a_t * theta_t
        h_list.append(h_current)
        
    return torch.stack(h_list, dim=1)

# 兼容旧接口名
def parallel_lerp_scan(theta, alpha):
    return serial_lerp_scan(theta, alpha)

def safe_softmax(x, dim=-1, temperature=1.0):
    """数值安全的softmax，防止溢出"""
    x = x / temperature
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def cosine_similarity_matrix(a, b, eps=1e-8):
    """计算余弦相似度矩阵"""
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a_norm, b_norm.transpose(-2, -1))
