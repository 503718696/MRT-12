#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 核心模型架构
流形循环变换器的完整实现

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .manifold_ops import RMSNorm, parallel_lerp_scan
import torch.utils.checkpoint as checkpoint


class CausalConv1d(nn.Module):
    """
    终极因果卷积修复版：使用手动左侧padding确保绝对无未来信息泄露。
    原理：在输入左侧填充 (kernel_size - 1) 个零，卷积层本身不自动填充。
    保证每个时间步 t 的输出仅依赖于 t, t-1, ..., t-k+1。
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        # 关闭自动填充，由我们手动控制
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=0, groups=groups)

    def forward(self, x):
        # x: (B, D, T)
        # 手动在左侧填充 (kernel_size - 1) 个零
        x_padded = F.pad(x, (self.kernel_size - 1, 0), mode='constant', value=0)
        return self.conv(x_padded)


class MRT12_Layer(nn.Module):
    """MRT-12单层架构：字符->概念跃迁的核心计算单元"""
    def __init__(self, d_model, layer_idx):
        super().__init__()
        self.d_model = d_model
        
        # 关键生成器：同时生成路径参数θ和混合系数α
        self.key_gen = nn.Linear(d_model, d_model * 2, bias=False)
        
        # 概念绑定器：通过因果卷积捕捉局部依赖关系，防止未来信息泄露
        self.binder = CausalConv1d(d_model, d_model, kernel_size=3, groups=d_model)
        
        # 概念激活器：非线性变换增强表达能力
        self.concept_act = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 归一化层
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # 生成路径参数
        res = self.key_gen(x)
        theta, alpha = torch.chunk(res, 2, dim=-1)
        
        # 演化路径计算：字符级别到概念级别的跃迁
        h_path = parallel_lerp_scan(torch.tanh(theta) * math.pi, torch.sigmoid(alpha))
        h_path = h_path.to(x.dtype)
        
        # 概念绑定：通过因果卷积捕捉序列中的局部模式，仅依赖历史信息
        h_bound = self.binder(h_path.transpose(1, 2)).transpose(1, 2)
        
        # 最终概念表示：路径演化 + 局部绑定 + 非线性激活
        h_final = self.concept_act(h_path + h_bound)
        
        # 归一化，移除层级缩放
        return self.norm(h_final)


class MRT12_Universal(nn.Module):
    """MRT-12通用拓扑模型：完整的语言模型架构"""
    def __init__(self, vocab_size, d_model=2048, num_layers=24):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 减小初始化方差
        nn.init.normal_(self.embedding.weight, std=0.002)
        
        # 位置编码：可学习的位置嵌入
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.002)
        
        # 多层MRT架构
        self.layers = nn.ModuleList([
            MRT12_Layer(d_model, i) for i in range(num_layers)
        ])
        
        # 可塑性突触：最后4层的神经可塑性连接
        self.plastic_synapses = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model, d_model)) 
            for _ in range(4)
        ])
        
        # 最终归一化
        self.final_norm = RMSNorm(d_model)
        
        # 语言模型头部
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定：输出层与嵌入层共享权重
        self.lm_head.weight = self.embedding.weight
        
        # 添加缺失的dropout层
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        """
        前向传播（启用梯度检查点以节省显存）
        Args:
            input_ids: [batch_size, seq_len] 输入token IDs
        Returns:
            logits: [batch_size, seq_len, vocab_size] 预测logits
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入 + 位置编码 + L2归一化
        x = F.normalize(
            self.embedding(input_ids) + self.pos_emb[:, :seq_len, :], 
            p=2, dim=-1
        )

        # 定义 checkpoint 封装函数
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # 逐层处理并应用梯度检查点
        for i, layer in enumerate(self.layers):
            delta = checkpoint.checkpoint(create_custom_forward(layer), x, use_reentrant=False) \
                if self.training else layer(x)

            # 神经可塑性机制：最后4层启用可塑性连接
            if i >= self.num_layers - 4:
                plastic_idx = i - (self.num_layers - 4)
                adaptation = torch.matmul(delta, self.plastic_synapses[plastic_idx])
                x = x + delta + adaptation
            else:
                x = x + delta

        # 最终输出处理
        x = self.final_norm(x)
        x = self.dropout(x)

        # 强力RMSNorm防止极端logit值
        rms_scale = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-8)
        x = x * rms_scale * 0.8

        logits = self.lm_head(x)

        # 输出logits归一化
        logits = logits / torch.std(logits, dim=-1, keepdim=True).clamp(min=1.0)
        logits = torch.clamp(logits, -10.0, 10.0)

        return logits

    def count_parameters(self):
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """文本生成函数"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    values, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[:, [-1]]] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估计模型FLOPs利用率"""
        # 假设每个token需要6*N*d^2 FLOPs (N=层数, d=隐藏维度)
        N = self.num_layers
        d = self.d_model
        L = 1024  # 序列长度假设
        flops_per_token = 6 * N * d**2 + 12 * N * d * L
        flops_per_iter = flops_per_token * L * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        # A100峰值性能约312 TFLOPS
        flops_promised = 312e12
        return flops_achieved / flops_promised
