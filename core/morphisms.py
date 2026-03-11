"""
MRT-12 Morphisms (态射模块)
=========================

实现范畴论中的态射概念：h_next = f(h, x)
将传统的神经网络层重构为纯函数式的几何变换

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
Based on: 英雄的范畴论框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RiemannianMorphism(nn.Module):
    """
    黎曼态射：定义流形上的基础变换
    这是MRT-12 的"钥匙"函数式表达
    
    数学定义：m: H × X → H
    其中 H 是流形状态空间，X 是输入空间
    """
    
    def __init__(self, d_model, experts=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.experts = experts
        
        # 高阶函数分发器：根据语境选择旋转逻辑
        self.context_dispatch = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, experts),
            nn.Softmax(dim=-1)
        )
        
        # 专家态射库：每个专家都是一个独立的几何变换
        self.expert_morphisms = nn.ModuleList([
            self._create_expert_morphism() for _ in range(experts)
        ])
        
        # 态射组合参数
        self.composition_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def _create_expert_morphism(self):
        """创建单个专家态射"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.SiLU(),
            nn.Linear(self.d_model * 4, self.d_model * 2),
            nn.SiLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
    
    def forward(self, h, x):
        """
        执行态射变换：h_next = m(h, x)
        
        Args:
            h: 当前流形状态 (B, D)
            x: 输入特征 (B, D)
            
        Returns:
            h_next: 下一状态 (B, D)
        """
        # 1. 上下文感知分发
        context = torch.cat([h, x], dim=-1)
        expert_weights = self.context_dispatch(context)  # (B, E)
        
        # 2. 并行执行所有专家态射
        expert_outputs = []
        for i, expert in enumerate(self.expert_morphisms):
            # 每个专家接收不同的上下文变换
            expert_input = h + x * (i / self.experts)  # 轻微扰动避免退化
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)
        
        # 3. 加权组合专家输出
        expert_stack = torch.stack(expert_outputs, dim=-2)  # (B, E, D)
        weighted_output = torch.sum(expert_stack * expert_weights.unsqueeze(-1), dim=-2)
        
        # 4. 态射组合：线性插值确保平滑过渡
        composition_factor = self.composition_gate(context).squeeze(-1)  # (B,)
        h_next = (1 - composition_factor.unsqueeze(-1)) * h + composition_factor.unsqueeze(-1) * weighted_output
        
        # 5. 正则化和稳定化
        h_next = self.norm(h_next)
        h_next = self.dropout(h_next)
        h_next = F.normalize(h_next, p=2, dim=-1)  # 保持单位球约束
        
        return h_next

class ParallelMorphismScan(nn.Module):
    """
    并行态射扫描：实现函数式编程中的fold操作
    将序列转换为语义轨迹的范畴映射
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.morphism = RiemannianMorphism(d_model, experts=4)
        
    def forward(self, x_seq):
        """
        并行扫描执行：fold(morphism, initial_state, sequence)
        
        Args:
            x_seq: 输入序列 (B, T, D)
            
        Returns:
            h_trajectory: 语义轨迹 (B, T, D)
        """
        B, T, D = x_seq.shape
        
        # 初始化状态球
        h_current = torch.zeros(B, D, device=x_seq.device)
        
        # 并行化处理（实际实现中可优化为真正的并行扫描）
        trajectory = []
        for t in range(T):
            x_t = x_seq[:, t, :]  # 当前时间步输入
            h_current = self.morphism(h_current, x_t)
            trajectory.append(h_current)
        
        return torch.stack(trajectory, dim=1)

class CategoricalConsistencyLoss(nn.Module):
    """
    范畴一致性损失：确保态射保持结构
    这是MRT-12 超越传统模型的关键
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, h1, h2, similarity_threshold=0.8):
        """
        计算范畴一致性损失
        
        Args:
            h1, h2: 两个应该保持同构的状态 (B, D)
            similarity_threshold: 相似度阈值
            
        Returns:
            consistency_loss: 一致性约束损失
        """
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(h1, h2, dim=-1)
        
        # 同义词应该映射到相近的状态
        consistency_loss = F.relu(similarity_threshold - cos_sim)
        
        return consistency_loss.mean()

# 便捷函数：数值稳定的并行扫描
def parallel_stable_scan(theta, alpha):
    """
    数值稳定的并行扫描实现（Log-Space）
    这是函数式fold操作的核心优化
    """
    B, T, D = theta.shape
    
    # 转换到float32确保数值稳定
    t_f32 = theta.to(torch.float32)
    a_f32 = alpha.to(torch.float32).clamp(0.01, 0.95)
    
    # Log-space计算避免exp爆炸
    log_1_a = torch.log(1 - a_f32)
    cum_log = torch.cumsum(log_1_a, dim=1)
    
    # 指数偏移技巧
    x = t_f32 * a_f32 * torch.exp(-cum_log)
    path = torch.cumsum(x, dim=1) * torch.exp(cum_log)
    
    return path.to(torch.bfloat16)