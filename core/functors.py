"""
MRT-12 Functors (函子模块)
========================

实现范畴论中的函子概念：范畴间的自然变换
将字符序列范畴映射到语义轨迹范畴

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .morphisms import RiemannianMorphism, ParallelMorphismScan, parallel_stable_scan

class CausalFunctor(nn.Module):
    """
    因果函子：C_char → C_concept
    将字符序列范畴映射到概念轨迹范畴
    
    数学性质：
    1. 保持因果结构（无未来信息泄露）
    2. 实现函数式fold操作
    3. 维护流形几何约束
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 态射生成器：产生变换参数
        self.morphism_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )
        
        # 因果绑定器：字符到概念的拓扑连接
        # 使用分组卷积保持通道独立性
        self.concept_binder = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0),  # 左填充确保因果性
            nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False),
            nn.SiLU()
        )
        
        # 流形稳定器：保持球面约束
        self.manifold_stabilizer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        函子映射：字符序列 → 语义轨迹
        
        Args:
            x: 字符嵌入序列 (B, T, D)
            
        Returns:
            concept_trajectory: 概念轨迹 (B, T, D)
        """
        B, T, D = x.shape
        
        # 1. 产生态射参数（钥匙生成）
        morphism_params = self.morphism_generator(x)  # (B, T, 2*D)
        theta, alpha = torch.chunk(morphism_params, 2, dim=-1)
        
        # 2. 并行关联扫描（函数式 fold 操作）
        # 这是MRT-12 的核心加速算子
        semantic_path = parallel_stable_scan(
            torch.tanh(theta), 
            torch.sigmoid(alpha)
        )
        
        # 3. 字符到概念的绑定（拓扑连接）
        # 通过因果卷积实现邻域信息融合
        concept_binding = self.concept_binder(
            semantic_path.transpose(1, 2)
        ).transpose(1, 2)
        
        # 4. 流形结构稳定
        stabilized_trajectory = self.manifold_stabilizer(
            semantic_path + 0.1 * concept_binding
        )
        
        # 5. 正则化
        output = self.dropout(stabilized_trajectory)
        output = F.normalize(output, p=2, dim=-1)  # 保持单位球约束
        
        # 恢复原始数据类型
        output = output.to(original_dtype)
        
        return output

class SemanticTrajectoryFunctor(nn.Module):
    """
    语义轨迹函子：C_concept → C_world
    将概念轨迹映射到世界模型范畴
    """
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        
        # 多头注意力实现概念间的关系建模
        self.multihead_attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.1
        )
        
        # 概念演化器：模拟时间维度上的语义漂移
        self.concept_evolver = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, concept_trajectory, causal_mask=None):
        """
        语义轨迹演化：概念 → 世界模型
        
        Args:
            concept_trajectory: 概念轨迹 (B, T, D)
            causal_mask: 因果掩码 (T, T)
            
        Returns:
            world_model: 世界模型表示 (B, T, D)
        """
        # 1. 概念间关系建模（自注意力）
        attended, _ = self.multihead_attention(
            concept_trajectory, 
            concept_trajectory, 
            concept_trajectory,
            attn_mask=causal_mask,
            need_weights=False
        )
        attended = self.norm1(concept_trajectory + attended)
        
        # 2. 概念演化（时间维度上的语义漂移）
        evolved = self.concept_evolver(attended)
        world_model = self.norm2(attended + evolved)
        
        return F.normalize(world_model, p=2, dim=-1)

class CategoryBinder(nn.Module):
    """
    范畴绑定器：实现跨范畴的自然变换
    确保从字符到概念到世界的结构保持
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.char_to_concept = CausalFunctor(d_model)
        self.concept_to_world = SemanticTrajectoryFunctor(d_model)
        
        # 范畴间的信息流动控制器
        self.inter_category_flow = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 控制信息混合比例
        )
        
    def create_causal_mask(self, seq_len, device):
        """创建因果掩码防止未来信息泄露"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
        
    def forward(self, char_sequence):
        """
        完整的范畴绑定流程：C_char → C_concept → C_world
        
        Args:
            char_sequence: 字符序列 (B, T, D)
            
        Returns:
            world_representation: 世界模型表示 (B, T, D)
        """
        B, T, D = char_sequence.shape
        
        # 1. 字符范畴 → 概念范畴
        concept_trajectory = self.char_to_concept(char_sequence)
        
        # 2. 概念范畴 → 世界范畴
        causal_mask = self.create_causal_mask(T, char_sequence.device)
        world_model = self.concept_to_world(concept_trajectory, causal_mask)
        
        # 3. 范畴间信息融合
        flow_control = self.inter_category_flow(concept_trajectory)
        unified_representation = (
            flow_control * concept_trajectory + 
            (1 - flow_control) * world_model
        )
        
        return F.normalize(unified_representation, p=2, dim=-1)