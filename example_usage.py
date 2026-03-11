#!/usr/bin/env python3
"""
MRT-12 使用示例脚本

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
from core.model_mrt12 import MRT12_Universal
from core.manifold_ops import RMSNorm, parallel_lerp_scan
from utils.logger import LogosLogger
from utils.checkpoint import CheckpointManager
import time


def example_1_basic_model():
    """示例 1: 基础模型使用"""
    print("=== 示例 1: 基础模型使用 ===")
    
    model = MRT12_Universal(vocab_size=1000, d_model=512, num_layers=6)
    print(f"模型参数数量：{model.get_num_params():,}")
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
        print(f"输入形状：{input_ids.shape}")
        print(f"输出形状：{logits.shape}")
    
    print("✅ 基础模型使用完成\n")


def example_2_logger_usage():
    """示例 2: 日志系统使用"""
    print("=== 示例 2: 日志系统使用 ===")
    
    logger = LogosLogger(
        log_dir="example_logs",
        max_mb=10,
        backup_count=2
    )
    
    logger.log_config({"example": "test", "batch_size": 32})
    logger.log_milestone("开始模拟训练")
    
    for step in range(5):
        fake_loss = 3.0 - step * 0.2
        fake_vram = 8.0 + step * 0.5
        logger.log_step(step=step, loss=fake_loss, vram_gb=fake_vram, learning_rate=1e-4)
        time.sleep(0.1)
    
    logger.log_milestone("模拟训练完成", {"final_loss": 2.0})
    logger.close()
    
    print("✅ 日志系统使用完成\n")


def example_3_checkpoint_management():
    """示例 3: 检查点管理"""
    print("=== 示例 3: 检查点管理 ===")
    
    ckpt_manager = CheckpointManager(
        checkpoint_dir="example_checkpoints",
        max_checkpoints=3,
        save_every_n_steps=2
    )
    
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in [1, 2, 3, 4, 5]:
        metrics = {"loss": 2.0 - step * 0.1, "accuracy": 0.5 + step * 0.05}
        
        if step % 2 == 0:
            ckpt_manager.save_checkpoint(model=model, optimizer=optimizer, step=step, metrics=metrics)
    
    checkpoints = ckpt_manager.list_checkpoints()
    print(f"保存的检查点数量：{len(checkpoints)}")
    
    print("✅ 检查点管理完成\n")


if __name__ == "__main__":
    try:
        example_1_basic_model()
        example_2_logger_usage()
        example_3_checkpoint_management()
        print("🎉 所有示例运行完成！")
    except Exception as e:
        print(f"\n❌ 示例运行失败：{e}")
        import traceback
        traceback.print_exc()
