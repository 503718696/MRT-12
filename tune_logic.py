#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 第二阶段：逻辑灌顶/指令微调 (Logic Tuning/SFT)
目标：在预训练底座基础上，使用逻辑数据进行高压灌顶训练

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model_mrt12 import MRT12_Universal
from data.dataset import load_data_final, build_or_load_vocab, WikiDataset, pad_collate_fn
from utils.logger import LogosLogger
from utils.checkpoint import CheckpointManager
from utils.common import load_checkpoint_safe

def tune():
    """逻辑微调主函数"""
    # === SFT 专用超参数 ===
    D_MODEL = 2048
    N_LAYERS = 32
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5  # 极低学习率，保护底座知识
    EPOCHS = 10           # 逻辑数据量小，需要多读几遍
    SAVE_EVERY = 1000     # 每1000步保存检查点
    
    # === 初始化日志和检查点管理器 ===
    logger = LogosLogger(
        log_dir="logic_tuning_logs",
        max_mb=50,
        backup_count=3,
        enable_tensorboard=True
    )
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="logic_tuning_checkpoints",
        max_checkpoints=5,
        save_every_n_steps=SAVE_EVERY,
        max_total_size_gb=5.0
    )
    
    # === 数据准备 ===
    logger.log_milestone("开始加载逻辑训练数据...")
    
    # 使用逻辑混合数据（1.9万条黄金数据）
    logic_data_file = "mrt_training_data_final.jsonl"
    sentences = load_data_final(logic_data_file, max_sentences=100000)
    
    # 构建或加载词表（应该与预训练阶段一致）
    w2i, i2w = build_or_load_vocab(sentences, vocab_file="mrt_vocab.json")
    
    logger.log_milestone(f"逻辑数据加载完成: {len(sentences)} 条句子")
    
    # 创建数据加载器
    dataset = WikiDataset(sentences, w2i, max_len=512)  # 更长序列以容纳逻辑推理
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # === 模型初始化 ===
    logger.log_milestone("初始化模型...")
    
    model = MRT12_Universal(
        vocab_size=len(w2i),
        d_model=D_MODEL,
        num_layers=N_LAYERS
    ).to("cuda")
    
    # === 加载预训练底座权重 ===
    base_checkpoint_path = "world_model_checkpoints/mrt12_step_100000_20240101_000000.pth"  # 示例路径
    
    if os.path.exists(base_checkpoint_path):
        logger.log_milestone(f"加载预训练底座权重: {base_checkpoint_path}")
        try:
            load_checkpoint_safe(base_checkpoint_path, model)
            logger.log_milestone("底座权重加载成功")
        except Exception as e:
            logger.log_milestone(f"底座权重加载失败: {e}")
            logger.log_milestone("警告：将从随机初始化开始训练")
    else:
        logger.log_milestone(f"警告：未找到预训练底座权重 {base_checkpoint_path}")
        logger.log_milestone("将从随机初始化开始训练")
    
    # === 优化器和混合精度 ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,  # 更轻的权重衰减
        eps=1e-8
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # === 检查是否从检查点恢复 ===
    latest_ckpt = checkpoint_manager.get_latest_checkpoint()
    start_epoch = 0
    start_step = 0
    
    if latest_ckpt:
        try:
            ckpt_path = os.path.join("logic_tuning_checkpoints", latest_ckpt["filename"])
            checkpoint = load_checkpoint_safe(ckpt_path, model, optimizer)
            start_epoch = checkpoint.get("epoch", 0)
            start_step = checkpoint["step"]
            logger.log_milestone(f"从检查点恢复训练: Epoch {start_epoch}, Step {start_step}")
        except Exception as e:
            logger.log_milestone(f"检查点加载失败，从头开始训练: {e}")
            start_epoch = 0
            start_step = 0
    
    # === 训练循环 ===
    logger.log_milestone(">>> MRT-12 逻辑灌顶阶段开始...")
    model.train()
    
    train_start_time = time.time()
    global_step = start_step
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            
            logger.log_milestone(f"开始 Epoch {epoch + 1}/{EPOCHS}")
            
            for batch_idx, batch in enumerate(dataloader):
                # 数据移到GPU
                batch = batch.to("cuda", non_blocking=True)
                
                # 梯度清零
                optimizer.zero_grad(set_to_none=True)
                
                # 前向传播 + 混合精度训练
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch)
                    # 计算交叉熵损失（忽略pad token）
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, len(w2i)),
                        batch[:, 1:].reshape(-1),
                        ignore_index=0
                    )
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更严格的裁剪
                
                # 参数更新
                scaler.step(optimizer)
                scaler.update()
                
                # 累积统计
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1
                
                # 记录训练进度
                if global_step % 10 == 0:
                    elapsed_time = time.time() - train_start_time
                    throughput = global_step * BATCH_SIZE * 512 / (elapsed_time + 1e-6)
                    
                    vram_gb = torch.cuda.memory_allocated() / 1e9
                    learning_rate = optimizer.param_groups[0]['lr']
                    
                    logger.log_step(
                        step=global_step,
                        loss=loss.item(),
                        vram_gb=vram_gb,
                        learning_rate=learning_rate,
                        throughput=throughput
                    )
                
                # 保存检查点
                if global_step % SAVE_EVERY == 0:
                    metrics = {
                        "loss": loss.item(),
                        "epoch_loss_avg": epoch_loss / epoch_steps if epoch_steps > 0 else 0,
                        "learning_rate": learning_rate,
                        "throughput": throughput
                    }
                    
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        epoch=epoch,
                        metrics=metrics
                    )
                
                # 显存清理
                if global_step % 500 == 0:
                    torch.cuda.empty_cache()
            
            # Epoch结束统计
            epoch_duration = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            
            logger.log_milestone(
                f"Epoch {epoch + 1} 完成 | "
                f"平均Loss: {avg_epoch_loss:.4f} | "
                f"耗时: {epoch_duration/60:.1f}分钟 | "
                f"总步数: {global_step}"
            )
            
            # 保存每个epoch的检查点
            epoch_metrics = {
                "epoch_avg_loss": avg_epoch_loss,
                "epoch_duration_sec": epoch_duration,
                "total_steps": global_step
            }
            
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=global_step,
                epoch=epoch + 1,
                metrics=epoch_metrics
            )
            
    except KeyboardInterrupt:
        logger.log_milestone("逻辑微调被用户中断")
    except Exception as e:
        logger.log_milestone(f"逻辑微调过程中发生错误: {e}")
        raise
    finally:
        # 保存最终检查点
        final_metrics = {
            "final_loss": loss.item() if 'loss' in locals() else 0.0,
            "epochs_completed": epoch + 1 if 'epoch' in locals() else 0,
            "total_steps": global_step
        }
        
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=global_step,
            epoch=EPOCHS,
            metrics=final_metrics
        )
        
        logger.log_milestone("逻辑微调完成")
        logger.close()

if __name__ == "__main__":
    tune()