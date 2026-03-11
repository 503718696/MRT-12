#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 第一阶段：大规模知识预训练 (Foundation Training)
目标：使用全量 Wiki/HugeCorpus 构建世界模型底座

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
from utils.common import get_recommended_config, load_checkpoint_safe

def main():
    """主训练函数"""
    # === 自动显存检测和配置选择 ===
    gpu_config = detect_gpu_memory()
    print(f"🔍 检测到GPU配置: {gpu_config['config']}")
    print(f"   总显存: {gpu_config['total_mem_gb']:.1f}GB")
    print(f"   推荐配置: D={gpu_config['d_model']}, L={gpu_config['layers']}, BS={gpu_config['batch_size']}")
    
    # === 配置区域 (终极显存优化) ===
    D_MODEL = gpu_config['d_model']
    N_LAYERS = gpu_config['layers']
    BATCH_SIZE = gpu_config['batch_size']
    LEARNING_RATE = 2e-4  # 保持较低学习率适应小batch
    MAX_STEPS = 100000
    SAVE_EVERY = 5000
    # 终极优化：物理batch=4，累积步数=32，等效batch=128
    GRADIENT_ACCUMULATION_STEPS = 32
    
    # === 设置显存优化环境变量 ===
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TensorFloat-32
    
    # === 初始化日志和检查点管理器 ===
    logger = LogosLogger(
        log_dir="world_model_logs",
        max_mb=50,
        backup_count=3,
        enable_tensorboard=True
    )
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="world_model_checkpoints",
        max_checkpoints=3,  # 减少检查点数量节省存储
        save_every_n_steps=SAVE_EVERY,
        max_total_size_gb=10.0  # 严格控制存储空间
    )
    
    # === 数据准备 ===
    logger.log_milestone("开始数据加载...")
    
    # 使用主训练语料
    data_file = "../zhwiki_dataset.jsonl"  # 2.4GB主语料
    sentences = load_data_final(data_file, max_sentences=50000000)
    
    # 构建或加载词表
    w2i, i2w = build_or_load_vocab(sentences, vocab_file="mrt_vocab.json")
    
    logger.log_milestone(f"数据加载完成: {len(sentences)} 条句子, 词表大小: {len(w2i)}")
    
    # 创建数据加载器（显存优化配置）
    dataset = WikiDataset(sentences, w2i, max_len=256)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=8,  # 减少worker数量降低内存压力
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
    
    # 禁用torch.compile以节省显存（已在模型层实现梯度检查点）
    logger.log_milestone("ℹ️  已启用梯度检查点，跳过torch.compile优化")
    
    # 记录模型参数量
    total_params = model.count_parameters()
    logger.log_milestone(f"模型参数量: {total_params:,}")
    logger.log_milestone(f"终极配置: D={D_MODEL}, L={N_LAYERS}, BS={BATCH_SIZE}")
    logger.log_milestone(f"梯度累积: {GRADIENT_ACCUMULATION_STEPS}步")
    logger.log_milestone(f"等效batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.log_milestone(f"学习率: {LEARNING_RATE}")
    
    # === 优化器和混合精度 ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # === 检查是否从检查点恢复 ===
    latest_ckpt = checkpoint_manager.get_latest_checkpoint()
    start_step = 0
    
    if latest_ckpt:
        try:
            ckpt_path = os.path.join("world_model_checkpoints", latest_ckpt["filename"])
            checkpoint = load_checkpoint_safe(ckpt_path, model, optimizer)
            start_step = checkpoint["step"]
            logger.log_milestone(f"从检查点恢复训练: Step {start_step}")
        except Exception as e:
            logger.log_milestone(f"检查点加载失败，从头开始训练: {e}")
            start_step = 0
    
    # === 训练循环 ===
    logger.log_milestone(">>> MRT-12 底座预训练启动...")
    logger.log_milestone(f"{gpu_config['config']}模式已启用")
    logger.log_milestone("🚀 The Great Cleanse 阶段开始 - 纯净因果预训练")
    model.train()
    
    train_start_time = time.time()
    step_counter = start_step
    accumulation_counter = 0
    
    try:
        while step_counter < MAX_STEPS:
            for batch_idx, batch in enumerate(dataloader):
                if step_counter >= MAX_STEPS:
                    break
                    
                # 数据移到GPU
                batch = batch.to("cuda", non_blocking=True)
                
                # 前向传播 + 混合精度训练
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch)
                    # 计算交叉熵损失（忽略pad token）
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, len(w2i)),
                        batch[:, 1:].reshape(-1),
                        ignore_index=0
                    )
                    # 梯度累积
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # 反向传播
                scaler.scale(loss).backward()
                
                accumulation_counter += 1
                
                # 梯度累积完成，执行优化步骤
                if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 参数更新
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 记录训练进度
                    if step_counter % 20 == 0:
                        elapsed_time = time.time() - train_start_time
                        throughput = (step_counter + 1) * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 256 / (elapsed_time + 1e-6)
                        
                        vram_gb = torch.cuda.memory_allocated() / 1e9
                        learning_rate = optimizer.param_groups[0]['lr']
                        
                        logger.log_step(
                            step=step_counter,
                            loss=loss.item() * GRADIENT_ACCUMULATION_STEPS,  # 恢复真实loss
                            vram_gb=vram_gb,
                            learning_rate=learning_rate,
                            throughput=throughput
                        )
                    
                    # 保存检查点
                    if step_counter % SAVE_EVERY == 0 and step_counter > 0:
                        metrics = {
                            "loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                            "learning_rate": learning_rate,
                            "throughput": throughput,
                            "vram_gb": vram_gb
                        }
                        
                        checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=step_counter,
                            metrics=metrics
                        )
                    
                    step_counter += 1
                
                # 显存清理和监控（更频繁）
                if step_counter % 500 == 0:
                    torch.cuda.empty_cache()
                    # 监控显存峰值
                    vram_peak = torch.cuda.max_memory_allocated() / 1e9
                    if vram_peak > 15.0:  # 更严格的显存阈值
                        logger.log_milestone(f"⚠️  显存使用高峰: {vram_peak:.1f}GB")
                    
    except KeyboardInterrupt:
        logger.log_milestone("训练被用户中断")
    except Exception as e:
        logger.log_milestone(f"训练过程中发生错误: {e}")
        raise
    finally:
        # 保存最终检查点
        final_metrics = {
            "final_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS if 'loss' in locals() else 0.0,
            "steps_completed": step_counter,
            "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9
        }
        
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step_counter,
            metrics=final_metrics
        )
        
        logger.log_milestone("训练完成")
        logger.close()

if __name__ == "__main__":
    main()