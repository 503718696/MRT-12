#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 公共工具函数
包含 GPU 检测、检查点加载等跨模块复用功能

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import os
import torch


def detect_gpu_memory(required_mem_gb=10.0):
    """
    检测 GPU 显存并判断是否足够加载模型
    
    Args:
        required_mem_gb: 模型所需显存（GB），默认 10.0
        
    Returns:
        dict: 包含 use_gpu, device, total_mem_gb, free_mem_gb 等信息
    """
    if not torch.cuda.is_available():
        print("⚠️  未检测到 CUDA 设备，将使用 CPU 模式")
        return {"use_gpu": False, "reason": "CUDA unavailable"}
    
    try:
        device = torch.cuda.current_device()
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated(device) / 1024**3
        free_mem_gb = total_mem_gb - allocated_mem
        
        print(f"🔍 GPU 检测:")
        print(f"   总显存：{total_mem_gb:.1f}GB")
        print(f"   已用显存：{allocated_mem:.1f}GB")
        print(f"   可用显存：{free_mem_gb:.1f}GB")
        print(f"   需要显存：{required_mem_gb:.1f}GB")
        
        if free_mem_gb >= required_mem_gb:
            print(f"✅ 显存充足，使用 GPU 模式")
            return {
                "use_gpu": True,
                "device": "cuda",
                "total_mem_gb": total_mem_gb,
                "free_mem_gb": free_mem_gb,
                "reason": "Sufficient VRAM"
            }
        else:
            print(f"⚠️  显存不足（剩余{free_mem_gb:.1f}GB < 需要{required_mem_gb:.1f}GB），将使用 CPU 模式")
            return {
                "use_gpu": False,
                "device": "cpu",
                "total_mem_gb": total_mem_gb,
                "free_mem_gb": free_mem_gb,
                "required_mem_gb": required_mem_gb,
                "reason": "Insufficient VRAM"
            }
            
    except Exception as e:
        print(f"⚠️  GPU 检测失败：{str(e)}，将使用 CPU 模式")
        return {"use_gpu": False, "reason": f"GPU detection error: {str(e)}"}


def load_checkpoint_safe(filepath, model, optimizer=None, device="cuda"):
    """
    安全加载检查点，处理 torch.compile 前缀问题
    
    Args:
        filepath: 检查点文件路径
        model: PyTorch 模型
        optimizer: 优化器（可选）
        device: 加载设备
        
    Returns:
        checkpoint: 检查点字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    state_dict = checkpoint["model_state_dict"]
    
    has_compile_prefix = any(key.startswith('_orig_mod.') for key in state_dict.keys())
    
    if has_compile_prefix:
        print("Detected compiled model checkpoint, removing _orig_mod prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        print("Loading regular model checkpoint...")
    
    model.load_state_dict(state_dict, strict=False)
    
    if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}")
    
    print(f"Checkpoint loaded. Step: {checkpoint['step']}, Epoch: {checkpoint.get('epoch', 'N/A')}")
    return checkpoint


def get_recommended_config():
    """
    根据 GPU 配置推荐训练参数
    
    Returns:
        dict: 包含 config, d_model, layers, batch_size 等配置
    """
    if not torch.cuda.is_available():
        return {"config": "cpu", "d_model": 768, "layers": 12, "batch_size": 2}
    
    device = torch.cuda.current_device()
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    if total_mem_gb < 16:
        return {"config": "low_memory", "d_model": 1024, "layers": 16, "batch_size": 4}
    elif total_mem_gb < 24:
        return {"config": "ultra_conservative", "d_model": 2048, "layers": 32, "batch_size": 4}
    else:
        return {"config": "high_performance", "d_model": 2048, "layers": 32, "batch_size": 8}
