#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 智能检查点管理器
自动保存、断点续训、硬盘空间管理

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import os
import json
import shutil
from datetime import datetime
import psutil

class CheckpointManager:
    """智能检查点管理器：自动保存、断点续训、硬盘空间管理"""
    
    def __init__(self, checkpoint_dir="checkpoints", max_checkpoints=5, 
                 save_every_n_steps=1000, max_total_size_gb=10.0):
        """
        初始化检查点管理器
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保留检查点数量
            save_every_n_steps: 每多少步保存一次
            max_total_size_gb: 检查点总大小限制(GB)
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_every_n_steps = save_every_n_steps
        self.max_total_size_bytes = max_total_size_gb * 1024**3
        
        # 创建检查点目录
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # 跟踪已保存的检查点
        self.saved_checkpoints = []
        self._load_existing_checkpoints()
        
        # 磁盘空间监控
        self._check_disk_space()

    def _check_disk_space(self):
        """检查磁盘空间并发出警告"""
        try:
            # 获取磁盘使用情况
            disk_usage = psutil.disk_usage(self.checkpoint_dir)
            free_space_gb = disk_usage.free / 1024**3
            total_space_gb = disk_usage.total / 1024**3
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            print(f"💾 磁盘空间状态: {free_space_gb:.1f}GB 可用 / {total_space_gb:.1f}GB 总计 ({used_percent:.1f}% 已使用)")
            
            # 发出警告
            if free_space_gb < 5.0:
                print("⚠️  警告: 可用磁盘空间不足5GB，请及时清理!")
            elif free_space_gb < 10.0:
                print("ℹ️  提示: 可用磁盘空间低于10GB，建议清理不必要的文件")
                
        except Exception as e:
            print(f"⚠️  磁盘空间检查失败: {e}")

    def _get_checkpoint_size(self, filepath):
        """获取检查点文件大小"""
        try:
            return os.path.getsize(filepath)
        except:
            return 0

    def _calculate_total_size(self):
        """计算所有检查点的总大小"""
        total_size = 0
        existing_checkpoints = []
        
        for ckpt in self.saved_checkpoints:
            filepath = os.path.join(self.checkpoint_dir, ckpt["filename"])
            if os.path.exists(filepath):
                size = self._get_checkpoint_size(filepath)
                total_size += size
                existing_checkpoints.append({**ckpt, "size_bytes": size})
            else:
                # 文件不存在，从列表中移除
                print(f"⚠️  检查点文件不存在，已移除记录: {filepath}")
        
        self.saved_checkpoints = existing_checkpoints
        return total_size

    def _cleanup_by_size(self):
        """按大小清理检查点，优先删除最旧的"""
        total_size = self._calculate_total_size()
        
        if total_size <= self.max_total_size_bytes:
            return
            
        print(f"🗑️  检查点总大小 {total_size/1024**3:.1f}GB 超过限制 {self.max_total_size_bytes/1024**3:.1f}GB")
        
        # 按时间排序（最旧的优先删除）
        self.saved_checkpoints.sort(key=lambda x: x["step"])
        
        removed_size = 0
        removed_count = 0
        
        for ckpt in self.saved_checkpoints[:]:
            if total_size - removed_size <= self.max_total_size_bytes:
                break
                
            filepath = os.path.join(self.checkpoint_dir, ckpt["filename"])
            if os.path.exists(filepath):
                file_size = self._get_checkpoint_size(filepath)
                try:
                    os.remove(filepath)
                    removed_size += file_size
                    removed_count += 1
                    print(f"  删除检查点: {ckpt['filename']} ({file_size/1024**2:.1f}MB)")
                except Exception as e:
                    print(f"  删除失败: {filepath} - {e}")
            
            self.saved_checkpoints.remove(ckpt)
        
        print(f"✅ 清理完成: 删除 {removed_count} 个检查点，释放 {removed_size/1024**3:.1f}GB 空间")

    def _load_existing_checkpoints(self):
        """加载已存在的检查点信息"""
        info_file = os.path.join(self.checkpoint_dir, "checkpoint_info.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r") as f:
                    info = json.load(f)
                    self.saved_checkpoints = info.get("checkpoints", [])
                # 验证现有检查点并清理
                self._cleanup_by_size()
            except Exception as e:
                print(f"⚠️  加载检查点信息失败: {e}")
                self.saved_checkpoints = []

    def _save_checkpoint_info(self):
        """保存检查点信息"""
        info_file = os.path.join(self.checkpoint_dir, "checkpoint_info.json")
        info = {
            "checkpoints": self.saved_checkpoints,
            "max_checkpoints": self.max_checkpoints,
            "max_total_size_gb": self.max_total_size_bytes / 1024**3,
            "last_updated": datetime.now().isoformat()
        }
        try:
            with open(info_file, "w") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            print(f"⚠️  保存检查点信息失败: {e}")

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，保持数量限制"""
        if len(self.saved_checkpoints) > self.max_checkpoints:
            # 按步数排序，删除最早的
            self.saved_checkpoints.sort(key=lambda x: x["step"])
            to_remove = self.saved_checkpoints[:-self.max_checkpoints]
            
            for ckpt in to_remove:
                ckpt_path = os.path.join(self.checkpoint_dir, ckpt["filename"])
                if os.path.exists(ckpt_path):
                    try:
                        os.remove(ckpt_path)
                        print(f"Removed old checkpoint: {ckpt_path}")
                    except Exception as e:
                        print(f"Failed to remove {ckpt_path}: {e}")
            
            # 保留最新的检查点
            self.saved_checkpoints = self.saved_checkpoints[-self.max_checkpoints:]

    def save_checkpoint(self, model, optimizer, step, epoch=None, 
                       metrics=None, extra_state=None):
        """
        保存检查点
        Args:
            model: 模型对象
            optimizer: 优化器对象
            step: 当前训练步数
            epoch: 当前epoch数
            metrics: 评估指标字典
            extra_state: 额外状态信息
        """
        # 构造检查点文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mrt12_step_{step:06d}_{timestamp}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 处理可能的编译模型前缀
        model_state = model.state_dict()
        # 检查是否有_orig_mod前缀并移除
        if any(key.startswith('_orig_mod.') for key in model_state.keys()):
            model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
            print("Saving compiled model checkpoint, removed _orig_mod prefix")
        
        # 构造检查点数据
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "step": step,
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics or {},
            "extra_state": extra_state or {}
        }
        
        # 保存检查点
        try:
            torch.save(checkpoint, filepath)
            print(f"💾 检查点已保存: {filename}")
        except Exception as e:
            print(f"❌ 检查点保存失败: {e}")
            return None
        
        # 更新跟踪列表
        ckpt_info = {
            "step": step,
            "epoch": epoch,
            "filename": filename,
            "timestamp": timestamp,
            "metrics": metrics or {}
        }
        self.saved_checkpoints.append(ckpt_info)
        
        # 执行清理策略
        self._cleanup_old_checkpoints()  # 数量限制清理
        self._cleanup_by_size()          # 大小限制清理
        self._save_checkpoint_info()     # 保存信息
        
        # 显示存储统计
        total_size = self._calculate_total_size()
        print(f"📊 检查点统计: {len(self.saved_checkpoints)} 个文件, "
              f"总大小 {total_size/1024**3:.1f}GB")
        
        return filepath

    def load_checkpoint(self, filepath, model, optimizer=None, device="cuda"):
        """
        加载检查点（兼容旧版本）
        Args:
            filepath: 检查点文件路径
            model: 模型对象
            optimizer: 优化器对象（可选）
            device: 设备
        Returns:
            checkpoint数据字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载优化器状态（如果提供）
        if optimizer and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        print(f"Checkpoint loaded. Step: {checkpoint['step']}, "
              f"Epoch: {checkpoint.get('epoch', 'N/A')}")
              
        return checkpoint

    def get_latest_checkpoint(self):
        """获取最新的检查点信息"""
        if not self.saved_checkpoints:
            return None
        return max(self.saved_checkpoints, key=lambda x: x["step"])

    def list_checkpoints(self):
        """列出所有检查点"""
        return sorted(self.saved_checkpoints, key=lambda x: x["step"])

    def get_storage_stats(self):
        """获取存储使用统计"""
        total_size = self._calculate_total_size()
        return {
            "checkpoint_count": len(self.saved_checkpoints),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / 1024**3,
            "max_allowed_gb": self.max_total_size_bytes / 1024**3,
            "usage_percent": (total_size / self.max_total_size_bytes) * 100 if self.max_total_size_bytes > 0 else 0
        }