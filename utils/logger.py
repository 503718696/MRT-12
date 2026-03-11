#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 工业级日志管理器
自动滚动、可选 TensorBoard 集成、硬盘保护

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import json
from datetime import datetime
import shutil
import glob

class LogosLogger:
    """工业级日志管理器：自动滚动、可选TensorBoard集成、硬盘保护"""
    
    def __init__(self, log_dir="logs", max_mb=50, backup_count=3, 
                 enable_tensorboard=True, max_tensorboard_size_gb=2.0):
        """
        初始化日志管理器
        Args:
            log_dir: 日志目录
            max_mb: 单个日志文件最大大小(MB)
            backup_count: 保留的备份文件数量
            enable_tensorboard: 是否启用TensorBoard
            max_tensorboard_size_gb: TensorBoard日志最大大小(GB)
        """
        self.log_dir = log_dir
        self.enable_tensorboard = enable_tensorboard
        self.max_tensorboard_bytes = max_tensorboard_size_gb * 1024**3
        self.writer = None
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 配置主日志器
        self.logger = logging.getLogger("MRT_Trainer")
        self.logger.setLevel(logging.INFO)
        
        # 清除已存在的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建滚动文件处理器：自动覆盖旧日志，保护硬盘空间
        log_file = os.path.join(log_dir, "train.log")
        handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_mb * 1024 * 1024,  # 最大大小
            backupCount=backup_count        # 备份数量
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        
        # 同时输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        # TensorBoard写入器（可选）
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard")
                self.writer = SummaryWriter(tb_dir)
                self.logger.info("TensorBoard enabled")
                # 检查并清理TensorBoard日志
                self._manage_tensorboard_storage()
            except ImportError:
                self.logger.warning("TensorBoard not available, continuing without it")
                self.writer = None
        else:
            self.writer = None
            
        self.step_count = 0
        self._log_storage_info()

    def _log_storage_info(self):
        """记录存储相关信息"""
        try:
            # 获取磁盘使用情况
            disk_usage = shutil.disk_usage(self.log_dir)
            free_space_gb = disk_usage.free / 1024**3
            self.logger.info(f"Storage available: {free_space_gb:.1f}GB free space")
            
            if free_space_gb < 5.0:
                self.logger.warning("Low disk space warning: less than 5GB available")
        except Exception as e:
            self.logger.warning(f"Failed to check disk space: {e}")

    def _manage_tensorboard_storage(self):
        """管理TensorBoard存储空间"""
        if not self.writer:
            return
            
        tb_dir = os.path.join(self.log_dir, "tensorboard")
        if not os.path.exists(tb_dir):
            return
            
        try:
            # 计算TensorBoard日志总大小
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(tb_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            self.logger.info(f"TensorBoard logs size: {total_size/1024**3:.2f}GB")
            
            # 如果超过限制，清理旧的日志
            if total_size > self.max_tensorboard_bytes:
                self.logger.warning(f"TensorBoard logs exceed limit ({self.max_tensorboard_bytes/1024**3:.1f}GB), "
                                  f"current size: {total_size/1024**3:.2f}GB")
                
                # 简单清理策略：删除最旧的event文件
                event_files = glob.glob(os.path.join(tb_dir, "**/events.out.tfevents.*"), recursive=True)
                if len(event_files) > 3:  # 保留最近3个
                    event_files.sort(key=os.path.getmtime)
                    for old_file in event_files[:-3]:
                        try:
                            os.remove(old_file)
                            self.logger.info(f"Removed old TensorBoard log: {old_file}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {old_file}: {e}")
                            
        except Exception as e:
            self.logger.warning(f"Failed to manage TensorBoard storage: {e}")

    def _cleanup_temp_files(self):
        """清理临时文件"""
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*_cache_*",
            "tmp_*"
        ]
        
        cleaned_count = 0
        for pattern in temp_patterns:
            temp_files = glob.glob(os.path.join(self.log_dir, pattern))
            for temp_file in temp_files:
                try:
                    if os.path.isfile(temp_file):
                        os.remove(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to clean temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned {cleaned_count} temporary files")

    def log_step(self, step, loss, vram_gb=None, learning_rate=None, 
                 grad_norm=None, throughput=None):
        """
        记录训练步骤信息
        Args:
            step: 当前步数
            loss: 损失值
            vram_gb: 显存使用量(GB)
            learning_rate: 学习率
            grad_norm: 梯度范数
            throughput: 吞吐量(tokens/sec)
        """
        self.step_count = step
        
        # 控制台实时显示（不写入硬盘）
        display_info = f"Step {step:6d} | Loss {loss:.4f}"
        if vram_gb is not None:
            display_info += f" | VRAM {vram_gb:.1f}G"
        if learning_rate is not None:
            display_info += f" | LR {learning_rate:.2e}"
        if grad_norm is not None:
            display_info += f" | GradNorm {grad_norm:.3f}"
        if throughput is not None:
            display_info += f" | Throughput {throughput:.0f} tok/s"
            
        print(f"\r{display_info}", end="", flush=True)
        
        # TensorBoard记录（二进制格式，节省空间）
        if self.writer is not None:
            self.writer.add_scalar("Loss/step", loss, step)
            if vram_gb is not None:
                self.writer.add_scalar("Hardware/VRAM_GB", vram_gb, step)
            if learning_rate is not None:
                self.writer.add_scalar("Training/Learning_Rate", learning_rate, step)
            if grad_norm is not None:
                self.writer.add_scalar("Training/Gradient_Norm", grad_norm, step)
            if throughput is not None:
                self.writer.add_scalar("Performance/Throughput", throughput, step)

    def log_milestone(self, msg, extra_data=None):
        """记录重要里程碑事件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[MILESTONE] {msg}"
        
        # 写入硬盘日志
        self.logger.info(log_msg)
        
        # 控制台输出
        print(f"\n[{timestamp}] {log_msg}")
        
        # 额外数据记录到JSON文件
        if extra_data:
            milestone_file = os.path.join(self.log_dir, "milestones.jsonl")
            record = {
                "timestamp": timestamp,
                "step": self.step_count,
                "message": msg,
                "data": extra_data
            }
            try:
                with open(milestone_file, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                self.logger.warning(f"Failed to write milestone: {e}")

    def log_config(self, config_dict):
        """记录训练配置"""
        config_file = os.path.join(self.log_dir, "config.json")
        try:
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def periodic_cleanup(self):
        """定期清理任务"""
        self._cleanup_temp_files()
        if self.enable_tensorboard:
            self._manage_tensorboard_storage()

    def close(self):
        """关闭日志器"""
        if self.writer is not None:
            self.writer.close()
        
        # 执行最终清理
        self.periodic_cleanup()
        
        logging.shutdown()
        
        # 显示最终存储统计
        try:
            disk_usage = shutil.disk_usage(self.log_dir)
            used_space_gb = (disk_usage.total - disk_usage.free) / 1024**3
            self.logger.info(f"Final storage usage: {used_space_gb:.2f}GB")
        except Exception as e:
            self.logger.warning(f"Failed to get final storage stats: {e}")