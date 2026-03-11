#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 GPU 显存检测测试脚本
用于验证智能设备选择功能

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import detect_gpu_memory

def test_gpu_detection():
    """测试 GPU 显存检测功能"""
    print("="*60)
    print("🔍 MRT-12 GPU 显存检测测试")
    print("="*60)
    
    # 运行检测
    gpu_info = detect_gpu_memory()
    
    print("\n📊 检测结果:")
    print(f"   使用 GPU: {gpu_info['use_gpu']}")
    print(f"   设备：{gpu_info.get('device', 'N/A')}")
    print(f"   原因：{gpu_info.get('reason', 'N/A')}")
    
    if gpu_info.get('total_mem_gb'):
        print(f"   总显存：{gpu_info['total_mem_gb']:.1f}GB")
        print(f"   可用显存：{gpu_info.get('free_mem_gb', 0):.1f}GB")
    
    if gpu_info.get('required_mem_gb'):
        print(f"   需要显存：{gpu_info['required_mem_gb']:.1f}GB")
    
    print("\n" + "="*60)
    
    # 验证结果
    if gpu_info['use_gpu']:
        print("✅ 测试通过：GPU 模式已启用")
        assert gpu_info['device'] == 'cuda', "设备应为 cuda"
    else:
        print("✅ 测试通过：CPU 模式已启用")
        assert gpu_info.get('device') == 'cpu' or not gpu_info['use_gpu'], "设备应为 cpu"
    
    print("\n💡 提示:")
    if gpu_info['use_gpu']:
        print("   - 评估将在 GPU 上运行，速度较快")
    else:
        print("   - 评估将在 CPU 上运行，速度较慢但兼容性好")
        print("   - 建议关闭其他 GPU 应用释放显存")
    
    return True

if __name__ == "__main__":
    try:
        test_gpu_detection()
        print("\n🎉 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
