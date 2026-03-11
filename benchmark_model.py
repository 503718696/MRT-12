#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 模型性能基准测试
测试不同配置下的模型性能指标

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_mrt12 import MRT12_Universal


def benchmark_config(vocab_size, d_model, num_layers, batch_size, seq_len, device="cuda"):
    """测试特定配置的性能"""
    
    # 创建模型
    model = MRT12_Universal(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers
    ).to(device)
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # 预热
    warmup_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    with torch.no_grad():
        _ = model(warmup_input)
    
    # 基准测试
    num_iterations = 10
    times = []
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            output = model(input_ids)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
    
    # 计算统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    tokens_per_sec = (batch_size * seq_len) / avg_time
    
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'device': device,
        'total_params': total_params,
        'avg_latency_ms': avg_time * 1000,
        'min_latency_ms': min_time * 1000,
        'max_latency_ms': max_time * 1000,
        'throughput_tokens_sec': tokens_per_sec,
    }


def print_benchmark(results):
    """打印基准测试结果"""
    print("\n" + "="*80)
    print("  MRT-12 模型性能基准测试")
    print("="*80)
    
    for i, r in enumerate(results, 1):
        print(f"\n[测试 {i}]")
        print(f"  配置：vocab={r['vocab_size']}, d_model={r['d_model']}, layers={r['num_layers']}")
        print(f"  Batch Size: {r['batch_size']}, Seq Len: {r['seq_len']}")
        print(f"  设备：{r['device'].upper()}")
        print(f"  参数量：{r['total_params']:,}")
        print(f"  平均延迟：{r['avg_latency_ms']:.2f} ms")
        print(f"  最小延迟：{r['min_latency_ms']:.2f} ms")
        print(f"  最大延迟：{r['max_latency_ms']:.2f} ms")
        print(f"  吞吐量：{r['throughput_tokens_sec']:,.0f} tokens/sec")
    
    print("\n" + "="*80)


def main():
    """主测试函数"""
    print("🔍 检测硬件环境...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备：{device.upper()}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print("\n开始性能基准测试...\n")
    
    test_configs = [
        # (vocab_size, d_model, num_layers, batch_size, seq_len)
        (1000, 256, 4, 4, 32),      # 小型配置
        (5000, 512, 8, 4, 64),      # 中型配置
        (8000, 768, 12, 2, 128),    # 大型配置
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n测试配置：vocab={config[0]}, d_model={config[1]}, layers={config[2]}, BS={config[3]}, seq={config[4]}")
        
        try:
            result = benchmark_config(*config, device=device)
            results.append(result)
            print(f"✅ 完成 - 吞吐量：{result['throughput_tokens_sec']:,.0f} tok/s")
        except Exception as e:
            print(f"❌ 失败：{e}")
            if device == "cuda" and "out of memory" in str(e).lower():
                print("   💡 显存不足，跳过此配置")
    
    print_benchmark(results)
    
    # 保存结果
    import json
    output_file = "benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 基准测试结果已保存到：{output_file}")


if __name__ == "__main__":
    main()
