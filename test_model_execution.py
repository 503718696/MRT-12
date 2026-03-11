#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 模型执行测试脚本
验证模型架构能否正常初始化、前向传播和生成文本

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_mrt12 import MRT12_Universal
from data.dataset import build_or_load_vocab


def print_section(title):
    """打印分节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_model_initialization():
    """测试 1: 模型初始化"""
    print_section("测试 1: 模型初始化")
    
    # 创建小型模型进行快速测试
    print("创建测试模型 (vocab_size=1000, d_model=512, num_layers=6)...")
    model = MRT12_Universal(
        vocab_size=1000,
        d_model=512,
        num_layers=6
    )
    
    total_params = model.count_parameters()
    print(f"✅ 模型初始化成功")
    print(f"   总参数量：{total_params:,}")
    print(f"   模型维度：d_model={model.d_model}, layers={model.num_layers}")
    
    return model


def test_forward_pass(model):
    """测试 2: 前向传播"""
    print_section("测试 2: 前向传播")
    
    batch_size = 2
    seq_len = 16
    
    # 创建随机输入
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    print(f"输入形状：{input_ids.shape}")
    
    # 前向传播
    print("执行前向传播...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model(input_ids)
    
    elapsed = time.time() - start_time
    
    print(f"✅ 前向传播成功")
    print(f"   输出形状：{output.shape}")
    print(f"   输出范围：[{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   推理时间：{elapsed*1000:.2f}ms")
    print(f"   吞吐量：{batch_size * seq_len / elapsed:.0f} tokens/sec")
    
    return output


def test_gradient_flow(model):
    """测试 3: 梯度流动"""
    print_section("测试 3: 梯度流动测试")
    
    batch_size = 2
    seq_len = 16
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("执行反向传播...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    output = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), 
        target.view(-1)
    )
    
    print(f"   Loss: {loss.item():.4f}")
    print("   反向传播中...")
    
    loss.backward()
    
    # 检查梯度
    has_grad = False
    grad_norm_total = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_norm_total += grad_norm ** 2
    
    grad_norm_total = grad_norm_total ** 0.5
    
    if has_grad:
        print(f"✅ 梯度流动正常")
        print(f"   总梯度范数：{grad_norm_total:.6f}")
    else:
        print(f"❌ 未检测到梯度")
    
    model.eval()
    return has_grad


def test_text_generation():
    """测试 4: 文本生成（使用真实词表）"""
    print_section("测试 4: 文本生成测试")
    
    # 尝试加载真实词表
    vocab_file = "mrt_vocab.json"
    if not os.path.exists(vocab_file):
        print(f"⚠️  未找到词表文件 {vocab_file}，使用模拟数据测试")
        
        # 使用模拟词表
        w2i = {chr(i): i for i in range(100)}
        i2w = {v: k for k, v in w2i.items()}
        
        model = MRT12_Universal(vocab_size=100, d_model=128, num_layers=2)
        device = "cpu"
    else:
        print(f"加载词表：{vocab_file}")
        w2i, i2w = build_or_load_vocab([], vocab_file=vocab_file)
        print(f"   词表大小：{len(w2i)}")
        
        # 创建实际大小的模型
        model = MRT12_Universal(
            vocab_size=len(w2i),
            d_model=256,
            num_layers=4
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    
    model.eval()
    
    # 测试提示词
    prompts = [
        "人工智能",
        "机器学习",
        "深度学习"
    ]
    
    print(f"\n运行设备：{device.upper()}")
    print(f"测试生成长度：20 tokens\n")
    
    for prompt in prompts:
        print(f"提示词：{prompt}")
        
        # 转换为 token IDs
        input_ids = [w2i.get(c, 1) for c in prompt]
        curr = torch.tensor([input_ids], device=device)
        
        generated = list(input_ids)
        
        with torch.no_grad():
            for _ in range(20):
                logits = model(curr)
                next_logits = logits[0, -1, :]
                
                # 简单采样
                probs = torch.softmax(next_logits / 0.8, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                
                char = i2w.get(next_id, '?')
                generated.append(next_id)
                
                curr = torch.cat([curr, torch.tensor([[next_id]], device=device)], dim=1)
        
        generated_text = ''.join([i2w.get(idx, '?') for idx in generated])
        print(f"生成：{generated_text}\n")
    
    print("✅ 文本生成测试完成")


def run_all_tests():
    """运行所有测试"""
    print_section("MRT-12 模型执行测试套件")
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    try:
        # 测试 1: 初始化
        model = test_model_initialization()
        results['初始化'] = True
        
        # 测试 2: 前向传播
        test_forward_pass(model)
        results['前向传播'] = True
        
        # 测试 3: 梯度流动
        test_gradient_flow(model)
        results['梯度流动'] = True
        
        # 测试 4: 文本生成
        test_text_generation()
        results['文本生成'] = True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 总结报告
    print_section("测试总结")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！MRT-12 模型可以正常执行")
    else:
        print("⚠️  部分测试未通过，请检查问题")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
