#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 验收评估脚本
集成 Top-P 采样和重复惩罚，输出真正的模型智力

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import datetime
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model_mrt12 import MRT12_Universal

def generate_smart(model, prompt, w2i, i2w, max_len=100, temp=0.8, top_p=0.9, rep_penalty=1.5, device="cpu"):
    """
    智能文本生成函数
    Args:
        model: 训练好的模型
        prompt: 输入提示文本
        w2i: 词到索引的映射
        i2w: 索引到词的映射
        max_len: 最大生成长度
        temp: 温度参数
        top_p: Top-P 采样阈值
        rep_penalty: 重复惩罚系数
        device: 运行设备 (cuda/cpu)
    """
    model.eval()
    
    # 将提示文本转换为 token ID
    ids = [w2i.get(c, 1) for c in prompt]  # 1 是<unk>的 ID
    curr = torch.tensor([ids], device=device)
    generated = list(ids)  # 用于重复惩罚跟踪
    
    print(f"\n[Prompt]: {prompt}")
    print("[MRT-12]: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_len):
            # 前向传播获取 logits
            logits = model(curr)
            next_logits = logits[0, -1, :] / temp  # 应用温度
            
            # 重复惩罚：降低已生成 token 的概率
            for token_id in set(generated):
                next_logits[token_id] /= rep_penalty
            
            # Top-P (Nucleus) 采样
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 至少保留一个 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0  # 保留概率最高的 token
            
            # 将被移除的 token 设置为负无穷
            next_logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
            
            # 从剩余 token 中采样
            next_id = torch.multinomial(F.softmax(next_logits, dim=-1), 1).item()
            char = i2w.get(next_id, '<unk>')
            
            # 输出生成的字符
            print(char, end="", flush=True)
            
            # 更新生成历史
            generated.append(next_id)
            curr = torch.cat([curr, torch.tensor([[next_id]], device=device)], dim=1)
            
            # 遇到终止符则停止
            if char in ["。", "！", "？", "<eos>"]:
                break
    
    print("\n" + "-"*50)
    return ''.join([i2w.get(idx, '<unk>') for idx in generated])

def calculate_diversity_score(generated_text):
    """计算生成文本的多样性得分"""
    chars = list(generated_text.replace('<unk>', '').replace(' ', ''))
    if len(chars) == 0:
        return 0.0
    
    char_counts = Counter(chars)
    total_chars = len(chars)
    
    # 计算香农熵
    entropy = 0.0
    for count in char_counts.values():
        prob = count / total_chars
        entropy -= prob * torch.log2(torch.tensor(prob)).item()
    
    # 最大可能熵（所有字符都不同）
    max_entropy = torch.log2(torch.tensor(len(set(chars)))).item()
    
    # 归一化多样性得分
    diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    return diversity

def detect_repetition(generated_text, window_size=10):
    """检测重复模式"""
    chars = list(generated_text)
    repetitions = []
    
    for i in range(len(chars) - window_size + 1):
        window = ''.join(chars[i:i + window_size])
        # 检查这个窗口是否在后续出现
        for j in range(i + window_size, len(chars) - window_size + 1):
            if ''.join(chars[j:j + window_size]) == window:
                repetitions.append(window)
                break
    
    return repetitions

def run_comprehensive_evaluation(model, w2i, i2w, device="cpu"):
    """运行全面的模型评估"""
    print("="*60)
    print("🚀 MRT-12 智力验收测试")
    print("="*60)
    
    # 测试提示词集合
    test_prompts = [
        "人工智能是",
        "如何学习编程？",
        "量子纠缠的原理",
        "写一段 Python 代码",
        "机器学习的核心思想",
        "深度学习的发展历程",
        "自然语言处理的应用",
        "计算机视觉的技术要点"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 测试 {i}/{len(test_prompts)}")
        print("-"*40)
        
        # 生成文本
        generated_text = generate_smart(model, prompt, w2i, i2w, max_len=80, temp=0.9, top_p=0.95, rep_penalty=1.5, device=device)
        
        # 计算多样性得分
        diversity = calculate_diversity_score(generated_text)
        
        # 检测重复
        reps = detect_repetition(generated_text)
        
        result = {
            "prompt": prompt,
            "generated": generated_text,
            "diversity_score": diversity,
            "repetitions": reps,
            "has_repetition": len(reps) > 0
        }
        results.append(result)
        
        # 实时反馈
        print(f"   多样性：{diversity:.3f}")
        if len(reps) > 0:
            print(f"   ⚠️ 检测到重复：{reps[0]}")
        else:
            print(f"   ✅ 无重复")
    
    # 汇总分析
    print("\n" + "="*60)
    print("📊 评估总结")
    print("="*60)
    
    avg_diversity = sum(r["diversity_score"] for r in results) / len(results)
    repetition_count = sum(1 for r in results if r["has_repetition"])
    
    print(f"平均多样性得分：{avg_diversity:.3f}")
    print(f"重复样本数：{repetition_count}/{len(results)}")
    
    # 决策建议
    print(f"\n💡 决策建议:")
    if avg_diversity > 0.95 and repetition_count == 0:
        print(f"✅ 模型状态优秀！多样性高且无重复")
    elif avg_diversity > 0.8 or repetition_count <= 2:
        print(f"⚠️ 模型可用但需优化")
        if avg_diversity < 0.9:
            print(f"   - 多样性偏低，建议使用更多样化的训练数据")
        if repetition_count > 0:
            print(f"   - 检测到重复，可增加重复惩罚或调整采样参数")
    else:
        print(f"❌ 模型存在严重问题，不建议使用")
        print(f"   - 多样性过低或重复严重")
        print(f"   - 建议重新训练或调整架构")
    
    # 保存结果
    evaluation_result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": device,
        "model_config": {
            "d_model": 2048,
            "num_layers": 32
        },
        "results": results,
        "summary": {
            "average_diversity": avg_diversity,
            "repetition_count": repetition_count,
            "total_samples": len(results)
        }
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细评估结果已保存到：evaluation_results.json")
    return evaluation_result


def detect_gpu_memory():
    """检测 GPU 显存并判断是否足够加载模型"""
    if not torch.cuda.is_available():
        print("⚠️  未检测到 CUDA 设备，将使用 CPU 模式")
        return {"use_gpu": False, "reason": "CUDA unavailable"}
    
    try:
        device = torch.cuda.current_device()
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated(device) / 1024**3
        free_mem_gb = total_mem_gb - allocated_mem
        required_mem_gb = 10.0
        
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


def main():
    """主评估函数"""
    vocab_file = "mrt_vocab.json"
    if not os.path.exists(vocab_file):
        print(f"❌ 错误：找不到词表文件 {vocab_file}")
        print("请先运行训练脚本生成词表")
        return
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        w2i = vocab_data["w2i"]
        i2w = {int(k): v for k, v in vocab_data["i2w"].items()}
    
    print(f"📚 词表加载完成：{len(w2i)} 个 token\n")
    
    gpu_info = detect_gpu_memory()
    print("")
    
    device = gpu_info["device"] if gpu_info["use_gpu"] else "cpu"
    
    print(f"🔧 初始化模型 (d_model=2048, num_layers=32)...")
    model = MRT12_Universal(
        vocab_size=len(w2i),
        d_model=2048,
        num_layers=32
    )
    
    if device == "cuda":
        print(f"🚀 将模型加载到 GPU...")
        model = model.to(device)
    else:
        print(f"💻 将模型保持在 CPU 模式...")
        model = model.to("cpu")
    
    # 加载训练好的权重 - 改进的检查点搜索逻辑
    checkpoint_dirs = [
        "world_model_checkpoints",
        "logic_tuning_checkpoints"
    ]
    
    candidate_checkpoints = []
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"🔍 搜索目录: {checkpoint_dir}")
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith("mrt12_step_") and filename.endswith(".pth"):
                    filepath = os.path.join(checkpoint_dir, filename)
                    try:
                        file_size = os.path.getsize(filepath)
                        file_time = os.path.getctime(filepath)
                        candidate_checkpoints.append({
                            'path': filepath,
                            'size': file_size,
                            'time': file_time,
                            'filename': filename
                        })
                    except OSError:
                        continue
    
    if not candidate_checkpoints:
        print("❌ 未找到任何检查点文件")
        return
    
    candidate_checkpoints.sort(key=lambda x: x['size'], reverse=True)
    
    print(f"📋 找到 {len(candidate_checkpoints)} 个候选检查点:")
    for i, cp in enumerate(candidate_checkpoints[:5]):
        size_mb = cp['size'] / (1024 * 1024)
        print(f"   {i+1}. {cp['filename']} ({size_mb:.1f}MB)")
    
    model_loaded = False
    
    # 依次尝试加载检查点
    for i, checkpoint_info in enumerate(candidate_checkpoints):
        filepath = checkpoint_info['path']
        filename = checkpoint_info['filename']
        size_mb = checkpoint_info['size'] / (1024 * 1024)
        
        print(f"\n📥 尝试加载检查点 {i+1}/{len(candidate_checkpoints)}: {filename}")
        print(f"   大小：{size_mb:.1f}MB")
        print(f"   路径：{filepath}")
        
        try:
            print("   🔍 验证文件完整性...")
            test_checkpoint = torch.load(filepath, map_location='cpu')
            print("   ✅ 文件结构验证通过")
            
            state_dict = test_checkpoint['model_state_dict']
            
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            print("   ✅ 模型权重加载成功")
            
            if 'step' in test_checkpoint:
                print(f"   📊 训练步数：{test_checkpoint['step']}")
            if 'metrics' in test_checkpoint:
                metrics = test_checkpoint['metrics']
                if 'loss' in metrics:
                    print(f"   📊 最终 Loss: {metrics['loss']:.4f}")
                if 'vram_gb' in metrics:
                    print(f"   📊 显存使用：{metrics['vram_gb']:.1f}GB")
            
            model_loaded = True
            break
            
        except Exception as e:
            print(f"   ❌ 加载失败：{str(e)}")
            if "data/" in str(e):
                print("   💡 提示：文件可能已损坏，尝试下一个检查点...")
            continue
    
    if not model_loaded:
        print("\n❌ 所有检查点都无法加载")
        print("可能的原因:")
        print("   - 检查点文件损坏")
        print("   - 磁盘空间不足")
        print("   - 文件权限问题")
        print("\n💡 建议:")
        print("   - 检查磁盘健康状态")
        print("   - 重新运行训练生成新检查点")
        print("   - 或联系技术支持")
        return
    
    print(f"\n🎉 模型加载成功！运行设备：{device.upper()}")
    print(f"💡 提示：评估过程将在{'GPU' if device == 'cuda' else 'CPU'}上运行（速度：{'快' if device == 'cuda' else '较慢'}）")
    
    # 运行评估 - 传递设备信息
    run_comprehensive_evaluation(model, w2i, i2w, device=device)

if __name__ == "__main__":
    main()