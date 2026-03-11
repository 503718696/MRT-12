# MRT-12 预训练模型文件

## 📦 包含的模型

### mrt12_step_070000_20260301_213409.pth
- **训练步数**: 70,000 步
- **训练日期**: 2026-03-01 21:34:09
- **文件大小**: 9.4 GB
- **模型配置**: d_model=2048, num_layers=32
- **训练阶段**: 第一阶段 - 知识底座预训练（世界模型）
- **训练数据**: zhwiki_dataset.jsonl (2.3GB 中文维基百科)

## 🎯 使用说明

### 加载模型进行评估
```bash
python evaluate.py
```

评估脚本将自动：
1. 检测 GPU 显存并选择最优设备
2. 搜索并加载此检查点
3. 运行完整的智力验收测试

### 手动加载模型
```python
import torch
from core.model_mrt12 import MRT12_Universal

# 初始化模型
model = MRT12_Universal(vocab_size=5000, d_model=2048, num_layers=32)

# 加载检查点
checkpoint = torch.load('models/mrt12_step_070000_20260301_213409.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"模型加载成功，训练步数：{checkpoint['step']}")
```

## 📊 预期性能

在 RTX 3090 Ti (24GB) 上：
- **生成速度**: ~85k tokens/sec
- **显存占用**: ~10GB
- **推理延迟**: <100ms per token

## ⚠️ 注意事项

1. **文件完整性**: 确保下载/复制过程中文件完整（可通过校验和验证）
2. **存储空间**: 建议保留至少 20GB 空闲空间用于检查点保存
3. **版本兼容**: 此模型仅与 MRT-12 架构兼容
4. **使用场景**: 
   - ✅ 评估和测试
   - ✅ 继续训练（从 70k 步继续）
   - ✅ 推理和生成
   - ❌ 不应用于商业目的（除非获得授权）

## 🔗 相关资源

- **训练数据**: `data/zhwiki_dataset.jsonl` (2.3GB)
- **词表文件**: `data/mrt_vocab.json` (5000 tokens)
- **训练脚本**: `train_foundation.py`
- **评估脚本**: `evaluate.py`

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>
