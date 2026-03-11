# MRT-12 数据和模型说明

## 📁 包含内容

### 数据文件 (data/ 目录)
- `zhwiki_dataset.jsonl` (2.3GB): 中文维基百科训练语料（1200 万句）
- `mrt_vocab.json` (201KB): 模型词汇表（7,429 tokens）

### 预训练模型 (models/ 目录)
- `mrt12_step_070000_20260301_213409.pth` (9.4GB): 
  - 训练步数：70,000 步
  - 最终 Loss: 3.346
  - 模型参数：8.4 亿（32 层 × 2048 维）
  - 显存使用：13.5GB（训练时）

### 逻辑训练数据
- `logic_library.json` (1.5KB): 逻辑规则库（精简版）
- `omega_logic_library.json` (3.2KB): Omega 逻辑库（完整版）

## 🗂️ 完整项目结构

```
MRT12_FULL_RELEASE/
├── core/                    # 核心算法模块
│   ├── functors.py         # 范畴论函子实现
│   ├── manifold_ops.py     # 黎曼流形算子
│   ├── model_mrt12.py      # MRT-12 主模型
│   ├── morphisms.py        # 态射模块
│   └── __init__.py
├── data/                    # 数据处理模块
│   ├── cleaning.py         # 数据清洗
│   ├── dataset.py          # 数据集加载
│   ├── mrt_vocab.json      # 词表文件
│   ├── zhwiki_dataset.jsonl # 训练语料
│   └── __init__.py
├── models/                  # 模型检查点
│   └── mrt12_step_070000_*.pth
├── utils/                   # 工具模块
│   ├── checkpoint.py       # 检查点管理
│   ├── common.py           # 公共函数（GPU 检测等）
│   ├── logger.py           # 日志管理
│   └── __init__.py
├── docs/                    # 技术文档
│   ├── README.md
│   ├── README_中文.md
│   ├── MRT12_TECHNICAL_DOCUMENTATION.md
│   └── MRT12 中文技术文档.md
├── evaluate.py              # 智能评估脚本
├── train_foundation.py      # 知识底座预训练
├── tune_logic.py            # 逻辑微调
├── verify_system.py         # 系统验证
├── example_usage.py         # 使用示例
├── test_gpu_detection.py    # GPU 检测测试
├── test_cpu_run.py          # ⚠️ CPU 测试（已弃用）
├── start_mrt12_complete.sh  # 交互式启动器
├── requirements.txt         # 依赖列表
├── logic_library.json       # 逻辑数据
├── omega_logic_library.json # Omega 逻辑数据
├── DATA_AND_MODELS.md       # 本文档
├── GPU_CPU_ADAPTIVE_GUIDE.md # GPU/CPU 指南
├── CLEANUP_SUMMARY.md       # 清理总结
└── README.md                # 项目说明
```

## 🚀 快速使用

### 1. 使用预训练模型
```python
from core.model_mrt12 import MRT12_Universal
import json
import torch

# 加载词汇表
with open('data/mrt_vocab.json', 'r') as f:
    vocab_data = json.load(f)
    vocab_size = len(vocab_data['w2i'])

# 初始化模型
model = MRT12_Universal(
    vocab_size=vocab_size,
    d_model=2048,
    num_layers=32
)

# 加载预训练权重
checkpoint = torch.load('models/mrt12_step_070000_20260301_213409.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# 生成文本
model.eval()
with torch.no_grad():
    # 使用 evaluate.py 中的生成逻辑
    pass
```

### 2. 继续训练
```bash
# 使用预训练模型继续训练
python train_foundation.py \
    --data_path data/zhwiki_dataset.jsonl \
    --vocab_path data/mrt_vocab.json \
    --resume_from models/mrt12_step_070000_20260301_213409.pth \
    --d_model 2048 \
    --num_layers 32
```

### 3. 模型评估
```bash
python evaluate.py --checkpoint models/mrt12_step_070000_20260301_213409.pth
```

## 📊 模型性能

### 训练指标
- **多样性得分**: 0.998 (优秀)
- **重复率**: 0% (最优)
- **生成质量**: 高连贯性，多样化输出

### 硬件要求
- **推荐配置**: NVIDIA RTX 3090 Ti (24GB)
- **最低配置**: NVIDIA RTX 3080 (10GB)
- **CPU 模式**: 可运行但速度较慢

## ⚠️ 注意事项

1. **存储空间**: 完整包约 12GB，请确保足够磁盘空间
2. **内存要求**: 运行时需要至少 16GB 系统内存
3. **CUDA 支持**: 需要 NVIDIA 驱动和 CUDA 环境
4. **Python 版本**: 建议使用 Python 3.10+

## 📚 更多信息

详细使用说明请参考:
- `docs/MRT12_TECHNICAL_DOCUMENTATION.md` (英文)
- `docs/MRT12 中文技术文档.md` (中文)
- `GPU_CPU_ADAPTIVE_GUIDE.md` (GPU/CPU 自适应指南)
- `CLEANUP_SUMMARY.md` (代码清理总结)

## 📞 联系与支持

如有问题、建议或合作意向，欢迎联系：
- **作者**: 罗兵 (Luo Bing)
- **邮箱**: <2712179753@qq.com>
- **微信**: 18368870543
- **抖音**: 1918705950

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>