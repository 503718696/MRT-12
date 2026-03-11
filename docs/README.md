# MRT-12: Manifold Recurrent Transformer

<p align="center">
  <img src="docs/mrt12_logo.png" alt="MRT-12 Logo" width="200"/>
</p>

<p align="center">
  <strong>A Geometric Approach to Language Modeling</strong>
</p>

<p align="center">
  <a href="https://github.com/robing-ai/mrt12/actions"><img src="https://github.com/robing-ai/mrt12/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/robing-ai/mrt12" alt="License"></a>
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://arxiv.org/badge.png" alt="Paper"></a>
</p>

## 🌟 What is MRT-12?

MRT-12 (Manifold Recurrent Transformer 12) is a novel neural architecture that merges **geometric manifold learning** with **transformer-based sequence modeling**. Unlike traditional transformers that operate in flat Euclidean space, MRT-12 performs computations on curved manifolds, enabling more sophisticated reasoning capabilities.

### Key Innovations
- **Geometric Layers**: Mathematical operations on Riemannian manifolds
- **Causal Convolution**: Ensures strict temporal causality 
- **Lerp Scanning**: Parallel and serial interpolation for smooth state transitions
- **Memory Efficiency**: Optimized for consumer GPUs with advanced techniques

## 🚀 Quick Start

### Requirements
- **GPU**: NVIDIA RTX 3090 Ti (24GB) or equivalent
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ free space
- **Software**: Python 3.10+, PyTorch 2.0+, CUDA 11.8+

### Installation
```bash
# Navigate to release directory
cd MRT12_FULL_RELEASE

# Install dependencies
pip install -r requirements.txt

# Validate system configuration
python verify_system.py

# Use pretrained model directly
python evaluate.py

# Or continue training with full pipeline
./start_mrt12_complete.sh
```

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | 840M |
| **Hidden Dimensions** | 2048 |
| **Layers** | 32 |
| **Vocabulary Size** | 7,429 |
| **Sequence Length** | 512 |
| **Training Data** | Chinese Wikipedia (12M sentences) |

## 🎯 Performance Highlights

- **Training Speed**: 13,534 tokens/second (RTX 3090 Ti)
- **Memory Usage**: 13.5GB VRAM during training
- **Final Loss**: 3.346 after 70K steps
- **Generation Quality**: 0.998 diversity score

## 📚 Documentation

- [📘 Technical Documentation](MRT12_TECHNICAL_DOCUMENTATION.md) - Complete technical reference
- [⚡ Quick Start Guide](QUICK_START.md) - Fast setup instructions
- [🔧 Performance Optimization](PERFORMANCE_OPTIMIZATION_GUIDE.md) - Advanced tuning guide
- [🛡️ Memory Management](MEMORY_OPTIMIZATION_GUIDE.md) - VRAM optimization strategies

## 🧪 Example Usage

```python
from core.model_mrt12 import MRT12_Universal

# Load trained model
model = MRT12_Universal.from_pretrained("models/mrt12_step_070000_20260301_213409.pth")

# Generate text
response = model.generate(
    prompt="人工智能是",
    max_length=100,
    temperature=0.8
)
print(response)  # "人工智能是个用于他们的技术。"
```

## 🛠️ Training Pipeline

The MRT-12 system follows a three-phase approach:

1. **Foundation Pre-training**: Learn basic language patterns using `train_foundation.py`
2. **Logic Fine-tuning**: Enhance reasoning capabilities with `tune_logic.py`
3. **Evaluation**: Comprehensive quality assessment with `evaluate.py`

Each phase is fully automated with the interactive launcher script `start_mrt12_complete.sh`.

## 📁 Project Structure

```
MRT12_FULL_RELEASE/
├── core/                    # 核心数学和模型实现
│   ├── functors.py         # 几何函数实现（因果卷积、语义轨迹）
│   ├── manifold_ops.py     # 黎曼算子和几何运算
│   ├── model_mrt12.py      # MRT-12 模型架构
│   ├── morphisms.py        # 流形变换和范畴论映射
│   └── __init__.py
├── data/                    # 数据处理模块
│   ├── cleaning.py         # 数据清洗和标准化
│   ├── dataset.py          # 数据集和词汇表管理
│   ├── mrt_vocab.json      # 模型词汇表 (7,429 tokens)
│   ├── zhwiki_dataset.jsonl # 中文维基百科语料 (1200 万句)
│   └── __init__.py
├── models/                  # 预训练模型检查点
│   └── mrt12_step_070000_20260301_213409.pth
├── utils/                   # 工具模块
│   ├── logger.py           # 工业级日志管理（50MB 限制）
│   ├── checkpoint.py       # 智能检查点管理
│   ├── common.py           # 公共工具函数（GPU 检测、检查点加载）
│   └── __init__.py
├── docs/                    # 详细技术文档
│   ├── MRT12_TECHNICAL_DOCUMENTATION.md
│   ├── MRT12 中文技术文档.md
│   ├── README.md
│   └── README_中文.md
├── logic_library.json       # 逻辑规则库（精简版）
├── omega_logic_library.json # Omega 逻辑库（完整版）
├── evaluate.py              # 模型评估（含 GPU/CPU 自适应）
├── example_usage.py         # 示例用法
├── train_foundation.py      # 基础训练脚本（知识底座预训练）
├── tune_logic.py            # 逻辑微调脚本
├── verify_system.py         # 系统验证
├── test_gpu_detection.py    # GPU 显存检测测试
├── test_cpu_run.py          # ⚠️ CPU 测试运行（已弃用）
├── start_mrt12_complete.sh  # 交互式启动脚本
├── requirements.txt         # Python 依赖列表
├── DATA_AND_MODELS.md       # 数据模型文档
├── GPU_CPU_ADAPTIVE_GUIDE.md # GPU/CPU 自适应指南
├── CLEANUP_SUMMARY.md       # 代码清理总结
└── README.md                # 项目说明
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
conda create -n mrt12-dev python=3.11
conda activate mrt12-dev
pip install -r requirements.txt
```

## 📈 Research Impact

MRT-12 demonstrates that geometric approaches can enhance traditional transformer architectures:

- **Novel Architecture**: First successful integration of manifold learning with transformers
- **Practical Viability**: Runs efficiently on consumer hardware
- **Quality Results**: Achieves competitive performance with unique reasoning patterns
- **Open Source**: Fully reproducible research

## 📖 Citation

If you use MRT-12 in your research, please cite:

```bibtex
@article{mrt122026,
  title={MRT-12: Manifold Recurrent Transformer for Geometric Language Modeling},
  author={Luo Bing},
  email={<271217953@qq.com>},
  website={https://www.youtube.com/@1918705950},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/robing-ai/mrt12/issues)
- **Documentation**: [Technical Docs](MRT12_TECHNICAL_DOCUMENTATION.md)
- **Discussion**: [Community Forum](https://github.com/robing-ai/mrt12/discussions)

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  </p>

# MRT-12 Universal Topology 项目

## 项目概述

MRT-12 是一个采用模块化架构设计的通用拓扑语言模型，具有以下特点：

- **模块化解耦**：核心数学、模型架构、数据处理、工具组件完全分离
- **工业级日志**：自动滚动日志，保护硬盘空间（最大 150MB）
- **智能检查点**：自动保存和管理训练检查点
- **在线进化**：支持交互式推理和实时学习
- **硬件优化**：针对 3090 Ti 等消费级 GPU 优化
- **智能显存管理**：自动检查 GPU 显存，防止 OOM 错误

## 目录结构

```
MRT12_FULL_RELEASE/
├── core/                    # 核心模块
│   ├── functors.py         # 几何函数实现（因果卷积、语义轨迹）
│   ├── manifold_ops.py     # 黎曼算子和数学运算
│   ├── model_mrt12.py      # MRT-12 模型架构
│   ├── morphisms.py        # 流形变换和范畴论映射
│   └── __init__.py
├── data/                    # 数据处理模块
│   ├── dataset.py          # 数据集和词表管理
│   └── cleaning.py         # 数据清洗和预处理
├── utils/                   # 工具模块
│   ├── logger.py           # 工业级日志管理
│   └── checkpoint.py       # 检查点管理
├── train_foundation.py          # 基础训练脚本
├── tune_logic.py                 # 逻辑微调脚本
└── README.md               # 项目说明文档
```

## 核心特性

### 1. 模块化解耦设计

**优势**：
- 复用模型定义进行推理（无需重复代码）
- 单独优化数据预处理（CPU 处理）
- 保护 GPU 性能（核心运算专门优化）
- 易于维护和扩展

### 2. 硬盘保护机制

通过 `LogosLogger` 的`RotatingFileHandler`：
- 自动滚动日志文件
- 最大限制 150MB（50MB × 3 个文件）
- 即使训练一年也不会撑爆硬盘

### 3. 智能检查点管理

- 自动保存最新 N 个检查点
- 自动清理旧检查点
- 支持断点续训
- 记录训练状态和指标

### 4. GPU 显存智能检查 ✨

**新增功能**：
- 训练前自动检查 GPU 显存是否充足
- 根据模型配置精确估算显存需求
- 显存不足时给出具体优化建议
- 支持多种硬件配置的安全运行

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install tensorboard
```

### 2. 训练模型

```bash
# 基础训练 (自动显存检查)
python train_foundation.py \
    --data_path zhwiki_dataset.jsonl \
    --batch_size 24 \
    --d_model 2048 \
    --num_layers 32

# 使用更多选项
python train_foundation.py \
    --data_path data/zhwiki_dataset.jsonl \
    --batch_size 24 \
    --accum_steps 4 \
    --learning_rate 5e-4 \
    --max_steps 70000 \
    --compile_model \
    --log_dir mrt12_training_logs \
    --checkpoint_dir models/

# 使用预训练模型推理
python example_usage.py
```

### 3. 交互式推理

```bash
# 启动交互助手（需要使用预训练模型）
python evaluate.py
```

## 模块详解

### Core 核心模块

#### `core/functors.py`
包含几何函数实现：
- `CausalFunctor`: 因果卷积运算
- `SemanticTrajectoryFunctor`: 语义轨迹追踪
- `CategoryBinder`: 范畴论绑定器
- 流形上的张量操作

#### `core/manifold_ops.py`
黎曼算子和数学运算：
- `RMSNorm`: Root Mean Square 归一化层
- 并行 LERP 路径扫描算法
- 余弦相似度矩阵计算
- 数值安全的 softmax

#### `core/model_mrt12.py`
MRT-12 模型架构：
- `MRT12_Layer`: 单层架构，实现字符到概念的跃迁
- `MRT12_Universal`: 完整模型，包含嵌入、多层处理、输出头
- 神经可塑性机制：最后 4 层启用可塑性连接
- 权重绑定：输出层与嵌入层共享权重

### Data 数据模块

#### `data/dataset.py`
数据处理核心：
- `VocabManager`: 词汇表管理器 (7,429 tokens)
- `WikiDataset`: Wiki 数据集类
- 数据加载和预处理函数
- 序列填充工具
- **高性能优化**：全量预读入、懒加载 Tokenization

#### `data/cleaning.py`
数据清洗工具：
- `DataCleaner`: 数据清洗器
- Unicode 标准化
- 标点符号标准化
- 数字处理策略
- 质量过滤机制

### Utils 工具模块

#### `utils/logger.py`
工业级日志管理：
- `LogosLogger`: 自动滚动日志管理器
- TensorBoard 集成
- 实时控制台显示
- 里程碑事件记录
- 硬盘空间保护（150MB 上限）

#### `utils/checkpoint.py`
智能检查点管理：
- `CheckpointManager`: 检查点管理器
- 自动保存和清理
- 断点续训支持
- 训练状态跟踪

## 训练配置说明

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|
| `--d_model` | 2048 | 模型维度 |
| `--num_layers` | 32 | 层数 |
| `--batch_size` | 24 | 批次大小 |
| `--accum_steps` | 4 | 梯度累积步数 |
| `--learning_rate` | 5e-4 | 学习率 |
| `--max_steps` | 70000 | 最大训练步数 |

### 硬件建议

- **推荐**: NVIDIA 3090 Ti (24GB VRAM)
- **最低**: NVIDIA RTX 3080 (10GB VRAM)
- **CPU**: 8 核以上
- **内存**: 32GB 以上

### 性能优化

```bash
# 启用模型编译（PyTorch 2.0+）
--compile_model

# 调整批次大小适应显存
--batch_size 16  # 或 32, 12 根据显存调整

# 使用梯度累积减少显存占用
--accum_steps 8  # 增加累积步数

# 3090Ti 推荐配置
--batch_size 24 --num_workers 16 --compile_model
```

## GPU 显存管理 ✨

### 自动显存检查
训练脚本会自动：
1. 检测可用 GPU 设备
2. 估算模型显存需求
3. 检查实际可用显存
4. 不足时给出优化建议并退出

### 显存优化建议
当显存不足时，系统会建议：
- 减少 `batch_size`
- 减少 `d_model` 或 `num_layers`
- 使用梯度累积
- 启用混合精度训练

### 3090Ti 优化配置
```
# 推荐配置 (显存使用约 8-10GB)
python train_foundation.py \
    --data_path data/zhwiki_dataset.jsonl \
    --batch_size 24 \
    --d_model 2048 \
    --num_layers 32 \
    --num_workers 16 \
    --compile_model
```

详细显存优化指南请参考相关文档

## 使用示例

### 1. 模型加载和使用

```python
# 直接使用模型
from core.model_mrt12 import MRT12_Universal

model = MRT12_Universal.from_pretrained("models/mrt12_step_070000_20260301_213409.pth")
model.eval()

with torch.no_grad():
    input_ids = torch.tensor([[501, 238, 780]])  # "人工智能"的 token IDs
    logits = model(input_ids)
    predictions = torch.argmax(logits, dim=-1)
```

### 2. 逻辑微调

使用 `tune_logic.py` 进行逻辑能力增强：
```bash
python tune_logic.py \
    --model_path models/mrt12_step_070000.pth \
    --data_path logic_training_data.jsonl \
    --max_steps 10000
```

## 故障排除

### 常见问题

1. **显存不足**
   ```bash
   # 减少批次大小
   --batch_size 16
   
   # 增加梯度累积
   --accum_steps 8
   
   # 查看详细显存需求估算
   python test_gpu_memory.py
   ```

2. **日志文件过大**
   - 日志系统自动管理，无需手动清理
   - 最大占用 150MB 硬盘空间

3. **检查点过多**
   ```bash
   --max_checkpoints 3  # 限制保留检查点数量
   ```

### 性能监控

训练过程中会实时显示：
- 当前步数和损失值
- 显存使用情况
- 学习率
- 训练吞吐量
- 梯度范数

## 相关文档

- [DATA_AND_MODELS.md](DATA_AND_MODELS.md) - 数据模型说明
- [MRT12_TECHNICAL_DOCUMENTATION.md](MRT12_TECHNICAL_DOCUMENTATION.md) - 技术文档

## 贡献指南

欢迎贡献代码！请遵循：

1. 保持模块化解耦
2. 添加适当的文档注释
3. 遵循现有代码风格
4. 提交前运行测试

## 许可证

MIT License

## 联系方式

如有问题，请提交 issue 或联系项目维护者。

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>