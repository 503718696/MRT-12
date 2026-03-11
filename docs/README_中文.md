# MRT-12: 流形循环变换器

<div align="center">

[🇨🇳 中文](README_中文.md) | [🇬🇧 English](README.md)

[项目主页](https://github.com/503718696/MRT-12.git) | [技术文档](./) | [夸克网盘下载](https://pan.quark.cn/s/c9101da1efe2)

</div>

---

## 📚 文档导航 | Documentation Navigation

### 📖 核心技术文档 | Technical Documentation

| 文档名称 | 中文 | English | 描述 |
|---------|------|---------|------|
| **技术文档** | [📄 查看](MRT12 中文技术文档.md) | [📄 View](MRT12_TECHNICAL_DOCUMENTATION.md) | 完整的技术架构、API 说明和使用指南 |
| **架构对比** | [📊 查看](ARCHITECTURE_COMPARISON_中文.md) | [📊 View](ARCHITECTURE_COMPARISON.md) | MRT-12 与 Transformer、Mamba 等架构的详细对比分析 |
| **核心优势** | [🏆 查看](MRT12_CORE_ADVANTAGES.md) | - | MRT-12 的独家优势和战略定位分析 |
| **快速入门** | [🚀 查看](README_中文.md) | [🚀 View](README.md) | 快速上手指南和项目介绍 |

### 🔧 使用指南 | User Guides

| 文档名称 | 文件路径 | 描述 |
|---------|---------|------|
| **数据与模型说明** | [DATA_AND_MODELS.md](../DATA_AND_MODELS.md) | 详细介绍训练数据、预训练模型和使用方法 |
| **GPU/CPU 自适应指南** | [GPU_CPU_ADAPTIVE_GUIDE.md](../GPU_CPU_ADAPTIVE_GUIDE.md) | GPU 和 CPU 环境下的部署和运行指南 |
| **代码清理总结** | [CLEANUP_SUMMARY.md](../CLEANUP_SUMMARY.md) | 代码优化和清理过程的总结报告 |
| **项目结构说明** | [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | 项目目录结构和模块功能详解 |

### 📊 测试报告 | Test Reports

| 文档名称 | 文件路径 | 描述 |
|---------|---------|------|
| **模型测试报告** | [MODEL_TEST_REPORT.md](../MODEL_TEST_REPORT.md) | 模型性能测试结果和分析 |
| **基准测试结果** | [benchmark_results.json](../benchmark_results.json) | 详细的性能基准测试数据 |

### 📝 逻辑库 | Logic Libraries

| 文档名称 | 文件路径 | 描述 |
|---------|---------|------|
| **逻辑规则库** | [logic_library.json](../logic_library.json) | 精简版逻辑规则库（1.5KB） |
| **Omega 逻辑库** | [omega_logic_library.json](../omega_logic_library.json) | 完整版 Omega 逻辑库（3.2KB） |

---

<p align="center">
  <img src="mrt12_logo.png" alt="MRT-12 Logo" width="200"/>
</p>

<p align="center">
  <strong>语言建模的几何方法</strong>
</p>

<p align="center">
  <a href="https://github.com/robing-ai/mrt12/actions"><img src="https://github.com/robing-ai/mrt12/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/robing-ai/mrt12" alt="License"></a>
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://arxiv.org/badge.png" alt="Paper"></a>
</p>

## 🌟 什么是MRT-12？

MRT-12（流形循环变换器 12）是一种新颖的神经架构，将**几何流形学习**与**基于变换器的序列建模**相融合。与在平坦欧几里得空间中运算的传统变换器不同，MRT-12 在弯曲流形上执行计算，从而实现更复杂的推理能力。

### 核心创新
- **几何层**：黎曼流形上的数学运算
- **因果卷积**：确保严格的时间因果性 
- **Lerp 扫描**：并行和串行插值实现平滑状态转换
- **内存效率**：针对消费级 GPU 的高级优化技术

## 🚀 快速开始

### 系统要求
- **GPU**：NVIDIA RTX 3090 Ti (24GB) 或同等配置
- **内存**：32GB+ 系统内存
- **存储**：500GB+ 可用空间
- **软件**：Python 3.10+, PyTorch 2.0+, CUDA 11.8+

### 安装步骤
``bash
# 导航到发布目录
cd MRT12_FULL_RELEASE

# 安装依赖
pip install -r requirements.txt

# 验证系统配置
python verify_system.py

# 直接使用预训练模型
python evaluate.py

# 或使用完整流水线继续训练
./start_mrt12_complete.sh
```

## 📊 模型规格

| 参数 | 数值 |
|------|------|
| **总参数量** | 8.4 亿 |
| **隐藏维度** | 2048 |
| **层数** | 32 |
| **词汇表大小** | 7,429 |
| **序列长度** | 512 |
| **训练数据** | 中文维基百科 (1200 万句子) |

## 🎯 性能亮点

- **训练速度**：13,534 tokens/秒 (RTX 3090 Ti)
- **内存使用**：训练期间 13.5GB 显存
- **最终 Loss**：70K 步后 3.346
- **生成质量**：0.998 多样性得分

## 📚 文档资源

- [📘 技术文档](MRT12_TECHNICAL_DOCUMENTATION.md) - 完整技术参考
- [⚡ 快速开始指南](QUICK_START.md) - 快速设置说明
- [🔧 性能优化](PERFORMANCE_OPTIMIZATION_GUIDE.md) - 高级调优指南
- [🛡️ 内存管理](MEMORY_OPTIMIZATION_GUIDE.md) - 显存优化策略

## 🧪 使用示例

``python
from core.model_mrt12 import MRT12_Universal

# 加载训练好的模型
model = MRT12_Universal.from_pretrained("models/mrt12_step_070000_20260301_213409.pth")

# 文本生成
response = model.generate(
    prompt="人工智能是",
    max_length=100,
    temperature=0.8
)
print(response)  # "人工智能是个用于他们的技术。"
```

## 🛠️ 训练流水线

MRT-12 系统遵循三阶段方法：

1. **基础预训练**：使用 `train_foundation.py` 学习基本语言模式
2. **逻辑微调**：使用 `tune_logic.py` 增强推理能力  
3. **评估**：使用 `evaluate.py` 进行全面质量评估

每个阶段都通过交互式启动脚本 `start_mrt12_complete.sh` 完全自动化。

## 📁 项目结构

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

## 🤝 贡献

我们欢迎贡献！详细信息请参阅我们的 [贡献指南](CONTRIBUTING.md)。

### 开发环境设置
```bash
conda create -n mrt12-dev python=3.11
conda activate mrt12-dev
pip install -r requirements.txt
```

## 📈 研究影响

MRT-12 证明了几何方法可以增强传统变换器架构：

- **新颖架构**：首次成功将流形学习与变换器融合
- **实际可行性**：在消费级硬件上高效运行
- **质量成果**：以独特的推理模式实现竞争性性能
- **开源**：完全可复现的研究

## 📖 引用

如果您在研究中使用 MRT-12，请引用：

``bibtex
@article{mrt122026,
  title={MRT-12: Manifold Recurrent Transformer for Geometric Language Modeling},
  author={Luo Bing},
  email={<271217953@qq.com>},
  website={https://www.youtube.com/@1918705950},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## 📞 支持

- **项目地址**: [https://github.com/503718696/MRT-12.git](https://github.com/503718696/MRT-12.git)
- **问题**：[GitHub Issues](https://github.com/503718696/MRT-12.git/issues)
- **文档**：[技术文档](MRT12_TECHNICAL_DOCUMENTATION.md)
- **讨论**：[社区论坛](https://github.com/503718696/MRT-12.git/discussions)

## 📜 许可证

本项目根据 MIT 许可证授权。详见 [LICENSE](LICENSE) 文件。

---

<p align="center">
</p>

# MRT-12 Universal Topology 项目

## 项目概述

MRT-12 是一个采用模块化架构设计的通用拓扑语言模型，具有以下特点：

- **模块化解耦**：核心数学、模型架构、数据处理、工具组件完全分离
```

```

```

```

```
