# MRT-12: 流形循环变换器

## 🌐 语言切换 | Language Switch

**中文版本**: [📖 查看中文 README](docs/README_中文.md)  
**English Version**: [📖 View English README](README.md)

---

## 🌟 项目简介

MRT-12 是一个创新的神经语言模型架构，将几何流形学习与变换器机制相结合，实现了 8.4 亿参数的大规模语言建模。

## 📚 文档导航 | Documentation Navigation

### 📖 核心技术文档 | Technical Documentation

| 文档名称 | 中文 | English | 描述 |
|---------|------|---------|------|
| **技术文档** | [📄 查看](docs/MRT12 中文技术文档.md) | [📄 View](docs/MRT12_TECHNICAL_DOCUMENTATION.md) | 完整的技术架构、API 说明和使用指南 |
| **架构对比** | [📊 查看](docs/ARCHITECTURE_COMPARISON_中文.md) | [📊 View](docs/ARCHITECTURE_COMPARISON.md) | MRT-12 与 Transformer、Mamba 等架构的详细对比分析 |
| **核心优势** | [🏆 查看](docs/MRT12_CORE_ADVANTAGES.md) | - | MRT-12 的独家优势和战略定位分析 |
| **快速入门** | [🚀 查看](docs/README_中文.md) | [🚀 View](README.md) | 快速上手指南和项目介绍 |

### 🔧 使用指南 | User Guides

| 文档名称 | 文件路径 | 描述 |
|---------|---------|------|
| **数据与模型说明** | [DATA_AND_MODELS.md](DATA_AND_MODELS.md) | 详细介绍训练数据、预训练模型和使用方法 |
| **GPU/CPU 自适应指南** | [GPU_CPU_ADAPTIVE_GUIDE.md](GPU_CPU_ADAPTIVE_GUIDE.md) | GPU 和 CPU 环境下的部署和运行指南 |
| **代码清理总结** | [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) | 代码优化和清理过程的总结报告 |
| **项目结构说明** | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 项目目录结构和模块功能详解 |

### 📊 测试报告 | Test Reports

| 文档名称 | 文件路径 | 描述 |
|---------|---------|------|
| **模型测试报告** | [MODEL_TEST_REPORT.md](MODEL_TEST_REPORT.md) | 模型性能测试结果和分析 |
| **基准测试结果** | [benchmark_results.json](benchmark_results.json) | 详细的性能基准测试数据 |

### 📝 逻辑库 | Logic Libraries

| 文档名称 | 文件路径 | 描述 |
|---------|---------|------|
| **逻辑规则库** | [logic_library.json](logic_library.json) | 精简版逻辑规则库（1.5KB） |
| **Omega 逻辑库** | [omega_logic_library.json](omega_logic_library.json) | 完整版 Omega 逻辑库（3.2KB） |

---

## 📦 发布包内容

### 核心组件
- ✅ 完整的 MRT-12 模型实现 (8.4 亿参数)
- ✅ 训练数据和词汇表
- ✅ 预训练模型检查点
- ✅ 完整的训练/评估脚本
- ✅ 详细的中英文技术文档

### 项目结构
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
│   ├── zhwiki_dataset.jsonl # 中文维基百科语料 (1200 万句，2.4GB) ⬇️ 从夸克网盘下载
│   └── __init__.py
├── models/                  # 预训练模型检查点
│   └── mrt12_step_070000_20260301_213409.pth ⬇️ 从夸克网盘下载
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
└── README.md                # 本文档
```

**📥 大型文件下载**：
- **模型检查点** (`mrt12_step_070000_20260301_213409.pth`) 和 **数据集** ([zhwiki_dataset.jsonl](file:///home/bing/Documents/MRT/MRT12_FULL_RELEASE/data/zhwiki_dataset.jsonl)) 请从夸克网盘下载
- **夸克网盘链接**: https://pan.quark.cn/s/c9101da1efe2
- **文件大小**: 模型 ~2.3GB，数据集 ~2.4GB

## 🚀 快速开始

### 系统要求
- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU (推荐 RTX 3090 Ti 24GB)
- 至少 16GB 系统内存
- 20GB 可用磁盘空间

### 安装使用
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证系统
python verify_system.py

# 3. 直接使用预训练模型
python evaluate.py

# 4. 或继续训练
./start_mrt12_complete.sh
```

## 🎯 核心特性
- **8.4 亿参数**几何语言模型
- **流形学习**与变换器融合
- **显存优化**支持消费级 GPU
- **完整训练流水线**自动化
- **包含真实训练数据**和预训练模型

## 📚 详细文档
查看 `docs/` 目录下的技术文档获取更多信息。

## 📊 预训练模型性能
- **训练步数**: 70,000 步
- **最终 Loss**: 3.346
- **多样性得分**: 0.998
- **重复率**: 0%

## 🔗 快速链接 | Quick Links

### 🎯 常用操作
- **新手入门**: [docs/README_中文.md](docs/README_中文.md)
- **API 参考**: [docs/MRT12 中文技术文档.md](docs/MRT12 中文技术文档.md)
- **架构详解**: [docs/ARCHITECTURE_COMPARISON_中文.md](docs/ARCHITECTURE_COMPARISON_中文.md)
- **优势分析**: [docs/MRT12_CORE_ADVANTAGES.md](docs/MRT12_CORE_ADVANTAGES.md)
- **数据下载**: [夸克网盘](https://pan.quark.cn/s/c9101da1efe2)

### 🛠️ 开发资源
- **示例代码**: [example_usage.py](example_usage.py)
- **训练脚本**: [train_foundation.py](train_foundation.py)
- **评估工具**: [evaluate.py](evaluate.py)
- **系统验证**: [verify_system.py](verify_system.py)

## 📞 贡献与联系

**项目地址**: [https://github.com/503718696/MRT-12.git](https://github.com/503718696/MRT-12.git)

如有问题或建议，欢迎提交 Issue 或在 GitHub 上讨论。

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>