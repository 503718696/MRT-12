# MRT-12: 流形循环变换器

## 🌟 项目简介

MRT-12 是一个创新的神经语言模型架构，将几何流形学习与变换器机制相结合，实现了 8.4 亿参数的大规模语言建模。

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
└── README.md                # 本文档
```

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

## 📞 贡献与联系

**项目地址**: [https://github.com/503718696/MRT-12.git](https://github.com/503718696/MRT-12.git)

如有问题或建议，欢迎提交 Issue 或在 GitHub 上讨论。

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>