# MRT-12 项目结构说明

## 📁 完整目录结构

```
MRT12_FULL_RELEASE/
│
├── 📂 core/                          # 核心算法模块
│   ├── __init__.py                   # 核心模块入口
│   ├── functors.py                   # 范畴论函子实现
│   ├── manifold_ops.py               # 黎曼流形算子
│   ├── model_mrt12.py                # MRT-12 主模型架构
│   └── morphisms.py                  # 态射模块
│
├── 📂 data/                          # 数据处理模块
│   ├── __init__.py                   # 数据模块入口
│   ├── cleaning.py                   # 数据清洗和标准化
│   ├── dataset.py                    # 数据集加载和管理
│   ├── mrt_vocab.json                # 模型词表（7,429 tokens）
│   └── zhwiki_dataset.jsonl          # 中文维基百科训练语料（1200 万句）
│
├── 📂 models/                        # 预训练模型存储
│   └── mrt12_step_070000_20260301_213409.pth  # 预训练检查点（9.4GB）
│
├── 📂 utils/                         # 工具模块
│   ├── __init__.py                   # 工具模块入口
│   ├── checkpoint.py                 # 智能检查点管理
│   ├── common.py                     # 公共工具函数
│   │   ├── detect_gpu_memory()       # GPU 显存检测
│   │   ├── load_checkpoint_safe()    # 安全加载检查点
│   │   └── get_recommended_config()  # GPU 配置推荐
│   ├── logger.py                     # 工业级日志管理
│   └── LogosLogger                   # 日志类
│
├── 📂 docs/                          # 技术文档目录
│   ├── README.md                     # 英文文档索引
│   ├── README_中文.md                 # 中文文档索引
│   ├── MRT12_TECHNICAL_DOCUMENTATION.md  # 英文技术文档
│   └── MRT12 中文技术文档.md              # 中文技术文档
│
├── 📄 evaluate.py                    # 智能评估脚本
│   ├── detect_gpu_memory()           # GPU 显存检测
│   ├── generate_smart()              # 智能文本生成
│   ├── run_comprehensive_evaluation() # 全面评估
│   └── main()                        # 主函数（含 CPU/GPU 自适应）
│
├── 📄 train_foundation.py            # 知识底座预训练
│   ├── detect_gpu_memory()           # GPU 配置检测
│   ├── load_checkpoint_safe()        # 检查点加载（从 utils.common）
│   └── main()                        # 主训练循环
│
├── 📄 tune_logic.py                  # 逻辑微调脚本
│   ├── load_checkpoint_safe()        # 检查点加载（从 utils.common）
│   └── tune()                        # 微调主函数
│
├── 📄 verify_system.py               # 系统完整性验证
│   ├── check_python_version()        # Python 版本检查
│   ├── check_required_packages()     # 依赖包检查
│   ├── check_gpu_availability()      # GPU 可用性检查
│   └── check_project_structure()     # 项目结构检查
│
├── 📄 example_usage.py               # 使用示例脚本
│   ├── example_1_basic_model()       # 基础模型使用
│   ├── example_2_logger_usage()      # 日志系统使用
│   └── example_3_checkpoint_management() # 检查点管理
│
├── 📄 test_gpu_detection.py          # GPU 显存检测测试
│   └── test_gpu_detection()          # GPU 检测功能测试
│
├── 📄 test_cpu_run.py                # ⚠️ CPU 测试运行（已弃用）
│   └── 功能已被 evaluate.py 替代，仅保留向后兼容
│
├── 📄 start_mrt12_complete.sh        # 交互式启动脚本
│   ├── 选项 1: 知识底座预训练
│   ├── 选项 2: 逻辑微调
│   └── 选项 4: 完整自动流程
│
├── 📄 requirements.txt               # Python 依赖列表
│   ├── torch
│   ├── torchvision
│   ├── torchaudio
│   ├── numpy
│   └── psutil
│
├── 📄 logic_library.json             # 逻辑规则库（精简版，1.5KB）
├── 📄 omega_logic_library.json       # Omega 逻辑库（完整版，3.2KB）
│
├── 📄 README.md                      # 项目主说明文档
├── 📄 DATA_AND_MODELS.md             # 数据和模型详细说明
├── 📄 GPU_CPU_ADAPTIVE_GUIDE.md      # GPU/CPU 自适应指南
├── 📄 CLEANUP_SUMMARY.md             # 代码清理总结报告
└── 📄 PROJECT_STRUCTURE.md           # 本文档（项目结构说明）
```

## 📊 模块功能说明

### 核心模块 (core/)

#### `model_mrt12.py` - MRT-12 主模型
- **MRT12_Universal**: 通用拓扑模型类
  - 参数：vocab_size, d_model=2048, num_layers=32
  - 方法：forward(), count_parameters(), generate()
- **MRT12_Layer**: 单层 MRT-12 架构
- **CausalConv1d**: 因果卷积实现

#### `manifold_ops.py` - 黎曼流形算子
- **RMSNorm**: 均方根归一化
- **parallel_lerp_scan**: 并行线性插值扫描

#### `functors.py` - 范畴论函子
- **ParallelAssociativeScan**: 并行关联扫描
- **GeometricFunctor**: 几何函子类

#### `morphisms.py` - 态射模块
- **RiemannianMorphism**: 黎曼态射
- **CategoricalConsistencyLoss**: 范畴一致性损失

### 数据模块 (data/)

#### `dataset.py` - 数据集管理
- **load_data_final()**: 加载最终数据
- **build_or_load_vocab()**: 构建或加载词表
- **WikiDataset**: 维基百科数据集类
- **pad_collate_fn**: 动态 Padding  collate 函数

#### `cleaning.py` - 数据清洗
- **clean_chinese_text()**: 中文文本清洗

### 工具模块 (utils/)

#### `common.py` - 公共工具函数
- **detect_gpu_memory(required_mem_gb=10.0)**: 
  - 检测 GPU 显存并判断是否足够
  - 返回：{use_gpu, device, total_mem_gb, free_mem_gb, reason}
- **load_checkpoint_safe(filepath, model, optimizer, device)**:
  - 安全加载检查点，处理 torch.compile 前缀
  - 返回：checkpoint 字典
- **get_recommended_config()**:
  - 根据 GPU 配置推荐训练参数
  - 返回：{config, d_model, layers, batch_size}

#### `logger.py` - 日志管理
- **LogosLogger**: 工业级日志管理器
  - 方法：log_config(), log_step(), log_milestone(), close()
  - 特性：自动轮转（50MB 限制），备份计数（3 份）

#### `checkpoint.py` - 检查点管理
- **CheckpointManager**: 智能检查点管理器
  - 方法：save_checkpoint(), list_checkpoints(), get_latest_checkpoint()
  - 特性：自动清理旧检查点，保留最近 N 个

## 🔄 数据流向图

```
原始数据 (zhwiki_dataset.jsonl)
    ↓
data/cleaning.py (清洗)
    ↓
data/dataset.py (加载 + 词表)
    ↓
train_foundation.py (预训练)
    ↓
models/mrt12_step_*.pth (检查点)
    ↓
tune_logic.py (微调)
    ↓
evaluate.py (评估)
    ↓
evaluation_results.json (结果)
```

## 🎯 关键文件用途

### 训练相关
| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| train_foundation.py | 知识底座预训练 | zhwiki_dataset.jsonl | world_model_checkpoints/ |
| tune_logic.py | 逻辑微调 | logic_library.json | logic_tuning_checkpoints/ |

### 评估相关
| 文件 | 功能 | 特性 |
|------|------|------|
| evaluate.py | 智能评估 | GPU/CPU 自适应，Top-P 采样，重复惩罚 |
| test_gpu_detection.py | GPU 检测测试 | 独立测试 GPU 显存检测功能 |

### 工具相关
| 文件 | 功能 | 使用场景 |
|------|------|---------|
| verify_system.py | 系统验证 | 首次运行时检查环境 |
| example_usage.py | 使用示例 | 学习基本用法 |
| start_mrt12_complete.sh | 交互启动器 | 一键启动训练流程 |

## 📝 已弃用文件

### test_cpu_run.py ⚠️
- **状态**: 已弃用（v1.0）
- **原因**: 功能已被 evaluate.py 的智能设备选择完全替代
- **移除计划**: v2.0 版本完全移除
- **替代方案**: 使用 `python evaluate.py`（自动选择最优设备）

## 🔧 依赖关系图

```
requirements.txt
├── torch (核心框架)
│   └── 被所有模块依赖
├── torchvision (辅助)
├── torchaudio (辅助)
├── numpy (数值计算)
│   └── 被 data/, utils/ 依赖
└── psutil (系统监控)
    └── 被 utils/logger.py 依赖
```

## 📚 文档层次结构

```
文档体系
├── README.md (主入口)
│   └── 快速开始、项目概览
├── DATA_AND_MODELS.md
│   └── 数据详情、模型规格
├── PROJECT_STRUCTURE.md (本文档)
│   └── 完整结构、模块说明
├── GPU_CPU_ADAPTIVE_GUIDE.md
│   └── 设备选择、性能优化
├── CLEANUP_SUMMARY.md
│   └── 代码维护、清理记录
└── docs/
    ├── README.md (文档索引)
    ├── MRT12_TECHNICAL_DOCUMENTATION.md (英文详参)
    └── MRT12 中文技术文档.md (中文详参)
```

## 🎯 推荐使用流程

```bash
# 1. 系统验证
python verify_system.py

# 2. GPU 检测（可选）
python test_gpu_detection.py

# 3. 查看示例
python example_usage.py

# 4. 训练（三选一）
./start_mrt12_complete.sh  # 交互式
python train_foundation.py  # 直接预训练
python tune_logic.py        # 直接微调

# 5. 评估
python evaluate.py  # 推荐：智能设备选择

# 6. 查看结果
cat evaluation_results.json
```

---

<p align="center">
  Built by 罗兵 (Luo Bing) for advancing geometric AI research | Email: <2712179753@qq.com> | WeChat: 18368870543 | Douyin: 1918705950
</p>
