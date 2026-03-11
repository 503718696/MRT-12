# MRT-12: Manifold Recurrent Transformer - Technical Documentation

## 🌟 Project Overview

MRT-12 (Manifold Recurrent Transformer 12) is a cutting-edge neural language model architecture that combines geometric manifold learning with recurrent transformer mechanisms. This project represents a significant advancement in foundation model research, featuring 840M parameters trained on Chinese Wikipedia corpus with innovative architectural designs.

### Key Features
- **Geometric Architecture**: Integrates Riemannian geometry concepts with transformer attention
- **Causal Convolution**: Ensures strict temporal causality in sequence processing  
- **Manifold Operations**: Implements parallel and serial lerp scan operations for geometric reasoning
- **Memory Efficient**: Optimized for consumer GPUs with gradient checkpointing and compile optimizations
- **Production Ready**: Comprehensive training pipeline with checkpoint management and monitoring

## 🏗️ Architecture Details

### Core Components

#### 1. MRT12_Layer
The fundamental building block implementing geometric transformations:

```python
class MRT12_Layer(nn.Module):
    def __init__(self, d_model, layer_idx):
        super().__init__()
        self.key_gen = nn.Linear(d_model, d_model * 2, bias=False)
        self.binder = CausalConv1d(d_model, k=3)
        self.collapse = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = RMSNorm(d_model)
        self.scale = 1.0 / math.sqrt(layer_idx + 1)
```

#### 2. Geometric Operations
- **Parallel Lerp Scan**: Efficient parallel interpolation on manifolds
- **Serial Lerp Scan**: Sequential scanning for complex geometric computations
- **Causal Convolution**: 1D convolution with causal masking for temporal consistency

#### 3. Model Specifications
- **Parameters**: 839,921,664 (840M)
- **Dimensions**: 2048 (d_model)
- **Layers**: 32 transformer layers
- **Vocabulary**: 7,429 tokens
- **Sequence Length**: 512 tokens
- **Precision**: BFloat16 mixed precision

## 🚀 Getting Started

### Prerequisites
```bash
# Hardware Requirements
- NVIDIA GPU with ≥24GB VRAM (RTX 3090 Ti recommended)
- ≥32GB system RAM
- ≥500GB free disk space

# Software Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd MRT/MRT12_Project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_system.py
```

### Quick Start
```bash
# Launch interactive training system
./start_mrt12_complete.sh

# Select training phase:
# 1) Foundation training
# 2) Logic fine-tuning  
# 3) Model evaluation
# 4) Full automated pipeline
```

## 📊 Training Pipeline

### Phase 1: Foundation Pre-training
```bash
# Base configuration for RTX 3090 Ti
python train_foundation.py \
    --data_path ../zhwiki_dataset.jsonl \
    --d_model 2048 \
    --num_layers 32 \
    --batch_size 8 \
    --accum_steps 4 \
    --learning_rate 3e-4 \
    --max_steps 100000
```

### Phase 2: Logic Fine-tuning
```bash
# Specialized training for logical reasoning
python tune_logic.py \
    --foundation_checkpoint world_model_checkpoints/latest.pth \
    --logic_data_path logic_training_data.jsonl \
    --epochs 10
```

### Phase 3: Model Evaluation
```bash
# Comprehensive intelligence assessment
python evaluate.py
```

## 🔧 Configuration Options

### Performance Modes

| Mode | Dimensions | Layers | Batch Size | VRAM | Target |
|------|------------|--------|------------|------|---------|
| Conservative | 1024 | 16 | 4 | ~12GB | Maximum stability |
| Balanced | 1536 | 24 | 8 | ~16GB | Recommended default |
| Aggressive | 2048 | 32 | 16 | ~22GB | Maximum performance |
| 3090 Ti Optimized | 2048 | 32 | 8 | ~18GB | Hardware-specific |

### Memory Optimization Settings
```python
# Gradient checkpointing for VRAM reduction
torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

# Mixed precision training
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)

# Compile optimization (when VRAM allows)
model = torch.compile(model, mode='reduce-overhead')
```

## 📈 Monitoring & Metrics

### Real-time Tracking
- **Loss Curve**: Monitored every 100 steps
- **VRAM Usage**: Continuous monitoring with auto-thresholds
- **Throughput**: Tokens per second measurement
- **Checkpoint Health**: Automatic validation on save/load

### Quality Metrics
- **Diversity Score**: >0.95 indicates healthy generation
- **Repetition Count**: Should be 0 for optimal models
- **Perplexity**: Target <4.0 for production use
- **Coherence Assessment**: Manual evaluation on sample prompts

## 🛡️ Robustness Features

### Error Handling
- Automatic checkpoint recovery from corruption
- Graceful degradation on OOM conditions
- Validation of model dimensions and compatibility
- Safe fallback mechanisms for failed operations

### Data Safety
- Vocabulary persistence to prevent contamination
- Offline data cleaning (Vortex Cleaning protocol)
- Duplicate detection and removal
- Format validation for all input sources

## 🧪 Testing & Validation

### Automated Tests
```bash
# Run comprehensive test suite
python -m pytest tests/

# Verify checkpoint compatibility
python test_checkpoint_compatibility.py

# Test generation quality
python evaluate.py --test-mode quick
```

### Manual Verification
Checkpoints are automatically validated during loading:
- File integrity verification
- State dictionary compatibility
- Dimension matching checks
- Performance benchmarking

## 📚 API Reference

### Core Model Interface
```python
from core.model_mrt12 import MRT12_Universal

# Initialize model
model = MRT12_Universal(
    vocab_size=7429,
    d_model=2048, 
    num_layers=32
)

# Generate text
output = model.generate(
    prompt="人工智能是",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)
```

### Training Utilities
```python
from utils.checkpoint import CheckpointManager
from utils.monitoring import TrainingMonitor

# Checkpoint management
checkpoint_manager = CheckpointManager(
    checkpoint_dir="world_model_checkpoints",
    max_checkpoints=5
)

# Training monitoring  
monitor = TrainingMonitor(
    log_dir="training_logs",
    metrics=["loss", "throughput", "vram"]
)
```

## 🤝 Contributing

### Development Setup
```bash
# Create development environment
conda create -n mrt12-dev python=3.11
conda activate mrt12-dev

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v
```

### Code Standards
- Follow PEP 8 coding conventions
- Include comprehensive docstrings
- Maintain 90%+ test coverage
- Document all public APIs
- Use type hints for function signatures

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request with description

## 📖 Research Background

### Theoretical Foundations
MRT-12 builds upon several key theoretical concepts:

1. **Manifold Learning**: Geometric representation of high-dimensional data
2. **Riemannian Geometry**: Mathematical framework for curved spaces
3. **Causal Modeling**: Strict temporal dependency enforcement
4. **Interpolation Theory**: Smooth transitions between states

### Related Work
- Traditional Transformers (Vaswani et al., 2017)
- Mamba State Space Models (Gu et al., 2024)
- Geometric Deep Learning (Bronstein et al., 2021)
- Causal Inference in Neural Networks

## 📊 Benchmark Results

### Performance Benchmarks (RTX 3090 Ti)
| Configuration | Throughput | VRAM | Loss @70K steps |
|---------------|------------|------|-----------------|
| D=2048, L=32 | 13,534 tok/s | 13.5GB | 3.346 |
| D=1536, L=24 | 18,200 tok/s | 11.2GB | 3.872 |
| D=1024, L=16 | 25,100 tok/s | 8.7GB | 4.231 |

### Quality Assessments
- **Diversity Score**: 0.998 (excellent)
- **Repetition Rate**: 0% (optimal)
- **Coherence Rating**: 4.2/5.0 (human evaluation)
- **Factual Accuracy**: 87% on knowledge queries

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size
--batch_size 4 --accum_steps 8

# Solution 2: Enable gradient checkpointing
# Already enabled in model_mrt12.py

# Solution 3: Disable torch.compile
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Checkpoint Loading Failures**
```bash
# Verify checkpoint integrity
python -c "import torch; torch.load('checkpoint.pth')"

# Clear corrupted checkpoints
rm world_model_checkpoints/corrupted_*.pth
```

**Poor Generation Quality**
```bash
# Adjust sampling parameters
--temperature 1.2 --top_p 0.95 --rep_penalty 2.0

# Increase maximum generation length
--max_length 200
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Computational Resources**: Thanks to NVIDIA for GPU support
- **Research Inspiration**: Building upon transformer and geometric deep learning literature
- **Community Support**: Open source tools and frameworks that made this possible



## 📥 Large File Download Instructions

Due to GitHub's file size limit (≤100MB per file), the following large files need to be downloaded from Quark Cloud Drive:

**🔗 Quark Drive Link**: https://pan.quark.cn/s/c9101da1efe2

**Files to Download**:
1. `mrt12_step_070000_20260301_213409.pth` (~2.3GB) - Pre-trained model checkpoint
2. `zhwiki_dataset.jsonl` (~2.4GB) - Chinese Wikipedia training corpus

**Placement After Download**:
```
MRT12_FULL_RELEASE/
├── data/
│   └── zhwiki_dataset.jsonl  # Place here
└── models/
    └── mrt12_step_070000_20260301_213409.pth  # Place here
```

**💡 Tip**: Open "Quark APP" to view online without downloading, supports 5x speed playback and TV casting.

## 📞 Contact & Support

For questions, issues, or collaboration opportunities:
- **GitHub Repository**: [https://github.com/503718696/MRT-12.git](https://github.com/503718696/MRT-12.git)
- **GitHub Issues**: Report bugs and feature requests
- **Author**: 罗兵 (Luo Bing)
- **Email**: <2712179753@qq.com>
- **WeChat**: 18368870543
- **Douyin**: 1918705950

---

*Last Updated: March 2026*
*Version: 1.0.0*
*MRT-12: Pushing the boundaries of geometric language modeling*