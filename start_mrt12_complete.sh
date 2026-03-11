#!/bin/bash

# MRT-12 完整训练流程一键启动脚本
# 作者：罗兵 (Luo Bing)
# 邮箱：<2712179753@qq.com>
# 微信：18368870543
# 抖音：1918705950
# 版本：1.4 (3090 Ti 优化版)

set -e  # 遇到错误立即退出

echo "=========================================="
echo "🚀 MRT-12 生产级训练系统启动器"
echo "=========================================="

# 检查环境
echo "🔍 检查运行环境..."

# 检查Python版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ Python版本: $PYTHON_VERSION"

# 检查PyTorch
if python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" &> /dev/null; then
    echo "✅ PyTorch已安装"
else
    echo "❌ 错误: 未安装PyTorch"
    echo "请运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# 检查GPU和显存
echo "🔍 检查GPU配置..."
if nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1)
    GPU_MEM=$(echo $GPU_INFO | cut -d',' -f2)
    echo "✅ GPU: $GPU_NAME (${GPU_MEM}MB)"
    
    # 3090 Ti特殊优化检测
    if [[ "$GPU_NAME" == *"3090 Ti"* ]] && [ "$GPU_MEM" -ge 24000 ]; then
        DEFAULT_MODE="3090ti_optimized"
        echo "🎯 检测到3090 Ti，启用专属优化配置"
    elif [ "$GPU_MEM" -lt 16000 ]; then
        DEFAULT_MODE="conservative"
        echo "⚠️  显存较低，推荐保守模式"
    elif [ "$GPU_MEM" -lt 24000 ]; then
        DEFAULT_MODE="balanced"
        echo "✅ 中等显存，推荐平衡模式"
    else
        DEFAULT_MODE="aggressive"
        echo "✅ 高显存，可使用激进模式"
    fi
else
    echo "⚠️  警告: 未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
    DEFAULT_MODE="cpu"
fi

# 检查数据文件
echo "📂 检查数据文件..."
DATA_FILES=(
    "../zhwiki_dataset.jsonl"
    "../zh_corpus_huge.txt" 
    "mrt_training_data_final.jsonl"
)

for file in "${DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "✅ 找到数据文件: $file ($size)"
    else
        echo "⚠️  数据文件不存在: $file"
    fi
done

# 创建必要的目录
echo "📁 创建工作目录..."
mkdir -p world_model_logs world_model_checkpoints
mkdir -p logic_tuning_logs logic_tuning_checkpoints
mkdir -p evaluation_results

echo ""
echo "🎯 训练流程选项:"
echo "1) 第一阶段: 知识底座预训练 (foundation training)"
echo "2) 第二阶段: 逻辑灌顶微调 (logic tuning)"  
echo "3) 第三阶段: 模型验收评估 (evaluation)"
echo "4) 完整流程: 1→2→3 (全自动)"
echo "5) 显存优化诊断"
echo "6) 性能模式选择"
echo "7) 3090 Ti专属优化"
echo "8) 退出"

read -p "请选择操作 (1-8): " choice

case $choice in
    1)
        echo "🔨 启动第一阶段训练..."
        echo "当前默认模式: $DEFAULT_MODE"
        echo ""
        echo "性能模式选择:"
        echo "1) 保守模式 (D=1024, L=16, BS=4) - 最稳定"
        echo "2) 平衡模式 (D=1536, L=24, BS=8) - 推荐默认"
        echo "3) 3090 Ti优化 (D=2048, L=32, BS=8) - 专属配置"
        echo "4) 激进模式 (D=2048, L=32, BS=16) - 最大性能"
        echo "5) 自动模式 (根据硬件推荐)"
        read -p "请选择性能模式 (1-5, 默认5): " perf_choice
        perf_choice=${perf_choice:-5}
        
        case $perf_choice in
            1) export MRT_PERF_MODE="conservative" ;;
            2) export MRT_PERF_MODE="balanced" ;;
            3) export MRT_PERF_MODE="3090ti_optimized" ;;
            4) export MRT_PERF_MODE="aggressive" ;;
            *) export MRT_PERF_MODE="$DEFAULT_MODE" ;;
        esac
        
        echo "选定模式: $MRT_PERF_MODE"
        
        case $MRT_PERF_MODE in
            "conservative")
                echo "预计时间: 15-25天"
                echo "目标: Loss降至3.5左右"
                echo "显存使用: ~8-10GB"
                ;;
            "balanced")
                echo "预计时间: 10-15天"
                echo "目标: Loss降至3.0左右"
                echo "显存使用: ~15-18GB"
                ;;
            "3090ti_optimized")
                echo "预计时间: 8-12天"
                echo "目标: Loss降至2.8左右"
                echo "显存使用: ~18-20GB"
                echo "特点: 3090 Ti专属优化，平衡性能与稳定性"
                ;;
            "aggressive")
                echo "预计时间: 7-12天"
                echo "目标: Loss降至2.8左右"
                echo "显存使用: ~20-22GB"
                ;;
        esac
        
        read -p "确认开始? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            # 设置显存优化环境变量
            export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
            python3 train_foundation.py
        fi
        ;;
    2)
        echo "🧠 启动第二阶段逻辑微调..."
        echo "前提: 第一阶段已完成并生成检查点"
        read -p "确认开始? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
            python3 tune_logic.py
        fi
        ;;
    3)
        echo "✅ 启动模型验收评估..."
        python3 evaluate.py
        ;;
    4)
        echo "🚀 启动完整训练流程..."
        echo "配置类型: $DEFAULT_MODE"
        echo "此过程将依次执行所有阶段"
        case $DEFAULT_MODE in
            "conservative")
                echo "总预计时间: 18-30天"
                ;;
            "balanced")
                echo "总预计时间: 12-20天"
                ;;
            "3090ti_optimized")
                echo "总预计时间: 10-15天"
                ;;
            "aggressive")
                echo "总预计时间: 9-15天"
                ;;
            *)
                echo "总预计时间: 25-40天"
                ;;
        esac
        read -p "确认开始完整训练? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            echo "🔧 阶段1: 知识底座预训练"
            export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
            python3 train_foundation.py
            
            echo "🔧 阶段2: 逻辑灌顶微调" 
            python3 tune_logic.py
            
            echo "🔧 阶段3: 模型验收评估"
            python3 evaluate.py
            
            echo "🎉 完整训练流程完成!"
        fi
        ;;
    5)
        echo "🔍 显存优化诊断..."
        echo "运行环境诊断脚本..."
        python3 verify_system.py
        echo ""
        echo "📊 显存使用建议:"
        echo "   当前配置: $DEFAULT_MODE"
        case $DEFAULT_MODE in
            "conservative")
                echo "   推荐配置: D=1024, L=16, BS=4"
                echo "   梯度累积: 8步"
                echo "   预期显存: ~8-10GB"
                ;;
            "balanced")
                echo "   推荐配置: D=1536, L=24, BS=8"
                echo "   梯度累积: 4步"
                echo "   预期显存: ~15-18GB"
                ;;
            "3090ti_optimized")
                echo "   推荐配置: D=2048, L=32, BS=8"
                echo "   梯度累积: 4步"
                echo "   预期显存: ~18-20GB"
                echo "   特点: 3090 Ti黄金配置点"
                ;;
            "aggressive")
                echo "   推荐配置: D=2048, L=32, BS=16"
                echo "   梯度累积: 2步"
                echo "   预期显存: ~20-22GB"
                ;;
        esac
        echo ""
        echo "💡 优化技巧:"
        echo "- 设置环境变量: export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'"
        echo "- 启用TensorFloat-32: torch.backends.cuda.matmul.allow_tf32 = True"
        echo "- 根据显存情况智能启用torch.compile"
        echo "- 定期执行torch.cuda.empty_cache()"
        ;;
    6)
        echo "⚙️  性能模式详细说明:"
        echo ""
        echo "🚀 激进模式 (Aggressive):"
        echo "   配置: D=2048, L=32, BS=16"
        echo "   特点: 最大化利用显存，最高训练速度"
        echo "   适用: 24GB+显存，追求极致性能"
        echo "   风险: 显存压力大，需要监控OOM"
        echo ""
        echo "⚖️  平衡模式 (Balanced):"
        echo "   配置: D=1536, L=24, BS=8"
        echo "   特点: 稳定可靠，性价比最优"
        echo "   适用: 16-24GB显存，推荐默认"
        echo "   优势: 训练稳定，资源利用率高"
        echo ""
        echo "🎯 3090 Ti优化模式:"
        echo "   配置: D=2048, L=32, BS=8"
        echo "   特点: 专为3090 Ti设计的黄金配置"
        echo "   适用: RTX 3090 Ti (24GB)"
        echo "   优势: 保持最大模型容量的同时确保稳定性"
        echo ""
        echo "🛡️  保守模式 (Conservative):"
        echo "   配置: D=1024, L=16, BS=4"
        echo "   特点: 最大安全保障，绝对稳定"
        echo "   适用: <16GB显存，或首次训练"
        echo "   代价: 训练速度相对较慢"
        echo ""
        echo "🎯 建议:"
        echo "   3090 Ti (24GB): 推荐3090 Ti优化模式"
        echo "   首次训练: 建议从平衡模式开始"
        echo "   生产环境: 根据稳定性要求选择"
        ;;
    7)
        echo "💎 3090 Ti专属优化说明:"
        echo ""
        echo "🎯 黄金配置点 (D=2048, L=32, BS=8):"
        echo "   • 保持完整的8.4亿参数模型架构"
        echo "   • 显存占用控制在18-20GB安全范围"
        echo "   • 通过梯度累积实现等效batch size=32"
        echo "   • 启用TensorFloat-32和混合精度训练"
        echo "   • 智能决定是否启用torch.compile"
        echo ""
        echo "🔧 优化策略:"
        echo "   1. 降低物理batch size，增加梯度累积"
        echo "   2. 启用BF16混合精度减少显存占用"
        echo "   3. 智能编译优化（显存充足时）"
        echo "   4. 环境变量优化防内存碎片"
        echo "   5. 实时显存监控和预警机制"
        echo ""
        echo "📊 预期表现:"
        echo "   • 显存使用: 18-20GB (安全范围)"
        echo "   • 训练速度: 优秀 (相比激进模式仅轻微下降)"
        echo "   • 模型质量: 保持完整8.4亿参数能力"
        echo "   • 稳定性: 高 (避免OOM风险)"
        echo ""
        echo "🚀 建议操作:"
        echo "   选择'3090 Ti优化模式'开始训练"
        echo "   让系统自动应用所有优化配置"
        echo "   监控显存使用保持在20GB以下"
        ;;
    8)
        echo "👋 退出程序"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "📊 训练监控命令:"
echo "  查看GPU状态: watch -n 1 nvidia-smi"
echo "  实时日志: tail -f world_model_logs/train.log"
echo "  检查点管理: ls -lh world_model_checkpoints/"
echo "  显存监控: nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1"

echo ""
echo "💡 提示:"
echo "  - 训练过程中可随时按 Ctrl+C 中断"
echo "  - 系统支持断点续训"
echo "  - 建议定期备份重要检查点"
echo "  - 如遇问题请查看相关文档"

echo ""
echo "🌟 MRT-12训练系统就绪，祝您训练顺利！"
echo "当前正在进行: The Great Cleanse 阶段 - 纯净因果预训练"