#!/bin/bash
# EcoSort GPU 训练脚本 - 带完整日志保存

set -e

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🚀 EcoSort GPU 训练 - 带日志保存                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 设置环境
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosort

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
ERROR_LOG="$LOG_DIR/training_${TIMESTAMP}_error.log"

echo "📁 日志将保存到: $LOG_FILE"
echo "📁 错误日志: $ERROR_LOG"
echo ""

# 检查GPU
echo "🔍 检查GPU状态..."
python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA不可用，将使用CPU')
    exit(1)
" || {
    echo "❌ GPU不可用，请先安装CUDA版本的PyTorch"
    exit 1
}

echo ""
echo "✓ GPU检查完成"
echo ""

# 配置
CONFIG_FILE="$PROJECT_ROOT/configs/baseline_resnet50.yaml"
DATA_ROOT="$PROJECT_ROOT/data/raw"
EXPERIMENT_NAME="gpu_training"

# 启动训练
echo "═══════════════════════════════════════════════════════════"
echo "🚀 启动GPU训练"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 使用 nohup 在后台运行，保存所有输出
nohup python experiments/train_baseline.py \
    --config "$CONFIG_FILE" \
    --data-root "$DATA_ROOT" \
    --exp-name "$EXPERIMENT_NAME" \
    --no-wandb \
    > "$LOG_FILE" 2> "$ERROR_LOG" &

TRAIN_PID=$!

echo "✓ 训练已启动"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📊 训练信息"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "进程 PID: $TRAIN_PID"
echo "配置文件: $CONFIG_FILE"
echo "数据目录: $DATA_ROOT"
echo "实验名称: $EXPERIMENT_NAME"
echo ""
echo "📁 日志文件:"
echo "  标准输出: $LOG_FILE"
echo "  错误输出: $ERROR_LOG"
echo ""
echo "📁 模型输出:"
echo "  输出目录: checkpoints/$EXPERIMENT_NAME/"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📈 实时监控命令"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. 查看训练日志 (实时):"
echo "   tail -f $LOG_FILE"
echo ""
echo "2. 查看错误日志:"
echo "   tail -f $ERROR_LOG"
echo ""
echo "3. 查看最新输出 (最后50行):"
echo "   tail -50 $LOG_FILE"
echo ""
echo "4. 检查进程状态:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "5. 查看GPU使用:"
echo "   nvidia-smi"
echo ""
echo "6. 查看生成的文件:"
echo "   watch -n 5 'ls -lht checkpoints/$EXPERIMENT_NAME/'"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "⏱️  预计时间 (GPU加速)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "使用 NVIDIA L20 GPU (3块):"
echo "  • 单个 Epoch: ~1-2 分钟 (vs CPU 6-8分钟)"
echo "  • 完整训练 (20 epochs): ~20-40 分钟"
echo "  • 预计完成: $(date '+%H:%M' -d '+30 minutes')"
echo ""
echo "⚡ 加速比: 约 5-10x"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# 保存PID到文件
echo $TRAIN_PID > "$LOG_DIR/training.pid"
echo "PID已保存到: $LOG_DIR/training.pid"

# 等待初始输出
echo "等待5秒查看初始输出..."
sleep 5

if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "========== 初始训练输出 =========="
    head -50 "$LOG_FILE"
    echo ""
fi

echo ""
echo "✅ 训练已在后台运行！"
echo ""
echo "💡 提示: 使用 'tail -f $LOG_FILE' 查看实时日志"
