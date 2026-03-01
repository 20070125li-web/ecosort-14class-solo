#!/bin/bash
# EcoSort TrashNet 6类训练脚本

set -e

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🚀 EcoSort TrashNet 6类训练                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 设置环境
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosort

# 检查GPU
echo "🔍 检查 GPU..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  使用 CPU 训练（速度较慢）')
"
echo ""

# 检查数据
echo "🔍 检查数据集..."
python -c "
from pathlib import Path
data_root = Path('data/raw')
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print('TrashNet 数据集:')
for cls in classes:
    count = len(list((data_root / cls).glob('*')))
    print(f'  {cls}: {count} 张')
"
echo ""

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/training_trashnet_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 开始训练"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 启动训练（使用 TrashNet 专用配置，包含类别加权）
nohup python experiments/train_baseline.py \
    --config configs/trashnet_resnet50.yaml \
    --data-root data/raw \
    --exp-name trashnet_resnet50 \
    --no-wandb \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✓ 训练已启动 (PID: $TRAIN_PID)"
echo ""
echo "📁 日志文件: $LOG_FILE"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 监控命令"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. 实时查看训练日志:"
echo "   tail -f $LOG_FILE"
echo ""
echo "2. 查看最新输出:"
echo "   tail -50 $LOG_FILE"
echo ""
echo "3. 检查进程状态:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "4. 查看 GPU 使用:"
echo "   nvidia-smi"
echo ""
echo "5. 查看生成的文件:"
echo "   watch -n 5 'ls -lht checkpoints/trashnet_resnet50/'"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "⏱️  预计时间"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "使用 GPU:"
echo "  • 单 Epoch: ~1-2 分钟"
echo "  • 50 Epochs: ~50-100 分钟"
echo "  • 早停可能提前结束 (10-15 epochs 无改善)"
echo ""
echo "使用 CPU:"
echo "  • 单 Epoch: ~5-10 分钟"
echo "  • 总计: ~4-8 小时"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 保存 PID
echo $TRAIN_PID > "$LOG_DIR/training.pid"

# 等待初始输出
sleep 10

echo "========== 初始训练输出 =========="
if [ -f "$LOG_FILE" ]; then
    head -50 "$LOG_FILE"
fi

echo ""
echo "✅ 训练已在后台运行！"
echo ""
echo "💡 提示: 使用 'tail -f $LOG_FILE' 查看实时日志"
