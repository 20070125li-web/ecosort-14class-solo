#!/bin/bash
# EcoSort TrashNet 6-Class Training Script
# Automates training setup, environment validation and background execution

set -e

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🚀 EcoSort TrashNet 6-Class Training                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configure environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosort

# GPU Validation
echo "🔍 Checking GPU Availability..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  Training on CPU (significantly slower)')
"
echo ""

# Dataset Validation
echo "🔍 Validating Dataset..."
python -c "
from pathlib import Path
data_root = Path('data/raw')
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print('TrashNet Dataset Summary:')
for cls in classes:
    count = len(list((data_root / cls).glob('*')))
    print(f'  {cls}: {count} images')
"
echo ""

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/training_trashnet_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 Starting Training Process"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Launch training (TrashNet-specific config with class weighting)
nohup python experiments/train_baseline.py \
    --config configs/trashnet_resnet50.yaml \
    --data-root data/raw \
    --exp-name trashnet_resnet50 \
    --no-wandb \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✓ Training Started (PID: $TRAIN_PID)"
echo ""
echo "📁 Log File: $LOG_FILE"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 Monitoring Commands"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Real-time training logs:"
echo "   tail -f $LOG_FILE"
echo ""
echo "2. Latest training output:"
echo "   tail -50 $LOG_FILE"
echo ""
echo "3. Check process status:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "4. GPU utilization:"
echo "   nvidia-smi"
echo ""
echo "5. Monitor checkpoint files:"
echo "   watch -n 5 'ls -lht checkpoints/trashnet_resnet50/'"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "⏱️  Estimated Training Time"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "With GPU:"
echo "  • Per Epoch: ~1-2 minutes"
echo "  • 50 Epochs: ~50-100 minutes"
echo "  • Early stopping may terminate early (10-15 epochs with no improvement)"
echo ""
echo "With CPU:"
echo "  • Per Epoch: ~5-10 minutes"
echo "  • Total: ~4-8 hours"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Save PID for process management
echo $TRAIN_PID > "$LOG_DIR/training.pid"

# Wait for initial output generation
sleep 10

echo "========== Initial Training Output =========="
if [ -f "$LOG_FILE" ]; then
    head -50 "$LOG_FILE"
fi

echo ""
echo "✅ Training running in background!"
echo ""
echo "💡 Tip: Use 'tail -f $LOG_FILE' to monitor training progress in real-time"
