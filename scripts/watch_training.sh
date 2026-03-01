#!/bin/bash
# 实时监控训练进度

LOGFILE=$(ls -t logs/train_optimized*.log 2>/dev/null | head -1)

if [ -z "$LOGFILE" ]; then
    echo "未找到训练日志"
    exit 1
fi

echo "监控日志: $LOGFILE"
echo "=========================================="
echo ""

# 获取最后几行关键信息
tail -100 "$LOGFILE" | grep -E "Epoch [0-9]+/|Val Acc:|Best Val|Training completed" | tail -20

echo ""
echo "=========================================="
echo "GPU使用情况:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "未检测到GPU"
