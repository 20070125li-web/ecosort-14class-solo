#!/bin/bash
# EcoSort 进度监控脚本

echo "🔍 EcoSort 系统状态监控"
echo "========================"
echo ""

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

# 1. 检查数据下载进度
echo "📥 数据下载状态:"
if [ -f "data/download/trashnet.zip" ]; then
    SIZE=$(ls -lh data/download/trashnet.zip | awk '{print $5}')
    echo "  ✓ 数据集已下载 ($SIZE)"
else
    echo "  ⏳ 正在下载..."
fi

# 2. 检查数据解压状态
echo ""
echo "📂 数据解压状态:"
if [ -d "data/download/dataset-resized" ]; then
    TOTAL=$(find data/download/dataset-resized -name "*.jpg" 2>/dev/null | wc -l)
    echo "  ✓ 已解压 $TOTAL 张图像"
else
    echo "  ⏳ 等待下载完成..."
fi

# 3. 检查数据转换状态
echo ""
echo "🔄 数据转换状态:"
if [ -d "data/raw" ]; then
    RECYCLABLE=$(ls data/raw/recyclable/ 2>/dev/null | wc -l)
    OTHER=$(ls data/raw/other/ 2>/dev/null | wc -l)
    echo "  recyclable: $RECYCLABLE 张"
    echo "  other: $OTHER 张"
    echo "  hazardous: $(ls data/raw/hazardous/ 2>/dev/null | wc -l) 张"
    echo "  kitchen: $(ls data/raw/kitchen/ 2>/dev/null | wc -l) 张"
else
    echo "  ⏳ 尚未开始"
fi

# 4. 检查 Conda 环境
echo ""
echo "🐚 Conda 环境:"
if conda env list | grep -q "ecosort"; then
    echo "  ✓ ecosort 环境已创建"
else
    echo "  ⏳ 正在创建中..."
    # 检查后台任务
    if pgrep -f "conda env create" > /dev/null; then
        echo "  🔄 进程运行中"
    else
        echo "  ⚠️  未检测到创建进程"
    fi
fi

# 5. 检查训练结果
echo ""
echo "🎯 训练结果:"
if [ -d "checkpoints" ]; then
    for exp in checkpoints/*/; do
        if [ -d "$exp" ]; then
            NAME=$(basename "$exp")
            if [ -f "${exp}best_model.pth" ]; then
                echo "  ✓ $exp - 已完成"
            fi
        fi
    done
else
    echo "  ⏳ 尚未开始训练"
fi

echo ""
echo "========================"
echo "提示: 运行 'watch -n 5 bash scripts/monitor_progress.sh' 实时监控"
