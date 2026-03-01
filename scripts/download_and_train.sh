#!/bin/bash
# EcoSort 数据下载与快速训练脚本

set -e

echo "=================================="
echo "EcoSort 数据准备与训练"
echo "=================================="
echo ""

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

# 步骤 1: 激活环境
echo "步骤 1: 激活 Conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh

# 检查环境是否存在
if conda env list | grep -q "ecosort"; then
    echo "  ✓ 环境已存在，激活中..."
    conda activate ecosort
else
    echo "  ✗ 环境不存在，请先运行: conda env create -f environment.yml"
    exit 1
fi

# 步骤 2: 下载数据集
echo ""
echo "步骤 2: 下载数据集..."
mkdir -p data/download

if [ ! -f "data/download/trashnet.zip" ]; then
    echo "  下载 TrashNet 数据集..."
    cd data/download

    # 尝试多个下载源
    DOWNLOAD_URLS=(
        "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
        "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
    )

    DOWNLOADED=false
    for url in "${DOWNLOAD_URLS[@]}"; do
        echo "  尝试从 $url 下载..."
        if wget -c --timeout=30 -O trashnet.zip "$url"; then
            DOWNLOADED=true
            break
        else
            echo "  下载失败，尝试下一个源..."
        fi
    done

    if [ "$DOWNLOADED" = false ]; then
        echo "  ✗ 所有下载源均失败"
        echo "  请手动下载: https://github.com/garythung/trashnet"
        exit 1
    fi

    cd "$PROJECT_ROOT"
else
    echo "  ✓ 数据集已存在"
fi

# 步骤 3: 解压数据
echo ""
echo "步骤 3: 解压数据集..."
if [ ! -d "data/download/dataset-resized" ]; then
    cd data/download
    unzip -q trashnet.zip
    cd "$PROJECT_ROOT"
    echo "  ✓ 数据解压完成"
else
    echo "  ✓ 数据已解压"
fi

# 步骤 4: 转换数据格式
echo ""
echo "步骤 4: 转换数据格式..."
python3 << 'EOF'
import os
import shutil
from pathlib import Path

# TrashNet 原始类别
trashnet_classes = {
    'cardboard': 'recyclable',
    'glass': 'recyclable',
    'metal': 'recyclable',
    'paper': 'recyclable',
    'plastic': 'recyclable',
    'trash': 'other'
}

# 源目录和目标目录
source_dir = Path('data/download/dataset-resized')
target_dir = Path('data/raw')

# 创建目标目录
for class_name in ['recyclable', 'hazardous', 'kitchen', 'other']:
    (target_dir / class_name).mkdir(parents=True, exist_ok=True)

# 转换数据
if source_dir.exists():
    for old_class, new_class in trashnet_classes.items():
        old_path = source_dir / old_class
        new_path = target_dir / new_class

        if old_path.exists():
            # 复制文件
            files = list(old_path.glob('*.jpg')) + list(old_path.glob('*.png'))
            for i, img_file in enumerate(files):
                if i < 100:  # 每类限制100张用于快速测试
                    shutil.copy2(img_file, new_path / f"{old_class}_{img_file.name}")

            print(f"  {old_class} -> {new_class}: {min(len(files), 100)} 张")
    print("  ✓ 数据转换完成")
else:
    print("  ✗ 源数据目录不存在")
    exit(1)

# 统计
print("\n数据集统计:")
for class_name in ['recyclable', 'hazardous', 'kitchen', 'other']:
    count = len(list((target_dir / class_name).glob('*')))
    print(f"  {class_name}: {count} 张")
EOF

# 步骤 5: 快速训练测试
echo ""
echo "步骤 5: 开始快速训练测试（10 epochs）..."
python experiments/train_baseline.py \
    --config configs/baseline_resnet50.yaml \
    --data-root data/raw \
    --exp-name quick_test \
    --no-wandb

echo ""
echo "=================================="
echo "训练完成!"
echo "=================================="
echo ""
echo "查看结果:"
echo "  - 模型检查点: checkpoints/quick_test/"
echo "  - 训练日志: logs/"
echo ""
echo "下一步:"
echo "  1. 查看训练结果: python experiments/evaluate.py --checkpoint checkpoints/quick_test/best_model.pth"
echo "  2. 启动 API 服务: python backend/app.py --model-path checkpoints/quick_test/best_model.pth"
echo ""
