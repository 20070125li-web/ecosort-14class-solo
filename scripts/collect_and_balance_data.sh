#!/bin/bash
# EcoSort 数据集收集和平衡脚本
# 目标：每类 500-1000 张图像，实现类别平衡

set -e

PROJECT_ROOT="/public/home/zhw/cptac/projects/ecosort"
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     📊 EcoSort 数据集收集和平衡工具                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查 Kaggle
echo "🔍 检查 Kaggle 安装..."
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI 未安装"
    echo ""
    echo "安装步骤:"
    echo "1. pip install kaggle"
    echo "2. 访问 https://www.kaggle.com/settings 生成 API Key"
    echo "3. 下载 kaggle.json 到 ~/.kaggle/"
    echo ""
    exit 1
fi

echo "✓ Kaggle CLI 已安装"
echo ""

# 检查当前数据
echo "═══════════════════════════════════════════════════════════"
echo "📊 当前数据集统计"
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 << 'EOF'
import os
from pathlib import Path

data_root = Path("data/raw")
categories = ["recyclable", "hazardous", "kitchen", "other"]

print("类别分布:")
print("-" * 50)

total = 0
for cat in categories:
    cat_path = data_root / cat
    if cat_path.exists():
        count = len(list(cat_path.glob("*")))
    else:
        count = 0
    total += count
    percentage = (count / total * 100) if total > 0 else 0
    print(f"  {cat:12s}: {count:4d} 张 ({percentage:5.1f}%)")

print("-" * 50)
print(f"  {'总计':12s}: {total:4d} 张")
print()

# 检查平衡性
if total > 0:
    counts = [len(list((data_root / cat).glob("*"))) if (data_root / cat).exists() else 0 for cat in categories]
    max_count = max(counts)
    min_count = min(counts)
    if max_count > 0 and min_count > 0:
        imbalance_ratio = max_count / min_count
        print(f"不平衡比例: {imbalance_ratio:.1f}:1")
        if imbalance_ratio > 3:
            print("⚠️  严重的类别不平衡！")
        elif imbalance_ratio > 1.5:
            print("⚠️  中度不平衡")
        else:
            print("✓ 类别平衡良好")
    else:
        print("❌ 存在空类别")
print()
EOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📥 推荐数据集"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. Garbage Classification (Kaggle)"
echo "   - 9,871 张图像，12 类"
echo "   - 包含: cardboard, glass, metal, paper, plastic, trash, 等"
echo "   - 下载: kaggle datasets download -d asdasdasasdas/garbage-classification"
echo ""
echo "2. Waste Classification Data (Kaggle)"
echo "   - ~3,000 张图像，4 类"
echo "   - 包含: organic, recyclable, hazardous, landfill"
echo "   - 下载: kaggle datasets download -d techsash/waste-classification"
echo ""
echo "3. TrashNet (Kaggle/GitHub)"
echo "   - 2,527 张图像，6 类"
echo "   - 经典基准数据集"
echo "   - 下载: kaggle datasets download -d fedherinoraminirez/trashnet"
echo ""

read -p "是否现在下载推荐数据集？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "跳过下载"
    exit 0
fi

# 创建下载目录
mkdir -p data/kaggle

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📥 开始下载数据集"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 下载数据集 1
echo "📦 [1/3] Garbage Classification..."
if [ ! -f "data/kaggle/garbage-classification.zip" ]; then
    kaggle datasets download -d asdasdasasdas/garbage-classification -p data/kaggle/
    echo "✓ 下载完成"
else
    echo "✓ 已存在，跳过"
fi

# 下载数据集 2
echo ""
echo "📦 [2/3] Waste Classification Data..."
if [ ! -f "data/kaggle/waste-classification.zip" ]; then
    kaggle datasets download -d techsash/waste-classification -p data/kaggle/
    echo "✓ 下载完成"
else
    echo "✓ 已存在，跳过"
fi

# 下载数据集 3
echo ""
echo "📦 [3/3] TrashNet..."
if [ ! -f "data/kaggle/trashnet.zip" ]; then
    kaggle datasets download -d fedherinoraminirez/trashnet -p data/kaggle/
    echo "✓ 下载完成"
else
    echo "✓ 已存在，跳过"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📂 解压数据集"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p data/kaggle/extracted

for zip in data/kaggle/*.zip; do
    if [ -f "$zip" ]; then
        echo "解压: $(basename "$zip")"
        unzip -q "$zip" -d data/kaggle/extracted/
    fi
done

echo "✓ 解压完成"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "🔍 查看下载的数据集结构"
echo "═══════════════════════════════════════════════════════════"
echo ""

find data/kaggle/extracted/ -maxdepth 3 -type d | head -20

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "📊 下一步"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. 查看下载的数据集类别分布"
echo "   python scripts/analyze_kaggle_datasets.py"
echo ""
echo "2. 合并和映射到 EcoSort 4 类"
echo "   python scripts/merge_datasets.py \\"
echo "       --source data/kaggle/extracted/ \\"
echo "       --target data/raw/ \\"
echo "       --mapping configs/dataset_mapping.yaml"
echo ""
echo "3. 平衡数据集（可选）"
echo "   python scripts/balance_dataset.py \\"
echo "       --data-root data/raw/ \\"
echo "       --target-per-class 500"
echo ""
echo "4. 验证平衡后数据集"
echo "   python scripts/verify_balance.py --data-root data/raw/"
echo ""
echo "5. 开始训练"
echo "   python experiments/train_baseline.py \\"
echo "       --config configs/baseline_resnet50.yaml \\"
echo "       --data-root data/raw/ \\"
echo "       --exp-name balanced_training"
echo ""
echo "✅ 数据集下载完成！"
