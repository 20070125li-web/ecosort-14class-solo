#!/usr/bin/env python3
"""
分析下载的 Kaggle 数据集结构和类别分布
"""

import os
from pathlib import Path
from collections import defaultdict
import json

def analyze_directory(root_dir, name):
    """分析数据集目录结构"""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")
    print(f"路径: {root_dir}")
    print()

    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    # 查找所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    class_counts = defaultdict(int)
    total_images = 0

    # 遍历目录结构
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                total_images += 1
                # 获取类别名（父目录名）
                class_name = Path(root).name
                class_counts[class_name] += 1

    if total_images == 0:
        print("❌ 未找到图像文件")
        return

    print(f"总图像数: {total_images}")
    print(f"类别数: {len(class_counts)}")
    print()

    print("类别分布:")
    print("-" * 70)

    # 按数量排序
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    for class_name, count in sorted_classes:
        percentage = (count / total_images) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {class_name:30s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print("-" * 70)
    print()

    # 检查平衡性
    if len(class_counts) > 0:
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        avg_count = sum(counts) / len(counts)

        print(f"统计摘要:")
        print(f"  最多: {max_count}")
        print(f"  最少: {min_count}")
        print(f"  平均: {avg_count:.1f}")

        if min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"  不平衡比例: {imbalance_ratio:.1f}:1")

            if imbalance_ratio > 5:
                print(f"  ⚠️  严重的类别不平衡！")
            elif imbalance_ratio > 2:
                print(f"  ⚠️  中度不平衡")
            else:
                print(f"  ✓ 类别平衡良好")
        print()

    return {
        'total_images': total_images,
        'num_classes': len(class_counts),
        'class_counts': dict(class_counts)
    }


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     📊 Kaggle 数据集分析工具                                 ║")
    print("╚════════════════════════════════════════════════════════════╝")

    kaggle_dir = Path("data/kaggle/extracted")

    if not kaggle_dir.exists():
        print(f"\n❌ Kaggle 数据集目录不存在: {kaggle_dir}")
        print("\n请先运行: bash scripts/collect_and_balance_data.sh")
        return

    # 查找所有子目录（每个数据集）
    datasets = [d for d in kaggle_dir.iterdir() if d.is_dir()]

    if not datasets:
        print(f"\n❌ 在 {kaggle_dir} 中未找到数据集")
        return

    results = {}

    for dataset_dir in sorted(datasets):
        result = analyze_directory(str(dataset_dir), dataset_dir.name)
        if result:
            results[dataset_dir.name] = result

    # 保存摘要
    print("\n" + "="*70)
    print("📋 数据集摘要")
    print("="*70)
    print()

    summary = []
    for name, result in results.items():
        summary.append({
            'name': name,
            'total_images': result['total_images'],
            'num_classes': result['num_classes'],
            'class_counts': result['class_counts']
        })

        print(f"{name}:")
        print(f"  总图像: {result['total_images']}")
        print(f"  类别数: {result['num_classes']}")
        print(f"  平均/类: {result['total_images'] // result['num_classes']}")
        print()

    # 保存到 JSON
    output_file = "data/kaggle/datasets_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ 分析结果已保存到: {output_file}")
    print()
    print("💡 下一步:")
    print("   查看 configs/dataset_mapping.yaml 以了解如何映射到 EcoSort 4 类")


if __name__ == "__main__":
    main()
