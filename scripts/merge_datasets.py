#!/usr/bin/env python3
"""
合并多个数据集并映射到 EcoSort 4 类系统
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import defaultdict
import argparse


def load_mapping(config_path):
    """加载类别映射配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def find_image_files(directory):
    """递归查找目录中的所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    images = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                images.append(Path(root) / file)

    return images


def guess_mapping(class_name, mapping_config):
    """根据类别名猜测映射（基于关键词匹配）"""
    class_lower = class_name.lower()

    # 检查关键词映射
    for target_class, keywords in mapping_config['custom_rules']['keyword_mapping'].items():
        for keyword in keywords:
            if keyword in class_lower:
                return target_class

    # 默认映射到 other
    return 'other'


def merge_datasets(source_dir, target_dir, mapping_config, dry_run=False):
    """合并数据集到目标目录"""
    target_path = Path(target_dir)
    source_path = Path(source_dir)

    # 创建目标目录
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = defaultdict(lambda: defaultdict(int))

    # 查找所有源数据集
    datasets = [d for d in source_path.iterdir() if d.is_dir()]

    print(f"找到 {len(datasets)} 个数据集:")
    for d in datasets:
        print(f"  - {d.name}")
    print()

    # 遍历每个数据集
    for dataset_dir in datasets:
        print(f"处理数据集: {dataset_dir.name}")
        print("-" * 70)

        # 查找该数据集的类别
        # 假设数据集结构为: dataset_name/class_name/*.jpg
        classes = [d for d in dataset_dir.iterdir() if d.is_dir()]

        if not classes:
            print(f"  ⚠️  未找到类别目录，跳过")
            continue

        # 遍历每个类别
        for class_dir in classes:
            class_name = class_dir.name

            # 确定映射目标
            target_class = None

            # 尝试从预定义映射中查找
            for mapping_name in ['trashnet_mapping', 'garbage_classification_mapping',
                                'waste_classification_mapping']:
                if mapping_name in mapping_config and class_name in mapping_config[mapping_name]:
                    target_class = mapping_config[mapping_name][class_name]
                    break

            # 如果未找到，使用猜测
            if target_class is None:
                target_class = guess_mapping(class_name, mapping_config)
                print(f"  {class_name:30s} -> {target_class:12s} (自动推断)")
            else:
                print(f"  {class_name:30s} -> {target_class:12s}")

            # 查找图像
            images = find_image_files(class_dir)
            print(f"    找到 {len(images)} 张图像")

            if len(images) == 0:
                continue

            # 复制/链接图像
            target_class_dir = target_path / target_class
            if not dry_run:
                target_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                # 生成唯一文件名: dataset_class_originalname
                new_name = f"{dataset_dir.name}_{class_name}_{img_path.name}"
                target_path_full = target_class_dir / new_name

                if not dry_run:
                    # 复制文件（或使用软链接节省空间）
                    shutil.copy2(img_path, target_path_full)

                stats[dataset_dir.name][target_class] += 1

        print()

    # 打印统计
    print("="*70)
    print("📊 合并统计")
    print("="*70)
    print()

    total = 0
    for dataset, class_stats in stats.items():
        print(f"{dataset}:")
        for target_class, count in class_stats.items():
            print(f"  {target_class}: {count}")
            total += count
        print()

    print(f"总计: {total} 张图像")
    print()

    # 打印最终类别分布
    print("="*70)
    print("📊 EcoSort 类别分布")
    print("="*70)
    print()

    final_stats = defaultdict(int)
    for dataset, class_stats in stats.items():
        for target_class, count in class_stats.items():
            final_stats[target_class] += count

    for target_class in ['recyclable', 'hazardous', 'kitchen', 'other']:
        count = final_stats.get(target_class, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {target_class:12s}: {count:5d} ({percentage:5.1f}%)")

    print()

    # 检查平衡性
    if len(final_stats) > 0:
        counts = list(final_stats.values())
        max_count = max(counts)
        min_count = min(counts)

        if min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"不平衡比例: {imbalance_ratio:.1f}:1")

            if imbalance_ratio > 3:
                print("⚠️  严重的类别不平衡，建议进行数据增强")
            elif imbalance_ratio > 1.5:
                print("⚠️  中度不平衡，可能需要轻微调整")
            else:
                print("✓ 类别平衡良好")
        else:
            print("❌ 存在空类别，需要补充数据")
        print()


def main():
    parser = argparse.ArgumentParser(description="合并多个数据集到 EcoSort 格式")
    parser.add_argument("--source", type=str, default="data/kaggle/extracted",
                        help="源数据集目录")
    parser.add_argument("--target", type=str, default="data/raw",
                        help="目标目录（EcoSort 格式）")
    parser.add_argument("--mapping", type=str, default="configs/dataset_mapping.yaml",
                        help="类别映射配置文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅分析，不实际复制文件")

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║     📦 EcoSort 数据集合并工具                               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # 检查源目录
    if not os.path.exists(args.source):
        print(f"❌ 源目录不存在: {args.source}")
        print("\n请先运行: bash scripts/collect_and_balance_data.sh")
        return

    # 加载映射配置
    mapping_config = load_mapping(args.mapping)

    # 备份现有数据
    target_path = Path(args.target)
    if target_path.exists() and not args.dry_run:
        backup_dir = Path(f"{args.target}_backup")
        print(f"⚠️  目标目录已存在，备份到: {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(args.target, backup_dir)
        print("✓ 备份完成")
        print()

    # 合并数据集
    if args.dry_run:
        print("⚠️  DRY RUN 模式 - 不会实际复制文件")
        print()

    merge_datasets(args.source, args.target, mapping_config, args.dry_run)

    if not args.dry_run:
        print("✅ 数据集合并完成！")
        print()
        print("💡 下一步:")
        print("   1. 检查数据质量: python scripts/verify_balance.py --data-root data/raw/")
        print("   2. 如果需要平衡: python scripts/balance_dataset.py --data-root data/raw/")
        print("   3. 开始训练: python experiments/train_baseline.py --config configs/baseline_resnet50.yaml")


if __name__ == "__main__":
    main()
