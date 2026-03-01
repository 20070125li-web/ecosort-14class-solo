#!/usr/bin/env python3
"""
平衡数据集：通过下采样或数据增强实现类别平衡
"""

import os
import shutil
from pathlib import Path
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms


def analyze_dataset(data_root):
    """分析数据集分布"""
    categories = ['recyclable', 'hazardous', 'kitchen', 'other']
    data_path = Path(data_root)

    class_counts = {}
    class_files = {}

    for cat in categories:
        cat_path = data_path / cat
        if cat_path.exists():
            files = list(cat_path.glob("*"))
            files = [f for f in files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
            class_counts[cat] = len(files)
            class_files[cat] = files
        else:
            class_counts[cat] = 0
            class_files[cat] = []

    return class_counts, class_files


def print_statistics(class_counts):
    """打印数据集统计"""
    print("="*70)
    print("📊 当前数据集统计")
    print("="*70)
    print()

    total = sum(class_counts.values())

    print("类别分布:")
    print("-" * 70)

    for cat, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {cat:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print("-" * 70)
    print(f"  {'总计':12s}: {total:5d}")
    print()

    # 检查平衡性
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)

    if min_count > 0:
        imbalance_ratio = max_count / min_count
        print(f"不平衡比例: {imbalance_ratio:.1f}:1")
        if imbalance_ratio > 5:
            print("⚠️  严重的类别不平衡！")
        elif imbalance_ratio > 2:
            print("⚠️  中度不平衡")
        else:
            print("✓ 类别平衡良好")
    else:
        print("❌ 存在空类别")
    print()

    return total, max_count, min_count


def downsample_majority_class(class_files, target_count, output_dir):
    """对多数类进行下采样"""
    print(f"下采样到 {target_count} 张/类...")

    for cat, files in class_files.items():
        cat_output_dir = Path(output_dir) / cat
        cat_output_dir.mkdir(parents=True, exist_ok=True)

        if len(files) <= target_count:
            # 不需要下采样，全部复制
            for img_path in files:
                shutil.copy2(img_path, cat_output_dir / img_path.name)
            print(f"  {cat}: {len(files)} 张 (无需下采样)")
        else:
            # 随机采样
            np.random.seed(42)
            sampled_indices = np.random.choice(len(files), target_count, replace=False)

            for idx in sampled_indices:
                img_path = files[idx]
                shutil.copy2(img_path, cat_output_dir / img_path.name)

            print(f"  {cat}: {len(files)} -> {target_count} 张 (下采样)")


def oversample_minority_classes(class_files, target_count, output_dir, augment=True):
    """对少数类进行上采样（数据增强）"""
    print(f"上采样到 {target_count} 张/类...")

    # 数据增强变换
    augmentation_transforms = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    ]

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for cat, files in class_files.items():
        cat_output_dir = Path(output_dir) / cat
        cat_output_dir.mkdir(parents=True, exist_ok=True)

        if len(files) >= target_count:
            # 不需要上采样，随机选择
            np.random.seed(42)
            sampled_indices = np.random.choice(len(files), target_count, replace=False)
            for idx in sampled_indices:
                img_path = files[idx]
                shutil.copy2(img_path, cat_output_dir / img_path.name)
            print(f"  {cat}: {len(files)} -> {target_count} 张 (随机选择)")
        else:
            # 复制所有原始图像
            for img_path in files:
                shutil.copy2(img_path, cat_output_dir / img_path.name)

            needed = target_count - len(files)
            print(f"  {cat}: {len(files)} -> {target_count} 张 (需要生成 {needed} 张增强图像)")

            if augment and needed > 0:
                # 通过数据增强生成额外图像
                aug_idx = 0
                while aug_idx < needed:
                    for img_path in files:
                        if aug_idx >= needed:
                            break

                        img = Image.open(img_path).convert('RGB')

                        # 随机选择增强方法
                        transform = np.random.choice(augmentation_transforms)
                        img_tensor = transform(img)

                        # 转回 PIL Image
                        img_aug = to_pil(img_tensor)

                        # 保存
                        aug_name = f"aug_{aug_idx:04d}_{img_path.name}"
                        save_path = cat_output_dir / aug_name
                        img_aug.save(save_path)

                        aug_idx += 1


def balance_to_target(data_root, output_dir, target_count, method='oversample'):
    """平衡数据集到目标数量"""
    print(f"╔════════════════════════════════════════════════════════════╗")
    print(f"║     🎯 数据集平衡工具 ({method})                            ║")
    print(f"╚════════════════════════════════════════════════════════════╝")
    print()

    # 分析数据集
    class_counts, class_files = analyze_dataset(data_root)

    # 打印统计
    total, max_count, min_count = print_statistics(class_counts)

    print(f"目标: 每类 {target_count} 张")
    print(f"方法: {method}")
    print()

    # 检查是否有空类别
    if min_count == 0:
        print("❌ 错误: 存在空类别，无法平衡")
        print("请先补充数据或使用其他数据集")
        return

    # 检查目标数量是否合理
    if method == 'oversample' and target_count > max_count * 5:
        print(f"⚠️  警告: 目标数量 {target_count} 远大于多数类 {max_count}")
        print(f"     这会导致大量重复图像，可能影响模型性能")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return

    # 创建输出目录
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"⚠️  输出目录已存在: {output_dir}")
        backup_dir = Path(f"{output_dir}_backup")
        print(f"   备份到: {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(output_dir, backup_dir)
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # 执行平衡
    print("="*70)
    print("🔄 开始平衡...")
    print("="*70)
    print()

    if method == 'downsample':
        downsample_majority_class(class_files, target_count, output_dir)
    elif method == 'oversample':
        oversample_minority_classes(class_files, target_count, output_dir, augment=True)
    else:
        print(f"❌ 未知方法: {method}")
        return

    print()
    print("="*70)
    print("✅ 平衡完成！")
    print("="*70)
    print()

    # 验证结果
    print("验证结果:")
    class_counts_after, _ = analyze_dataset(output_dir)
    print_statistics(class_counts_after)

    print(f"✓ 平衡后的数据集已保存到: {output_dir}")


def auto_balance(data_root, output_dir, max_imbalance_ratio=1.5):
    """自动平衡：根据当前分布选择最佳方法"""
    class_counts, _ = analyze_dataset(data_root)

    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)

    if min_count == 0:
        print("❌ 存在空类别，无法自动平衡")
        print("请先补充数据")
        return

    # 根据不平衡比例选择策略
    imbalance_ratio = max_count / min_count

    if imbalance_ratio <= max_imbalance_ratio:
        print("✓ 数据集已经足够平衡，无需处理")
        return

    # 选择目标数量（使用中位数）
    sorted_counts = sorted(counts)
    median_count = sorted_counts[len(sorted_counts) // 2]

    # 如果多数类远大于少数类，考虑下采样
    # 如果少数类太少，使用上采样
    if max_count > median_count * 3:
        target_count = median_count
        method = 'downsample'
    else:
        target_count = median_count
        method = 'oversample'

    print(f"自动决策:")
    print(f"  不平衡比例: {imbalance_ratio:.1f}:1")
    print(f"  目标数量: {target_count}")
    print(f"  使用方法: {method}")
    print()

    balance_to_target(data_root, output_dir, target_count, method)


def main():
    parser = argparse.ArgumentParser(description="平衡垃圾分类数据集")
    parser.add_argument("--data-root", type=str, default="data/raw",
                        help="原始数据目录")
    parser.add_argument("--output", type=str, default="data/balanced",
                        help="输出目录")
    parser.add_argument("--target", type=int, help="每类目标数量")
    parser.add_argument("--method", type=str, choices=['downsample', 'oversample'],
                        help="平衡方法: downsample (下采样) 或 oversample (上采样+增强)")
    parser.add_argument("--auto", action="store_true",
                        help="自动选择最佳策略")

    args = parser.parse_args()

    if args.auto:
        auto_balance(args.data_root, args.output)
    elif args.target:
        if not args.method:
            # 默认使用上采样
            args.method = 'oversample'
        balance_to_target(args.data_root, args.output, args.target, args.method)
    else:
        print("请指定 --target N (每类目标数量) 或使用 --auto")
        print()
        print("示例:")
        print("  下采样到每类 500 张: --target 500 --method downsample")
        print("  上采样到每类 1000 张: --target 1000 --method oversample")
        print("  自动选择: --auto")


if __name__ == "__main__":
    main()
