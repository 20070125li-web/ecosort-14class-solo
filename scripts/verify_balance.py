#!/usr/bin/env python3
"""
验证平衡后的数据集
"""

import os
from pathlib import Path
import argparse


def verify_dataset(data_root):
    """验证数据集平衡性和质量"""
    categories = ['recyclable', 'hazardous', 'kitchen', 'other']
    data_path = Path(data_root)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║     ✓ 数据集验证工具                                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print(f"数据目录: {data_root}")
    print()

    # 检查目录是否存在
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        return False

    # 统计每个类别的图像数量
    class_stats = {}
    total = 0

    print("="*70)
    print("📊 类别分布")
    print("="*70)
    print()

    for cat in categories:
        cat_path = data_path / cat

        if not cat_path.exists():
            print(f"❌ 缺失类别目录: {cat}")
            class_stats[cat] = 0
            continue

        # 统计图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        files = [f for f in cat_path.glob("*") if f.suffix.lower() in image_extensions]

        count = len(files)
        class_stats[cat] = count
        total += count

        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)

        status = "✓" if count > 0 else "❌"
        print(f"{status} {cat:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print()
    print(f"总计: {total} 张图像")
    print()

    # 检查平衡性
    print("="*70)
    print("⚖️  平衡性检查")
    print("="*70)
    print()

    counts = list(class_stats.values())
    max_count = max(counts)
    min_count = min(counts)

    if min_count == 0:
        print("❌ 结果: 失败")
        print("   原因: 存在空类别")
        print()
        print("建议:")
        empty_classes = [cat for cat, count in class_stats.items() if count == 0]
        for cat in empty_classes:
            print(f"   - 为 '{cat}' 类别补充至少 400-500 张图像")
        return False

    imbalance_ratio = max_count / min_count
    print(f"不平衡比例: {imbalance_ratio:.1f}:1")
    print()

    if imbalance_ratio <= 1.2:
        print("✓ 结果: 优秀")
        print("   类别非常平衡，适合训练")
        balance_status = "优秀"
    elif imbalance_ratio <= 1.5:
        print("✓ 结果: 良好")
        print("   类别平衡良好，可以开始训练")
        balance_status = "良好"
    elif imbalance_ratio <= 2.0:
        print("⚠️  结果: 一般")
        print("   轻度不平衡，建议使用类别加权损失")
        balance_status = "一般"
    elif imbalance_ratio <= 3.0:
        print("⚠️  结果: 较差")
        print("   中度不平衡，强烈建议进行数据平衡")
        balance_status = "较差"
    else:
        print("❌ 结果: 很差")
        print("   严重的类别不平衡，必须进行数据平衡")
        balance_status = "很差"

    print()

    # 检查数据量
    print("="*70)
    print("📏 数据量检查")
    print("="*70)
    print()

    avg_count = total / 4

    print(f"平均每类: {avg_count:.0f} 张")
    print()

    if avg_count >= 1000:
        print("✓ 数据量: 充足")
        print("   每类样本数 > 1000，适合深度学习训练")
        data_status = "充足"
    elif avg_count >= 500:
        print("✓ 数据量: 良好")
        print("   每类样本数 > 500，可以使用迁移学习")
        data_status = "良好"
    elif avg_count >= 300:
        print("⚠️  数据量: 一般")
        print("   每类样本数 > 300，建议使用数据增强")
        data_status = "一般"
    elif avg_count >= 100:
        print("⚠️  数据量: 较少")
        print("   每类样本数 < 300，模型性能可能受限")
        data_status = "较少"
    else:
        print("❌ 数据量: 不足")
        print("   每类样本数 < 100，建议收集更多数据")
        data_status = "不足"

    print()

    # 综合评估
    print("="*70)
    print("🎯 综合评估")
    print("="*70)
    print()

    if balance_status in ["优秀", "良好"] and data_status in ["充足", "良好"]:
        print("✅ 数据集质量: 优秀")
        print()
        print("可以开始训练！")
        print()
        print("推荐配置:")
        print(f"  python experiments/train_baseline.py \\")
        print(f"      --config configs/baseline_resnet50.yaml \\")
        print(f"      --data-root {data_root} \\")
        print(f"      --exp-name balanced_training")
        return True
    elif balance_status in ["一般", "较差", "很差"] and data_status in ["一般", "较少", "不足"]:
        print("⚠️  数据集质量: 需要改进")
        print()
        print("建议:")
        if balance_status in ["较差", "很差"]:
            print("  1. 使用数据平衡工具:")
            print("     python scripts/balance_dataset.py --auto")
        if data_status in ["较少", "不足"]:
            print("  2. 下载更多数据:")
            print("     bash scripts/collect_and_balance_data.sh")
        return False
    elif balance_status in ["一般", "较差", "很差"]:
        print("⚠️  主要问题: 类别不平衡")
        print()
        print("建议:")
        print("  python scripts/balance_dataset.py \\")
        print("      --data-root {} \\ --auto".format(data_root))
        return False
    elif data_status in ["一般", "较少", "不足"]:
        print("⚠️  主要问题: 数据量不足")
        print()
        print("建议:")
        print("  bash scripts/collect_and_balance_data.sh")
        return False
    else:
        print("✅ 数据集质量: 可接受")
        print()
        print("可以开始训练，但建议使用类别加权损失")
        return True


def main():
    parser = argparse.ArgumentParser(description="验证平衡后的数据集")
    parser.add_argument("--data-root", type=str, default="data/raw",
                        help="数据目录")

    args = parser.parse_args()

    success = verify_dataset(args.data_root)

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
