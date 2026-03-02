#!/usr/bin/env python3
"""
Validate Balanced Dataset
Check dataset balance, completeness and quality metrics
"""

import os
from pathlib import Path
import argparse


def verify_dataset(data_root):
    """Validate dataset balance and quality metrics"""
    categories = ['recyclable', 'hazardous', 'kitchen', 'other']
    data_path = Path(data_root)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║     ✓ Dataset Validation Tool                              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print(f"Data Directory: {data_root}")
    print()

    # Check if root directory exists
    if not data_path.exists():
        print(f"❌ Data directory does not exist: {data_root}")
        return False

    # Count images per category
    class_stats = {}
    total = 0

    print("="*70)
    print("📊 Class Distribution")
    print("="*70)
    print()

    for cat in categories:
        cat_path = data_path / cat

        if not cat_path.exists():
            print(f"❌ Missing category directory: {cat}")
            class_stats[cat] = 0
            continue

        # Count image files (support common formats)
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
    print(f"Total Images: {total}")
    print()

    # Check balance metrics
    print("="*70)
    print("⚖️  Balance Check")
    print("="*70)
    print()

    counts = list(class_stats.values())
    max_count = max(counts)
    min_count = min(counts)

    if min_count == 0:
        print("❌ Result: FAILED")
        print("   Reason: Empty categories detected")
        print()
        print("Recommendations:")
        empty_classes = [cat for cat, count in class_stats.items() if count == 0]
        for cat in empty_classes:
            print(f"   - Add at least 400-500 images to '{cat}' category")
        return False

    imbalance_ratio = max_count / min_count
    print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print()

    if imbalance_ratio <= 1.2:
        print("✓ Result: EXCELLENT")
        print("   Classes are well-balanced - ideal for training")
        balance_status = "EXCELLENT"
    elif imbalance_ratio <= 1.5:
        print("✓ Result: GOOD")
        print("   Classes are reasonably balanced - ready for training")
        balance_status = "GOOD"
    elif imbalance_ratio <= 2.0:
        print("⚠️  Result: FAIR")
        print("   Mild imbalance detected - recommend class-weighted loss")
        balance_status = "FAIR"
    elif imbalance_ratio <= 3.0:
        print("⚠️  Result: POOR")
        print("   Moderate imbalance - strong recommendation to balance data")
        balance_status = "POOR"
    else:
        print("❌ Result: VERY POOR")
        print("   Severe class imbalance - data balancing required")
        balance_status = "VERY POOR"

    print()

    # Check data volume adequacy
    print("="*70)
    print("📏 Data Volume Check")
    print("="*70)
    print()

    avg_count = total / 4

    print(f"Average per class: {avg_count:.0f} images")
    print()

    if avg_count >= 1000:
        print("✓ Data Volume: SUFFICIENT")
        print("   >1000 samples per class - ideal for deep learning training")
        data_status = "SUFFICIENT"
    elif avg_count >= 500:
        print("✓ Data Volume: GOOD")
        print("   >500 samples per class - suitable for transfer learning")
        data_status = "GOOD"
    elif avg_count >= 300:
        print("⚠️  Data Volume: FAIR")
        print("   >300 samples per class - recommend data augmentation")
        data_status = "FAIR"
    elif avg_count >= 100:
        print("⚠️  Data Volume: LOW")
        print("   <300 samples per class - model performance may be limited")
        data_status = "LOW"
    else:
        print("❌ Data Volume: INSUFFICIENT")
        print("   <100 samples per class - recommend collecting more data")
        data_status = "INSUFFICIENT"

    print()

    # Comprehensive assessment
    print("="*70)
    print("🎯 Comprehensive Assessment")
    print("="*70)
    print()

    if balance_status in ["EXCELLENT", "GOOD"] and data_status in ["SUFFICIENT", "GOOD"]:
        print("✅ Dataset Quality: EXCELLENT")
        print()
        print("Ready for training!")
        print()
        print("Recommended Training Command:")
        print(f"  python experiments/train_baseline.py \\")
        print(f"      --config configs/baseline_resnet50.yaml \\")
        print(f"      --data-root {data_root} \\")
        print(f"      --exp-name balanced_training")
        return True
    elif balance_status in ["FAIR", "POOR", "VERY POOR"] and data_status in ["FAIR", "LOW", "INSUFFICIENT"]:
        print("⚠️  Dataset Quality: NEEDS IMPROVEMENT")
        print()
        print("Recommendations:")
        if balance_status in ["POOR", "VERY POOR"]:
            print("  1. Run data balancing tool:")
            print("     python scripts/balance_dataset.py --auto")
        if data_status in ["LOW", "INSUFFICIENT"]:
            print("  2. Download additional data:")
            print("     bash scripts/collect_and_balance_data.sh")
        return False
    elif balance_status in ["FAIR", "POOR", "VERY POOR"]:
        print("⚠️  Primary Issue: Class Imbalance")
        print()
        print("Recommendation:")
        print("  python scripts/balance_dataset.py \\")
        print(f"      --data-root {data_root} --auto")
        return False
    elif data_status in ["FAIR", "LOW", "INSUFFICIENT"]:
        print("⚠️  Primary Issue: Insufficient Data Volume")
        print()
        print("Recommendation:")
        print("  bash scripts/collect_and_balance_data.sh")
        return False
    else:
        print("✅ Dataset Quality: ACCEPTABLE")
        print()
        print("Training can proceed - recommend using class-weighted loss function")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate balanced dataset quality and balance")
    parser.add_argument("--data-root", type=str, default="data/raw",
                        help="Root directory of dataset")

    args = parser.parse_args()

    success = verify_dataset(args.data_root)

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
