#!/usr/bin/env python3
"""
Balance Dataset: Achieve class balance through downsampling or data augmentation
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
    """Analyze dataset class distribution"""
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
    """Print dataset statistics and distribution"""
    print("="*70)
    print("📊 Current Dataset Statistics")
    print("="*70)
    print()

    total = sum(class_counts.values())

    print("Class Distribution:")
    print("-" * 70)

    for cat, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {cat:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print("-" * 70)
    print(f"  {'Total':12s}: {total:5d}")
    print()

    # Check balance metrics
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)

    if min_count > 0:
        imbalance_ratio = max_count / min_count
        print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")
        if imbalance_ratio > 5:
            print("⚠️  Severe class imbalance detected!")
        elif imbalance_ratio > 2:
            print("⚠️  Moderate class imbalance")
        else:
            print("✓ Good class balance")
    else:
        print("❌ Empty classes detected")
    print()

    return total, max_count, min_count


def downsample_majority_class(class_files, target_count, output_dir):
    """Downsample majority classes to target count"""
    print(f"Downsampling to {target_count} images per class...")

    for cat, files in class_files.items():
        cat_output_dir = Path(output_dir) / cat
        cat_output_dir.mkdir(parents=True, exist_ok=True)

        if len(files) <= target_count:
            # No downsampling needed - copy all files
            for img_path in files:
                shutil.copy2(img_path, cat_output_dir / img_path.name)
            print(f"  {cat}: {len(files)} images (no downsampling needed)")
        else:
            # Random sampling without replacement
            np.random.seed(42)
            sampled_indices = np.random.choice(len(files), target_count, replace=False)

            for idx in sampled_indices:
                img_path = files[idx]
                shutil.copy2(img_path, cat_output_dir / img_path.name)

            print(f"  {cat}: {len(files)} -> {target_count} images (downsampled)")


def oversample_minority_classes(class_files, target_count, output_dir, augment=True):
    """Oversample minority classes using data augmentation"""
    print(f"Oversampling to {target_count} images per class...")

    # Data augmentation transformations
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
            # No oversampling needed - random selection
            np.random.seed(42)
            sampled_indices = np.random.choice(len(files), target_count, replace=False)
            for idx in sampled_indices:
                img_path = files[idx]
                shutil.copy2(img_path, cat_output_dir / img_path.name)
            print(f"  {cat}: {len(files)} -> {target_count} images (random selection)")
        else:
            # Copy all original images first
            for img_path in files:
                shutil.copy2(img_path, cat_output_dir / img_path.name)

            needed = target_count - len(files)
            print(f"  {cat}: {len(files)} -> {target_count} images (need {needed} augmented images)")

            if augment and needed > 0:
                # Generate additional images with augmentation
                aug_idx = 0
                while aug_idx < needed:
                    for img_path in files:
                        if aug_idx >= needed:
                            break

                        img = Image.open(img_path).convert('RGB')

                        # Randomly select augmentation method
                        transform = np.random.choice(augmentation_transforms)
                        img_tensor = transform(img)

                        # Convert back to PIL Image
                        img_aug = to_pil(img_tensor)

                        # Save augmented image
                        aug_name = f"aug_{aug_idx:04d}_{img_path.name}"
                        save_path = cat_output_dir / aug_name
                        img_aug.save(save_path)

                        aug_idx += 1


def balance_to_target(data_root, output_dir, target_count, method='oversample'):
    """Balance dataset to target count per class"""
    print(f"╔════════════════════════════════════════════════════════════╗")
    print(f"║     🎯 Dataset Balancing Tool ({method})                    ║")
    print(f"╚════════════════════════════════════════════════════════════╝")
    print()

    # Analyze dataset
    class_counts, class_files = analyze_dataset(data_root)

    # Print statistics
    total, max_count, min_count = print_statistics(class_counts)

    print(f"Target: {target_count} images per class")
    print(f"Method: {method}")
    print()

    # Check for empty classes
    if min_count == 0:
        print("❌ Error: Empty classes detected - cannot balance dataset")
        print("Please add data to empty classes or use a different dataset")
        return

    # Validate target count
    if method == 'oversample' and target_count > max_count * 5:
        print(f"⚠️  Warning: Target count ({target_count}) is much higher than majority class ({max_count})")
        print(f"     This will create many duplicate/augmented images which may hurt model performance")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"⚠️  Output directory exists: {output_dir}")
        backup_dir = Path(f"{output_dir}_backup")
        print(f"   Backing up to: {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(output_dir, backup_dir)
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # Execute balancing
    print("="*70)
    print("🔄 Starting dataset balancing...")
    print("="*70)
    print()

    if method == 'downsample':
        downsample_majority_class(class_files, target_count, output_dir)
    elif method == 'oversample':
        oversample_minority_classes(class_files, target_count, output_dir, augment=True)
    else:
        print(f"❌ Unknown method: {method}")
        return

    print()
    print("="*70)
    print("✅ Dataset balancing completed!")
    print("="*70)
    print()

    # Verify results
    print("Verification Results:")
    class_counts_after, _ = analyze_dataset(output_dir)
    print_statistics(class_counts_after)

    print(f"✓ Balanced dataset saved to: {output_dir}")


def auto_balance(data_root, output_dir, max_imbalance_ratio=1.5):
    """Auto-balance: Select optimal strategy based on current distribution"""
    class_counts, _ = analyze_dataset(data_root)

    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)

    if min_count == 0:
        print("❌ Empty classes detected - cannot auto-balance")
        print("Please add data to empty classes first")
        return

    # Calculate imbalance ratio
    imbalance_ratio = max_count / min_count

    if imbalance_ratio <= max_imbalance_ratio:
        print("✓ Dataset is already sufficiently balanced - no action needed")
        return

    # Select target count (using median)
    sorted_counts = sorted(counts)
    median_count = sorted_counts[len(sorted_counts) // 2]

    # Choose strategy based on distribution
    if max_count > median_count * 3:
        target_count = median_count
        method = 'downsample'
    else:
        target_count = median_count
        method = 'oversample'

    print(f"Auto-Decision:")
    print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"  Target Count: {target_count}")
    print(f"  Selected Method: {method}")
    print()

    balance_to_target(data_root, output_dir, target_count, method)


def main():
    parser = argparse.ArgumentParser(description="Trash Classification Dataset Balancer")
    parser.add_argument("--data-root", type=str, default="data/raw",
                        help="Root directory of original dataset")
    parser.add_argument("--output", type=str, default="data/balanced",
                        help="Output directory for balanced dataset")
    parser.add_argument("--target", type=int,
                        help="Target number of images per class")
    parser.add_argument("--method", type=str, choices=['downsample', 'oversample'],
                        help="Balancing method: downsample (majority) or oversample (minority with augmentation)")
    parser.add_argument("--auto", action="store_true",
                        help="Automatically select optimal balancing strategy")

    args = parser.parse_args()

    if args.auto:
        auto_balance(args.data_root, args.output)
    elif args.target:
        if not args.method:
            # Default to oversampling
            args.method = 'oversample'
        balance_to_target(args.data_root, args.output, args.target, args.method)
    else:
        print("Please specify either --target N (target images per class) or use --auto")
        print()
        print("Examples:")
        print("  Downsample to 500 images per class: --target 500 --method downsample")
        print("  Oversample to 1000 images per class: --target 1000 --method oversample")
        print("  Auto-balance: --auto")


if __name__ == "__main__":
    main()
