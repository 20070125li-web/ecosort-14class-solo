#!/usr/bin/env python3
"""
Merge Multiple Datasets and Map to EcoSort 4-Class System
Unifies diverse trash classification datasets into a standardized 4-class structure
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import defaultdict
import argparse


def load_mapping(config_path):
    """Load class mapping configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def find_image_files(directory):
    """Recursively find all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    images = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                images.append(Path(root) / file)

    return images


def guess_mapping(class_name, mapping_config):
    """Guess target class based on keyword matching when no explicit mapping exists"""
    class_lower = class_name.lower()

    # Check keyword-based mapping rules
    for target_class, keywords in mapping_config['custom_rules']['keyword_mapping'].items():
        for keyword in keywords:
            if keyword in class_lower:
                return target_class

    # Default to 'other' if no match found
    return 'other'


def merge_datasets(source_dir, target_dir, mapping_config, dry_run=False):
    """Merge multiple datasets into target directory with standardized class mapping"""
    target_path = Path(target_dir)
    source_path = Path(source_dir)

    # Create target directory structure
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = defaultdict(lambda: defaultdict(int))

    # Find all source datasets (subdirectories)
    datasets = [d for d in source_path.iterdir() if d.is_dir()]

    print(f"Found {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d.name}")
    print()

    # Process each dataset
    for dataset_dir in datasets:
        print(f"Processing dataset: {dataset_dir.name}")
        print("-" * 70)

        # Find class directories in current dataset
        # Assumes structure: dataset_name/class_name/*.jpg
        classes = [d for d in dataset_dir.iterdir() if d.is_dir()]

        if not classes:
            print(f"  ⚠️  No class directories found - skipping dataset")
            continue

        # Process each class in dataset
        for class_dir in classes:
            class_name = class_dir.name

            # Determine target class mapping
            target_class = None

            # Check predefined mappings first
            for mapping_name in ['trashnet_mapping', 'garbage_classification_mapping',
                                'waste_classification_mapping']:
                if mapping_name in mapping_config and class_name in mapping_config[mapping_name]:
                    target_class = mapping_config[mapping_name][class_name]
                    break

            # Use keyword guessing if no explicit mapping found
            if target_class is None:
                target_class = guess_mapping(class_name, mapping_config)
                print(f"  {class_name:30s} -> {target_class:12s} (auto-inferred)")
            else:
                print(f"  {class_name:30s} -> {target_class:12s}")

            # Find all images in current class directory
            images = find_image_files(class_dir)
            print(f"    Found {len(images)} images")

            if len(images) == 0:
                continue

            # Prepare target directory for mapped class
            target_class_dir = target_path / target_class
            if not dry_run:
                target_class_dir.mkdir(parents=True, exist_ok=True)

            # Copy/link images to standardized structure
            for img_path in images:
                # Generate unique filename: dataset_class_originalname
                new_name = f"{dataset_dir.name}_{class_name}_{img_path.name}"
                target_path_full = target_class_dir / new_name

                if not dry_run:
                    # Copy file (use symlink for space efficiency: shutil.symlink instead)
                    shutil.copy2(img_path, target_path_full)

                stats[dataset_dir.name][target_class] += 1

        print()

    # Print merge statistics
    print("="*70)
    print("📊 Merge Statistics")
    print("="*70)
    print()

    total = 0
    for dataset, class_stats in stats.items():
        print(f"{dataset}:")
        for target_class, count in class_stats.items():
            print(f"  {target_class}: {count}")
            total += count
        print()

    print(f"Total Images Processed: {total}")
    print()

    # Print final EcoSort class distribution
    print("="*70)
    print("📊 EcoSort Class Distribution")
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

    # Check class balance
    if len(final_stats) > 0:
        counts = list(final_stats.values())
        max_count = max(counts)
        min_count = min(counts)

        if min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"Imbalance Ratio: {imbalance_ratio:.1f}:1")

            if imbalance_ratio > 3:
                print("⚠️  Severe class imbalance detected - recommend data augmentation")
            elif imbalance_ratio > 1.5:
                print("⚠️  Moderate imbalance - minor adjustments recommended")
            else:
                print("✓ Good class balance achieved")
        else:
            print("❌ Empty classes detected - need to add more data")
        print()


def main():
    parser = argparse.ArgumentParser(description="Merge multiple datasets into standardized EcoSort 4-class format")
    parser.add_argument("--source", type=str, default="data/kaggle/extracted",
                        help="Source directory containing multiple datasets")
    parser.add_argument("--target", type=str, default="data/raw",
                        help="Target directory for standardized EcoSort dataset")
    parser.add_argument("--mapping", type=str, default="configs/dataset_mapping.yaml",
                        help="Path to class mapping configuration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only - no actual file copying")

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║     📦 EcoSort Dataset Merging Tool                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # Validate source directory
    if not os.path.exists(args.source):
        print(f"❌ Source directory does not exist: {args.source}")
        print("\nPlease run first: bash scripts/collect_and_balance_data.sh")
        return

    # Load mapping configuration
    mapping_config = load_mapping(args.mapping)

    # Backup existing target data if it exists
    target_path = Path(args.target)
    if target_path.exists() and not args.dry_run:
        backup_dir = Path(f"{args.target}_backup")
        print(f"⚠️  Target directory exists - backing up to: {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(args.target, backup_dir)
        print("✓ Backup completed successfully")
        print()

    # Execute dataset merging
    if args.dry_run:
        print("⚠️  DRY RUN MODE - no files will be copied")
        print()

    merge_datasets(args.source, args.target, mapping_config, args.dry_run)

    if not args.dry_run:
        print("✅ Dataset merging completed successfully!")
        print()
        print("💡 Next Steps:")
        print("   1. Verify data quality: python scripts/verify_balance.py --data-root data/raw/")
        print("   2. Balance if needed: python scripts/balance_dataset.py --data-root data/raw/")
        print("   3. Start training: python experiments/train_baseline.py --config configs/baseline_resnet50.yaml")


if __name__ == "__main__":
    main()
