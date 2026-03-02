#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_valid_images(input_root: Path, min_w: int, min_h: int):
    per_class: Dict[str, List[Path]] = {}
    invalid_files: List[str] = []

    class_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        images = []
        for fp in class_dir.iterdir():
            if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(fp) as img:
                    w, h = img.size
                if w < min_w or h < min_h:
                    invalid_files.append(str(fp))
                    continue
                images.append(fp)
            except Exception:
                invalid_files.append(str(fp))
                continue
        if images:
            per_class[class_dir.name] = images

    return per_class, invalid_files


def global_dedup(per_class: Dict[str, List[Path]]):
    seen: Dict[str, str] = {}
    deduped: Dict[str, List[Path]] = {}
    dropped_dup = []

    for cls in sorted(per_class.keys()):
        deduped[cls] = []
        for fp in per_class[cls]:
            try:
                h = sha1_file(fp)
            except Exception:
                continue
            if h in seen:
                dropped_dup.append((str(fp), seen[h]))
                continue
            seen[h] = str(fp)
            deduped[cls].append(fp)
    return deduped, dropped_dup


def split_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if n <= 1:
        return n, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_val == 0:
        n_val = 1
    if n_test == 0:
        n_test = 1

    if n_train <= 0:
        n_train = max(1, n - n_val - n_test)

    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    while n_train + n_val + n_test < n:
        n_train += 1

    return n_train, n_val, n_test


def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        src_h = sha1_file(src)
        if dst.exists() and sha1_file(dst) == src_h:
            return
    except Exception:
        pass

    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def preprocess(
    input_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    min_w: int,
    min_h: int,
    seed: int,
):
    random.seed(seed)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    per_class, invalid_files = read_valid_images(input_root, min_w=min_w, min_h=min_h)
    per_class, dropped_dup = global_dedup(per_class)

    split_stats = {}
    manifest_rows = []

    for cls in sorted(per_class.keys()):
        files = per_class[cls]
        random.shuffle(files)

        n = len(files)
        n_train, n_val, n_test = split_counts(n, train_ratio, val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:n_train + n_val + n_test]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for src in split_files:
                dst = output_root / split_name / cls / src.name
                safe_link_or_copy(src, dst)
                manifest_rows.append([split_name, cls, str(src), str(dst)])

        split_stats[cls] = {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "min_size": [min_w, min_h],
        "seed": seed,
        "class_count": len(split_stats),
        "invalid_file_count": len(invalid_files),
        "duplicate_file_count": len(dropped_dup),
        "split_stats": split_stats,
    }

    with open(output_root / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(output_root / "invalid_files.txt", "w", encoding="utf-8") as f:
        for item in invalid_files:
            f.write(item + "\n")

    with open(output_root / "duplicate_files.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dropped_file", "kept_file"])
        writer.writerows(dropped_dup)

    with open(output_root / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_name", "source_path", "target_path"])
        writer.writerows(manifest_rows)

    with open(output_root / "class_distribution.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "total", "train", "val", "test"])
        for cls in sorted(split_stats.keys()):
            s = split_stats[cls]
            writer.writerow([cls, s["total"], s["train"], s["val"], s["test"]])

    print("=" * 70)
    print("Preprocess completed")
    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {len(split_stats)}")
    print(f"Invalid filtered: {len(invalid_files)}")
    print(f"Duplicates removed: {len(dropped_dup)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Preprocess crawled dataset")
    parser.add_argument("--input-root", type=str, default="data/raw")
    parser.add_argument("--output-root", type=str, default="data/proc/crawled_v1")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 比例之和需为1.0，当前为 {ratio_sum}")

    preprocess(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_w=args.min_width,
        min_h=args.min_height,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
