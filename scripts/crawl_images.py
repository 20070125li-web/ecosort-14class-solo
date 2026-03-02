#!/usr/bin/env python3
"""
EcoSort Image Crawling Script (Bing)

Features:
- Batch image crawling by category keywords
- Minimum resolution filtering
- File hash-based deduplication (cross-keyword/cross-batch)
- Metadata output for traceability

Usage:
python scripts/crawl_images.py \
  --config configs/crawl_keywords.yaml \
  --output-root data/raw \
  --per-keyword 120
"""

import argparse
import csv
import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image
from icrawler.builtin import BingImageCrawler

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha1_file(path: Path) -> str:
    """Calculate SHA1 hash of file content for deduplication"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path, min_w: int, min_h: int) -> bool:
    """Validate image file format and minimum resolution"""
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= min_w and h >= min_h
    except Exception:
        return False


def build_existing_hashes(output_root: Path) -> Dict[str, str]:
    """Build hash map of existing images to detect duplicates"""
    hashes = {}
    if not output_root.exists():
        return hashes
    for p in output_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            try:
                hashes[sha1_file(p)] = str(p)
            except Exception:
                continue
    return hashes


def load_config(path: Path) -> Dict:
    """Load crawling configuration from YAML file"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "classes" not in cfg or not isinstance(cfg["classes"], dict):
        raise ValueError("Config file must contain 'classes' field")
    return cfg


def crawl_one_keyword(keyword: str, max_num: int, workers: int, out_dir: Path):
    """Crawl images for single keyword using BingImageCrawler"""
    crawler = BingImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=workers,
        storage={"root_dir": str(out_dir)},
    )
    crawler.crawl(keyword=keyword, max_num=max_num)


def main():
    parser = argparse.ArgumentParser(description="EcoSort Image Crawler (Bing)")
    parser.add_argument("--config", type=str, default="configs/crawl_keywords.yaml",
                        help="Path to YAML config file with crawl keywords")
    parser.add_argument("--output-root", type=str, default="data/raw",
                        help="Root directory for crawled images")
    parser.add_argument("--per-keyword", type=int, default=120,
                        help="Maximum images to crawl per keyword")
    parser.add_argument("--min-width", type=int, default=256,
                        help="Minimum image width (pixels)")
    parser.add_argument("--min-height", type=int, default=256,
                        help="Minimum image height (pixels)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of download worker threads")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Sleep seconds between keyword crawls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate crawl without downloading files")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    existing_hashes = build_existing_hashes(output_root)
    metadata_path = output_root / "crawl_metadata.csv"

    new_rows: List[List[str]] = []
    summary = {}

    for class_name, class_cfg in config["classes"].items():
        keywords = class_cfg.get("keywords", [])
        target_count = int(class_cfg.get("target_count", 300))
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len([p for p in class_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        need = max(target_count - existing_count, 0)

        print(f"\n[Class] {class_name} | existing={existing_count} target={target_count} need={need}")

        added = 0
        invalid = 0
        dup = 0

        if need == 0:
            summary[class_name] = {"added": 0, "invalid": 0, "duplicate": 0, "final": existing_count}
            continue

        for kw in keywords:
            if added >= need:
                break

            to_crawl = min(args.per_keyword, need - added)
            print(f"  - keyword: {kw} | request={to_crawl}")

            if args.dry_run:
                time.sleep(args.sleep)
                continue

            with tempfile.TemporaryDirectory(prefix=f"crawl_{class_name}_") as td:
                td_path = Path(td)
                crawl_one_keyword(kw, to_crawl, args.workers, td_path)

                for p in td_path.rglob("*"):
                    if not p.is_file():
                        continue

                    if not is_valid_image(p, args.min_width, args.min_height):
                        invalid += 1
                        continue

                    try:
                        file_hash = sha1_file(p)
                    except Exception:
                        invalid += 1
                        continue

                    if file_hash in existing_hashes:
                        dup += 1
                        continue

                    ts = int(time.time() * 1000)
                    new_name = f"{class_name}_{ts}_{added:05d}{p.suffix.lower()}"
                    dst = class_dir / new_name
                    shutil.move(str(p), str(dst))

                    existing_hashes[file_hash] = str(dst)
                    added += 1

                    new_rows.append([
                        class_name,
                        kw,
                        str(dst),
                        file_hash,
                        str(args.min_width),
                        str(args.min_height),
                    ])

                    if added >= need:
                        break

            time.sleep(args.sleep)

        final_count = len([p for p in class_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        summary[class_name] = {
            "added": added,
            "invalid": invalid,
            "duplicate": dup,
            "final": final_count,
        }

    if not args.dry_run and new_rows:
        write_header = not metadata_path.exists()
        with open(metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["class_name", "keyword", "file_path", "sha1", "min_width", "min_height"])
            writer.writerows(new_rows)

    print("\n========== Crawl Summary ==========")
    for cls, stat in summary.items():
        print(
            f"{cls:12s} | added={stat['added']:4d} invalid={stat['invalid']:4d} "
            f"dup={stat['duplicate']:4d} final={stat['final']:4d}"
        )
    print("===================================")
    if args.dry_run:
        print("DRY-RUN completed (no files downloaded)")
    else:
        print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
