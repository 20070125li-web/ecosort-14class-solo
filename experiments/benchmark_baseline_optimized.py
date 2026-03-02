"""
Baseline vs Optimized (Quantized) Comparison Benchmark Script

Functionality:
1. Load trained checkpoint as Baseline model
2. Generate INT8 dynamic quantized model as Optimized version
3. Evaluate Accuracy/F1/Latency/Model Size on the same data split and device (CPU)
4. Save results to checkpoints/<exp>/benchmark/
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.data.dataset import TrashDataset, get_data_transforms
from src.models.efficientnet_classifier import create_efficientnet_model
from src.models.resnet_classifier import create_resnet_model
from src.train.trainer import load_checkpoint
from src.utils.quantization import post_training_quantization


def _infer_model_spec(checkpoint: Dict) -> Tuple[str, str, int, float, bool]:
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    model_type = model_cfg.get("type", "resnet")
    backbone = model_cfg.get("backbone", "resnet50")

    num_classes = model_cfg.get("num_classes")
    if num_classes is None:
        class_counts = data_cfg.get("class_counts", [])
        num_classes = len(class_counts) if class_counts else len(TrashDataset.CLASS_NAMES)

    dropout = float(model_cfg.get("dropout", 0.3))
    use_attention = bool(model_cfg.get("use_attention", False))
    return model_type, backbone, int(num_classes), dropout, use_attention


def _build_model(model_type: str, backbone: str, num_classes: int, dropout: float, use_attention: bool):
    if model_type == "resnet":
        return create_resnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
            use_attention=use_attention,
        )
    if model_type == "efficientnet":
        return create_efficientnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _state_dict_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = Path(f.name).stat().st_size / (1024 * 1024)
    return float(size_mb)


def _evaluate_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _benchmark_latency_ms(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_samples: int = 120) -> float:
    model.eval()
    latencies = []

    warmup = 10
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= warmup:
                break
            _ = model(images.to(device))

    counted = 0
    with torch.no_grad():
        for images, _ in loader:
            for sample in images:
                if counted >= max_samples:
                    break
                sample = sample.unsqueeze(0).to(device)
                start = time.perf_counter()
                _ = model(sample)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)
                counted += 1
            if counted >= max_samples:
                break

    return float(np.mean(latencies)) if latencies else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline vs quantized model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to baseline checkpoint .pth")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Dataset root directory")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size (pixels)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for metric evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader worker processes")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--max-latency-samples", type=int, default=120, help="Maximum samples for latency benchmarking")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (default: checkpoints/<exp>/benchmark)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    exp_dir = checkpoint_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    checkpoint_raw = torch.load(checkpoint_path, map_location="cpu")
    model_type, backbone, num_classes, dropout, use_attention = _infer_model_spec(checkpoint_raw)

    baseline_model = _build_model(model_type, backbone, num_classes, dropout, use_attention)
    _ = load_checkpoint(str(checkpoint_path), baseline_model)
    baseline_model = baseline_model.to(device).eval()

    quantized_model = post_training_quantization(baseline_model).to(device).eval()

    val_dataset = TrashDataset(
        root_dir=args.data_root,
        transform=get_data_transforms("val", args.img_size),
        split="val",
        val_split=args.val_split,
    )

    eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    latency_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, 16),
        shuffle=False,
        num_workers=args.num_workers,
    )

    baseline_metrics = _evaluate_metrics(baseline_model, eval_loader, device)
    quant_metrics = _evaluate_metrics(quantized_model, eval_loader, device)

    baseline_latency = _benchmark_latency_ms(
        baseline_model, latency_loader, device, max_samples=args.max_latency_samples
    )
    quant_latency = _benchmark_latency_ms(
        quantized_model, latency_loader, device, max_samples=args.max_latency_samples
    )

    baseline_size = _state_dict_size_mb(baseline_model)
    quant_size = _state_dict_size_mb(quantized_model)

    quant_model_path = output_dir / "optimized_int8_model.pth"
    torch.save(quantized_model, quant_model_path)

    summary = {
        "experiment": exp_dir.name,
        "checkpoint": str(checkpoint_path),
        "data_root": args.data_root,
        "num_classes": num_classes,
        "class_names": TrashDataset.CLASS_NAMES[:num_classes],
        "device": str(device),
        "baseline": {
            **baseline_metrics,
            "latency_ms": baseline_latency,
            "model_size_mb": baseline_size,
        },
        "optimized_int8": {
            **quant_metrics,
            "latency_ms": quant_latency,
            "model_size_mb": quant_size,
        },
        "deltas": {
            "accuracy_drop": float(quant_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "f1_drop": float(quant_metrics["f1_macro"] - baseline_metrics["f1_macro"]),
            "latency_speedup_x": float(baseline_latency / quant_latency) if quant_latency > 0 else None,
            "size_reduction_pct": float((1 - quant_size / baseline_size) * 100) if baseline_size > 0 else None,
        },
        "artifacts": {
            "optimized_model": str(quant_model_path),
        },
    }

    with open(output_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    markdown = [
        "# Baseline vs Optimized (INT8) Benchmark Report",
        "",
        f"- Experiment: {summary['experiment']}",
        f"- Device: {summary['device']}",
        "",
        "| Model | Accuracy | F1(macro) | Latency(ms) | Size(MB) |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline | {summary['baseline']['accuracy']:.4f} | {summary['baseline']['f1_macro']:.4f} | {summary['baseline']['latency_ms']:.2f} | {summary['baseline']['model_size_mb']:.2f} |",
        f"| Optimized INT8 | {summary['optimized_int8']['accuracy']:.4f} | {summary['optimized_int8']['f1_macro']:.4f} | {summary['optimized_int8']['latency_ms']:.2f} | {summary['optimized_int8']['model_size_mb']:.2f} |",
        "",
        "## Performance Delta",
        f"- Accuracy change: {summary['deltas']['accuracy_drop']:+.4f}",
        f"- F1 change: {summary['deltas']['f1_drop']:+.4f}",
        f"- Inference speedup: {summary['deltas']['latency_speedup_x']:.2f}x" if summary['deltas']['latency_speedup_x'] else "- Inference speedup: N/A",
        f"- Model size reduction: {summary['deltas']['size_reduction_pct']:.2f}%" if summary['deltas']['size_reduction_pct'] is not None else "- Model size reduction: N/A",
    ]

    with open(output_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))

    print("=" * 60)
    print("Benchmark completed successfully")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
