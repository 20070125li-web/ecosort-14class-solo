"""
EcoSort Streamlit 对比 Demo

展示 Baseline vs Optimized(INT8):
- 预测类别与置信度
- 单张推理时延
- 模型体积
- 最近一次 benchmark 摘要
"""

import argparse
import io
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.models.efficientnet_classifier import create_efficientnet_model
from src.models.resnet_classifier import create_resnet_model
from src.train.trainer import load_checkpoint
from src.utils.quantization import post_training_quantization


DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def _detect_model_type_from_state_dict(state_dict: Dict) -> Tuple[str, str]:
    """从 state_dict keys 自动检测模型类型

    Args:
        state_dict: 模型权重字典

    Returns:
        (model_type, backbone): 模型类型和骨架名称
    """
    keys = list(state_dict.keys())

    # EfficientNet 特征: backbone._conv_stem, backbone._blocks
    if any(k.startswith("backbone._conv_stem") for k in keys):
        return "efficientnet", "efficientnet-b3"

    # EfficientNet 特征: backbone._blocks
    if any(k.startswith("backbone._blocks") for k in keys):
        return "efficientnet", "efficientnet-b3"

    # ResNet 特征: features.0.weight 或 layer1.0.conv1.weight
    if any(k.startswith("features.") or k.startswith("layer") for k in keys):
        return "resnet", "resnet50"

    # 默认尝试 EfficientNet (更常见)
    return "efficientnet", "efficientnet-b3"


def _infer_model_spec(checkpoint: Dict) -> Tuple:
    """从 checkpoint 推断模型规格

    Args:
        checkpoint: checkpoint 字典

    Returns:
        (model_type, backbone, num_classes, dropout, use_attention)
    """
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    # 优先从 config 获取
    model_type = model_cfg.get("type")
    backbone = model_cfg.get("backbone")

    # 如果 config 缺失，从 state_dict 自动检测
    if not model_type or not backbone:
        state_dict = checkpoint.get("model_state_dict", {})
        if state_dict:
            detected_type, detected_backbone = _detect_model_type_from_state_dict(state_dict)
            model_type = model_type or detected_type
            backbone = backbone or detected_backbone

    # 默认值
    if not model_type:
        model_type = "efficientnet"
    if not backbone:
        backbone = "efficientnet-b3"

    num_classes = model_cfg.get("num_classes")
    if num_classes is None:
        class_counts = data_cfg.get("class_counts", [])
        num_classes = len(class_counts) if class_counts else len(DEFAULT_CLASSES)
    dropout = float(model_cfg.get("dropout", 0.3))
    use_attention = bool(model_cfg.get("use_attention", False))
    return model_type, backbone, int(num_classes), dropout, use_attention


def _infer_class_names(checkpoint: Dict, data_root: Optional[str], num_classes: int) -> List[str]:
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    config_names = data_cfg.get("class_names")
    if isinstance(config_names, list) and len(config_names) == num_classes:
        return [str(x) for x in config_names]

    if data_root:
        train_dir = Path(data_root) / "train"
        if train_dir.exists():
            names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
            if len(names) == num_classes:
                return names

    if num_classes <= len(DEFAULT_CLASSES):
        return DEFAULT_CLASSES[:num_classes]

    return [f"class_{i}" for i in range(num_classes)]


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


def _model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return Path(f.name).stat().st_size / (1024 * 1024)


@st.cache_resource
def load_models(checkpoint_path: str, data_root: str = ""):
    """加载基线模型 (INT8 量化模型将在用户选择时按需加载)

    Returns:
        baseline: 基线模型
        None: 量化模型占位符 (按需加载)
        meta: 元数据字典
    """
    start_load = time.perf_counter()

    # 加载 checkpoint
    checkpoint_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type, backbone, num_classes, dropout, use_attention = _infer_model_spec(checkpoint_raw)

    # 构建并加载基线模型
    baseline = _build_model(model_type, backbone, num_classes, dropout, use_attention)
    try:
        _ = load_checkpoint(checkpoint_path, baseline)
    except Exception as e:
        st.error(f"加载模型权重失败: {e}")
        st.info(f"检测到模型类型: {model_type}, backbone: {backbone}")
        st.stop()

    baseline = baseline.eval().cpu()
    load_time = time.perf_counter() - start_load

    class_names = _infer_class_names(checkpoint_raw, data_root, num_classes)

    meta = {
        "model_type": model_type,
        "backbone": backbone,
        "num_classes": num_classes,
        "class_names": class_names,
        "baseline_size_mb": _model_size_mb(baseline),
        "load_time_sec": load_time,
        "checkpoint_path": checkpoint_path,
    }
    return baseline, None, meta


@st.cache_resource
def load_quantized_model(_baseline: torch.nn.Module):
    """按需加载 INT8 量化模型 (懒加载)

    Args:
        _baseline: 基线模型 (用于缓存键)

    Returns:
        optimized: 量化后的模型
    """
    start_q = time.perf_counter()
    optimized = post_training_quantization(_baseline).eval().cpu()
    quant_time = time.perf_counter() - start_q
    return optimized, quant_time


def preprocess(image: Image.Image, img_size: int = 224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(image).unsqueeze(0)


def extract_video_frames(video_bytes: bytes, sample_every_n: int, max_frames: int) -> List[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        frame_idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            frame_idx += 1
        cap.release()
    return frames


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names,
    confidence_threshold: float = 0.70,
    margin_threshold: float = 0.15,
):
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(image_tensor)
        end = time.perf_counter()
        probs = torch.softmax(logits, dim=1)[0]
        top_values, top_indices = torch.topk(probs, k=min(3, len(class_names)))

    top1_conf = float(top_values[0].item())
    top1_idx = int(top_indices[0].item())
    top1_name = class_names[top1_idx]
    top2_conf = float(top_values[1].item()) if len(top_values) > 1 else 0.0
    margin = top1_conf - top2_conf

    uncertain = top1_conf < confidence_threshold or margin < margin_threshold
    final_label = "unknown_review_needed" if uncertain else top1_name

    return {
        "class_name": final_label,
        "raw_top1_class": top1_name,
        "confidence": top1_conf,
        "top2_confidence": top2_conf,
        "margin": margin,
        "uncertain": uncertain,
        "latency_ms": (end - start) * 1000.0,
        "probs": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
    }


def summarize_video_predictions(predictions: List[Dict]) -> Dict:
    labels = [p["class_name"] for p in predictions]
    raw_labels = [p["raw_top1_class"] for p in predictions]
    confs = [p["confidence"] for p in predictions]
    uncertain_count = sum(1 for p in predictions if p["uncertain"])

    voted_label = Counter(labels).most_common(1)[0][0] if labels else "unknown_review_needed"
    voted_raw = Counter(raw_labels).most_common(1)[0][0] if raw_labels else "unknown"

    return {
        "final_label": voted_label,
        "raw_majority_label": voted_raw,
        "avg_confidence": float(np.mean(confs)) if confs else 0.0,
        "uncertain_ratio": (uncertain_count / len(predictions)) if predictions else 1.0,
        "frame_count": len(predictions),
    }


def render_report_if_exists(checkpoint_path: str):
    benchmark_json = Path(checkpoint_path).parent / "benchmark" / "comparison_report.json"
    if benchmark_json.exists():
        with open(benchmark_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        st.subheader("最近一次基准对比")
        st.json(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/expansion24_effb3_reviewed_longstop_20260224_155117/best_model.pth",
        help="Baseline checkpoint path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/proc/expansion24_freeze16_v1",
        help="Data root used to infer class names from train directory",
    )
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="EcoSort 垃圾分类网页演示", layout="wide")
    st.title("EcoSort 垃圾分类演示")
    st.caption("支持上传图片或视频进行垃圾分类识别（含低置信度保护）")

    if not Path(args.checkpoint).exists():
        st.error(f"Checkpoint not found: {args.checkpoint}")
        st.stop()

    baseline, _, meta = load_models(args.checkpoint, args.data_root)

    # 显示模型信息和加载时间
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("模型类型", f"{meta['model_type']} / {meta['backbone']}")
    with col2:
        st.metric("类别数", meta['num_classes'])
    with col3:
        st.metric("加载耗时", f"{meta['load_time_sec']:.2f}s")

    st.caption(f"模型大小: {meta['baseline_size_mb']:.2f} MB")

    st.subheader("不确定性保护")
    conf_th = st.slider("最低置信度阈值", min_value=0.40, max_value=0.95, value=0.70, step=0.01)
    margin_th = st.slider("Top1-Top2 最小间隔阈值", min_value=0.00, max_value=0.40, value=0.15, step=0.01)

    # 模型选择与懒加载
    model_choice = st.radio("推理模型", options=["Baseline", "Optimized (INT8)"], horizontal=True)

    if model_choice == "Baseline":
        selected_model = baseline
        model_size_mb = meta['baseline_size_mb']
    else:
        # 懒加载 INT8 量化模型
        if "optimized_model" not in st.session_state:
            with st.spinner("正在进行 INT8 量化 (首次使用较慢)..."):
                st.session_state.optimized_model, st.session_state.quant_time = load_quantized_model(baseline)
        selected_model = st.session_state.optimized_model
        model_size_mb = meta.get("optimized_size_mb", _model_size_mb(selected_model))

    st.write(f"当前模型大小: {model_size_mb:.2f} MB")
    if model_choice == "Optimized (INT8)" and "quant_time" in st.session_state:
        st.caption(f"量化耗时: {st.session_state.quant_time:.2f}s")

    input_mode = st.radio("输入类型", options=["图片", "视频"], horizontal=True)

    if input_mode == "图片":
        uploaded = st.file_uploader("上传垃圾图片", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(image, caption="输入图片", width=360)
            tensor = preprocess(image, img_size=300)
            res = predict(
                selected_model,
                tensor,
                meta["class_names"],
                confidence_threshold=conf_th,
                margin_threshold=margin_th,
            )
            st.subheader("图片识别结果")
            st.write(f"预测: **{res['class_name']}**")
            st.write(f"原始Top1: {res['raw_top1_class']}")
            st.write(f"置信度: {res['confidence']:.4f}")
            st.write(f"Top1-Top2 间隔: {res['margin']:.4f}")
            st.write(f"推理耗时: {res['latency_ms']:.2f} ms")
            if res["uncertain"]:
                st.warning("低置信度或类别接近，建议人工复核")
            st.bar_chart(res["probs"])
    else:
        sample_every_n = st.slider("视频抽帧间隔（每 N 帧取 1 帧）", min_value=5, max_value=60, value=20, step=5)
        max_frames = st.slider("最多分析帧数", min_value=10, max_value=240, value=60, step=10)
        uploaded_video = st.file_uploader("上传垃圾视频", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video is not None:
            video_bytes = uploaded_video.read()
            st.video(video_bytes)

            with st.spinner("正在抽帧并识别..."):
                frames = extract_video_frames(video_bytes, sample_every_n=sample_every_n, max_frames=max_frames)

                predictions = []
                for frame in frames:
                    tensor = preprocess(frame, img_size=300)
                    predictions.append(
                        predict(
                            selected_model,
                            tensor,
                            meta["class_names"],
                            confidence_threshold=conf_th,
                            margin_threshold=margin_th,
                        )
                    )

            summary = summarize_video_predictions(predictions)
            st.subheader("视频识别结果")
            st.write(f"最终投票类别: **{summary['final_label']}**")
            st.write(f"原始多数Top1: {summary['raw_majority_label']}")
            st.write(f"平均置信度: {summary['avg_confidence']:.4f}")
            st.write(f"不确定帧占比: {summary['uncertain_ratio']:.2%}")
            st.write(f"分析帧数: {summary['frame_count']}")

            if predictions:
                frame_label_counts = Counter([p["class_name"] for p in predictions])
                st.bar_chart(dict(frame_label_counts))
                if summary["uncertain_ratio"] >= 0.5:
                    st.warning("视频中超过一半帧为低置信度，建议人工复核")
            else:
                st.error("未从视频中解析到有效帧，请更换视频或减小抽帧间隔")

    render_report_if_exists(args.checkpoint)


if __name__ == "__main__":
    main()
