"""
EcoSort Backend API
Flask REST API 服务器
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.train.trainer import load_checkpoint


app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
model = None
device = None
class_names = ['recyclable', 'hazardous', 'kitchen', 'other']
transform = None
confidence_threshold = 0.7
margin_threshold = 0.15


def load_model(model_path: str, model_type: str = 'resnet'):
    """加载模型"""
    global model, device, transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    if model_type == 'resnet':
        model = create_resnet_model(
            backbone='resnet50',
            num_classes=4,
            pretrained=False
        )
    elif model_type == 'efficientnet':
        model = create_efficientnet_model(
            backbone='efficientnet-b3',
            num_classes=4,
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载权重
    checkpoint = load_checkpoint(model_path, model)
    model = model.to(device)
    model.eval()

    # 创建预处理变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f"Model loaded from {model_path}")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """单张图像分类

    Request:
        - image: base64 编码的图像字符串
        - format: 'base64' (默认)

    Returns:
        - class_name: 类别名称
        - class_id: 类别 ID (0-3)
        - confidence: 置信度
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # 获取请求数据
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # 解码图像
        image_data = data['image']

        # 处理 base64 编码的图像
        if data.get('format') == 'base64':
            # 移除 data:image/xxx;base64, 前缀
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            return jsonify({'error': 'Unsupported format'}), 400

        # 预处理
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_values, top_indices = torch.topk(probabilities, k=min(2, len(class_names)))

        top1_conf = float(top_values[0].item())
        top1_idx = int(top_indices[0].item())
        top1_name = class_names[top1_idx]
        top2_conf = float(top_values[1].item()) if len(top_values) > 1 else 0.0
        margin = top1_conf - top2_conf
        uncertain = top1_conf < confidence_threshold or margin < margin_threshold
        final_name = 'unknown_review_needed' if uncertain else top1_name

        # 构建响应
        response = {
            'class_name': final_name,
            'raw_top1_class': top1_name,
            'class_id': top1_idx,
            'confidence': top1_conf,
            'top2_confidence': top2_conf,
            'margin': margin,
            'uncertain': uncertain,
            'probabilities': {
                class_names[i]: float(probabilities[i].item())
                for i in range(len(class_names))
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量图像分类

    Request:
        - images: base64 编码的图像列表

    Returns:
        - predictions: 预测结果列表
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        if 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400

        images = data['images']
        predictions = []

        for img_data in images:
            try:
                # 解码图像
                if ',' in img_data:
                    img_data = img_data.split(',')[1]

                image_bytes = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # 预处理
                input_tensor = transform(image).unsqueeze(0).to(device)

                # 推理
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    confidence, prediction = torch.max(probabilities, 0)

                predictions.append({
                    'class_name': class_names[prediction.item()],
                    'class_id': int(prediction.item()),
                    'confidence': float(confidence.item())
                })

            except Exception as e:
                predictions.append({'error': str(e)})

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'model_type': type(model).__name__,
        'num_classes': len(class_names),
        'class_names': class_names,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'device': str(device)
    }

    return jsonify(info)


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """重新加载模型

    Request:
        - model_path: 模型路径
        - model_type: 模型类型 (resnet/efficientnet)
    """
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        model_type = data.get('model_type', 'resnet')

        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400

        load_model(model_path, model_type)

        return jsonify({'status': 'success', 'message': 'Model reloaded'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """启动服务器"""
    import argparse

    parser = argparse.ArgumentParser(description='EcoSort Backend API')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='resnet',
                        choices=['resnet', 'efficientnet'],
                        help='Model type')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')

    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    load_model(args.model_path, args.model_type)
    print("Model loaded successfully!")

    # 启动服务器
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Available endpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("  POST /batch_predict")
    print("  GET  /model_info")
    print("  POST /reload_model")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
