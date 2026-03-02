"""
EcoSort Backend API
Flask REST API Server (Local Primary + Manual AI 42-Class Deep Review)
"""
import base64
import os
import urllib.request
import json
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml

# Use the latest SDK import method
try:
    from google import genai
except ImportError:
    genai = None

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.data.letterbox import ResizeLongestSideWithPadding

app = Flask(__name__, static_folder='static')

def _parse_cors_origins() -> List[str]:
    raw = os.getenv('ECOSORT_CORS_ORIGINS', '*').strip()
    if not raw or raw == '*':
        return ['*']
    return [origin.strip() for origin in raw.split(',') if origin.strip()]

cors_origins = _parse_cors_origins()
CORS(
    app,
    resources={r"/*": {"origins": cors_origins}},
    methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization'],
)

# Global variables
model = None
device = None
class_names = [
    'brick_ceramic',
    'e_waste',
    'foam',
    'glass',
    'haz_battery',
    'haz_device',
    'haz_medicine',
    'hygiene_contaminated',
    'metal',
    'organic_food',
    'paper_family',
    'plastic',
    'small_hard_items',
    'textile',
]
fine_to_coarse = {
    'paper_family': 'recyclable',
    'plastic': 'recyclable',
    'glass': 'recyclable',
    'metal': 'recyclable',
    'textile': 'recyclable',
    'haz_battery': 'hazardous',
    'haz_device': 'hazardous',
    'haz_medicine': 'hazardous',
    'e_waste': 'hazardous',
    'organic_food': 'kitchen',
    'hygiene_contaminated': 'other',
    'small_hard_items': 'other',
    'foam': 'other',
    'brick_ceramic': 'other',
}
transform = None
confidence_threshold = 0.7
inference_image_size = int(os.getenv('ECOSORT_INFER_IMAGE_SIZE', '224'))
use_cuda_autocast = os.getenv('ECOSORT_USE_CUDA_AUTOCAST', '1').strip() != '0'

# ===============================
# Gemini Production Config
# ===============================
GEMINI_API_KEY = os.getenv("ECOSORT_GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv(
    "ECOSORT_GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

print("\n========== GEMINI DEBUG ==========")
print("SDK Installed:", genai is not None)
print("API KEY Loaded:", bool(GEMINI_API_KEY))
print("API KEY Length:", len(GEMINI_API_KEY) if GEMINI_API_KEY else 0)
print("Model Name:", GEMINI_MODEL_NAME)
gemini_client = None

if GEMINI_API_KEY and genai is not None:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Client Initialized: SUCCESS")
        
        # Verify if the current model exists
        available_models = [m.name for m in gemini_client.models.list()]
        if f"models/{GEMINI_MODEL_NAME}" in available_models:
            print("Model Verified:", GEMINI_MODEL_NAME)
        else:
            print("WARNING: Model not found in available models list")
            
    except Exception as e:
        print("Client Initialization Failed:", str(e))
else:
    print("Gemini not configured properly.")
print("===================================\n")

def _download_model_if_needed(model_path: str) -> str:
    # Set default filename to v2.0
    if not model_path:  
        model_path = 'model_weight_v2.0.pth'
        
    if os.path.exists(model_path):
        return model_path
        
    # Correct release asset URL (download endpoint)
    default_url = 'https://github.com/20070125li-web/ecosort-14class/releases/download/v2.0/model_weight_v2.0.pth'

    candidate_urls = [
        os.getenv('HF_MODEL_URL', '').strip(),
        os.getenv('ECOSORT_MODEL_URL', '').strip(),
        default_url,
    ]

    downloaded = False
    for model_url in candidate_urls:
        if not model_url or not model_url.startswith('http'):
            continue

        print(f"Model not found locally. Downloading from {model_url} ...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Successfully downloaded model to {model_path}")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {model_url}: {e}")

    if not downloaded and not os.path.exists(model_path):
        print(f"Warning: could not download model. Expected local file at: {model_path}")
        
    return model_path

# ==========================================
# Core VLM Parsing Logic (Supports 42 Classes)
# ==========================================
def _call_vlm_primary(image: Image.Image) -> Dict[str, Any]:
    """Manual trigger: Call cloud VLM, output 42 classes and probability ranking"""
    if not GEMINI_API_KEY or gemini_client is None:
        raise ValueError("AI review failed: VLM API key not configured or SDK missing")

    try:
        prompt = """
        You are a top expert in waste classification standards.
        Please identify the main object in the image and strictly select the most accurate classification from the following 42 labels:
        cardboard, glass, metal, paper, plastic, clothes, shoes, bag, e_waste, plastic_toy, 
        battery, bulb, chemical, paint, medicine, thermometer, cosmetic, 
        fruit_peel, vegetable, leftover, tea, egg_shell, bone, flower, cake, 
        cigarette, tissue, chopsticks, toothpick, cup, mask, ceramic, brick, diaper, 
        wet_wipe, plastic_bag, pen, comb, towel, lighter, foam, trash.
        Also, provide its coarse category (recyclable, hazardous, kitchen, other).

        Based on the image content, deduce your top 3 most confident classifications and their respective confidence probabilities (ranging from 0 to 1, summing to no more than 1, with the first place having the highest probability).

        You must ONLY output valid JSON format strictly as follows:
        {
            "class_name": "label_name (must be one of the 42 above)",
            "coarse_category": "recyclable/hazardous/kitchen/other",
            "confidence": 0.95,
            "top_predictions": [
                {"class_name": "1st_place_label", "confidence": 0.95},
                {"class_name": "2nd_place_label", "confidence": 0.04},
                {"class_name": "3rd_place_label", "confidence": 0.01}
            ]
        }
        """
        
        # Call VLM using the new SDK syntax
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[prompt, image]
        )
        
        text = (response.text or '').strip()
        text = re.sub(r'^```(json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        data = json.loads(text)

        top_preds = []
        for pred in data.get("top_predictions", []):
            label = pred.get("class_name", "trash")
            top_preds.append({
                "class_name": label,
                "daily_label": label,  
                "confidence": float(pred.get("confidence", 0.0)),
                "coarse_category": data.get("coarse_category", "other")
            })

        main_label = data.get('class_name', 'trash')
        
        return {
            'class_name': main_label,
            'raw_top1_class': main_label,
            'daily_label': main_label,
            'coarse_category': data.get('coarse_category', 'other'),
            'confidence': float(data.get('confidence', 0.9)),
            'top_predictions': top_preds,
            'vlm_fallback_used': True,
            'advice': f'AI depth review completed, matched with 42-class standard subdivisions.',
            'uncertain': False
        }

    except Exception as exc:
        print(f"VLM Primary Error: {exc}")
        return {
            'class_name': 'trash',
            'raw_top1_class': 'trash',
            'daily_label': 'trash',
            'coarse_category': 'other',
            'confidence': 0.0,
            'top_predictions': [],
            'vlm_fallback_used': True,
            'vlm_raw_output': str(exc),
            'advice': 'VLM review request failed, please try again.',
            'uncertain': True
        }

def _decode_base64_image(image_data: str) -> Image.Image:
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

# ==========================================
# Local 14-Class Core Logic
# ==========================================
def _infer_image(image: Image.Image) -> Dict[str, Any]:
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        if device is not None and str(device).startswith('cuda') and use_cuda_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_tensor)
        else:
            outputs = model(input_tensor)
            
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_values, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
        top1_idx = int(top_indices[0].item())

    top1_name = class_names[top1_idx]
    top1_conf = float(top_values[0].item())
    top1_coarse = fine_to_coarse.get(top1_name, 'other')

    top_predictions = [
        {
            'class_name': class_names[int(top_indices[i].item())],
            'daily_label': class_names[int(top_indices[i].item())],
            'confidence': float(top_values[i].item()),
            'coarse_category': fine_to_coarse.get(class_names[int(top_indices[i].item())], 'other')
        }
        for i in range(len(top_indices))
    ]

    uncertain = top1_conf < confidence_threshold
    return {
        'class_name': top1_name,
        'raw_top1_class': top1_name,
        'daily_label': top1_name,
        'coarse_category': top1_coarse,
        'confidence': top1_conf,
        'top_predictions': top_predictions,
        'uncertain': uncertain,
        'vlm_fallback_used': False,
        'advice': 'Low confidence from local model. If you disagree, please click [AI Review]' if uncertain else ''
    }

def _detect_model_type_from_state_dict(state_dict: dict) -> Tuple[str, str]:
    keys = list(state_dict.keys())
    if any(k.startswith("backbone._conv_stem") or k.startswith("backbone._blocks") for k in keys):
        return "efficientnet", "efficientnet-b3"
    if any(k.startswith("features.") or k.startswith("layer") for k in keys):
        return "resnet", "resnet50"
    return "efficientnet", "efficientnet-b3"

def load_model(model_path: str, model_type: str = None, num_classes: int = None):
    global model, device, transform, class_names
    
    torch_threads = max(1, int(os.getenv('ECOSORT_TORCH_THREADS', '1')))
    torch.set_num_threads(torch_threads)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint_raw = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Local model failed to load: {e}")
        return "Local model failed to load."

    state_dict = checkpoint_raw.get('model_state_dict', checkpoint_raw)

    if model_type is None:
        model_type, backbone = _detect_model_type_from_state_dict(state_dict)
    else:
        backbone = 'resnet50' if model_type == 'resnet' else 'efficientnet-b3'
        
    config = checkpoint_raw.get('config', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    # Trust checkpoint config first when available.
    config_model_type = model_cfg.get('type')
    config_backbone = model_cfg.get('backbone')
    if config_model_type in {'resnet', 'efficientnet'}:
        model_type = config_model_type
    if isinstance(config_backbone, str) and config_backbone:
        backbone = config_backbone
    
    config_names = data_cfg.get('class_names')
    if isinstance(config_names, list) and len(config_names) > 0:
        class_names = config_names
        num_classes = len(class_names)
    elif num_classes is None:
        class_counts = data_cfg.get('class_counts', [])
        num_classes = len(class_counts) if class_counts else 14
        
    if num_classes is None or num_classes == 4:
        for key in ['backbone._fc.3.weight', 'classifier.3.weight']:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                break
    if num_classes is None:
        num_classes = 14
                
    print(f"Creating model: {model_type} / {backbone} / num_classes={num_classes}")
    
    if model_type == 'resnet':
        model = create_resnet_model(backbone=backbone, num_classes=num_classes, pretrained=False, dropout=0.3, use_attention=False)
    elif model_type == 'efficientnet':
        model = create_efficientnet_model(backbone=backbone, num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        stem_shape = None
        if 'backbone._conv_stem.weight' in state_dict:
            stem_shape = tuple(state_dict['backbone._conv_stem.weight'].shape)
        raise RuntimeError(
            "Checkpoint-model mismatch while loading state_dict. "
            f"model_type={model_type}, backbone={backbone}, num_classes={num_classes}, "
            f"checkpoint_stem_shape={stem_shape}. Original error: {e}"
        )
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        ResizeLongestSideWithPadding(inference_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return f"Loaded local {model_type} with {num_classes} classes"

# ==========================================
# API Routes
# ==========================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vlm_configured': bool(GEMINI_API_KEY and gemini_client is not None),
        'vlm_model': GEMINI_MODEL_NAME,
        'mode': 'Local Default + AI Verify On Demand'
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    return jsonify({
        'coarse_labels': ['recyclable', 'hazardous', 'kitchen', 'other'],
        'class_names': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        image = _decode_base64_image(data['image'])
        ai_verify = data.get('ai_verify', False) or data.get('force_vlm', False)
        
        if ai_verify:
            if not GEMINI_API_KEY:
                return jsonify({'error': 'VLM API Key is not configured.'}), 500
            response = _call_vlm_primary(image)
        else:
            if model is None:
                return jsonify({'error': 'Local model is not ready.'}), 500
            response = _infer_image(image)
            
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_vlm', methods=['POST'])
def predict_vlm():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        image = _decode_base64_image(data['image'])
        
        if not GEMINI_API_KEY:
            return jsonify({'error': 'VLM API Key is not configured.'}), 500
            
        response = _call_vlm_primary(image)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

EDGE_CASE_LOG_FILE = "edge_cases_dataset.csv"

@app.route('/report_edge_case', methods=['POST'])
def report_edge_case():
    try:
        data = request.get_json()
        image_base64 = data.get("image", "")
        expected_label = data.get("expected_label", "unknown")
        model_prediction = data.get("model_prediction", "unknown")
        
        file_exists = os.path.isfile(EDGE_CASE_LOG_FILE)
        with open(EDGE_CASE_LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Expected Label", "Model Guessed", "Image Header"])
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            img_preview = image_base64[:50] + "..." if image_base64 else "None"
            writer.writerow([timestamp, expected_label, model_prediction, img_preview])
        
        print(f"[EDGE CASE COLLECTED] Expected: {expected_label} | Model Guessed: {model_prediction}")
        return jsonify({"status": "success", "message": "Edge case recorded successfully."})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    import argparse
    
    # Updated the default model path to search for v2.0
    _mod_env = (
        os.getenv('MODEL_PATH', '').strip()
        or os.getenv('ECOSORT_MODEL_PATH', '').strip()
        or 'model_weight_v2.0.pth'
    )
    
    parser = argparse.ArgumentParser(description='EcoSort Backend API')
    parser.add_argument('--model-path', type=str, default=_mod_env)
    parser.add_argument('--model-type', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 7860)))
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    final_model_path = _download_model_if_needed(args.model_path)
    
    print(f"Loading local model from {final_model_path}...")
    msg = load_model(final_model_path, args.model_type, args.num_classes)
    print(msg)
    
    print(f"\nStarting server on {args.host}:{args.port}")
    if GEMINI_API_KEY:
        print(f"VLM Verification Ready: Model {GEMINI_MODEL_NAME} is armed for deep check.")
        
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
