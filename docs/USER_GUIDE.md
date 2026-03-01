# EcoSort 使用指南

## 目录
1. [安装指南](#安装指南)
2. [快速开始](#快速开始)
3. [API 使用](#api-使用)
4. [Android 应用](#android-应用)
5. [常见问题](#常见问题)

---

## 安装指南

### 环境要求

- Python 3.8+
- CUDA 11.8 (可选, 用于 GPU 加速)
- 8GB+ RAM
- 20GB+ 磁盘空间

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/yourusername/ecosort.git
cd ecosort
```

#### 2. 创建 Conda 环境
```bash
conda env create -f environment.yml
conda activate ecosort
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 验证安装
```bash
python -c "import torch; print(torch.__version__)"
python -c "import flask; print('Flask installed')"
```

---

## 快速开始

### 准备数据集

#### 1. 数据集结构
```
data/raw/
├── recyclable/          # 可回收物
│   ├── plastic_001.jpg
│   ├── paper_002.jpg
│   └── ...
├── hazardous/           # 有害垃圾
│   ├── battery_001.jpg
│   └── ...
├── kitchen/             # 厨余垃圾
│   ├── apple_001.jpg
│   └── ...
└── other/               # 其他垃圾
    ├── ceramic_001.jpg
    └── ...
```

#### 2. 数据集来源

推荐数据集:
- [TrashNet](https://github.com/garythung/trashnet)
- [Kaggle Waste Classification](https://www.kaggle.com/datasets)
- 自行收集网络图片

#### 3. 数据清洗
```bash
# 移除损坏的图像
python scripts/clean_data.py --data-dir data/raw

# 数据集统计
python scripts/analyze_dataset.py --data-dir data/raw
```

### 训练模型

#### 1. 训练 Baseline
```bash
python experiments/train_baseline.py \
    --config configs/baseline_resnet50.yaml \
    --data-root data/raw \
    --exp-name my_first_model
```

#### 2. 训练 EfficientNet
```bash
python experiments/train_baseline.py \
    --config configs/efficientnet_b3.yaml \
    --data-root data/raw
```

#### 3. 超参数优化
```bash
python experiments/hpo_optuna.py \
    --config configs/baseline_resnet50.yaml \
    --n-trials 50
```

### 评估模型

#### 1. 运行评估
```bash
python experiments/evaluate.py \
    --checkpoint checkpoints/my_first_model/best_model.pth \
    --data-root data/raw \
    --output-dir checkpoints/evaluation
```

#### 2. 查看结果
- `confusion_matrix.png` - 混淆矩阵
- `per_class_accuracy.png` - 每类准确率
- `evaluation_report.json` - 详细报告

---

## API 使用

### 启动服务

#### 本地启动
```bash
python backend/app.py \
    --model-path checkpoints/best_model.pth \
    --model-type resnet \
    --port 5000
```

#### 生产部署
```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

### API 端点

#### 1. 健康检查
```bash
GET /health
```

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### 2. 单张图像分类
```bash
POST /predict
Content-Type: application/json
```

**请求:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "format": "base64"
}
```

**响应:**
```json
{
  "class_name": "recyclable",
  "class_id": 0,
  "confidence": 0.9234,
  "probabilities": {
    "recyclable": 0.9234,
    "hazardous": 0.0321,
    "kitchen": 0.0289,
    "other": 0.0156
  }
}
```

#### 3. 批量分类
```bash
POST /batch_predict
```

**请求:**
```json
{
  "images": [
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,..."
  ]
}
```

**响应:**
```json
{
  "predictions": [
    {"class_name": "recyclable", "confidence": 0.92},
    {"class_name": "kitchen", "confidence": 0.88}
  ]
}
```

### Python 客户端示例

```python
import requests
import base64
from PIL import Image
import io

def classify_image(image_path, api_url="http://localhost:5000/predict"):
    # 读取图像
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Base64 编码
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    # 发送请求
    response = requests.post(api_url, json={
        "image": f"data:image/jpeg;base64,{base64_string}",
        "format": "base64"
    })

    return response.json()

# 使用
result = classify_image("test_image.jpg")
print(f"类别: {result['class_name']}")
print(f"置信度: {result['confidence']:.2%}")
```

---

## Android 应用

### 安装应用

#### 方法 1: 从 APK 安装
1. 下载 `app-release.apk`
2. 在手机上启用"未知来源"安装
3. 打开 APK 文件安装

#### 方法 2: Android Studio
1. 克隆项目
2. 用 Android Studio 打开 `mobile/` 目录
3. 点击 Run 按钮

### 配置服务器

#### 修改 API 地址
编辑 `mobile/app/src/main/java/com/example/ecosort/ApiClient.java`:

```java
private static final String BASE_URL = "http://YOUR_SERVER_IP:5000";
```

#### 局域网配置
1. 确保手机和服务器在同一 WiFi
2. 查找服务器 IP:
   ```bash
   hostname -I
   ```
3. 在 Android 设置中更新 BASE_URL

### 使用应用

#### 拍照识别
1. 点击"拍照"按钮
2. 授予相机权限
3. 对准垃圾物品拍照
4. 查看识别结果

#### 图库选择
1. 点击"图库"按钮
2. 选择已有照片
3. 查看识别结果

#### 查看详细结果
- 类别名称 (中文)
- 置信度百分比
- 各类别概率分布

---

## 常见问题

### 训练相关

**Q: 显存不足错误 (CUDA out of memory)**
```yaml
# 解决方案: 减小 batch_size
data:
  batch_size: 16  # 从 32 减小到 16

# 或使用梯度累积
training:
  gradient_accumulation_steps: 2
```

**Q: 训练速度慢**
```yaml
# 解决方案: 启用混合精度
training:
  use_amp: true

# 增加数据加载线程
data:
  num_workers: 8
```

**Q: 准确率不理想**
```yaml
# 解决方案: 更强的数据增强
augmentation:
  horizontal_flip_prob: 0.5
  rotation_degrees: 30
  color_jitter:
    brightness: 0.5
    contrast: 0.5
```

### 部署相关

**Q: Android 无法连接服务器**
- 检查 BASE_URL 配置
- 确认服务器正在运行
- 测试连通性:
  ```bash
  curl http://YOUR_IP:5000/health
  ```

**Q: API 响应慢**
```bash
# 使用 Gunicorn 多进程
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app

# 或使用量化模型
python backend/app.py --model-path checkpoints/model_quantized.pth
```

**Q: 模型文件过大**
```bash
# 量化模型
python scripts/quantize_model.py \
    --input checkpoints/best_model.pth \
    --output checkpoints/model_quantized.pth
```

### 数据相关

**Q: 数据集不平衡**
```python
# 使用加权损失
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

**Q: 数据集太小**
- 使用数据增强
- 迁移学习 (预训练模型)
- 收集更多数据

---

## 高级用法

### 自定义模型

```python
# 创建自定义模型
from src.models.resnet_classifier import ResNetClassifier

model = ResNetClassifier(
    num_classes=4,
    backbone='resnet101',  # 使用 ResNet-101
    pretrained=True,
    dropout=0.5,
    use_attention=True
)
```

### 自定义数据增强

```python
import albumentations as A

transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
])
```

### 模型集成

```python
# 多模型投票
models = [model1, model2, model3]
predictions = []

for model in models:
    pred = model(image)
    predictions.append(pred)

# 投票
final_pred = torch.mode(torch.stack(predictions), dim=0).values
```

---

## 技术支持

遇到问题?
- 查看 [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)
- 提交 Issue
- 发送邮件至: your.email@example.com

---

**祝您使用愉快! 🚀**
