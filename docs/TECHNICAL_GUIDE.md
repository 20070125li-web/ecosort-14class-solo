# EcoSort 技术指南

## 目录
1. [系统架构](#系统架构)
2. [模型设计](#模型设计)
3. [训练流程](#训练流程)
4. [部署方案](#部署方案)
5. [优化策略](#优化策略)

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        EcoSort System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │   Android   │         │  Web Client │                   │
│  │    App      │         │  (Future)   │                   │
│  └──────┬──────┘         └──────┬──────┘                   │
│         │                        │                          │
│         └────────────┬───────────┘                          │
│                      │                                      │
│               ┌──────▼──────┐                               │
│               │  Flask API  │                               │
│               │   Backend   │                               │
│               └──────┬──────┘                               │
│                      │                                      │
│               ┌──────▼──────┐                               │
│               │  PyTorch    │                               │
│               │   Model     │                               │
│               └─────────────┘                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 数据流程

1. **用户交互**
   - Android App 拍照/选择图片
   - 压缩图像 (256x256, JPEG 80%)
   - Base64 编码

2. **API 请求**
   - POST /predict
   - 传输 Base64 图像数据

3. **模型推理**
   - 图像预处理 (Resize, Normalize)
   - 模型前向传播
   - Softmax 概率计算

4. **结果返回**
   - 类别名称 + ID
   - 置信度
   - 所有类别概率

---

## 模型设计

### ResNet-50 Baseline

**架构:**
```
Input (3, 256, 256)
    ↓
Conv1 (7x7, 64, stride=2)
    ↓
MaxPool (3x3, stride=2)
    ↓
Residual Blocks (4 stages)
    ↓
Global Average Pooling
    ↓
FC (2048 → 512 → 4)
    ↓
Output (4 classes)
```

**特点:**
- 预训练权重 (ImageNet)
- 迁移学习
- 可选 CBAM 注意力机制

### EfficientNet-B3

**架构:**
```
Input (3, 256, 256)
    ↓
Stem Conv (3x3, 40)
    ↓
MBConv Blocks (7 stages)
    ↓
Head Conv (1x1, 1536)
    ↓
Global Average Pooling
    ↓
FC (1536 → 512 → 4)
    ↓
Output (4 classes)
```

**特点:**
- 复合缩放 (深度/宽度/分辨率)
- MBConv 倒残差结构
- SE 模块
- 参数量更少，性能更高

---

## 训练流程

### 1. 数据准备

**数据增强策略:**
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

**数据集划分:**
- 训练集: 80%
- 验证集: 20%
- 分层采样保证类别平衡

### 2. 训练配置

**优化器:**
- AdamW (β1=0.9, β2=0.999)
- 学习率: 1e-3
- 权重衰减: 1e-4

**学习率调度:**
- CosineAnnealingLR
- T_max = epochs
- η_min = 1e-6

**损失函数:**
- CrossEntropyLoss
- Label Smoothing = 0.1

### 3. 训练技巧

**混合精度训练 (AMP):**
```python
scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**早停:**
- 监控验证准确率
- Patience = 10 epochs
- 保存最佳模型

---

## 部署方案

### 本地部署

**Flask API:**
```bash
python backend/app.py \
    --model-path checkpoints/best_model.pth \
    --host 0.0.0.0 \
    --port 5000
```

### 云端部署

**Docker 容器化:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "backend.app:app"]
```

**Docker Compose:**
```yaml
version: '3'
services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./checkpoints:/app/checkpoints
```

### 移动端部署

**APK 构建:**
```bash
cd mobile
./gradlew assembleDebug
```

**TFLite 转换:**
```python
import torch

# 导出 ONNX
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, "model.onnx")

# 转换 TFLite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_onnx_model("model.onnx")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 优化策略

### 1. 模型量化

**动态量化:**
- 减少模型大小 ~75%
- 加速推理 ~2-3x
- 精度损失 <1%

**静态量化:**
- 需要校准数据集
- 更好的性能
- 适合部署

### 2. 模型剪枝

**结构化剪枝:**
- 移除整个卷积核
- 硬件友好
- 加速明显

**非结构化剪枝:**
- 移除单个权重
- 需要特殊硬件支持
- 压缩率高

### 3. 知识蒸馏

**Teacher-Student:**
```python
# Teacher: ResNet-50
# Student: ResNet-18

loss = α * KL(student_logits, teacher_logits) +
       (1-α) * CE(student_logits, labels)
```

### 4. 推理优化

**TensorRT:**
- 图优化
- 层融合
- 精度校准

**ONNX Runtime:**
- 跨平台
- 高性能
- 易于部署

---

## 性能基准

### 推理速度 (ms)

| 平台 | FP32 | INT8 | Speedup |
|------|------|------|---------|
| GPU (V100) | 15 | 8 | 1.9x |
| CPU (Xeon) | 120 | 45 | 2.7x |
| Mobile (Snapdragon) | 350 | 150 | 2.3x |

### 模型大小 (MB)

| 模型 | FP32 | INT8 | Reduction |
|------|------|------|-----------|
| ResNet-50 | 102 | 28 | 72.5% |
| EfficientNet-B3 | 48 | 14 | 70.8% |

---

## 常见问题

**Q: 如何提高准确率?**
1. 增加数据量
2. 使用更强的数据增强
3. 超参数优化
4. 集成学习

**Q: 如何加速训练?**
1. 使用混合精度 (AMP)
2. 增大 batch size
3. 分布式训练 (DDP)
4. Gradient checkpointing

**Q: 如何减少显存占用?**
1. 减小 batch size
2. 使用梯度累积
3. 减小图像分辨率
4. 模型并行

---

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
