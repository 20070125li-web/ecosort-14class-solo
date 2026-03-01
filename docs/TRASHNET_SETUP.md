# ✅ TrashNet 6类数据集设置完成

## 📊 数据集信息

**TrashNet 标准数据集（学术基准）**

| 类别 | 图像数 | 占比 | 说明 |
|------|--------|------|------|
| paper | 594 | 23.5% | 纸张 |
| glass | 501 | 19.8% | 玻璃 |
| plastic | 482 | 19.1% | 塑料 |
| metal | 410 | 16.2% | 金属 |
| cardboard | 403 | 15.9% | 纸板 |
| trash | 137 | 5.4% | 其他垃圾 |
| **总计** | **2,527** | **100%** | |

**数据质量：**
- ✅ 标准学术基准（多篇论文使用）
- ✅ 数据量充足（平均每类 421 张）
- ✅ 不平衡比例 4.3:1（可接受范围）
- ✅ 图像清晰，标注准确

---

## 🎯 代码更新

### 已更新的文件

1. **`src/data/dataset.py`**
   - 更新为支持 6 类
   - 类别：cardboard, glass, metal, paper, plastic, trash

2. **`configs/baseline_resnet50.yaml`**
   - `num_classes: 4` → `num_classes: 6`

3. **`configs/efficientnet_b3.yaml`**
   - `num_classes: 4` → `num_classes: 6`

4. **`configs/trashnet_resnet50.yaml`**（新建）
   - 专门针对 TrashNet 优化的配置
   - 包含类别加权损失
   - 针对 6 类的数据增强

5. **`src/train/trainer.py`**
   - 新增类别加权损失支持
   - 自动根据 class_counts 计算权重

6. **`scripts/train_trashnet.sh`**（新建）
   - 一键启动训练脚本

---

## 🚀 开始训练

### 方法 1：使用训练脚本（推荐）

```bash
cd /public/home/zhw/cptac/projects/ecosort
bash scripts/train_trashnet.sh
```

### 方法 2：手动启动

```bash
cd /public/home/zhw/cptac/projects/ecosort

# 激活环境
export PYTHONPATH=/public/home/zhw/cptac/projects/ecosort:$PYTHONPATH
conda activate ecosort

# 使用 ResNet-50 训练
python experiments/train_baseline.py \
    --config configs/trashnet_resnet50.yaml \
    --data-root data/raw \
    --exp-name trashnet_resnet50 \
    --no-wandb

# 或使用 EfficientNet-B3（可能效果更好）
python experiments/train_baseline.py \
    --config configs/efficientnet_b3.yaml \
    --data-root data/raw \
    --exp-name trashnet_efficientnet \
    --no-wandb
```

---

## 📊 预期性能（基于 SOTA 论文）

| 模型 | 准确率 | 训练时间（GPU） | 推荐度 |
|------|--------|-----------------|--------|
| ResNet-50 | ~93% | ~50-100 分钟 | ⭐⭐⭐⭐ |
| EfficientNet-B3 | ~95% | ~60-120 分钟 | ⭐⭐⭐⭐⭐ |
| ResNet-101 | ~94% | ~80-150 分钟 | ⭐⭐⭐⭐ |
| Vision Transformer | ~96% | ~120-200 分钟 | ⭐⭐⭐ |

**推荐配置：**
- 快速训练：ResNet-50（1-2 小时）
- 最佳性能：EfficientNet-B3（2 小时）

---

## 📈 训练监控

### 查看训练日志

```bash
# 实时查看
tail -f logs/training_trashnet_*.log

# 查看最后 50 行
tail -50 logs/training_trashnet_*.log
```

### 监控 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看 GPU 使用详情
nvidia-smi --query-gpu=index,name,temperature,utilization.gpu,memory.used,memory.total --format=csv
```

### 检查生成的文件

```bash
# 查看检查点文件
ls -lht checkpoints/trashnet_resnet50/

# 查看训练摘要
cat checkpoints/trashnet_resnet50/training_summary.json
```

---

## 🔍 类别加权说明

为了处理 trash 类样本较少（137 张）的问题，训练时会使用类别加权：

```
类别权重（自动计算）:
  paper:     1.00  (594 张)
  glass:     1.19  (501 张)
  plastic:   1.23  (482 张)
  metal:     1.45  (410 张)
  cardboard: 1.47  (403 张)
  trash:     4.34  (137 张) ← 加权最高
```

这确保模型不会忽略少数类（trash）。

---

## 📁 输出文件

训练完成后，会在 `checkpoints/trashnet_resnet50/` 生成：

```
checkpoints/trashnet_resnet50/
├── best_model.pth              # 最佳模型（完整状态）
├── checkpoint_epoch_XX.pth     # 定期检查点
├── training_summary.json       # 训练摘要
├── loss_curve.png             # 损失曲线
├── confusion_matrix.png       # 混淆矩阵
└── class_accuracy.png         # 各类别准确率
```

---

## 🎓 真实场景识别

TrashNet 6 类非常适合真实场景：

### 优点
1. **细粒度分类**：区分不同材质（纸板、玻璃、金属等）
2. **实用性强**：直接对应实际垃圾分类需求
3. **数据真实**：真实拍摄，包含多种角度和光照
4. **学术认可**：多篇论文使用，结果可对比

### 识别能力
训练完成后，模型可以识别：
- ✅ 纸板箱（cardboard）
- ✅ 玻璃瓶/杯（glass）
- ✅ 金属罐（metal）
- ✅ 纸张（paper）
- ✅ 塑料瓶/袋（plastic）
- ✅ 其他垃圾（trash）

### 部署到真实场景
```python
# 加载模型进行预测
from torchvision import transforms
from PIL import Image

# 预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载图像
image = Image.open("your_photo.jpg")
input_tensor = transform(image).unsqueeze(0)

# 预测
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(1).item()

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(f"预测结果: {classes[prediction]}")
```

---

## 💡 下一步

### 训练完成后
1. **评估性能**
   - 查看混淆矩阵
   - 分析各类别准确率
   - 对比 SOTA 基准

2. **模型优化**（可选）
   - 尝试不同模型（EfficientNet-B3）
   - 调整数据增强
   - 使用集成学习

3. **部署测试**
   - 使用真实照片/视频测试
   - 部署到 Flask API
   - 集成到 Android 应用

### 性能优化
如果需要更高准确率：
- 使用 EfficientNet-B3（预期 95%+）
- 增加数据增强
- 尝试集成学习
- 使用更大的模型（ResNet-101, ViT）

---

## 📚 参考资源

### SOTA 论文
- "Classification of TrashNet Dataset Based on Deep Learning Models" (2022)
- "Multi-Class Image Benchmark for Automated Waste Segregation" (2024)

### TrashNet 数据集
- 原始来源: Stanford & Toronto University
- GitHub: https://github.com/garythung/TrashNet
- Kaggle: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

---

## ✅ 检查清单

训练前：
- [x] ✅ 数据已准备好（2,527 张，6 类）
- [x] ✅ 代码已更新为 6 类
- [x] ✅ 配置文件已更新
- [x] ✅ 类别加权已配置

训练中：
- [ ] 监控训练日志
- [ ] 检查 GPU 使用
- [ ] 观察损失和准确率

训练后：
- [ ] 评估性能指标
- [ ] 分析混淆矩阵
- [ ] 测试真实图像
- [ ] 保存最佳模型

---

*准备时间: 2026-02-15*
*数据集: TrashNet 6类*
*状态: ✅ 准备就绪，可以开始训练*
