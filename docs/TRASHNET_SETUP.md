# ✅ TrashNet 6-Class Dataset Setup Complete

## 📊 Dataset Information

**TrashNet Standard Dataset (Academic Baseline)**

| Class | # Images | Ratio | Description |
|------|----------|-------|-------------|
| paper | 594 | 23.5% | Paper |
| glass | 501 | 19.8% | Glass |
| plastic | 482 | 19.1% | Plastic |
| metal | 410 | 16.2% | Metal |
| cardboard | 403 | 15.9% | Cardboard |
| trash | 137 | 5.4% | Other waste |
| **Total** | **2,527** | **100%** | |

**Data quality:**
- ✅ Standard academic benchmark (used in multiple papers)
- ✅ Sufficient dataset volume (421 images per class on average)
- ✅ Imbalance ratio 4.3:1 (acceptable range)
- ✅ Clear images and reliable labels

---

## 🎯 Code Updates

### Updated Files

1. **`src/data/dataset.py`**
   - Updated to support 6 classes
   - Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`

2. **`configs/baseline_resnet50.yaml`**
   - `num_classes: 4` → `num_classes: 6`

3. **`configs/efficientnet_b3.yaml`**
   - `num_classes: 4` → `num_classes: 6`

4. **`configs/trashnet_resnet50.yaml`** (new)
   - Configuration optimized specifically for TrashNet
   - Includes class-weighted loss
   - Includes 6-class-focused augmentation

5. **`src/train/trainer.py`**
   - Added class-weighted loss support
   - Automatically computes weights from `class_counts`

6. **`scripts/train_trashnet.sh`** (new)
   - One-click training script

---

## 🚀 Start Training

### Method 1: Use Training Script (Recommended)

```bash
cd /public/home/zhw/cptac/projects/ecosort
bash scripts/train_trashnet.sh
```

### Method 2: Run Manually

```bash
cd /public/home/zhw/cptac/projects/ecosort

# Activate environment
export PYTHONPATH=/public/home/zhw/cptac/projects/ecosort:$PYTHONPATH
conda activate ecosort

# Train with ResNet-50
python experiments/train_baseline.py \
    --config configs/trashnet_resnet50.yaml \
    --data-root data/raw \
    --exp-name trashnet_resnet50 \
    --no-wandb

# Or train with EfficientNet-B3 (may perform better)
python experiments/train_baseline.py \
    --config configs/efficientnet_b3.yaml \
    --data-root data/raw \
    --exp-name trashnet_efficientnet \
    --no-wandb
```

---

## 📊 Expected Performance (Based on SOTA Papers)

| Model | Accuracy | Training Time (GPU) | Recommendation |
|------|----------|----------------------|----------------|
| ResNet-50 | ~93% | ~50–100 min | ⭐⭐⭐⭐ |
| EfficientNet-B3 | ~95% | ~60–120 min | ⭐⭐⭐⭐⭐ |
| ResNet-101 | ~94% | ~80–150 min | ⭐⭐⭐⭐ |
| Vision Transformer | ~96% | ~120–200 min | ⭐⭐⭐ |

**Suggested setup:**
- Fast training: ResNet-50 (1–2 hours)
- Best performance: EfficientNet-B3 (~2 hours)

---

## 📈 Training Monitoring

### View training logs

```bash
# Real-time monitoring
tail -f logs/training_trashnet_*.log

# Last 50 lines
tail -50 logs/training_trashnet_*.log
```

### Monitor GPU usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU usage
nvidia-smi --query-gpu=index,name,temperature,utilization.gpu,memory.used,memory.total --format=csv
```

### Check generated files

```bash
# List checkpoint files
ls -lht checkpoints/trashnet_resnet50/

# View training summary
cat checkpoints/trashnet_resnet50/training_summary.json
```

---

## 🔍 Class Weighting Notes

To address the low sample count of the `trash` class (137 images), class weighting is used during training:

```text
Class weights (auto-computed):
  paper:     1.00  (594 images)
  glass:     1.19  (501 images)
  plastic:   1.23  (482 images)
  metal:     1.45  (410 images)
  cardboard: 1.47  (403 images)
  trash:     4.34  (137 images)  <-- highest weight
```

This ensures the model does not ignore minority classes.

---

## 📁 Output Files

After training, the following files are generated in `checkpoints/trashnet_resnet50/`:

```text
checkpoints/trashnet_resnet50/
├── best_model.pth              # Best model (full state)
├── checkpoint_epoch_XX.pth     # Periodic checkpoints
├── training_summary.json       # Training summary
├── loss_curve.png              # Loss curve
├── confusion_matrix.png        # Confusion matrix
└── class_accuracy.png          # Per-class accuracy
```

---

## 🎓 Real-World Recognition Capability

TrashNet 6-class setup is practical for real-world use:

### Advantages
1. **Fine-grained material classes**: separates cardboard, glass, metal, etc.
2. **High practical relevance**: directly aligned with common sorting scenarios.
3. **Realistic imagery**: includes varied viewpoints and lighting.
4. **Academic comparability**: benchmark used in many studies.

### What the model can recognize
- ✅ Cardboard boxes (`cardboard`)
- ✅ Glass bottles/cups (`glass`)
- ✅ Metal cans (`metal`)
- ✅ Paper (`paper`)
- ✅ Plastic bottles/bags (`plastic`)
- ✅ Other waste (`trash`)

### Example deployment inference

```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("your_photo.jpg")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(1).item()

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(f"Prediction: {classes[prediction]}")
```

---

## 💡 Next Steps

### After training
1. **Evaluate performance**
   - Inspect confusion matrix
   - Analyze per-class accuracy
   - Compare against SOTA baselines

2. **Optional model optimization**
   - Try alternative backbones (e.g., EfficientNet-B3)
   - Tune augmentation policy
   - Apply ensembling

3. **Deployment testing**
   - Test on real photos/videos
   - Deploy with Flask API
   - Integrate into Android app

### Performance upgrade options
- Use EfficientNet-B3 (target 95%+)
- Increase augmentation diversity
- Try model ensembling
- Use larger backbones (ResNet-101, ViT)

---

## 📚 Reference Resources

### SOTA papers
- *Classification of TrashNet Dataset Based on Deep Learning Models* (2022)
- *Multi-Class Image Benchmark for Automated Waste Segregation* (2024)

### TrashNet dataset
- Original source: Stanford & Toronto University
- GitHub: https://github.com/garythung/TrashNet
- Kaggle: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

---

## ✅ Checklist

Before training:
- [x] ✅ Data is ready (2,527 images, 6 classes)
- [x] ✅ Code updated for 6-class support
- [x] ✅ Config files updated
- [x] ✅ Class weighting configured

During training:
- [ ] Monitor training logs
- [ ] Monitor GPU usage
- [ ] Track loss and accuracy trends

After training:
- [ ] Evaluate final metrics
- [ ] Analyze confusion matrix
- [ ] Test on real images
- [ ] Save the best model

---

*Prepared on: 2026-02-15*  
*Dataset: TrashNet 6 classes*  
*Status: ✅ Setup complete and ready for training*
