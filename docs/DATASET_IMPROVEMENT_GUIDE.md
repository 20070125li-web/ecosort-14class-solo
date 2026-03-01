# 📊 EcoSort 数据集改进完整指南

## 🎯 概述

本指南提供了完整的改进方案，解决当前 EcoSort 数据集的类别不平衡问题，使模型达到 SOTA 性能。

---

## 📋 当前问题总结

### 数据集现状
```yaml
总图像数: 2,137 张
有效类别: 2/4 (50%)
类别分布:
  recyclable: 2,000 张 (93.6%) ← 多数类
  other:       137 张 (6.4%)  ← 少数类
  hazardous:      0 张 (0%)   ← 缺失
  kitchen:        0 张 (0%)   ← 缺失

不平衡比例: 14.6:1 (严重)
模型性能: Acc=93.4%, F1=0.48 (虚高的准确率)
```

### 根本原因
- **类别严重不平衡**：模型学会总是预测多数类
- **缺失类别数据**：hazardous 和 kitchen 完全没有数据
- **样本不足**：minority class 仅 137 张（远低于 400-500 最小要求）

---

## 🏆 SOTA 数据集标准

### 最佳实践（来自研究）
| 指标 | 最小值 | 推荐值 | 理想值 |
|------|--------|--------|--------|
| 每类样本数 | 400-500 | 800-1000 | 1000+ |
| 类别比例 | 1.5:1 | 1.2:1 | 1:1 |
| 总数据量 | 1,600 | 3,200 | 5,000+ |
| F1-Score | 0.85 | 0.90 | 0.94+ |

### SOTA 模型使用的数据集
1. **TrashNet**: 2,527 张, 6 类, 每类 400-500 张
2. **Waste89**: ~5,000 张, 6-10 类, 每类 500-1000 张
3. **2024 New Benchmark**: ~10,000 张, 10 类

详细分析见: [DATASET_RESEARCH.md](DATASET_RESEARCH.md)

---

## 🚀 改进方案

### 方案对比

| 方案 | 优点 | 缺点 | 预期 F1 | 时间成本 |
|------|------|------|---------|----------|
| **1. 补充完整数据** | 性能最佳 | 需要下载/收集 | 0.92+ | 2-3 小时 |
| **2. 部分补充 + 平衡** | 平衡效果/工作量 | 某些类数据少 | 0.85-0.90 | 1-2 小时 |
| **3. 仅数据平衡** | 快速 | 性能受限 | 0.70-0.75 | 30 分钟 |
| **4. 类别加权** | 无需改数据 | 效果有限 | 0.60-0.70 | 10 分钟 |

---

## 📖 完整工作流程（推荐方案 1）

### 步骤 1: 安装 Kaggle CLI (首次)

```bash
# 1. 安装 Kaggle
pip install kaggle

# 2. 访问 https://www.kaggle.com/settings
#    点击 "Create New API Token" 下载 kaggle.json

# 3. 配置 API Key
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 步骤 2: 下载补充数据

```bash
# 方式 1: 自动化脚本（推荐）
cd /public/home/zhw/cptac/projects/ecosort
bash scripts/collect_and_balance_data.sh

# 方式 2: 手动下载
kaggle datasets download -d asdasdasasdas/garbage-classification -p data/kaggle/
kaggle datasets download -d techsash/waste-classification -p data/kaggle/
kaggle datasets download -d fedherinoraminirez/trashnet -p data/kaggle/

# 解压
mkdir -p data/kaggle/extracted
unzip data/kaggle/*.zip -d data/kaggle/extracted/
```

### 步骤 3: 分析下载的数据

```bash
python scripts/analyze_kaggle_datasets.py
```

**输出示例：**
```
Garbage Classification:
  总图像: 9,871
  类别数: 12
  类别分布:
    cardboard:   938
    glass:       861
    metal:       753
    ...
```

### 步骤 4: 合并和映射到 EcoSort 4 类

```bash
# 先预览（不实际复制）
python scripts/merge_datasets.py \
    --source data/kaggle/extracted/ \
    --target data/raw/ \
    --dry-run

# 确认无误后执行
python scripts/merge_datasets.py \
    --source data/kaggle/extracted/ \
    --target data/raw/
```

**映射规则：**
- recyclable ← cardboard, glass, metal, paper, plastic, 等
- hazardous ← batteries, 等
- kitchen ← biological, organic, 等
- other ← trash, landfill, 等

### 步骤 5: 验证数据集

```bash
python scripts/verify_balance.py --data-root data/raw/
```

**预期输出：**
```
✓ recyclable : 2500 (25.0%)
✓ hazardous  :  800 ( 8.0%)
✓ kitchen    : 1200 (12.0%)
✓ other      : 1000 (10.0%)

不平衡比例: 3.1:1
⚠️  中度不平衡，建议进行轻微调整
```

### 步骤 6: 平衡数据集（可选）

```bash
# 自动平衡
python scripts/balance_dataset.py \
    --data-root data/raw/ \
    --output data/balanced/ \
    --auto

# 或手动指定目标
python scripts/balance_dataset.py \
    --data-root data/raw/ \
    --output data/balanced/ \
    --target 500 \
    --method oversample  # 或 downsample
```

### 步骤 7: 验证最终数据集

```bash
python scripts/verify_balance.py --data-root data/balanced/
```

**预期输出（成功）：**
```
✓ recyclable :  500 (25.0%) ██████████
✓ hazardous  :  500 (25.0%) ██████████
✓ kitchen    :  500 (25.0%) ██████████
✓ other      :  500 (25.0%) ██████████

不平衡比例: 1.0:1
✅ 数据集质量: 优秀

可以开始训练！
```

### 步骤 8: 重新训练模型

```bash
export PYTHONPATH=/public/home/zhw/cptac/projects/ecosort:$PYTHONPATH
conda activate ecosort

python experiments/train_baseline.py \
    --config configs/baseline_resnet50.yaml \
    --data-root data/balanced/ \
    --exp-name balanced_training \
    --epochs 50 \
    --no-wandb
```

---

## 🎯 快速方案（方案 3: 仅数据平衡）

如果暂时无法下载新数据，可以只对现有数据进行平衡：

```bash
# 使用上采样（数据增强）
python scripts/balance_dataset.py \
    --data-root data/raw/ \
    --output data/balanced/ \
    --target 137 \
    --method oversample

# 验证
python scripts/verify_balance.py --data-root data/balanced/

# 训练
python experiments/train_baseline.py \
    --config configs/baseline_resnet50.yaml \
    --data-root data/balanced/ \
    --exp-name balanced_137 \
    --epochs 50
```

**注意：** 此方案性能受限，因为 minority class 样本数仍然很少。

---

## 🛠️ 临时方案（方案 4: 类别加权）

修改训练脚本添加类别权重：

```python
# experiments/train_baseline.py 或 src/train/trainer.py

import torch
import torch.nn as nn

# 根据类别分布计算权重
# recyclable: 2000, hazardous: 0, kitchen: 0, other: 137
class_counts = [2000, 1, 1, 137]  # 避免除以0
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * 4

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**预期效果：**
- F1-Score: 0.48 → 0.60-0.70
- 不改变数据集
- 效果有限，因为 minority class 样本仍不足

---

## 📊 性能预期

### 各方案性能对比

| 方案 | Accuracy | F1-Score | Recall (minority) | 说明 |
|------|----------|----------|-------------------|------|
| 当前 (不平衡) | 93.4% | 0.48 | ~0% | 虚高准确率 |
| 仅类别加权 | 85% | 0.65 | 40-60% | 快速但效果有限 |
| 仅数据平衡 (137/类) | 82% | 0.72 | 60-70% | 受限于样本数 |
| 平衡数据集 (500/类) | 90% | 0.88 | 85% | 推荐配置 |
| 平衡数据集 (1000/类) | 92% | 0.92 | 90% | 理想配置 |
| 迁移学习 (5000+ 预训练) | 94% | 0.94 | 93% | SOTA 水平 |

---

## 📁 脚本说明

### 数据收集和平衡脚本

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `collect_and_balance_data.sh` | 从 Kaggle 下载数据集 | - | data/kaggle/*.zip |
| `analyze_kaggle_datasets.py` | 分析下载的数据集 | data/kaggle/extracted/ | 统计信息 |
| `merge_datasets.py` | 合并并映射到 4 类 | data/kaggle/extracted/ | data/raw/ |
| `balance_dataset.py` | 平衡数据集 | data/raw/ | data/balanced/ |
| `verify_balance.py` | 验证数据集质量 | data/raw/ 或 data/balanced/ | 验证报告 |

---

## 🔧 常见问题

### Q1: Kaggle 下载失败怎么办？
**A:** 可能是网络问题，尝试：
```bash
# 使用代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 或手动下载
# 访问 Kaggle 网站手动下载，然后上传到服务器
```

### Q2: 内存不足怎么办？
**A:** 使用分批处理或下采样：
```bash
# 方法 1: 下采样到较小数量
python scripts/balance_dataset.py \
    --data-root data/raw/ \
    --output data/balanced_small/ \
    --target 300 \
    --method downsample

# 方法 2: 只使用部分数据集
```

### Q3: 没有 hazardous 和 kitchen 数据怎么办？
**A:** 选项：
1. 从 Kaggle 下载包含这些类别的数据集（推荐）
2. 暂时使用 2 类分类（recyclable vs other）
3. 网络爬虫收集相关图像并标注

### Q4: 平衡后训练变慢？
**A:** 可能的原因：
1. 数据量增加（正常）
2. 使用了上采样（图像生成开销）
3. 解决：使用下采样或减少 batch size

### Q5: 如何选择 target 数量？
**A:** 根据现有数据：
- 如果多数类有 2000 张，可下采样到 500-1000
- 如果 minority class 有 100-200 张，上采样到 300-500
- 推荐：500-1000 是性能和计算成本的平衡点

---

## 🎓 最佳实践建议

### 数据收集
1. **优先使用平衡数据集**：从 Kaggle 下载的预分类数据集通常已经平衡
2. **注意类别映射**：确保外部数据集的类别能正确映射到 EcoSort 4 类
3. **数据质量 > 数量**：500 张高质量图像优于 2000 张低质量图像

### 数据平衡
1. **推荐使用上采样 + 数据增强**：保留所有数据，通过增强增加 minority class
2. **下采样作为备选**：仅当数据量充足时使用（避免丢失信息）
3. **验证平衡结果**：始终使用 verify_balance.py 检查

### 训练
1. **使用类别加权损失**：即使平衡后也推荐使用（轻微加权可提升性能）
2. **监控 F1-Score**：不要只看 accuracy
3. **使用早停**：避免过拟合

---

## 📚 参考资源

### 论文
- "Multi-Class Image Benchmark for Automated Waste Segregation" (2024)
- "Classification of TrashNet Dataset Based on Deep Learning Models" (2022)

### 数据集
- Kaggle: Garbage Classification
- Kaggle: Waste Classification Data
- GitHub: ZeroWaste Dataset

### 工具文档
- [PyTorch 数据加载](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [类别不平衡处理](https://pytorch.org/docs/stable/nn.html#crossentropyloss)
- [数据增强](https://pytorch.org/vision/stable/transforms.html)

---

## ✅ 检查清单

### 改进前检查
- [ ] 了解当前数据集问题（类别不平衡、缺失类别）
- [ ] 阅读 DATASET_RESEARCH.md 了解 SOTA 标准
- [ ] 决定使用哪种改进方案

### 数据收集
- [ ] 安装并配置 Kaggle CLI
- [ ] 下载补充数据集
- [ ] 分析下载的数据集质量

### 数据处理
- [ ] 合并数据集到 EcoSort 格式
- [ ] 验证类别映射正确性
- [ ] 检查数据质量（损坏图像、标注错误）

### 数据平衡
- [ ] 选择平衡策略（上采样/下采样）
- [ ] 执行数据平衡
- [ ] 验证平衡结果

### 训练准备
- [ ] 验证最终数据集质量
- [ ] 调整训练配置（learning rate, batch size）
- [ ] 启动训练并监控指标

### 性能评估
- [ ] 对比改进前后的 F1-Score
- [ ] 分析混淆矩阵
- [ ] 记录实验结果

---

## 🎯 总结

### 推荐行动方案

**如果您有 2-3 小时：**
1. ✅ 下载 Kaggle 数据集
2. ✅ 合并和平衡到每类 500-1000 张
3. ✅ 重新训练模型
4. ✅ 预期 F1-Score: 0.88-0.92

**如果您只有 30 分钟：**
1. ✅ 使用现有数据平衡（上采样）
2. ✅ 快速验证
3. ✅ 训练基线模型
4. ✅ 预期 F1-Score: 0.70-0.75

**如果您需要最佳性能：**
1. ✅ 下载多个数据集
2. ✅ 仔细清理和标注
3. ✅ 平衡到每类 1000+ 张
4. ✅ 使用迁移学习（先在大规模数据预训练）
5. ✅ 集成学习（ResNet + EfficientNet）
6. ✅ 预期 F1-Score: 0.94+ (SOTA)

---

*最后更新: 2026-02-15*
*完整文档: docs/DATASET_RESEARCH.md*
*问题反馈: Issues on GitHub*
