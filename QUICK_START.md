# EcoSort Quick Start

## 1. Create Environment
```bash
conda env create -f environment.yml
conda activate ecosort
pip install -r requirements.txt
```

## 2. Prepare Data
Expected structure:
```text
data/proc/<dataset_name>/
├── train/
├── val/
└── test/
```

If you are using DVC-managed artifacts:
```bash
dvc pull
```

## 3. Train
```bash
python experiments/train_baseline.py \
  --config configs/efficientnet_b3.yaml \
  --data-root data/proc
```

Optional V6 configs:
```bash
python experiments/train_baseline.py \
  --config configs/ecosort_v6_15class_effb3_letterbox224.yaml
```

## 4. Evaluate
```bash
python experiments/evaluate.py \
  --checkpoint checkpoints/<exp_name>/best_model.pth \
  --data-root data/proc \
  --model-type efficientnet
```

## 5. Run Services
Backend API:
```bash
python backend/app.py
```

Streamlit demo:
```bash
streamlit run streamlit_demo.py
```

## 6. Handover Sanity Check
- Ensure there is at least one runnable checkpoint path in `checkpoints/`.
- Ensure `configs/dataset_mapping.yaml` is aligned with runtime labels.
- Ensure no large raw data or secrets are staged in Git (`git status`).
