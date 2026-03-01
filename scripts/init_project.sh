#!/bin/bash
# EcoSort 项目初始化脚本

set -e

echo "=================================="
echo "EcoSort 项目初始化"
echo "=================================="
echo ""

# 检查 Conda 环境
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 Conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 创建环境
echo "1. 创建 Conda 环境..."
conda env create -f environment.yml || {
    echo "环境已存在，更新中..."
    conda env update -f environment.yml
}

# 激活环境
echo "2. 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecosort

# 安装 Python 依赖
echo "3. 安装 Python 依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "4. 创建项目目录..."
mkdir -p data/{raw,proc}
mkdir -p checkpoints
mkdir -p logs

# 初始化 DVC
echo "5. 初始化 DVC..."
if [ ! -d .dvc ]; then
    dvc init
fi

# 初始化 Git (如果需要)
if [ ! -d .git ]; then
    echo "6. 初始化 Git 仓库..."
    git init
    git add .
    git commit -m "feat: initial EcoSort project"
fi

# 生成示例配置
echo "7. 生成示例配置..."
cat > configs/custom_config.yaml << EOF
# 自定义配置模板
model:
  type: "resnet"
  backbone: "resnet50"
  num_classes: 4
  pretrained: true
  dropout: 0.3

data:
  root_dir: "data/raw"
  img_size: 256
  batch_size: 32
  num_workers: 4

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adamw"
  use_amp: true
  early_stopping_patience: 10
EOF

# 创建 .gitignore (如果不存在)
if [ ! -f .gitignore ]; then
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Data & Models
data/raw/
data/proc/
checkpoints/
*.pth
*.pt

# Logs
logs/
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
environment.yml.bak
EOF
fi

echo ""
echo "=================================="
echo "初始化完成!"
echo "=================================="
echo ""
echo "下一步:"
echo "1. 准备数据集到 data/raw/ 目录"
echo "2. 运行训练: python experiments/train_baseline.py --config configs/baseline_resnet50.yaml"
echo "3. 启动服务: python backend/app.py --model-path checkpoints/best_model.pth"
echo ""
echo "环境激活命令: conda activate ecosort"
echo ""
