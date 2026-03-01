#!/bin/bash
# EcoSort 优化训练启动脚本

set -e

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ecosort

# 切换到项目目录
cd /public/home/zhw/cptac/projects/ecosort

# 设置PYTHONPATH
export PYTHONPATH=/public/home/zhw/cptac/projects/ecosort:$PYTHONPATH

# 启动训练
python experiments/train_baseline.py \
    --config configs/optimized_resnet50.yaml \
    --exp-name crawled42_optimized \
    2>&1 | tee logs/train_optimized_$(date +%Y%m%d_%H%M%S).log
