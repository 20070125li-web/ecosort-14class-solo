#!/usr/bin/env python3
"""
EcoSort 快速启动脚本 - 无需conda环境
使用系统Python直接运行
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

print("="*60)
print("EcoSort 快速启动 - 数据下载与训练")
print("="*60)
print()

# 项目路径
PROJECT_ROOT = Path("/public/home/zhw/cptac/projects/ecosort")
os.chdir(PROJECT_ROOT)

# 步骤1: 下载数据集
print("步骤 1/4: 下载 TrashNet 数据集...")
print("-" * 40)

DATA_DIR = PROJECT_ROOT / "data" / "download"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_FILE = DATA_DIR / "trashnet.zip"
DATA_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

if not ZIP_FILE.exists():
    print(f"正在从 {DATA_URL} 下载...")
    print("文件大小: ~41MB")

    def download_with_progress(url, destination):
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r下载进度: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # 新行

    try:
        download_with_progress(DATA_URL, ZIP_FILE)
        print("✓ 下载完成!")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n请手动下载:")
        print(f"  wget {DATA_URL} -O {ZIP_FILE}")
        sys.exit(1)
else:
    print("✓ 数据集已存在")

# 步骤2: 解压数据
print("\n步骤 2/4: 解压数据集...")
print("-" * 40)

EXTRACTED_DIR = DATA_DIR / "dataset-resized"

if not EXTRACTED_DIR.exists():
    print("正在解压...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("✓ 解压完成!")
else:
    print("✓ 数据已解压")

# 步骤3: 转换数据格式
print("\n步骤 3/4: 转换数据格式...")
print("-" * 40)

import shutil

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "raw"

# 创建目标目录
for class_name in ['recyclable', 'hazardous', 'kitchen', 'other']:
    (TARGET_DIR / class_name).mkdir(parents=True, exist_ok=True)

# TrashNet 类别映射
trashnet_mapping = {
    'cardboard': 'recyclable',
    'glass': 'recyclable',
    'metal': 'recyclable',
    'paper': 'recyclable',
    'plastic': 'recyclable',
    'trash': 'other'
}

# 转换数据
print("正在转换数据格式...")
total_copied = 0

for old_class, new_class in trashnet_mapping.items():
    old_path = EXTRACTED_DIR / old_class
    new_path = TARGET_DIR / new_class

    if old_path.exists():
        # 复制文件 (限制每类100张用于快速测试)
        files = list(old_path.glob('*.jpg')) + list(old_path.glob('*.png'))
        for i, img_file in enumerate(files[:100]):  # 限制100张
            shutil.copy2(img_file, new_path / f"{old_class}_{img_file.name}")
            total_copied += 1

        print(f"  {old_class:12} -> {new_class:12} ({min(len(files), 100)} 张)")

print(f"✓ 转换完成! 总计 {total_copied} 张图像")

# 步骤4: 快速训练
print("\n步骤 4/4: 开始快速训练...")
print("-" * 40)
print("注意: 这将使用系统Python运行训练")
print("如果遇到依赖问题，请等待conda环境创建完成")
print()

# 检查必要的Python包
required_packages = ['torch', 'torchvision', 'PIL', 'numpy', 'matplotlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("⚠️  缺少以下Python包:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\n请先安装:")
    print(f"   pip install {' '.join(missing_packages)}")
    print("\n或者等待conda环境创建完成后使用:")
    print("   bash scripts/download_and_train.sh")
    sys.exit(1)

# 创建快速训练配置
quick_config = """
# 快速训练配置
model:
  type: "resnet"
  backbone: "resnet18"
  num_classes: 4
  pretrained: false
  dropout: 0.3

data:
  root_dir: "data/raw"
  img_size: 128
  batch_size: 16
  num_workers: 2
  val_split: 0.2

training:
  epochs: 5
  learning_rate: 0.001
  optimizer: "adam"
  use_amp: false
  early_stopping_patience: 3
"""

config_path = PROJECT_ROOT / "configs" / "quick_test.yaml"
with open(config_path, 'w') as f:
    f.write(quick_config)

print("配置文件已创建: configs/quick_test.yaml")
print("\n训练参数:")
print("  - 模型: ResNet-18 (轻量级)")
print("  - Epochs: 5")
print("  - 图像尺寸: 128x128")
print("  - Batch Size: 16")
print()

# 启动训练
print("启动训练...")
print("-" * 40)

try:
    # 使用Python直接运行训练脚本
    cmd = [
        sys.executable, "-m", "experiments.train_baseline",
        "--config", str(config_path),
        "--data-root", str(TARGET_DIR),
        "--exp-name", "quick_test",
        "--no-wandb"
    ]

    print(f"命令: {' '.join(cmd)}")
    print()

    # 子进程运行训练
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ 训练完成!")
        print("="*60)
        print(f"\n结果保存在: checkpoints/quick_test/")
        print("\n下一步:")
        print("  1. 查看训练结果: ls checkpoints/quick_test/")
        print("  2. 评估模型: python experiments/evaluate.py --checkpoint checkpoints/quick_test/best_model.pth")
        print("  3. 启动API: python backend/app.py --model-path checkpoints/quick_test/best_model.pth")
    else:
        print("\n训练过程中出现错误，请检查日志")

except Exception as e:
    print(f"\n✗ 训练失败: {e}")
    print("\n如果遇到问题，建议:")
    print("  1. 等待conda环境创建完成")
    print("  2. 使用完整脚本: bash scripts/download_and_train.sh")
    sys.exit(1)

print()
print("="*60)
print("脚本执行完成!")
print("="*60)
