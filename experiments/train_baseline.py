"""
EcoSort 训练入口脚本
使用方式:
    python experiments/train_baseline.py --config configs/baseline_resnet50.yaml
"""

import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.train.trainer import Trainer


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    """根据配置创建模型"""
    model_type = config['model']['type']

    if model_type == 'resnet':
        return create_resnet_model(
            backbone=config['model']['backbone'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3),
            use_attention=config['model'].get('use_attention', False)
        )
    elif model_type == 'efficientnet':
        return create_efficientnet_model(
            backbone=config['model']['backbone'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='EcoSort Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root directory')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (overrides config)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖配置
    if args.data_root:
        config['data']['root_dir'] = args.data_root
    if args.exp_name:
        config['experiment_name'] = args.exp_name
    else:
        config['experiment_name'] = Path(args.config).stem

    print(f"\n{'='*60}")
    print(f"EcoSort Training: {config['experiment_name']}")
    print(f"{'='*60}\n")

    # 打印配置
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # 创建数据加载器
    print("\nCreating dataloaders...")
    try:
        config_class_names = config.get('data', {}).get('class_names')
        strong_aug = config.get('augmentation', {}).get('random_erasing_prob', 0) > 0
        train_loader, val_loader = create_dataloaders(
            data_root=config['data']['root_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            img_size=config['data']['img_size'],
            val_split=config['data']['val_split'],
            class_names=config_class_names,
            strong_aug=strong_aug
        )

        # 动态同步类别信息
        inferred_class_names = train_loader.dataset.class_names
        config['model']['num_classes'] = len(inferred_class_names)
        config['data']['class_names'] = inferred_class_names

        class_counts = train_loader.dataset.get_class_distribution()
        ordered_counts = [class_counts[name] for name in inferred_class_names]
        config['data']['class_counts'] = ordered_counts

        print(f"Detected {len(inferred_class_names)} classes")
        print(f"Class names: {inferred_class_names}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("\nPlease ensure your data is organized as:")
        print("data/raw/")
        print("  ├── recyclable/")
        print("  ├── hazardous/")
        print("  ├── kitchen/")
        print("  └── other/")
        return

    # 创建模型
    print("\nCreating model...")
    model = create_model(config)

    # 创建训练器
    print("\nInitializing trainer...")
    trainer_config = dict(config['training'])
    trainer_config['data'] = config.get('data', {})
    trainer_config['loss'] = config.get('loss', {})

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=config['experiment_name'],
        use_wandb=not args.no_wandb
    )

    # 开始训练
    print("\nStarting training...\n")
    trainer.train()

    print("\nTraining completed!")
    print(f"Checkpoints saved to: {Path(args.checkpoint_dir) / config['experiment_name']}")


if __name__ == '__main__':
    main()
