"""
EcoSort 超参数优化 (Optuna)
使用方式:
    python experiments/hpo_optuna.py --config configs/baseline_resnet50.yaml
"""

import argparse
import yaml
from pathlib import Path

import optuna
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


def create_model_from_config(config: dict, trial=None):
    """根据配置创建模型，支持 Optuna 超参数搜索"""
    model_type = config['model']['type']

    # Optuna 超参数搜索
    if trial is not None:
        if model_type == 'resnet':
            config['model']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            config['model']['use_attention'] = trial.suggest_categorical('use_attention', [True, False])

        # 训练超参数
        config['training']['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    # 创建模型
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


def objective(trial, config, train_loader, val_loader):
    """Optuna 优化目标函数"""

    # 创建模型（使用试验的超参数）
    model = create_model_from_config(config, trial)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        checkpoint_dir=f"checkpoints/hpo/trial_{trial.number}",
        experiment_name=f"trial_{trial.number}",
        use_wandb=False  # HPO 时不使用 WandB
    )

    # 训练（减少 epochs 加快搜索）
    original_epochs = config['training']['epochs']
    config['training']['epochs'] = 10  # HPO 时使用较少的 epochs
    trainer.train()
    config['training']['epochs'] = original_epochs

    # 返回验证准确率作为优化目标
    return trainer.best_val_acc


def main():
    parser = argparse.ArgumentParser(description='EcoSort Hyperparameter Optimization')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root directory')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    if args.data_root:
        config['data']['root_dir'] = args.data_root

    print(f"\n{'='*60}")
    print(f"EcoSort Hyperparameter Optimization")
    print(f"n_trials: {args.n_trials}")
    print(f"{'='*60}\n")

    # 创建数据加载器
    print("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_root=config['data']['root_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            img_size=config['data']['img_size'],
            val_split=config['data']['val_split']
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # 创建 Optuna study
    study = optuna.create_study(
        study_name='ecosort_hpo',
        direction='maximize',
        storage='sqlite:///checkpoints/hpo/optuna.db',
        load_if_exists=True
    )

    # 运行优化
    study.optimize(
        lambda trial: objective(trial, config, train_loader, val_loader),
        n_trials=args.n_trials,
        n_jobs=1  # 单任务运行
    )

    # 打印结果
    print("\n" + "="*60)
    print("Hyperparameter Optimization Completed")
    print("="*60)
    print(f"\nBest validation accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存最佳配置
    best_config = config.copy()
    best_config.update(study.best_params)

    output_path = 'checkpoints/hpo/best_config.yaml'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"\nBest config saved to: {output_path}")


if __name__ == '__main__':
    main()
