"""
EcoSort 系统测试脚本
验证各个模块是否正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试依赖导入"""
    print("测试 1: 依赖导入...")
    try:
        import torch
        import torchvision
        import flask
        import PIL
        import numpy as np
        import sklearn
        print("  ✓ 所有依赖导入成功")
        print(f"  PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False


def test_data_module():
    """测试数据模块"""
    print("\n测试 2: 数据模块...")
    try:
        from src.data.dataset import TrashDataset, get_data_transforms

        # 测试数据变换
        transform = get_data_transforms('train')
        print("  ✓ 数据变换创建成功")

        # 测试数据集类
        print(f"  ✓ 类别: {TrashDataset.CLASS_NAMES}")
        print(f"  ✓ 类别映射: {TrashDataset.CLASS_TO_IDX}")
        return True
    except Exception as e:
        print(f"  ✗ 数据模块测试失败: {e}")
        return False


def test_model_module():
    """测试模型模块"""
    print("\n测试 3: 模型模块...")
    try:
        from src.models.resnet_classifier import create_resnet_model
        from src.models.efficientnet_classifier import create_efficientnet_model

        import torch

        # 测试 ResNet
        model_resnet = create_resnet_model(
            backbone='resnet50',
            num_classes=4,
            pretrained=False
        )
        print("  ✓ ResNet 模型创建成功")

        # 测试前向传播
        x = torch.randn(2, 3, 256, 256)
        y = model_resnet(x)
        assert y.shape == (2, 4), f"输出形状错误: {y.shape}"
        print("  ✓ ResNet 前向传播成功")

        # 测试 EfficientNet
        model_effnet = create_efficientnet_model(
            backbone='efficientnet-b3',
            num_classes=4,
            pretrained=False
        )
        print("  ✓ EfficientNet 模型创建成功")

        return True
    except Exception as e:
        print(f"  ✗ 模型模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_module():
    """测试训练器模块"""
    print("\n测试 4: 训练器模块...")
    try:
        from src.train.trainer import Trainer
        print("  ✓ 训练器导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 训练器模块测试失败: {e}")
        return False


def test_utils_module():
    """测试工具模块"""
    print("\n测试 5: 工具模块...")
    try:
        from src.utils.quantization import post_training_quantization
        print("  ✓ 量化工具导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 工具模块测试失败: {e}")
        return False


def test_backend_module():
    """测试后端模块"""
    print("\n测试 6: 后端模块...")
    try:
        # 检查后端文件是否存在
        backend_file = project_root / 'backend' / 'app.py'
        if backend_file.exists():
            print("  ✓ 后端文件存在")
            return True
        else:
            print("  ✗ 后端文件不存在")
            return False
    except Exception as e:
        print(f"  ✗ 后端模块测试失败: {e}")
        return False


def test_mobile_module():
    """测试移动端模块"""
    print("\n测试 7: 移动端模块...")
    try:
        # 检查 Android 文件是否存在
        main_activity = project_root / 'mobile' / 'app' / 'src' / 'main' / 'java' / 'com' / 'example' / 'ecosort' / 'MainActivity.java'
        api_client = project_root / 'mobile' / 'app' / 'src' / 'main' / 'java' / 'com' / 'example' / 'ecosort' / 'ApiClient.java'

        if main_activity.exists() and api_client.exists():
            print("  ✓ Android 源文件存在")
            return True
        else:
            print("  ✗ Android 源文件不存在")
            return False
    except Exception as e:
        print(f"  ✗ 移动端模块测试失败: {e}")
        return False


def test_configs():
    """测试配置文件"""
    print("\n测试 8: 配置文件...")
    try:
        import yaml

        config_file = project_root / 'configs' / 'baseline_resnet50.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print("  ✓ 配置文件存在且有效")
            print(f"  模型类型: {config['model']['type']}")
            return True
        else:
            print("  ✗ 配置文件不存在")
            return False
    except Exception as e:
        print(f"  ✗ 配置文件测试失败: {e}")
        return False


def test_dummy_data_creation():
    """测试虚拟数据创建"""
    print("\n测试 9: 虚拟数据创建...")
    try:
        from scripts.create_dummy_data import create_dummy_dataset

        # 创建少量测试数据
        test_dir = project_root / 'data' / 'test'
        create_dummy_dataset(output_dir=str(test_dir), num_samples_per_class=5)
        print("  ✓ 虚拟数据创建成功")

        # 清理测试数据
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("  ✓ 测试数据已清理")

        return True
    except Exception as e:
        print(f"  ✗ 虚拟数据创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """端到端测试"""
    print("\n测试 10: 端到端流程...")
    try:
        from src.data.dataset import TrashDataset, get_data_transforms
        from src.models.resnet_classifier import create_resnet_model
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        # 创建虚拟数据
        print("  创建虚拟数据集...")
        dummy_data = TensorDataset(
            torch.randn(20, 3, 256, 256),
            torch.randint(0, 4, (20,))
        )
        dummy_loader = DataLoader(dummy_data, batch_size=4)

        # 创建模型
        print("  创建模型...")
        model = create_resnet_model(
            backbone='resnet50',
            num_classes=4,
            pretrained=False
        )

        # 测试训练循环
        print("  测试前向传播...")
        model.train()
        for images, labels in dummy_loader:
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            print(f"    Loss: {loss.item():.4f}")
            break

        print("  ✓ 端到端测试成功")
        return True
    except Exception as e:
        print(f"  ✗ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("EcoSort 系统测试")
    print("="*60)

    tests = [
        test_imports,
        test_data_module,
        test_model_module,
        test_trainer_module,
        test_utils_module,
        test_backend_module,
        test_mobile_module,
        test_configs,
        test_dummy_data_creation,
        test_end_to_end,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过! 系统已就绪。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查相关模块。")
        return 1


if __name__ == '__main__':
    exit(main())
