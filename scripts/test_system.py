"""
EcoSort System Test Script
Validate all core modules functionality and integration
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test dependency imports"""
    print("Test 1: Dependency Imports...")
    try:
        import torch
        import torchvision
        import flask
        import PIL
        import numpy as np
        import sklearn
        print("  ✓ All dependencies imported successfully")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_data_module():
    """Test data handling module"""
    print("\nTest 2: Data Module...")
    try:
        from src.data.dataset import TrashDataset, get_data_transforms

        # Test data transformations
        transform = get_data_transforms('train')
        print("  ✓ Data transforms created successfully")

        # Test dataset class
        print(f"  ✓ Classes: {TrashDataset.CLASS_NAMES}")
        print(f"  ✓ Class mapping: {TrashDataset.CLASS_TO_IDX}")
        return True
    except Exception as e:
        print(f"  ✗ Data module test failed: {e}")
        return False


def test_model_module():
    """Test model architecture module"""
    print("\nTest 3: Model Module...")
    try:
        from src.models.resnet_classifier import create_resnet_model
        from src.models.efficientnet_classifier import create_efficientnet_model

        import torch

        # Test ResNet model
        model_resnet = create_resnet_model(
            backbone='resnet50',
            num_classes=4,
            pretrained=False
        )
        print("  ✓ ResNet model created successfully")

        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        y = model_resnet(x)
        assert y.shape == (2, 4), f"Incorrect output shape: {y.shape}"
        print("  ✓ ResNet forward pass successful")

        # Test EfficientNet model
        model_effnet = create_efficientnet_model(
            backbone='efficientnet-b3',
            num_classes=4,
            pretrained=False
        )
        print("  ✓ EfficientNet model created successfully")

        return True
    except Exception as e:
        print(f"  ✗ Model module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_module():
    """Test training manager module"""
    print("\nTest 4: Trainer Module...")
    try:
        from src.train.trainer import Trainer
        print("  ✓ Trainer imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Trainer module test failed: {e}")
        return False


def test_utils_module():
    """Test utility functions module"""
    print("\nTest 5: Utilities Module...")
    try:
        from src.utils.quantization import post_training_quantization
        print("  ✓ Quantization utilities imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Utilities module test failed: {e}")
        return False


def test_backend_module():
    """Test backend API module"""
    print("\nTest 6: Backend Module...")
    try:
        # Check backend file existence
        backend_file = project_root / 'backend' / 'app.py'
        if backend_file.exists():
            print("  ✓ Backend files exist")
            return True
        else:
            print("  ✗ Backend files not found")
            return False
    except Exception as e:
        print(f"  ✗ Backend module test failed: {e}")
        return False


def test_mobile_module():
    """Test mobile application module"""
    print("\nTest 7: Mobile Module...")
    try:
        # Check Android source files existence
        main_activity = project_root / 'mobile' / 'app' / 'src' / 'main' / 'java' / 'com' / 'example' / 'ecosort' / 'MainActivity.java'
        api_client = project_root / 'mobile' / 'app' / 'src' / 'main' / 'java' / 'com' / 'example' / 'ecosort' / 'ApiClient.java'

        if main_activity.exists() and api_client.exists():
            print("  ✓ Android source files exist")
            return True
        else:
            print("  ✗ Android source files not found")
            return False
    except Exception as e:
        print(f"  ✗ Mobile module test failed: {e}")
        return False


def test_configs():
    """Test configuration files"""
    print("\nTest 8: Configuration Files...")
    try:
        import yaml

        config_file = project_root / 'configs' / 'baseline_resnet50.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print("  ✓ Configuration file exists and is valid")
            print(f"  Model type: {config['model']['type']}")
            return True
        else:
            print("  ✗ Configuration file not found")
            return False
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_dummy_data_creation():
    """Test dummy dataset generation"""
    print("\nTest 9: Dummy Data Creation...")
    try:
        from scripts.create_dummy_data import create_dummy_dataset

        # Create small test dataset
        test_dir = project_root / 'data' / 'test'
        create_dummy_dataset(output_dir=str(test_dir), num_samples_per_class=5)
        print("  ✓ Dummy dataset created successfully")

        # Clean up test data
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("  ✓ Test data cleaned up")

        return True
    except Exception as e:
        print(f"  ✗ Dummy data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """End-to-end integration test"""
    print("\nTest 10: End-to-End Pipeline...")
    try:
        from src.data.dataset import TrashDataset, get_data_transforms
        from src.models.resnet_classifier import create_resnet_model
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        # Create dummy dataset
        print("  Creating dummy dataset...")
        dummy_data = TensorDataset(
            torch.randn(20, 3, 256, 256),
            torch.randint(0, 4, (20,))
        )
        dummy_loader = DataLoader(dummy_data, batch_size=4)

        # Create model
        print("  Initializing model...")
        model = create_resnet_model(
            backbone='resnet50',
            num_classes=4,
            pretrained=False
        )

        # Test training loop basics
        print("  Testing forward propagation...")
        model.train()
        for images, labels in dummy_loader:
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            print(f"    Loss: {loss.item():.4f}")
            break

        print("  ✓ End-to-end test successful")
        return True
    except Exception as e:
        print(f"  ✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all system tests"""
    print("="*60)
    print("EcoSort System Validation")
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
            print(f"\nTest exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Test summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - please check the relevant modules.")
        return 1


if __name__ == '__main__':
    exit(main())
