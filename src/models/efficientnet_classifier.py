"""
EcoSort EfficientNet Classifier
基于EfficientNet的高效垃圾分类模型
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """EfficientNet 垃圾分类器

    支持的 backbone:
    - efficientnet-b0 到 efficientnet-b7
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = 'efficientnet-b3',
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_connect_rate: float = 0.2
    ):
        """
        Args:
            num_classes: 分类类别数 (4类垃圾)
            backbone: EfficientNet 变体
            pretrained: 是否使用 ImageNet 预训练权重
            dropout: Dropout 概率
            drop_connect_rate: DropConnect 概率
        """
        super(EfficientNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # 加载预训练 EfficientNet
        self.backbone = EfficientNet.from_pretrained(
            backbone if pretrained else 'efficientnet-b0',
            num_classes=num_classes,
            dropout_rate=dropout,
            drop_connect_rate=drop_connect_rate
        )

        # 获取特征维度
        if 'b0' in backbone:
            self.feature_dim = 1280
        elif 'b1' in backbone:
            self.feature_dim = 1280
        elif 'b2' in backbone:
            self.feature_dim = 1408
        elif 'b3' in backbone:
            self.feature_dim = 1536
        elif 'b4' in backbone:
            self.feature_dim = 1792
        else:
            self.feature_dim = 1280

        # 替换最后的分类层
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

        # 初始化分类层权重
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for m in self.backbone._fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图像

        Returns:
            logits: (B, num_classes) 分类logits
        """
        logits = self.backbone(x)
        return logits

    def get_features(self, x):
        """提取特征 (用于可视化或 t-SNE)"""
        # 移除最后的分类层
        features = self.backbone.extract_features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.view(pooled.size(0), -1)


def create_efficientnet_model(
    backbone: str = 'efficientnet-b3',
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> EfficientNetClassifier:
    """创建 EfficientNet 分类器的工厂函数

    Args:
        backbone: 模型类型
        num_classes: 类别数
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数

    Returns:
        model: EfficientNetClassifier 实例
    """
    model = EfficientNetClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs
    )

    return model


if __name__ == '__main__':
    # 测试模型
    model = create_efficientnet_model(
        backbone='efficientnet-b3',
        num_classes=4,
        pretrained=False
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # 测试特征提取
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
