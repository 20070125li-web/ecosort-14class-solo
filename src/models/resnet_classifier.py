"""
EcoSort ResNet Classifier
基于ResNet的垃圾分类模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetClassifier(nn.Module):
    """ResNet 垃圾分类器

    支持的 backbone:
    - resnet50
    - resnet101
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_attention: bool = False
    ):
        """
        Args:
            num_classes: 分类类别数 (4类垃圾)
            backbone: ResNet 变体
            pretrained: 是否使用 ImageNet 预训练权重
            dropout: Dropout 概率
            use_attention: 是否使用注意力机制
        """
        super(ResNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_attention = use_attention

        # 加载预训练 ResNet
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 提取特征提取器 (除去最后的 FC 层)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 注意力机制 (可选)
        if use_attention:
            self.attention = CBAM(self.feature_dim)
        else:
            self.attention = None

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

        # 初始化分类头权重
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for m in self.classifier.modules():
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
        # 特征提取
        features = self.features(x)  # (B, 2048, H/32, W/32)

        # 注意力机制
        if self.attention is not None:
            features = self.attention(features)

        # 池化
        pooled = self.avgpool(features)  # (B, 2048, 1, 1)

        # 分类
        logits = self.classifier(pooled)

        return logits

    def get_features(self, x):
        """提取特征 (用于可视化或 t-SNE)"""
        features = self.features(x)
        if self.attention is not None:
            features = self.attention(features)
        pooled = self.avgpool(features)
        return pooled.view(pooled.size(0), -1)


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    参考: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(CBAM, self).__init__()

        # 通道注意力
        self.channel_attention = SequentialPolarized(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=True)
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W) 加权后的特征
        """
        # 通道注意力
        ca = self.channel_attention(x)
        ca = torch.sigmoid(ca)
        x = x * ca

        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x


class SequentialPolarized(nn.Module):
    """用于通道注意力的简化模块"""
    def __init__(self, *args):
        super(SequentialPolarized, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


def create_resnet_model(
    backbone: str = 'resnet50',
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> ResNetClassifier:
    """创建 ResNet 分类器的工厂函数

    Args:
        backbone: 模型类型
        num_classes: 类别数
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数

    Returns:
        model: ResNetClassifier 实例
    """
    model = ResNetClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs
    )

    return model


if __name__ == '__main__':
    # 测试模型
    model = create_resnet_model(
        backbone='resnet50',
        num_classes=4,
        pretrained=False,
        use_attention=True
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
