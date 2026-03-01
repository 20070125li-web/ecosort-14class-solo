"""
创建示例垃圾数据集
用于测试和演示 (仅生成虚拟图像)
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_dummy_image(text: str, size: tuple = (256, 256), color: tuple = (255, 255, 255)) -> Image.Image:
    """创建带文字的虚拟图像"""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)

    # 绘制文字
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()

    # 获取文字边界框
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 居中绘制
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, text, fill=(0, 0, 0), font=font)

    return img


def create_dummy_dataset(
    output_dir: str = 'data/raw',
    num_samples_per_class: int = 100
):
    """创建虚拟垃圾数据集"""

    class_config = {
        'recyclable': {
            'text': '♻️',
            'color': (100, 200, 100),
            'variations': ['瓶', '罐', '纸', '盒']
        },
        'hazardous': {
            'text': '☠️',
            'color': (200, 100, 100),
            'variations': ['电池', '药', '灯', '毒']
        },
        'kitchen': {
            'text': '🍎',
            'color': (100, 100, 200),
            'variations': ['果', '菜', '饭', '骨']
        },
        'other': {
            'text': '🗑️',
            'color': (200, 200, 100),
            'variations': ['土', '陶', '砖', '瓦']
        }
    }

    output_path = Path(output_dir)

    print(f"创建虚拟数据集到: {output_dir}")
    print(f"每类样本数: {num_samples_per_class}\n")

    for class_name, config in class_config.items():
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"生成 {class_name} 类...")

        for i in range(num_samples_per_class):
            # 随机选择变体
            text = np.random.choice(config['variations'])

            # 创建图像
            img = create_dummy_image(
                text=text,
                color=config['color']
            )

            # 保存
            filename = f"{class_name}_{i:04d}.jpg"
            img.save(class_dir / filename, quality=85)

        print(f"  ✓ 生成 {num_samples_per_class} 张图像")

    print("\n虚拟数据集创建完成!")
    print(f"位置: {output_dir}")
    print(f"总计: {num_samples_per_class * len(class_config)} 张图像")

    # 打印统计
    print("\n数据集统计:")
    for class_name in class_config.keys():
        class_dir = output_path / class_name
        count = len(list(class_dir.glob('*.jpg')))
        print(f"  {class_name}: {count} 张")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='创建虚拟垃圾数据集')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='每类样本数量')

    args = parser.parse_args()

    create_dummy_dataset(
        output_dir=args.output_dir,
        num_samples_per_class=args.num_samples
    )
