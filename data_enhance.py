import os
import random
from PIL import Image, ImageEnhance


def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def random_rotation(image):
    angle = random.uniform(-5, 5)  # 随机旋转角度范围
    return image.rotate(angle, expand=True)


def random_flip(image):
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def save_image(image, path, name, suffix):
    if not os.path.exists(path):
        os.makedirs(path)
    # 获取文件的基本名和扩展名
    base, ext = os.path.splitext(name)
    filename = f"{base}_{suffix}{ext}"
    image.save(os.path.join(path, filename))


def augment_image(image_path, output_path):
    image = Image.open(image_path)

    # # 原始图像
    # save_image(image, output_path, os.path.basename(image_path), 'original')

    # 调整亮度
    brightness_factor = random.uniform(0.5, 1.5)
    brightness_image = adjust_brightness(image, brightness_factor)
    save_image(brightness_image, output_path, os.path.basename(image_path), 'brightness')


    # # 调整亮度
    # brightness_factor2 = random.uniform(0.2, 2.0)
    # brightness_image = adjust_brightness(image, brightness_factor2)
    # save_image(brightness_image, output_path, os.path.basename(image_path), 'brightness2')

    # 调整对比度
    contrast_factor = random.uniform(0.5, 1.5)
    contrast_image = adjust_contrast(image, contrast_factor)
    save_image(contrast_image, output_path, os.path.basename(image_path), 'contrast')

    # 旋转
    rotated_image = random_rotation(image)
    save_image(rotated_image, output_path, os.path.basename(image_path), 'rotated')

    # 翻转
    flipped_image = random_flip(image)
    save_image(flipped_image, output_path, os.path.basename(image_path), 'flipped')

    # 全处理
    final_image = random_flip(contrast_image)
    final_image = adjust_brightness(final_image, brightness_factor)
    save_image(final_image, output_path, os.path.basename(image_path), 'final')


def augment_dataset(dataset_path, output_path):
    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_path, image_name)
            augment_image(image_path, output_path)

# 使用示例
dataset_path = 'boundary/train/1'  # 你的数据集路径
output_path = 'boundary/train/1'  # 增强后的数据集保存路径
augment_dataset(dataset_path, output_path)