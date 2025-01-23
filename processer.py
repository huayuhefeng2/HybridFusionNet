import os
import cv2
import numpy as np
import torch
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift

def load_images_from_folder(folder, image_size=(224, 224), batch_size=8):
    """
    从文件夹加载图像并调整为固定尺寸。
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
        if len(images) == batch_size:
            break
    images = np.stack(images, axis=0)  # 将图像堆叠为批量
    images = images / 255.0  # 归一化
    images = np.transpose(images, (0, 3, 1, 2))  # 转换为 [N, C, H, W]
    return torch.tensor(images, dtype=torch.float32)


def extract_texture_features(images, batch_size):
    """
    提取纹理特征（灰度共生矩阵简化版）。
    """
    gray_images = np.mean(images, axis=1, keepdims=True)  # 转为灰度
    return torch.tensor(gray_images, dtype=torch.float32)


def extract_frequency_features(images, target_size=(128, 128)):
    """
    提取频域特征（FFT）。
    """
    freq_features = []
    for img in images:
        gray = np.mean(img, axis=0)  # 转灰度
        gray_resized = cv2.resize(gray, target_size)
        f_transform = fftshift(fft2(gray_resized))
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)
        freq_features.append(magnitude_spectrum)
    freq_features = np.stack(freq_features, axis=0)  # 堆叠
    return torch.tensor(freq_features, dtype=torch.float32)


def extract_edge_features(images):
    """
    提取边缘特征（Canny 边缘检测）。
    """
    edge_images = []
    for img in images:
        gray = np.mean(img, axis=0)  # 转灰度
        edges = cv2.Canny(np.uint8(gray * 255), 100, 200)
        edges = edges / 255.0  # 归一化
        edge_images.append(edges)
    edge_images = np.stack(edge_images, axis=0)
    edge_images = np.expand_dims(edge_images, axis=1)  # 增加通道维度
    return torch.tensor(edge_images, dtype=torch.float32)


def extract_geometric_features(images):
    """
    提取几何特征（LBP）。
    """
    geometric_features = []
    for img in images:
        gray = np.mean(img, axis=0)  # 转灰度
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(257), density=True)
        geometric_features.append(hist)
    geometric_features = np.stack(geometric_features, axis=0)
    return torch.tensor(geometric_features, dtype=torch.float32)


# 主函数
if __name__ == "__main__":
    # 设置文件夹路径和参数
    folder = "path_to_images"  # 替换为图像文件夹路径
    batch_size = 8
    image_size = (224, 224)

    # 加载图像
    images = load_images_from_folder(folder, image_size=image_size, batch_size=batch_size).numpy()

    # 提取特征
    x = torch.tensor(images, dtype=torch.float32)  # 主图像
    texture = extract_texture_features(images, batch_size)  # 纹理图像
    frequency = extract_frequency_features(images)  # 频域特征
    edge = extract_edge_features(images)  # 边缘图像
    geometric = extract_geometric_features(images)  # 几何特征

    # 打印形状验证
    print("主图像形状:", x.shape)  # [batch_size, 3, 224, 224]
    print("纹理图像形状:", texture.shape)  # [batch_size, 1, 224, 224]
    print("频域特征形状:", frequency.shape)  # [batch_size, 128, 128]
    print("边缘图像形状:", edge.shape)  # [batch_size, 1, 224, 224]
    print("几何特征形状:", geometric.shape)  # [batch_size, 256]
