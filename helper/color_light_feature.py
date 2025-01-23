import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_color_features(image):
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算HSV色彩直方图
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    # 计算颜色熵
    hist_hue = hist_hue / hist_hue.sum()  # 归一化
    hist_saturation = hist_saturation / hist_saturation.sum()
    hist_value = hist_value / hist_value.sum()

    entropy_hue = -np.sum(hist_hue * np.log2(hist_hue + 1e-6))
    entropy_saturation = -np.sum(hist_saturation * np.log2(hist_saturation + 1e-6))
    entropy_value = -np.sum(hist_value * np.log2(hist_value + 1e-6))

    return hist_hue, hist_saturation, hist_value, entropy_hue, entropy_saturation, entropy_value

def extract_lighting_features(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子计算梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算光照方向和强度
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi

    return gradient_magnitude, gradient_orientation

def plot_color_features(hist_hue, hist_saturation, hist_value):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Hue Histogram')
    plt.plot(hist_hue, color='r')
    
    plt.subplot(1, 3, 2)
    plt.title('Saturation Histogram')
    plt.plot(hist_saturation, color='g')
    
    plt.subplot(1, 3, 3)
    plt.title('Value Histogram')
    plt.plot(hist_value, color='b')

    plt.show()

def plot_lighting_features(gradient_magnitude, gradient_orientation):
    fig = plt.figure(figsize=(10, 6))
    
    ax = fig.add_subplot(121)
    ax.imshow(gradient_magnitude, cmap='gray')
    ax.set_title('Gradient Magnitude')

    ax = fig.add_subplot(122)
    ax.imshow(gradient_orientation, cmap='hsv')
    ax.set_title('Gradient Orientation')

    plt.show()

if __name__ == "__main__":
    # 读取图像
    image_path = 'D:/image/HybridFusionNet/assets/0/0AEhR5Q82pV7x7vk.jpg'  # 修改为你的图像路径
    image_path = 'D:/image/HybridFusionNet/assets/1/0G62M3zvb85koOiD.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        exit(1)

    # 提取颜色特征
    hist_hue, hist_saturation, hist_value, entropy_hue, entropy_saturation, entropy_value = extract_color_features(image)

    # 提取光照特征
    gradient_magnitude, gradient_orientation = extract_lighting_features(image)

    # 打印颜色熵值
    print(f"Entropy of Hue: {entropy_hue}")
    print(f"Entropy of Saturation: {entropy_saturation}")
    print(f"Entropy of Value: {entropy_value}")

    # 可视化颜色特征
    plot_color_features(hist_hue, hist_saturation, hist_value)

    # 可视化光照特征
    plot_lighting_features(gradient_magnitude, gradient_orientation)
