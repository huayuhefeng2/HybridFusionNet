import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None
        self.gradient = None

        # 注册钩子
        self.hook_layers()

    def hook_layers(self):
        """在目标卷积层注册钩子，以获取激活图和梯度"""
        self.target_layer.register_forward_hook(self.save_feature_map)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_feature_map(self, module, input, output):
        """保存卷积层的输出特征图"""
        self.feature_map = output

    def save_gradient(self, module, grad_input, grad_output):
        """保存梯度信息"""
        self.gradient = grad_output[0]

    def generate_cam(self):
        """根据保存的特征图和梯度生成 Grad-CAM"""
        # 计算梯度对特征图的加权平均
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)  # 对H和W维度进行平均
        cam = torch.sum(weights * self.feature_map, dim=1, keepdim=True)
        cam = F.relu(cam)  # 只保留正值，ReLU激活
        cam = cam.squeeze().cpu().detach().numpy()

        # 缩放为原始图像的尺寸
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam)  # 归一化到[0, 1]范围
        return cam

    def visualize(self, image, cam):
        """将 Grad-CAM 热力图与原始图像叠加"""
        cam = np.uint8(255 * cam)  # 转换为 0-255 之间的值
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 使用 Jet 色图
        heatmap = np.float32(heatmap) / 255

        # 图像叠加热力图
        image = np.array(image).astype(np.float32) / 255
        superimposed_img = 0.5 * image + 0.5 * heatmap  # 图像与热力图加权合成
        return np.uint8(255 * superimposed_img)
