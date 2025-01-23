import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s
from timm import create_model
from safetensors.torch import load_file


class HybridFusionNet(nn.Module):
    def __init__(self):
        super(HybridFusionNet, self).__init__()

        # 主干网络：EfficientNet 和 Swin Transformer
        self.backbone_efficientnet = create_model('tf_efficientnetv2_b2', pretrained=False, num_classes=0)       
        self.backbone_swin = create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=0)

        # 加载EfficientNetV2 B2的safetensors权重
        efficientnet_weights = load_file('../pretrain/tf_efficientnetv2_b2.safetensors')
        self.backbone_efficientnet.load_state_dict(efficientnet_weights)

        # 加载Swin Transformer的safetensors权重
        swin_weights = load_file('../pretrain/swin_small_patch4_window7_224.safetensors')
        self.backbone_swin.load_state_dict(swin_weights)


        # 冻结 EfficientNet 部分层
        for param in self.backbone_efficientnet.conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone_efficientnet.bn1.parameters():
            param.requires_grad = False

        # 冻结 Swin Transformer 的前两个阶段
        for param in self.backbone_swin.layers[0].parameters():
            param.requires_grad = False
        for param in self.backbone_swin.layers[1].parameters():
            param.requires_grad = False

        # 调整主干输出尺寸
        self.fc_backbone = nn.Linear(1536 + 768, 2048)

        # 纹理特征提取模块（GLCM + CNN）
        self.texture_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 32)
        )

        # 频域特征提取模块（FFT + 小波变换）
        self.frequency_fc = nn.Sequential(
            nn.Linear(128 * 128, 128),
            nn.ReLU()
        )

        # 边缘特征提取模块（HED + CNN）
        self.edge_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 64)
        )

        # 局部几何特征提取模块（LBP）
        self.geometric_fc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU()
        )

        # 自适应注意力模块
        self.attention_fc = nn.Sequential(
            nn.Linear(32 + 128 + 64 + 32, 4),
            nn.Softmax(dim=1)
        )

        # 动态融合模块
        self.fusion_fc = nn.Sequential(
            nn.Linear(2048 + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, texture, frequency, edge, geometric):
        # 主干网络特征提取
        backbone_features_efficientnet = self.backbone_efficientnet.features(x)
        backbone_features_efficientnet = F.adaptive_avg_pool2d(backbone_features_efficientnet, 1).view(x.size(0), -1)
        backbone_features_swin = self.backbone_swin(x)

        backbone_features = torch.cat([backbone_features_efficientnet, backbone_features_swin], dim=1)
        backbone_features = self.fc_backbone(backbone_features)

        # 模态特征提取
        texture_features = self.texture_cnn(texture)
        frequency_features = self.frequency_fc(frequency.view(frequency.size(0), -1))
        edge_features = self.edge_cnn(edge)
        geometric_features = self.geometric_fc(geometric)

        # 拼接模态特征
        multimodal_features = torch.cat(
            [texture_features, frequency_features, edge_features, geometric_features], dim=1
        )

        # 自适应注意力
        attention_weights = self.attention_fc(multimodal_features)
        weighted_features = torch.cat([
            texture_features * attention_weights[:, 0:1],
            frequency_features * attention_weights[:, 1:2],
            edge_features * attention_weights[:, 2:3],
            geometric_features * attention_weights[:, 3:4],
        ], dim=1)

        # 动态融合
        combined_features = torch.cat([backbone_features, weighted_features], dim=1)
        fusion_output = self.fusion_fc(combined_features)

        # 分类输出
        output = self.classifier(fusion_output)
        return output


# 测试模型
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 8
    x = torch.rand(batch_size, 3, 224, 224)  # 主图像
    texture = torch.rand(batch_size, 1, 224, 224)  # 纹理图像
    frequency = torch.rand(batch_size, 128, 128)  # 频域特征
    edge = torch.rand(batch_size, 1, 224, 224)  # 边缘图像
    geometric = torch.rand(batch_size, 256)  # 几何特征

    # 初始化模型并测试前向传播
    model = HybridFusionNet()
    output = model(x, texture, frequency, edge, geometric)
    print("模型输出尺寸:", output.shape)  # 输出尺寸应为 [batch_size, 1]
