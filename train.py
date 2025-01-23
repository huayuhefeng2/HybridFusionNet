from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from torchvision.transforms import ToTensor
import glob
from model.HybridFusionNet import HybridFusionNet
from processer import *
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from model.GradCAM import *

# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, feature_extraction_funcs, image_size=(224, 224)):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extraction_funcs = feature_extraction_funcs
        self.image_size = image_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB').resize(self.image_size)
        img_tensor = ToTensor()(img)

        texture = extract_texture_features(img_tensor.numpy(), batch_size=1).squeeze(0)
        frequency = extract_frequency_features(img_tensor.numpy()).squeeze(0)
        edge = extract_edge_features(img_tensor.numpy()).squeeze(0)
        geometric = extract_geometric_features(img_tensor.numpy()).squeeze(0)

        return {
            "image": img_tensor,
            "texture": texture,
            "frequency": frequency,
            "edge": edge,
            "geometric": geometric
        }, torch.tensor(label, dtype=torch.long)


def load_data(dataset_path):
    """加载数据并根据文件夹名称分配标签"""
    file_paths = []
    labels = []

    for label_name, label_value in [("real", 1), ("fake", 0)]:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder):
            raise ValueError(f"Folder '{folder}' does not exist in '{dataset_path}'.")

        image_files = glob.glob(os.path.join(folder, "*.*"))
        file_paths.extend(image_files)
        labels.extend([label_value] * len(image_files))

    return file_paths, labels


def train_or_test(mode, dataset_path, model_path=None, save_model_path="checkpoints", 
                  output_csv="results.csv", test_input=None, epochs=10, batch_size=8, lr=1e-3):
    print("Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridFusionNet().to(device)

    if mode == "test":
        if model_path is None:
            raise ValueError("In test mode, --model_path is required.")

        model.load_state_dict(torch.load(model_path))
        model.eval()

        if os.path.isfile(test_input):
            print(f"Processing single image: {test_input}")
            process_single_image(model, test_input, device)
        elif os.path.isdir(test_input):
            print(f"Processing folder: {test_input}")
            process_folder(model, test_input, output_csv, device)
        else:
            raise ValueError(f"Invalid test_input: {test_input}. Must be a file or directory.")
        return

    # Training setup
    file_paths, labels = load_data(dataset_path)
    train_files, val_files, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_files, train_labels, feature_extraction_funcs=[])
    val_dataset = CustomDataset(val_files, val_labels, feature_extraction_funcs=[])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard writer
    writer = SummaryWriter(log_dir="logs")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_acc += (preds == labels).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 每个epoch保存模型到checkpoints文件夹
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_file_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_file_path)
        print(f"Model saved to {model_file_path}")

        # 只有验证集准确率提升时才保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model saved with Val Acc: {best_acc:.4f}")

    writer.close()


def process_single_image(model, image_path, device, target_layer='conv5_block3_out'):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    texture = extract_texture_features(img_tensor.cpu().numpy(), batch_size=1).to(device)
    frequency = extract_frequency_features(img_tensor.cpu().numpy()).to(device)
    edge = extract_edge_features(img_tensor.cpu().numpy()).to(device)
    geometric = extract_geometric_features(img_tensor.cpu().numpy()).to(device)

    inputs = {
        "image": img_tensor,
        "texture": texture,
        "frequency": frequency,
        "edge": edge,
        "geometric": geometric,
    }

    model.eval()

    # Grad-CAM 处理
    target_conv_layer = dict(model.named_modules())[target_layer]  # 选定目标卷积层
    gradcam = GradCAM(model, target_conv_layer)

    # 获取输出并计算损失
    with torch.no_grad():
        outputs = model(**inputs)
        fake_prob = torch.softmax(outputs, dim=1)[0, 0].item()
        is_fake = fake_prob > 0.5

        # 计算 Grad-CAM
        outputs.backward()  # 反向传播以计算梯度
        cam = gradcam.generate_cam()
        
        # 可视化 Grad-CAM
        cam_image = gradcam.visualize(img, cam)

        # 显示结果
        plt.imshow(cam_image)
        plt.title(f"Fake Probability: {fake_prob:.4f}, Prediction: {'Fake' if is_fake else 'Real'}")
        plt.show()

def process_folder(model, folder_path, output_csv, device, target_layer='conv5_block3_out'):
    files = glob.glob(os.path.join(folder_path, "*.*"))
    results = []

    for file in files:
        img = Image.open(file).convert('RGB').resize((224, 224))
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

        texture = extract_texture_features(img_tensor.cpu().numpy(), batch_size=1).to(device)
        frequency = extract_frequency_features(img_tensor.cpu().numpy()).to(device)
        edge = extract_edge_features(img_tensor.cpu().numpy()).to(device)
        geometric = extract_geometric_features(img_tensor.cpu().numpy()).to(device)

        inputs = {
            "image": img_tensor,
            "texture": texture,
            "frequency": frequency,
            "edge": edge,
            "geometric": geometric,
        }

        model.eval()

        # Grad-CAM 处理
        target_conv_layer = dict(model.named_modules())[target_layer]  # 选定目标卷积层
        gradcam = GradCAM(model, target_conv_layer)

        with torch.no_grad():
            outputs = model(**inputs)
            fake_prob = torch.softmax(outputs, dim=1)[0, 0].item()
            is_fake = fake_prob > 0.5

            # 计算 Grad-CAM
            outputs.backward()  # 反向传播以计算梯度
            cam = gradcam.generate_cam()

            # 可视化 Grad-CAM
            cam_image = gradcam.visualize(img, cam)

            # 保存或显示 Grad-CAM 图像
            plt.imshow(cam_image)
            plt.title(f"File: {os.path.basename(file)} - Fake Probability: {fake_prob:.4f}, Prediction: {'Fake' if is_fake else 'Real'}")
            plt.show()

            results.append([os.path.basename(file), fake_prob, int(is_fake)])

    # 保存结果到 CSV 文件
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Fake Probability", "Is Fake"])
        writer.writerows(results)
    print(f"Results saved to {output_csv}")


def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
            total_acc += (preds == labels).float().mean().item()

    total_loss /= len(data_loader) if criterion else 0
    total_acc /= len(data_loader)
    return total_loss, total_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="运行模式: train 或 test")
    parser.add_argument("--dataset_path", type=str, help="数据集文件夹路径 (train 模式必需)")
    parser.add_argument("--model_path", type=str, help="加载模型路径 (test 模式必需)")
    parser.add_argument("--save_model_path", type=str, default="checkpoints/best_model.pth", help="保存模型的路径")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="测试结果 CSV 文件路径")
    parser.add_argument("--test_input", type=str, help="测试输入文件或文件夹路径 (test 模式必需)")
    parser.add_argument("--epochs", type=int, default=10, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    args = parser.parse_args()

    train_or_test(
        mode=args.mode,
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        save_model_path=args.save_model_path,
        output_csv=args.output_csv,
        test_input=args.test_input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )