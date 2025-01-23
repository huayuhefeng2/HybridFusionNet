import os
import shutil
import pandas as pd

def separate_images_by_class(csv_file, image_folder, class_0_folder, class_1_folder):
    # 支持的图片格式
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # 读取CSV文件
    df = pd.read_csv(csv_file, header=None)
    
    # 确保文件夹存在
    os.makedirs(class_0_folder, exist_ok=True)
    os.makedirs(class_1_folder, exist_ok=True)
    
    # 遍历每一行，获取文件名和类别
    for index, row in df.iterrows():
        filename = row[0]  # 文件名（去掉扩展名）
        label = row[1]     # 类别 0 或 1
        
        # 查找图片的完整路径（支持多种格式）
        moved = False
        for ext in valid_extensions:
            image_path = os.path.join(image_folder, filename + ext)
            if os.path.exists(image_path):
                # 根据类别移动文件
                if label == 0:
                    shutil.move(image_path, os.path.join(class_0_folder, filename + ext))
                elif label == 1:
                    shutil.move(image_path, os.path.join(class_1_folder, filename + ext))
                moved = True
                break
        
        if not moved:
            print(f"图片 {filename} 的图片未找到，跳过该文件")

# 示例使用
csv_file = 'cla_gt_C1.csv'  # 替换为你的 CSV 文件路径
image_folder = './assets'          # 替换为存放图片的文件夹路径
class_0_folder = './assets/0' # 替换为类别0的文件夹路径
class_1_folder = './assets/1' # 替换为类别1的文件夹路径

separate_images_by_class(csv_file, image_folder, class_0_folder, class_1_folder)
