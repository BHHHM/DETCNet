import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 定义LeNet模型（只保留卷积层部分）
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

# 图像预处理函数
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    return transform(image).unsqueeze(0).to(device)

# 特征可视化函数（改进版）
def visualize_and_save_features(feature_maps, save_dir):
    feature_maps_normalized = feature_maps.cpu().detach().numpy()[0]
    # 归一化特征图以增强对比度
    feature_maps_normalized = (feature_maps_normalized - feature_maps_normalized.min()) / \
                              (feature_maps_normalized.max() - feature_maps_normalized.min())

    # 对特征图进行高斯模糊
    blur_kernel_size = (7,7)  # 高斯模糊的核大小
    feature_maps_blurred = []
    for feature_map in feature_maps_normalized:
        blurred_map = cv2.GaussianBlur(feature_map, blur_kernel_size, 0)
        feature_maps_blurred.append(blurred_map)
    feature_maps_blurred = np.array(feature_maps_blurred)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存并显示每个通道的特征图
    num_features = feature_maps_blurred.shape[0]
    for i in range(num_features):
        # 保存特征图
        save_path = os.path.join(save_dir, f"feature_map_{i}.png")
        plt.imsave(save_path, feature_maps_blurred[i], cmap='jet')

# 主函数
if __name__ == "__main__":
    # 定义包含图片的目录
    pic_dir = r'F:\AAABHM\GuiLin\tool\hunhe_40'

    # 获取目录中的所有图片文件
    image_files = [f for f in os.listdir(pic_dir) if f.startswith('final_blended_image_') and f.endswith('.jpg')]

    # 检查是否正好有四张图片
    if len(image_files) != 4:
        print(f"Error: Expected 4 images, but found {len(image_files)} images.")
        exit()

    # 定义保存特征图的基目录
    base_save_dir = r'F:\AAABHM\GuiLin\tool\feature_maps2_2'

    # 加载模型
    model = FeatureExtractor().to(device)

    # 遍历每张图片
    for image_file in image_files:
        # 完整的图片路径
        image_path = os.path.join(pic_dir, image_file)

        # 创建保存特征图的目录（基于图片名称）
        save_dir = os.path.join(base_save_dir, os.path.splitext(image_file)[0])

        # 处理图片并提取特征
        img = process_image(image_path)
        features = model(img)

        # 可视化并保存特征图（应用高斯模糊）
        visualize_and_save_features(features, save_dir)

        print(f"Features for {image_file} saved to {save_dir}")