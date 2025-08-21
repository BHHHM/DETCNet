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
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 5, 1, 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        return x

# 图像预处理函数
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    return transform(image).unsqueeze(0).to(device)

# 特征可视化函数
def visualize_features(feature_maps):
    feature_maps = feature_maps.cpu().detach().numpy()[0]
    num_features = feature_maps.shape[0]
    cols = 8
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_features:
                ax = axes[i, j]
                ax.imshow(feature_maps[idx], cmap='jet')
                ax.axis('off')
            else:
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

# 主函数
if __name__ == "__main__":
    pic_dir = r'F:\AAABHM\GuiLin\tool\final_blended_image_1.jpg'
    img = process_image(pic_dir)
    model = FeatureExtractor().to(device)
    features = model(img)
    visualize_features(features)