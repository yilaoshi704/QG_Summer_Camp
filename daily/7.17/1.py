import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)
print(device)

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取标签信息
data_dir = 'D:/data'  # 修改为您的数据集路径
train_csv_path = os.path.join(data_dir, 'train.csv')
train_df = pd.read_csv(train_csv_path)

# 划分训练集和验证集
np.random.seed(42)  # 确保结果可复现
train_df = train_df.sample(frac=1).reset_index(drop=True)  # 打乱数据
split_index = int(0.9 * len(train_df))  # 90%用于训练，10%用于验证
train_df, valid_df = train_df[:split_index], train_df[split_index:]

# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = CustomDataset(dataframe=train_df,
                              root_dir=os.path.join(data_dir, 'train_images'),  # 修改为您的训练图像文件夹路径
                              transform=train_transform)
valid_dataset = CustomDataset(dataframe=valid_df,
                              root_dir=os.path.join(data_dir, 'train_images'),  # 修改为您的训练图像文件夹路径
                              transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 修改输出层为5个类别
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)

# 使用 Adam 优化器
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # 学习率可以调整

# 训练模型
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            except Exception as e:
                print(f"Error during training: {e}")
                break

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 验证模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Validation Accuracy: {accuracy:.2f}%')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), './output/best_model.pth')

# 调用训练函数
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10)
