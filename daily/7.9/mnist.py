# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据预处理
# 定义训练集和测试集的图像变换流程
# 包括随机水平翻转、尺寸调整、转换为张量以及标准化
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])  # 使用MNIST的标准均值和方差进行归一化

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载MNIST数据集
# 应用预处理变换加载训练集和测试集
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

# 创建数据加载器
# 通过DataLoader为训练集和测试集创建批次加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)  # 批次大小为64，随机洗牌
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)  # 批次大小为32，不洗牌


# 定义CNN模型
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # 第一层卷积
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # 卷积层，输入通道1，输出通道16，3x3滤波器
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，2x2窗口
        )

        # 第二层卷积
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层，输入通道16，输出通道32
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，2x2窗口
        )

        # 全连接层
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),  # 输入尺寸为32x8x8，输出节点数64
            nn.ReLU(),  # 激活函数
            nn.Dropout(p=0.5),  # 50%的丢弃率
            nn.Linear(64, 10)  # 输出节点数10，对应10个数字类别
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(-1, 32 * 8 * 8)  # 扁平化特征
        x = self.fc_layer(x)
        return x


# 实例化模型、定义损失函数和优化器
# 将模型放在GPU上运行，如果可用
model = MNIST_CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 计算F1分数的辅助函数
def calculate_f1_score(predicted_labels, true_labels):
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    tp = np.sum(np.logical_and(predicted_labels == 1, true_labels == 1))  # 真阳性
    fp = np.sum(np.logical_and(predicted_labels == 1, true_labels == 0))  # 假阳性
    fn = np.sum(np.logical_and(predicted_labels == 0, true_labels == 1))  # 假阴性
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确度
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # F1分数
    return f1_score


# 训练模型
# 迭代训练过程，每个epoch遍历整个训练集，并在测试集上计算准确率和F1分数
num_epochs = 5  # 总共训练5个周期
total_step = len(train_loader)  # 总步骤数
loss_list = []  # 记录训练损失
accuracy_list = []  # 记录测试准确率
f1_score_list = []  # 记录测试F1分数

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # 图像数据到设备
        labels = labels.to(device)  # 标签数据到设备

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        # 每10步打印一次损失值
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    # 计算测试集上的准确率和F1分数
    with torch.no_grad():
        correct = 0
        total = 0
        f1_scores = []
        model.eval()  # 将模型设置为评估模式
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1_scores.append(calculate_f1_score(predicted, labels))
        accuracy = correct / total  # 测试集准确率
        f1_score = np.mean(f1_scores)  # 平均F1分数
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}')
        accuracy_list.append(accuracy)
        f1_score_list.append(f1_score)


plt.figure(figsize=(10, 6))

plt.plot(loss_list, marker='o', linewidth=2, label='Training Loss')
plt.plot(accuracy_list, marker='o', linewidth=2, label='Test Accuracy')
plt.plot(f1_score_list, marker='o', linewidth=2, label='F1 Score')

plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss/F1 Score')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.title('Training Progression')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
