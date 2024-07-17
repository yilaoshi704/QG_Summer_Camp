import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_dataloader(train=True):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root="./data", train=train, transform=transform_train, download=True)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)  # 增大 batch_size 以更好利用 GPU
    return data_loader


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28) # 每一个像素（28*28）都是一个特征
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_net = network().to(device)  # 将模型移动到 GPU
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)

# 检查模型和优化器状态文件是否存在
model_path = "model/mnist_net.pt"
optimizer_path = "results/mnist_optimizer.pt"

if os.path.exists(model_path) and os.path.exists(optimizer_path):
    mnist_net.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
    print("模型和优化器状态已加载。")
else:
    print("未找到模型和优化器状态文件，将进行训练。")

train_dataloader = get_dataloader(True)


def train(epoch):
    losses = []
    for i, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
        output = mnist_net(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 100 == 0:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            torch.save(mnist_net.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f'Epoch {epoch}, i: {i}/{len(train_dataloader)}, loss: {loss:.4f}')
    return sum(losses) / len(losses)


def test():
    mnist_net.eval()  # 设置模型为评估模式
    loss_total = []
    accuracy_total = []
    test_dataloader = get_dataloader(False)
    with torch.no_grad():  # 评估时无需进行梯度计算
        for i, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = mnist_net(data)
            cur_loss = F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # 获得该维度最大值
            cur_accuracy = pred.eq(target.view_as(pred)).sum().item() # 转化为标量
            loss_total.append(cur_loss)
            accuracy_total.append(cur_accuracy)
    avg_loss = np.sum(loss_total) / len(test_dataloader.dataset)
    avg_accuracy = np.sum(accuracy_total) / len(test_dataloader.dataset)
    print(f"平均损失率: {avg_loss:.4f}, 平均准确率: {avg_accuracy:.4f}")


if __name__ == "__main__":
    if os.path.exists(model_path) and os.path.exists(optimizer_path):
        test()
    else:
        all_losses = []
        for i in range(3):
            epoch_losses = train(i)
            all_losses.append(epoch_losses)

        # 绘制损失率图像
        plt.figure(figsize=(10, 5))
        plt.plot(all_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()

