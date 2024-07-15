# CNN卷积神经网络及其变种

## 1.卷积所具备的性质

$$
\int_{-\infty}^{\infty}f\left(\tau\right)g\left(x-\tau\right)d\tau 
$$

- 平移不变性：只关注**局部特征**，在图像识别中可以理解为不依赖于图像位置，若图中只有数字1，那么无论左上角还是右上角都没有影响。
- **参数共享**：卷积核的权重在整个输入数据上共享，减少了模型的参数数量。

## 2.卷积层

- 对输入和卷积核权重进行互相关运算，添加标量偏置后产生输出，我们训练的是卷积核的权重以及标量的偏置。我们随机初始化卷积核权重。

- **步幅**：卷积核滑动的距离，会影响卷积运算的得到的特征图大小

- 填充：填充使得输出特征图的尺寸与输入数据相同。通常在每一边添加 ((k-1)/2) 个像素（对于奇数大小的卷积核），其中 (k) 是卷积核的尺寸。

- 计算公式：
	$$
	[ H_{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1 ]\\ [ W_{out} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1 ]
	$$
	
	

- 多通道运算：检测边缘，获得更多特征
	每个卷积层和相应的卷积核进行互相关运算，然后相同位置相加得到特征图。
	 输入通道数等于卷积核通道数，输出通道数等于卷积核个数

## 3.池化层

- 减少空间维度，提取主要特征，防止过拟合
- 最大池化运算，找最大值
- 平均池化操作，求平均值得到特征图   
- 计算特征图大小一致，卷积核的高宽换成感受野的高宽

## 4.项目实战——fashion_mnist

补充：F1分数，精确率（查准率）和召回率（查全率）的调和平均数
$$
[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} ]
$$
<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240709101532548.png" alt="image-20240709101532548" style="zoom: 50%;" />

```
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))  # 假设FashionMNIST数据集已经预处理为[0,1]区间
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.286,), (0.353,))
])

# 导入dataset
train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)

# 导入dataloader
train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x

# 实例化模型、定义损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 计算宏平均F1分数的辅助函数
def calculate_macro_f1_score(predictions, true_labels, num_classes=10):
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(num_classes):
        tp = (predictions == i).sum().item()
        fp = (predictions == i).sum().item() - tp
        fn = (true_labels == i).sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
    macro_f1_score = np.mean(f1_scores)
    return macro_f1_score

# 训练模型
num_epochs = 10
loss_list = []
accuracy_list = []
macro_f1_score_list = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct_preds = 0
    total_samples = 0
    for i,(images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_preds += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        if (i+1) % 200 == 0:
            print(f"第{i:.1f}批,Loss:{loss:.4f}")
    
    epoch_loss /= len(train_dataloader)
    loss_list.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {correct_preds / total_samples:.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    total_preds = []
    total_labels = []
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_preds.extend(predicted.view(-1).cpu().numpy())
        total_labels.extend(labels.view(-1).cpu().numpy())
    accuracy = correct_preds / total_samples
    macro_f1_score = calculate_macro_f1_score(np.array(total_preds), np.array(total_labels))
    print(f'Test Accuracy: {accuracy:.4f}, Test Macro F1 Score: {macro_f1_score:.4f}')

# 绘制训练过程
plt.figure(figsize=(10, 6))

plt.plot(range(1, num_epochs + 1), loss_list, marker='o', linewidth=2, label='Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.title('Training Progression')
plt.grid(True, linestyle='-', alpha=0.5)
plt.tight_layout()
plt.show()
```



## 5.卷积网络变种

nn.MaxPool2d(kernel_size=2, stride=2) 池化后特征图缩小一半一半 16\*16   8\*8

卷积核1*1实现不改变尺寸，但控制通道数

最大池化要写步幅不然默认为感受野的大小

### 5.1 LeNet-5

LeNet-5是Yann Lecun等人于1998年提出的卷积神经网络模型，被广泛应用于手写数字识别等任务。LeNet-5的结构相对简单，包含了两个卷积层和三个全连接层。卷积层使用了5x5的卷积核，并通过 引入了非线性。池化层则使用了2x2的平均池化操作，降低了特征图的尺寸。LeNet-5通过在卷积层和全连接层之间交替使用卷积和池化操作，从而实现了对输入图像的特征提取和分类。

最开始用于处理灰度图

**现代卷积网络更喜欢使用RELU函数**

<img src="C:/Users/张奕霖/AppData/Roaming/Typora/typora-user-images/image-20240710153925708.png" alt="image-20240710153925708" style="zoom:33%;" />

```python
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 输入通道数为1（灰度图像），输出通道数为6，卷积核大小为5x5
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化层，池化核大小为2x2，步幅为2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层，输入大小为16 * 5 * 5，输出大小为120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入大小为120，输出大小为84
        self.fc3 = nn.Linear(84, num_classes)  # 全连接层，输入大小为84，输出大小为num_classes（默认10）

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))  # 通过第一个卷积层并应用Sigmoid激活函数
        x = self.pool(x)  # 通过第一个池化层
        x = torch.sigmoid(self.conv2(x))  # 通过第二个卷积层并应用Sigmoid激活函数
        x = self.pool(x)  # 通过第二个池化层
        x = x.view(-1, 16 * 5 * 5)  # 展平张量
        x = torch.sigmoid(self.fc1(x))  # 通过第一个全连接层并应用Sigmoid激活函数
        x = torch.sigmoid(self.fc2(x))  # 通过第二个全连接层并应用Sigmoid激活函数
        x = self.fc3(x)  # 通过第三个全连接层
        return x
```



### 5.2 AlexNet

AlexNet是由Alex Krizhevsky等人于2012年提出的卷积神经网络模型，是第一个在ImageNet图像识别比赛中取得优胜的模型。AlexNet相比LeNet-5更加深层，并包含了多个卷积层和全连接层。AlexNet使用了ReLU激活函数，并引入了Dropout和数据增强等技术，从而进一步提高了模型的性能和鲁棒性。

用来处理RGB三通道图像。

**使用RELU函数，多了三卷积层，使用最大池化，使用Dropout层正则化，使用数据增强（翻转，随机裁剪）**，使用LRN正则化多gpu并行计算

Dropout层：随机丢弃，正则化，避免过拟合

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240710154002515.png" alt="image-20240710154002515" style="zoom: 50%;" />

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### 5.3 VGG

VGGNet是由Karen Simonyan和Andrew Zisserman于2014年提出的卷积神经网络模型，以其深度和简洁的结构而闻名。VGGNet通过多次堆叠3x3的卷积层和2x2的最大池化层来提取特征，从而增加了网络的深度。VGGNet的结构相对一致，使得其更加容易理解和扩展。6种结构，以VGG-16为例子。

优势：**更深的网络结构，统一使用3x3的卷积核（多个同样的卷积核堆叠），使用较小的步长（通常为1）和2x2的最大池化窗口（保留更多特征，分辨率更高）**

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240710154051325.png" alt="image-20240710154051325" style="zoom:33%;" />

```python
"""
import torchvision.models as models
# 加载预训练的VGG16模型
model = models.vgg16(pretrained=True)
"""

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

```

**补充：模型太深会出现梯度爆炸，进行权重初始化**

```python
# 权重设置为0
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    else 
```

### 5.4 GoogleNet

GoogLeNet是由Google团队于2014年提出的卷积神经网络模型，以其高效的网络结构而闻名。GoogLeNet引入了"Inception"模块，通过并行使用不同尺寸的卷积核和池化操作，并在网络中进行了混合，从而在保持网络深度较浅的同时，提高了网络的感受野和特征提取能力。

**优势：**

- 1x1 卷积进行降维，减少计算量和参数量；
- 由多个 Inception 模块（**调节该模块超参数以控制输出通道数**）堆叠而成，形成一个深度为 22 层的网络。
- 在中间层引入了两个辅助分类器，帮助梯度传播和正则化
- 参数量和计算量显著低于 VGG，且性能更好
- **通道合并**
- **全局平均池化**：全局平均池化将每个特征图的所有元素取平均值，从而将每个特征图缩减为一个单一的数值。这种操作可以显著减少参数量，并且有助于防止过拟合。
	**局限：**信息丢失，适用性有限，局部特征不敏感，过度平滑，只适用于googlenet和resnet，对噪声敏感，
	梯度爆炸或消失
- Inception
  <img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240710154141472.png" alt="image-20240710154141472" style="zoom: 50%;" />

```python
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.p1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
    def forward(self, x):
        p1 = self.relu(self.p1(x))
        p2 = self.relu(self.p2(x))
        p3 = self.relu(self.p3(x))
        p4 = self.relu(self.p4(x))
        return torch.cat([p1, p2, p3, p4])
```

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240710154235030.png" alt="image-20240710154235030" style="zoom:33%;" />

```python
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self)__init__.()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            
            nn.Dropout(0.4),
            nn.Linear(1024, 1000)
        )
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = nn.Linear(1024, 1000)(x)
        return x
```

### 5.5 ResNet(重要)

ResNet是由Kaiming He等人于2015年提出的卷积神经网络模型，以其深层的结构和"残差"连接而闻名。ResNet通过引入"跳跃连接"，允许网络中的信息在跳过一些层时保持不变，从而解决了深层网络中的梯度消失和网络退化问题。ResNet的结构更加深层，从而可以进一步提高网络的性能。

优势：**引入残差块解决退化问题（误差先减小后变大）**，进行批量规范化（BN），跨层输入，模型的泛化能力，在残差块中将输入值x和经过卷积的数值相加（二者尺寸一致）。

<img src="C:/Users/张奕霖/AppData/Roaming/Typora/typora-user-images/image-20240711093439943.png" alt="image-20240711093439943" style="zoom:50%;" />

BN层（批量规范化）：以往只在数据清洗（数据输入前进行规范化或归一化），现在在任意一层，任意一批进行规范化。

1.求每批次均值；2.求每批次的方差；

3.归一化             $\hat{x_{i}}\leftarrow\frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}$；

**4.尺寸偏移**               $y_i\leftarrow\gamma\hat{x_i}+\beta=\mathrm{BN}_{\gamma,\beta}(x_i)$ 最后返回新引入的俩参数

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240711100147092.png" alt="image-20240711100147092" style="zoom:33%;" />

 

```
import torch
import torch.nn as nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_1=False,stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if use_1:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride)
        else:
            self.conv3 = None
    
    def forward(self, x):
        identity = x  # Save the input for the shortcut connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3 is not None:
            identity = self.conv3(identity)  # Apply the 1x1 convolution to the input
        out += identity  # Add the shortcut connection
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            ResidualBlock(64, 64, use_1=False,stride=1),
            ResidualBlock(64, 64, use_1=False,stride=1)
        )
        self.b3 = nn.Sequential(
            ResidualBlock(64, 128, use_1=True,stride=2),
            ResidualBlock(128, 128, use_1=False,stride=1)
        )
        self.b4 = nn.Sequential(
            ResidualBlock(128, 256, use_1=True,stride=2),
            ResidualBlock(256, 256, use_1=False,stride=1)
        )
        self.b5 = nn.Sequential(
            ResidualBlock(256, 512, use_1=True,stride=2),
            ResidualBlock(512, 512, use_1=False,stride=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(ResidualBlock).to(device)
    print(summary(model, (3, 224, 224)))
```

