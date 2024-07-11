# Pytorch

感知机是只有输入输出的神经网络，若干输入一个输出

线性定义
$$
\begin{aligned}&\mathrm{f(x1+x2)=y1+y2}\\&\mathrm{f(kx1)=ky1}\end{aligned}
$$

### 1.张量 与numpy多维数组类似

1.约等于numpy多维数组

2..permute和.transpose方法均为交换维度

3.torch自带自己的数据类型，tensor([],dtype=torch.int32)

4.x.add_(y)会就地修改x+=y

5.使用GPU计算Tensor

```python
import torch

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")
# 创建一个张量
x = torch.randn(3, 3)
# 将张量移动到 GPU
x = x.to(device)
print("Tensor on GPU:", x)
# 创建一个简单的模型
model = torch.nn.Linear(3, 3)
# 将模型移动到 GPU
model = model.to(device)
# 将输入张量传递给模型
output = model(x)
print("Output on GPU:", output)
```

### 2.梯度下降

设置.requires_grad为True，或者.requires_grad_(True) 

反向传播前先充值梯度optimizer.zero_grad()  # 清零梯度

输出.backward()求其梯度，损失函数一般为标量，若为向量需要添加其他参数

## 3.torchAPI

```python

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# nn.Module
class Lr(nn.Module):
    def __init__(self):
        super(Lr, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        out = self.linear(x)
        return out 

model = Lr().to(device)
criterion = nn.MSELoss() # 均方误差
# optim优化器，使用随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn([500, 1], requires_grad=False).to(device) # 保持数据在同一设备
y = x * 3 + 0.8

num = 1000
losses = []
for n in range(num):
    output = model(x) # 预测值
    loss = criterion(output, y) # 计算损失函数
    optimizer.zero_grad() # 清空梯度
    loss.backward() # 反向传播找梯度
    optimizer.step() # 调用优化器更新参数
    losses.append(loss.item())
    if (n % 100) == 0:
        print(f"loss is {loss.item():.4f}")

for p in model.parameters():
    print(p)
# 绘制损失变化曲线
plt.plot(losses)
plt.xlabel('n')
plt.ylabel('Loss')
plt.title('Loss over n')
plt.show()

# model.train()
模型设置为训练模式。这会启用 Dropout 和 BatchNorm 层的训练行为
# model.eval()
将模型设置为评估模式。这会禁用 Dropout 层，并使 BatchNorm 层使用累积的均值和方差。

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在评估模式下，不需要计算梯度
    output = model(x)
    eval_loss = criterion(output, y)
    print(f"Evaluation Loss: {eval_loss.item():.4f}")
```

## 4.常见优化算法

### 梯度下降BGD

- 全局最优，把全部样本扔进去计算。

### 随机梯度下降

- 随机抽出一个样本，进行梯度计算。
- API: torch.optim.SGD()

### 小批量梯度下降MBGD

- 单个样本可能存在噪声（偏离），随机抽取一小部分数据来进行梯度计算然后取得平均值。

### 动量法

- MBGD可能存在梯度震荡，即在最优点附近徘徊，故学习率很难挑选

- 该方法基于梯度移动的加权平均 

- $$
	\begin{array}{l}v=0.8v+0.2\nabla w\\w=w-\alpha v\end{array},
	$$

- 用前一次的梯度加权然后加上本次梯度乘加权值的积

### AdaGrad

- 动态学习率,\delta 一般为极小值10^-7

- $$
	gradent=history\_gradent+(\nabla w)^2\\w=w-\frac\alpha{\sqrt{gradent}+\delta}\nabla w
	$$

### RMSProp

- 上面二者结合,加权学习率

- $$
	\begin{aligned}&gradent=0.8*history_gradent+0.2*(\nabla w)^2\\&w=w-\alpha\frac{\nabla w}{\sqrt{gradent}+\delta}\end{aligned}
	$$

### Adam

- 结合

- $$
	\$$
	\begin{array}{l}
	\text{一阶矩估计的衰减率} \ (\beta_1) \ \text{（通常取 0.9）} \\
	\text{二阶矩估计的衰减率} \ (\beta_2) \ \text{（通常取 0.999）} \\
	\text{防止除零的小常数} \ (\epsilon) \ \text{（通常取 } 10^{-8} \text{）}
	\end{array}
	\$$
	$$

- $$
	\$$
	\begin{aligned}
	&\text{时间步} \ ( t \leftarrow t + 1 ) \\
	&\text{计算梯度} \ ( g_t ) \ \text{（即损失函数对参数的梯度）} \\
	&\text{更新一阶矩估计：} \ [ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t ] \\
	&\text{更新二阶矩估计：} \ [ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 ] \\
	&\text{进行偏差校正：} \ [ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} ] \ [ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} ] \\
	&\text{更新参数：} \ [ \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} ]
	\end{aligned}
	\$$
	$$

## 5.torch数据加载

准备好dataset实例，然后把dataset交给dataloader打乱顺序，组成batch

### 数据集类

- torch.utils.data.Dataset,继承这个基类，加载数据

- ```python
	# 继承datatset
	class MyDataset(Dataset):
		def __init__(self):
	        self.lines = open(data.path).readlines()
	        
	    def __getitem__(self, index):
	        # 根据索引返回值
	        cur_line = self.lines[index].strip()
	        label = cur_line[:4].strip()
	        content = cur_line[4:].strip()
	    
	    def __len__(self):
	        return len(self.lines)
	    
	"""
	from torch.utils.data import Dataset
	
	class MyDataset(Dataset):
	    def __init__(self, data, labels):
	        self.data = data
	        self.labels = labels
	
	    def __len__(self):
	        return len(self.data)
	
	    def __getitem__(self, idx):
	        return self.data[idx], self.labels[idx]
	"""
	    
	if __name__ == "__main__":
	    dataset = MyDataset()
	    print(dataset[0])
	```

### 数据加载器类

划分数据集，向上取整

```python
from torch.utils.data import DataLoader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True，drop_last=True)
```

### 自带数据集

torchvision：图像

```python
from torchvision.datasets import MNIST
minst = MNIST(root='./data', train=True, download=False)
print(minst[0][0].show())

```

torchtext：文本

## 项目：BP神经网络实现手写数字识别

### 1.思路

a.准备数据 （加载和清洗）
b.构建模型
c.训练
d.保存
e.评估

### 2.准备训练集

图像变矩阵变tensor（三维）  012    201

ps：你处理的是图像，那就只需要记住[ 高，宽，通道数 ]；你定义了一个[c, n, m]的矩阵，那就只需要记住[ 层，行，列 ]

#### 2.1torchvision.transforms图像数据处理

- 把取值为[0,255]的PIL.Image或者shape为(H,W,C)(高宽通道数)的numpy.ndarray，转换成形状为[C,H,W],取值范围为[0,1.0]的torch.FloatTensor,使用torchvison.transforms.ToTensor()的方法
- 黑白图片通道数为1，彩色图通道数为3(RGB),每个像素取值均为[0,255]

- torchvision.transforms.Normalize(mean, std)归一化处理
- transforms.Compose组合，依次进行处理

2.2准备数据集的dataset和dataloader

### 3.构建模型

* 激活函数 import torch.nn.functional  as F
	x = F.relu(x)

* 关注每一层数据的形状

* 交叉熵计算损失,有效处理概率损失计算问题
	$$
	[ L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)] ]
	$$

### 4.模型的训练

- 实例化模型，设置训练模式
- 实例化优化器，实例化优化函数
- 获取遍历dataloader
- 设置梯度
- 前向传播
- 计算损失
- 反向传播
- 更新参数
- 循环

### 5.模型的保存和加载

#### 保存

```python
torch.save(mnist_net.state_dict(),"model/mnist_net.pk1")
torch.save(optimizer.state.dict(),"results/mnist_optimizer.pk1")
```

#### 加载

```python
mnist_net.load_state_dict(torch.load("model/mnist_net.pk1"))
optimizer.load_state_dict(torch.load("results/mnist_optimizer.pk1"))
```

### 6.模型评估

- 无需计算梯度
- 收集损失和准确率，计算平均损失和平均准确率
- 计算损失方法相同
- 准确率的计算
	- 模型输出为[batch_size, 10]的形状
	- 最大值位置是预测的目标值（softmax后验概率，分母相同，分子越大，概率越大）
	- torch.argmax获取最大值及其位置
	- 返回最大值位置之后和真实值进行对比，相同则表示预测成功



## 补充评估

在机器学习和深度学习中，评估模型性能的手段有很多，F1分数是其中之一。不同的评估指标适用于不同的任务和场景。以下是一些常见的评估指标及其适用场景：

### 1. **准确率（Accuracy）**

- **定义**：准确率是正确预测的样本数占总样本数的比例。
- **公式**：[ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} ]
- **适用场景**：适用于类别分布均衡的分类任务。

### 2. **精确率（Precision）**

- **定义**：精确率是正确预测的正样本数占所有预测为正样本的样本数的比例。

- **公式**：
	$$
	[ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} ]
	$$
	
- **适用场景**：适用于关注假阳性（False Positive）较少的场景，如垃圾邮件检测。

### 3. **召回率（Recall）**

- **定义**：召回率是正确预测的正样本数占所有实际为正样本的样本数的比例。

- **公式**：
	$$
	[ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} ]
	$$
	
- **适用场景**：适用于关注假阴性（False Negative）较少的场景，如疾病检测。

### 4. **F1分数（F1 Score）多分类要用MACRO_f1**

- **定义**：F1分数是精确率和召回率的调和平均数。

- **公式**：
	$$
	[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} ]
	$$
	
- **适用场景**：适用于类别不平衡的分类任务，综合考虑精确率和召回率。

- ```python
	def f1(predicted_labels, true_labels):
	    predicted_labels = predicted_labels.cpu().numpy()
	    true_labels = true_labels.cpu().numpy()
	    tp = np.sum(np.logical_and(predicted_labels == 1, true_labels == 1))
	    fp = np.sum(np.logical_and(predicted_labels == 1, true_labels == 0))
	    fn = np.sum(np.logical_and(predicted_labels == 0, true_labels == 1))
	    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
	    return f1_score
	```

	

### 5. **ROC曲线和AUC（Area Under Curve）**

- **定义**：ROC曲线是以假阳性率（FPR）为横轴，真阳性率（TPR）为纵轴绘制的曲线。AUC是ROC曲线下的面积。
- **适用场景**：适用于评估二分类模型的整体性能，特别是在类别不平衡的情况下。

### 6. **混淆矩阵（Confusion Matrix）**

- **定义**：混淆矩阵是一个表格，用于描述分类模型的性能。它显示了实际类别和预测类别的分布。
- **适用场景**：适用于详细分析分类模型的性能，特别是多分类任务。

### 7. **平均绝对误差（Mean Absolute Error, MAE）**

- **定义**：MAE是预测值与实际值之差的绝对值的平均数。

- **公式**：

- $$
	[ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| ]
	$$

- 

- **适用场景**：适用于回归任务，衡量预测值与实际值的平均偏差。

### 8. **均方误差（Mean Squared Error, MSE）**

- **定义**：MSE是预测值与实际值之差的平方的平均数。

- **公式**：
	$$
	[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 ]
	$$
	
- **适用场景**：适用于回归任务，衡量预测值与实际值的平均偏差，较大误差的影响更大。

### 9. **均方根误差（Root Mean Squared Error, RMSE）**

- **定义**：RMSE是MSE的平方根。

- **公式**：
	$$
	[ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} ]
	$$
	
- **适用场景**：适用于回归任务，衡量预测值与实际值的平均偏差，较大误差的影响更大。

### 10. **R²（决定系数）**

- **定义**：R²表示模型解释变量总变异的比例。

- **公式**：
	$$
	[ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}*i)^2}{\sum*{i=1}^{n} (y_i - \bar{y})^2} ]
	$$
	
- **适用场景**：适用于回归任务，衡量模型的解释能力。
