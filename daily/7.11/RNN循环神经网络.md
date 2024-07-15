# RNN循环神经网络

## 1.简单介绍

### RNN特点

前馈神经网络不考虑数据之间的关联性，网络的输出只和当前时刻网络的输入相关。然而在解决很多实际问题的时候我们发现，现实问题中存在着很多序列型的数据（文本、语音以及视频等），现实场景如室外的温度是随着气候的变化而周期性的变化的，以及我们的语言也需要通过上下文的关系来确认所表达的含义。

这些**序列型的数据**往往都是具有时序上的关联性的，**既某一时刻网络的输出除了与当前时刻的输入相关之外，还与之前某一时刻或某几个时刻的输出相关**。而前馈神经网络并不能处理好这种关联性，因为它没有记忆能力，所以前面时刻的输出不能传递到后面的时刻。

因此，就有了现在的循环神经网络，其本质是：**拥有记忆的能力**，并且会根据这些记忆的内容来进行推断。因此，它的输出就依赖于当前的输入和记忆。相比于前馈神经网络，该网络内部具有很强的记忆性，它可以利用内部的记忆来处理任意时序的输入序列。

![img](https://img-blog.csdnimg.cn/0c06937e7d2e4d4eaaa53534323ed256.png)

由上图可见：一个典型的 RNN 网络架构包含一个输入，一个输出和一个神经网络单元 。和普通的前馈神经网络的区别在于：RNN 的神经网络单元不但与输入和输出存在联系，而且**自身也存在一个循环** / 回路 / 环路 / 回环 (loop)。这种回路允许信息从网络中的一步传递到下一步。

![img](https://img-blog.csdnimg.cn/38576e9805dd4913bf7c4878fbd4b00a.png)

 以上架构不仅揭示了 RNN 的实质：**上一个时刻的网络状态将会作用于（影响）到下一个时刻的网络状态，还表明 RNN 和序列数据密切相关。同时，RNN 要求每一个时刻都有一个输入，但是不一定每个时刻都需要有输出。**

![img](https://img-blog.csdnimg.cn/19a99b88abb14d6b840d5a075f03a660.png)。



圆形的箭头表示隐藏层的自连接。在RNN中，每一层都共享参数U、V、W，降低了网络中需要学习的参数，提高学习效率。

### RNN类型

1. **One-to-One（单输入单输出）**：
	- **应用**：**图像分类**等任务。
	- **描述**：输入一个固定大小的向量，输出一个固定大小的向量。
2. **One-to-Many（单输入多输出）**：
	- **应用**：**图像描述生成**（Image Captioning）。
	- **描述**：输入一个固定大小的向量，输出一个序列。例如，输入一张图片，输出一段描述这张图片的文字。
3. **Many-to-One（多输入单输出）**：
	- **应用**：情感分析（Sentiment Analysis）、**文本分类**。
	- **描述**：输入一个序列，输出一个固定大小的向量。例如，输入一段文字，输出一个情感标签（正面或负面）。
4. **Many-to-Many（多输入多输出）**：
	- **应用**：**机器翻译**（Machine Translation）、视频分类。
	- **描述**：输入一个序列，输出一个序列。例如，输入一句话的英文，输出对应的中文翻译。
5. **Many-to-Many（多输入多输出，时间步对齐）**：
	- **应用**：**视频帧标签**（Video Frame Labeling）、命名实体识别（Named Entity Recognition）。
	- **描述**：输入一个序列，输出一个序列，且输入和输出的序列长度相同。例如，输入一个视频的每一帧，输出每一帧的标签。

## 2.RNN变种

### LSTM——long shoty-term memory

#### 1.特点

- 弥补了RNN长期记忆能力不足的缺点

- 引入门控机制，输入门，遗忘门，输出门

- 交叉熵损失，F_nll_loss(F.log_softmax(x, dim=1))

<img src="C:/Users/张奕霖/AppData/Roaming/Typora/typora-user-images/image-20240711203405207.png" alt="image-20240711203405207" style="zoom:50%;" />

#### 2.核心

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240711203653894.png" alt="image-20240711203653894" style="zoom:33%;" />

实现记忆功能的线，用门选择是否让信息通过。

门：sigmoid和矩阵点乘

w权重，h~t-1~前一刻状态，同理x~t~当前输入，b~t~偏置，$\sigma$sigmoid函数

输入门：$ [ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) ]$

遗忘门：$ [ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) ]$

输出门：$ [ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) ]$与tanh的结果相乘放到记忆功能的线，输出之前的，现在的，和隐藏状态。

3.API

```
torch.nn.LSTM(input_size,hidden_size,num_layers,batch_first,bidirectional,dropout)
```

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240711210636967.png" alt="image-20240711210636967" style="zoom: 50%;" />

**LSTM 的输入：input，（h_0，c_0）**

input：输入数据，shape 为（句子长度seq_len, 句子数量batch, 每个单词向量的长度input_size）；
**隐藏状态形状**h_0：默认为0，shape 为（num_layers * num_directions单向为1双向为2, batch, 隐藏层节点数hidden_size）；
**细胞状态形状**c_0：默认为0，shape 为（num_layers * num_directions, batch, hidden_size）；

**LSTM 的输出：output，（h_n，c_n）** 

output：输出的 shape 为（seq_len, batch, num_directions * hidden_size）；
h_n：shape 为（num_layers * num_directions, batch, hidden_size）；
c_n：shape 为（num_layers * num_directions, batch, hidden_size）

#### 补充：Embedding

将离散数据映射为连续变量，捕捉潜在关系——将文本转换为连续向量，基于分布式假设捕捉语义信息。

```
Image Embedding（图像嵌入）
定义与目的：图像嵌入是将图像转换为低维向量，以简化处理并保留关键信息供机器学习使用。
方法与技术：利用深度学习模型（如CNN）抽取图像特征，通过降维技术映射到低维空间，训练优化嵌入向量。
应用与优势：图像嵌入广泛应用于图像分类、检索等任务，提升模型性能，降低计算需求，增强泛化能力。
```

```
Word Embedding（词嵌入）
定义与目的：词嵌入是将单词映射为数值向量，以捕捉单词间的语义和句法关系，为自然语言处理任务提供有效的特征表示。
方法与技术：词嵌入通过预测单词上下文（如Word2Vec）或全局词频统计（如GloVe）来学习，也可使用深度神经网络捕捉更复杂的语言特征。
应用与优势：词嵌入广泛应用于文本分类、机器翻译等自然语言处理任务，有效提升模型性能，因其能捕捉语义信息和缓解词汇鸿沟问题。
```

```python
torch.nn.Embedding(num_embeddings,  # 词典单词数
                   embedding_dim,  # 你希望将每个词映入几维的向量
                   padding_idx=None, 
                   max_norm=None, 
                   norm_type=2.0, 
                   scale_grad_by_freq=False, 
                   sparse=False, 
                   _weight=None, 
                   _freeze=False, 
                   device=None, 
                   dtype=None)
```

#### 3.实战：IMBD文本情感分类

