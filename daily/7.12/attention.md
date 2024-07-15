# attention注意力机制

注意力机制(Attention Mechanism)是机器学习中的一种数据处理方法，广泛应用在自然语言处理、图像识别以及语音识别等各种不同类型的机器学习任务中。注意力机制对不同信息的关注程度（重要程度）由**权值**来体现，注意力机制可以视为**查询矩阵**(Query)、**键**(key)以及**加权平均值**构成了多层感知机(Multilayer Perceptron, MLP)。

注意力的思想，类似于寻址。给定Target中的某个元素Query，通过计算Query和各个Key的**相似性**或**相关性**，得到每个Key对应Value的**权重系数**，然后对Value进行加权求和，即得到最终的Attention数值。所以，本质上Attention机制是Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。

K == V

## 1.三个阶段

### 一阶段

计算query（查询对象）和key相关性，公示如下：

点积$Similarity(Query,Key_i)=Query*Key_i$

cosin相似性$Similarity(Query,Key_i)=\frac{Query*Key_i}{||Query||*||Key_i||}$

MLP网络$Similarity(Query,Key_i)=MLP(Query,Key_i)$



![在这里插入图片描述](https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/f76f7fff5a6949508cfb86d8a35d8a64.png)

### 二阶段

使用softmax将得到的值归一化。

$a_i =Softmax(Sim_i )=\frac{e^{Sim_i}}{\sum_{j=1}^{L_x}e^{Sim_j}}$

### 三阶段

根据得到的权重系数对Value进行加权求和，得到注意力数值。

$Attention(Query,Source)=\sum_{i=1}^{L_x}a_i*Value_i$

ps：进行softmax之前要进行缩放，避免出现极端情况。

Q K V均通过输入特征后X训练出相应的权重矩阵后矩阵乘法得到。

## 2.自注意力机制—注意力机制的子集

简单理解为只关注输入或者只关注输出的特殊注意力计算机制。

**自注意力机制的查询和键则都是来自于同一组的元素 K\==V\==Q**

相较于RNN和LSTM拥有并行计算和句法特征和语意特征

## 3.掩码自注意力机制

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240712210059733.png" alt="image-20240712210059733" style="zoom:33%;" />



## 4.多头（h）自注意力机制

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240712210410716.png" alt="image-20240712210410716" style="zoom:33%;" />

将输入X拆分成h块，分别进行注意力的过程，最后得到多个Z进行线性变换后拼接

## 补充：为什么需要编码

ATTENTION缺点：

1.开销变大；2.无位置关系（顺序关系）

<img src="https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240712212025688.png" alt="image-20240712212025688" style="zoom:25%;" />

位置编码