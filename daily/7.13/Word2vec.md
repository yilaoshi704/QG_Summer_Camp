# Word2vec

api：

```python
import gensim
import torch
import torch.nn as nn
# 使用gensim训练Word2Vec模型
sentences = [["hello", "world"], ["goodbye", "world"]]
model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
# 获取词汇表和嵌入矩阵
vocab = model.wv.key_to_index
embedding_matrix = model.wv.vectors
# 将嵌入矩阵转换为PyTorch嵌入层
embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
# 示例：获取单词"hello"的嵌入
word_idx = vocab["hello"]
word_embedding = embedding_layer(torch.tensor([word_idx]))
print(word_embedding)

```

Word2Vec是一种用于获取词嵌入（词向量）的技术，通过将文本中的单词映射到高维空间中的实数向量，从而捕捉单词之间的语义关系。

Skip-gram模型通过输入词来预测上下文的词，而CBOW模型则相反，通过上下文的词来预测输入词。

在Word2Vec算法中，损失函数的选择对于模型的训练效果至关重要。通常使用的损失函数包括交叉熵损失函数和层次Softmax损失函数。同时，优化器的选择也会影响模型训练的速度和效果，常用的优化器包括随机梯度下降（SGD）和Adam优化器。

## 1.one-hot

One-hot编码是一种将分类变量转换为数值形式的方法。在这种编码中，每个类别都由一个唯一的二进制向量表示，其中除了一个位置是1之外，其余所有位置都是0。

### 特点

- **稀疏性**：One-hot编码通常会产生一个非常稀疏的向量，因为大多数位置都是0。
- **无序性**：One-hot编码不包含任何关于类别之间顺序的信息，即它不表示任何类别的优先级或顺序。
- **维度高**：如果分类变量有N个可能的值，那么One-hot编码将生成一个N维的向量。

### 示例

假设我们有一个简单的词汇表，包含三个词："apple", "banana", "cherry"。使用One-hot编码，我们可以这样表示这三个词：

- "apple" → [1, 0, 0]
- "banana" → [0, 1, 0]
- "cherry" → [0, 0, 1]

### 优点

- 简单易懂，实现容易。
- 可以避免模型学习到错误的数值顺序关系。

### 缺点

- 维度灾难：当类别数量非常大时，One-hot编码会导致向量维度非常高，这会增加计算复杂度和内存消耗。
- 稀疏性：大多数位置都是0，这可能导致模型训练效率低下。

### 改进

——word2vec

## 2.CBOW

### 基本步骤

读取语料，统计词频信息
构建词典，并初始化Huffman树以及随机初始化每个词的对应向量（维度默认是200）
以行为单位训练模型（输入文件都在一行上，会按照最大1000个词切割为多行）
获取当前行中的一个输入样本（当前词向量以及相邻几个的词的词向量）
累加上下文词向量中每个维度的值并求平均得到投影层向量X(w)（对应代码中的neu1）
遍历当前词到根节点（输出层的Huffman树）经过的每个中间节点
计算中间节点对应的梯度 g * 学习速率（与中间节点的权重向量 syn1 和投影层向量 neu1 相关）
刷新投影层到该中间节点的误差向量（与梯度和中间节点向量相关）
刷新中间结点向量（与梯度和投影层向量相关）
刷新上下文词向量（其实就是将误差向量累加到初始向量中）

### 三层结构

输入层：包含context(w)中个词的词向量，其中，表示单词的向量化表示函数，相当于此函数把一个个单词转化成了对应的向量化表示(类似one-hot编码似的)，表示上下文取的总词数，表示向量的维度；
投影层：将输入层的个向量做累加求和；
输出层：按理我们要通过确定的上下文决定一个我们想要的中心词，但怎么决定想要的中心词具体是  中的哪个呢？
通过计算各个可能中心词的概率大小，取概率最大的词便是我们想要的中心词，相当于是针对一个N维数组进行多分类，但计算复杂度太大，所以输出层改造成了一棵Huffman树，以语料中出现过的词当叶子结点，然后各个词出现的频率大小做权重



## 3. Skip-gram



## 4.代码

```
import gensim
import torch
import torch.nn as nn

# 使用gensim训练Word2Vec模型
sentences = [["hello", "world"], ["goodbye", "world"]]
model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取词汇表和嵌入矩阵
vocab = model.wv.key_to_index
embedding_matrix = model.wv.vectors

# 将嵌入矩阵转换为PyTorch嵌入层
embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

# 示例：获取单词"hello"的嵌入
word_idx = vocab["hello"]
word_embedding = embedding_layer(torch.tensor([word_idx]))
print(word_embedding)

```



