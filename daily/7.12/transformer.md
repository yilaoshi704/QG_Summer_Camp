# Transformer

## 优缺点

优点：
1.效果好
2.可以并行训练，速度快
3.很好地解决了长距离依赖的问题
缺点：
1.完全基于self-attention，对于词语位置之间的信息有一定的丢失，虽然加入了positional encoding来解决这个问题，但也还存在着可以优化的地方

<img src="https://img-blog.csdnimg.cn/20200324202147216.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 80%;" />。

## Transformer模型的基本工作流程包括以下步骤：

1. 输入处理：将输入文本转换为向量，添加位置信息。（套用模型）

2. Encoder编码器工作流程：多头注意力、残差连接与层归一化。

	![image-20240712224004579](https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/image-20240712224004579.png)

3. Decoder解码器工作流程：带掩码的多头注意力、前馈网络。

	![img](https://img-blog.csdnimg.cn/a4069a2544024ddbb3f9561adbd6b2ea.png)

	![img](https://yilaoshi.oss-cn-guangzhou.aliyuncs.com/picture/038419d2d9d94a78954f9283e211dee4.webp)

4. 输出处理：获取句子的编码信息矩阵

  

  



## 注意事项

**解码器掩码：确保解码器在计算自注意力时，只能使用当前位置之前的词，而不能看到当前位置之后的词，以保持因果关系，防止信息泄漏**

**残差连接：防止模型退化**

**位置编码：attention无位置关系**

批量归一化：数据更加稳定

decoder的输入：一种是训练输入，一种是预测输入



## 代码

```py
import torch
import torch.nn as nn

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.transformer = nn.Transformer(d_model, nhead, nlayers, nlayers, nhid, dropout)
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, src, src_mask)
        output = self.decoder(output)
        return output

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 示例参数
ntokens = 1000  # 词汇表大小
d_model = 512  # 嵌入维度
nhead = 8  # 注意力头数
nhid = 2048  # 前馈网络隐藏层维度
nlayers = 6  # Transformer层数
dropout = 0.5  # Dropout概率

model = TransformerModel(ntokens, d_model, nhead, nhid, nlayers, dropout)

```

