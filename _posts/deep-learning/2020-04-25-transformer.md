---
layout: post
title: 'Transformer'
date: 2020-04-25
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea28751c2a9a83be5467bc1.jpg'
tags: 深度学习
---

> Transformer，基于Multi-head self-attention的Seq2Seq模型.

- paper：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- code：[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

**Transformer**是一个基于[多头自注意力](https://0809zheng.github.io/2020/04/24/self-attention.html#3-multi-head-self-attention)(**Multi-head Self-Attention**)机制的模型，成为继多层感知机、卷积神经网络和循环神经网络之后又一个常用的深度学习模型。在原文中**Transformer**被提出用于进行[序列到序列](https://0809zheng.github.io/2020/04/21/sequence-2-sequence.html)(**Seq2Seq**)建模，并适用于机器翻译等任务。目前该模型也被广泛应用于其他自然语言处理以及计算机视觉等领域。

**Transformer**的基本结构如下图所示。
网络结构可以分成**编码器Encoder**和**解码器Decoder**两部分。根据不同的任务，有时候会用到不同的部分，如**编码器**部分常用于文本编码分类，**解码器**部分用于语言模型生成，完整的**编码器-解码器**结构用于机器翻译。

![](https://pic.imgdb.cn/item/618b94ea2ab3f51d91f6d24e.jpg)


```python
d_model = 512   # 词嵌入 Embedding 的维度
d_ff = 2048     # 前馈神经网络的隐藏层维度
d_k = d_v = 64  # K(=Q), V向量的维度 
n_layers = 6    # 编码器和解码器堆叠层数
n_heads = 8     # 自注意力头数

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):                         # enc_inputs: [batch_size, src_len]  
                                                                       # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)         # enc_outputs: [batch_size, src_len, d_model], 
                                                                       # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                       # dec_outpus    : [batch_size, tgt_len, d_model], 
                                                                       # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], 
                                                                       # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                      # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```

## 1. 网络结构

### ① 编码器

![](https://pic.imgdb.cn/item/60ebe1f55132923bf8acdd67.jpg)


编码器由$N$层模块堆叠而成(设置`n_layers=6`)。序列数据首先经过**词嵌入**(**embedding**)变换为词向量(长度为`d_model=512`)，与位置编码(**positional encoding**)相加后作为输入。

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)                     # 词嵌入
        self.pos_emb = PositionalEncoding(d_model)                               # 位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]   
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model], 
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```

由于输入序列中可能存在占位符等没有意义的**token**，因此使用`get_attn_pad_mask`函数生成注意力**mask**，在计算注意力时将这些位置置零。实现过程是首先找出这些位置(标记为$1$)，并在后续的注意力计算中将这些位置赋予一个较大的负值(如$-1e9$)，这样经过**softmax**函数后该位置就趋近于$0$。

```python
def get_attn_pad_mask(seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          # 判断占位符P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成多维度
```

编码器的每层模块包含两个子层，即一个[多头自注意力](https://0809zheng.github.io/2020/04/24/self-attention.html#3-multi-head-self-attention)(**Multi-head self-attention**)层和一个逐位置的前馈神经网络(**Feed Forward**)层：

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, d_model]
                                                                                      # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model], 
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]                                                                   
        enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
```

多头自注意力机制如下。基本的注意力计算采用缩放点积注意力，序列每个位置的**query, key, value**向量是由其自身(单头)或自身的线性变换(多头)表示的，因此称为“自”(**self**)注意力。其中**query, key**向量的长度为`d_k=64`，**value**向量的长度为`d_v=64`。

$$ \text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$

引入缩放因子$1/\sqrt{d_k}$的原因是**softmax**函数将输入的每一行规范化为概率分布，由于**softmax**函数对较大的数值比较敏感，数值较大的位置更有可能趋近于$1$，使得其他位置趋近于$0$，为了减少这种过度的“二值化”，对注意力计算的数值进行缩放。

由于单头的自注意力运算没有可学习参数，因此其表示能力受限。多头自注意力机制是指将输入序列映射到$h$个不同的子空间(设置`n_head=8`，应满足`n_head*d_k=n_model`)，在每个子空间中应用自注意力运算，将结果连接起来再映射回原空间中。这种做法类似于卷积网络中使用多个卷积核，使得模型具有$h$次机会倾向于学习合适的注意力关系，从而增强模型的表达能力。

多头自注意力机制后还应用了残差连接和[Layer Norm](https://0809zheng.github.io/2020/03/04/normalization.html#9-layer-normalization)。使用**LayerNorm**而不是**BatchNorm**的原因是，序列数据通常具有不同的长度，通过补$0$进行长度对齐。若在所有样本的某一个特征维度上进行标准化(**BatchNorm**)，其计算得到的均值和方差变化较大，不利于存储滑动平均值。而对每个样本的所有特征维度进行标准化(**LayerNorm**)则比较稳定。

![](https://pic.imgdb.cn/item/618b82b52ab3f51d91ec31ba.jpg)

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                             # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果是占位符P就等于 0 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        return self.layernorm(output + residual), attn
```

前馈神经网络层采用两层全连接层，全连接层作用于序列的每个位置，其中间特征维度为`d_ff=2048`。该层最后也使用了残差连接和**Layer Norm**：

$$ \text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2 $$

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)   # [batch_size, seq_len, d_model]  
```

### ② 解码器

![](https://pic.imgdb.cn/item/60ebe1f55132923bf8acdd67.jpg)


解码器也由$N$层模块堆叠而成(设置`n_layers=6`)。解码器采用自回归式的输入方式，即每次输入应为目标句子的一部分(右移**shifted right**的目标序列，初始为`[START]`)，经过词嵌入后与位置编码相加。在实践中可以对解码器的输入序列进行**mask**，即对每一个输入**token**，在计算注意力时**mask**掉其后所有**token**，使得每一个输入**token**只能和其之前的输入**token**交互，通过这种**mask**机制可以在一次前向传播过程中实现所有自回归过程。

```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                               # dec_inputs: [batch_size, tgt_len]
                                                                                          # enc_intpus: [batch_size, src_len]
                                                                                          # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                            # [batch_size, tgt_len, d_model]       
        dec_outputs = self.pos_emb(dec_outputs)                                           # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)                # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)            # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + 
                                       dec_self_attn_subsequence_mask), 0)                # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)                     # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                             # dec_outputs: [batch_size, tgt_len, d_model]
                                                              # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                              # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```

除了使用`get_attn_pad_mask`函数**mask**掉解码器输入和编码器输入中没有意义的占位符，还使用`get_attn_subsequence_mask`函数生成自回归的**mask**，表现为一个上三角矩阵(值为$1$即会被**mask**掉)。

```python
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]          # 注意力矩阵：QK^T
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask  
```

解码器的每层模块包含三个子层，即一个带掩码的多头自注意力层、一个多头自注意力层和一个逐位置的前馈神经网络层。其中带掩码的多头自注意力层将自回归**mask**应用到注意力计算中；多头自注意力层中的**query**来自前一个输出，**key, value**来自编码器的输出。

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask): # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                       # enc_outputs: [batch_size, src_len, d_model]
                                                                                       # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                       # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                 dec_inputs, dec_self_attn_mask)   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                enc_outputs, dec_enc_attn_mask)    # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                    # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn
```

### ③ 位置编码
自注意力机制无法捕捉位置信息，这是因为其计算注意力时的无序性，导致打乱任意顺序的序列其每个对应位置会得到相同的结果。通过引入位置编码把位置信息直接编码到输入序列中。

每个位置的位置编码也应具有长度`d_model=512`。作者使用一种三角形式的位置编码，使得每一位置的编码表示为之前位置编码的线性函数(三角函数的和差公式)。第$pos$位置的第$i$和$i+1$个编码表示为：

$$ PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}}) $$

在实践中由于词嵌入的数值相对于位置编码较小，因此将词嵌入的结果乘以$\sqrt{d_{model}}$后与位置编码相加。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) 
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)                      # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)
```

### ④ 模型比较

![](https://pic.imgdb.cn/item/618b97032ab3f51d91f83cfd.jpg)

上表展示了自注意力机制、循环网络、卷据网络以及一种受限的自注意力机制的计算性能对比。其中$n$是序列长度，$d$是序列每个**token**的特征维度(词嵌入维度)，$k$是(1d)卷积核尺寸，$r$表示对每个位置只计算其附近$r$个位置的注意力。
- **Complexity per Layer**：即每层的计算复杂度。循环网络和卷积网络的复杂度接近，与自注意力的复杂度比较主要取决于$n$和$d$的大小。
- **Sequential Operations**：所需等待的序列操作数，只有循环网络需要顺序执行(当前位置依赖于之前位置的计算)。
- **Maximum Path Length**：连接任意两位置所需路径长度的最大值。自注意力可以建立任意两个位置之间的关系；循环网络需要顺序遍历完整的序列才能建立全局关系；卷积网络受卷积核(感受野)限制，需要堆叠多层才能获得全局感受野。

## 2. 实验分析

### ① 网络设置
作者设计了几种不同大小的模型，如下表所示：

![](https://pic.imgdb.cn/item/618b994d2ab3f51d91f97b8a.jpg)

## ② 训练设置

训练集使用**WMT 2014**英语-德语数据集和英语-法语数据集。前者包含$450$万对句子，使用**byte-pair**编码句子，即按照划分词根进行编码，减少同一个单词不同时态造成的冗余。源域和目标域语言共享包含$37000$个**token**的词典。后者则更大，包含$3600$万对句子。

训练使用了$8$块**P100 GPU**。**base**模型每次训练耗时$0.4$秒，共进行了$10$万次训练，总耗时$12$小时。**big**模型每次训练耗时$1$秒，共进行了$30$万次训练，总耗时$3.5$天。

使用**Adam**优化器，基本参数$\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$。设置$warmup\_steps=4000$，学习率公式如下：

$$ lrate=d_{model}^{-0.5}\cdot \min (step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5}) $$

在每个子层的残差连接、编码器和解码器的词嵌入和位置编码相加处使用了**dropout**，设置$P_{drop}=0.1$。

设置$\epsilon_{ls}=0.1$的**label smoothing**降低学习难度，即当概率超过$0.1$时认为是对的结果(总类别数较多)。尽管这降低了模型预测的困惑度，但提高了准确率。

### 3. 实验结果
作者给出了在机器翻译任务上的模型表现：

![](https://pic.imgdb.cn/item/618b9d192ab3f51d91fb53d3.jpg)

使用**multi-head**机制，既可以捕捉到近距离依赖关系，又可以捕捉到远距离依赖关系；且模型具有较好的可解释性。由于计算得到每一个**token**与其他所有**token**的自注意力，因此可以定量衡量不同**token**之间的相关性程度。下图展示了两个句子，其每个句子的每个**token**（此处为单词）与句子中其他单词之间的相关性：

![](https://pic.downk.cc/item/5ea2a8e3c2a9a83be570cfb4.jpg)

