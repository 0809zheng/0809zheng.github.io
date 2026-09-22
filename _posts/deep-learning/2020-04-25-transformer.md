---
layout: post
title: 'Transformer及其结构改进'
date: 2020-04-25
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-000-5ea28751.jpg'
tags: 深度学习
---

> Transformer and Its Architectural Evolution.

循环神经网络沿时间步串行传播状态，卷积神经网络则依靠固定局部窗口逐层扩大感受野。**Transformer**把序列建模改写为全局的信息路由：任意位置都可以在一层内直接读取其他位置，再用逐位置的非线性变换更新表示。它由此获得更短的信息路径和更高的训练并行度，也付出了注意力复杂度随序列长度平方增长、缺少固有顺序归纳偏置等代价。

最初的**Transformer**是用于机器翻译的编码器—解码器模型，随后演化出编码器、解码器与编码器—解码器三种主干。围绕这套骨架的结构改进主要回答五个问题：注意力内部怎样形成并组合路由；残差支路如何在深层网络中稳定传播；前馈网络如何提高参数利用率；注意力与前馈网络应如何排列、共享和裁剪；以及现代解码器为何普遍采用**Pre-RMSNorm、RoPE、GQA**与**SwiGLU**。

本文目录：
1. 原始**Transformer**：网络结构、实验设置与结果分析
2. **Transformer**的结构优化
  - 2.1 归一化与残差路径
  - 2.2 前馈网络与局部计算单元
  - 2.3 注意力通信、层组织与弹性深度
3. 哪些改进真正改变了**Transformer**

**符号约定**：输入序列长度为$n$，模型维度为$d_{\text{model}}$，注意力头数为$h$，单头的键和值维度分别为$d_k,d_v$，前馈网络中间维度为$d_{\text{ff}}$。输入表示记为$$X\in\mathbb{R}^{n\times d_{\text{model}}}$$，查询、键和值矩阵记为$$Q,K,V$$。为简化表达，公式中的偏置项常被省略。

# 1. 原始**Transformer**

**Transformer**的基本结构如下图所示。
网络结构可以分成**编码器Encoder**和**解码器Decoder**两部分。根据不同的任务，有时候会用到不同的部分，如**编码器**部分常用于文本编码分类，**解码器**部分用于语言模型生成，完整的**编码器-解码器**结构用于机器翻译。

![](pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-001-transformer.png)


<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

## 1. 网络结构

### ① 编码器

编码器由$N$层模块堆叠而成(设置`n_layers=6`)。序列数据首先经过**词嵌入**(**embedding**)变换为词向量(长度为`d_model=512`)，与位置编码(**positional encoding**)相加后作为输入。

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

由于输入序列中可能存在占位符等没有意义的**token**，因此使用`get_attn_pad_mask`函数生成注意力**mask**，在计算注意力时将这些位置置零。实现过程是首先找出这些位置(标记为$1$)，并在后续的注意力计算中将这些位置赋予一个较大的负值(如$-1e9$)，这样经过**softmax**函数后该位置就趋近于$0$。

<details markdown="1">
  <summary>点击展开代码</summary>

```python
def get_attn_pad_mask(seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          # 判断占位符P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成多维度
```
</details>

编码器的每层模块包含两个子层，即一个[多头自注意力](https://0809zheng.github.io/2020/04/24/self-attention.html#3-multi-head-self-attention)(**Multi-head self-attention**)层和一个逐位置的前馈神经网络(**Feed Forward**)层：

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

多头自注意力机制如下。基本的注意力计算采用缩放点积注意力，序列每个位置的**query, key, value**向量是由其自身(单头)或自身的线性变换(多头)表示的，因此称为“自”(**self**)注意力。其中**query, key**向量的长度为`d_k=64`，**value**向量的长度为`d_v=64`。

$$ \text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$

引入缩放因子$1/\sqrt{d_k}$的原因是**softmax**函数将输入的每一行规范化为概率分布，由于**softmax**函数对较大的数值比较敏感，数值较大的位置更有可能趋近于$1$，使得其他位置趋近于$0$，为了减少这种过度的“二值化”，对注意力计算的数值进行缩放。

由于单头的自注意力运算没有可学习参数，因此其表示能力受限。多头自注意力机制是指将输入序列映射到$h$个不同的子空间(设置`n_head=8`，应满足`n_head*d_k=n_model`)，在每个子空间中应用自注意力运算，将结果连接起来再映射回原空间中。这种做法类似于卷积网络中使用多个卷积核，使得模型具有$h$次机会倾向于学习合适的注意力关系，从而增强模型的表达能力。

多头自注意力机制后还应用了残差连接和[Layer Norm](https://0809zheng.github.io/2020/03/04/normalization.html#9-layer-normalization)。使用**LayerNorm**而不是**BatchNorm**的原因是，序列数据通常具有不同的长度，通过补$0$进行长度对齐。若在所有样本的某一个特征维度上进行标准化(**BatchNorm**)，其计算得到的均值和方差变化较大，不利于存储滑动平均值。而对每个样本的所有特征维度进行标准化(**LayerNorm**)则比较稳定。


![](pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-002-mha.png)

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

前馈神经网络层采用两层全连接层，全连接层作用于序列的每个位置，其中间特征维度为`d_ff=2048`。该层最后也使用了残差连接和**Layer Norm**：

$$ \text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2 $$

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

### ② 解码器

解码器也由$N$层模块堆叠而成(设置`n_layers=6`)。解码器采用自回归式的输入方式，即每次输入应为目标句子的一部分(右移**shifted right**的目标序列，初始为`[START]`)，经过词嵌入后与位置编码相加。在实践中可以对解码器的输入序列进行**mask**，即对每一个输入**token**，在计算注意力时**mask**掉其后所有**token**，使得每一个输入**token**只能和其之前的输入**token**交互，通过这种**mask**机制可以在一次前向传播过程中实现所有自回归过程。

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

除了使用`get_attn_pad_mask`函数**mask**掉解码器输入和编码器输入中没有意义的占位符，还使用`get_attn_subsequence_mask`函数生成自回归的**mask**，表现为一个上三角矩阵(值为$1$即会被**mask**掉)。

<details markdown="1">
  <summary>点击展开代码</summary>

```python
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]          # 注意力矩阵：QK^T
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask  
```
</details>

解码器的每层模块包含三个子层，即一个带掩码的多头自注意力层、一个多头自注意力层和一个逐位置的前馈神经网络层。其中带掩码的多头自注意力层将自回归**mask**应用到注意力计算中；多头自注意力层中的**query**来自前一个输出，**key, value**来自编码器的输出。

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

### ③ 位置编码
自注意力机制无法捕捉位置信息，这是因为其计算注意力时的无序性，导致打乱任意顺序的序列其每个对应位置会得到相同的结果。通过引入位置编码把位置信息直接编码到输入序列中。

每个位置的位置编码也应具有长度`d_model=512`。作者使用一种三角形式的位置编码，使得每一位置的编码表示为之前位置编码的线性函数(三角函数的和差公式)。第$pos$位置的第$i$和$i+1$个编码表示为：

$$ PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}}) $$

在实践中由于词嵌入的数值相对于位置编码较小，因此将词嵌入的结果乘以$\sqrt{d_{model}}$后与位置编码相加。

<details markdown="1">
  <summary>点击展开代码</summary>

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
</details>

### ④ 模型比较

![](pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-003-compare.png)

上表展示了自注意力机制、循环网络、卷据网络以及一种受限的自注意力机制的计算性能对比。其中$n$是序列长度，$d$是序列每个**token**的特征维度(词嵌入维度)，$k$是(1d)卷积核尺寸，$r$表示对每个位置只计算其附近$r$个位置的注意力。
- **Complexity per Layer**：即每层的计算复杂度。循环网络和卷积网络的复杂度接近，与自注意力的复杂度比较主要取决于$n$和$d$的大小。
- **Sequential Operations**：所需等待的序列操作数，只有循环网络需要顺序执行(当前位置依赖于之前位置的计算)。
- **Maximum Path Length**：连接任意两位置所需路径长度的最大值。自注意力可以建立任意两个位置之间的关系；循环网络需要顺序遍历完整的序列才能建立全局关系；卷积网络受卷积核(感受野)限制，需要堆叠多层才能获得全局感受野。

## 2. 实验设置

训练集使用**WMT 2014**英语-德语数据集和英语-法语数据集。前者包含$450$万对句子，使用**byte-pair**编码句子，即按照划分词根进行编码，减少同一个单词不同时态造成的冗余。源域和目标域语言共享包含$37000$个**token**的词典。后者则更大，包含$3600$万对句子。

训练使用了$8$块**P100 GPU**。**base**模型每次训练耗时$0.4$秒，共进行了$10$万次训练，总耗时$12$小时。**big**模型每次训练耗时$1$秒，共进行了$30$万次训练，总耗时$3.5$天。

使用**Adam**优化器，基本参数$\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$。设置$warmup\_steps=4000$，学习率公式如下：

$$ lrate=d_{model}^{-0.5}\cdot \min (step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5}) $$

在每个子层的残差连接、编码器和解码器的词嵌入和位置编码相加处使用了**dropout**，设置$P_{drop}=0.1$。

设置$\epsilon_{ls}=0.1$的**label smoothing**降低学习难度，即当概率超过$0.1$时认为是对的结果(总类别数较多)。尽管这降低了模型预测的困惑度，但提高了准确率。

### 3. 实验结果
作者给出了在机器翻译任务上的模型表现：

![](pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-005-result.png)

使用**multi-head**机制，既可以捕捉到近距离依赖关系，又可以捕捉到远距离依赖关系；且模型具有较好的可解释性。由于计算得到每一个**token**与其他所有**token**的自注意力，因此可以定量衡量不同**token**之间的相关性程度。下图展示了两个句子，其每个句子的每个**token**（此处为单词）与句子中其他单词之间的相关性：

![](pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-004-attn-vis.png)

# 2. Transformer的结构优化

## 2.1 归一化与残差路径

残差连接使每层只需学习对当前表示的增量，归一化则控制激活尺度。二者的位置关系决定梯度是否拥有一条不经过归一化和非线性变换的恒等路径，是深层**Transformer**最关键的结构选择之一。

### (1) **Post-LN**与**Pre-LN**

原始**Post-LN**为

$$
x_{l+1}=\operatorname{LN}(x_l+F_l(x_l)).
$$

每层输出都被重新归一化，最终表示通常无需额外的输出归一化。但反向传播必须连续穿过各层归一化，在深层网络中更依赖学习率预热和谨慎初始化。

### ⚪ **Pre-LN**：把归一化移到残差分支之前
- **paper**：[**Transformers without Tears: Improving the Normalization of Self-Attention**](https://arxiv.org/abs/1910.05895)

**Pre-LN**把结构改为

$$
x_{l+1}=x_l+F_l(\operatorname{LN}(x_l)),
$$

并在网络末端增加一次归一化。梯度可以沿$x_l\rightarrow x_{l+1}$的恒等支路直接传播，因而通常更容易扩深，对预热的依赖也更弱。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-normalization-009-transformer-norm-positions.png)

### ⚪ **LayerNorm理论分析**：解释**Pre-LN**为何更易训练
- **paper**：[**On Layer Normalization in the Transformer Architecture**](https://arxiv.org/abs/2002.04745)

均值场分析表明，初始化时**Post-LN**靠近输出层的梯度可能很大，而**Pre-LN**的梯度尺度更稳定。这解释了为什么学习率预热对原始结构重要，也解释了现代大模型普遍采用预归一化。

但“更易优化”不等于“最终质量一定更好”。把**Pre-LN**递推展开可得

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l(\operatorname{LN}(x_l)).
$$

若残差流范数随深度持续增长，而每个归一化后分支的输出尺度近似固定，那么单层增量相对于$x_l$会越来越小，深层容易接近恒等更新；有研究把这描述为有效深度不足。在训练充分、初始化合适时，**Post-LN**也可能取得更好的最终性能。两者的取舍应区分梯度能否传播、各层是否得到充分更新以及最终质量，而不能只比较“是否训崩”。更完整的变体见[归一化方法](https://0809zheng.github.io/2020/03/04/normalization.html#26-%E5%BD%92%E4%B8%80%E5%8C%96%E5%9C%A8transformer%E4%B8%AD%E7%9A%84%E4%BD%8D%E7%BD%AE)。

### (2) 简化归一化与残差门控

### ⚪ **RMSNorm**：只按均方根缩放
- **paper**：[**Root Mean Square Layer Normalization**](https://arxiv.org/abs/1910.07467)

层归一化同时减去均值并除以标准差：

$$
\operatorname{LN}(x)
=\gamma\odot\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}+\beta.
$$

**RMSNorm**去掉中心化，只保留尺度归一化：

$$
\operatorname{RMSNorm}(x)
=\gamma\odot
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}.
$$

它减少统计量与逐元素操作，并保留对整体缩放的不变性。现代解码器常采用**Pre-RMSNorm**。

### ⚪ **ReZero**：用零初始化残差门保持初始恒等映射
- **paper**：[**ReZero is All You Need: Fast Convergence at Large Depth**](https://arxiv.org/abs/2003.04887)

**ReZero**为每个残差分支加入可学习标量$\alpha_l$：

$$
x_{l+1}=x_l+\alpha_lF_l(x_l),\qquad \alpha_l=0.
$$

网络初始化时严格等于恒等映射，训练再逐步打开各层。它在受控任务中展示了极深网络的可训练性，也说明“先让残差支路接近零”是稳定深层网络的一条通用原则。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-normalization-016-rezero.png)

**ReZero**可以移除归一化，但并未取代现代语言模型中的**RMSNorm**。零门控会改变早期优化动力学，其千层结果也主要来自较小任务，不能直接外推到大规模预训练。

### ⚪ **GTrXL**：用门控残差控制状态写入
- **paper**：[**Stabilizing Transformers for Reinforcement Learning**](https://arxiv.org/abs/1910.06764)

**GTrXL(Gated Transformer-XL)**先把归一化移到子层输入，使恒等支路不经过归一化，再用类似**GRU**的门控替代无条件相加。其更新可以写为

$$
\begin{aligned}
r &= \sigma(W_ry+U_rx) \\
z &= \sigma(W_z y + U_z x - b_g) \\
\hat{h} &= \tanh(W_g y + U_g (r \odot x)) \\ 
y&=(1-z)\odot x+z\odot \tilde{h}
\end{aligned}
$$

把门偏置初始化为偏向$x$，可让网络开始时接近恒等映射，之后再学习何时写入新信息。它与**ReZero**共享“先保护捷径、再逐步开启残差分支”的原则，但门是逐元素且依赖输入的，表达力和开销都更高。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-006-gtrxl.png)

该方法主要解决强化学习中长时序、非平稳目标带来的优化困难；实验优势不能直接外推到大规模语言建模。它的重要启示是残差相加本身也可以被视为一个需要参数化的信息写入操作。

### (3) 残差尺度与初始化

即使归一化位置不变，残差分支的初始尺度也会随深度累积。若各层增量近似独立且方差相当，残差流方差会随层数增长；若分支过大，单次更新会剧烈扰动网络，若分支过小，深层又可能长期接近恒等映射。初始化因此不仅要保持单个矩阵的方差，还要控制整条残差路径的累计更新量。

### ⚪ **Admin**：按实测方差校准残差依赖
- **paper**：[**Understanding the Difficulty of Training Transformers**](https://arxiv.org/abs/2004.08249)

**Admin(adaptive model initialization)**把训练不稳定归因于残差分支的**放大效应(amplification effect)**：上层参数的微小变化可能经残差依赖被逐层放大。它先用一次前向统计估计各分支输出方差，再为捷径设置层相关系数

$$
x_{l+1}=\omega_l x_l+F_l(x_l),
$$

使初始化时各层对最终表示的依赖更均衡。与固定按层数缩放相比，$$\omega_l$$利用了具体模型和数据的实测信号；代价是初始化流程更复杂，且训练后还需把系数吸收到参数中。

**T-Fixup、Admin、DeepNorm**等方法表面上分别修改初始化、捷径或归一化，核心都在控制“每层增量相对残差流有多大”。相关初始化推导见[权重初始化](https://0809zheng.github.io/2020/03/05/initialization.html#24-%E6%AE%8B%E5%B7%AE%E7%BD%91%E7%BB%9C%E7%9A%84%E5%88%9D%E5%A7%8B%E5%8C%96)。

### (4) 深层网络的稳定化

### ⚪ **NormFormer**：在**Pre-LN**内部补充归一化
- **paper**：[**NormFormer: Improved Transformer Pretraining with Extra Normalization**](https://arxiv.org/abs/2110.09456)

**Pre-LN**虽然稳定，却可能出现不同深度的梯度幅度失衡。**NormFormer**在注意力输出后加入层归一化，对每个注意力头引入可学习缩放，并在前馈网络第一层激活后再次归一化。这些额外操作改善了中小规模模型的困惑度与下游性能，但会增加约数个百分点的训练开销。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-007-normformer.png)

### ⚪ **DeepNorm**：用深度相关缩放训练千层网络
- **paper**：[**DeepNet: Scaling Transformers to 1,000 Layers**](https://arxiv.org/abs/2203.00555)

**DeepNorm**用随深度变化的系数放大恒等捷径，并配套缩小残差分支的参数初始化，使每次参数更新引起的模型输出变化保持有界：

$$
x_{l+1}=\operatorname{LN}(\alpha x_l+F_l(x_l)).
$$

不同的编码器、解码器和编码器—解码器结构对应不同$\alpha$与初始化系数。该方法把**Post-LN**扩展到最高$1000$层，说明深度本身并非不可突破；但其配方依赖结构和层数，工程采用度远低于**Pre-RMSNorm**。

### (5) 注意力分数的归一化

残差流稳定并不保证注意力分数稳定。随着模型变大，查询与键的范数可能持续增长，使$$QK^\top$$进入大幅值区，注意力熵下降、训练对精度和学习率更敏感。此时提高数值精度只能推迟溢出，不能消除分数尺度本身不断增长的问题；直接约束查询和键更接近根因。

### ⚪ **QK-Norm**：归一化查询和键并学习分数尺度
- **paper**：[**Query-Key Normalization for Transformers**](https://arxiv.org/abs/2010.04245)

原始**QK-Norm**对每个头的查询与键做$L_2$归一化，并用可学习缩放$\tau$代替固定的$1/\sqrt{d_k}$：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(
\tau\frac{Q}{\lVert Q\rVert}
\frac{K^\top}{\lVert K\rVert}
\right)V.
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-normalization-034-vit22b.png)

现代模型中“**QK-Norm**”也常指在查询、键投影之后施加**LayerNorm**或**RMSNorm**。两者目标相同，都是控制分数尺度。

## 2.2 前馈网络与局部计算单元

前馈网络不混合位置，却决定每个词元能进行多复杂的通道变换。原始**ReLU FFN**只有一条上投影支路；后来的门控结构增加第二条支路，用一个内容相关的门选择要写回残差流的信息。少数结构还在注意力投影附近加入局部卷积，把序列归纳偏置嵌入基本模块。

### (1) 从硬阈值到平滑激活

### ⚪ **GELU**：用输入幅值进行平滑随机门控
- **paper**：[**Gaussian Error Linear Units**](https://arxiv.org/abs/1606.08415)

**GELU**定义为

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

其中$\Phi$是标准高斯分布的累积分布函数。与**ReLU**的硬阈值不同，它按输入幅值平滑地缩放激活，后来成为**BERT**等编码器的默认激活。

### ⚪ **Primer EZ**：平方激活与局部深度卷积
- **paper**：[**Primer: Searching for Efficient Transformers for Language Modeling**](https://arxiv.org/abs/2109.08668)

**Primer**通过架构搜索得到两项可以独立移植的改动。第一项把前馈激活替换为

$$
\operatorname{SquaredReLU}(x)=\max(x,0)^2,
$$

用平方进一步放大强激活并压低弱激活；第二项在查询、键和值投影之后沿序列维加入核宽为$3$的逐通道因果卷积，为每个通道提供显式局部混合。只保留这两项的简化结构称为**Primer EZ**。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-008-primer.png)

两项修改解决的问题不同：平方激活改变通道非线性，深度卷积增加局部序列归纳偏置。后者会改变注意力投影前后的数据流，也不再是完全依赖全局注意力的纯模块。论文在自回归语言建模中报告了更好的训练计算效率。

### (2) 门控前馈网络

### ⚪ **GLU**：用一条分支控制另一条分支
- **paper**：[**Language Modeling with Gated Convolutional Networks**](https://arxiv.org/abs/1612.08083)

**GLU(Gated Linear Unit)**最初用于卷积语言模型，其核心形式为

$$
\operatorname{GLU}(x)
=(xW_v)\odot\sigma(xW_g).
$$

一条线性支路提供内容，另一条支路产生门值。把它放入**Transformer FFN**后，还需用输出投影映射回模型维度。

### ⚪ **GEGLU与SwiGLU**：门控前馈网络的系统比较
- **paper**：[**GLU Variants Improve Transformer**](https://arxiv.org/abs/2002.05202)

门控前馈网络可以统一写为

$$
\operatorname{FFN}_{\text{gated}}(x)
=W_o\left[\phi(xW_g)\odot(xW_v)\right],
$$

其中**GEGLU**令$$\phi=\operatorname{GELU}$$，**SwiGLU**令$$\phi=\operatorname{Swish/SiLU}$$。实验表明这些变体在相近计算量下通常优于普通**ReLU/GELU FFN**。

门控结构有三个大矩阵，而普通前馈网络只有两个。为保持参数量与计算量近似不变，中间维度通常从$$4d_{\text{model}}$$缩小到约$$\frac{8}{3}d_{\text{model}}$$，并进一步取硬件友好的倍数。直接保持原宽度会增加约$50\%$的前馈参数。**SwiGLU**已成为现代解码器的常见选择。

## 2.3 注意力通信、层组织与弹性深度

标准模块的各注意力头独立计算，并按“注意力→前馈网络”串行堆叠具有独立参数的固定层数。结构改进可以分别打通头间或层间路由、循环共享同一转换、重排或并行子层，以及让执行深度可裁剪或动态变化。

### (1) 深度递归与参数共享

### ⚪ **Universal Transformer**：沿深度循环应用同一转换
- **paper**：[**Universal Transformers**](https://arxiv.org/abs/1807.03819)

**Universal Transformer**在“深度时间”上反复应用共享的注意力与前馈转换，并可通过**Adaptive Computation Time**让不同位置执行不同次数。它结合了自注意力的全局交互与循环网络的深度递归，在理论上具有更强的算法归纳偏置。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-009-universal-transformer.png)

参数共享显著减少存储，却不会同比减少计算；循环层之间还存在顺序依赖，动态停止也不利于批量并行。因此它更适合研究迭代推理。

### ⚪ **ALBERT**：分解嵌入并跨层共享参数
- **paper**：[**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**](https://arxiv.org/abs/1909.11942)

**ALBERT**把词嵌入维度$E$与隐藏维度$H$解耦，用$$V\times E$$和$$E\times H$$两个矩阵替代巨大的$$V\times H$$嵌入矩阵；同时让不同层共享注意力和前馈参数。

跨层共享减少参数存储，但每层仍要执行相同计算，因此**FLOPs**和激活显存不会按参数比例下降。全部共享还可能损害表达能力，尤其是前馈网络共享；“参数更少”不等于“推理更快”。

### (2) 注意力内部的跨层与跨头通信

标准多头注意力在两个方向上彼此隔离：同一层的各个头在输出投影前不通信，不同层的注意力分数也会重新计算。下面两种方法分别打破这两个边界，但都保持完整的全局注意力，因此不会消除$O(n^2)$复杂度。

### ⚪ **Talking-Heads Attention**：在**softmax**前后混合注意力头
- **paper**：[**Talking-Heads Attention**](https://arxiv.org/abs/2003.02436)

**Talking-Heads Attention**在查询—键点积之后，先沿头维做一次线性投影，再计算**softmax**，随后对各头的注意力权重再做一次头维投影。这样一个头形成的关系可以参与另一个值头的聚合，不再要求“打分头”和“读值头”一一对应。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-010-talkinghead.png)

跨头投影增加的参数很少，但它直接作用于长度为$n^2$的分数张量，长序列下仍会增加激活操作和访存。该方法说明“多头”不必等于完全独立的多组注意力，却没有成为现代大模型的默认组件；简单的独立头更容易被高效内核融合。

### ⚪ **RealFormer**：沿深度累加注意力分数
- **paper**：[**RealFormer: Transformer Likes Residual Attention**](https://arxiv.org/abs/2012.11747)

**RealFormer**把上一层尚未经过**softmax**的注意力分数加入当前层：

$$
S_l=\frac{Q_lK_l^\top}{\sqrt{d_k}}+S_{l-1},\qquad
A_l=\operatorname{softmax}(S_l).
$$

普通残差连接只在隐藏表示上跨层传递信息，**RealFormer**则额外保留一条“注意力分数残差”，让相邻层逐步修正而非完全重建路由。它可以使注意力模式更连贯、训练更稳定，但也把各层的头数和头对应关系绑定起来，并需要额外保存或传递分数张量。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-011-realformer.png)

注意力分数残差与残差流不是同一件事：前者传递“读哪里”的路由，后者传递“当前表示是什么”的状态。二者可以同时存在，也会产生不同的显存和优化影响。

### (3) 子层重排与并行

### ⚪ **Macaron Transformer**：在注意力两侧各放半步前馈网络
- **paper**：[**Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View**](https://arxiv.org/abs/1906.02762)

**Macaron**结构把标准“注意力→前馈”改为

$$
\frac{1}{2}\operatorname{FFN}
\rightarrow\operatorname{Attention}
\rightarrow\frac{1}{2}\operatorname{FFN}.
$$

它可以从微分方程的**Strang splitting**解释：前馈网络负责单位置演化，注意力负责位置间相互作用，对称排列提供更高阶的离散近似。该结构在翻译和语音模型中有效，也被**Conformer**等模型采用。若两个前馈模块不共享参数，执行成本会增加。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-transformer-012-macaron.png)

### ⚪ **Parallel Transformer Block**：并行计算注意力与前馈网络
- **paper**：[**PaLM: Scaling Language Modeling with Pathways**](https://arxiv.org/abs/2204.02311)

串行**Pre-LN**模块为

$$
\begin{aligned}
u&=x+\operatorname{Attn}(\operatorname{LN}(x)),\\
y&=u+\operatorname{FFN}(\operatorname{LN}(u)).
\end{aligned}
$$

并行模块让两条分支读取同一个归一化输入：

$$
y=x+\operatorname{Attn}(\operatorname{LN}(x))
+\operatorname{FFN}(\operatorname{LN}(x)).
$$

这样可以把注意力与前馈矩阵乘法并行调度，减少通信和串行依赖。**PaLM**报告大规模训练约$15\%$的速度提升；较小模型消融出现轻微质量下降，而较大模型未观察到明显下降。

### (4) 弹性深度

固定深度要求每个词元、每个样本都执行相同层数，但实际难度并不相同。弹性深度有两条路线：训练一个可以裁剪的层集合，或在推理时根据中间置信度动态停止。前者硬件规则、易于批处理，后者粒度更细，却容易因不同词元退出时间不同而破坏并行。

### ⚪ **LayerDrop**：训练可按需裁剪的层集合
- **paper**：[**Reducing Transformer Depth on Demand with Structured Dropout**](https://arxiv.org/abs/1909.11556)

**LayerDrop**在训练时随机跳过整个**Transformer**层，使同一组参数暴露于多种子网络深度。推理时可以按固定间隔删除层，在无需为每个目标深度单独训练的情况下交换质量与延迟。

它并不为单个样本动态选择计算量，也不能保证任意删层顺序都等价；训练深度、丢弃率和目标裁剪方式仍需匹配。其本质同时属于结构弹性与随机深度正则化，更完整的正则化视角见[正则化方法](https://0809zheng.github.io/2020/03/03/regularization.html)。

**Universal Transformer**的**Adaptive Computation Time**则属于动态路线：每个位置累计停机概率，达到阈值后停止深度递归。它能让不同词元使用不同计算步数，但在现代加速器上，批次通常仍要等待最慢位置，理论上节省的层数不一定转化为同等墙钟加速。动态深度的关键瓶颈往往不是退出规则，而是如何把不规则计算重新组织成高利用率批次。

# 3. 讨论：哪些改进真正改变了**Transformer**

第一层是**训练参数化**：**Pre-LN、RMSNorm、ReZero、Admin、DeepNorm、QK-Norm**主要改变优化路径和数值尺度，不改变模型可以建立哪些位置关系。它们决定模型是否容易训练、能否扩深，但不必然提高同等训练预算下的最终能力。

第二层是**局部计算单元**：**SwiGLU、Primer EZ、Macaron**和并行分支改变单层怎样处理通道、引入局部先验或排列子层。它们通常带来稳定但有限的收益，效果依赖是否在相近参数量、**FLOPs**和训练数据下公平比较。

第三层是**路由组织与容量分配**：**Talking-Heads、RealFormer**改变注意力头之间或层之间如何复用路由，参数共享让多层复用同一函数，**LayerDrop**让一组参数支持多个执行深度。这些方法分别改变中间信息、参数存储与实际计算量。

第四层是**信息流拓扑**：编码器、解码器、交叉注意力、稀疏注意力、递归状态和外部记忆决定谁能读取谁。这一层真正改变模型的可见范围、计算复杂度和因果结构，也是最可能改变能力边界的部分。

**Transformer**之所以泛用，是提供了一种可组合的接口：残差流保存当前状态，注意力在词元之间路由信息，前馈网络在通道维更新表示，归一化控制尺度，掩码定义可见关系。结构改进之所以层出不穷，正是因为这几个接口可以独立替换；而判断一项改进是否重要，也应回到它究竟改变了哪一个接口、付出了什么代价，以及收益能否跨规模和任务复现。
