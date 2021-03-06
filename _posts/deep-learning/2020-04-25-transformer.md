---
layout: post
title: 'Transformer'
date: 2020-04-25
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea28751c2a9a83be5467bc1.jpg'
tags: 深度学习
---

> Transformer，基于Multi-head self-attention的Seq2Seq模型.

- paper：Attention Is All You Need
- arXiv：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**Transformer**是一个基于**多头自注意力(Multi-head Self-Attention)**的**序列到序列(Seq2Seq)**模型，网络结构可以分成**编码器Encoder**和**解码器Decoder**两部分：

![](https://pic.imgdb.cn/item/60ebe1f55132923bf8acdd67.jpg)

# 1. Encoder
1. **Embedding**：输入序列经过词嵌入得到词嵌入向量；
2. **Positional Encoding**：词嵌入向量加上位置编码；
3. **Multi-head self-attention**：[多头自注意力](https://0809zheng.github.io/2020/04/24/self-attention.html#3-multi-head-self-attention)层；
4. **Add & Norm**：残差连接和[Layer Norm](https://0809zheng.github.io/2020/03/04/normalization.html#9-layer-normalization)；
5. **Feed Forward**:逐位置的前馈神经网络$$\text{FFN}(z)=W_2 \text{ReLU}(W_1z+b_1)+b_2$$


# 2. Decoder
1. **Embedding**：训练时使用右移(**shifted right**)的目标序列，经过词嵌入得到词嵌入向量；
2. **Positional Encoding**：词嵌入向量加上位置编码；
3. **Masked Self-Attention**：使用自注意力模型对已生成的前缀序列进行编码，通过$mask$阻止每个位置选择后面的输入信息；
4. **Multi-head self-attention**：[多头自注意力](https://0809zheng.github.io/2020/04/24/self-attention.html#4-multi-head-self-attention)层；
5. **Feed Forward**:逐位置的前馈神经网络。


# 3. 实验结果
使用**multi-head**机制，既可以捕捉到近距离依赖关系，又可以捕捉到远距离依赖关系；且模型具有较好的可解释性：

![](https://pic.downk.cc/item/5ea2a8e3c2a9a83be570cfb4.jpg)

