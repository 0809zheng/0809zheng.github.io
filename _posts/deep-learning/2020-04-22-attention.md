---
layout: post
title: '注意力机制'
date: 2020-04-22
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e9fbd69c2a9a83be53b80dd.jpg'
tags: 深度学习
---

> Attention  Mechanism.

在条件Seq2Seq模型中，将输入文本通过编码器Encoder转换为一个**上下文向量**$c$，喂入解码器Decoder。

![](https://pic.downk.cc/item/5e9ed899c2a9a83be5966fe5.jpg)

在比较简单的任务（比如文本分类）中，只需要编码一些对分类有用的信息，因此用一个向量来表示文本语义是可行的。

但是在复杂的任务（比如阅读理解）中，给定的背景文章（Background Document）一般比较长，如果用循环神经网络来将其转换为向量表示，那么这个编码向量很难反映出背景文章的所有语义。

引入**注意力机制（Attention Mechanism）**，选择一些关键的信息输入进行处理，来提高模型的效率。

注意力机制的实现过程：

1. 在所有输入信息上计算**注意力分布**；
2. 根据注意力分布来计算输入信息的**加权平均**。

### ①注意力分布
给定输入向量$$[x_1,...,x_N]$$;

对于文本输入，输入向量可以是RNN的隐状态向量：

![](https://pic.downk.cc/item/5e9fabb5c2a9a83be52dfb27.jpg)

对于图像输入，输入向量可以是不同区域的特征向量：

![](https://pic.downk.cc/item/5e9fab83c2a9a83be52de2b4.jpg)

给定**查询向量(query vertor)**$q$，可以是动态生成的，也可以是可学习的参数。

则**注意力分布（Attention Distribution）**$α_n$定义为：

$$ α_n = softmax(s(x_n,q)) = \frac{exp(s(x_n,q))}{\sum_{n=1}^{N} {exp(s(x_n,q))}} $$

其中$s(x,q)$是**注意力得分函数**(**相似函数（alignment model）**)，常用的计算方式：

1. **加性模型**：$$s(x,q) = v^Ttanh(Wx+Uq)$$,其中$v$、$W$、$U$是可学习参数；
2. **点积模型**：$$s(x,q) = x^Tq$$,相比加性模型使用矩阵乘积，提高计算效率;
3. **缩放点积模型**：$$s(x,q) = \frac{x^Tq}{\sqrt{D}}$$,其中$D$是输入向量的维度，相比点积模型方差小;
4. **双线性模型**：$$s(x,q) = x^TWq$$,其中$W$是可学习参数，相比点积模型引入了非对称性。

### ②加权平均
采用**加权平均(weighted sum)**对输入信息进行汇总：

$$ c = \sum_{n=1}^{N} {α_nx_n} $$

将$c$作为Decoder的输入，并反复使用注意力机制。

上述方法是一种**软性注意力机制（Soft Attention Mechanism）**,也叫**全局(global)注意力**。

注意力机制还存在一些变化的模型：
1. **硬性注意力（Hard Attention）**,也叫**局部(local)注意力**：$$ c = x_{\hat{n}}, \quad \hat{n} = argmax_{(n)}α_n $$，硬性注意力的缺点是最大采样不能使用反向传播，需要用强化学习进行训练;
2. **键值对注意力（key-value pair Attention）**：输入信息用键值对$$x_n = (k_n,v_n)$$表示，则用键计算注意力分布$$α_n = softmax(s(k_n,q))$$，用值计算加权平均$$c = \sum_{n=1}^{N} {α_nv_n}$$；
![](https://pic.downk.cc/item/5e9fbfa1c2a9a83be53d2b6a.jpg)
3. **多头注意力（Multi-Head Attention）**:使用多个查询向量$q_1,...,q_M$并行地从输入信息中选取多组信息$$c_1,...,c_M$$,并进行向量拼接。
![](https://pic.downk.cc/item/5e9fc024c2a9a83be53d7ad2.jpg)

### 正则化
当同时使用多个输入文本或图像时，每一个样本的注意力得分总和应该相近，引入正则化：

$$ \sum_{i}^{} {(τ-\sum_{n}^{} {s^i(x_n,q)})} $$

其中$τ$是给定的常数。

### 自注意力模型

**自注意力（Self-Attention）**也称为**内部注意力（Intra-Attention）**，利用注意力机制来“动态”地生成不同连接的权重，对于变长的输入序列生成相同长度的输出向量序列。

自注意力模型采用**查询-键-值（Query-Key-Value，QKV）**模式。

![](https://pic.downk.cc/item/5ea11f31c2a9a83be59bea46.jpg)

假设输入序列为$$X=[x_1,...,x_N] \in \Bbb{R}^{D_x×N}$$,输出序列为$$H=[h_1,...,h_N] \in \Bbb{R}^{D_v×N}$$，

将输入序列线性映射到三个不同的空间，得到**查询向量**$$Q=W_qX \in \Bbb{R}^{D_k×N}$$、**键向量**$$K=W_kX \in \Bbb{R}^{D_k×N}$$和**值向量**$$V=W_vX \in \Bbb{R}^{D_v×N}$$。

使用**键值对注意力机制**得到输出向量$H$：

$$ H = Vsoftmax(s(Q,K)) $$

其中注意力得分函数$s$常选用**缩放点积**：

$$ s(Q,K) = \frac{K^TQ}{\sqrt{D_k}} $$

自注意力模型的优点：
1. 提高并行计算效率
2. 捕捉长距离的依赖关系

自注意力模型可以看作在一个线性投影空间中建立$X$中不同向量之间的交互关系。

为了提取更多的交互信息，可以使用**多头自注意力（Multi-Head Self-Attention）**，在多个不同的投影空间中捕捉不同的交互信息。

$$ H = W_o[H_1;...;H_M] $$

$$ H_m = V_msoftmax(s(Q_m,K_m)) $$
