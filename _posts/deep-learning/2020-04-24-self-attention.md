---
layout: post
title: '自注意力模型'
date: 2020-04-24
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea28825c2a9a83be5477d93.jpg'
tags: 深度学习
---

> Self-Attention Model.

**自注意力（Self-Attention）**也称为**内部注意力（Intra-Attention）**，利用注意力机制来“动态”地生成不同连接的权重，对于变长的输入序列生成相同长度的输出向量序列。

**本文目录**：
1. Background
2. Self-Attention
3. Multi-Head Self-Attention
4. Positional Encoding


# 1. Background
通常使用**循环神经网络RNN**实现Seq2Seq模型，双向RNN可以读取全局信息：

![](https://pic.downk.cc/item/5ea28d62c2a9a83be54e2d83.jpg)

这种方法的**弊端**是对于输入序列是顺序处理的，不能并行(parallel)实现。

可以用(1维)**卷积神经网络CNN**代替RNN实现Seq2Seq模型：

![](https://pic.downk.cc/item/5ea28dafc2a9a83be54e9e5c.jpg)

这种方法的**弊端**是每一个卷积核只能感受局部的信息，要获得更大的receptive field需要加深层数。

**自注意力模型**可以用来代替RNN，每个输出基于**全局信息**，并且可以**并行化**计算：

![](https://pic.downk.cc/item/5ea28e2bc2a9a83be54f42d5.jpg)

# 2. Self-Attention
自注意力模型采用**查询-键-值（Query-Key-Value，QKV）**模式。

### (1)
假设输入序列为$$X=[x_1,...,x_N] \in \Bbb{R}^{D_x×N}$$,经过**词嵌入**得到$$A=[a_1,...,a_N] \in \Bbb{R}^{D_a×N}$$;

将词嵌入向量线性映射到三个不同的空间，得到
1. **查询向量**$$Q=[q_1,...,q_N] \in \Bbb{R}^{D_k×N}$$
2. **键向量**$$K=[k_1,...,k_N] \in \Bbb{R}^{D_k×N}$$
3. **值向量**$$V=[v_1,...,v_N] \in \Bbb{R}^{D_v×N}$$;

![](https://pic.downk.cc/item/5ea2912ec2a9a83be5531a8a.jpg)

矩阵运算如下：

$$ Q = W^qA, \quad W^q \in \Bbb{R}^{D_k×D_a} $$

$$ K = W^kA, \quad W^k \in \Bbb{R}^{D_k×D_a} $$

$$ V = W^vA, \quad W^v \in \Bbb{R}^{D_v×D_a} $$

![](https://pic.downk.cc/item/5ea2928ac2a9a83be554b900.jpg)

### (2)
对于每个查询向量$q_i$使用**键值对注意力机制**,得到注意力分布$$\hat{a}_{1,1},...,\hat{a}_{1,N}$$：

![](https://pic.downk.cc/item/5ea29327c2a9a83be5558220.jpg)

矩阵运算如下：

$$ A = \frac{K^TQ}{\sqrt{D_k}} $$

$$ \hat{A} = softmax(A) $$

![](https://pic.downk.cc/item/5ea29480c2a9a83be557541f.jpg)

其中注意力得分选用**缩放点积Scaled Dot-Product**；Softmax函数按列运算。

### (3)
根据注意力分布$\hat{A}$，**加权求和**得到输出：

![](https://pic.downk.cc/item/5ea29582c2a9a83be558a1c7.jpg)

矩阵运算如下：

$$ B = V\hat{A} $$

![](https://pic.downk.cc/item/5ea295c1c2a9a83be558eecd.jpg)


自注意力模型的**优点**：
1. 提高并行计算效率;
2. 捕捉长距离的依赖关系。

自注意力模型可以看作在一个线性投影空间中建立$X$中不同向量之间的交互关系。


# 3. Multi-Head Self-Attention
为了提取更多的交互信息，可以使用**多头自注意力（Multi-Head Self-Attention）**，在多个不同的投影空间中捕捉不同的交互信息。

![](https://pic.downk.cc/item/5ea296f0c2a9a83be55a5633.jpg)

假设使用$M$个$head$，矩阵运算如下：

$$ A_m = \frac{K_m^TQ_m}{\sqrt{D_k}} $$

$$ B_m = V_msoftmax(A_m) $$

$$ B = W^o \begin{bmatrix} B_1 \\ ... \\ B_M \\ \end{bmatrix} $$

# 4. Positional Encoding
自注意力模型忽略了序列$$[x_1,...,x_N]$$中每个$x$的位置信息，因此显式的引入**位置编码**$e$：

![](https://pic.downk.cc/item/5ea29902c2a9a83be55cdc7b.jpg)

位置编码$e$并不是从数据中学习得到的，而是人为定义并加到词嵌入向量上的。

对位置编码的一些**解释**：

### (1)为什么是$add$而不是$concatenate$?
假设位置编码为$one$-$hot$形式，$concatenate$到输入向量上进行词嵌入：

![](https://pic.downk.cc/item/5ea29a5ac2a9a83be55e6d0f.jpg)

结果等价于先进行词嵌入，再加上位置编码。

### (2)设置位置编码
位置编码通过下面方式进行预定义：

$$ e_{t,2i} = sin(\frac{t}{10000^{\frac{2i}{D}}}) $$

$$ e_{t,2i+1} = cos(\frac{t}{10000^{\frac{2i}{D}}}) $$

其中$e_{t,2i}$表示第$t$个位置的编码向量的第$2i$维，$D$是编码向量的维度。

### (3)位置编码的可视化
位置编码矩阵可视化如下：

![](https://pic.downk.cc/item/5ea29a8ac2a9a83be55ea47e.jpg)