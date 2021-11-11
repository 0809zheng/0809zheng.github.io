---
layout: post
title: '自注意力模型'
date: 2020-04-24
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea28825c2a9a83be5477d93.jpg'
tags: 深度学习
---

> Self-Attention Model.

**自注意力(Self-Attention)**机制也称为**内部注意力(Intra-Attention)**，是一种特殊的[注意力机制](https://0809zheng.github.io/2020/04/22/attention.html)。自注意力机制作为一种新型的网络结构被广泛应用于自然语言处理与计算机视觉等任务中。本文首先讨论注意力机制与自注意力机制的区别；其次对比卷积神经网络、循环神经网络和自注意力机制；最后介绍自注意力机制的实现细节。

**本文目录**：
1. Attention and Self-Attention
1. CNN, RNN, and Self-Attention
2. Self-Attention
3. Multi-Head Self-Attention
4. Position Encoding

# 1. Attention and Self-Attention
[注意力机制](https://0809zheng.github.io/2020/04/22/attention.html)(**attention mechanism**)最早是在序列到序列模型中提出的，用于解决机器翻译任务。在该任务中需要把输入序列$$\{x_1,x_2,...,x_j\}$$转换为输出序列$$\{y_1,y_2,...,y_i\}$$，因此序列到序列模型采用编码器-解码器结构，即引入可学习的权重参数$w_{ij}$，使得：

$$ y_i = \sum_{j}^{}w_{ij}x_j $$

其中$w_{ij}$表示输入序列的第$i$个**token**对输出序列的第$j$个**token**的重要性程度。在实际中引入约束$\sum_{j}^{}w_{ij}=1$，这一步是通过**softmax**函数实现的。自注意力机制是一种特殊的注意力机制，其主要区别在于前者的权重参数$w_{ij}$并不是直接学习得到的，而是由输入计算得到的，如通过**点积**的方式计算两个输入**token**的相关性：

$$ w_{ij}=x_i^Tx_j $$

注意力机制与自注意力机制的主要区别包括：
1. 注意力机制的权重参数是一个全局可学习参数，对于模型来说是**固定**的；而自注意力机制的权重参数是由输入决定的，即使是同一个模型，对于不同的输入也会有**不同**的权重参数。
2. 注意力机制的输出序列长度与输入序列长度可以是**不同**的；而自注意力机制的的输出序列长度与输入序列长度必须是**相同**的。
3. 注意力机制在一个模型中通常只使用一次，作为编码器和解码器之间的**连接**部分；而自注意力机制在同一个模型中可以使用很多次，作为网络**结构**的一部分。
4. 注意力机制擅长捕捉两个**序列之间**的关系，如机器翻译任务中将一个序列映射为另一个序列；而自注意力机制擅长捕捉单个**序列内部**的关系，如作为预训练语言模型的基本结构。

# 2. CNN, RNN, and Self-Attention
卷积神经网络、循环神经网络和自注意力机制都可以用于自然语言处理任务。在自然语言处理任务中，首先对输入序列(如句子)进行分词，将每个词转化成对应的词向量；即可将输入序列表示为$X=(x_1,x_2,...,x_n)\in \Bbb{R}^{n \times d}$，其中$x_i$表示第$i$个词的维度为$d$的词向量。因此对输入序列的处理等价于对该序列进行编码：

![](https://pic.imgdb.cn/item/60ebc1765132923bf88a64f9.jpg)

### (a) 卷积神经网络
可以用($1$维)**卷积神经网络**对该输入序列进行处理。即使用卷积核进行滑动窗口遍历，如长度为$3$的卷积核：

$$ y_i = f(x_{i-1},x_{i},x_{i+1}) $$

![](https://pic.downk.cc/item/5ea28dafc2a9a83be54e9e5c.jpg)

卷积神经网络容易并行，可以捕捉一些全局的结构信息。但其**弊端**是每一个卷积核只能感受局部的信息，要获得更大的**receptive field**需要加深层数。

### (b) 循环神经网络

**循环神经网络**是自然语言处理任务中最常用的模型，其计算过程是通过递归实现的：

$$ y_i = f(h_{i-1},x_{i}) $$

![](https://pic.downk.cc/item/5ea28d62c2a9a83be54e2d83.jpg)

循环神经网络本身结构简单，适合序列建模。但其**弊端**是对于输入序列是顺序处理的，速度较慢，不能并行(**parallel**)实现；且循环神经网络无法很好的学习到全局结构信息(尽管上图的双向结构一定程度上缓解了这个问题)。

### (c) 自注意力
**自注意力模型**的每个输出基于**全局信息**，并且可以**并行化**计算：

$$ y_i = f(x_{1},x_{2},...,x_{n}) $$

![](https://pic.downk.cc/item/5ea28e2bc2a9a83be54f42d5.jpg)


### (d) 对不同网络结构的讨论

1. 卷积神经网络事实上只能获得局部信息，需要通过**堆叠**更多层数来增大感受野；循环神经网络需要通过**递归**获得全局信息，因此一般采用双向形式；自注意力机制能够直接获得**全局信息**。
1. 常用的神经网络模型(如多层感知机)，其每一层的权重参数经过训练后是**固定**的，与输入无关；而自注意力层的权重是由输入决定的，但其只能生成固定长度的输入序列。
1. 循环神经网络是递归计算的，无法并行；卷积神经网络的不同卷积核之间可以并行计算；自注意力机制的计算是高度并行的，很容易被**GPU**等加速。

若输入序列长度为$n$，特征维度为$d$。则上述模型每层的计算复杂度、序列操作数(越大表示可并行化程度越差)和最大路径长度分别为：

$$
\begin{array}{c|ccc}
    \text{Layer Type} & \text{Complexity per Layer} & \text{Sequential Operations} & \text{Maximum Path Length} \\
    \hline
    \text{Convolutional} & O(k \cdot n \cdot d^2) & O(1) & O(log_k(n)) \\
    \text{Recurrent} & O(n \cdot d^2) & O(n) & O(n) \\
    \text{Self-Attention} & O(n^2 \cdot d) & O(1) & O(1) \\ 
\end{array}
$$

# 3. Self-Attention
本小节介绍自注意力机制的运算过程。自注意力模型采用**查询-键-值(Query-Key-Value,QKV)**模式。

### (1) 计算查询矩阵Q,键矩阵K,值矩阵V
假设输入序列为$$X=[x_1,...,x_N] \in \Bbb{R}^{D_x×N}$$,经过**词嵌入**得到$$A=[a_1,...,a_N] \in \Bbb{R}^{D_a×N}$$;
将词嵌入矩阵线性映射到三个不同的空间，得到
1. **查询矩阵**$$Q=[q_1,...,q_N] \in \Bbb{R}^{D_k×N}$$
2. **键矩阵**$$K=[k_1,...,k_N] \in \Bbb{R}^{D_k×N}$$
3. **值矩阵**$$V=[v_1,...,v_N] \in \Bbb{R}^{D_v×N}$$;

![](https://pic.downk.cc/item/5ea2912ec2a9a83be5531a8a.jpg)

矩阵运算如下：

$$ Q = W^qA, \quad W^q \in \Bbb{R}^{D_k×D_a} $$

$$ K = W^kA, \quad W^k \in \Bbb{R}^{D_k×D_a} $$

$$ V = W^vA, \quad W^v \in \Bbb{R}^{D_v×D_a} $$

![](https://pic.downk.cc/item/5ea2928ac2a9a83be554b900.jpg)

### (2) 计算注意力分布
对于每个查询向量$q_i$使用**键值对注意力机制**,得到注意力分布$$\hat{a}_{1,1},...,\hat{a}_{1,N}$$：

![](https://pic.downk.cc/item/5ea29327c2a9a83be5558220.jpg)

矩阵运算如下：

$$ A = \frac{K^TQ}{\sqrt{D_k}} $$

$$ \hat{A} = softmax(A) $$

![](https://pic.downk.cc/item/5ea29480c2a9a83be557541f.jpg)

其中注意力得分选用**缩放点积Scaled Dot-Product**，其原因是后续的**Softmax**函数对较大或较小的输入非常敏感(容易映射到$1$或$0$)，因此通过因子$\sqrt{D_k}$进行缩放；**Softmax**函数按**列**运算。

### (3) 加权求和
根据注意力分布$\hat{A}$，**加权求和**得到输出：

![](https://pic.downk.cc/item/5ea29582c2a9a83be558a1c7.jpg)

矩阵运算如下：

$$ B = V\hat{A} $$

![](https://pic.downk.cc/item/5ea295c1c2a9a83be558eecd.jpg)

自注意力模型的**优点**：
1. 提高并行计算效率;
2. 捕捉长距离的依赖关系。

自注意力模型可以看作在一个线性投影空间中建立$X$中不同向量之间的交互关系。上述自注意力运算的计算复杂度为$O(N^2)$。实践中有些问题并不需要捕捉全局结构，只依赖于局部信息，此时可以使用**restricted**自注意力机制，即假设当前词只与前后$r$个词发生联系(类似于卷积中的滑动窗口)，此时计算复杂度为$O(rN)$。


# 4. Multi-Head Self-Attention
为了提取更多的交互信息，可以使用**多头自注意力(Multi-Head Self-Attention)**，即在多个不同的投影空间中捕捉不同的交互信息。不妨类比于卷积神经网络，其一个卷积核通常用于捕捉某一类**pattern**的信息，故采用多个卷积核。自注意力机制采用多个**head**，便可以捕捉不同的相关性。在实践中，首先通过$M$个**head**生成$M$个不同的输出$B_1,B_2,...,B_M$，将其合并后再通过一层全连接层进行线性变换：

![](https://pic.downk.cc/item/5ea296f0c2a9a83be55a5633.jpg)

假设使用$M$个**head**，矩阵运算如下：

$$ A_m = \frac{K_m^TQ_m}{\sqrt{D_k}} $$

$$ B_m = V_msoftmax(A_m) $$

$$ B = W^o \begin{bmatrix} B_1 \\ ... \\ B_M \\ \end{bmatrix} $$

在实现多头自注意力时，有两种常用的形式：
1. **Narrow Self-Attention**：把输入词向量切割成$h$块，每一块使用一次自注意力运算；这种方法速度快，节省内存，但是效果不好；
2. **Wide Self-Attention**：对输入词向量独立地使用$h$次自注意力运算；这种方法效果更好，但花费更多时间和内存

# 5. Position Encoding
自注意力模型忽略了序列$$[x_1,...,x_N]$$中每个$x$的位置信息，即将该序列打乱后并不影响输出结果。因此在模型中显式的引入**位置编码(position encoding)**$e$：

![](https://pic.downk.cc/item/5ea29902c2a9a83be55cdc7b.jpg)

位置编码$e$是自注意力机制中获取序列位置信息的唯一来源，是模型重要的组成部分。位置编码并不是从数据中学习得到的，而是人为定义并加到词嵌入向量上的。对位置编码的一些**解释**：

### (1)为什么是$add$而不是$concatenate$?
假设位置编码为$one$-$hot$形式，$concatenate$到输入向量上进行词嵌入：

![](https://pic.downk.cc/item/5ea29a5ac2a9a83be55e6d0f.jpg)

结果等价于先对位置索引和输入序列分别进行词嵌入，再相加。此时的位置编码是一种**位置嵌入(position embedding)**

### (2)设置位置编码
位置编码通过下面方式进行预定义：

$$ e_{t,2i} = sin(\frac{t}{10000^{\frac{2i}{D}}}) $$

$$ e_{t,2i+1} = cos(\frac{t}{10000^{\frac{2i}{D}}}) $$

其中$e_{t,2i}$表示第$t$个位置的编码向量的第$2i$维，$D$是编码向量的维度。

选用该形式的位置编码的思路是，由于有:

$$ sin(\alpha+\beta)=sin(\alpha)cos(\beta)+cos(\alpha)sin(\beta) $$

$$ cos(\alpha+\beta)=cos(\alpha)cos(\beta)-sin(\alpha)sin(\beta) $$

因此位置$\alpha+\beta$处的编码很容易被位置$\alpha$和位置$\beta$处的编码表示。通过实验发现位置嵌入和上述位置编码效果是接近的，因此直接选用后者。

上述位置编码也被称为**Sinusoidal**位置编码，其编码矩阵可视化如下：

![](https://pic.downk.cc/item/5ea29a8ac2a9a83be55ea47e.jpg)