---
layout: post
title: '降低Transformer的计算复杂度: 稀疏化和线性化'
date: 2021-07-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60ebbe525132923bf878e6c3.jpg'
tags: 深度学习
---

> Efficient Transformers.

本文目录：
1. Transformer的计算复杂度
2. 改进自注意力机制

# 1. Transformer的计算复杂度

## (1) Transformer的典型结构
![](https://pic.imgdb.cn/item/60ebbe165132923bf8778fed.jpg)

典型的**Transformer**结构如上图所示，其整体计算量来源于模型中的[自注意力层](https://0809zheng.github.io/2020/04/24/self-attention.html)和全连接层两部分，本文主要讨论自注意力层的改进。

## (2) 自注意力机制的运算

由于计算机中乘法的计算速度比加法慢，因此在衡量计算复杂度时主要考虑**乘法**。对于矩阵乘法$(a, b)\times(b,c)$，计算复杂度为$O(abc)$。

![](https://pic.downk.cc/item/5ea28825c2a9a83be5477d93.jpg)



### ① 计算查询矩阵Q,键矩阵K,值矩阵V
对于经过词嵌入并加入位置编码的输入序列$$X=[x_1,...,x_N] \in \Bbb{R}^{D_x×N}$$，将其通过仿射变换映射为查询矩阵$Q$,键矩阵$K$,值矩阵$V$：

$$ Q = W^qX \in \Bbb{R}^{D_k×N}, \quad W^q \in \Bbb{R}^{D_k×D_x} $$

$$ K = W^kX \in \Bbb{R}^{D_k×N}, \quad W^k \in \Bbb{R}^{D_k×D_x} $$

$$ V = W^vX \in \Bbb{R}^{D_v×N}, \quad W^v \in \Bbb{R}^{D_v×D_x} $$

这一步的计算量为$O(2D_kND_x+D_vND_x)=O(ND_x(2D_k+D_v)))=O(N)$。

### ② 计算注意力矩阵A
根据查询矩阵$Q$和键矩阵$K$计算注意力矩阵$$A \in \Bbb{R}^{N×N}$$(这一步是在计算每个输入与其他输入的相关性)：

$$ A = \frac{K^TQ}{\sqrt{D_k}} $$

这一步的计算量为$O(N^2D_k)=O(N^2)$。

### ③ 归一化注意力矩阵A

使用**softmax**函数对注意力矩阵$A$进行归一化：

$$ \hat{A} = \text{softmax}(A)_{\text{dim}=1} $$

这一步的计算量为$O(1)$。

### ④ 加权求和计算输出H
根据归一化的注意力矩阵$\hat{A}$和值矩阵$V$加权求和计算输出：

$$ H = V\hat{A} \in \Bbb{R}^{D_x×N} $$

这一步的计算量为$O(D_xN^2)=O(N^2)$。

根据上述分析可知，自注意力机制的整体运算复杂度为:

$$ O(2D_kND_x+D_vND_x+N^2D_k+1+D_xN^2) = O(N^2) $$

其中$N$是输入序列长度。一般地，选择向量的特征维度$D_k=D_v=D_x=d$，在自然语言处理等任务中一般有$N>d$。

## (3) 自注意力层和全连接层的比较
通常认为**Transformer**的计算量主要来源于自注意力层，然而全连接层的计算量也不可忽略。下面进行分析。

根据上述分析可知，自注意力层的整体运算复杂度为:

$$ O(2D_kND_x+D_vND_x+N^2D_k+1+D_xN^2) $$

其中$N$是输入序列长度。一般地，选择向量的特征维度$D_k=D_v=D_x=d$，在自然语言处理等任务中一般有$N>d$。则自注意力层的计算复杂度简化为：

$$ O(2dNd+dNd+N^2d+1+dN^2) = O(3Nd^2+2N^2d) $$

**Transformer**中的全连接层一般设置两层，第一层的特征维度$d\to 4d$，第二层的特征维度$4d\to d$。因此全连接层的计算复杂度为：

$$ O(Nd4d+N4dd) = O(8Nd^2) $$

若假设自注意力层的计算复杂度超过全连接层，则有：

$$ 3Nd^2+2N^2d>8Nd^2 $$

解上式得$n>2.5d$。对于**base**版本的**Transformer**，$d=768$；则只有序列长度超过$n=1920$时自注意力层的计算量才会超过全连接层。当输入序列长度较小时，模型主要计算量来源于全连接层，其计算复杂度仍然是近似线性的。

综上所述，下述改进**Transformer**效率的工作，大多是在序列长度较大时进行的。而当输入序列长度有限时，这些改进并不明显。

# 2. 改进自注意力机制
目前已经有大量改进**Transformer**中的自注意力运算，进而降低其计算复杂度的方法。从第一节的分析中可以看出，**计算自注意力矩阵**以及**加权求和计算输出**这两个步骤引入了$O(N^2)$的计算复杂度。因此可以改进这两个步骤，从而降低计算复杂度。

## (1) 改进注意力矩阵A的计算: 稀疏化
这类方法的改进思路是使得注意力矩阵的计算**稀疏化**，即对输入序列中的每一个位置只计算其与一部分位置(而不是全部位置)之间的相关性，表现为注意力矩阵是稀疏的。

### ① 标准自注意力矩阵

标准的自注意力机制使用**缩放点积**(**scaled dot-product**)计算注意力矩阵，表示为$A = \frac{K^TQ}{\sqrt{D_k}}$。对于长度为$N$的输入序列，其每一个位置都会和该序列的所有位置进行交互并计算注意力(相关度)，从而得到$N^2$大小的注意力矩阵。该矩阵的第$i$列代表第$i$个输入位置与所有位置的相关性，该矩阵通常是一个**稠密**(**dense**)矩阵。

![](https://pic.imgdb.cn/item/60ed16a45132923bf8e602d3.jpg)

### ② 一些稀疏化方法

- [Sparse Transformer](https://0809zheng.github.io/2021/07/13/sparsetransformer.html)：窗口注意力+空洞注意力

![](https://pic.imgdb.cn/item/60ed1c745132923bf80f980e.jpg)

- [Reformer](https://0809zheng.github.io/2021/08/11/reformer.html)：使用局部敏感哈希选择注意力位置

![](https://pic.imgdb.cn/item/6115e64d5132923bf86784a5.jpg)

- [Longformer](https://0809zheng.github.io/2021/08/14/longformer.html)：窗口注意力+空洞注意力+全局注意力

![](https://pic.imgdb.cn/item/61179cb95132923bf8100609.jpg)

- [Big Bird](https://0809zheng.github.io/2020/08/08/bigbird.html)：随机注意力+窗口注意力+全局注意力

![](https://pic.downk.cc/item/5f2e340114195aa594463791.jpg)



## (2) 改进输出的加权求和: 线性化
这类方法的改进思路是使得自注意力的计算**线性化**。

### ① 从矩阵角度理解线性化
标准的**Attention**计算为:

$$ \text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d}})V=\hat{A}V $$

其计算复杂度为$O(N^2d)=O(N^2)$。如下图所示(图中$L$相当于$N$，表示序列长度)：

![](https://pic.imgdb.cn/item/61138cae5132923bf86415dc.jpg)

自注意力的线性化是指通过某种矩阵分解$\hat{A}=Q'{K'}^T$，使得**Attention**计算变为$\hat{A}V=Q'{K'}^TV=Q'({K'}^TV)$。由矩阵乘法的**结合律**，可以先计算$K^TV$，从而使得计算复杂度降低为$O(2Nrd)=O(N)$。因此线性化的关键是寻找合适的矩阵分解。

### ② 从向量角度理解线性化
标准的**Attention**也可以由向量表示。若记查询向量$q$,键向量$k$,值向量$v$；则对于序列的第$i$个输入$x_i$，其输出计算为:

$$ (\text{softmax}(\frac{QK^T}{\sqrt{d}})V)_i=\frac{\sum_{j=1}^{N}e^{\frac{q_i^Tk_j}{\sqrt{d}}}v_j}{\sum_{j=1}^{N}e^{\frac{q_i^Tk_j}{\sqrt{d}}}}=\frac{\sum_{j=1}^{N}\text{sim}(q_i,k_j)v_j}{\sum_{j=1}^{N}\text{sim}(q_i,k_j)} $$

上式可以表示为使用输入$x_i$的查询向量$q_i$和所有输入的键向量$k_j$计算相似度函数$\text{sim}(q_i,k_j)≥0$，并对所有输入的值向量$v_j$进行加权求和。对于标准的缩放点积，相似度函数选用：

$$ \text{sim}(q_i,k_j) = e^{\frac{q_i^Tk_j}{\sqrt{d}}} $$

若选择合适的相似度函数使得$\text{sim}(q_i,k_j)=\phi(q_i)^T\psi(k_j)$，则有：

$$ \frac{\sum_{j=1}^{N}\text{sim}(q_i,k_j)v_j}{\sum_{j=1}^{N}\text{sim}(q_i,k_j)}=\frac{\sum_{j=1}^{N}\phi(q_i)^T\psi(k_j)v_j}{\sum_{j=1}^{N}\phi(q_i)^T\psi(k_j)}=\frac{\phi(q_i)^T\sum_{j=1}^{N}\psi(k_j)v_j^T}{\phi(q_i)^T\sum_{j=1}^{N}\psi(k_j)} $$

注意到通过上述转换，将计算复杂度从$O(N^2)$降为$O(N)$。
因此线性化的关键是寻找合适的相似度函数。

### ③ 一些线性化方法

- [Efficient Attention](https://0809zheng.github.io/2021/08/15/efficient.html)：对$Q$的每一行,$K$的每一列进行归一化：

$$ \text{Attention}(Q,K,V)==\rho_Q(Q)\rho_K(K)^TV $$

- [Synthesizer](https://0809zheng.github.io/2020/07/14/synthesizer.html)：使用全连接神经网络(或随机)生成自注意力矩阵:

$$ \text{Attention}(Q,K,V)=\text{FFN}(X)V $$

- [Linformer](https://0809zheng.github.io/2021/08/13/linformer.html)：
为$K$和$V$引入了低秩映射$E,F \in \Bbb{R}^{k \times N}$：

$$ \text{Attention}(Q,K,V)=\text{softmax}(\frac{Q(EK)^T}{\sqrt{d}})(FV) $$

- [Linear Transformer](https://0809zheng.github.io/2021/08/10/linear.html)：使用线性注意力实现快速自回归的Transformer。

$$ \text{sim}(q,k)=(\text{elu}(q)+1)^T(\text{elu}(k)+1), \quad \text{elu}(x)=\begin{cases} x, & x>0 \\ e^x-1, & x≤0 \end{cases} $$


- [Performer](https://0809zheng.github.io/2021/08/12/performer.html)：通过随机投影将Attention的复杂度线性化。

$$ \text{sim}(q,k) =e^{q\cdot k} = \Bbb{E}_{\omega \text{~} \mathcal{N}(\omega;0,1_d)} [e^{w\cdot q-||q||^2/2}\times e^{w\cdot k-||k||^2/2}] \\ ≈ \frac{1}{\sqrt{m}} \begin{pmatrix} e^{w_1\cdot q-||q||^2/2} \\ e^{w_2\cdot q-||q||^2/2} \\ \cdots \\ e^{w_m\cdot q-||q||^2/2} \end{pmatrix} \cdot \frac{1}{\sqrt{m}} \begin{pmatrix} e^{w_1\cdot k-||k||^2/2} \\ e^{w_2\cdot k-||k||^2/2} \\ \cdots \\ e^{w_m\cdot k-||k||^2/2} \end{pmatrix}  = \tilde{q}\cdot\tilde{k} $$

- [Nyströmformer](https://0809zheng.github.io/2021/04/29/nystromformer.html)：使用Nyström方法近似自注意力运算。

$$ \text{Attention}(Q,K,V)= (\tilde{F}\times\tilde{A}\times\tilde{B})V $$

$$ \tilde{F} =  \text{softmax}(\frac{Q\tilde{K}^T}{\sqrt{d_q}} ) , \tilde{B} =  \text{softmax}(\frac{\tilde{Q}K^T}{\sqrt{d_q}} ), \tilde{A} = \text{softmax}(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_q}} )^{-1} $$

- [External Attention](https://0809zheng.github.io/2021/08/09/external.html)：将$K$和$V$用全局共享的记忆单元$M_k$和$M_v$表示。

$$ \text{Attention}(Q,M_k,M_v) = \text{Norm}(Q^TM_k)M_v $$

- [FLASH](https://0809zheng.github.io/2022/03/05/flash.html)：对输入序列不重叠地分块，并使用局部注意力和全局注意力的混合。

$$ \text{Attention}(Q^{quad},K^{quad},V)=\frac{1}{ns} \text{relu}^2(Q_g^{quad}{K_g^{quad}}^T)V_g \\ \text{Attention}(Q^{lin},K^{lin},V) =\frac{1}{n}Q_g^{lin} \sum_{h=1}^{n/c} {K_h^{lin}}^TV_h $$

## (3) 从秩的角度理解自注意力机制的改进
下面从信息损失的角度分析自注意力机制。自注意力机制中最重要的步骤之一是计算自注意力矩阵$A\in \Bbb{R}^{N \times N}$，它是由$Q,K\in \Bbb{R}^{N \times d}$通过$\text{softmax}(QK^T)$计算得到。$N$是输入序列长度，$d$是注意力的**key size**，通常$d<N$。因此矩阵$QK^T$的秩不超过$d$，离满秩$N$差距较大。通常更大的秩能保留更多有效信息，使得信息瓶颈效应更弱。而**softmax**函数计算$e^{QK^T}$，由于指数函数可能会使矩阵的秩增加(比如秩为$0$的全$0$矩阵取指数后变为秩为$1$的全$1$矩阵)，因此**softmax**函数使得注意力矩阵具有**升秩**的可能性，从而提高有效处理信息的能力。

自注意力运算的**线性化**是指寻找近似矩阵$Q',K'\in \Bbb{R}^{N \times m}$使得$A=Q'{K'}^T$。因此线性自注意力矩阵的秩不超过$m$。为了弥补秩较小带来的损失，通常设置$m>d$，如**Performer**中设置$m=4d$。由于扩大了**key size**，在处理短序列时线性注意力的计算可能会比标准自注意力的计算还要慢。

自注意力矩阵的**稀疏化**则有时能够提高自注意力矩阵的秩。一般而言，稀疏的自注意力矩阵表示输入序列的每一个位置能够显著地与序列中的有限个位置关联。标准**Attention**中的指数运算$e^{QK^T}$能够放大不同$q\cdot k$
之间的差距，使得自注意力矩阵具有稀疏化的趋势。而线性**Attention**的注意力结果是稠密的，当序列长度$N$较大时，结果趋近于平均池化。


# ⚪ 参考文献
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732v2)：(arXiv2009)一篇高效Transformer综述。
- [<font color=Blue>Efficient Attention: Attention with Linear Complexities</font>](https://0809zheng.github.io/2021/08/15/efficient.html)：(arXiv1812)具有线性复杂度的高效自注意力机制。
- [<font color=Blue>Generating Long Sequences with Sparse Transformers</font>](https://0809zheng.github.io/2021/07/13/sparsetransformer.html)：(arXiv1904)Sparse Transformer：使用稀疏注意力的Transformer。
- [<font color=Blue>Reformer: The Efficient Transformer</font>](https://0809zheng.github.io/2021/08/11/reformer.html)：(arXiv2001)Reformer: 使用局部敏感哈希和可逆FFN实现高效Transformer。
- [<font color=Blue>Longformer: The Long-Document Transformer</font>](https://0809zheng.github.io/2021/08/14/longformer.html)：(arXiv2004)Longformer: 适用于长文本的Transformer。
- [<font color=Blue>Synthesizer: Rethinking Self-Attention in Transformer Models</font>](https://0809zheng.github.io/2020/07/14/synthesizer.html)：(arXiv2005)Synthesizer：使用合成注意力的Transformer模型。
- [<font color=Blue>Linformer: Self-Attention with Linear Complexity</font>](https://0809zheng.github.io/2021/08/13/linformer.html)：(arXiv2006)Linformer: 线性复杂度的自注意力机制。
- [<font color=Blue>Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention</font>](https://0809zheng.github.io/2021/08/10/linear.html)：(arXiv2006)Linear Transformer: 使用线性注意力实现快速自回归的Transformer。
- [<font color=Blue>Big Bird: Transformers for Longer Sequences</font>](https://0809zheng.github.io/2020/08/08/bigbird.html)：(arXiv2007)Big Bird：一种应用于长序列的Transformer模型。
- [<font color=Blue>Rethinking Attention with Performers</font>](https://0809zheng.github.io/2021/08/12/performer.html)：(arXiv2009)Performer: 通过随机投影将Attention的复杂度线性化。
- [<font color=Blue>Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention</font>](https://0809zheng.github.io/2021/04/29/nystromformer.html)：(arXiv2102)Nyströmformer：使用Nyström方法近似自注意力运算。
- [<font color=Blue>Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks</font>](https://0809zheng.github.io/2021/08/09/external.html)：(arXiv2105)External Attention: 使用两个外部记忆单元的注意力机制。
- [<font color=Blue>Transformer Quality in Linear Time</font>](https://0809zheng.github.io/2022/03/05/flash.html)：(arXiv2202)FLASH: 基于门控注意力单元的线性Transformer。