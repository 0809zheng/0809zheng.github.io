---
layout: post
title: 'Rethinking Attention with Performers'
date: 2021-08-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61136ff95132923bf8281465.jpg'
tags: 论文阅读
---

> Performer: 通过随机投影将Attention的复杂度线性化.

- paper：Rethinking Attention with Performers
- arXiv：[link](https://arxiv.org/abs/2009.14794)

标准的**Attention**首先将输入序列$$X=[x_1,...,x_n] \in \Bbb{R}^{n×d}$$($n$个维度为$d$的特征向量，通常$n>d$)转换成查询矩阵$Q$,键矩阵$K$,值矩阵$V$：

$$ Q = XW_q \in \Bbb{R}^{n×d}, \quad W_q \in \Bbb{R}^{d×d} $$

$$ K = XW_k \in \Bbb{R}^{n×d}, \quad W_k \in \Bbb{R}^{d×d} $$

$$ V = XW_v \in \Bbb{R}^{n×d}, \quad W_v \in \Bbb{R}^{d×d} $$

并通过下式计算自注意力，对于第$i$个输入$x_i$，其输出计算为:

$$ \text{Attention}(Q,K,V)_i=(\text{softmax}(\frac{QK^T}{\sqrt{d}})V)_i=\frac{\sum_{j=1}^{n}e^{\frac{q_i^Tk_j}{\sqrt{d}}}v_j}{\sum_{j=1}^{n}e^{\frac{q_i^Tk_j}{\sqrt{d}}}}=\frac{\sum_{j=1}^{n}\text{sim}(q_i,k_j)v_j}{\sum_{j=1}^{n}\text{sim}(q_i,k_j)} $$

上式可以表示为使用查询向量和键向量计算相似度函数$\text{sim}(q,k)≥0$，并对值向量进行加权求和。对于标准的点积自注意力机制，相似度函数选用：

$$ \text{sim}(q_i,k_j) = e^{\frac{q_i^Tk_j}{\sqrt{d}}} $$

作者的出发点还是上述标准自注意力，为简化省略了缩放因子，即表示为$\text{sim}(q_i,k_j) = e^{q_i\cdot k_j}$。若希望将计算复杂度线性化，则寻找新的向量满足：

$$ \text{sim}(q,k) ≈ \tilde{q}\cdot\tilde{k} $$

作者找到了一种映射：

$$ e^{q\cdot k} = \Bbb{E}_{\omega \text{~} \mathcal{N}(\omega;0,1_d)} [e^{w\cdot q-||q||^2/2}\times e^{w\cdot k-||k||^2/2}] \\ ≈ \frac{1}{\sqrt{m}} \begin{pmatrix} e^{w_1\cdot q-||q||^2/2} \\ e^{w_2\cdot q-||q||^2/2} \\ \cdots \\ e^{w_m\cdot q-||q||^2/2} \end{pmatrix} \cdot \frac{1}{\sqrt{m}} \begin{pmatrix} e^{w_1\cdot k-||k||^2/2} \\ e^{w_2\cdot k-||k||^2/2} \\ \cdots \\ e^{w_m\cdot k-||k||^2/2} \end{pmatrix} \\ = \tilde{q}\cdot\tilde{k} $$

上式表示从标准正态分布$\mathcal{N}(\omega;0,1_d)$中采样足够多的$\omega$，然后计算$e^{w\cdot q-||q||^2/2}\times e^{w\cdot k-||k||^2/2}$的数学期望，结果等于$e^{q\cdot k}$。实际中$\omega$只能采集有限个，因此采集$m$个并作上述近似。上式将两个$d$维向量的内积的指数$e^{q\cdot k}$转化为两个$m$维向量的内积$\tilde{q}\cdot\tilde{k}$，则对应的自注意力计算为：

$$ \text{Attention}(Q,K,V)_i=\frac{\sum_{j=1}^{n}(\tilde{q}_i\cdot\tilde{k}_j)v_j}{\sum_{j=1}^{n}\tilde{q}_i\cdot\tilde{k}_j} = \frac{\tilde{q}_i\sum_{j=1}^{n}\tilde{k}_j\cdot v_j}{\tilde{q}_i\sum_{j=1}^{n}\tilde{k}_j}  $$

根据矩阵乘法的结合律，计算复杂度为$O(n^2)$的$QK^T$变为计算复杂度为$O(n)$的$\tilde{K}V^T$，从而将自注意力运算线性化。

该映射推导如下。注意到$e^{q\cdot k}$可以改写为：

$$ e^{q\cdot k} = e^{||q||^2/2+||k||^2/2-||q-k||^2/2} $$

对$e^{-\|\|q-k\|\|^2/2}$做傅里叶变换：

$$ \mathcal{F}(e^{-||q-k||^2/2})=\frac{1}{\sqrt{2\pi}}\int_{}^{} e^{-||q-k||^2/2}e^{-i\omega (q-k)}d(q-k)=\frac{1}{\sqrt{2\pi}}\int_{}^{} e^{-||t||^2/2}e^{-i\omega t}dt \\ = \frac{1}{\sqrt{2\pi}}\int_{}^{} e^{-(||t||^2+2i\omega t-\omega^2+\omega^2)/2}dt = e^{-\omega^2/2}\frac{1}{\sqrt{2\pi}}\int_{}^{} e^{-(t+2i\omega)^2/2}dt = e^{-\omega^2/2}  $$

再应用傅里叶逆变换：

$$ e^{-||q-k||^2/2} = \mathcal{F}^{-1}(e^{-\omega^2/2})=\frac{1}{\sqrt{2\pi}}\int_{}^{} e^{-\omega^2/2}e^{i\omega (q-k)}d\omega  $$

代入原式得：

$$ e^{q\cdot k} = e^{||q||^2/2+||k||^2/2}\frac{1}{\sqrt{2\pi}}\int_{}^{} e^{-\omega^2/2}e^{i\omega (q-k)}d\omega \\ = \frac{e^{||q||^2/2+||k||^2/2}}{\sqrt{2\pi}}\int_{}^{} e^{-\omega^2/2+i\omega (q-k)}d\omega $$

对于上式，若令$q \to -iq,k \to ik$，则有：

$$ e^{q\cdot k}  = \frac{e^{-||q||^2/2-||k||^2/2}}{\sqrt{2\pi}}\int_{}^{} e^{-\omega^2/2+\omega (q+k)}d\omega \\ = \int_{}^{} \frac{e^{-\omega^2/2}}{\sqrt{2\pi}}e^{\omega q-||q||^2/2}\cdot e^{\omega k-||k||^2/2}d\omega \\ = \Bbb{E}_{\omega \text{~} \mathcal{N}(\omega;0,1_d)} [e^{w\cdot q-||q||^2/2}\times e^{w\cdot k-||k||^2/2}] $$


$\omega_1,\omega_2,...,\omega_m$是独立地从标准正态分布$\mathcal{N}(\omega;0,1_d)$中采样得到的。作者指出，若将$\omega_i$进行正交化，能够有效地降低估计方差，提高估计的平均精度。这是因为分布$\mathcal{N}(\omega;0,1_d)$是各向同性的，即在方向上是均匀的，向量正交化使得采样结果更均匀，从而降低估计方差。值得一提的是，上述正交化仅对$m≤d$有效；若$m>d$，则每$d$个向量进行分组，组内正交化。

作者使用上述机制设计了**Performer**，当输入序列长度$n$较大时(比如超过$2048$)，比标准的**Transformer**具有明显的速度优势：

![](https://pic.imgdb.cn/item/611389405132923bf85d49d8.jpg)

作者比较了不同设置下输出的近似程度。采用正交化采样比独立采样更有效；本文所提方法比直接把$e^{q\cdot k}$展开为**sin,cos**函数的形式更精确：

![](https://pic.imgdb.cn/item/61138ae25132923bf8608d4a.jpg)

理论上**Performer**与标准的**Transformer**可以相互转换。但实验表明**Performer**直接加载**Transformer**的权重效果较差，经过微调后能够获得较高的准确率：

![](https://pic.imgdb.cn/item/61138c5c5132923bf86376c3.jpg)