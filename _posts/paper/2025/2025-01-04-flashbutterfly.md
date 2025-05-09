---
layout: post
title: 'Simple Hardware-Efficient Long Convolutions for Sequence Modeling'
date: 2025-01-04
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/681d738558cb8da5c8e7a2ed.png'
tags: 论文阅读
---

> 用于序列建模的简单的硬件高效长卷积.

- paper：[Simple Hardware-Efficient Long Convolutions for Sequence Modeling](https://arxiv.org/abs/2302.06646)


# 0. TL; DR

本文提出了一种简单高效的长卷积方法，用于序列建模。通过直接学习长卷积核，该方法在长序列建模任务中达到了与状态空间模型（**SSMs**）相当的性能，同时避免了复杂的初始化和训练技巧。此外，作者还开发了一种名为**FlashButterfly**的**IO**感知算法，通过**Butterfly**分解减少**GPU**内存**IO**并提高**FLOP**利用率，显著提升了长卷积的运行效率。在多个任务上，该方法不仅匹配甚至超越了**SSMs**的性能，还展示了更好的稳定性和扩展性。

# 1. 背景介绍
近年来，基于状态空间模型（**SSMs**）的序列模型因其在长序列建模中的高效性和优越性能而受到关注。**SSMs**在长序列建模任务中表现出色，但在实际应用中，它们依赖于复杂的数学结构和精心设计的初始化方法，这使得训练过程变得复杂且难以调优。因此，一个自然的问题是：是否可以直接参数化长卷积核，以简化模型并提高效率？

长卷积在序列建模中的应用面临着两个主要挑战：质量和运行效率。一方面，直接参数化的长卷积核在性能上通常不如**SSMs**；另一方面，尽管长卷积可以通过快速傅里叶变换（**FFT**）在$O(N \log N)$的时间复杂度内计算，但在现代硬件上，由于**GPU**内存**IO**的限制，其运行效率往往不如二次算法（如自注意力机制）。本文旨在通过简单的正则化技术和**IO**感知的卷积算法解决这些挑战。

# 2. 方法介绍

![](https://pic1.imgdb.cn/item/681db03858cb8da5c8e8d1f9.png)

## 2.1 长卷积的正则化技术
为了提高长卷积在序列建模中的性能，作者首先研究了长卷积核的平滑性。通过可视化卷积核，发现直接学习的长卷积核是非平滑的，而**SSMs**的卷积核则是平滑的。

![](https://pic1.imgdb.cn/item/681d9a2f58cb8da5c8e8781d.png)

为了使长卷积核平滑，作者提出了两种简单的正则化技术：**Squash**和**Smooth**。

- **Squash操作**：通过减少卷积核权重的幅度来强制稀疏性，从而在频域中实现平滑性。引入超参数$\lambda$控制平滑的程度，**Squash**操作定义为：

$$
  K = \text{sign}(K) \odot \max(|K| - \lambda, 0)
$$


- **Smooth操作**：通过在时间域中对卷积核权重应用简单的平均池化来促进频域中的平滑性。引入池化的宽度$p$，**Smooth**操作定义为：

$$
  K_k = \frac{1}{2p + 1} \sum_{j=1}^{2p+1} K_{k+j-p}
$$

通过这些正则化技术，长卷积在多个任务上的性能得到了显著提升，甚至超过了**SSMs**。

![](https://pic1.imgdb.cn/item/681dad9c58cb8da5c8e8c33a.png)

## 2.2 FlashButterfly算法
为了提高长卷积的运行效率，作者开发了**FlashButterfly**算法。**FlashButterfly**通过**Butterfly**分解将**FFT**卷积重写为一系列块稀疏**Butterfly**矩阵，从而减少**GPU**内存**IO**并提高**FLOP**利用率。

**Butterfly**分解是一种经典的**FFT**算法，它将**FFT**分解为一系列块对角**Butterfly**矩阵。具体来说，对于长度为N的序列，**FFT**可以分解为：

$$
F_N = P (I_{N_2} \otimes F_{N_1}) P^T D (I_{N_1} \otimes F_{N_2}) P
$$

其中，$P$是置换矩阵，$D$是对角矩阵，$I_{N_i}$和$F_{N_i}$分别是大小为$N_i \times N_i$的单位矩阵和**DFT**矩阵。

为了进一步减少长序列的**IO**需求，作者提出了一个三步算法。该算法将**FFT**卷积分解为三个步骤：一个**Butterfly**矩阵乘法（可以单次读取输入序列），多个并行的**FFT**卷积，以及一个最终的**Butterfly**矩阵乘法（也可以单次读取输入序列）。具体步骤如下：
1. **Butterfly矩阵乘法**：将输入序列$u$与**Butterfly**矩阵$B^{-1}$相乘。
2. **并行FFT卷积**：对中间结果进行多个并行的FFT卷积。
3. **最终Butterfly矩阵乘法**：将并行FFT卷积的结果与**Butterfly**矩阵$B$相乘。

通过这种方式，整个卷积可以在三次读取输入序列的情况下完成，显著减少了**IO**需求。

![](https://pic1.imgdb.cn/item/681dad8a58cb8da5c8e8c2b7.png)

**FlashButterfly**算法不仅提高了长卷积的运行效率，还通过学习**Butterfly**矩阵的参数进一步提高了模型的表达能力。具体来说，作者提出了一个扩展，允许模型学习**Butterfly**矩阵中的参数，而不是使用固定的**FFT**矩阵。这种扩展在不增加额外**FLOPS**的情况下增加了模型的参数数量，从而提高了模型的性能。

# 3. 实验分析

### 3.1 长序列建模：长范围竞技场（LRA）
作者在长范围竞技场（**LRA**）基准测试中评估了长卷积的性能。**LRA**包含六个长序列建模任务，涵盖了文本、自然图像和合成图像等多种模态。实验结果表明，经过正则化处理的长卷积在所有任务上的性能都接近或超过了**SSMs**。此外，长卷积对初始化的选择更为鲁棒，即使在完全随机初始化的情况下，也能达到与几何衰减初始化相当的性能。

![](https://pic1.imgdb.cn/item/681dae4258cb8da5c8e8c7f4.png)

### 3.2 图像分类
作者进一步在图像分类任务上评估了长卷积的性能。实验包括一维像素级图像分类和二维图像分类。在**1D**图像分类任务中，长卷积在随机初始化的情况下达到了91.0%的准确率，使用几何衰减初始化时达到了92.1%。在**2D**图像分类任务中，长卷积达到了89.1%的准确率，接近**S4ND**模型的89.9%。这些结果表明，长卷积在图像分类任务中具有很强的竞争力。

![](https://pic1.imgdb.cn/item/681dae6d58cb8da5c8e8c9a0.png)

### 3.3 文本建模
在文本建模任务中，作者使用**OpenWebText**和**The Pile**数据集评估了长卷积的性能。实验结果表明，长卷积在**OpenWebText**上的性能接近**SSMs**，测试困惑度达到了**19.9 PPL**。在**The Pile**数据集上，长卷积在训练了15B个token后，测试困惑度达到了**10.3 PPL**，与**SSMs**相当。这些结果表明，长卷积在文本建模任务中具有很强的竞争力。

![](https://pic1.imgdb.cn/item/681dae9858cb8da5c8e8cb3e.png)

### 3.4 脑fMRI分析
作者还在脑**fMRI**分析任务中评估了长卷积的性能。实验结果表明，长卷积在预测脑活动的任务中表现优于**Transformer**和**SSMs**，平均绝对误差（**MAE**）达到了0.54。

![](https://pic1.imgdb.cn/item/681daec058cb8da5c8e8ccb0.png)

### 3.5 FlashButterfly的运行效率
作者在长范围竞技场（**LRA**）速度基准测试中评估了**FlashButterfly**的运行效率。实验结果表明，**FlashButterfly**在标准序列建模工作负载中的运行速度比**Transformer**、**FlashAttention**、**SSMs**快。

![](https://pic1.imgdb.cn/item/681daf2e58cb8da5c8e8cee2.png)

此外，**FlashButterfly**在长序列任务中也表现出色，例如在**Path256**任务中，**FlashButterfly**的训练时间比之前的最佳模型快7.2倍，同时在64K序列长度的任务中达到了92.2%的准确率。

![](https://pic1.imgdb.cn/item/681daf4658cb8da5c8e8cf34.png)

### 3.6 学习Butterfly扩展
作者在顺序**CIFAR**和**WikiText103**任务中评估了学习**Butterfly**扩展的性能。在顺序**CIFAR**任务中，使用学习**Butterfly**扩展的长卷积达到了92.5%的准确率，比固定**Butterfly**矩阵的基线提高了1.5个百分点。在**WikiText103**任务中，作者用学习**Butterfly**扩展的长卷积替换了**Transformer**中的**MLP**层，结果表明，该扩展在参数数量减少30%的情况下，测试困惑度达到了**20.4 PPL**，优于基线**Transformer**的**20.6 PPL**。

![](https://pic1.imgdb.cn/item/681daf7658cb8da5c8e8cfd6.png)