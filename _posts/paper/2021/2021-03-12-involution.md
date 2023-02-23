---
layout: post
title: 'Involution: Inverting the Inherence of Convolution for Visual Recognition'
date: 2021-03-12
author: 郑之杰
cover: 'https://img.imgdb.cn/item/604ac4f15aedab222c7f2e94.jpg'
tags: 论文阅读
---

> Involution：空间独立通道共享的卷积核.

- paper：Involution: Inverting the Inherence of Convolution for Visual Recognition
- arXiv：[link](https://arxiv.org/abs/2103.06255)

# 1. 卷积神经网络
卷积是用于构建视觉任务的神经网络结构的主要组件。假设输入特征是$X \in R^{H \times W \times C_i}$，其中$H$,$W$,$C_i$分别表示高度、宽度和输入通道数；其中的每个空间像素$X_{i,j} \in R^{C_i}$都可看做具有丰富语义信息(在通道中)的特征。记尺寸为$K \times K$的卷积核为$F \in R^{C_o \times C_i \times K \times K}$，通过滑动窗口的方式作用于输入特征，计算输出特征$Y \in R^{H \times W \times C_o}$。空间位置$(i,j)$的第$k$个通道的输出特征计算过程如下：

$$ Y_{i,j,k} = \sum_{c=1}^{C_i} \sum_{(u,v) \in \Delta_{K}}^{} F_{k,c,u+\lfloor K/2 \rfloor,v+\lfloor K/2 \rfloor}X_{i+u,j+v,c} $$

其中$\Delta_{K}$为对中心像素进行卷积的邻域偏移量集合，用笛卡尔积表示为：

$$ \Delta_{K} = [-\lfloor K/2 \rfloor,...,\lfloor K/2 \rfloor] \times [-\lfloor K/2 \rfloor,...,\lfloor K/2 \rfloor] $$

卷积操作具有两个性质：**spatial-agnostic**和**channel-specific**。**spatial-agnostic**是指在不同的空间位置共享卷积核，实现了平移不变性，这有助于捕捉与空间位置无关的视觉特征；但是这种性质阻碍了卷积核在不同空间位置适应不同视觉模式的能力。**channel-specific**是指卷积在不同通道具有不同的值，用于收集不同的语义信息；但是这种性质受到通道间冗余的影响，限制了卷积核的灵活性。

# 2. Involution
作者设计了一种新的卷积形式：**involution**。与标准卷积操作相反，新的卷积操作具有**spatial-specific**和**channel-agnostic**的特点。**spatial-specific**是指**involution**卷积在不同空间位置具有不同的值，能够捕捉更丰富的空间信息。**channel-agnostic**是指在不同的通道中共享卷积核，减少卷积核的冗余。

![](https://img.imgdb.cn/item/604ad1775aedab222c849949.jpg)

**involution**卷积的实现如上图所示。**involution**卷积核$H_{i,j,:,:,g} \in R^{K \times K}$是专门为特征位置$(i,j)$定制的，可以应用$G$个，其空间位置$(i,j)$的第$k$个通道的输出特征计算过程如下：

$$ Y_{i,j,k} = \sum_{(u,v) \in \Delta_{K}}^{} H_{i,j,u+\lfloor K/2 \rfloor,v+\lfloor K/2 \rfloor, \lceil kG/C \rceil}X_{i+u,j+v,k} $$

**involution**卷积核$H_{i,j} \in R^{K \times K}$的生成是由空间像素$X_{i,j} \in R^{C}$得到的。引入线性变换$W_0 \in R^{\frac{C}{r} \times C}$和$W_1 \in R^{(K \times K \times G) \times \frac{C}{r}}$作为**bottleneck**和非线性激活函数，则核参数计算如下：

$$ H_{i,j} = W_1 \sigma (W_0 X_{i,j}) $$

**Pytorch**风格的伪代码如下：

![](https://img.imgdb.cn/item/604ad6435aedab222c86add5.jpg)

# 3. RedNet
作者在**ResNet**的主干网络上使用**involution**卷积核替换$3 \times 3$卷积，但保留了$1 \times 1$卷积用于通道映射和融合，从而构造出一种高效**Backbone**网络，称为**RedNet**。在图像分类、目标检测和语义分割任务上取得了更好的性能。

![](https://img.imgdb.cn/item/604ad7345aedab222c873dd9.jpg)



