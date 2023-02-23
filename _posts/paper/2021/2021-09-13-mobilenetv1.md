---
layout: post
title: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'
date: 2021-09-13
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/613ab6b344eaada739491b51.jpg'
tags: 论文阅读
---

> MobileNet: 使用深度可分离卷积构造轻量网络.

- paper：MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- arXiv：[link](https://arxiv.org/abs/1704.04861)

作者提出了**MobileNet**，使用深度可分离卷积代替普通卷积，在准确率轻微下降的情况下极大地减少了模型的参数量和计算量，从而使得模型可以部署在移动端。

# 1. 深度可分离卷积

![](https://pic.imgdb.cn/item/613ab7e044eaada7394a87a7.jpg)

普通的卷积操作能够将一个尺寸为$D_H \times D_W \times M$的特征图转换为尺寸为$D_H \times D_W \times N$的特征图。实现过程是使用$N$个尺寸为$D_K \times D_K \times M$的卷积核。卷积操作的参数量(**Parameters**)为：

$$ \text{params}_{\text{conv}} = N \times D_K \times D_K \times M $$

若计每一次乘加运算(**Mult-Adds**)为一次计算量，则卷积操作的计算量为（不考虑偏置）：

$$ \text{macs}_{\text{conv}} = D_K \times D_K \times M \times D_H \times D_W \times N $$

**深度可分离卷积**(**depthwise separable convolution**)把普通卷积近似拆分成两部分：**深度卷积**(**depthwise convolution**)和**逐点卷积**(**pointwise convolution**)。深度可分离卷积也能够实现将尺寸为$D_H \times D_W \times M$的特征图转换为尺寸为$D_H \times D_W \times N$的特征图。

深度卷积是指对输入特征图的每一个通道使用一个单通道卷积核进行处理，从而实现通道独立、空间交互的卷积操作。对于输入尺寸为$D_H \times D_W \times M$的特征图，使用$M$个尺寸为$D_K \times D_K$的卷积核，生成尺寸为$D_H \times D_W \times M$的中间特征图。深度卷积的参数量和运算量分别为：

$$ \text{params}_{\text{dconv}} = M \times D_K \times D_K  $$

$$ \text{macs}_{\text{dconv}} = D_K \times D_K \times D_H \times D_W \times M $$

逐点卷积是指对输入特征图的每一个空间位置使用一个$1\times 1$卷积核进行处理，从而实现空间独立、通道交互的卷积操作。对于输入尺寸为$D_H \times D_W \times M$的中间特征图，使用$N$个尺寸为$1 \times 1 \times M$的卷积核，生成尺寸为$D_H \times D_W \times N$的输出结果。逐点卷积的参数量和运算量分别为：

$$ \text{params}_{\text{pconv}} = N \times 1 \times 1 \times M  $$

$$ \text{macs}_{\text{pconv}} = 1 \times 1 \times M \times D_H \times D_W \times N $$

则深度可分离卷积的参数量和运算量分别为：

$$ \text{params}_{\text{dsconv}} = M \times (D_K \times D_K+N)  $$

$$ \text{macs}_{\text{dsconv}} = M \times D_H \times D_W \times (D_K \times D_K+N) $$

深度可分离卷积与普通卷积的参数量之比为：

$$ \frac{\text{params}_{\text{dsconv}}}{\text{params}_{\text{conv}}} = \frac{M \times (D_K \times D_K+N)}{N \times D_K \times D_K \times M} = \frac{1}{N}+ \frac{1}{D_K^2} $$

深度可分离卷积与普通卷积的计算量之比为：

$$ \frac{\text{macs}_{\text{dsconv}}}{\text{macs}_{\text{conv}}} = \frac{M \times D_H \times D_W \times (D_K \times D_K+N)}{D_K \times D_K \times M \times D_H \times D_W \times N} = \frac{1}{N}+ \frac{1}{D_K^2} $$

综上所述，深度可分离卷积的参数量和计算量仅为普通卷积的$\frac{1}{N}+ \frac{1}{D_K^2}$。通常$N$较大，而$D_K$选用$3$，则前者只有后者的$1/9$。

# 2. MobileNet
**MobileNet**把标准的$3\times 3$卷积模块替换成了深度可分离卷积模块，如下图所示。

![](https://pic.imgdb.cn/item/613ac32e44eaada739590d11.jpg)

其中的激活函数**ReLU**替换为**ReLU6**，即限制了**ReLU**的最大输出为$6$。这是为了在移动端设备使用低精度的数据类型(**float16/int8**)时，也能有较好的数值分辨率。否则无上界的激活值可能分布在一个很大的范围内，低精度数据可能会造成精度损失。

$$ \text{ReLU6}(x)=\min(\max(x,0),6) =\begin{cases} 6, & x\geq 6 \\ x, & 0\leq x<6 \\ 0, &x<0 \end{cases} $$

作者设计的**MobileNet**的基本结构如下表所示，该网络极大地降低了计算量和参数量，在分类任务上准确率仅下降了$1.1\%$。

![](https://pic.imgdb.cn/item/613affdd44eaada739b0bda6.jpg)

通过统计，网络的计算量主要来源于逐点卷积($94.86\%$)，参数量主要来源于逐点卷积($74.59\%$)和全连接层($24.33\%$)。

