---
layout: post
title: 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'
date: 2020-08-11
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f571067160a154a679113c6.jpg'
tags: 论文阅读
---

> ESPCN：基于PixelShuffle上采样的超分辨率网络.

- paper：Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
- arXiv：[link](https://arxiv.org/abs/1609.05158)
- note：[Is the deconvolution layer the same as a convolutional layer?](https://arxiv.org/abs/1609.07009)

作者称先做上采样（如双三次插值）再进行卷积操作的超分辨率模型为**high-resolution(HR) networks**；对应的先进行卷积操作再用可学习的上采样的超分辨率模型为**low-resolution(LR) networks**。

作者提出了一种**LR**网络模型，通过被称为**sub-pixel conv**（也称**PixelShuffle**）的上采样方法进行图像超分辨率任务。

# 1. Transposed convolution and sub­pixel convolutional layers
**反卷积（deconvolution）**，也被称为**转置卷积（transposed convolutional）**、**fractional convolutional**、**inverse, up or backward convolutional**。本节比较卷积、转置卷积和子像素卷积的区别。

以$1D$空间为例。

下图是一个步长为$2$的卷积操作。长度为$8$的输入信号$x$经过两端填充$2$的**padding**之后，使用长度为$4$的滤波器$f$进行卷积操作得到长度为$5$的输出信号$y$。这是一个下采样的过程。该操作可以被表示为一个矩阵运算，灰色部分表示为$0$的元素。

![](https://pic.downk.cc/item/5f5716e6160a154a679297e7.jpg)

下图是一个步长为$2$的经过**cropping**的转置卷积操作。该操作引入的矩阵恰好是卷积操作中矩阵的转置，转置卷积因此得名。需要注意的是，对应卷积中的**padding**操作，此处需要引入**cropping**操作去掉输出信号$y$两端的值。

![](https://pic.downk.cc/item/5f571a46160a154a679367fb.jpg)

下图是一个步长为$\frac{1}{2}$的子像素卷积操作。想象在输入像素中间插入了子像素（**sub­pixel**）。对比得到转置卷积和子像素卷积的结果除了顺序互逆之外完全相同。因此这两种操作可以学到类似的结果。

![](https://pic.downk.cc/item/5f571a54160a154a67936ae2.jpg)

# 2. Deconvolution layer vs Convolution in LR
本节说明**sub­pixel**卷积和在**LR**空间中进行卷积的等价关系。具体地，使用尺寸为$(or^2,i,k,k)$的卷积核进行卷积等价于使用尺寸为$(o,i,kr,kr)$的卷积核进行**sub­pixel**卷积。

下图是使用尺寸为$(4,1,2,2)$的卷积核进行卷积，并对输出特征进行**periodic shuffling**得到$2$倍尺寸的输出特征：

![](https://pic.downk.cc/item/5f571e1d160a154a679447b1.jpg)

下图是使用尺寸为$(1,1,4,4)$的卷积核进行**sub­pixel**卷积得到$2$倍尺寸的输出特征：

![](https://pic.downk.cc/item/5f571e0d160a154a67944492.jpg)

两个输出特征完全相同。这表明上采样操作可以通过卷积和周期重排实现。

下图是将尺寸为$(9,32,3,3)$的卷积核变为尺寸为$(1,32,9,9)$的卷积核的可视化过程：

![](https://pic.downk.cc/item/5f571e64160a154a67945740.jpg)

# 3. What does this mean?

![](https://pic.downk.cc/item/5f571f93160a154a6794984c.jpg)

该方法可以把尺寸为$(c×r^2,H,W)$的特征图变为尺寸为$(c,H×r,W×r)$的特征图，实现了上采样过程。具体步骤包括卷积和通道的周期重排。

`Pytorch`中给出了实现：
```
class torch.nn.PixleShuffle(upscale_factor)
```
