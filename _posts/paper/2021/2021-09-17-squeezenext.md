---
layout: post
title: 'SqueezeNext: Hardware-Aware Neural Network Design'
date: 2021-09-17
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6170ccce2ab3f51d915dd4b5.jpg'
tags: 论文阅读
---

> SqueezeNext: 针对硬件特性的神经网络设计.

- paper：SqueezeNext: Hardware-Aware Neural Network Design
- arXiv：[link](https://arxiv.org/abs/1803.10615)

作者提出了**SqueezeNext**，相比于**AlexNet**参数量减少$112$倍，相比于**VGG19**参数量减少$31$倍，相比于**MobileNet**参数量减少$1.3$倍，且避免使用在移动处理器上低效的深度可分离卷积。

相比于**SqueezeNet**，**SqueezeNext**做了如下改进：
1. 引入两层**squeeze**卷积层($1\times 1$卷积)，大幅减少特征通道数；
2. 将$3\times 3$卷积拆分成串联的$3\times 1$卷积和$1\times 3$卷积，而不是之前并联的$1\times 1$卷积和$3\times 3$卷积；
3. 引入残差连接，可以训练更深的网络；
4. 通过模拟在多处理器嵌入式系统上的性能，优化网络结构。

下面给出**ResNet**模块、**SqueezeNet**模块和**SqueezeNext**模块的对比：

![](https://pic.imgdb.cn/item/6170d0772ab3f51d91602bca.jpg)

**SqueezeNext**中使用的网络模块的设计思路如下：
- 低秩卷积核：将$K\times K$卷积分解为$K\times 1$卷积和$1\times K$卷积，将参数量从$K^2$降低为$2K$，同时增加了网络深度。
- 瓶颈模块：使用两层$1\times 1$卷积降低输入特征的通道数，并在输出端使用$1\times 1$卷积恢复通道数。
- 全连接层：网络参数主要在全连接层，在全连接层之前使用$1\times 1$卷积降低通道数，从而降低全连接层的输入维度。

![](https://pic.imgdb.cn/item/6170d2c22ab3f51d9161c3db.jpg)

一个完整的**SqueezeNext**结构如下，首先使用一个卷积和池化层降低特征的空间尺寸，增加通道数；然后分别在$4$种通道数下堆叠$(6,6,8,1)$个**SqueezeNext**模块，最后使用平均池化和全连接层输出预测结果。

![](https://pic.imgdb.cn/item/6170d6502ab3f51d91644e4d.jpg)