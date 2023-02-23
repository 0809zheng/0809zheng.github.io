---
layout: post
title: 'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'
date: 2021-09-16
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/616ffb2a2ab3f51d91c39aaa.jpg'
tags: 论文阅读
---

> SqueezeNet: 与AlexNet精度相当的轻量级模型.

- paper：SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
- arXiv：[link](https://arxiv.org/abs/1602.07360)


本文提出了**SqueezeNet**，在**ImageNet**数据集上实现了**AlexNet**级别的准确率，但参数量减少了$50$倍，且压缩后模型尺寸小于$0.5$MB（是**AlexNet**的$510$倍）。在同等精度下，参数量更小的模型具有以下优势：
- 更有效的分布式训练：参数量越小，服务器之间的通信开销越少，分布式训练的可扩展性越好；
- 将新模型导出到客户端的开销更少：一些公司会使用**架空更新**(**over-the-air update**)，即定期将新模型更新到产品中。网络更新需要大量的数据传输。
- 可以部署到**FPGA**和嵌入式设备上：**FPGA**的内存通常少于$10$MB，小模型不会受到带宽限制。

# 1. SqueezeNet
**SqueezeNet**的设计思路如下：
1. 使用$1 \times 1$卷积替换$3 \times 3$卷积：参数量降低$9$倍；
2. 降低$3 \times 3$卷积的输入通道数量：进一步降低参数量；
3. 延迟下采样：将下采样放在网络后期，使早期的卷积层具有较大的特征图，在参数量受限的情况下尽可能提高分类精度。

**SqueezeNet**是由**Fire**模块组成的。**Fire**模块包括一个**squeeze**卷积层和一个**expand**层。**squeeze**卷积层使用$1\times 1$卷积，由参数$s_{1x1}$控制通道数；**expand**层使用$1\times 1$卷积和$3\times 3$卷积，由参数$e_{1x1}$和$e_{3x3}$控制通道数。通常$s_{1x1}<e_{1x1}+e_{3x3}$。

![](https://pic.imgdb.cn/item/6170c17a2ab3f51d91570b76.jpg)

**SqueezeNet**的结构如下，分别表示无跳跃连接、带有简单跳跃连接和带有复杂跳跃连接的网络。

![](https://pic.imgdb.cn/item/6170c2ce2ab3f51d9157bf24.jpg)

![](https://pic.imgdb.cn/item/6170c2e32ab3f51d9157ce9c.jpg)

# 2. 实验分析
作者比较了**SqueezeNet**与**AlexNet**及其压缩网络的性能。即使未压缩**SqueezeNet**，模型也小于压缩后的**AlexNet**，并保持相当的准确率。作者进一步使用深度压缩技术对**SqueezeNet**进行压缩，并没有造成明显的精度损失。这表明即使是小模型也有继续压缩的空间。

![](https://pic.imgdb.cn/item/6170c4ae2ab3f51d9158d88b.jpg)