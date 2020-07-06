---
layout: post
title: 'Group Normalization'
date: 2020-06-28
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f02876214195aa594bff838.jpg'
tags: 论文阅读
---

> 深度学习中的组归一化方法.

- paper：Group Normalization
- arXiv：[link](https://arxiv.org/abs/1803.08494)

**Batch Norm**已经证明对深度网络中的特征进行归一化能够加速训练。作者研究**VGG**网络的**Conv5_3**卷积层特征，将其特征数值按从小到大进行排列，在训练中绘制第1、20、80和99个特征值的变化：

![](https://pic.downk.cc/item/5f02888314195aa594c08d20.jpg)

由上图发现，当不引入归一化方法时，特征的变化很大；引入归一化方法能够控制特征的分布，使其更接近正态化。

**Batch Norm**最大的问题在于依赖于**batch size**，当**batch size**较小的时候计算统计量会有比较大的偏差，从而影响结果。作者提出了**Group Norm**，对每一个样本的一组特征通道计算统计量，其几乎不会受到**batch size**的影响：

![](https://pic.downk.cc/item/5f0289e414195aa594c1565b.jpg)

几种不同的**Norm**方法实现如下，其中$N$是**批量轴 batch axis**, $C$是**通道轴 channel axis**, $(H, W)$是**空间轴 spatial axes**。

![](https://pic.downk.cc/item/5f02894714195aa594c0ff90.jpg)

tensorflow形式的**Group Norm**实现如下，实现过程非常简单：

![](https://pic.downk.cc/item/5f028a7714195aa594c1abac.jpg)

**Group Norm**的一个弊端是额外引入了超参数$G$：

![](https://pic.downk.cc/item/5f028aa714195aa594c1c66e.jpg)
