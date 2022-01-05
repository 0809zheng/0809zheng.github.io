---
layout: post
title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
date: 2021-09-11
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61b303ae2ab3f51d9165bc0a.jpg'
tags: 论文阅读
---

> EfficientNet: 重新考虑卷积神经网络的缩放.

- paper：EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- arXiv：[link](https://arxiv.org/abs/1905.11946)

本文作者研究了卷积神经网络模型的缩放，即改变模型的深度、宽度和分辨率。通过均匀地缩放这三个维度，作者使用神经结构搜索设计了**EfficientNet**，在**ImageNet**数据集上获得
**84.4% top-1 / 97.1% top-5**的表现，并且比目前最好的卷积网络小$8.4$倍，速度快$6.1$倍。

![](https://pic.imgdb.cn/item/61d4ed982ab3f51d91ddc9ed.jpg)

宽度缩放是指改变每一层卷积层使用的卷积核数量（即特征通道数）；深度缩放是指改变堆叠的网络层数；分辨率缩放是指改变输入图像的空间分辨率（高度和宽度）。

通常一个卷积神经网络可以划分成$s$个阶段，每个阶段的卷积层具有相同的结构；比如**ResNet**划分成$5$个阶段。若第$i$个阶段的特征$X$的空间尺寸为$(H_i,W_i)$，通道数为$C_i$；第$i$个阶段重复堆叠$L_i$个相同的卷积层$\mathcal{F}_i$，则整个卷积网络$\mathcal{N}$可以表示为：

$$ \mathcal{N} = \mathop{\bigodot}_{i=1,...,s} \mathcal{F}_i^{L_i}(X_{<H_i,W_i,C_i>}) $$


使用缩放因子可以对模型的深度、宽度和分辨率分别缩放。通过$r$可以调整模型的特征分辨率，通过$d$可以调整模型的深度，通过$w$可以调整模型的宽度。作者旨在给定资源约束的条件下获得精度尽可能高的卷积网络，表述为一个优化问题：

$$ \begin{align} \mathop{\max}_{d,w,r} & Accuracy(N(d,w,r)) \\ s.t. & N(d,w,r)=\mathop{\bigodot}_{i=1,...,s} \mathcal{F}_i^{d\cdot L_i}(X_{<r\cdot H_i,r\cdot W_i,w\cdot C_i>}) \\ & \text{Memory}(N)<\text{target memory} \\ & \text{FLOPS}(N)<\text{target flops} \end{align} $$

作者通过实验发现，单独增大网络的深度、宽度或分辨率都能够提高网络的精度，但是模型较大时精度的提升会趋于饱和。

![](https://pic.imgdb.cn/item/61d4f55d2ab3f51d91e25a38.jpg)

网络的深度、宽度或分辨率并不是相互独立的。比如对于更大分辨率的输入图像，应该增加网络的深度以获得更大的感受野。同时应增加网络的宽度以捕捉更多像素的细粒度模式。作者通过实验发现，在不同的深度和分辨率系数下，改变网络的宽度会使网络收敛到不同的精度水平。因此平衡网络的深度、宽度或分辨率相当重要。

![](https://pic.imgdb.cn/item/61d4f66b2ab3f51d91e2f546.jpg)

作者提出了一种**复合缩放**(**compound scaling**)方法，通过复合系数$\phi$均匀地缩放网络的深度、宽度或分辨率：

$$ \begin{align} \text{depth}: &d=\alpha^{\phi} \\ \text{width}: &w=\beta^{\phi} \\ \text{resolution}: &r=\gamma^{\phi} \\ s.t. &\alpha \cdot \beta^2 \cdot \gamma^2 ≈2 \\ &\alpha \geq 1, \beta \geq 1, \gamma \geq 1 \end{align} $$

常规的卷积网络的**FLOPS**与网络的深度$d$成正比，与宽度的平方$w^2$和分辨率的平方$r^2$成正比。参数$\alpha,\beta,\gamma$是通过网格搜索得到的常数，表示将可用的网络资源分配给深度、宽度和分辨率的程度。若约束$\alpha \cdot \beta^2 \cdot \gamma^2 ≈2$（初始可用资源为$2$倍），则用户指定复合系数$\phi$后网络的**FLOPS**会增长$2^{\phi}$倍。

作者通过神经结构搜索设计了基线网络**EfficientNet-B0**，搜索目标是$ACC\times [FLOPS/T]^{-0.07}$，$T$为目标FLOPS，设置为$400$M。**EfficientNet-B0**的基本结构如下。其中基本模块采用[MobileNetV2](https://0809zheng.github.io/2021/09/14/mobilenetv2.html)中的**MBConv**（**mobile inverted bottleneck**）。

![](https://pic.imgdb.cn/item/61d4fac42ab3f51d91e5af64.jpg)

通过复合缩放，可以在**EfficientNet-B0**的基础上获得更大的模型。缩放步骤如下：
1. 固定$\phi=1$，假设可用资源为$2$倍，即$\alpha \cdot \beta^2 \cdot \gamma^2 ≈2$，通过网格搜索获得$\alpha,\beta,\gamma$。本文搜索得到$\alpha=1.2,\beta=1.1,\gamma=1.15$。
2. 固定$\alpha,\beta,\gamma$，通过不同的$\phi$获得不同的网络结构。本文得到的**EfficientNet-B1**到**B7**如下。

![](https://pic.imgdb.cn/item/61d4fbef2ab3f51d91e6769d.jpg)

实验表明**EfficientNet**实现了最先进的精度，并且具有更少的参数量和FLOPS。

![](https://pic.imgdb.cn/item/61d4ff202ab3f51d91e8a264.jpg)

作者随机展示了几张不同输入图像的类别激活图。图中显示具有复合缩放的模型倾向于同时关注更多的目标和更多细节。而其他模型要么缺乏目标的细节，要么无法捕获图中的所有目标。

![](https://pic.imgdb.cn/item/61d4fe802ab3f51d91e83472.jpg)