---
layout: post
title: 'Jigsaw Clustering for Unsupervised Visual Representation Learning'
date: 2022-10-31
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63e4a5b14757feff331ec02f.jpg'
tags: 论文阅读
---

> 无监督视觉表示学习的拼图聚类方法.

- paper：[Jigsaw Clustering for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2104.00323)

本文提出了一种用于无监督表示学习的拼图聚类方法。把一批图像拆分成$m\times m$的图像块，打乱后构成一批新的图像。则训练目标是把打乱的图像恢复成原始图像。

![](https://pic.imgdb.cn/item/63e4aa3c4757feff332813d5.jpg)

作者设计了两种无监督损失。把打乱的图像通过卷积网络提取特征后，把特征拆分成$m\times m$的特征向量，每个特征向量对应一个输入图像块。

![](https://pic.imgdb.cn/item/63e4aad54757feff33292620.jpg)

基于对比学习可以构造聚类损失。对于每一个图像块特征$z_i$，属于同一个原始图像的特征$z_j$为正样本，其余特征为负样本；则可构造聚类损失：

$$ \mathcal{L}_{clu} = \frac{1}{nmm} \sum_i \frac{1}{mm-1} \sum_{j \in C_i} - \log \frac{\exp(\cos(z_i,z_j)/\tau)}{\sum_{k=1,...,nmm;k \neq i}\exp(\cos(z_i,z_k)/\tau)} $$

基于交叉熵可以构造定位损失。每个图像块的位置信息已知的，因此可以构造一个$mm$分类问题，并构造定位损失：

$$  \mathcal{L}_{loc} = CrossEntropy(L,L_{gt}) $$

作者通过实验发现，在切分图像时保留一定的重叠会有更好的效果。切分后对每个图像块单独应用数据增强也能提高表现。

![](https://pic.imgdb.cn/item/63e4af7d4757feff3331162e.jpg)