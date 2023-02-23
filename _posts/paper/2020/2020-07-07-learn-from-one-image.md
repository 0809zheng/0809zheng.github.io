---
layout: post
title: 'A critical analysis of self-supervision, or what we can learn from a single image'
date: 2020-07-07
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f03f01514195aa594635775.jpg'
tags: 论文阅读
---

> 使用单张图像进行自监督学习.

- paper：A critical analysis of self-supervision, or what we can learn from a single image
- arXiv：[link](https://arxiv.org/abs/1904.13132v2)


作者通过实验发现：
1. 深度网络浅层的权重包含有限的自然图像的统计信息；
2. 浅层的统计信息可以通过自监督学习，和监督学习获得的效果是类似的；
3. 浅层的统计信息能够从单张图像及其合成转换（数据增强）中获得，而不需要大量数据集。

作者选用的三张训练图像如下：

![](https://pic.downk.cc/item/5f03e64e14195aa5945f8f0d.jpg)

作者使用了大量图像增强的方法，训练了三类自监督模型：
- 生成模型，使用**BiGAN**。
- 旋转，使用**RotNet**，通过旋转图像产生伪标签。
- 聚类，使用**DeepCluster**，通过图像聚类产生伪标签。

作者选择从浅到深的五层卷积神经网络特征（$conv1$~$conv5$），通过预训练产生特征后使用**linear probes**来测试这些特征的好坏。由于**linear probes**是一个线性分类模型，其模型复杂度低，因此分类结果主要取决于特征。

作者通过实验发现，对于卷积网络的浅层特征，使用单张图像的自监督学习学到的特征表示要优于使用大量图像的自监督学习，甚至超过了监督学习的结果；随着网络层数的加深，学习到的特征表示越来越差：

![](https://pic.downk.cc/item/5f03e9f314195aa59460fec8.jpg)

![](https://pic.downk.cc/item/5f03ea5f14195aa594612d24.jpg)

作者还可视化了使用单张图像训练得到的$conv1$卷积核特征：

![](https://pic.downk.cc/item/5f03eab014195aa594614e7c.jpg)
