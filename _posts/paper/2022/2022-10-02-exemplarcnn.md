---
layout: post
title: 'Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks'
date: 2022-10-02
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63c0d642be43e0d30e782142.jpg'
tags: 论文阅读
---

> 通过Exemplar-CNN实现判别无监督特征学习.

- paper：[Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks](https://arxiv.org/abs/1406.6909)

对于图像数据集，在图像中增加小的扰动不会改变图像的原始语义信息或几何形式，因此轻微扰动的图像被认为和原始图像是相同的，卷积神经网络应该从中学习到对扰动具有不变性的特征。

本文作者提出了**Exemplar-CNN**，使用无标签图像中的图像块构造了一个代理训练集，以进行无监督的特征表示学习。

**Exemplar-CNN**从图像数据集的梯度较大区域（通常覆盖边缘并包含目标的一部分）中采样$32 \times 32$大小的图像块，把这些图像块称作**exemplary patch**。对每一个图像块应用不同的随机图像增强，同一个图像块的增强样本属于同一个代理类别。自监督学习的前置任务旨在区分不同的代理类别。理论上可以任意创造足够多的代理类别。

![](https://pic.imgdb.cn/item/63c0d8a9be43e0d30e7cc5b5.jpg)

作者报告了不同数量的代理类别对下游分类任务的影响：

![](https://pic.imgdb.cn/item/63c0d989be43e0d30e7ed1d1.jpg)

作者还报告了每个代理类别中增强图像的数量对下游分类任务的影响：

![](https://pic.imgdb.cn/item/63c0d9e3be43e0d30e7f9582.jpg)

通过**Exemplar-CNN**还可以分析对不同数据增强程度的不变性，作者展示了在不同增强程度下网络学习到的特征向量与原始特征向量之间的归一化距离(a-c)以及在不同增强程度下分类任务的准确率变化：

![](https://pic.imgdb.cn/item/63c0db14be43e0d30e81434d.jpg)

