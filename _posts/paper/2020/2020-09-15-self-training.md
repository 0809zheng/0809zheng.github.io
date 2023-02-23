---
layout: post
title: 'Rethinking Pre-training and Self-training'
date: 2020-09-15
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f608568160a154a672bdad4.jpg'
tags: 论文阅读
---

> 对计算机视觉任务中预训练和自训练的一些讨论.

- paper：Rethinking Pre-training and Self-training
- arXiv：[link](https://arxiv.org/abs/2006.06882)

作者通过大量实验探索了在目标检测和图像分割任务上**预训练（pre-training）**和**自训练（self-training）**的效果。
- **预训练（pre-training）**：监督学习方法，在图像分类任务数据集上（如**ImageNet**）训练卷积网络，将其应用到检测和分割任务中。
- **自训练（self-training）**：半监督学习方法，先用少量有标签数据训练网络，再用网络对大量无标签的数据生成伪标签，最后将这些数据结合起来一起训练网络。

# 实验分析

![](https://pic.downk.cc/item/5f60a032160a154a6732f290.jpg)

实验设置如下图所示，分别为：
- 数据增强：使用四种不同的（依次增强）数据增强方法。
- 初始化：模型使用**Efficient-B7**，分别使用随机初始化、预训练初始化和[**Noisy Student**](https://0809zheng.github.io/2020/08/07/noisy-student-training.html)自训练初始化。

### （1）Pre-training

![](https://pic.downk.cc/item/5f60a192160a154a6733604c.jpg)

对预训练进行的实验表明：
- 当使用更强的数据增强方法时，预训练反而会损害模型的性能；
- 当可用的已标注数据越多时，预训练的作用越小。

### （2）Self-training

![](https://pic.downk.cc/item/5f60a349160a154a6733bc35.jpg)

![](https://pic.downk.cc/item/5f60a38d160a154a6733ca37.jpg)

对自训练进行的实验表明：
- 使用自训练在不同强度的数据增强方法下都能够提升性能；
- 使用自训练在不同大小的标注数据规模下都能够提升性能。

### （3）Self-supervised pre-training

![](https://pic.downk.cc/item/5f60a432160a154a6733ed10.jpg)

实验表明，自监督的预训练在高强度数据增强（**Augment-S4**）和可用的已标注数据多（**100% COCO**）时会损害模型的性能。

### （4）Joint-training

![](https://pic.downk.cc/item/5f60a561160a154a67343482.jpg)

实验表明，联合训练（同时训练图像分类和目标检测）能够提升模型性能，并可以和其他方法一起使用。

# 讨论
通过实验，作者发现当采用更强的数据增强方法或有更多的标注数据可以利用时，采用预训练模型反而会影响模型性能；而自训练方法均能够对模型性能有所提升。

作者认为，预训练模型的效果不好是因为用图像分类任务进行预训练无法感知检测或分割任务感兴趣的地方并且无法适应，例如**ImageNet**上训练好的特征可能忽视了检测任务所需的位置信息。

作者认为，自训练模型之所以效果好，是因为它能够进行**task alignment**，自动地对齐任务（如标注的类别不一致，通过自训练能够生成符合当前任务类别的伪标注）。但是自训练相比于预训练需要$1.3$到$8$倍的训练时间，成本更高。
