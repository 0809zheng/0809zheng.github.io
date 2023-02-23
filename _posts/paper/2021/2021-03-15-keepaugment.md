---
layout: post
title: 'KeepAugment: A Simple Information-Preserving Data Augmentation Approach'
date: 2021-03-15
author: 郑之杰
cover: 'https://img.imgdb.cn/item/604ecbdd5aedab222c6263c5.jpg'
tags: 论文阅读
---

> KeepAugment：提高保真度的图像增强方法.

- paper：KeepAugment: A Simple Information-Preserving Data Augmentation Approach
- arXiv：[link](https://arxiv.org/abs/2011.11778)

**数据增强**能够提高网络的性能。在图像任务中，尽管针对图像的增强方法能够增加有效样本的数量和训练数据的多样性，但不可避免地引入了具有噪声和歧义的样本。当前的图像增强方法可分为两类，**区域级**图像增强方法(如**Coutout**、**CutMix**)通常遮挡或修改图像的随机矩形区域，**图像级**图像增强方法(如**AutoAugment**、**RandAugment**)通过强化学习寻找变换组合(如旋转、改变颜色)的最佳策略。

尽管目前的数据增强方法能够增加有效样本数，但如果增强幅度不合适，可能会引入噪声和歧义导致信息丢失。在**CIFAR-10**数据集上，分别对**Coutout**和**RandAugment**两种方法进行实验，实验结果如下图所示，两种数据增强都提高了模型的泛化能力(表现为原始数据的训练和测试准确率之间的差距)。但当增强程度太大时，模型准确率均下降。

![](https://img.imgdb.cn/item/604ed2315aedab222c668477.jpg)

作者提出了一种图像数据增强方法：**KeepAugment**。首先通过计算**saliency map**找到图像中对结果影响较大的区域，保留重要性得分较高的矩形区域后应用增强方法。对于**Coutout**，避免剪切重要的区域；对于**RandAugment**，将重要区域粘贴到图像上。

![](https://img.imgdb.cn/item/604ed7de5aedab222c69ad1b.jpg)

![](https://img.imgdb.cn/item/604ed7b95aedab222c699a0b.jpg)

**KeepAugment**对每张输入图像通过反向传播计算**saliency map**，计算成本较高。在论文中，作者提出两种有效的策略降低计算量，均不会导致性能的下降。

第一种是基于**低分辨率**的近似方法，即先把输入图像通过下采样生成一个低分辨率的图像，再对该低分辨率图像计算**saliency map**；将该**saliency map**上采样恢复到原始分辨率，能够显著降低计算量。

![](https://img.imgdb.cn/item/604ed6765aedab222c68e203.jpg)

第二种是基于**early loss**的近似方法，即在浅层网络计算**loss**，通过该**loss**反向传播计算**saliency map**，降低计算量。

![](https://img.imgdb.cn/item/604ed6885aedab222c68ef5b.jpg)
