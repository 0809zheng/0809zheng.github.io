---
layout: post
title: 'Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation'
date: 2020-12-18
author: 郑之杰
cover: 'https://pic.downk.cc/item/5fdc71e63ffa7d37b379cc0e.jpg'
tags: 论文阅读
---

> 一种用于实例分割的复制粘贴数据增强方法.

- paper：Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation
- arXiv：[link](https://arxiv.org/abs/2012.07177)

作者提出了一种可用于目标检测和实例分割的数据增强方法：**复制粘贴(Copy-Paste)**。该方法的主要流程如下：
1. 随机选择两幅图像；
2. 对两幅图像应用随机**尺度抖动(scale jitting)**；
3. 对两幅图像应用随机**水平翻转(horizontal flipping)**；
4. 从一张图像上随机选择一个目标的子集；
5. 将其粘贴在另一张图像上；
6. 对标注应用相同的调整。

该方法产生的增强图像如下图所示：

![](https://pic.downk.cc/item/5fdc72333ffa7d37b37a1796.jpg)

作者特别指出，采用更大的尺度抖动效果更好。下图左边是一组标准的尺度抖动缩放，缩放倍率是$0.8$-$1.25$倍；右边是一种实际效果更好的尺度抖动缩放，缩放倍率是$0.1$-$2.0$倍。

![](https://pic.downk.cc/item/5fdc72d83ffa7d37b37acc79.jpg)

作者进行了很多实验，与**SOTA**的目标检测和实例分割模型做对比，在**COCO**数据集上最好结果分别提高了$1.4$和$0.5$个点：

![](https://pic.downk.cc/item/5fdc72b83ffa7d37b37aa863.jpg)