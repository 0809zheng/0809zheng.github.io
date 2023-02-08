---
layout: post
title: 'Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations'
date: 2022-10-28
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63e346cb4757feff33fb8666.jpg'
tags: 论文阅读
---

> 通过背景增强改进自监督学习的鲁棒性.

- paper：[Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations](https://arxiv.org/abs/2103.12719)

在对比学习中，图像的背景可能会干扰图像语义特征的学习，本文作者设计了一种**背景增强(Background Augmentation)**策略，用于增强对比学习的性能表现。

作者使用显著性图生成方法**DeepUSPS2**生成图像的前景和背景**mask**，用于提取图像的前景区域。

![](https://pic.imgdb.cn/item/63e349f24757feff33015ec4.jpg)

在获得图像前景区域的基础上，构造三种图像增强策略：
- **BG_RM**：移除背景，把背景设置为零像素；
- **BG_Random**：随机增加背景；
- **BG_Swaps**：为**anchor**样本和负样本设置同一个背景，为正样本设置另一个背景。

![](https://pic.imgdb.cn/item/63e34a8b4757feff33026ea2.jpg)

实验结果表明，该方法能改进不同对比学习方法的性能表现。

![](https://pic.imgdb.cn/item/63e34b024757feff33033fcd.jpg)