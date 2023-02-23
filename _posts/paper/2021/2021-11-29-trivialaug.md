---
layout: post
title: 'TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation'
date: 2021-11-29
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61d8f1182ab3f51d91aaf3da.jpg'
tags: 论文阅读
---

> TrivialAugment: 无需调优的先进数据增强.

- paper：TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation
- arXiv：[link](https://arxiv.org/abs/2103.10158)

数据增强的自动搜索通常需要在简单性、成本和性能之间权衡，本文作者提出了一种简单的增强方法**TrivialAugment**，该增强没有任何参数，对每个输入图像应用单个增强。实验证明该增强的优越性。

**TrivialAugment**预先设置$\mathcal{A}$种增强方法及应用增强的幅度$$m=\{0,...,30\}$$。对于每张图像$x$，均匀地从$\mathcal{A}$种增强方法中采样一种，并均匀地采样增强幅度$m$。

![](https://pic.imgdb.cn/item/61d8f55d2ab3f51d91adeeef.jpg)

**TrivialAugment**与其他模型之间的表现对比如下：

![](https://pic.imgdb.cn/item/61d8f7fa2ab3f51d91afcb6a.jpg)

由于不需要搜索超参数，**TrivialAugment**的训练时间比大多数自动增强方法要短：

![](https://pic.imgdb.cn/item/61d8f9502ab3f51d91b0ba5b.jpg)