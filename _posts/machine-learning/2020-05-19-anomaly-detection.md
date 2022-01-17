---
layout: post
title: 'Anomaly Detection：异常检测'
date: 2020-05-19
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eb51281c2a9a83be56f3c55.jpg'
tags: 机器学习
---

> Anomaly Detection.

**异常检测（Anomaly Detection）**是指判断数据集中是否存在异常点，或者一个新的数据点是否正常。

若把训练数据看作一个概率分布，则异常点是出现概率较低的数据点，如**outlier**。

异常点 $anomaly$ 相对于正常点 $normal$ 出现的频率低，较难收集。

假设数据集$$\{x^1,x^2,...,x^N\}$$，根据是否有对应的标签可以把异常检测分为**有标签的异常检测**和**无标签的异常检测**。

# 1. With labels
对带有标签的数据进行异常检测的方法也称作**Open-set Recognition**。

训练一个分类器，对数据进行分类的同时输出一个置信度$c$，表示该数据是正常的概率;

选择一个阈值$λ$，若置信度$c$高于阈值$λ$，则认为数据点是正常的；否则是异常的。

![](https://pic.downk.cc/item/5eb501c2c2a9a83be55e0e9e.jpg)

$$ f(x) = \begin{cases} normal, & c(x) > λ \\ anomaly, & c(x) ≤ λ \\ \end{cases} $$

### (1)置信度$c$的选择
置信度$c$可以使用网络经过Softmax之后得到概率分布的最大值或其熵的负值。

也可以训练一个网络分别计算概率分布和置信度：

![](https://pic.downk.cc/item/5eb50267c2a9a83be55ec058.jpg)

- 参考论文：[Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/abs/1802.04865)

### (2)阈值$λ$的选择
用验证集$dev$ $set$选择$λ$。

对于一个选定的$λ$，置信度$c$低于阈值$λ$的数据是被**检测detected**出来认为异常的，高于阈值的数据是**没有检测not det**到的（认为是正常的）。

在检测为异常点的数据中也存在正常点，称为**false alarm**；在认为是正常点的数据中也存在异常点，称为**missing**。

![](https://pic.downk.cc/item/5eb50d5dc2a9a83be56a12a0.jpg)

也可以用其他指标衡量，如**area under ROC curve**。

# 2. Without labels
对于没有标签的数据，可以对数据集的概率分布进行建模$p(x)$，

对于一个新的数据点$x^i$，选择一个阈值$λ$；

当$$p(x^i)≥λ$$时认为数据点是正常的，否则$$p(x^i)<λ$$时认为数据点是异常点。

通常用正态分布对数据分布建模：

$$ f_{\mu, \Sigma}(x) = \frac{1}{(2 \pi)^{\frac{d}{2}}} \frac{1}{\mid \Sigma \mid^\frac{1}{2}} exp(-\frac{1}{2}(x- \mu)^T \Sigma ^{-1} (x-\mu)) $$

用极大似然估计估计参数：

$$ \mu^* = \frac{1}{N} \sum_{n=1}^{N} {x^n} $$

$$ \Sigma^* = \frac{1}{N} \sum_{n=1}^{N} {(x^n-\mu^*)(x^n-\mu^*)^T} $$
