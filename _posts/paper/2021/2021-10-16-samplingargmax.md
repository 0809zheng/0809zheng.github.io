---
layout: post
title: 'Localization with Sampling-Argmax'
date: 2021-10-16
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/679060bbd0e0a243d4f65b67.png'
tags: 论文阅读
---

> 通过Sampling-Argmax来定位.

- paper：[Localization with Sampling-Argmax](https://arxiv.org/abs/2110.08825)

## TL; DR

本文提出了一种名为 **Sampling-Argmax** 的可微分训练方法，用于改进基于检测的定位任务中的概率图形状，从而提高定位精度。该方法通过最小化定位误差的期望值来隐式约束概率图的形状，并通过可微分采样过程近似计算期望值。实验表明，**Sampling-Argmax** 在多个定位任务上优于传统的 **Soft-Argmax** 及其变体，并且能够生成更可靠的置信度分数。

## 1. 背景介绍

在计算机视觉领域，从输入数据中定位目标位置是一个基础任务，广泛应用于人体姿态估计、面部关键点定位、立体匹配和目标关键点估计等任务。

常见的定位方法分为回归方法和基于检测的方法。基于检测的方法通过预测概率图（或热图）来指示目标位置的可能性，并通过 **Argmax** 操作获取最高概率的位置。然而，**Argmax** 操作不可微分且存在量化误差。为此，**Soft-Argmax** 被提出作为一种可微分的近似方法，但其训练机制存在局限性：仅约束概率图的期望值，而不约束其形状，导致模型在训练时缺乏像素级监督，影响性能。

![](https://pic1.imgdb.cn/item/6790955fd0e0a243d4f67c79.png)

为了解决这一问题，本文提出了一种新的可微分训练方法 **Sampling-Argmax**，通过最小化定位误差的期望值来隐式约束概率图的形状，从而提高定位精度。

## 2. Sampling-Argmax

**Sampling-Argmax**提出了一种新的训练目标：最小化定位误差的期望值，而不是最小化期望值的误差。

传统的 **Soft-Argmax** 方法通过最小化预测坐标与真实位置之间的误差来训练模型：

$$ L=d(y_t,E[y])≈d(y_t,∑_iπ_{y_i}y_i) $$

其中 $y_t$ 是真实位置，$π_{y_i}$ 是预测热图中坐标 $y_i$ 的概率，$d$是距离函数。

**Sampling-Argmax**提出的新目标函数为：

$$ L=E_y[d(y_t,y)] $$

该目标函数鼓励模型在真实位置附近生成更高的概率值，从而隐式约束概率图的形状。

为了近似计算期望值，**Sampling-Argmax**将目标位置的概率分布建模为连续的混合分布。具体来说，将概率图划分为多个子区间，并在每个子区间内使用标准概率密度函数（如均匀分布、三角分布或高斯分布）来近似原始分布。混合分布的形式为：

$$ p(y)=∑_{i=1}^n w_i f_i(y) $$

![](https://pic1.imgdb.cn/item/67909d51d0e0a243d4f68162.png)

通过上式可以将预测结果的离散概率分布重建为一个近似的连续分布。为了实现该连续分布的可微分采样，本文采用了 [Gumbel-Softmax](https://0809zheng.github.io/2022/04/24/repere.html#-gumbel-max%E6%96%B9%E6%B3%95) 方法。具体步骤如下：
1. 使用 **Gumbel-Softmax** 从概率图中采样分类权重：
$$ \pi_i=\frac{\exp((g_i+\log π_i)/τ)}{∑_k\exp((g_k+\log π_k)/τ)} $$
2. 从每个子分布 $f_i(y)$ 中采样$y_i\sim f_i(y)$；
3. 对采样结果进行加权求和：

$$ Y=∑_{i=1}^n \pi_i y_i $$
​
最终通过计算所有采样的平均误差来近似期望值：

$$ L\approx \frac{1}{N} \sum_{k=1}^N d(y_t,Y_k) $$

![](https://pic1.imgdb.cn/item/67909f09d0e0a243d4f68222.png)

实验表明，采样数量$N$对性能有一定影响，但过多的采样并不会显著提升性能。在 **COCO Keypoint** 数据集上，仅使用一个采样点即可获得较高的性能：

![](https://pic1.imgdb.cn/item/67909f82d0e0a243d4f68277.png)

为了验证 **Sampling-Argmax** 生成的概率图是否更可靠，本文计算了概率图峰值与预测正确性之间的皮尔逊相关系数。结果表明，**Sampling-Argmax** 的相关性显著高于 **Soft-Argmax**：

![](https://pic1.imgdb.cn/item/67909fc0d0e0a243d4f6829a.png)