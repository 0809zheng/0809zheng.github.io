---
layout: post
title: 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'
date: 2020-12-24
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/62158e0a2ab3f51d91a39b2e.jpg'
tags: 论文阅读
---

> 大批量分布式训练的线性缩放规则和warmup.

- paper：[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

在分布式训练深度网络时，由于数据批量变大将会导致优化困难。作者提出了一种调整学习率的线性缩放规则，将其作为批量大小的函数；并提出了一种学习率**warmup**方案，以克服训练早期的困难。通过这些技术，作者使用$256$块**GPU**和$8192$的批量大小实现了在一小时内训练**ResNet-50**。

## 1. 线性缩放规则

学习率的**线性缩放规则**(**linear scaling rule**)是指：
- 当批量大小增大$k$倍时，学习率也增大$k$倍，并保持其它超参数不变。

对于通常的梯度更新，经过$k$次迭代后参数更新为：

$$ w_{t+k} = w_t - \eta \frac{1}{|\mathcal{B}|}\sum_{j<k}^{}\sum_{x \in \mathcal{B}_j}^{} \nabla l(x,w_{t+j}) $$

如果在单次更新时使用的数据批量为$∪_j\mathcal{B}_j$，则参数更新为：

$$  \hat{w}_{t+1} = w_t - \hat{\eta} \frac{1}{k|\mathcal{B}|}\sum_{j<k}^{}\sum_{x \in \mathcal{B}_j}^{} \nabla l(x,w_t) $$

如果假设$\nabla l(x,w_{t+j})≈\nabla l(x,w_t)$，则使得上述两种参数更新等价$w_{t+k} =\hat{w}_{t+1}$的条件为$\hat{\eta}=k\eta$。

上述假设在一些情况下不成立。如在训练初期，梯度快速变化；批量的规模也不能无限扩大，超过某点时精度迅速下降。

## 2. warmup
在训练早期阶段，梯度变化剧烈，学习率的线性缩放规则不成立。该问题可以通过设计适当的**warmup**来缓解，即训练开始时使用较小的学习率。
- **constant warmuup**：在训练前$5$轮使用较低的常数学习率，然后恢复正常学习率。
- **gradual warmuup**：将学习率从一个较小的值开始逐渐提高，经过$5$轮恢复为正常学习率，避免学习率的突然增大。

## 3. 分布式SGD的一些注意事项

在训练分布式(或多**GPU**)模型时，有一些常见的实现错误导致模型训练误差较高。

### (1) 权重衰减

参数的梯度更新公式如下：

$$ w_{t+1} = w_t - \eta \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} \nabla l(x,w_t) $$

此时对学习率$\eta$的缩放等价于对损失函数$l$的缩放。

权重衰减相当于在损失函数中增加了梯度的**l2**正则化项：

$$ w_{t+1} = w_t - \eta \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} \nabla [l(x,w_t)+\frac{\lambda}{2}||w_t||^2] \\ = w_t -\eta \lambda w_t- \eta \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} \nabla l(x,w_t) $$

注意到权重衰减项$\eta \lambda w_t$与批量无关，因此可以单独计算后增加到聚合梯度中。此时学习率缩放不在$\eta \lambda w_t$中执行。

- 当使用权重衰减时，学习率缩放不等于损失函数缩放。

### (2) 动量修正

当梯度更新过程中引入动量时，参数更新公式如下：

$$ u_{t+1} = mu_t+ \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} \nabla l(x,w_t) \\ w_{t+1} = w_t - \eta u_{t+1} $$

一种流行的变体是将学习率融合到动量更新中：

$$ v_{t+1} = mv_t+ \eta \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} \nabla l(x,w_t) \\ w_{t+1} = w_t -  v_{t+1} $$

注意到在上述两种更新方式中，动量$u$与学习率$\eta$无关，而动量$v$与学习率$\eta$有关。当学习率改变时，为了保证两者的等价性，变体公式应修改为：

$$ v_{t+1} = m \frac{\eta_{t+1}}{\eta_t}v_t+ \eta_{t+1} \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} \nabla l(x,w_t) \\ w_{t+1} = w_t -  v_{t+1} $$

$\frac{\eta_{t+1}}{\eta_t}$为动量修正的因子。当$\eta_{t+1}>>\eta_t$时，动量修正很重要，否则将会导致训练不稳定。

- 如果使用动量梯度下降的变体公式，应该应用动量修正。

### (3) 梯度累积
对于$k$个**GPU**计算的梯度结果，需要累积后用于一次梯度更新。累积结果不能单纯的求和，而是应该求平均值。一种简单有效的方法是将$1/k$平均缩放放到每个**GPU**的损失计算中，从而避免了对整体梯度的缩放，只需要对分布式求得的梯度求和即可。

- 对每个**GPU**的损失函数使用缩放系数$\frac{1}{k\|\mathcal{B}\|}$而不是$\frac{1}{\|\mathcal{B}\|}$。

### (4) 数据打乱

梯度下降应该随机采样数据。在每轮**epoch**中应该对数据集进行随机打乱，然后划分成$k$部分，每部分数据交给一个**GPU**处理。

- 在每轮**epoch**中对数据集进行随机打乱。

## 4. 实验分析
实验在**ImageNet**上进行，在$k=8$个**GPU**和每个**GPU**中$n=32$批量(相当于总批量$256$)时，默认学习率为$\eta=0.1$。通过设置**gradual warmuup**，实现了大批量训练与普通的小批量训练接近的训练结果。

![](https://pic.imgdb.cn/item/621734ed2ab3f51d91f211b6.jpg)

