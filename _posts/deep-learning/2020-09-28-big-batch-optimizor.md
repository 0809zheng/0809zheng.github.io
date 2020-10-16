---
layout: post
title: '为large-batch训练设计的Optimizor'
date: 2020-09-28
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f728a0c160a154a67555f58.jpg'
tags: 深度学习
---

> Optimizors designed for large-batch training.

加速训练卷积神经网络的一个方法是**分布式训练**，即采用数据并行或模型并行的方法加载多计算节点。此时整体**batch size**设置较大，每个**GPU**节点接收整体**batch**的一部分。

但是当增大**batch**时，训练的模型精度会剧烈降低。这是因为当总训练**epoch**保持不变时，大**batch**意味着权重更新的次数减少。

一些针对大**batch**设计的优化方法由此被提出。本文主要介绍：
1. **Linear Scaling**
2. **Warm up**
3. **LARS**
4. **LAMB**
5. **NovoGrad**

# 1. Linear Scaling
- paper：[One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)

一种直观的解决方法是增大学习率。若记**batch size**为$B$，学习率的增大通常遵循**Linear Scaling**原则：

- 当我们对$B$增大$k$倍时，应同样对学习率增大$k$倍，并保持其它超参数不变。

然而使用更大的学习率会使得网络在初始阶段就发散，需要解决高学习率带来的不稳定性。

# 2. Warm up
- paper：[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

**Warm up**方法是指训练开始时先选择使用一个较小的学习率，然后逐渐增大为预先设置的学习率。
- **Constant warmup**：阶梯增长，缺点是学习率会突然增长。
- **Gradual warmup**：线性增长，训练更稳定。

# 3. LARS
- paper：[Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)

为了分析在大学习率下训练的稳定性，作者分析了模型中每层的**权值范数**和更新时的**梯度范数**。
作者观察到，当两者比例很低时，训练就变得不稳定；当比例很高时，权值的更新变得很慢。
由于这个比例在不同层之间差异很大，所以有必要对神经网络中的每一层都使用不同的学习率。

作者提出了一种**Layer-wise Adaptive Rate Scaling（LARS）**方法。其中第$l$层的更新梯度计算为：

$$ \Delta w_{t}^{l} = \gamma * \lambda^l * \Delta L(w_{t}^{l}) $$

其中$\Delta L(w_{t}^{l})$是反向传播计算得到的梯度，$\gamma$是全局学习率（**global LR**）；$\lambda^l$是局部学习率（**local LR**），计算为：

$$ \lambda^l = \eta \times \frac{\mid\mid w^l 
\mid\mid}{\mid\mid \Delta L(w^{l}) 
\mid\mid} $$

其中系数$\eta$表明我们在多少程度下相信该层会在一次更新中权重将会改变。注意到此时梯度更新的幅值与梯度本身的幅值无关。

引入权重衰减因子$\beta$平衡局部学习率：

$$ \lambda^l = \eta \times \frac{\mid\mid w^l 
\mid\mid}{\mid\mid \Delta L(w^{l}) 
\mid\mid + \beta * \mid\mid w^l 
\mid\mid} $$

结合**LARS**与**momentum**的优化算法表示如下：

![](https://pic.downk.cc/item/5f73dd90160a154a67ae3865.jpg)

# 4. LAMB
- paper：[Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)

**Layer-wise Adaptive Moments optimizer for Batching training (LAMB)**在**LARS**的基础上做了进一步改进，引入**momentum**和**rmsprop**的思想：

![](https://pic.downk.cc/item/5f73de03160a154a67ae5122.jpg)

# 5. NovoGrad
- paper：[Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks](https://arxiv.org/abs/1905.11286)

**NovoGrad**使用二阶动量(**2nd moment**)作为标准化(**normalization**)，将权重衰减(**weight decay**)从随机梯度(**stochastic gradient**)中解耦作为正则化(**regularization**)：

![](https://pic.downk.cc/item/5f73e1ca160a154a67af1658.jpg)