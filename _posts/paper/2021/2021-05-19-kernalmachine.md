---
layout: post
title: 'Every Model Learned by Gradient Descent Is Approximately a Kernel Machine'
date: 2021-05-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60a383cf6ae4f77d35980995.jpg'
tags: 论文阅读
---

> 使用梯度下降优化的深度学习模型近似于核方法.

- paper：Every Model Learned by Gradient Descent Is Approximately a Kernel Machine
- arXiv：[link](https://arxiv.org/abs/2012.00152)

通常认为深度学习之所以能够成功，是因为它能够自动从数据中提取新的特征表示，而不是使用手工设计的特征。在本文中，作者认为，通过标准梯度下降算法训练得到的深度网络，在数学上近似等价于核方法，即记录所有训练数据并通过一个相似性函数(核)直接进行预测。这为深度网络的权重提供了一种可解释性：权重是所有训练样本的**叠加态(superposition)**。

## 1. 核方法 Kernel Machine
核方法通常表示为：

$$ y=g(\sum_{i}^{} a_i K(x,x_i)+b) $$

其中$x$为输入的查询数据；$x_i$为训练集中的数据点；$a_i$和$b$是可学习参数，其中$a_i$在监督学习中通常是训练数据的标签；$K$是一个核函数，用于衡量其两个参数的相似性；$g$是可选的非线性函数。

众多核方法的不同主要体现在其选择的核函数不同。

## 2. 路径核 Path Kernel
两个数据点的**路径核(path kernel)**是指两个数据点对应的网络输出$y$相对于网络权重$w$的梯度的点积沿梯度下降路径$c(t)$的积分(假设学习率无穷小，梯度下降是一个连续的过程)：

$$ K(x,x') = \int_{c(t)}^{} \Delta_wy(x) \cdot \Delta_wy(x') dt $$

直观上，路径核衡量模型在这两个数据点训练时变化的相似性。下图给出了一种图形解释，在模型梯度下降的整个过程中(从$w_0$到$w_{final}$)，若数据$x_1$与数据$x$相对于权重的梯度的点积的平均值相较于数据$x_2$更大，则可以认为标签$y_2$对预测结果$y$的影响相较于标签$y_1$更大。

![](https://pic.imgdb.cn/item/60a39c326ae4f77d353b0106.jpg)

## 3. 梯度下降 Gradient Descent
梯度下降是一种常用的深度神经网络优化方法。给定网络权重的初始值$w_0$和损失函数$L=\sum_{i}^{} L(y_i^*,y_i)$，梯度下降可以表示为：

$$ w_{s+1} = w_s - \epsilon \Delta_w L(w_s) $$

## 4. 使用梯度下降优化的深度网络近似于使用路径核的核方法

### 引理
函数$f_w(x)$在参数向量$w$上的**正切核(tangent kernel)**计算为：

$$ K_{f,w}^{g} (x,x') = \Delta_w f_w(x) \cdot \Delta_w f_w(x') $$

函数$f_w(x)$在曲线$c(t)$上的路径核可以借助正切核表示为：

$$ K_{f,c}^{p} (x,x') = \int_{c(t)}^{} K_{f,w}^{g} (x,x') dt $$

### 本文的主要结论
假设模型$y=f_w(x)$是在训练集$$\{(x_i,y_i^*)\}_{i=1}^m$$上通过梯度下降算法优化得到的，损失函数为$L=\sum_{i}^{} L(y_i^*,y_i)$，学习率为$\epsilon$。则有：

$$ \mathop{\lim}_{\epsilon → 0} y = \sum_{i=1}^{m} a_i K(x,x_i)+b $$

其中$K(x,x_i)$是沿梯度下降路径的路径核，$a_i$沿路径由其正切核加权平均的损失函数相对于输出的负梯度，$b$是初始模型。

### 证明
将梯度下降公式改写为：

$$ \frac{w_{s+1} - w_s}{\epsilon} =  - \Delta_w L(w_s) $$

当$\epsilon → 0$，上式变为微分方程：

$$ \frac{dw(t)}{dt} = - \Delta_w L(w(t)) $$

另一方面，由链式法则，将$w$沿其维度拆分可得：

$$ \frac{dy}{dt} = \sum_{j=1}^{d} \frac{\partial y}{\partial w_j} \frac{\partial w_j}{\partial t} $$

将上述两式整合，得：

$$ \frac{dy}{dt} = \sum_{j=1}^{d} \frac{\partial y}{\partial w_j} (-\frac{L}{\partial w_j}) $$

再次根据链式法则，将损失函数沿所有样本点拆分可得：

$$ \frac{dy}{dt} = \sum_{j=1}^{d} \frac{\partial y}{\partial w_j} (-\sum_{i=1}^{m} \frac{L}{\partial y_i} \frac{\partial y_i}{\partial w_j}) $$

将上式重排为：

$$ \frac{dy}{dt} = -\sum_{i=1}^{m} \frac{L}{\partial y_i} \sum_{j=1}^{d} \frac{\partial y}{\partial w_j} \frac{\partial y_i}{\partial w_j} $$

使用正切核简化上述表达：

$$ \frac{dy}{dt} = -\sum_{i=1}^{m} L'(y_i^*,y_i) \sum_{j=1}^{d} K_{f,w(t)}^{g} (x,x_i) $$

解上述微分方程，可以得到输出表达式：

$$ \mathop{\lim}_{\epsilon → 0} y = y_0 - \int_{c(t)}^{} \sum_{i=1}^{m} L'(y_i^*,y_i) K_{f,w(t)}^{g} (x,x_i) dt $$

将上式进一步表示为：

$$ \mathop{\lim}_{\epsilon → 0} y = y_0 - \sum_{i=1}^{m} \frac{\int_{c(t)}^{} L'(y_i^*,y_i) K_{f,w(t)}^{g} (x,x_i) dt}{\int_{c(t)}^{} K_{f,w(t)}^{g} (x,x_i) dt} \int_{c(t)}^{} K_{f,w(t)}^{g} (x,x_i) dt $$

上式可以记作：

$$ \mathop{\lim}_{\epsilon → 0} y = y_0 - \sum_{i=1}^{m} \overline{L'}(y_i^*,y_i) K_{f,c}^{p} (x,x_i) \\ = \sum_{i=1}^{m} a_i K(x,x_i)+b $$

