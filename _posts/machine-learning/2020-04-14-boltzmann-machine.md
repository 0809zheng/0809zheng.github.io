---
layout: post
title: 'Boltzmann Machine：玻尔兹曼机'
date: 2020-04-14
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e954cbbc2a9a83be594d120.jpg'
tags: 机器学习
---

> Boltzmann Machine.

**本文目录**：
1. 模型介绍
2. 推断
3. 学习


# 1. 模型介绍
**玻尔兹曼机(Boltzmann Machine)**是一个概率无向图模型，具有与[离散型Hopfield网络](https://0809zheng.github.io/2020/04/13/hopfield-network.html)相似的特点：
1. 每个神经元代表的随机变量是二值的，所有随机变量可以用一个随机向量$$X \in \{0,1\}^K$$表示；
2. 所有节点是全连接的；
3. 每两个节点之间的连接权重是对称的。

玻尔兹曼机的神经元可以分为**可观测变量(visible variable)**$v$和**隐变量(latent variable)**$h$，下图给出了包含3个可观测变量和3个隐变量的玻尔兹曼机：

![](https://pic.downk.cc/item/5e952a3dc2a9a83be57fae50.jpg)

随机向量$$X = (v,h)$$的联合概率服从**玻尔兹曼分布(Boltzmann Distribution)**:

$$ p(x) = \frac{1}{Z} exp(\frac{-E(x)}{T}) $$

其中$Z$是归一化因子，称为**配分函数(Partition Function)**;

能量函数$E(x)$定义为：

$$ E(x) = -(\sum_{i<j}^{} {w_{ij}x_ix_j} + \sum_{i}^{} {b_{i}x_i}) $$

能量函数值越小，则$X$发生的概率越大。

如果令玻尔兹曼机中的每个变量代表一个基本假设，其取值为 1 或 0 分别表示模型接受或拒绝该假设，那么变量之间连接的权重代表了两个假设之间的弱约束关系。
- 一个正的连接权重表示两个假设可以互相支持；
- 一个负的连接权重表示两个假设不能同时被接受。

$T$代表**温度**。
- 温度越高($$T → ∞$$)，$p → 0.5$,每个变量状态的改变十分容易，很快可以达到热平衡;
- 温度越低($$T → 0$$)，当系统能量$E(x)>0$时$p(x=1) → 1$,当系统能量$E(x)<0$时$p(x=0) → 1$,时，玻尔兹曼机退化为Hopfield网络。

与Hopfield网络的比较：
- Hopfield网络是一种确定性的动力系统，每次状态更新都会使得系统的能量降低；
- 玻尔兹曼机是一种随机性的动力系统，以一定的概率使得系统的能量上升。

# 2. 推断
**推断(Inference)**问题是指当给定变量之间的连接权重时，根据观测值生成一组二值向量，使得整个网络的能量最低。

在玻尔兹曼机中，配分函数$Z$通常难以计算，因此，联合概率分布$$p(x)$$一般通过MCMC方法（如Gibbs采样）来近似。

玻尔兹曼机的**Gibbs采样**过程为：
1. 随机选择一个变量$X_i$,计算其全条件概率$$p(x_i \mid x_{-i})$$；
2. 以$$p(x_i = 1 \mid x_{-i})$$的概率将变量设为1，否则设为0；
3. 在固定温度$T$的情况下，在运行足够时间之后，玻尔兹曼机会达到热平衡;
4. 最终任何全局状态的概率服从玻尔兹曼分布$p(x)$，只与系统的能量有关，与初始状态无关。

**全条件概率**$$p(x_i \mid x_{-i})$$计算如下：

$$ p(x_i = 1 \mid x_{-i}) = sigmoid(\frac{\sum_{j}^{} {w_{ij}x_j} + b_i}{T}) $$

$$ p(x_i = 0 \mid x_{-i}) = 1 - p(x_i = 1 \mid x_{-i}) $$

玻尔兹曼机采用**模拟退火（Simulated Annealing）**的方法，让系统刚开始在一个比较高的温度下运行达到热平衡，然后逐渐降低，直到系统在一个比较低的温度下运行达到热平衡。

模拟退火是一种寻找**全局最优**的近似方法，可以证明，模拟退火算法所得解依概率收敛到全局最优解。

# 3. 学习
**学习(Learning)**问题是指当给定变量的多组观测值时，学习模型的最优参数。

假设一个玻尔兹曼机有$K$个变量，包括$K_v$个可观测变量$$v \in \{0,1\}^{K_v}$$和$K_h$个隐变量$$h \in \{0,1\}^{K_h}$$。

给定可观测向量$$D = \{\hat{v}^1,\hat{v}_2,...,\hat{v}^N\}$$为训练集，需要学习参数$W$和$b$使得训练集所有样本的对数似然函数最大：

$$ W_{MLE},b_{MLE} = argmax_{(W,b)} L(D;W,b) $$

其中对数似然函数：

$$ L(D;W,b) = \frac{1}{N} \sum_{n=1}^{N} {logp(\hat{v}^n \mid W,b)} = \frac{1}{N} \sum_{n=1}^{N} {log\sum_{h}^{} {p(\hat{v}^n,h,W,b)}} $$

计算得对数似然函数对参数的梯度为：

$$ \frac{\partial L(D;W,b)}{\partial w_{ij}} = E_{\hat{p}(v)}E_{p(h \mid v)}(x_ix_j)-E_{p(v,h)}(x_ix_j) $$

$$ \frac{\partial L(D;W,b)}{\partial b_{i}} = E_{\hat{p}(v)}E_{p(h \mid v)}(x_i)-E_{p(v,h)}(x_i) $$

上述计算涉及配分函数和期望，很难精确计算。玻尔兹曼机一般通过MCMC方法（如吉布斯采样）来进行近似求解。

以参数$w_{ij}$的梯度为例:
1. 对于第一项，固定可观测变量$$v$$，只对隐变量$$h$$进行Gibbs采样，在训练集上所有的训练样本上重复此过程，得到$x_ix_j$的近似期望$$(x_ix_j)_{data}$$;
2. 对于第二项，对所有变量进行Gibbs采样。当玻尔兹曼机达到热平衡状态时，采样$x_ix_j$的值，得到近似期望$$(x_ix_j)_{model}$$。

采用**梯度上升法**更新权重：

$$ w_{ij} = w_{ij} + α((x_ix_j)_{data} - (x_ix_j)_{model}) $$
