---
layout: post
title: 'Gradientless Descent: High-Dimensional Zeroth-Order Optimization'
date: 2022-03-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/62286e495baa1a80ab2364ef.jpg'
tags: 论文阅读
---

> 高维参数空间中的零阶优化方法.

- paper：[Gradientless Descent: High-Dimensional Zeroth-Order Optimization](https://arxiv.org/abs/1911.06317)


**零阶**(**zeroth-order**)优化又称无梯度(**gradient-free**)优化或土匪(**bandit**)优化，泛指不需要梯度信息的优化方法。对于不可导的函数，可以通过采样和差分这两种方法来估计参数更新的方向，不依赖于梯度计算。由于这两种方法都需要在参数空间中随机采样，当参数空间的维度比较大时，更新过程中的方差较大，优化效率较低。

### ⚪ 基于差分的零阶优化

对于任意函数$f(x)$，由于其可导性未知，因此无法直接求梯度。可以通过采样和差分求得梯度的近似表示：

$$ \tilde{\nabla}_xf(x) = \Bbb{E}_{u\text{~}p(u)}[\frac{f(x+\epsilon u)-f(x)}{\epsilon} u] $$

其中$\epsilon$是小正数；$p(u)$是具有零均值和单位协方差矩阵的分布，通常用标准正态分布。

从分布$p(u)$中采样若干点，便可以对梯度进行估计。对估计的梯度套用梯度下降算法，便是零阶优化的基本思路：

$$ x_{t+1} = x_t-\tilde{\nabla}_xf(x_t) $$

注意到如果函数$f(x)$可导，有如下**Taylor**展开：

$$ f(x+\epsilon u) = f(x) + \epsilon u^T\nabla_xf(x) + \mathcal{O}(\epsilon^2) $$

此时梯度估计等价于真实的梯度：

$$ \tilde{\nabla}_xf(x) = \Bbb{E}_{u\text{~}p(u)}[\frac{f(x+\epsilon u)-f(x)}{\epsilon} u] \\ = \Bbb{E}_{u\text{~}p(u)}[\frac{\epsilon u^T\nabla_xf(x)}{\epsilon} u] = \int_{}^{} p(u)uu^T\nabla_xf(x) du = \nabla_xf(x) $$

### ⚪ 基于采样的零阶优化

基于采样的**无梯度下降(gradient-less descent)**的基本思路是给定采样分布$\mathcal{D}$和参数初始值$x_0$，在第$t$轮循环中设置一个标量半径$r_t$，从以$x_t$为中心的分布$r_t\mathcal{D}$中采样$y_t$。如果$f(y_t)<f(x_t)$，则更新$x_{t+1}=y_t$；否则$x_{t+1}=x_t$。

尽管零阶优化中的采样分布是固定的(通常选择均匀分布)，可以在算法的每次迭代中选择采样半径$r_t$。如简单地通过二分搜索设置一系列半径：

![](https://pic.imgdb.cn/item/62295fb15baa1a80abd04cf6.jpg)

如果优化函数具有一个良好的条件数上界，则可以在迭代过程中逐渐减小采样半径，从而降低优化方差：

![](https://pic.imgdb.cn/item/622960415baa1a80abd0b18f.jpg)

