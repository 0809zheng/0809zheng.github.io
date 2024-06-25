---
layout: post
title: 'Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow'
date: 2022-09-17
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6679194cd9c307b7e9a040cd.png'
tags: 论文阅读
---

> 通过整流流实现数据的生成与转换.

- paper：[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)

给定初始数据$x_T$，扩散模型旨在学习一个演化过程，生成目标数据$x_0$，该过程可以用一个**常微分方程 (ordinary differential equation, ODE)**来描述：

$$
\frac{d x_t}{dt} = f_t(x_t)
$$

其中$p_T(x_T),p_0(x_0)$是已知的，上述**ODE**旨在设计一个函数$f_t(x_t)$，使其对应的演化轨迹构成给定分布$p_T(x_T),p_0(x_0)$之间的一个变换。

随机选定$x_0\sim p_0(x_0),x_T\sim p_T(x_T)$，设计函数：

$$
x_t = \phi_t(x_0,x_T)
$$

则有微分方程：

$$
\frac{d x_t}{dt} = \frac{\partial \phi_t(x_0,x_T)}{\partial t}
$$

引入一个函数$s_\theta(x_t,t)$逼近上式右端：

$$
\mathbb{E}_{x_0\sim p_0(x_0),x_T\sim p_T(x_T)}\left[ \left\| s_\theta(x_t,t) - \frac{\partial \phi_t(x_0,x_T)}{\partial t} \right\|^2 \right]
$$

下面不妨考虑$[0,1]$的扩散过程，设计变化轨迹$\phi_t(x_0,x_T)$为直线：

$$
x_t = \phi_t(x_0,x_T) = (x_1-x_0)t+x_0
$$

对应微分方程：

$$
\frac{d x_t}{dt} = \frac{\partial \phi_t(x_0,x_T)}{\partial t} = x_1-x_0
$$

此时训练目标为：

$$
\mathbb{E}_{x_0\sim p_0(x_0),x_T\sim p_T(x_T)}\left[ \left\| s_\theta(x_t,t) - \frac{\partial \phi_t(x_0,x_T)}{\partial t} \right\|^2 \right] \\
 = \mathbb{E}_{x_0\sim p_0(x_0),x_T\sim p_T(x_T)}\left[ \left\| s_\theta((x_1-x_0)t+x_0,t) - (x_1-x_0) \right\|^2 \right]
$$

该模型可以把任何一种数据或噪声转换成另外一种数据，并且只需一步计算就直接产生高质量的结果，而不需要调用计算量大的数值求解器来迭代式地模拟整个扩散过程。

![](https://pic.imgdb.cn/item/66792a62d9c307b7e9c58c19.png)