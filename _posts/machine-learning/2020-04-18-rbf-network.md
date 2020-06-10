---
layout: post
title: '径向基函数网络'
date: 2020-04-18
author: 郑之杰
cover: 'https://pic.downk.cc/item/5edf57bfc2a9a83be5dd5cc4.jpg'
tags: 机器学习
---

> Radial Basis Function Networks.

# 1. RBF Network
**径向基函数网络（Radial Basis Function Networks，RBF Network）**是一种特殊的神经网络，它用**径向基函数**替代神经元，输出是这些径向基函数的(加权)投票结果：

![](https://pic.downk.cc/item/5edf44d1c2a9a83be5bf41b2.jpg)

假设选定$M$个中心$μ_1,...,μ_M$，给定一个样本$x$，径向基函数$rbf$用来度量这个样本与给定中心的**相似度**；径向基函数网络模型可以表示为：

$$ h(x) = Output(\sum_{m=1}^{M} {w_m·rbf(x,μ_m)}) $$

径向基函数网络也可以看作一种特殊的**特征变换**，即给定中心$μ_1,...,μ_M$和一个新的样本点$x$，先对样本点$x$进行特征变换：

$$ z = φ(x) = (rbf(x,μ_1),rbf(x,μ_2),...,rbf(x,μ_m)) $$

网络模型可表示为：

$$ h(x) = Output(\sum_{m=1}^{M} {w_m·rbf(x,μ_m)}) = Output(w^Tz) $$

### 径向基函数
**径向基函数（radial basis function，rbf）**用来衡量两个向量的相似性。常用的径向基函数包括：
- **高斯(Gaussian)径向基函数**：$exp(-γ\mid\mid x-x' \mid\mid^2)$
- **截断(Truncated)径向基函数**：$\[\mid\mid x-x' \mid\mid ≤ 1\](1-\mid\mid x-x' \mid\mid)^2$
- $[x=x']$

# 2. Full RBF Network
**Full RBF Network**把每一个样本点$x_1,...,x_N$都看做中心，即$M=N$;

### 手工选择权重
若选定$w_m = y_m$，径向基函数为高斯径向基函数，模型表示为：

$$ h(x) = Output(\sum_{m=1}^{N} {y_m·rbf(x,x_m)}) = Output(\sum_{m=1}^{N} {y_mexp(-γ\mid\mid x-x_m \mid\mid^2)}) $$

对于分类问题使用符号函数，对于回归问题使用平均值作为输出。

其几何解释为样本距离越近的中心点对应的输出对结果影响越大（$rbf(0)=1$）;样本距离越y远的中心点对应的输出对结果影响越小（$rbf(∞)=0$）。

当径向基函数采用输出与给定样本最接近的1个或k个样本输出决定时，算法变为[最近邻或k近邻算法](https://0809zheng.github.io/2020/03/23/knn.html)。

### 使用样本集选择权重
若给定样本集$x_1,...,x_N$及其标签$y_1,...,y_N$，求解最优的$w$；

此时径向基函数网络的特征变换可表示为：

$$ z = φ(x) = (rbf(x,x_1),rbf(x,x_2),...,rbf(x,x_N)) $$

输出采用线性模型：

$$ h(x) = \sum_{m=1}^{N} {w_m·rbf(x,μ_m)} = w^Tz $$

这是一个线性回归问题，若记特征变换后的输入样本为$Z \in \Bbb{R}^{N×N}$，其解为：

$$ w = (Z^TZ)^{-1}Z^Ty $$

如果所有的$x_n$不同，且使用高斯径向基函数，则$Z$是**可逆**的。

为了避免发生过拟合，可以引入正则项$λ$，得到$w$的最优解为：

$$ w = (Z^TZ+λI)^{-1}Z^Ty $$


# 3. RBF Network using k-Means
把每一个样本点$x_1,...,x_N$都看做中心的计算代价大，实际操作中选择合适的$M$个样本点作为中心。

中心点的选择可参考[K-Means](https://0809zheng.github.io/2020/05/02/kmeans.html)等聚类算法。