---
layout: post
title: '感知机'
date: 2020-03-11
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eca7715c2a9a83be521b412.jpg'
tags: 机器学习
---

> Perceptron.

本文目录：
1. 感知机模型
2. 感知机学习算法（PLA）
3. 算法的对偶形式
4. 算法的收敛性
5. 口袋算法

# 1. 感知机模型
**感知机（perceptron）**是一类简单的二分类模型。

数据集$${(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$$，其中$x_i \in \Bbb{R^d}$，$y_i \in {-1,+1}$。

感知机模型是对数据点的每一维特征进行加权求和，与阈值进行比较，可表示为：

$$ f(x) = sign(w^Tx+b) $$

该模型的**几何意义**：寻找数据的特征空间中的一个**分离超平面（separable hyperplane）**。

![](https://pic.downk.cc/item/5eca5f20c2a9a83be50895dd.jpg)

有时也对参数$w$和数据$x$扩充第零维，以简化表示：

$$ f(x) = sign(w^Tx) $$

# 2. 感知机学习算法
**感知机学习算法（perceptron learning algorithm，PLA）**的思想是**错误驱动**。

感知机的损失函数定义为**错误分类的点数**：

$$ L(w,b) = \sum_{i=1}^{N} {[y_i ≠ sign(w^Tx_i+b)]} $$

上述损失函数不是连续函数，优化困难。进一步，使用**分类错误点的函数间隔**：

$$ L(w,b) = \sum_{y_i ≠ sign(w^Tx_i+b)}^{} {-y_i(w^Tx_i+b)} $$

则可得优化问题：

$$ min_{w,b} \quad \sum_{y_i ≠ sign(w^Tx_i+b)}^{} {-y_i(w^Tx_i+b)} $$

对目标函数求梯度，得：

$$ \frac{\partial L(w,b)}{\partial w} = \sum_{y_i ≠ sign(w^Tx_i+b)}^{} {-y_ix_i} $$

$$ \frac{\partial L(w,b)}{\partial b} = \sum_{y_i ≠ sign(w^Tx_i+b)}^{} {-y_i} $$

使用随机梯度下降的方法（设学习率为1），每次选择一个被错误分类的点$(x_i,y_i)$，可得参数更新公式：

$$ w^{(t+1)} = w^{(t)} + y_ix_i $$

$$ b^{(t+1)} = b^{(t)} + y_i $$

参数更新的**几何解释**：对于被错误分类的点，
- 若为正样本，则参数$w$和数据$x$夹角大于90°（内积为负），此时对$w$加上$x$来更新参数；
- 若为负样本，则参数$w$和数据$x$夹角小于90°（内积为正），此时对$w$减去$x$来更新参数；

![](https://pic.downk.cc/item/5eca636fc2a9a83be50d0ea7.jpg)

在实践中遍历样本集，发现错误的点就进行修正，直至（线性可分的）样本集被完全分类正确，这种方法称为**cycle PLA**。

# 3. 算法的对偶形式
若假设PLA算法的初值选择$$w^{(0)}=0$$，$$b^{(0)}=0$$，经过循环后参数更新为：

$$ w = \sum_{i=1}^{N} {α_iy_ix_i} $$

$$ b = \sum_{i=1}^{N} {α_iy_i} $$

其中$α_i$表示第$i$个样本点在更新中被错误分类的次数。

由于在训练中，对于样本点$(x_j,y_j)$，需要判断其是否被分类错误，即计算：

$$ sign(w^Tx_j+b) = sign(\sum_{i=1}^{N} {α_iy_ix_i^Tx_j}) + \sum_{i=1}^{N} {α_iy_i} $$

在训练前可以预先存储数据集的**Gram矩阵**$$G=[x_i^Tx_j]_{N×N}$$，从而减少不必要的重复计算。

# 4. 算法的收敛性
当样本集**线性可分（linear separable）**时，感知机学习算法收敛。但解不唯一，取决于初值的选择和错误分类点的选择顺序。

**Novikoff定理**：若训练集线性可分，则：
1. 存在一个超平面${\hat{w}}^Tx=0$能将正负样本完全分开；
2. 若记$R=max \quad \mid\mid x_i \mid\mid$,$ρ=min \quad y_i{\hat{w}}^Tx_i$,则算法的迭代次数$T$满足：

$$ T ≤ \frac{R^2}{ρ^2} $$

证明：

记第$t$次更新时参数为$w^{(t)}$，选择被错误分类的点$(x_i,y_i)$更新参数：

$$ w^{(t+1)} = w^{(t)} + y_ix_i $$

我们想要证明在更新中$w^{(t+1)}$与最优值$\hat{w}$越来越接近，从两个方面出发：
- 两者的内积越来越大；
- $w^{(t+1)}$的长度增长较慢；

不妨取$$\mid\mid \hat{w} \mid\mid = 1$$，不影响分类的结果;

计算更新后参数$w^{(t+1)}$与最优值$\hat{w}$的内积：

$$ \hat{w}^Tw^{(t+1)} = \hat{w}^T(w^{(t)} + y_ix_i) \\ = \hat{w}^Tw^{(t)} + y_i\hat{w}^Tx_i \\ ≥ \hat{w}^Tw^{(t)} + min \quad y_i{\hat{w}}^Tx_i \\ = \hat{w}^Tw^{(t)} + ρ $$

计算$w^{(t+1)}$的长度：

$$ \mid\mid w^{(t+1)} \mid\mid^2 = \mid\mid w^{(t)} + y_ix_i \mid\mid^2 \\ = \mid\mid w^{(t)} \mid\mid^2 + 2y_iw^{(t)}x_i + \mid\mid y_ix_i \mid\mid^2 \\ ≤ \mid\mid w^{(t)} \mid\mid^2 + \mid\mid x_i \mid\mid^2 \\ ≤ \mid\mid w^{(t)} \mid\mid^2 + max \quad \mid\mid x_i \mid\mid^2 \\ = \mid\mid w^{(t)} \mid\mid^2 + R^2 $$

若经过$T$次迭代后$$$$，由上面两式可以得到：

$$ \hat{w}^Tw^{(T)} ≥ \hat{w}^Tw^{(T-1)} + ρ ≥ Tρ $$

$$ \mid\mid w^{(T)} \mid\mid^2 ≤ \mid\mid w^{(T-1)} \mid\mid^2 + R^2 ≤ TR^2 $$

则计算参数$w^{(T)}$与最优值$\hat{w}$归一化后的内积：

$$ \frac{w^{(T)}}{\mid\mid w^{(T)} \mid\mid} · \frac{\hat{w}}{\mid\mid \hat{w} \mid\mid} ≥ \frac{Tρ}{\sqrt{TR^2} · \mid\mid \hat{w} \mid\mid^2} = \sqrt{T}\frac{ρ}{R} $$

即在更新时随$T$的增大参数$w^{(T)}$与最优值$\hat{w}$归一化后的内积不断增大，注意到这个值具有上限$1$,因此更新过程使得参数逐步趋近于最优值，这个算法是收敛的。

则可由下面的不等式：

$$ Tρ ≤ \hat{w}^Tw^{(T)} ≤ \mid\mid \hat{w} \mid\mid · \mid\mid w^{(T)} \mid\mid = \mid\mid w^{(T)} \mid\mid ≤ \sqrt{TR^2} $$

得到算法迭代次数的上界：

$$ T ≤ \frac{R^2}{ρ^2} $$

# 5. 口袋算法
当样本集线性不可分时，感知机是不收敛的。**口袋算法（pocket algorithm）**用来解决这种情况。

口袋算法的思想是，在感知机学习算法中，仍然每次选择一个错误分类的点更新参数；

参数更新后，如果此时分类错误的点数比之前记录的最优参数所分类错误的点数还要少，则把这个参数放入口袋中，作为最优参数的备选参数；

由于样本集线性不可分，这个循环过程是无限进行的，人为设置最大的循环次数，将最终存在于口袋中的参数作为算法得到的最优参数。

口袋算法相比于普通的感知机学习算法的速度要慢，原因在于：
- 需要格外的空间存储备选的最优参数；
- 需要额外的计算判断本次更新的参数是否是最优参数；