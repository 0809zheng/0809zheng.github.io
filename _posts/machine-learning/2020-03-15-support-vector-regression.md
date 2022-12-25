---
layout: post
title: '支持向量回归(Support Vector Regression, SVR)'
date: 2020-03-15
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ed759b9c2a9a83be54d254c.jpg'
tags: 机器学习
---

> Support Vector Regression.

**支持向量回归**（**Support Vector Regression，SVR**）是一种启发于[支持向量机](https://0809zheng.github.io/2020/03/14/SVM.html)和[Tube回归](https://0809zheng.github.io/2020/03/29/tube.html)的回归方法。**SVR**从使用了**L2**正则化的**Tube**回归出发，借鉴了支持向量机中的“支持向量”的思想，使用稀疏的样本点来决定回归函数。

![](https://pic.downk.cc/item/5ed759b9c2a9a83be54d254c.jpg)

# 1. 原问题
若样本集$$X=\{x^{(1)},...,x^{(N)}\}$$，标签集$$y=\{y^{(1)},...,y^{(N)}\}$$，则使用了**L2**正则化的**Tube**回归的目标函数定义如下：

$$ \mathop{\min}_{w,b}  \frac{λ}{N}w^Tw + \frac{1}{N} \sum_{n=1}^{N} {\max(0,| w^Tx^{(n)}+b-y^{(n)} | - ε)} $$

参数$ε$表明我们能容忍预测值$w^Tx^{(n)}$和真实值$y^{(n)}$之间最多有$ε$的偏差。这相当于在回归线两侧设置了以宽度为$2ε$的中立区，当样本点落入中立区中则被认为是正确的，只有当样本点的位置远离了回归线的中立区，才计算其损失。

借鉴支持向量机的流程，用参数$C$取代参数$λ$，平衡正则化和对损失的容忍:

$$ \mathop{\min}_{w,b}  \frac{1}{2}w^Tw + C \sum_{n=1}^{N} {\max(0,| w^Tx^{(n)}+b-y^{(n)} | - ε)} $$

引入**松弛变量**$ξ_n^-,ξ_n^+$放松中立区(中立区两侧的松弛程度可能有所不同)，其中$ξ_n^+$表示**upper tube violations**，$ξ_n^-$表示**lower tube violations**。使得放松后的中立区能够覆盖所有样本点，此时问题转化成约束优化问题(所有样本点都在中立区内)，优化的经验风险变为引入的松弛变量尽可能地小：

$$ \begin{aligned} \mathop{\min}_{w,b,ξ}  & \frac{1}{2}w^Tw + C \sum_{n=1}^{N} {ξ_n^++ξ_n^-} \\ \text{s.t. }  &  -ε-ξ_n^- ≤ w^Tx^{(n)}+b-y^{(n)} ≤ ε+ξ_n^+ \\ &ξ_n^+ ≥ 0,ξ_n^- ≥ 0 \end{aligned} $$

上式是标准的二次规划问题。支持向量回归中需要设置的超参数是$C$和$ε$，求解的二次规划问题具有$2N+d+1$个变量，具有$2N+2$个约束条件。由于变量数量与样本特征维度$d$有关，为了减少计算量引入对偶问题。

# 2. 对偶问题
引入拉格朗日乘子$α_n^-$、$α_n^+$、$β_n^-$、$β_n^+$，则拉格朗日函数：

$$ \begin{aligned} L(α,β) = &\frac{1}{2}w^Tw + C \sum_{n=1}^{N} {(ξ_n^++ξ_n^-)} \\ &+ \sum_{n=1}^{N} {α_n^- (-ε-ξ_n^- - w^Tx^{(n)}+b-y^{(n)})} \\ &+ \sum_{n=1}^{N} {α_n^+ (w^Tx^{(n)}+b-y^{(n)} - ε+ξ_n^+)} \\ &+ \sum_{n=1}^{N} {-β_n^-ξ_n^-} + \sum_{n=1}^{N} {-β_n^+ξ_n^+} \end{aligned} $$

原问题记为：

$$ \begin{aligned} \mathop{\min}_{w,b,ξ}  \mathop{\max}_{α,β}  & L(α,β) \\ \text{s.t. }  &-ε-ξ_n^- ≤ w^Tx^{(n)}+b-y^{(n)} ≤ ε+ξ_n^+ \\ & ξ_n^+ ≥ 0,ξ_n^- ≥ 0 \end{aligned} $$

根据[约束优化的对偶理论](https://0809zheng.github.io/2022/09/23/minimax.html)转化为对偶问题：

$$ \begin{aligned} \mathop{\max}_{α,β}  \mathop{\min}_{w,b,ξ} & L(α,β) \\ \text{s.t. }& α_n^- ≥ 0,α_n^+ ≥ 0,β_n^- ≥ 0,β_n^+ ≥ 0 \end{aligned} $$

由$KKT$条件：
- 原问题的约束条件：$-ε-ξ_n^- ≤ w^Tx^{(n)}+b-y^{(n)} ≤ ε+ξ_n^+ , ξ_n^+ ≥ 0,ξ_n^- ≥ 0$
- 对偶问题的约束条件：$α_n^- ≥ 0,α_n^+ ≥ 0,β_n^- ≥ 0,β_n^+ ≥ 0$
- 拉格朗日函数的梯度为零：由$\frac{\partial L(α,β)}{\partial w_i}=0$得$w=\sum_{n=1}^{N} {(α_n^+-α_n^-)x^{(n)}}$；由$\frac{\partial L(α,β)}{\partial b}=0$得$\sum_{n=1}^{N} {(α_n^+-α_n^-)}=0$
- **互补松弛条件complementary slackness**：$α_n^+ (w^Tx^{(n)}+b-y^{(n)} - ε+ξ_n^+)=0$、$α_n^- (-ε-ξ_n^- - w^Tx^{(n)}+b-y^{(n)})=0$

整理并化简：

$$ \begin{aligned} \mathop{\min}_{α} & \frac{1}{2}\sum_{n=1}^{N} {\sum_{m=1}^{N} {(α_n^+-α_n^-)(α_m^+-α_m^-){(x^{(n)})}^Tx^{(m)}}} \\&+ \sum_{n=1}^{N} {((ε-y^{(n)})α_n^++(ε+y^{(n)})α_n^-)} \\ \text{s.t. }& \sum_{n=1}^{N} {(α_n^+-α_n^-)}=0 \\& C ≥ α_n^- ≥ 0,C ≥ α_n^+ ≥ 0 \end{aligned} $$

这也是一个二次规划问题，具有$2N$个变量，具有$2N+1$个约束条件，与样本维度$d$无关。

求解得到$α_n^-$、$α_n^+$后，原问题的解为：

$$ \begin{aligned} w&=\sum_{n=1}^{N} {(α_n^+-α_n^-)x^{(n)}} \\ y&=w^Tx=\sum_{n=1}^{N} {(α_n^+-α_n^-)(x^{(n)})^Tx} \end{aligned} $$

稀疏性的说明：
- 对于分布在中立区$\| w^Tx^{(n)}+b-y^{(n)} \| < ε+ξ_n$内的点，由互补松弛条件，$α_n^-=0$、$α_n^+=0$；因此这些样本点对结果没有贡献。而位于中立区之上的点，起到支持向量的作用。

# 3. 核方法
将[核方法](https://0809zheng.github.io/2021/07/23/kernel.html)引入支持向量回归，可以增强其非线性表示能力。

引入核函数$K(x,x')={φ(x)}^Tφ(x')$来代替样本的特征转换和内积运算，可以得到支持向量回归的对偶形式：

$$ \begin{aligned} \mathop{\min}_{α} & \frac{1}{2}\sum_{n=1}^{N} \sum_{m=1}^{N} {(α_n^+-α_n^-)(α_m^+-α_m^-){K(x^{(n)},x^{(m)})}} \\&+ \sum_{n=1}^{N} {((ε-y^{(n)})α_n^++(ε+y^{(n)})α_n^-)} \\ \text{s.t. }& \sum_{n=1}^{N} {(α_n^+-α_n^-)}=0 \\ &C ≥ α_n^- ≥ 0,C ≥ α_n^+ ≥ 0 \end{aligned} $$

引入核方法后支持向量回归的最终结果为：

$$ y=w^Tx=\sum_{n=1}^{N} {(α_n^+-α_n^-)K(x^{(n)},x)} $$
