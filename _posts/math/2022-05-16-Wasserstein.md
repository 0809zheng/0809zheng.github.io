---
layout: post
title: '最优传输(Optimal Transport)问题与Wasserstein距离'
date: 2022-05-16
author: 郑之杰
cover: ''
tags: 数学
---

> Wasserstein Distance.

本文目录：
1. 最优传输问题 Optimal Transport Problem
2. 最优传输问题的对偶问题 Dual Problem
3. Wasserstein距离及其对偶形式

# 1. 最优传输问题 Optimal Transport Problem

对于两个概率分布$p(\textbf{x})$和$q(\textbf{x})$，**最优传输问题(optimal transport problem)**是指通过最少的成本把$p(\textbf{x})$转变为$q(\textbf{x})$，而最佳运输方案所对应的最低成本则称为**Wasserstein距离**。

若假设概率分布$p(\textbf{x})$和$q(\textbf{x})$代表两堆石子，则问题等价于如何移动一堆石子，通过最小的累积移动距离把它堆成另外一堆石子。因此**Wasserstein**距离也被称作**推土机距离(Earth Mover's Distance)**。

![](https://pic.imgdb.cn/item/6281fd3e0947543129711307.jpg)

记从位置$x$运输到位置$y$的成本为$d(x,y)$，联合分布$\gamma(x,y)$描述了一种可行的运输方案，表示应该从位置$x$处运输多少货物到位置$y$处，才能使$p(\textbf{x})$和$q(\textbf{x})$具有相同的概率分布。在离散形势下，联合分布$\gamma(x,y)$表示为一个矩阵：

![](https://pic1.imgdb.cn/item/6331819916f2c2beb1ca5a64.jpg)

其中矩阵的每一行代表概率分布$p(\textbf{x})$的某个位置$x_p$要分配到概率分布$q(\textbf{x})$不同位置处的值；每一列代表概率分布$q(\textbf{x})$的某个位置$x_q$接收到概率分布$p(\textbf{x})$的不同位置分配的值。在该联合分布下，概率分布变换的总成本为：

$$ \sum_{x_p,x_q} \gamma(x_p,x_q) d(x_p,x_q) = \Bbb{E}_{(x,y) \in \gamma(\textbf{x},\textbf{y})} [d(x,y)] $$

一般地，**Wasserstein**距离定义如下：

$$ \begin{aligned} \mathcal{W}[p,q] &= \mathop{\inf}_{\gamma \in \Pi[p,q]} \Bbb{E}_{(x,y) \in \gamma(\textbf{x},\textbf{y})} [d(x,y)] \\ & = \mathop{\inf}_{\gamma \in \Pi[p,q]} \int \int \gamma(x,y) d(x,y) dxdy \end{aligned} $$

其中$\Pi[p,q]$是$p$和$q$的所有可能联合分布的集合，下确界**infimum**表示寻找总运输成本最小的方案。不失一般性地假设$p(\textbf{x})$是原始分布，$q(\textbf{y})$是目标分布，则约束$p(\textbf{x})$和$q(\textbf{y})$是联合分布$\gamma(\textbf{x},\textbf{y})$的边缘分布：

$$ \int \gamma(x,y) dy = p(\textbf{x}), \quad \int \gamma(x,y)dx = q(\textbf{y}) $$

上式分别表示从每个$dy$处把$\gamma(x,y)$的物品搬回到$x$处，从而还原概率分布$p(\textbf{x})$；以及从每个$dx$处搬运$\gamma(x,y)$的物品到$y$处，从而得到概率分布$p(\textbf{y})$。



最优传输问题等价于如下约束优化问题：

$$ \begin{aligned} \mathop{\inf}_{\gamma \in \Pi[p,q]} & \int \int \gamma(x,y) d(x,y) dxdy \\ \text{s.t. } & \int \gamma(x,y) dy = p(x) \\ & \int \gamma(x,y)dx = q(y) \\ & \gamma(x,y) \geq 0 \end{aligned} $$

若$p,q$为离散型概率分布，则该优化问题可以表示为矩阵形式。记：

$$ \Gamma = \begin{pmatrix} \gamma(x_1,y_1) \\ \gamma(x_1,y_2) \\ \vdots \\ \gamma(x_1,y_n) \\ \gamma(x_2,y_1) \\ \gamma(x_2,y_2) \\ \vdots \\ \gamma(x_2,y_n) \\ \vdots \\ \gamma(x_n,y_1) \\ \gamma(x_n,y_2) \\ \vdots \\ \gamma(x_n,y_n) \end{pmatrix}, \quad D = \begin{pmatrix} d(x_1,y_1) \\ d(x_1,y_2) \\ \vdots \\ d(x_1,y_n) \\ d(x_2,y_1) \\ d(x_2,y_2) \\ \vdots \\ d(x_2,y_n) \\ \vdots \\ d(x_n,y_1) \\ d(x_n,y_2) \\ \vdots \\ d(x_n,y_n) \end{pmatrix} $$

则最优传输问题的目标函数可以用列向量$\Gamma$和$D$的内积表示：

$$ \int \int \gamma(x,y) d(x,y) dxdy = <\Gamma, D> $$

另一方面，两个边缘分布的约束条件也可以统一写作矩阵形式：

$$ \begin{pmatrix} 1 & 1 & \cdots & 1 & 0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 \\0 & 0 & \cdots & 0 & 1 & 1 & \cdots & 1 & 0 & 0 & \cdots & 0 \\\vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 & 1 & 1 & \cdots & 1 \\ 1 & 0 & \cdots & 0 & 1 & 0 & \cdots & 0 & 1 & 0 & \cdots & 0 \\0 & 1 & \cdots & 0 & 0 & 1 & \cdots & 0 & 0 & 1 & \cdots & 0 \\\vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & \cdots & 1 & 0 & 0 & \cdots & 1 & 0 & 0 & \cdots & 1  \end{pmatrix}  \begin{pmatrix} \gamma(x_1,y_1) \\ \gamma(x_1,y_2) \\ \vdots \\ \gamma(x_1,y_n) \\ \gamma(x_2,y_1) \\ \gamma(x_2,y_2) \\ \vdots \\ \gamma(x_2,y_n) \\ \vdots \\ \gamma(x_n,y_1) \\ \gamma(x_n,y_2) \\ \vdots \\ \gamma(x_n,y_n) \end{pmatrix} = \begin{pmatrix} p(x_1) \\ p(x_2) \\ \vdots \\ p(x_n) \\ q(y_1) \\ q(y_2) \\ \vdots \\ q(y_n)  \end{pmatrix} $$

将上式记作$A\Gamma = b$，并将非负约束记作$\Gamma \geq 0$。则最优传输问题也可以表示为一个线性规划问题：

$$ \mathop{\min}_{\Gamma} \{ <\Gamma, D> | A\Gamma = b, \Gamma \geq 0 \} $$

# 2. 最优传输问题的对偶问题 Dual Problem

对于一个线性规划问题，总可以写出其[<font color=blue>对偶形式</font>](https://0809zheng.github.io/2022/09/22/dual.html):

$$ \mathop{\min}_{\Gamma} \{ <\Gamma, D> | A\Gamma = b, \Gamma \geq 0 \} = \mathop{\max}_{F} \{ <b, F> | A^TF \leq D \} $$

不失一般性地，可以将$F$记为：

$$ F = \begin{pmatrix} f(x_1) & f(x_2) & \cdots & f(x_n) & g(y_1) & g(y_2) & \cdots & g(y_n)  \end{pmatrix}^T $$

则对偶问题的目标函数为：

$$ \begin{aligned} <b, F> &= \sum_n p(x_n)f(x_n) + \sum_nq(y_n)g(y_n) \\ &= \sum_n p(x_n)f(x_n) + \sum_nq(x_n)g(x_n) \\ &= \int [p(x) f(x) +q(x)g(x)]dx   \end{aligned} $$

约束条件$A^TF \leq D$为：

$$ \begin{pmatrix} 1 & 0 & \cdots & 0 & 1 & 0 & \cdots & 0  \\ 1 & 0 & \cdots & 0 & 0 & 1 & \cdots & 0  \\ \vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots  \\ 1 & 0 & \cdots & 0 & 0 & 0 & \cdots & 1  \\ 0 & 1 & \cdots & 0 & 1 & 0 & \cdots & 0  \\ 0 & 1 & \cdots & 0 & 0 & 1 & \cdots & 0  \\ \vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots  \\ 0 & 1 & \cdots & 0 & 0 & 0 & \cdots & 1 \\ 0 & 0 & \cdots & 1 & 1 & 0 & \cdots & 0  \\ 0 & 0 & \cdots & 1 & 0 & 1 & \cdots & 0  \\ \vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots  \\ 0 & 0 & \cdots & 1 & 0 & 0 & \cdots & 1  \end{pmatrix}  \begin{pmatrix} f(x_1) \\ f(x_2) \\ \vdots \\ f(x_n) \\ g(y_1) \\ g(y_2) \\ \vdots \\ g(y_n)  \end{pmatrix} \leq \begin{pmatrix} d(x_1,y_1) \\ d(x_1,y_2) \\ \vdots \\ d(x_1,y_n) \\ d(x_2,y_1) \\ d(x_2,y_2) \\ \vdots \\ d(x_2,y_n) \\ \vdots \\ d(x_n,y_1) \\ d(x_n,y_2) \\ \vdots \\ d(x_n,y_n) \end{pmatrix} $$

上式等价于：

$$ \forall i,j \quad f(x_i) + g(y_i) \leq d(x_i+y_i) $$

或写作：

$$ \forall x,y \quad f(x) + g(y) \leq d(x+y) $$

因此，最优传输问题的对偶问题为：

$$ \begin{aligned} \mathop{\sup}_{f,g}  & \int [p(x) f(x) +q(x)g(x)]dx \\ \text{s.t. } & f(x)+g(y) \leq  d(x,y) \end{aligned} $$

其中上确界**supremum**表示寻找使得目标函数最大的$f$和$g$。

# 3. Wasserstein距离及其对偶形式

**Wasserstein**距离定义为如下最优化问题：

$$ \begin{aligned} \mathcal{W}[p,q] = \mathop{\inf}_{\gamma \in \Pi[p,q]} & \int \int \gamma(x,y) d(x,y) dxdy \\ \text{s.t. } & \int \gamma(x,y) dy = p(x) \\ & \int \gamma(x,y)dx = q(y) \\ & \gamma(x,y) \geq 0 \end{aligned} $$

**Wasserstein**距离的一个对偶形式为：

$$ \mathcal{W}[p,q] =  \mathop{\sup}_{f,g} \{  \int [p(x) f(x) +q(x)g(x)]dx | f(x)+g(y) \leq  d(x,y) \} $$

注意到：

$$ f(x)+g(x) \leq  d(x,x) = 0 $$

因此有$g(x) \leq - f(x)$，则目标函数：

$$ p(x) f(x) +q(x)g(x) \leq p(x) f(x) -q(x)f(x) $$

放大后的上确界不会小于原来的上确界，因此不妨取$f(x)=-g(x)$。则**Wasserstein**距离简化为：

$$ \mathcal{W}[p,q] =  \mathop{\sup}_{f} \{  \int [p(x) f(x) -q(x)f(x)]dx | f(x)-f(y) \leq  d(x,y) \} $$

其中$f(x)-f(y) \leq  d(x,y)$为**Lipschitz约束**，记为$$\|f\|_L \leq 1$$；$p,q$是概率分布，因此积分可以写作采样形式。因此**Wasserstein**距离也可以写作：

$$ \mathcal{W}[p,q] =  \mathop{\sup}_{f, ||f||_L \leq 1} \{  \Bbb{E}_{x \text{~} p(x)} [ f(x)] -\Bbb{E}_{x \text{~}q(x)}[f(x)]\} $$