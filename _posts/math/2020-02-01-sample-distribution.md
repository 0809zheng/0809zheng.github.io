---
layout: post
title: '抽样分布(Sampling Distribution)'
date: 2020-02-01
author: 郑之杰
cover: 'https://github.com/0809zheng/imagebed_math_0/raw/main/5e8c6999504f4bcb0428b777.jpg'
tags: 数学
---

> Sampling Distribution.

# 抽样分布
**抽样分布(Sampling Distribution)**是指统计量的分布。

**本文目录**：
1. $\chi^2$分布
2. $t$分布
3. $F$分布
4. 抽样分布的重要定理

## 1. $\chi^2$分布
设一组样本$X_1,X_2,...,X_k$独立同分布于$N(0,1)$，记$X = \sum_{i=1}^{k} {X_i^2}$，则$X$服从自由度为$k$的$\chi^2$分布，记为$X$ ~ $\chi^2_k$。

其概率密度函数为：

$$ f_k(x)= \begin{cases} \frac{1}{2^{\frac{k}{2}}Γ(\frac{k}{2})} x^{\frac{k}{2}-1}e^{-\frac{x}{2}}, & x > 0 \\ 0, & x ≤ 0  \end{cases} $$

![](https://github.com/0809zheng/imagebed_math_0/raw/main/5e8c7bac504f4bcb04378571.png)

$\chi^2$分布的性质：
- $E(X) = k$
- $Var(X) = 2k$
- 设$$X_1$$ ~ $$\chi^2_{k_1}$$，$X_2$ ~ $$\chi^2_{k_2}$$，且$X_1$和$X_2$独立，则$X_1+X_2$ ~ $\chi^2_{k_1+k_2}$

## 2. $t$分布
$t$分布是英国统计学家W.S.Gosset在1908年以笔名Student发表的论文中提出的。

设随机变量$X$ ~ $N(0,1)$，$Y$ ~ $\chi^2_v$，且$X$和$Y$独立；记$T = \frac{X}{\sqrt{\frac{Y}{v}}}$，则$T$服从自由度为$v$的$t$分布，记为$T$ ~ $t_v$。

其概率密度函数为：

$$ p(x) = \frac{Γ(\frac{v+1}{2})}{Γ(\frac{v}{2})\sqrt{v\pi}} (1+\frac{x^2}{v})^{-\frac{v+1}{2}} $$

![](https://github.com/0809zheng/imagebed_math_0/raw/main/5e8c80d0504f4bcb043bb35b.png)

$t$分布的性质：
- $E(T) = 0, \quad v ≥ 2$
- $Var(T) = \frac{v}{v-2}, \quad v ≥ 3$
- 当$v$趋向于$∞$时，$t$分布趋向于标准正态分布$N(0,1)$。

## 3. $F$分布
设随机变量$X$ ~ $$\chi^2_{d_1}$$，$Y$ ~ $$\chi^2_{d_2}$$，且$X$和$Y$独立；记$F = \frac {\frac{X} {d_1}} {\frac{Y} {d_2}}$，则$F$服从自由度分别为$d_1$和$d_2$的$F$分布，记为$F$ ~ $$F_{d_1,d_2}$$。

其概率密度函数为：

$$ f_{d_1,d_2}(x)= \begin{cases} \frac{Γ(\frac{d_1+d_2}{2})}{Γ(\frac{d_1}{2})Γ(\frac{d_2}{2})} d_1^{\frac{d_1}{2}} d_2^{\frac{d_2}{2}} x^{\frac{d_1}{2}-1} (d_2+d_1x)^{-\frac{d_1+d_2}{2}}, & x > 0 \\ 0, & x ≤ 0  \end{cases} $$

![](https://github.com/0809zheng/imagebed_math_0/raw/main/5e8c829e504f4bcb043d234d.png)

$F$分布的性质：
- $E(F) = \frac{d_1}{d_2-2}, \quad d_2 ≥ 3$
- $Var(F) = \frac{2d_2^2(d_1+d_2-2)}{d_1(d_2-2)^2(d_2-4)}, \quad d_2 ≥ 5$
- 若$F$ ~ $$F_{m,n}$$，则$\frac{1}{F}$ ~ $$F_{n,m}$$
- 若$T$ ~ $$t_n$$，则$T^2$ ~ $$F_{1,n}$$
- $F_{m,n}(1-α) = \frac{1}{F_{n,m}(α)}$

## 4. 抽样分布的重要定理

### 定理1：单正态分布总体，方差已知

设一组样本$X_1,X_2,...,X_n$独立同分布于正态分布$N(μ,σ^2)$，$\overline{X}$为样本均值，$S^2$为样本方差，则有：
1. $$\overline{X}$$ ~ $$N(μ,\frac{σ^2}{n})$$；
2. $$(n-1) \frac{S^2}{σ^2}$$ ~ $$\chi^2_{n-1}$$；
3. $\overline{X}$和$S^2$独立。

### 定理2：单正态分布总体，方差未知

设一组样本$X_1,X_2,...,X_n$独立同分布于正态分布$N(μ,σ^2)$，$\overline{X}$为样本均值，$S^2$为样本方差，若记：

$$ T = \frac{\sqrt{n}(\overline{X}-μ)}{S} $$

则$T$~ $t_{n-1}$。

### 定理3：双正态分布总体，均值差

设一组样本$X_1,X_2,...,X_m$独立同分布于正态分布$N(μ_1,σ_1^2)$，另一组样本$Y_1,Y_2,...,Y_n$独立同分布于正态分布$N(μ_2,σ_2^2)$，

假定$$σ_1^2=σ_2^2=σ^2$$，样本$X_1,X_2,...,X_m$与$Y_1,Y_2,...,Y_n$独立，记：

$$ T = \frac{(\overline{X} - \overline{Y})-(μ_1-μ_2)}{S_w} \sqrt{\frac{mn}{n+m}} $$

其中$$(n+m-2)S_w^2 = (m-1)S_1^2+(n-1)S_2^2$$,

则$T$~ $t_{n+m-2}$。

### 定理4：双正态分布总体，方差比

设一组样本$X_1,X_2,...,X_m$独立同分布于正态分布$N(μ_1,σ_1^2)$，另一组样本$Y_1,Y_2,...,Y_n$独立同分布于正态分布$N(μ_2,σ_2^2)$，样本$X_1,X_2,...,X_m$与$Y_1,Y_2,...,Y_n$独立，记：

$$ F = \frac{\frac{S_1^2}{σ_1^2}}{\frac{S_2^2}{σ_2^2}} $$

则$F$~ $F_{m-1,n-1}$。

### 定理5：指数分布总体

设一组样本$X_1,X_2,...,X_n$独立同分布于指数分布$$f(x,λ)=λe^{-λx}I_{[x>0]}$$，

则$$2λn\overline{X}$$ ~ $$\chi^2_{2n}$$。
