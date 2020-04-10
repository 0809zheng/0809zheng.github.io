---
layout: post
title: '数理统计学的基本概念'
date: 2020-02-02
author: 郑之杰
cover: ''
tags: 数学
---

> Mathematic Statistics.

# 数理统计学的基本概念

**数理统计学(Mathematic Statistics)**，是指使用概率论的方法，通过试验或观察收集带有随机误差的数据，并在设定的统计模型之下，对数据进行统计分析，对所研究的问题做出统计推断。

## 1. 总体
**总体(population)**是与所研究的问题有关的全部个体组成的集合。

通常赋予总体一个概率分布，称为**统计总体**。

总体分布是某个概率分布族的一员。

## 2. 样本
**样本(sample)**是按一定的规定从总体中抽出的一部分个体。

记一组样本为$X_1,X_2,...,X_n$，$n$是样本容量。

通常假设样本独立同分布(i.i.d.)于总体分布，称为**独立随机样本**。

对样本的认识：
- 在计算阶段，把样本看作数据，即样本观测值；
- 在理论研究阶段，把样本看作随机变量。

## 3. 统计量
**统计量(statistics)**是完全由样本所决定的量。

统计量只依赖于样本，不包含其他未知量；特别地，不能依赖于总体分布中的未知参数。

常用统计量：
- 样本均值：$\overline{X} = \frac{1}{n}(X_1+X_2+...+X_n) = \frac{1}{n} \sum_{i=1}^{n} {X_i}$
- 样本方差：$S^2 = \frac{1}{n-1} \sum_{i=1}^{n} {(X_i-\overline{X})^2}$
- 样本$k$阶原点矩：$a_k = \frac{1}{n} \sum_{i=1}^{n} {X_i^k}$
- 样本$k$阶中心矩：$m_k = \frac{1}{n} \sum_{i=1}^{n} {(X_i-\overline{X})^k}$
- 顺序统计量：$$X_{(1)}=min(X_1,X_2,...,X_n)$$, $$X_{(n)}=max(X_1,X_2,...,X_n)$$


