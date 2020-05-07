---
layout: post
title: '层次聚类'
date: 2020-05-03
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eb376e3c2a9a83be5dfbdb8.jpg'
tags: 机器学习
---

> Hierarchical Clustering.

**层次聚类Hierarchical Clustering**是对数据集进行层次的分解。

根据分解的顺序，层次聚类可以分为**凝聚的agglomerate**和**分裂的divisive**。

层次聚类最终得到数据对象组成的一颗聚类的**树**。

层次聚类的**局限**在于，一旦一次合并或分裂被执行，就不能修改。

**本文目录**：
1. Agglomerative Nesting
2. Divisive Analysis
3. 簇间距离

# 1. Agglomerative Nesting
**Agglomerative Nesting(AGNES)**是一种**凝聚**的**层次聚类**算法。

**凝聚**的方法是一种**自底向上**的方法。一开始将每一个数据点作为一簇，然后相继合并相邻的簇，直到所有的数据点合并为一簇或满足终止条件。

# 2. Divisive Analysis
**Divisive Analysis(DIANA)**是一种**分裂**的**层次聚类**算法。

**分裂**的方法是一种**自顶向下**的方法。一开始将所有数据点置于一个簇，每一次迭代把一个簇分裂成更小的簇，直至每个数据单独为一簇或满足终止条件。

# 3. 簇间距离
假设两个簇$C_i$和$C_j$分别有$n_i$和$n_j$个数据，其质心为$m_i$和$m_j$,

普遍采用的**簇间距离**度量方法：

1. 最小距离：$$d_{min}(C_i,C_j)=min_{(p \in C_i,p' \in C_j)}\mid p-p' \mid$$
2. 最大距离：$$d_{max}(C_i,C_j)=max_{(p \in C_i,p' \in C_j)}\mid p-p' \mid$$
3. 平均距离：$$d_{avg}(C_i,C_j)=\frac{1}{n_in_j} \sum_{p \in C_i}^{} {\sum_{p' \in C_j}^{} {\mid p-p' \mid}}$$
4. 均值距离：$$d_{mean}(C_i,C_j)= \mid m_i-m_j \mid$$
