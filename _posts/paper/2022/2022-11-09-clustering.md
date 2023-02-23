---
layout: post
title: 'Deep Metric Learning via Facility Location'
date: 2022-11-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63ca15d5be43e0d30e45feab.jpg'
tags: 论文阅读
---

> 通过设施位置实现深度度量学习.

- paper：[Deep Metric Learning via Facility Location](https://arxiv.org/abs/1612.01213)

在进行深度度量学习时，仅考虑样本对的局部关系可能会学习到错误的度量。比如一对正样本(由蓝色线连接的紫色圆)可能会由于负样本的选择而彼此分离，从而形成两个聚类簇。本文提出了**clustering loss**，预先为每个类别的样本指定一个聚类中心，使得每一类样本的距离之和尽可能小，不同类样本间的距离尽可能大。

![](https://pic.imgdb.cn/item/63ca176ebe43e0d30e48fcc8.jpg)

给定一个聚类中心集合$S$，则把样本集$X$划分到不同聚类簇的聚类得分是一个设施位置问题(**Facility location problem**)，定义为：

$$ F(X,S;\theta) = -\sum_{i \in |X|} \mathop{\min}_{j \in S} ||f_{\theta}(X_i)-f_{\theta}(X_j)|| $$

对于有标签的样本集$(X,Y)$，从每个类别中选择一个样本为聚类中心，则最佳聚类得分定义为：

$$ \tilde{F}(X,Y;\theta)=\sum_k^{|Y|} \mathop{\max}_{j \in \{ i:Y_i=k \} }  F(X_{\{ i:Y_i=k \} },\{j\};\theta) $$

**clustering loss**使得最佳聚类得分$$\tilde{F}$$比任意其他聚类划分$g(S)$的聚类得分不低于结构化边界$\Delta$：

$$ \max(0, \mathop{\max}_{S \subset V,|S| = |Y|} \{ F(X,S;\theta)+\gamma \Delta(g(S),Y) \} - \tilde{F}(X,Y;\theta) ) $$

其中结构化边界$\Delta$通过归一化互信息(**normalized mutual informati**)定义：

$$ \Delta(y_1,y_2) = 1-\frac{MI(y_1,y_2)}{\sqrt{H(y_1)H(y_2)}} $$

![](https://pic.imgdb.cn/item/63ca1d5fbe43e0d30e5041c4.jpg)