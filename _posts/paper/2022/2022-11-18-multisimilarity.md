---
layout: post
title: 'Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning'
date: 2022-11-18
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63cdf3dfbe43e0d30e292cdd.jpg'
tags: 论文阅读
---

> 深度度量学习的多重相似性损失与通用对加权.

- paper：[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](https://arxiv.org/abs/1904.06627)

在深度度量学习中，有一系列的基于对(**pair-based**)的损失函数被提出。本文提出了一种基于对的度量损失的通用加权框架，称为通用对加权(**General Pair Weighting**)，把深度度量学习中的采样问题看成通过梯度分析的对加权问题，在此基础上提出了一个新的损失函数：多重相似性损失(**multi-similarity loss**)。

# 1. 通用对加权 General Pair Weighting

基于对的度量损失函数$$\mathcal{L}$$可以表示为样本的相似性矩阵$S$和标签$y$的函数$$\mathcal{L}(S,y)$$，模型参数$\theta$在迭代中的导数可以表示为：

$$ \frac{\partial \mathcal{L}(S,y)}{\partial \theta} = \frac{\partial \mathcal{L}(S,y)}{\partial S} \frac{\partial S}{\partial \theta} = \sum_{i=1}^N \sum_{j=1}^N \frac{\partial \mathcal{L}(S,y)}{\partial S_{ij}} \frac{\partial S_{ij}}{\partial \theta} $$

把上式用一个新的函数$$\mathcal{F}$$写成一个新的形式：

$$ \mathcal{F}(S,y) = \sum_{i=1}^N \sum_{j=1}^N \frac{\partial \mathcal{L}(S,y)}{\partial S_{ij}}  S_{ij} $$

深度度量学习的中心思想就是让正样本对更近，让负样本对更远。对于一个基于对的损失函数$$\mathcal{L}$$，负样本对$(i,j)$满足$$\frac{\partial \mathcal{L}(S,y)}{\partial S_{ij}} \leq 0$$，正样本对$(i,j)$满足$$\frac{\partial \mathcal{L}(S,y)}{\partial S_{ij}} \geq 0$$。因此上式可以变换成下面这种通用对加权的形式：

$$ \begin{aligned} \mathcal{F}(S,y) &= \sum_{i=1}^N (\sum_{y_j \neq y_i} |\frac{\partial \mathcal{L}(S,y)}{\partial S_{ij}}|  S_{ij}-\sum_{y_j = y_i}| \frac{\partial \mathcal{L}(S,y)}{\partial S_{ij}}|  S_{ij}) \\ &= \sum_{i=1}^N (\sum_{y_j \neq y_i} w_{ij}^-  S_{ij}-\sum_{y_j = y_i}w_{ij}^+  S_{ij}) \end{aligned} $$

上式表示基于对的方法能够被公式化为一个逐对(**pair-wise**)加权的相似性表达式，样本对$(i,j)$的权重是$w_{ij}$。

# 2. 多重相似性损失 multi-similarity loss

通用对加权公式在学习过程中给每一个样本对$(i,j)$动态赋予一个权重。给样本赋权的核心在于判断样本的局部分布，即它们之间的相似性，局部样本之间的分布和相互关系并不仅仅取决于当前两个样本之间的距离或相似性，还取决于当前样本对与其周围样本对之间的关系。

因此对于每一个样本对$(i,j)$，不仅需要考虑样本对本身的自相似性，同时还要考虑它与其它样本对的相对相似性。其中相对相似性又可以分为正相对相似性 (正样本)、负相对相似性（负样本）两种相似性。
- 自相似性(**Self-similarity**)：根据样本对计算出的相似性，自相似性很难完整地描述嵌入空间的样本分布情况；**Contrastive loss**和**Binomial Deviance Loss**基于这个准则。
- 正相对相似性(**Positive relative similarity**)：不仅考虑当前样本对的相似性，还考虑局部邻域内正样本对之间的相对关系；**Triplet loss**基于这个准则。
- 负相对相似性(**Negative relative similarity**)：不仅考虑当前样本对的相似性，还考虑局部邻域内负样本对之间的相对关系；**Lifted Structure Loss**基于这个准则。

![](https://pic.imgdb.cn/item/63cdfb41588a5d166c716c32.jpg)

![](https://pic.imgdb.cn/item/63cdfb6f588a5d166c71991b.jpg)


**Multi-Similarity Loss**通过定义样本对的自相似性和相对相似性，在训练过程中更加全面地考虑了局部样本分布，从而能更高效精确的对重要样本对进行采样和加权。

![](https://pic.imgdb.cn/item/63cdffc6588a5d166c79e696.jpg)

对于给定的负样本对$(i,j)$，联合计算其自相似度和负相对相似度设置权重：

$$ w_{ij}^- = \frac{1}{e^{\beta(\lambda-S_{ij})} + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik}-S_{ij})}} $$

对于给定的正样本对$(i,j)$，联合计算其自相似度和正相对相似度设置权重：

$$ w_{ij}^+ = \frac{1}{e^{-\alpha(\lambda-S_{ij})} + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik}-S_{ij})}} $$

根据通用对加权公式，损失函数可以构造为：

$$ \frac{1}{\alpha} \log(1+\sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik}-\lambda)}) + \frac{1}{\beta} \log(1+\sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik}-\lambda)}) $$
