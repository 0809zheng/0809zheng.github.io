---
layout: post
title: 'Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere'
date: 2022-10-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63d48b23face21e9ef1eeecb.jpg'
tags: 论文阅读
---

> 通过超球面上的对齐和一致性理解对比表示学习.

- paper：[Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242)

本文提出了对比学习损失的两种性质：**对齐性(Alignment)**和**一致性(Uniformity)**；并提出了衡量这两种性质的评价指标；通过优化这两个指标学到的特征在下游任务上表现更好。

对比学习是指使得相似的样本具有相近的特征表示，不相似的样本具有不同的特征表示。给定数据分布$p_{data}(\cdot)$和正样本对分布$p_{pos}(\cdot,\cdot)$，这两个分布应该满足：
- **对称性(Symmetry)**：$\forall x,x^+，p_{pos}(x,x^+)=p_{pos}(x^+,x)$
- **匹配边缘分布(Matching marginal)**：$\forall x,\int p_{pos}(x,x^+)dx^+ = p_{data}(x)$

神经网络$f(x)$学习到的特征向量经过**L2**归一化，相当于分布在单位超球面上；此时内积$f(x_1)^Tf(x_2)$等价于余弦相似度；对应的对比损失定义如下：

$$ \mathcal{L}_{contrastive} = \Bbb{E}_{(x,x^+)\text{~}p_{pos},\{x_i^-\}_{i=1}^M\text{~}p_{data}} [-\log \frac{\exp(f(x)^Tf(x^+)/\tau)}{\exp(f(x)^Tf(x^+)/\tau)+\sum_{i=1}^{M}\exp(f(x)^Tf(x^-_i)/\tau)}] $$

对比学习中通常负样本的数量远大于正样本，则上式可以写作：

$$ \begin{aligned} \mathcal{L}_{contrastive} &= \Bbb{E}_{(x,x^+)\text{~}p_{pos},\{x_i^-\}_{i=1}^M\text{~}p_{data}} [- f(x)^Tf(x^+)/\tau + \log (\exp(f(x)^Tf(x^+)/\tau)+\sum_{i=1}^{M}\exp(f(x)^Tf(x^-_i)/\tau))] \\ &\approx \Bbb{E}_{(x,x^+)\text{~}p_{pos},\{x_i^-\}_{i=1}^M\text{~}p_{data}} [- f(x)^Tf(x^+)/\tau + \log (\sum_{i=1}^{M}\exp(f(x)^Tf(x^-_i)/\tau))] \\ &= -\frac{1}{\tau} \Bbb{E}_{(x,x^+)\text{~}p_{pos}}[f(x)^Tf(x^+)] + \Bbb{E}_{x\text{~}p_{data}}[\log \Bbb{E}_{x^-\text{~}p_{data}}[\sum_{i=1}^{M}\exp(f(x)^Tf(x^-_i)/\tau)]] \end{aligned} $$

上式中第一项为**对齐性(Alignment)**，用于衡量正样本对之间的相似程度；第二项为**一致性(Uniformity)**，用于衡量归一化的特征在超球面上分布的均匀性。

![](https://pic.imgdb.cn/item/63d4917eface21e9ef2ca8e7.jpg)

让特征分布在单位超球面上，能够提高训练的稳定性，并且使得不同类别的特征被很好的聚类，在整个特征空间上类别是更容易被线性可分。

![](https://pic.imgdb.cn/item/63d492aeface21e9ef2ef0f6.jpg)

基于对比损失的分解，作者重新设计了量化**对齐性(Alignment)**和**一致性(Uniformity)**的损失函数。对齐性采用欧氏距离衡量，一致性采用高斯势核函数衡量：

$$ \begin{aligned} \mathcal{L}_{align}(f;\alpha) &=  \Bbb{E}_{(x,y)\text{~}p_{pos}} [||f(x)-f(y)||_2^{\alpha}] \\ \mathcal{L}_{uniform}(f;t) &= \log \Bbb{E}_{(x,y)\text{~}p_{data}} [e^{-t||f(x)-f(y)||_2^2}] \end{aligned} $$

采用高斯核的损失形式更简单，不需要做**softmax**计算，并且无监督训练的特征分布更加均匀。

![](https://pic.imgdb.cn/item/63d49573face21e9ef34a00a.jpg)

使用**Pytorch**实现上述损失的计算：

![](https://pic.imgdb.cn/item/63d49640face21e9ef363ab4.jpg)

实验结果表明，两个损失越小，相应的下游任务的效果越好。

![](https://pic.imgdb.cn/item/63d4975cface21e9ef387927.jpg)

对两个损失的权重分析如下：

![](https://pic.imgdb.cn/item/63d4972aface21e9ef380fc7.jpg)