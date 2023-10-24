---
layout: post
title: 'Learning from Noisy Anchors for One-stage Object Detection'
date: 2021-06-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6535c5aac458853aef0d4952.jpg'
tags: 论文阅读
---

> 单阶段目标检测器从噪声Anchor中学习.

- paper：[Learning from Noisy Anchors for One-stage Object Detection](https://arxiv.org/abs/1912.05086)


目前最先进的目标检测器依赖于人为设置一系列可能的**Anchor**并回归和分类，这些**Anchor**根据它们与相应的**GT**的**IoU**分为正样本和负样本。这样的设置方法会导致歧义性标签的产生，这可能会产生噪音，并且对训练具有挑战性。

作者通过设计与**Anchor**相关联的**cleanliness score**来缓解由不完美的标签分配产生的噪声影响。在不增加任何额外计算开销的情况下估计出的**cleanliness score**，不仅可以作为软标签来监督分类分支的训练，而且作为样本重加权因子来提高定位和分类精度。


根据**IOU**选出的**TOP-N**样本分别作为候选正样本$A_{pos}$和候选负样本$A_{neg}$，并为其设置软标签：

$$
c = \begin{cases}
\alpha\cdot \text{loc\_a} + (1-\alpha)\cdot \text{cls\_c}, & b \in A_{pos} \\
0, & b \in A_{neg}
\end{cases}
$$

**loc_a**表示定位置信度，采用预测**box**和对应的**GT**之间的**IOU**衡量；**cls_c**表示分类置信度，通过网络**head**直接预测。

对于候选正样本$A_{pos}$，作者进一步引入了损失函数的软权重：

$$
r = \left( \alpha\cdot f(\text{loc\_a}) + (1-\alpha)\cdot f(\text{cls\_c}) \right)^\gamma
$$

其中$f(x)=1/(1-x)$，和$γ$都是用来增大方差的。

![](https://pic.imgdb.cn/item/65362c12c458853aef186508.jpg)

