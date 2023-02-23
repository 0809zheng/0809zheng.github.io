---
layout: post
title: 'ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring'
date: 2022-09-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63be815ebe43e0d30ed53812.jpg'
tags: 论文阅读
---

> ReMixMatch：通过分布对齐和增强锚点实现半监督学习.

- paper：[ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785)

**ReMixMatch**是一种半监督学习方法，它在[<font color=blue>MixMatch</font>](https://0809zheng.github.io/2022/09/11/mixmatch.html)方法的基础上进行了两点改进：分布对齐(**Distribution Alignment**)和增强锚点(**Augmentation Anchoring**)。

![](https://pic.imgdb.cn/item/63bf70d6be43e0d30e3efb43.jpg)

分布对齐是指把构造的伪标签的分布$$p(\hat{y})$$调整为更接近已标注样本的标签分布$p(y)$。记伪标签分布的滑动平均$$\tilde{p}(\hat{y})$$，则调整伪标签样本$u$的伪标签：

$$ f_{\theta}(u) \leftarrow f_{\theta}(u) \cdot \frac{p(y)}{\tilde{p}(\hat{y})} $$

增强锚点是指给定未标注样本$u$，首先通过较弱的数据增强生成一个样本锚点，然后通过$K$次较强增强的预测均值构造伪标签。较强的数据增强选用**CTAugment**，只采样使得模型预测在网络容忍度之内的增强。

![](https://pic.imgdb.cn/item/63bf7222be43e0d30e40c6d9.jpg)

**ReMixMatch**的损失函数包括：
- 对已标注数据应用数据增强和**MixUp**后的监督损失；
- 对未标注数据应用数据增强和**MixUp**后的无监督损失（以伪标签为目标）；
- 对无标签增强样本的交叉熵损失；
- 自监督形式的旋转损失。

![](https://pic.imgdb.cn/item/63bf7335be43e0d30e424869.jpg)