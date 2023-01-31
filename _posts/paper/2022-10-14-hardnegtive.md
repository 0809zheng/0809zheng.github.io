---
layout: post
title: 'Contrastive Learning with Hard Negative Samples'
date: 2022-10-14
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63d4e824face21e9eff3e3c9.jpg'
tags: 论文阅读
---

> 使用难例负样本进行对比学习.

- paper：[Contrastive Learning with Hard Negative Samples](https://arxiv.org/abs/2010.04592)

本文作者设计了一种在对比学习损失中进行难例负样本挖掘的方法。

![](https://pic.imgdb.cn/item/63d5dfe7face21e9efcd2951.jpg)

受[<font color=blue>Debiased Contrastive Loss</font>](https://0809zheng.github.io/2022/10/13/debiased.html)启发，由于样本的真实标签是未知的，因此负样本可能采样到假阴性样本。在构造对比损失时，对负样本项进行偏差修正：

$$ g(x,\{u_i\}_{i=1}^N,\{v_i\}_{i=1}^M) = \max(\frac{1}{\eta^-}(\frac{1}{N}\sum_{i=1}^N \exp(f(x)^Tf(u_i))-\frac{\eta^+}{M}\sum_{i=1}^M \exp(f(x)^Tf(v_i))),\exp(-1/\tau)) \\ \mathcal{L}_{unbiased} = \Bbb{E}_{x,\{u_i\}_{i=1}^N\text{~}p;x^+,\{v_i\}_{i=1}^M\text{~}p_x^+} [-\log \frac{\exp(f(x)^Tf(x^+))}{\exp(f(x)^Tf(x^+))+Ng(x,\{u_i\}_{i=1}^N,\{v_i\}_{i=1}^M)}] $$

为了把难例负样本嵌入到损失函数中，考虑对损失中的负样本对项$$\exp(f(x)^Tf(u_i))$$进行加权，权重正比于负样本与**anchor**样本的相似度，设置为：

$$ \frac{\beta \exp(f(x)^Tf(u_i))}{\sum_i \exp(f(x)^Tf(u_i))} $$

![](https://pic.imgdb.cn/item/63d5e057face21e9efcdf6dc.jpg)