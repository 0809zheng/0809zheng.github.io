---
layout: post
title: 'When BERT Plays the Lottery, All Tickets Are Winning'
date: 2020-07-11
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f093ab414195aa5942435fc.jpg'
tags: 论文阅读
---

> 讨论BERT剪枝中的“彩票假设”.

- paper：When BERT Plays the Lottery, All Tickets Are Winning
- arXiv：[link](https://arxiv.org/abs/2005.00561)

作者讨论了**BERT**中的**lottery ticket hypothesis**，观察到：
- 通过对**BERT**剪枝得到表现较好的子模型；
- 从**BERT**中采样尺寸相似的子模型表现较差；但经过微调后也能得到较好的表现。

# BERT的剪枝
**BERT**是由多头自注意力模块和多层感知机组成的预训练模型。作者使用了$12$层、每层$12$个自注意力头的模型。

对于第$l$层，**BERT**实现了如下操作：

$$ MHAtt^{(l)} = \sum_{h=1}^{N_h} {Att^{(l)}_{W_k^{(h,l)},W_q^{(h,l)},W_v^{(h,l)},W_o^{(h,l)}}(x)} $$

$$ MLP^{(l)}_{out}(z) = MLP^{(l)}(z) + z $$

**BERT**在进行剪枝时，随机丢弃某一层中的某一个自注意力头或某一层的多层感知机，即引入取值为$$\{0,1\}$$的**mask**变量$ξ$和$\nu$，使得：

$$ MHAtt^{(l)} = \sum_{h=1}^{N_h} {ξ^{(h,l)}Att^{(l)}_{W_k^{(h,l)},W_q^{(h,l)},W_v^{(h,l)},W_o^{(h,l)}}(x)} $$

$$ MLP^{(l)}_{out}(z) = \nu^{(l)}MLP^{(l)}(z) + z $$

根据部件的重要性进行剪枝，其重要性由下式衡量：

$$ I_h^{(h,l)} = E_{x \text{~} X}\mid \frac{\partial L(x)}{\partial ξ^{(h,l)}} \mid $$

$$ I_{mlp}^{(l)} = E_{x \text{~} X}\mid \frac{\partial L(x)}{\partial \nu^{(l)}} \mid $$

# 实验结果
作者在**GLUE**的9个任务中进行实验：

![](https://pic.downk.cc/item/5f095fff14195aa5942eb941.jpg)

作者对每个任务设置了五次随机数种子，记录每个自注意力头或每层感知层在这9个任务中的平均存活次数及其方差；通过实验发现对于不同的任务，并不存在对这些任务都很重要的组件，大部分组件都在不同的任务中起到不同的作用：

![](https://pic.downk.cc/item/5f0961b614195aa5942f5442.jpg)

作者还实验探究了不同任务中同时存活的自注意力头（共144个）或感知层（共12个）个数。通过实验发现，不同任务中存活的部件相差比较大，证明这些部件对不同的任务起到不同的作用：

![](https://pic.downk.cc/item/5f09620b14195aa5942f7e10.jpg)

作者进一步对好的子网络和差的子网络进行微调，发现剪枝后较差的子网络通过微调也能达到不错的效果：

![](https://pic.downk.cc/item/5f0962dc14195aa5942fc9c4.jpg)
