---
layout: post
title: 'Synthesizer: Rethinking Self-Attention in Transformer Models'
date: 2020-07-14
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f0d9e5014195aa594e1ad35.jpg'
tags: 论文阅读
---

> Synthesizer：使用合成注意力的Transformer模型.

- paper：Synthesizer: Rethinking Self-Attention in Transformer Models
- arXiv：[link](https://arxiv.org/abs/1811.02486v1)

**Transformer**模型中使用了点积实现的多头自注意力机制，即对于输入$X$，分别使用**query**、**key**、**value**仿射变换得到$Q(X)$、$K(X)$、$G(X)$，再通过**query**和**key**的交互实现自注意力分布$B=K(X)Q(X)$，最后计算输出结果$Y=Softmax(B)G(X)$。

作者提出了一种**合成注意力（synthetic attention）**机制，即不引入仿射变换$Q(X)$和$K(X)$，而是直接对输入$X$进行变换得到自注意力分布$B=F(X)$，具体地对每一个时间步上的向量$X_i$使用一个两层的神经网络$F(X)=W(σ_R(W(X)+b))+b$，再计算输出结果$Y=Softmax(B)G(X)$。上述方法称为**Dense Synthesizer**。

该方法并没有显式地进行每一个时间步上向量之间的交互，而是用一个神经网络模拟这个交互过程，预测自注意力分布。

作者还提出了**Random Synthesizer**。即使用一个随机矩阵$R$来作为自注意力分布，直接计算输出结果$Y=Softmax(R)G(X)$。

![](https://pic.downk.cc/item/5f0d990e14195aa594e01fe2.jpg)

为了减小参数量，作者还在合成注意力中引入了矩阵分解的方法，这些方法的对比如下：

![](https://pic.downk.cc/item/5f0d9cdd14195aa594e13f60.jpg)

作者在机器翻译任务上进行试验，证明合成注意力机制可以取代点击自注意力机制，并且达到类似的效果。将两者结合使用时效果会更好：

![](https://pic.downk.cc/item/5f0d9de114195aa594e18460.jpg)