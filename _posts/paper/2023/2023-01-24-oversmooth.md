---
layout: post
title: 'Improve Vision Transformers Training by Suppressing Over-smoothing'
date: 2023-01-24
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/643bc7a30d2dde5777c4843d.jpg'
tags: 论文阅读
---

> 通过抑制过度平滑改进视觉Transformer.

- paper：[Improve Vision Transformers Training by Suppressing Over-smoothing](https://arxiv.org/abs/2104.12753)

直接在视觉任务上训练视觉**Transformer**模型容易产生不稳定和次优的结果。之前有的工作通过加入卷积以更有效地提取**low-level**的特征，从而加速和稳定视觉**Transformer**模型的训练，这相当于改变了**Transformer**的结构。而本文所做的工作是在不改变原有**Transformer**模型任何结构的前提下通过改变训练方式来稳定训练。

本文作者把视觉**Transformer**模型训练不稳定的原因归结为：过平滑问题 (**Over-smoothing problem**)，即：不同**token**之间的相似性随着模型的加深而增加，深层**self-attention**层趋向于把输入图片的不同**patch**映射成差不多的**latent representation**。

图片划分成**patch**之后，每个**patch**会经过一个**Transformer Block**层得到**token**，不同的**pacth**对应着不同的**token**。但是，当模型越来越深时，不同的**token**之间的相似性越来越大，大到甚至无法有效地进行区分了。作者称这种现象为过平滑问题。

作者首先定义了一种不同**patch representation**之间的**Layer-wise cosine similarity**：假设输入图片$x$及其对应的**patch representation**为$h=(h^{cls},h_1,...,h_n)$，定义**smoothness**为它的**patch representation**中所有**token**之间的**cosine similarity**：

$$
\cos Sim(h) = \frac{1}{n(n-1)}\sum_{i \neq j} \frac{h_i^Th_j}{\left\|h_i \right\| \left\|h_j \right\|}
$$

如下图a所示为所有层的**cosine similarity**值，可以观察到随着层数的加深，**DeiT**模型的**cosine similarity**越来越大。一个**24**层的**DeiT-Base**模型的最后一层的不同**token**之间的**pairwise**相似度达到了**0.9**，表明所学到的**patch representation**之间的高度相关性和重复性。

![](https://pic.imgdb.cn/item/643bc9a50d2dde5777c6b332.jpg)

此外，作者还通过另外一个指标来描述这种相似性：**Layer-wise standard deviation**。通常希望每一层的某一个**patch**，它**attend to**其他所有**patch**的程度是不一样的，这样才能捕获到图片中有意义的区域的信息。**Layer-wise standard deviation**就是为了衡量这种能力的大小。假设某个**patch** $h_i$，其**softmax attention score**为$S(h_i)\in R^n$，这是一个$n$维的向量，将其求标准差来表示相似度的大小，相似度越小，标准差就越大。每一个**layer**能求出来$n$个标准差。将它们取平均值来代表这一层的不同**token**之间的相似度的大小，如上图b所示。

从结果可以看出，传统的**Transformer**模型每一层的不同**token**之间的**standard deviation**都很小，表示相似度都很大，也就造成了**over-smoothing**的问题。但是在**NLP**模型中，每个**patch**都有其对应的**label**，所以不同的**token**之间的相似性不易变得很大，也就不容易出现**over-smoothing**的问题。


为了解决**over-smoothing**问题，作者提出了一系列方法：

### ⚪ 添加相似度罚项

最直接的避免每一层的不同**token**之间的相似度过大的办法是添加惩罚项，基于此作者使用了一个新的**loss**函数，对于最后一层的**patch representation** $h=(h^{cls},h_1,...,h_n)$来讲：

$$
\mathcal{l}_{cos}= \frac{1}{n(n-1)}\sum_{i \neq j} \frac{h_i^Th_j}{\left\|h_i \right\| \left\|h_j \right\|}
$$

来最小化相似性。这种做法可以看做是增加最后一层的**patch representation**的表达能力。

### ⚪ Patch Contrastive Loss

作者认为浅层的**patch**应该与深层对应的那个的**patch**的值比较接近，这样一来**patch**之间的区分度就能够在整个网络中维持住。对于一个给定的图片，假设$e$是第$1$层的**patch representations**，而$h$是最后$1$层的**patch representations**，作者使用对比学习的损失：

$$
\mathcal{l}_{cons} = -\frac{1}{n} \sum_{i=1}^n \log \frac{\exp(e_i^Th_i)}{\exp(e_i^Th_i)+\exp(e_i^T(\sum_{j=1}^nh_i/n))}
$$

来迫使浅层和深层相对应的**patch representations**接近，而与其他的**patch representations**疏远。

![](https://pic.imgdb.cn/item/643bcc650d2dde5777c9a2b7.jpg)

### ⚪ Patch Mixing Loss

作者希望每个**patch**都能有监督信息，这样一来就可以使用**CutMix**这个数据增强策略。具体而言是给每个**patch**的向量通过一个共享的分类器得到分类的结果，这些**patch**都会有对应的监督信息，所以模型能够学习到更具有信息量的**patch representation**。

**Patch Mixing Loss**定义为：

$$
\mathcal{l}_{token} = \frac{1}{n} \sum_{i=1}^n \mathcal{l}_{ce}(g(h_i),y_i)
$$

