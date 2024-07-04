---
layout: post
title: 'VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting'
date: 2023-12-27
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/66824a94d9c307b7e96d54a6.png'
tags: 论文阅读
---

> VLCounter：零样本目标计数的文本感知视觉表示.

- paper：[VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting](https://arxiv.org/abs/2312.16580)

本文提出了一种简化的零样本目标计数框架，即视觉语言基线(**Visual-Language Baseline, VLBase**)，它由**CLIP**编码器和计数解码器组成。利用**CLIP**的嵌入空间，实现语义和补丁嵌入的隐式关联，从而对目标物体进行定位。在**VLBase**的基础上作者进一步提出了**VLCounter**，通过引入语义条件提示微调、可学习仿射变换与语义感知跳跃连接，实现对目标物体的计数。

![](https://pic.imgdb.cn/item/66824eacd9c307b7e9739c62.png)

## 1. 视觉语言基线 Visual-Language Baseline

给定输入查询图像和类别名, **VLBase**分别使用**CLIP**视觉编码器 $V(·)$和文本编码器 $T(·)$获得图像嵌入$V$和文本语义嵌入$T$。通过计算$T$与$V$之间的余弦相似度，得到相似度图$S$。


$$
S_{i j}(\mathcal{V},\mathcal{T})=\frac{v_{i j}\mathcal{T}^T}{||v_{i j}||\cdot ||\mathcal{T}||}
$$

**CLIP**编码的文本嵌入和图像嵌入之间的相似度图可以充分表明图像和文本嵌入之间的语义相似程度。这种相似度图是解码器定位目标对象的一个很好的线索。因此基于**CNN**的计数解码器利用图像嵌入$V$和相似度图$S$的特征来预测密度图。最后，通过对密度图中的所有值求和得出目标计数预测。

## 2. 语义条件提示微调 Semantic-conditioned Prompt Tuning

为了在不牺牲其泛化能力的情况下赋予**CLIP**图像编码器任务特异性，作者引入了语义条件提示调优(**SPT**)，它利用语义信息和可学习的**token**来帮助图像编码器提取目标语义突出显示的视觉特征。

**SPT**对每个编码层都引入新的可学习**token**。第$l$层的可学习**token**定义为$P^l = [p^l_1, p^l_2,...,p^l_M]$，然后将这些**token**与线性投影文本嵌入$T$相加，以生成语义条件提示符：

$$
\hat{\mathcal{P}}^l=[p_1^l+\hat{\mathcal{T}},p_2^l+\hat{\mathcal{T}},p_M^l+\hat{\mathcal{T}}]
$$

因此图像编码器第$l$层的**patch**嵌入过程可表示为:

$$
[[cls],\_,\mathcal{V}^{l+1}]=Layer_{\mathrm{enc}}^{l}([[cls],\hat{\mathcal{P}}^{l},\mathcal{V}^{l}])
$$

![](https://pic.imgdb.cn/item/66825166d9c307b7e97803a5.png)

## 3. 可学习仿射变换 Learnable Affine Transformation

相似度图$S$表示目标类的相应区域被突出显示的视觉表示。然而，由于目标计数的本质是发现目标的中心点，而不是包含整个目标区域，因此相似图中包含的信息与训练过程中需要反向传播的损失之间可能产生差异。作者提出了可学习仿射变换矩阵来促进相似度图$S$转换到计数图$\hat{S}$：联

$$
\hat{S}=W\otimes S+B
$$

使用等级感知的对比损失来优化计数图$\hat{S}$，以学习目标计数的适当激活程度。为了设计等级感知对比损失的分层指导，首先将真值密度图归一化，以映射在0和1之间。然后使用不同的阈值迭代批处理K次，以准备正集和负集：如果真值密度图中对应的**patch**的值高于阈值，则被收集为正值；否则为负值。形式上，正集与负集的秩对比损失表示为:

$$
\mathcal{L}_{\mathrm{rank}}=-\sum_{k=1}^K\log\frac{\sum_{\hat{S}_i\in\hat{S}_r^{\mathrm{pos}}}\exp(\hat{S}_i/\tau)}{\sum_{\hat{S}_j\in(\hat{S}_r^{\mathrm{pos}}\cup\hat{S}_r^{\mathrm{neg}})}\exp(\hat{S}_j/\tau)}
$$

## 4. 语义感知跳跃连接 Semantic-aware Skip Connection

模型在推理过程中可能遇到看不见的类，因此在保持泛化能力的同时训练一个为目标计数量身定制的解码器是很重要的。作者采用跳跃连接，将编码器的中间特征合并到解码器中的对应部分。

具体地，编码器输出的视觉特征$V$在空间上进行连接和投影，然后乘以计数图$\hat{S}$来强调目标相关的**token**。最后将这些**patch**特征添加到解码器的对应层特征中：

$$
\mathcal{F}^k=Layer_{\mathrm{dec}}^k(\mathcal{F}^{k-1}+\phi_{\mathrm{proj}}^k(\mathcal{V}^l)\otimes\hat{S})
$$

![](https://pic.imgdb.cn/item/66825406d9c307b7e97cab42.png)

## 5. 实验分析

结果表明，尽管**VLBase**的设计很简单，但性能可以与两阶段方法相媲美。**VLcounter**明显超过了其他基线。单阶段方法(**VLBase**和**VLCounter**)只需要计算目标的时间，因此它们的推理速度比两阶段方法快得多，需要学习的参数更少，训练时间更短。

![](https://pic.imgdb.cn/item/66825629d9c307b7e9808b5b.png)

下图比较了**VLBase**和**VLCounter**的相似度图和预测密度图。通过传递语义条件和对相似度图进行微调，相似度图保留了更紧凑的显著区域；背景中的激活被抑制，目标区域被明确定位。然后通过将丰富语义的多层次表示与解码器中的这些相似图聚合在一起，密度图能够获得更高质量的结果，特别是对于密集填充的图像。

![](https://pic.imgdb.cn/item/668256c7d9c307b7e9816981.png)