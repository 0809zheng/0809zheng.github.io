---
layout: post
title: 'Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation'
date: 2022-11-30
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6687ac8dd9c307b7e9bb2c84.png'
tags: 论文阅读
---

> 特征蒸馏使对比学习在微调时击败了掩码图像建模.

- paper：[Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation](https://arxiv.org/abs/2205.14141)

本文提出了特征蒸馏(**Feature Distillation, FD**)方法，可以处理任意预训练方法获得的特征，已经学习到的特征会再被蒸馏成为全新的特征，可以极大地提升自监督学习模型的微调性能。

对于任意基于对比的自监督预训练模型，**FD**使用特征作为蒸馏的目标。在所有的训练目标里面，使用全部特征图得到的结果是最好的；与使用其他简化的特征作为蒸馏目标的方法相比，对完整特征图的蒸馏可以保持教师模型中涉及的更多信息，这会得到更好的性能。

![](https://pic.imgdb.cn/item/6687bf64d9c307b7e9dd7a47.png)

为了使得教师和学生模型的特征相当，作者对每张输入图像使用了相同的数据增强策略；为了使得特征的维度得到匹配，作者额外使用了 1×1 卷积$g(\cdot)$将学生模型的特征映射成和教师模型相同的大小。

作者提出了一些非常有用的设计，包括白化蒸馏目标 (**whitened distillation targets**)，共享相对位置编码 (**shared relative position bias**)，以及非对称的 **Drop Path** 率 (**asymmetric drop path rates**)。在以上这些精心设计的方法之下，基于对比的自监督预训练方法的微调性能达到与掩码图像建模方法相当的表现。

白化蒸馏目标是指对要蒸馏的教师特征进行白化处理，通过降低每个特征之间的关联性并统一每个特征的方差，减少冗余信息。白化蒸馏目标是一个不含 **scaling** 参数和 **bias** 参数的 **Layer Normalization**。

在蒸馏中作者使用了 **smooth L1 loss** 来优化学生和教师模型的特征：

$$
L(s,t) = \begin{cases}
\frac{1}{2}(g(s)-whiten(t))^2/\beta, & |g(s)-whiten(t)| \leq \beta \\
(g(s)-whiten(t) - \frac{1}{2}\beta), & g(s)-whiten(t)| > \beta
\end{cases}
$$

在特征蒸馏的框架中，作者重新思考了学生网络中的位置编码方式。共享相对位置编码就是每一层共享一样的相对位置编码，实验结果表明这样做效果最好。

特征蒸馏的双分支框架允许教师和学生模型使用非对称的 **Drop Path** 率。作者发现，对学生和教师模型使用不同的 **Drop Path** 率对于学生模型学习到更好的表征是更加有利的。适度增加学生网络的 **Drop Path** 率是有益的，可能是由于过拟合的缓解。

作者对使用特征蒸馏前后模型每一层的每个注意力头的平均注意力距离和头之间的余弦相似度进行可视化。平均注意力距离可以部分反映出每个注意力头的感受野大小。对于特征蒸馏之前的特征而言，随着层数的加深，不同头的注意力距离范围变得更加接近了，最终都落在了一个较小的区间中。这表明不同的头学习到了非常相似的视觉线索，可能会浪费模型的容量。但是对于特征蒸馏之后的特征而言，所有的表征变得更加不同，注意力的距离分布更加平均，尤其是在更深的层中，这一现象变得更加明显。每一层注意力不同头之间的余弦相似度结果也呈现出了类似的特征。

![](https://pic.imgdb.cn/item/6687c15bd9c307b7e9e0d16a.png)

作者对使用特征蒸馏前后的平均注意力图进行可视化。注意力模式呈现出了2个明显的特征，即对角线以及列状。对角线模式对应的是某些固定相对位置的图像 **Patch** 之间的关系，而列状模式则表示某些位置的图像 **Patch** 对其他所有位置的影响。从图中可以看出，特征蒸馏后的平均注意力图具有更多的对角模式，这意味着特征蒸馏之后的模型更多地依赖于相对位置关系对图像 **Patch** 进行视觉建模。这表明模型具有更好的平移不变性，而平移不变性通常是各种视觉任务的有利属性。

![](https://pic.imgdb.cn/item/6687c1cdd9c307b7e9e1a629.png)

注意到学生网络中包含了共享相对位置编码 (**RPB**)，为了研究其效果，作者还尝试在学生体系结构中使用绝对位置编码 (**APE**)，其注意力映射如图所示。即使是用了 **APE**，特征蒸馏得到的表征依然呈对角线形状，这表明更多的对角线模式主要是由特征蒸馏算法本身，而不是由位置编码的方式造成的。

![](https://pic.imgdb.cn/item/6687c1ffd9c307b7e9e2031a.png)

作者可对不同模型的损失/精度**landscape**进行可视化。在该可视化方法中，模型的训练权重被一系列不同程度的高斯噪声干扰，为了考虑不同模型的权重幅值变化的影响，每个噪声级在定义时依据每个滤波器的$l_2$范数。特征蒸馏后的表征比蒸馏前的表征更加平坦，这与其从结果上更好的微调性能是一致的。

![](https://pic.imgdb.cn/item/6687c266d9c307b7e9e2bc63.png)

作者还对掩码图像建模 (**MIM**) 方法的平均注意力距离和损失**landscape**进行可视化。使用 **MAE** 预训练得到的不同的注意力头之间的平均注意力距离更大，损失**landscape**相对平坦。结果表明，特征蒸馏后处理带来的良好的微调性能与掩码图像建模方法在功能上有一定的重叠。

![](https://pic.imgdb.cn/item/6687c2b4d9c307b7e9e34225.png)