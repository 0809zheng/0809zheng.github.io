---
layout: post
title: 'Few-shot Object Counting with Similarity-Aware Feature Enhancement'
date: 2023-05-05
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/667cd665d9c307b7e9db915f.png'
tags: 论文阅读
---

> 通过相似度感知特征增强实现少样本目标计数.

- paper：[Few-shot Object Counting with Similarity-Aware Feature Enhancement](https://arxiv.org/abs/2201.08959)

小样本计数(**Few-Shot Counting, FSC**)是指给定一张或几张**support**描述计数的物体类别，并给定一张待计数的图像**query**，**FSC**希望计算出该类别的物体在**query**中出现的个数。除了在训练集中出现的类别 (称为**base classes**)，在测试阶段，**FSC**还需要处理完全没有见过的类别 (称为**novel classes**)。

![](https://pic.imgdb.cn/item/667cd747d9c307b7e9dcee92.png)

小样本计数的**baselines**主要有两类。第一类是基于特征的方法(下图a)，将**support**和**query**的特征连接后训练一个回归头部，计算密度图。第二类是基于相似度的方法(下图b)，将**support**和**query**的特征计算一个距离度量，得到一张相似度图，之后从相似度图回归密度图。前者对于语义信息的保持更好，后者对于**support**和**query**之间的关系感知更好。

![](https://pic.imgdb.cn/item/667cd758d9c307b7e9dd04bf.png)

本文提出一种**Similarity-Aware Feature Enhancement**(**SAFECount**)模块。首先采用**Similarity Comparison Module (SCM)** 对比**support**和**query**的特征并生成相似度图；然后采用**Feature Enhancement Module (FEM)**将相似度图作为引导，用**support**的特征来提升**query**的特征；最后从提升过的特征中回归密度图。**SAFECount**既保持了很好的语义信息，又对于**support**和**query**之间的关系有良好的感知。

![](https://pic.imgdb.cn/item/667cd778d9c307b7e9dd3107.png)

在**Similarity Comparison Module (SCM)**中，首先将**support**和**query**的特征利用共享的卷积层投影到一个对比空间。之后将**support**作为卷积核，在**query**上滑动，计算得到一张得分图。最后通过**exemplar norm**（从**support**的维度进行**norm**）和**spatial norm**（从空间的维度进行**norm**）对得分图的值进行**norm**，得到一张相似度图。

在**Feature Enhancement Module (FEM)**中，首先将相似度图作为权重，对**support**进行加权。具体的，对**support**翻转作为卷积核，对相似度图进行卷积，得到一张相似度加权的特征。之后将相似度加权的特征与**query**的特征进行融合，得到提升过的特征。

提升过的特征与**query**的特征具有相同的形状。因此提升过的**feature**可以再次作为**query**的特征输入**SAFECount Block**。实验结果证明，只需要一个**block**即可达到非常高的精度，增加**block**可以进一步提升精度。最后使用一个回归头部用于从提升过的特征中预测得到密度图。损失函数为预测密度图与**ground-truth**之间的**MSE loss**。

**SAFECount**在小样本计数的数据集**FSC-147**的**Test Set**上显著高于其他模型。可视化结果表明**SAFECount**不仅可以提供精准的计数结果，还可以提准精准的定位信息。

![](https://pic.imgdb.cn/item/667cd7c9d9c307b7e9dda04f.png)

对相似度图的可视化结果表明，**SAFECount**可以在成堆且相互遮挡的物体中，得到非常清晰的物体边界信息，这有助于区分这些成堆且相互遮挡的物体，所以可以得到更加准确的计数结果。

![](https://pic.imgdb.cn/item/667cd790d9c307b7e9dd534d.png)