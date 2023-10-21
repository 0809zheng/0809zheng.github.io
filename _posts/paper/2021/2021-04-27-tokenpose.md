---
layout: post
title: 'TokenPose: Learning Keypoint Tokens for Human Pose Estimation'
date: 2021-04-27
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64cf1cec1ddac507cca6b948.jpg'
tags: 论文阅读
---

> TokenPose：学习人体姿态估计的关键点token.

- paper：[TokenPose: Learning Keypoint Tokens for Human Pose Estimation](https://arxiv.org/abs/2104.03516)

人体姿态估计任务主要依赖两方面的信息：视觉信息（图像纹理信息）和解剖学的约束信息（关节之间的连接关系）。对于**CNN**来说，其优势在于对于图像纹理信息的特征提取能力极强，能学习到高质量的视觉表征，但在约束信息的学习上则有所不足。本文利用了**Transformer**中多头注意力机制的特点，能学到位置关系上的约束，以及不同关键点之间的关联性，并且极大地减少了模型的参数量和计算量。

![](https://pic.imgdb.cn/item/64cf1f171ddac507ccab97c2.jpg)

**Token**在**NLP**中是指每个词或字符用一个特征向量来表示。在本工作中设置了两种**Token**类型，一种是**visual token**，是将特征图按**patch**拆分后拉成的一维特征向量；另一种是**keypoint token**，专门学习每一个关键点的特征表示。将两种Token特征一起放入**Transformer**，于是模型可以同时学习到图像纹理信息和关键点连接的约束信息。

**TokenPose**先通过一个基于**CNN**的骨干网络来提取特征图，将特征图拆分为**patch**后拉平为一维向量，经过一个线性函数(全连接层)投影到**d**维空间，这些向量称为**visual tokens**，负责图片纹理信息的学习。考虑到姿态估计任务对于位置信息是高度敏感的，因此还要给**token**加上**2d**位置编码。然后通过随机初始化一些可学习的**d**维向量来作为**keypoint tokens**，每个**token**对应一个**keypoint**。将两种**token**一起送入**transformer**进行学习，并将输出的**keypoint tokens**通过一个**MLP**映射到**HxW**维，以此来预测**heatmap**。

![](https://pic.imgdb.cn/item/64cf201d1ddac507ccadc8fc.jpg)

在实验部分作者通过对注意力进行可视化的方式，验证了关键点约束信息的学习情况。对不同**transformer**层对应的每个关键点注意力进行了可视化，可以清晰地看到一个逐渐确定到对应关节的过程：

![](https://pic.imgdb.cn/item/64cf20661ddac507ccae6b33.jpg)

对学习完成的**keypoint tokens**计算内积相似度也可以发现，相邻关节点和左右对称位置的关节的**token**相似度是最高的：

![](https://pic.imgdb.cn/item/64cf21a41ddac507ccb145a9.jpg)
