---
layout: post
title: 'RMPE: Regional Multi-person Pose Estimation'
date: 2021-04-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64a7644e1ddac507cc9416e0.jpg'
tags: 论文阅读
---

> RMPE：基于区域的多人姿态估计.

- paper：[RMPE: Regional Multi-person Pose Estimation](https://arxiv.org/abs/1612.00137)

本文提出了一种**Top-Down**的多人姿态估计方法，即先检测人，再识别人体姿态。对于自顶向下的方法，主要问题在于人体检测**位置不准确**和**检测冗余**，即使在检测任务中是正确的，提取的**proposal**也不一定适用于单人的姿态估计方法；同时冗余的检测框也使得单人的姿态被重复估计。本文致力于解决**imperfect proposal**问题，通过调整，使得裁剪的目标区域能够被单人姿态估计方法很好的识别，从而克服检测带来的定位误差。

![](https://pic.imgdb.cn/item/64a7668d1ddac507cc994602.jpg)

本文提出了**区域多人姿态估计（Region Multi-Person Pose Estimation，RMPE）**框架，包含三部分：
1. **Symmetric Spatial Transformer Network（SSTN）**：对称空间变换网络，用于在不准确的人体检测框中提取准确的单人区域；
2. **Parametric Pose Non-Maximum-Suppression（p-Pose NMS）**：参数化姿态的非最大值抑制，用于解决人体检测框冗余问题；
3. **Pose-Guided Proposals Generator（PGPG）**：姿态引导区域框生成器，用于数据增强。

本文使用**SSD**进行人体检测，使用**Stacked Hourglass**进行单人姿态估计。整个过程分为三步：
1. 用**SSD**检测人，获得**human proposal**；将**human proposal**在长宽方向上延长$20\%$，以确保可以把人完整的框起来。经过验证，这样确实可以把大部分的人整个框起来。
2. 将**proposal**输入到两个并行的分支里面，上面的分支是**STN+SPPE+SDTN**的结构，即**Spatial Transformer Networks + Single Person Pose Estimation + Spatial de- Transformer Networks**。**STN**接收的是**human proposal**，将延伸过的图像进行仿射变换，可以生成一个比较精确的、适合作为**SPPE**输入；**SPPE**是一个单人人体姿态估计器；**SDTN**把**SPPE**的输出经过与前边相反的**STN**变换，将坐标变换回原来的坐标系，产生**pose proposal**。下面并行的分支充当额外的正则化矫正器。
3. 对**pose proposal**做**Pose NMS**（非最大值抑制），用来消除冗余的**pose proposal**。

![](https://pic.imgdb.cn/item/64a7651d1ddac507cc964064.jpg)

### ⚪ Symmetric STN

**STN**相当于在传统的一层卷积中间引入矩阵的仿射变换，可以使得传统的卷积带有了裁剪、平移、缩放、旋转等特性；目标是可以减少**CNN**的训练数据量，以及减少对**data argument**的依赖，让**CNN**自己学会数据的形状变换，将输入图像做任意空间变换。

![](https://pic.imgdb.cn/item/64a76c6c1ddac507cca4083f.jpg)

### ⚪ Parallel SPPE

并行的**SPPE**作为正则化作用，用来进一步加强**STN**提取优质的**human proposal**。这一支的**label**设置为**single person pose**。训练时使用**2**条支路输出的总误差来训练网络，**parallel SPPE**所有层参数在训练阶段是固定的，**Parallel SPPE**分支和真实姿态的标注进行比较，反向传播中心位置的姿态误差给**STN**模块。如果**STN**的姿态不是中心定位，**parallel SPPE**反向传播较大的误差。通过反向传播的方式帮助**STN**聚焦正确的区域，实现提取高质量人体区域。测试阶将下面的**Parallel SPPE**丢掉，只使用**Symmetric STN**进行前向传播。

### ⚪ Parametric Pose NMS

首先选择置信度最高的**pose**作为参考，靠近它的**pose**通过淘汰标准来消除。对于剩下的**pose**重复上述过程，直到消除冗余姿势，并且仅返回唯一的**pose**。

### ⚪ Pose-guided Proposals Generator
在训练过程中通过数据增强增加**proposal**的数量，每一张图片的$K$个人会产生$K$个**bbox**，根据**ground truth**的**proposals**，生成和其分布相同的多个**proposals**一起训练。


