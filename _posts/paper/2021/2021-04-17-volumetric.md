---
layout: post
title: 'Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose'
date: 2021-04-17
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64a65f2e1ddac507cc96b4f0.jpg'
tags: 论文阅读
---

> 单图像人体姿态的由粗到细的体素预测.

- paper：[Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose](https://arxiv.org/abs/1611.07828)

使用单张图像进行人体姿态估计的方法通常采用两步解决方案，包括用于 **2D** 关节估计的卷积网络和用于恢复 **3D** 姿势的后续优化步骤。本文提出了一种端到端的单图像**3D**姿态估计方法，从**2D**图像中直接得到体素（**Volumetric**）表示，而不是直接回归关节点的坐标，并取最大值的位置作为每个关节点的输出。

![](https://pic.imgdb.cn/item/64a661d61ddac507cc9c9a5e.jpg)

对于每个关节，创建一个大小为 $w × h × d$ 的体积。让 $p_n(i,j,k)$ 表示关节 $n$ 在体素 $(i, j, k)$ 中的预测可能性。为了训练这个网络，还以体积形式提供监督。每个关节的目标是一个以 **3D** 网格中关节位置$x_{gt}^n = (x, y, z)$为中心的 **3D** 高斯体积:

$$
G_{i,j,k}(x_{gt}^n) = \frac{1}{2\pi \sigma^2} e^{-\frac{(x-i)^2+(y-j)^2+(z-k)^2}{2\sigma^2}}
$$

损失函数构建为**L2**损失：

$$
L = \sum_n \sum_{i,j,k} ||G_{i,j,k}(x_{gt}^n) -p_n(i,j,k)||^2 
$$

体积表示的一个主要优点是它将直接 **3D** 坐标回归的高度非线性问题转换为离散空间中更易于管理的预测形式。在这种情况下，预测不一定针对每个关节的唯一位置，而是为每个体素提供置信度估计。这使得网络更容易学习目标映射。

就网络架构而言，体积表示的一个重要好处是它可以使用全卷积网络进行预测。网络结构采用[<font color=blue>Stacked Hourglass Network</font>](https://0809zheng.github.io/2021/04/03/hourglass.html)，网络的输出是四维的，即$(w×h×d×N)$，但实际上将其组织在通道中，因此输出是三维的，即 $w × h × dN$。在每个 **3D** 网格中具有最大响应的体素被选为关节的 **3D** 位置。

![](https://pic.imgdb.cn/item/64a661f71ddac507cc9cf4da.jpg)

受 **2D** 姿势背景下迭代细化的成功启发，网络采用了逐步细化方案。然而三维表示的维度很大，对于具有 $16$ 个关节的 $64 × 64 × 64$ 的三维体素需要估计超过 **400** 万个体素的可能性。为了处理这种维度灾难，使用从粗到精的预测方案，即使用较低分辨率对于$z$维进行监督。准确地说，在网络的每个阶段使用每个关节大小为 $64 × 64 × d$ 的目标，其中 $d$ 通常从集合 $$\{1, 2, 4, 8, 16, 32, 64\}$$ 中取值。

实验结果表明，基于体素的方法比基于回归的方法具有优势。

![](https://pic.imgdb.cn/item/64a663141ddac507cca002ee.jpg)