---
layout: post
title: 'SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos'
date: 2021-10-18
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6790b51ad0e0a243d4f69241.png'
tags: 论文阅读
---

> SmoothNet：一种用于视频中人体姿态细化的即插即用网络.

- paper：[SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos](https://arxiv.org/abs/2207.10387)

## TL; DR

本文提出了一种名为 **SmoothNet** 的即插即用网络，用于改善视频中人体姿态估计的抖动问题。**SmoothNet** 通过学习人体运动的自然平滑特性，利用长时序关系对每个关节进行建模，显著降低了姿态估计中的抖动误差，同时提高了关键帧的估计精度。该方法在多个数据集和多种姿态估计器上进行了验证，表现出色，尤其是在处理罕见姿态和遮挡情况下的长时序抖动时效果显著。

## 1. 背景介绍

人体姿态估计在运动分析、人机交互等领域有着广泛的应用。尽管深度学习技术取得了显著进展，但在处理罕见姿态或遮挡情况时，现有的姿态估计器仍然存在较大的误差，导致视频中出现明显的抖动。这些抖动不仅影响视觉效果，还可能误导后续的分析任务。

作者根据持续时间将抖动问题分成了两类：瞬间抖动和长期抖动。由于抖动导致的误差分成相邻帧之间的抖动误差$J$和**ground truth**跟平滑后的姿态之间的偏离误差$S$。本文主要关注如何减小瞬间抖动的抖动误差$J$。

![](https://pic1.imgdb.cn/item/679c8b1ed0e0a243d4f8c110.png)

传统的解决方案包括使用时空模型联合优化每帧的精度和时序平滑性，但这些方法在处理长时序抖动时效果不佳。本文提出了一种新的方法——**SmoothNet**，它专注于建模人体运动的时序平滑性，通过学习长时序关系来缓解姿态估计中的抖动问题。

![](https://pic1.imgdb.cn/item/679c8e00d0e0a243d4f8c13d.png)

## 2. 方法介绍

**SmoothNet** 是一个专门用于姿态估计抖动缓解的时序细化网络。它通过学习人体运动的自然平滑特性，利用长时序关系对每个关节进行建模，从而显著降低姿态估计中的抖动误差。**SmoothNet** 的核心思想是将姿态估计的抖动问题转化为一个时序建模问题，通过优化时序平滑性来间接提升姿态估计的精度。

基础 **SmoothNet** 采用简单的全连接网络（**FCN**）作为骨干网络，通过多个残差连接的块来捕捉时序关系。具体来说，**SmoothNet** 的每一层计算可以表示为：

$$
\hat{Y}_{i, t}^{l+1}=\sigma(\sum_{t=1}^{T} w_{t}^l \cdot Y_{i, t}^l+b^l)
$$

其中，$\hat{Y}_{i, t}^{l+1}$是第 $l$ 层第 $i$ 个关节在第 $t$ 帧的输入，$w_t^l,b^l$是可学习的权重和偏置，$σ$ 是非线性激活函数（默认使用 **LeakyReLU**）。**SmoothNet** 通过滑动窗口方案处理整个输入序列，窗口大小为 $T$，步长为 $s$。

![](https://pic1.imgdb.cn/item/679c8eded0e0a243d4f8c15a.png)

为了进一步提升性能，**SmoothNet** 引入了运动信息，显式地建模速度和加速度。给定输入$\hat{Y}$，首先计算每个关节的速度和加速度：

$$
\hat{V}_{i, t} = \hat{Y}_{i, t} - \hat{Y}_{i, t-1} \\
\hat{A}_{i, t} = \hat{V}_{i, t} - \hat{V}_{i, t-1}
$$

 
然后，**SmoothNet** 通过三个分支分别对位置、速度和加速度进行细化，最后通过一个融合层将三个分支的输出进行线性融合，得到最终的细化结果。

![](https://pic1.imgdb.cn/item/679c9001d0e0a243d4f8c19c.png)

**SmoothNet** 的目标是最小化位置误差和加速度误差。具体损失函数如下：

$$
L_{pose} = \frac{1}{TC} \sum_{t=0}^T \sum_{i=0}^C | \hat{G}^i_t - Y^i_t | \\
L_{acc} = \frac{1}{(T-2)C} \sum_{t=0}^T \sum_{i=0}^C | {\hat{G}^i_t}^{\prime \prime} - A^i_t |
$$

其中，${\hat{G}^i_t}^{\prime \prime}$是从预测姿态$\hat{G}^i_t$计算得到的加速度，$A^i_t$是真实加速度。最终的损失函数为$L_{pose} + L_{acc}$。


## 3. 实验分析

**SmoothNet** 与三种常用滤波器（**One-Euro、Savitzky-Golay 和 Gaussian1d**）进行了比较。实验结果表明，**SmoothNet** 在减少抖动方面表现最佳，尤其是在处理长时序抖动时效果显著。例如，在 **AIST++** 数据集上，**SmoothNet** 将加速度误差从 **31.64 mm/frame²** 降低到 **4.15 mm/frame²**，同时将 **MPJPE** 从 **106.90 mm** 降低到 **97.47 mm**。

![](https://pic1.imgdb.cn/item/679c91e4d0e0a243d4f8c1ce.png)

**SmoothNet** 作为一种即插即用网络，可以与任何现有的姿态估计器结合使用。实验结果表明，**SmoothNet** 在多个数据集和多种姿态估计器上均显著降低了抖动误差，同时提高了估计精度。例如，在 **Human3.6M** 数据集上，**SmoothNet** 将 **HRNet** 的加速度误差从 **1.01 mm/frame²** 降低到 **0.13 mm/frame²**，同时将 **MPJPE** 从 **4.59 mm** 降低到 **4.54 mm**。

![](https://pic1.imgdb.cn/item/679c9225d0e0a243d4f8c1d1.png)

**SmoothNet** 与现有的时空模型进行了比较，结果表明，**SmoothNet** 在处理长时序抖动时效果更好。例如，在 **3DPW** 数据集上，**SmoothNet** 将 **VIBE** 的加速度误差从 **23.2 mm/frame²** 降低到 **6.0 mm/frame²**，同时将 **MPJPE** 从 **83.0 mm** 降低到 **81.5 mm**。

![](https://pic1.imgdb.cn/item/679c9254d0e0a243d4f8c1d6.png)

此外，作者进行了一系列消融实验：
- 不同时序模型的比较：**SmoothNet** 与 **TCN** 和 **Transformer** 等时序模型进行了比较。实验结果表明，**SmoothNet** 在减少抖动方面表现更好，尤其是在处理长时序抖动时。例如，在 **AIST++** 数据集上，**SmoothNet** 的加速度误差为 **4.15 mm/frame²**，而 **TCN** 的加速度误差为 **14.46 mm/frame²**，**Transformer** 的加速度误差为 **6.15 mm/frame²**。![](https://pic1.imgdb.cn/item/679c931cd0e0a243d4f8c1e2.png)
- 不同窗口大小的影响：**SmoothNet** 的窗口大小 $W$ 对平滑效果有显著影响。实验结果表明，窗口大小越大，加速度误差越小，但位置误差在窗口大小超过 $64$ 帧后开始略有增加。因此，$64$ 帧是一个平衡平滑效果和精度的合适窗口大小。![](https://pic1.imgdb.cn/item/679c932bd0e0a243d4f8c1e3.png)
