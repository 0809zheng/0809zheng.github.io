---
layout: post
title: 'PanFlowNet: A Flow-Based Deep Network for Pan-sharpening'
date: 2024-10-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/673f14e1d29ded1a8c016cef.png'
tags: 论文阅读
---

> PanFlowNet：基于流的深度网络用于全色锐化.

- paper：[PanFlowNet: A Flow-Based Deep Network for Pan-sharpening](https://arxiv.org/abs/2305.07774)

## TL; DR

本文提出了一种名为**PanFlowNet**的基于流的深度网络，用于全色（**PAN**）与多光谱（**MS**）图像融合任务，即图像锐化。**PanFlowNet**能够学习**HRMS**图像的显式分布，并生成多样化的高分辨率多光谱图像。实验结果表明，**PanFlowNet**在视觉和量化评估上均优于其他最先进的方法。

## 1. 背景介绍

随着卫星传感器技术的快速发展，遥感图像在环境监测、分类和目标检测等领域得到了广泛应用。全色（**PAN**）图像具有高空间分辨率，而多光谱（**MS**）图像则包含丰富的光谱信息。如何将这两种互补信息有效融合，以生成同时具有高空间和高光谱分辨率的图像，是图像锐化任务的核心挑战。

传统的图像锐化方法主要依赖于数学模型来融合空间和光谱信息，但这些方法往往无法充分利用深度学习技术的强大非线性拟合能力。近年来，基于深度学习的图像锐化方法逐渐兴起，但这些方法大多学习一个确定性的映射，从**LRMS**和**PAN**图像中仅恢复出一个**HRMS**图像，忽略了**HRMS**图像的多样性。

针对上述问题，本文提出了一种基于流的深度网络**PanFlowNet**，用于全色图像锐化任务。该网络能够学习**HRMS**图像的显式分布，并生成多样化的高分辨率多光谱图像。

![](https://pic.imgdb.cn/item/673f15b9d29ded1a8c02029b.png)

## 2. 方法介绍

**PanFlowNet**是一个基于流的深度网络，由一系列可逆的条件仿射耦合块（**Conditional Affine Coupling Block，CACB**）构成。这些块通过一系列可逆变换，将基础分布映射到复杂的**HRMS**图像分布。

网络架构如图所示，输入包括**LRMS**图像、**PAN**图像以及来自给定高斯分布的噪声样本。通过堆叠多个**CACB**，网络能够学习到从输入到**HRMS**图像的复杂映射。

![](https://pic.imgdb.cn/item/673f15ead29ded1a8c02245d.png)

条件仿射耦合块**CACB**是**PanFlowNet**的核心组件，它实现了输入到输出的可逆变换。每个**CACB**接受前一层的输出和条件信息（如如**PAN**和**MS**图像）作为输入，并输出变换后的特征图。

具体来说，**CACB**通过仿射变换对输入特征图$h_n=[h_n^1,h_n^2]$进行缩放和平移，这种设计使得网络能够在保持可逆性的同时，灵活地捕捉输入之间的复杂关系：

$$
h_{n+1}^1 = h_n^1 \odot \exp(s_1(h_n^2))+t_1(h_n^2) \\
h_{n+1}^2 = h_n^2 \odot \exp(s_2(h_{n+1}^1))+t_2(h_{n+1}^1) \\
\downarrow \\
h_n^2 = (h_{n+1}^2-t_2(h_{n+1}^1)) / \exp(s_2(h_{n+1}^1)) \\
h_n^1 = (h_{n+1}^1-t_1(h_n^2)) / \exp(s_1(h_n^2))
$$

变换参数$s,t$是由条件信息和前一层的输出共同决定:

![](https://pic.imgdb.cn/item/673f17d4d29ded1a8c038f63.png)


## 3. 实验分析

为了验证**PanFlowNet**的有效性，本文在多个卫星数据集上进行了实验。这些数据集包括不同分辨率和光谱波段的全色和多光谱图像。

评估指标包括峰值信噪比（**PSNR**）、结构相似性（**SSIM**）等常用图像质量评价指标。此外，为了评估生成**HRMS**图像的多样性，本文还引入了额外的评估指标，如生成图像的熵和互信息等。

实验结果表明，**PanFlowNet**在视觉和量化评估上均优于其他最先进的方法。具体来说，**PanFlowNet**能够生成具有高空间和高光谱分辨率的**HRMS**图像，同时保持图像的真实感和细节。

![](https://pic.imgdb.cn/item/673f182ed29ded1a8c03d164.png)
![](https://pic.imgdb.cn/item/673f1821d29ded1a8c03c751.png)

与确定性映射方法相比，**PanFlowNet**能够生成多样化的**HRMS**图像，从而更好地反映输入**LRMS**和**PAN**图像之间的复杂关系。此外，**PanFlowNet**还具有较好的泛化能力，能够在不同数据集上取得一致的良好表现。

![](https://pic.imgdb.cn/item/673f1850d29ded1a8c03e8a6.png)