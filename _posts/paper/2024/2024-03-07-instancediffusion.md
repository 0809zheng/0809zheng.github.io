---
layout: post
title: 'InstanceDiffusion: Instance-level Control for Image Generation'
date: 2024-03-07
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/673700c5d29ded1a8c6c14df.png'
tags: 论文阅读
---

> InstanceDiffusion：图像生成的实例级控制.

- paper：[InstanceDiffusion: Instance-level Control for Image Generation](https://arxiv.org/abs/2402.03290)

## TL; DR

本文提出了**InstanceDiffusion**，一种为文本到图像的扩散模型添加精确实例级控制的框架。**InstanceDiffusion**支持自由形式的控制条件，允许用户以单点、涂鸦、边界框或复杂的实例分割掩码等形式灵活地指定目标位置。通过 **UniFusion** 模块为模型引入了实例级条件，通过**ScaleU** 模块提高了图像保真度，通过**Multi-instance Sampler**改进了多个实例的生成过程，**InstanceDiffusion**能够生成高质量且符合用户指定实例级条件和属性的图像。

![](https://pic.imgdb.cn/item/67370209d29ded1a8c6d0519.png)


## 1. 背景介绍

文本到图像的扩散模型能够生成高质量的图像，但它们缺乏对图像中单个实例的精细控制。这限制了用户根据具体需求定制图像内容的能力。尽管一些方法如**ControlNet**和**GLIGEN**通过允许用户包含额外的图像或语义分割掩码来添加空间控制，但它们仍然存在局限性。例如**ControlNet**主要关注空间条件，而**GLIGEN**使用目标类别作为文本提示，缺乏详细的实例级提示的训练。为了克服这些限制，本文提出了**InstanceDiffusion**，它允许用户以灵活的方式指定多个实例的位置和属性，从而生成符合要求的图像。


## 2. 方法介绍

**InstanceDiffusion**通过在文本到图像扩散模型中引入实例级条件，结合**UniFusion**、**ScaleU**和**Multi-instance Sampler**三大模块，实现了对图像生成的精确控制。它支持多种位置指定方式，包括单点、涂鸦、边界框和实例分割掩码等。

![](https://pic.imgdb.cn/item/673704acd29ded1a8c6ee9bc.png)

### （1）UniFusion模块

**UniFusion**模块将各种形式的实例级条件投影到相同的特征空间，并将实例级布局和文本描述融合到视觉**token**中。这使得模型能够理解和处理复杂的实例级条件。

![](https://pic.imgdb.cn/item/67370598d29ded1a8c6f9b66.png)

把单点、涂鸦、边界框和实例分割掩码转换成不同数量的点集，然后通过不同的**Tokenizer**提取对应**token**。坐标点先通过**Fourier mapping**转换，与类别嵌入连接后使用**MLP**进行编码。不同控制条件的**token**通过掩码自注意力计算与视觉**token**进行融合，再融合进主路经中。

### （2）ScaleU模块

**ScaleU**模块通过重新校准**UNet**中的主路径特征和跳跃连接特征，增强了模型精确遵循指定布局条件的能力。这有助于生成更符合用户要求的图像。

**UNet**中的特征包括主路径特征$F_b$和跳跃连接特征$F_s$，主路径特征用于降噪，跳跃连接特征用于补充高频特征。通过减少跳跃连接特征中的低频部分，能够增强主路径特征。

**ScaleU**模块引入两个可学习的通道级向量$s_b,s_s$校准两个特征：

$$
F_b^\prime = F_b \otimes (\tanh(s_b)+1) \\
F_s^\prime = IFFT(FFT(F_s) \odot \alpha), \alpha(r) = \begin{cases} \tanh(s_s) + 1 & \text{if } r \leq r_{thresh} \\ 1 & \text{otherwise} \end{cases}
$$

### （3）Multi-instance Sampler模块

**Multi-instance Sampler**模块采用多实例采样策略，确保每个实例在生成过程中都能保持其独特性和准确性，减少多个实例（文本+布局）之间条件之间的信息泄露和混淆。

![](https://pic.imgdb.cn/item/6737090bd29ded1a8c73133b.png)

对于$T$步采样过程，首先分别为每个实例采样$M$步，将特征融合后再采样$T-M$步。


## 3. 实验分析

由于获取大规模的配对（实例，图像）数据集很困难，本文使用最先进的识别系统自动生成了一个包含实例级位置和文本标注的数据集。首先使用开放词汇图像标记模型**RAM**为图像生成一个标签列表，然后使用**Grounded-SAM**根据每个标签生成检测框，最后使用视觉语言模型**BLIP-V2**为每个框生成描述。

为了评估**InstanceDiffusion**的性能，本文引入了多种评估指标，对于边界框和掩码输入，采用**AP**和**IoU**指标；对于点和涂鸦输入，采用实例级精度（**PiM**）指标；此外所有输入都采用**FID**指标评估。这些指标能够全面反映模型在遵循实例级条件和属性方面的能力。

实验结果表明，**InstanceDiffusion**在多个评估指标上显著优于现有方法。它不仅能够生成高质量的图像，还能够精确遵循用户指定的实例级条件和属性。

![](https://pic.imgdb.cn/item/67370ae2d29ded1a8c75ad34.png)

此外**InstanceDiffusion**还支持多种位置指定方式，具有高度的灵活性和实用性。

![](https://pic.imgdb.cn/item/67370b23d29ded1a8c760c25.png)

**InstanceDiffusion**的精确实例控制能力使其在多轮图像生成中表现出色。用户可以在保持先前生成目标和整体场景一致性的同时，将目标放置在特定位置。这种能力为未来的细粒度图像生成研究提供了更多的可能性。

![](https://pic.imgdb.cn/item/67370b5cd29ded1a8c766431.png)