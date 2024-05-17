---
layout: post
title: 'DetDiffusion: Synergizing Generative and Perceptive Models for Enhanced Data Generation and Perception'
date: 2024-03-20
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/664171e80ea9cb14034c4722.png'
tags: 论文阅读
---

> DetDiffusion：用于增强数据生成与感知的协同生成与感知模型.

- paper：[DetDiffusion: Synergizing Generative and Perceptive Models for Enhanced Data Generation and Perception](https://arxiv.org/abs/2403.13304)

通过提供类别标签、分割热图和目标边界框等标注信息，生成模型的合成数据可以提高感知模型在下游任务(如分类、目标检测和分割)的性能。大多数方法侧重于分别改进生成模型或感知模型。

本文研究了生成模型和感知模型的协同作用：使生成模型能够利用来自感知模型的信息，从而增强其控制生成的能力；同时有针对性地生成数据，从而提高感知模型的性能。

![](https://pic.imgdb.cn/item/66417a120ea9cb140357e1bc.png)

## 1. 模型设计

为了提高检测模型的性能，首先从预训练的检测器中提取目标属性，这些属性包含了目标检测所必需的关键视觉特征；然后将提取的属性集成到生成模型的训练机制中。确保生成的图像更有利于训练检测器，从而有可能显著提高检测精度和可靠性。

为了提高生成模型在生成高质量、感知对齐图像的能力，本文设计了目标的感知属性，并设计了感知损失进行更细致的图像重建和图像属性控制。

![](https://pic.imgdb.cn/item/664182350ea9cb14036385cc.png)

### ⚪ 感知属性

对于每一个图像，使用预训练的检测器提取边界框$b=[b_1,...,b_n]$，并且提供标签的真值框$o=[o_1,...,o_m]$，**感知属性(Perception-Aware Attribute)**定义为每个真值框的检测难度。对于每个真值框$o_i$，通过与$n$个预测框的交集来评估其检测难度：

$$
d_i = \begin{cases}
[\text{easy}], & \exist j, \text{IoU}(b_j, o_i) > \beta \\
[\text{hard}], & \text{else}
\end{cases}
$$

此时图像的每个真值框都具有三种属性：类别属性$c$、位置属性$l$和感知属性$d$，设计如下文本提示符：

$$
\text{An image with \{objects\}} \\ 
\text{objects} = [(c_1,l_1,d_1),...,(c_m,l_m,d_m)]
$$

![](https://pic.imgdb.cn/item/66417de40ea9cb14035d0e3d.png)

### ⚪ 感知损失

在训练生成扩散模型时，目标是最小化预测图像与真实图像之间的重构距离。本文提出一种**感知损失(perception-aware loss)**，利用丰富的视觉特征促进更细致的图像重建，并精确控制图像属性。

使用预训练的**UNet**模型能够从图像中获取多层次特征$f=[f_1,...,f_k]$，额外引入一个分割头将这些特征解码为预测掩码$M=[m_1,...,m_k]$。在优化模型的高维特征空间时，引入掩码损失$\mathcal{L}_m$和**Dice**损失$\mathcal{L}_d$，并引入去噪扩散概率模型(**DDPM**)中的$\sqrt{\overline{\alpha}}_t$来平衡特征中的噪声成分，减少具有较高噪声水平的特征图的影响，从而强调具有较低噪声的特征图。从而构造感知损失：

$$
\mathcal{L}_p = \sqrt{\overline{\alpha}}_t(\mathcal{L}_m+\mathcal{L}_d)
$$

## 2. 实验分析

作者在**COCO-Thing-Stuff**基准上进行实验，该任务包括118,287张训练图像和5,000张验证图像。每张图像都用边界框标注了80个类别的目标，像用素级分割标注了91个类别的目标。

### ⚪ 保真度实验

布局引导生成要求生成的目标尽可能与原始图像中保持一致，同时保证生成高质量的图像。因此对保真度实验进行综合分析。
- 使用**Frechet Inception Distance (FID)**评估生成图像的整体视觉质量，**FID**使用**ImageNet**预训练的**InceptionV3**网络对真实图像和生成图像之间的特征分布进行比较。
- 使用**YOLO Score**评估生成模型中目标检测的精度。**YOLO Score**使用预训练的**YOLOv4**模型检测生成图像上80个目标类别的边界框的平均精度(**mAP**)。

结果表明**DetDiffusion**获得了最好的**FID(19.28)**，并且在**YOLO Score**上优于其他模型。这表明使用感知损失和引入感知属性可以生成更真实的图像。此外通过将感知属性人为指定为**easy/hard**能够控制生成图像的检测难度级别，比如全部指定为**easy**生成的图像易于检测器感知。

![](https://pic.imgdb.cn/item/6641850c0ea9cb1403679507.png)

### ⚪ 检测实验

生成目标检测数据的一个重要目的是其对下游目标检测的适用性，因此给出可训练性实验。可训练性的评估包括使用预训练生成模型从原始标签中创建新的合成训练集。然后使用原始和合成训练集来训练检测器。

![](https://pic.imgdb.cn/item/664187e50ea9cb14036af921.png)

从**COCO2017**数据集中包含3至8个目标的标签进行生成，产生了一个包含47,429张图像和210,893个目标的训练集。结果表明，模型生成的数据显著增强了下游的训练，并且全部指定为**hard**生成的图像作为一种更强的数据增强形式，在所有检测器指标中表现出最大的改进。

![](https://pic.imgdb.cn/item/6641876c0ea9cb14036a6d01.png)

作者可视化了感知属性的选择，比较了**easy/hard**实例。**easy**属性的图像生成的重点是目标的内在特征，确保清晰度和无噪声。**hard**的属性的图像生成包含了通过遮挡、照明和其他复杂性引入噪声的额外元素。

![](https://pic.imgdb.cn/item/6641b5a90ea9cb1403a83dd6.png)