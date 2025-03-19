---
layout: post
title: 'Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model'
date: 2024-07-13
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/67c947b7066befcec6ded7ac.png'
tags: 论文阅读
---

> Vision Mamba: 使用双向状态空间模型实现高效视觉表示学习.

- paper：[Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)

## 0. TL;DR

本文提出了**Vision Mamba（Vim）**，一种基于双向状态空间模型（**SSM**）的新型视觉骨干网络。**Vim**通过引入位置嵌入和双向**SSM**块，实现了对视觉数据的全局上下文建模，同时克服了传统**SSM**模型在视觉任务中的局限性。实验结果表明，**Vim**在**ImageNet**分类、**COCO**目标检测和**ADE20K**语义分割等任务上均优于现有的视觉**Transformer**模型（如**DeiT**），并且在处理高分辨率图像时具有显著的计算和内存效率优势。

## 1. 背景介绍

近年来，状态空间模型（**SSMs**）在序列建模领域逐渐崭露头角，成为**Transformer**架构的有力竞争者。**SSMs**的优势在于其线性时间推理、高度并行化的训练以及在长序列处理任务中的强大性能。**Mamba**作为一种基于**SSM**的模型，通过其选择性机制和硬件感知设计，实现了令人印象深刻的性能，成为挑战注意力机制的**Transformer**架构的有力候选者。

与此同时，视觉**Transformer（ViTs）**在视觉表示学习领域取得了巨大成功，特别是在大规模自监督预训练和下游任务的高性能方面。然而，**Transformer**的自注意力机制在处理长距离视觉依赖时，例如高分辨率图像，会带来速度和内存使用方面的挑战。

为了将**Mamba**的成功从语言建模领域转移到视觉领域，本文提出了**Vim**，一种基于纯**SSM**的通用视觉骨干网络。**Vim**通过引入双向**SSM**块和位置嵌入，实现了对视觉数据的全局上下文建模，同时保持了**SSM**的高效性。


## 2. Vim 模型

**Vim**模型结合了双向SSM，用于数据依赖的全局视觉上下文建模；并通过位置嵌入实现位置感知的视觉识别。

![](https://pic1.imgdb.cn/item/67c94ca7066befcec6dedc60.png)

为了使用**Mamba**处理视觉任务，作者设计了如下数据处理流程：
- 使用**16×16**的核大小投影层将二维图像拆分成二维图像块；
- 将图像块展平为向量，进行线性投影，并添加位置编码；
- 引入类标记，使用**Vim**模型处理序列；
- 对输出类标记进行线性投影，得到分类结果。

原始的**Mamba**模块是为一维序列设计的，不适合需要空间感知理解的视觉任务。故**Vim Block**结合了双向序列建模以用于视觉任务。
- 输入序列首先通过归一化层；
- 将归一化的序列线性投影为$x,z$；
- $x$通过双向**SSM**块，第一个块以前向处理视觉序列，第二个块以后向处理视觉序列；
- 通过$z$对双向的输出序列进行门控；
- 相加并通过线性投影得到输出序列。

![](https://pic1.imgdb.cn/item/67c950b9066befcec6dee3ad.png)


## 3. 实验分析

作者对**Vim**的双向设计进行了消融实验，结果表明，使用双向**SSM**，并且在 **SSM** 之前添加一个**Conv1d**的效果最好。

![](https://pic1.imgdb.cn/item/67c951f4066befcec6dee454.png)

对于图像分类任务，**Vim**在**ImageNet-1K**数据集上的表现优于现有的卷积网络、**Transformer**网络和**SSM**网络。具体来说，**Vim-Ti**的**Top-1**准确率为**76.1%**，高于**DeiT-Ti**的**72.2%**。**Vim-S**的**Top-1**准确率为**80.3%**，高于**DeiT-S**的**79.8%**。此外，**Vim**在处理高分辨率图像时具有显著的速度和内存优势。

![](https://pic1.imgdb.cn/item/67c952b8066befcec6dee68a.png)

对于语义分割任务，**Vim**在**ADE20K**数据集上的表现也优于**DeiT**。具体来说，**Vim-Ti**的**mIoU**为**41.0%**，高于**DeiT-Ti**的**39.2%**。**Vim-S**的**mIoU**为**44.9%**，高于**DeiT-S**的**44.0%**。

![](https://pic1.imgdb.cn/item/67c95398066befcec6dee8b0.png)

对于目标检测和实例分割，**Vim**在**COCO 2017**数据集上的表现也优于**DeiT**。具体来说，**Vim-Ti**的目标检测**AP**为**45.7%**，高于**DeiT-Ti**的**44.4%**。实例分割的**AP**为**39.2%**，高于**DeiT-Ti**的**38.1%**。

![](https://pic1.imgdb.cn/item/67c953f0066befcec6dee926.png)

