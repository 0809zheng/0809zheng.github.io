---
layout: post
title: 'LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation'
date: 2024-03-06
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6736eafbd29ded1a8c5ab380.png'
tags: 论文阅读
---

> LayoutDiffuse：调整基础扩散模型实现布局到图像生成.

- paper：[LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation](https://arxiv.org/abs/2302.08908)

## TL; DR

本文提出了一种名为**LayoutDiffuse**的方法，该方法通过微调预训练的基础扩散模型实现了从布局到图像的生成。**LayoutDiffuse**通过在**U-Net**的每一层之后添加布局注意力（**layout attention**）层来引入布局条件；此外还添加了任务感知提示（**task-aware prompt**）来指示模型生成任务的更改。实验结果显示，该方法在多个数据集上显著优于其他基于**GAN**、**VQ-VAE**和扩散模型的生成方法，不仅生成图像质量高，而且与布局保持一致。此外，**LayoutDiffuse**通过微调预训练模型显著减少了训练时间，提高了数据效率。

![](https://pic.imgdb.cn/item/6736ed26d29ded1a8c5c4d23.png)



## 1. 背景介绍

布局到图像生成旨在根据语义布局合成逼真的图像。生成的图像不仅需要视觉上合理，还需要与布局保持一致。近年来，扩散模型在图像合成方面取得了突破性进展，能够生成具有更好样本质量的图像，并且擅长生成语义信息。尽管扩散模型已被应用于文本到图像的生成、图像修复和超分辨率等条件生成任务，但如何扩展到布局到图像生成的研究却很少。

本文提出了一种新的方法，即**LayoutDiffuse**，它微调了预训练的基础扩散模型（可以是仅基于图像的或文本到图像的），用于布局到图像的生成。通过引入新颖的基于布局注意力和任务感知提示，**LayoutDiffuse**能够高效地训练，生成具有高感知质量和布局一致性的图像，并且所需数据更少。

## 2. 方法介绍

**LayoutDiffuse**通过**Adaptor**微调隐扩散模型用于布局到图像的生成，它已证明能够以较低的计算成本生成高质量图像。本文提出了两个组件：布局注意力和任务感知提示，用于微调扩散模型进行布局到图像的生成。

![](https://pic.imgdb.cn/item/6736f7fad29ded1a8c64e438.png)

### （1）布局注意力
为了将布局条件引入图像生成中，**LayoutDiffuse**在扩散模型中**U-Net**的每一层中添加了布局注意力层。布局注意力层在标准自注意力层的基础上引入了实例提示（**instance prompt**），对于每个类别$k$引入一个可学习的**token** $e_k$，如果区域$r$中出现了类别$k$，则将$e_k$添加到该区域的特征$a$中，并在该区域中计算自注意力。对于没有目标出现的区域，也学习一个**token** $e_\phi$并计算自注意力。根据前景和背景**mask**融合这两种自注意力结果。

![](https://pic.imgdb.cn/item/6736f720d29ded1a8c64362d.png)

### （2）任务自适应提示

任务自适应提示旨在识别模型生成任务是否已从预训练任务（即无条件图像生成或文本到图像生成）更改为布局到图像生成任务。**LayoutDiffuse**通过在预训练模型的输入前添加一个可学习的嵌入向量来实现这一点，该向量在训练过程中被优化以生成与布局条件一致的图像。

## 3. 实验分析

本文使用**COCO Stuff**和**Visual Genome（VG）**数据集来评估边界框布局到图像的性能。在测试时遵循以前的工作，过滤了目标数量不属于从3到8（COCO）和3到30（VG）的图像，并消除了小于图像面积2\%的目标。本文使用**CelebA-Mask**数据集来评估掩码布局到图像的性能。该数据集包含30,000张面部图像和相应的语义掩码，涵盖19个面部属性类别；本文在2,993张测试拆分图像上评估性能。

对于边界框布局到图像生成，通过**FID**, **IS**和检测准确性来定量评估生成图像的感知质量；此外还进行了人类评估，要求参与者基于整体质量或布局保真度从结果中选择一个最好的。对于掩码布局到图像生成，通过**FID**和**mIoU**来评估性能。

实验结果表明，**LayoutDiffuse**在边界框和掩码布局到图像生成任务上都取得了显著优于基线方法的结果。在COCO Stuff和Visual Genome数据集上，LayoutDiffuse在FID和IS方面均取得了最佳性能。在CelebA-Mask数据集上，LayoutDiffuse也取得了最高的mIoU分数。

![](https://pic.imgdb.cn/item/6736f9fcd29ded1a8c666a4c.png)


此外**LayoutDiffuse**的训练时间比基线方法短得多，只需数小时即可达到最佳性能，而基线方法通常需要数天。这得益于**LayoutDiffuse**对预训练扩散模型的微调策略，以及新设计的任务自适应提示。

![](https://pic.imgdb.cn/item/6736faa1d29ded1a8c66ef9f.png)
