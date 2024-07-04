---
layout: post
title: 'CLIP-Count: Towards Text-Guided Zero-Shot Object Counting'
date: 2023-05-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/667bc72fd9c307b7e9469e73.png'
tags: 论文阅读
---

> CLIP-Count：文本引导的零样本目标计数.

- paper：[CLIP-Count: Towards Text-Guided Zero-Shot Object Counting](https://arxiv.org/abs/2305.07304)

给定一张图像$I \in \mathbb{R}^{H\times W\times 3}$，本文的目标是通过指定感兴趣目标的文本提示$t$，估计指定目标的二元密度图$\hat{y} \in \mathbb{R}^{H\times W}$，并通过对密度图求和得到目标的计数结果$N_{pred}=SUM(\hat{y})$。

**CLIP**模型通过对图像和文本对进行大规模预训练，学习对齐的文本和图像表示；但**CLIP**模型本身缺乏定位能力。为了增强**CLIP**模型的定位能力，本文设计了块-文本对比损失，将**CLIP**模型的视觉编码器特征块与目标类别文本对齐；并进一步设计了层次化文本-块交互模块生成不同分辨率的特征图，并将这些特征图解码为目标的密度图。

![](https://pic.imgdb.cn/item/667bd0f8d9c307b7e9583f9f.png)

## 1. 块-文本对比损失 patch-text contrastive loss

**CLIP**模型的视觉编码器把图像$I \in \mathbb{R}^{H\times W\times 3}$编码为图像块特征$z_x \in \mathbb{R}^{H/16\times W/16\times d}$。首先应用线性投影$\phi_p$把图像块特征$z_x$投影到与文本嵌入$\mathcal{E}_t$相同的通道维度：

$$
\mathcal{E}_p = \phi_p(z_x)
$$

为了确定目标的位置，把**ground truth**密度图进行最大池化，生成与图像块特征具有相同尺寸的二元掩码；根据掩码值将图像块特征的不同**patch**划分为正集$\mathcal{P}$和负集$\mathcal{N}$，进而构造对比损失：

$$
\mathcal{L}_{con} = -\log \frac{\sum_{i \in \mathcal{P}} \exp(s(\mathcal{E}_p^i,\mathcal{E}_t)/\tau)}{\sum_{i \in \mathcal{P}} \exp(s(\mathcal{E}_p^i,\mathcal{E}_t)/\tau)+\sum_{k \in \mathcal{N}} \exp(s(\mathcal{E}_p^k,\mathcal{E}_t)/\tau)}
$$

![](https://pic.imgdb.cn/item/667bd4ced9c307b7e95eb211.png)

通过该方式将正的图像块嵌入拉近到固定的文本嵌入，将负的图像块嵌入推远，将图像块嵌入空间与文本嵌入空间对齐，以增强图像块特征的定位能力。

![](https://pic.imgdb.cn/item/667bd503d9c307b7e95f06bb.png)

在微调视觉编码器时，为了有效地利用**CLIP**中预训练的知识，将**CLIP ViT**中的参数冻结，并将少量可训练的参数以视觉提示（**visual prompt**）的形式连接到每层的输入。这使模型能够使用更少的数据和内存来更新**CLIP**中的知识，同时仍然受益于**CLIP**学习的丰富的视觉表示。

## 2. 层次化文本-块交互模块 hierarchical text-patch interaction module

场景中的物体通常具有多种尺度，但自注意力结构本质上缺乏对尺度不变特征建模的归纳偏差，导致密度估计不准确。本文提出了一个层次化文本-块交互模块，该模块能够生成不同分辨率的特征图，并使用这些特征图解码为目标的密度图。

该模块包括两个层次，每个层次依次应用多头自注意力(**MHSA**)，多头交叉注意力(**MHCA**)和两层**MLP** (**FFN**)。**MHSA**捕获全局特征关系，而**MHCA**将嵌入文本的语义信息传递给视觉特征。在两层之间使用双线性插值层将特征图的分辨率提高了一倍，在更细的粒度上捕捉文本和图像之间的关系。交互模块的最终输出是两层多模态特征图$M_c,M_f$。解码器将特征图$M_c,M_f$映射为二元密度图$\hat{y}$。

![](https://pic.imgdb.cn/item/667bd741d9c307b7e962caf4.png)

## 3. 实验分析

实验设计的数据集基准包括：
- **FSC-147**：由147个类别的6135张图像组成，每个训练图像使用类别名作为文本提示；
- **CARPK**：提供了1448张停车场鸟瞰图，共包含89,777辆汽车。
- **ShanghaiTech**：是一个综合性的人群统计数据集。它由A和B两部分组成，总共有1198张注释图像。Part A包含482张图片，其中400张用于训练，182张用于测试。B部分包括716张图像，其中400张用于训练，316张用于测试。

采用平均绝对误差(**MAE**)和均方根误差(**RMSE**)来评估目标计数的性能。

结果表明，该方法在**FSC147**数据集上的表现明显优于最先进的计数方法；在不经过微调的情况下，泛化到**CARPK**和**ShanghaiTech**数据集时，所提通用计数方法优于代表性的特定类别计数方法。

![](https://pic.imgdb.cn/item/667bd911d9c307b7e966ea53.png)

将预测的密度图覆盖在输入图像上，结果表明所提方法可以更好地将高密度定位在高保真的物体中心，并能够对不同类别、形状、大小和密度的物体进行鲁棒的定位和计数。

![](https://pic.imgdb.cn/item/667bd96dd9c307b7e96797a6.png)