---
layout: post
title: 'GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation'
date: 2024-03-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6641c65b0ea9cb1403c50736.png'
tags: 论文阅读
---

> GeoDiffusion: 目标检测数据生成的文本提示几何控制.

- paper：[GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation](https://arxiv.org/abs/2306.04607)

本文提出了**GeoDiffusion**，这是一种文本到图像生成模型，可以灵活地将边界框或几何控制（例如自动驾驶场景中的相机视图）转换为文本提示，实现高质量的检测数据生成，有助于提高目标检测器的性能。

![](https://pic.imgdb.cn/item/6641d02d0ea9cb1403d6f850.png)

## 1. 模型设计

### ⚪ 布局引导的图像生成

具有$N$个边界框的**几何布局(geometric layout)**记为$$L=(v,\{(c_i,b_i)\}_{i=1}^N)$$，其中$v$是与布局相关的几何体条件，$c_i$是语义类别，$b_i$是边界框的位置。

布局引导的图像生成是指学习一个模型$\mathcal{G}$，在给定几何布局$L$的条件下生成图像$I$：

$$
I = \mathcal{G}(L,z) ,z\sim \mathcal{N}(0,1)
$$

### ⚪ 编码几何布局

语义类别$c_i$和几何条件$v$可以通过替换为相应的文本进行编码。边界框的位置$b_i$通过将图像划分为位置网格来离散化连续坐标，每个离散位置对应一个唯一的位置标记。

给定网格尺寸$(H_{bin},W_{bin})$，任意点$(x_0,y_0)$对应的位置标记为：

$$
\sigma(x_0,y_0)=\mathcal{T}[y_{bin} \cdot W_{bin}+x_{bin}] \\
x_{bin} = \lfloor \frac{x_0}{W} \cdot W_{bin}\rfloor , y_{bin} = \lfloor \frac{y_0}{H} \cdot H_{bin}\rfloor
$$

通过随机排序策略，每次向模型输入布局时随机打乱目标框序列；用以下模板构造文本提示：

$$
\text{An image of \{view\} camera with \{boxes\}}
$$

![](https://pic.imgdb.cn/item/6641d4ee0ea9cb1403dccfab.png)

### ⚪ 前景优先重加权

用于训练生成模型的图像重构损失是在空间坐标上均匀先验分布的假设下设计的。由于前景和背景之间的极度不平衡，本文进一步引入一个自适应重加权**mask**与重构损失相乘，使模型能够更加关注前景生成，解决前景与背景不平衡问题。
- **Constant re-weighting**：为了区分前景和背景区域，引入一种重加权策略，即为前景区域分配一个权重$w > 1$，大于分配给背景区域的权重。
- **Area re-weighting**：面积重加权方法动态地为较小的目标框分配更高的权重。对于尺寸为$H\times W$的**mask**，像素$(i,j)$所属边界框的面积为$c_{ij}$，则为像素设置权重：

$$
m_{ij}^\prime = \begin{cases}
\frac{w}{c_{ij}^p}, & (i,j) \in \text{foreground} \\
\frac{1}{(HW)^p}, & (i,j) \in \text{background}
\end{cases}
$$

为了提高微调过程中的数值稳定性，对重加权掩码进行了归一化:

$$
m_{ij} = HW \cdot \frac{m_{ij}^\prime}{\sum_{i,j}m_{ij}^\prime}
$$

![](https://pic.imgdb.cn/item/6641d4910ea9cb1403dc58d8.png)

## 2. 实验分析

实验主要用**NuImages**数据集，该数据集由60K训练样本和15K验证样本组成，其中包含来自10个语义类的高质量边界框注释。该数据集从6个摄像头视图(前、左前、右前、后、左后和右后)捕获图像。

目标检测数据的质量取决于三个关键标准:保真度、可训练性和泛化性。
- **保真度 (fidelity)**要求在符合几何布局的同时生成足够真实的目标。
- **可训练性 (trainability)**要求生成的图像对训练目标检测器是有用的。
- **泛化性 (fidelity)**要求能够模拟真实数据集中的新场景。

### ⚪ 保真度

保真度实验采用如下评估指标：
- 使用**Frechet Inception Distance (FID)**评估生成图像的感知质量，**FID**使用**ImageNet**预训练的**InceptionV3**网络对真实图像和生成图像之间的特征分布进行比较。
- 使用**YOLO Score**评估生成图像和几何布局之间的一致性。**YOLO Score**使用预训练的**YOLOv4**模型检测生成图像上80个目标类别的边界框的平均精度(**mAP**)。

所提方法在感知质量**(FID)**和布局一致性**(mAP)**方面超过了所有基线，并伴随着4倍的训练加速(**64 vs 256 epoch**)，这表明文本提示的几何控制是一种有效的方法。

![](https://pic.imgdb.cn/item/6641db5c0ea9cb1403e883a3.png)

### ⚪ 可训练性

在目标检测器的训练过程中使用生成图像作为增强样本，以进一步评估提出的模型的可训练性。以**NuImages**训练集的数据注释作为输入，首先过滤小于图像面积0.2\%的边界框，然后以0.5的概率随机翻转和不超过256个像素的移动来增强边界框。使用**ImageNet**预训练权值初始化**Faster R-CNN**进行训练，并在验证集上进行评估。结果表明，通过缓解稀有类别的标签稀缺性，模型对几乎所有语义类都实现一致改进。

![](https://pic.imgdb.cn/item/6641dc650ea9cb1403e9ca7e.png)

随机抽取真实训练集的10\%、25\%、50\%和75\%，每个子集分别与生成的图像一起训练**Faster R-CNN**。结果表明，模型在不同的真实训练数据预算上实现了一致的改进。真实数据越稀缺，取得的改进就越显著，充分表明生成图像确实有助于缓解数据需求。

![](https://pic.imgdb.cn/item/6641dd140ea9cb1403eaa383.png)

### ⚪ 泛化性

为了保证输入几何布局是合理的，首先从验证集中随机采样一个查询布局，在此基础上通过翻转和随机移动进一步干扰查询边界框，以评估模型在新布局上的泛化性。结果表明模型可以在新的布局上进行生成。

![](https://pic.imgdb.cn/item/6641df400ea9cb1403ed226b.png)


