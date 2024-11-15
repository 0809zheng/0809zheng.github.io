---
layout: post
title: '布局引导图像生成(Layout-to-Image Generation)'
date: 2024-03-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/66457c7dd9c307b7e9094560.png'
tags: 深度学习
---

> Layout-to-Image Generation.

[**布局引导图像生成 (Layout-to-Image Generation)**](https://paperswithcode.com/task/layout-to-image-generation)是图像感知任务（如目标检测、图像分割）的逆过程，即根据给定的布局生成对应的图像。

布局描述了包含在图像中的目标信息，可以通过目标类别、边界框、分割**mask**、关键点、边缘图、深度图等来定义。布局引导的图像生成是指学习一个条件生成模型$\mathcal{G}$，在给定几何布局$L$的条件下生成图像$I$：

$$
I = \mathcal{G}(L,z) , \quad z\sim \mathcal{N}(0,1)
$$

根据布局控制条件的输入形式，布局引导的图像生成模型包括：
- 文本级**L2I**模型：通过将空间布局转换成文本**token**实现布局控制，如**ReCo**, **LayoutDiffusion**, **GeoDiffusion**, **DetDiffusion**。
- 像素级**L2I**模型：通过提供像素级空间对齐条件实现布局控制，如**GLIGEN**, **LayoutDiffuse**, **ControlNet**, **InstanceDiffusion**。

## 1. 文本级L2I模型

### ⚪ [<font color=Blue>ReCo</font>](https://0809zheng.github.io/2024/03/04/reco.html)
- （arXiv2211）ReCo: Region-Controlled Text-to-Image Generation

**ReCo**的输入布局包括图像的文本描述和每个目标框的类别、位置、文本描述。**ReCo**在预训练文本词嵌入$T$的同时引入位置标记$P$，用四个浮点数表示每个区域的位置标记$P$，即边界框的左上坐标和右下坐标$$<x_1>,<y_1>,<x_2>,<y_2>$$。输入序列设计为图像描述+多个位置标记和相应的区域描述。预训练CLIP文本编码器将标记编码为序列嵌入。

![](https://pic.imgdb.cn/item/665d8eb75e6d1bfa05f3a4ea.png)

### ⚪ [<font color=Blue>LayoutDiffusion</font>](https://0809zheng.github.io/2024/03/03/layoutdiffusion.html)
- （arXiv2303）LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation

**LayoutDiffusion**模型由布局融合模块(**Layout Fusion Module, LFM**)和目标感知交叉注意机制(**object-aware Cross-Attention Mechanism, OaCA**)组成。该模型的输入布局包括每个目标框的类别、位置。
- **LFM**融合了每个目标的信息，并对多个目标之间的关系进行建模，从而提供了整个布局的潜在表示。
- **OaCA**将图像**patch**特征与布局在统一的坐标空间中进行交叉注意力计算，将两者的位置表示为边界框，使模型更加关注与目标相关的信息。

![](https://pic.imgdb.cn/item/6650509ed9c307b7e9623aba.png)

### ⚪ [<font color=Blue>GeoDiffusion</font>](https://0809zheng.github.io/2024/03/19/geodiffusion.html)
- （arXiv2306）GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation
  
**GeoDiffusion**可以灵活地将边界框或几何控制（例如自动驾驶场景中的相机视图）转换为文本提示，实现高质量的检测数据生成，有助于提高目标检测器的性能。该模型的输入布局包括与布局相关的几何条件和每个目标框的类别、位置。

**GeoDiffusion**进一步引入一个自适应重加权**mask**与重构损失相乘，使模型能够更加关注前景生成，解决前景与背景不平衡问题。
- **Constant re-weighting**：为了区分前景和背景区域，引入一种重加权策略，即为前景区域分配一个权重$w > 1$，大于分配给背景区域的权重。
- **Area re-weighting**：面积重加权方法动态地为较小的目标框分配更高的权重。对于尺寸为$H\times W$的**mask**，像素$(i,j)$所属边界框的面积为$c_{ij}$，则为像素设置权重：

$$
m_{ij}^\prime = \begin{cases}
\frac{w}{c_{ij}^p}, & (i,j) \in \text{foreground} \\
\frac{1}{(HW)^p}, & (i,j) \in \text{background}
\end{cases}
$$

![](https://pic.imgdb.cn/item/6641d4ee0ea9cb1403dccfab.png)

### ⚪ [<font color=Blue>DetDiffusion</font>](https://0809zheng.github.io/2024/03/20/detdiffusion.html)
- （arXiv2403）DetDiffusion: Synergizing Generative and Perceptive Models for Enhanced Data Generation and Perception

**DetDiffusion**使用感知模型为生成模型提供信息，从而增强后者生成控制的能力。该模型的输入布局包括每个目标框的类别、位置和感知属性。
- 构造每个目标框的**感知属性**：使用预训练的目标检测器提取边界框$b=[b_1,...,b_n]$，感知属性定义为每个真值框$o=[o_1,...,o_m]$的检测难度。对于每个真值框$o_i$，通过与$n$个预测框的交集来评估其检测难度：

$$
d_i = \begin{cases}
[\text{easy}], & \exists j, \text{IoU}(b_j, o_i) > \beta \\
[\text{hard}], & \text{else}
\end{cases}
$$

- 构造感知损失促进更细致的图像重建，并精确控制图像属性：使用预训练的图像分割模型提取多层次掩码伪标签$M=[m_1,...,m_k]$。在优化模型的高维特征空间时，引入掩码损失$\mathcal{L}_m$和**Dice**损失$\mathcal{L}_d$。

![](https://pic.imgdb.cn/item/664182350ea9cb14036385cc.png)


## 2. 像素级L2I模型

### ⚪ [<font color=Blue>GLIGEN</font>](https://0809zheng.github.io/2024/03/02/gligen.html)
- （arXiv2301）GLIGEN: Open-Set Grounded Text-to-Image Generation

**GLIGEN**冻结原始模型的权重，并通过引入门控自注意力层来将输入布局信息引入模型。输入布局信息采用不同的方式进行编码：
- 图像的文本描述和目标实体的文本描述使用预训练文本编码器获得每个单词的上下文文本特征；
- 边界框、关键点等点坐标集合通过傅里叶嵌入提取每个点坐标特征；
- 输入条件图像使用图像编码器获得特征；
- 空间对齐条件（如边缘图、深度图、法线图和语义图）将**UNet**的第一个卷积层设置为可训练，将这类条件通过一个下采样网络并与输入图像连接后输入**UNet**。

![](https://pic.imgdb.cn/item/6645cf7bd9c307b7e99083fd.png)

### ⚪ [<font color=Blue>LayoutDiffuse</font>](https://0809zheng.github.io/2024/03/06/layoutdiffuse.html)
- （arXiv2302）LayoutDiffuse: Adapting Foundational Diffusion Models for Layout-to-Image Generation

**LayoutDiffuse**通过微调预训练扩散模型实现了布局到图像的生成。**LayoutDiffuse**通过在**U-Net**的每一层之后添加布局注意力（**layout attention**）层来引入布局条件，布局注意力层为每个类别和背景特征引入可学习的**token**，并分别计算特征自注意力再融合；此外还添加了任务感知提示（**task-aware prompt**）来指示模型生成任务的更改，任务感知提示通过在预训练模型的输入前添加一个可学习的嵌入向量来实现。

![](https://pic.imgdb.cn/item/6736f7fad29ded1a8c64e438.png)


### ⚪ [<font color=Blue>ControlNet</font>](https://0809zheng.github.io/2024/03/05/controlnet.html)
- （arXiv2302）Adding Conditional Control to Text-to-Image Diffusion Models

**ControlNet**冻结神经网络原始模块的参数$\Theta$，并拷贝该模块为具有参数$\Theta_c$的可训练版本。将拷贝模块通过两个零卷积$$\mathcal{Z}$$（参数初始化为$0$的$1\times 1$卷积）连接到原模块，接收输入布局条件$c$作为输入。零卷积保证了训练的初始阶段不会引入有害的训练噪声，保留了大型预训练模型的生成能力。

$$
y_c = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x+\mathcal{Z}(c; \Theta_1); \Theta_c); \Theta_2)
$$

**ControlNet**接受边缘、**Hough**线、用户涂鸦、人体关键点、分割图、形状法线、深度图等形式的布局图像，使用由4 × 4卷积核、步长为2的4个卷积层组成的网络(由ReLU激活，分别使用16、32、64、128个通道，用高斯初始化并与完整模型联合训练)将图像编码为输入布局条件$c$。

![](https://pic.imgdb.cn/item/66a8aaf9d9c307b7e901e311.png)

### ⚪ [<font color=Blue>InstanceDiffusion</font>](https://0809zheng.github.io/2024/03/07/instancediffusion.html)
- （arXiv2402）InstanceDiffusion: Instance-level Control for Image Generation

**InstanceDiffusion**支持单点、涂鸦、边界框和实例分割掩码等输入布局，使用**UniFusion**、**ScaleU**和**Multi-instance Sampler**三大模块实现了对图像生成的精确控制。
- **UniFusion**模块将各种形式的实例级条件投影到相同的特征空间，并将实例级布局和文本描述融合到视觉**token**中。
- **ScaleU**模块通过重新校准**UNet**中的主路径特征和跳跃连接特征，增强了模型精确遵循指定布局条件的能力。
- **Multi-instance Sampler**模块采用多实例采样策略，确保每个实例在生成过程中都能保持其独特性和准确性，减少多个实例之间条件之间的信息泄露和混淆。

![](https://pic.imgdb.cn/item/673704acd29ded1a8c6ee9bc.png)




