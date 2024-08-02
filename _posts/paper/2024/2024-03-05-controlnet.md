---
layout: post
title: 'Adding Conditional Control to Text-to-Image Diffusion Models'
date: 2024-03-05
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/66a89854d9c307b7e9f085a1.png'
tags: 论文阅读
---

> 向文本到图像扩散模型添加条件控制.

- paper：[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

这项工作提出了**ControlNet**，可以实现大型预训练文本到图像扩散模型的条件控制。**ControlNet**可以通过各种条件输入来控制**Stable Diffusion**，包括边缘、**Hough**线、用户涂鸦、人体关键点、分割图、形状法线、深度等。

![](https://pic.imgdb.cn/item/66a8a4d3d9c307b7e9fd3ea7.png)

**ControlNet**向神经网络的模块中引入额外的条件。对于一个训练好的网络模块$$\mathcal{F}(\cdot; \Theta)$$，将输入特征映射$x$转换为另一个特征映射$y$:

$$
y = \mathcal{F}(x; \Theta)
$$

**ControlNet**冻结原始模块的参数$\Theta$，并拷贝该模块为具有参数$\Theta_c$的可训练版本。将拷贝模块通过两个零卷积$$\mathcal{Z}$$（参数初始化为$0$的$1\times 1$卷积）连接到原模块：

$$
y_c = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(\mathcal{Z}(x; \Theta_1); \Theta); \Theta_2)
$$

![](https://pic.imgdb.cn/item/66a8aaf9d9c307b7e901e311.png)

零卷积保证了训练的初始阶段不会引入有害的训练噪声，保留了大型预训练模型的功能，并能够通过额外参数进一步学习。

将**ControlNet**引入**Stable Diffusion**如图所示。**Stable Diffusion**是典型的**U-net**结构，使用**ControlNet**创建了**Stable Diffusion**的12个编码块和1个中间块的可训练副本。由于原始参数是冻结的，不需要梯度计算来进行微调，能够加快训练速度并节省**GPU**内存。

![](https://pic.imgdb.cn/item/66a8acd4d9c307b7e904af82.png)

**ControlNet**使用由4 × 4卷积核、步长为2的4个卷积层组成的网络(由ReLU激活，分别使用16、32、64、128个通道，用高斯初始化并与完整模型联合训练)将图像空间条件$c_i$编码为特征空间条件向量$c_f$。

在训练过程中，由于零卷积不会给网络增加噪声，因此模型应该总是能够预测高质量的图像。实验观察到，模型不是逐渐学习控制条件，而是在不到10K的优化步骤内突然成功地根据输入条件生成图像，作者称之为“突然收敛现象”。

![](https://pic.imgdb.cn/item/66ac8ed7d9c307b7e92781f9.png)

作者对**ControlNet**的添加形式进行消融实验（零卷积、常规卷积核直接添加），结果显示当替换零卷积时，**ControlNet**的性能下降，这表明在微调期间预训练模型的知识被破坏。

![](https://pic.imgdb.cn/item/66ac90e3d9c307b7e929b124.png)

