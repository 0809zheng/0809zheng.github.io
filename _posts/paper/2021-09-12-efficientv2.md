---
layout: post
title: 'EfficientNetV2: Smaller Models and Faster Training'
date: 2021-09-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61d52d062ab3f51d910b942c.jpg'
tags: 论文阅读
---

> EfficientNetV2: 更小的模型和更快的训练.

- paper：EfficientNetV2: Smaller Models and Faster Training
- arXiv：[link](https://arxiv.org/abs/2104.00298)


作者提出了**EfficientNetV2**，其结构是通过训练感知(**train-aware**)的神经结构搜索和模型缩放得到的。
在搜索空间中使用了**Fused-MBConv**等新操作。
与**EfficientNet**相比，**EfficientNetV2**训练速度快$11$倍，并且模型参数量小$6.8$倍。
作者在训练过程中逐渐增大输入图像的大小以加快训练，为了补偿精度下降，训练时随图像大小自适应地调整图像增强等正则化程度。

**EfficientNet**使用神经结构搜索获得基线模型**EfficientNet-B0**，然后使用复合缩放策略进一步获得**B1-B7**。在训练该模型时存在一些训练瓶颈，导致模型的训练速度是次优的，这些瓶颈包括：
- 使用非常大的图像进行训练很慢。尺寸大的图像会导致大量的内存使用，由于硬件的总内存是固定的，只能减少训练的批量，导致训练速度的降低。
- 深度卷积在浅层较慢在深层高效。尽管深度卷积比常规卷积具有更少的参数，但它们不能充分地被现代加速器加速。
- 平均地缩放网络的每个阶段是次优的。比如平均地增加每一阶段的层数，每一层深度对训练速度和参数效率的贡献不是平等的。

**EfficientNetV2**的搜索空间包括两种卷积，即[MobileNetV3](https://0809zheng.github.io/2021/09/15/mobilenetv3.html)中的**MBConv**和一种改进的**Fused-MBConv**。

![](https://pic.imgdb.cn/item/61d558bb2ab3f51d91428d0c.jpg)

搜索目标综合考虑模型精度$A$、训练时间$S$和参数量$P$，设置为$A\cdot S^{-0.07} \cdot P^{-0.05}$。搜索得到的结构称为**EfficientNetV2-S**：

![](https://pic.imgdb.cn/item/61d545dc2ab3f51d9126915e.jpg)

与**EfficientNet**类似，**EfficientNetV2**也可以通过模型缩放进行扩展。在扩展时将图像最大尺寸限制为$480$，以防止更大的图像导致的内存和训练开销。不同于平均缩放，作者在更深的网络阶段增加更多的层，以便在不显著增加运行开销的情况下增加网络容量。

![](https://pic.imgdb.cn/item/61d551752ab3f51d9138c24a.jpg)

训练时作者采用了一种渐进式学习方法，即在训练早期使用尺寸较小的图像和较弱的正则化方法，随着训练轮数增加逐步应用更大的图像和更强的正则化。

假设训练共有$N$轮，将其分成$M$个阶段。每个阶段应用图像尺寸$S_i$和正则化程度$\Phi_i$。指定最后一阶段使用的图像尺寸$S_e$和正则化程度$\Phi_e$，并随机初始化图像尺寸$S_0$和正则化程度$\Phi_0$，则每个阶段使用的图像尺寸和正则化程度通过线性插值得到。整个流程如下：

![](https://pic.imgdb.cn/item/61d551da2ab3f51d913909c7.jpg)

可以调整的正则化方法包括：
- **Dropout**：一种网络级的正则化，通过随机丢弃通道减少过度依赖，可以调整概率$\gamma$。
- **RandAugment**：一种针对每个图像的数据增强，可以调整增强幅度$\epsilon$。
- **Mixup**：一种跨图像的数据增强，可以调整混合率$\lambda$。

对于不同的网络结构，设置参数如下：

![](https://pic.imgdb.cn/item/61d5534b2ab3f51d913bfa2e.jpg)

实验如下：

![](https://pic.imgdb.cn/item/61d554612ab3f51d913db5f4.jpg)