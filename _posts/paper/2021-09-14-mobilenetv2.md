---
layout: post
title: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks'
date: 2021-09-14
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/613b02d044eaada739b51d31.jpg'
tags: 论文阅读
---

> MobileNetV2: 倒残差与线性瓶颈.

- paper：MobileNetV2: Inverted Residuals and Linear Bottlenecks
- arXiv：[link](https://arxiv.org/abs/1801.04381)

# 1. Linear Bottleneck
**MobileNet**使用深度可分离卷积代替了普通卷积，减少了网络计算量和参数量，提高了网络运算速度。然而在实际训练中，深度卷积部分的卷积核训练后容易变成稀疏的，即大部分值为$0$。

![](https://pic.imgdb.cn/item/613b05bf44eaada739b9f09a.jpg)

作者认为，这一现象是激活函数**ReLU**导致的。为此作者进行了实验分析。对于二维空间中的螺旋线，通过随机矩阵$T$将其映射到高维空间，在高维空间中应用**ReLU**激活函数，再用逆矩阵$T^{-1}$恢复数据。实验结果表明，映射到较低维空间的特征应用**ReLU**后会损失较多信息；而对高维空间的特征应用**ReLU**后信息损失较小。

![](https://pic.imgdb.cn/item/613b07b644eaada739bd0140.jpg)

上述现象可以解释为什么深度可分离卷积中的深度卷积的卷积核学习效果较差。由于深度卷积的输入通常是上一层的输出经过**ReLU**的结果，导致信息损失。为此，作者将网络每一个模块最后一层逐点卷积后面的**ReLU**去掉（相当于应用线性激活函数），将其称作**Linear Bottleneck**。

# 2. Inverted Residual
**MobileNet**中的深度可分离卷积模块首先使用深度卷积，然而深度卷积不改变特征的通道数，导致经过深度卷积处理的特征维度较低，通过**ReLU**后仍然会产生信息损失。而**ReLU**能增加网络非线性，不可能全部去除，因此作者在深度卷积之前额外增加了逐点卷积(即$1\times 1$卷积)增加特征通道数。

![](https://pic.imgdb.cn/item/613b0c6b44eaada739c4dfae.jpg)

增加的这一层被称为**expansion layer**。改进后的深度可分离模块如上图所示。该模块首先对输入特征进行通道扩张(相当于解压缩)，然后使用深度卷积进行处理(相当于过滤)，最后对特征进行通道降维(相当于压缩)。额外引入了残差链接。

![](https://pic.imgdb.cn/item/613b0de144eaada739c70c43.jpg)

对比**ResNet**网络模块和**MobileNet V2**网络模块，都采用了$1\times 1 -> 3 \times 3 -> 1\times 1$卷积和残差连接的形式。然而，**ResNet**网络模块沿通道维度先降维再升维，而**MobileNet V2**网络模块沿通道维度先升维再降维。因此后者被称为**inverted residual**。

# 3. MobileNet V2
结合线性瓶颈和倒残差，**MobileNet V2**中的深度可分离卷积模块定义如下。注意每一次卷积后都使用了**Batch Norm**。

![](https://pic.imgdb.cn/item/613b0f7c44eaada739c96ce9.jpg)

**MobileNet V2**的结构如下。**MobileNet V2**具有$54$层，而**MobileNet**具有$28$层。尽管前者层数更多，但计算量和参数量更少，且表现更好。

![](https://pic.imgdb.cn/item/613b104844eaada739ca99bd.jpg)