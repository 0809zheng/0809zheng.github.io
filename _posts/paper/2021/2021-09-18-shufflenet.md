---
layout: post
title: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'
date: 2021-09-18
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61710f222ab3f51d918ace5a.jpg'
tags: 论文阅读
---

> ShuffleNet: 使用组卷积与通道打乱构造高效网络.

- paper：ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
- arXiv：[link](https://arxiv.org/abs/1707.01083)

作者提出了**ShuffleNet**，一个针对移动端设备设计的高效卷积神经网络。**ShuffleNet**的设计中使用了组卷积与通道打乱。

**组卷积**(**group convolution**)是指对特征沿通道维度进行分组，每个卷积只作用于组内的特征，从而显著降低计算成本。

下图(a)展示了两层堆叠的组卷积，显然每个组的输出只与组内的特征有关，这限制了不同通道组之间的信息交流，削弱了模型的表示能力。

![](https://pic.imgdb.cn/item/617112152ab3f51d918d0497.jpg)

如果允许组卷积不同组的输入特征，则输入通道和输出通道将完全相关。作者提出了**通道打乱**(**channel shuffle**)操作，如上图(b,c)所示。可以先把每个组的通道划分成几个子组，然后在下一层为每个组划分不同的子组特征。

假设特征划分成$g$个组，每组具有$n$个特征，则输入特征的尺寸表示为$b\times gn \times h \times w$。首先将其调整为$b\times g\times n \times h \times w$，并进行通道置换得到$b\times n \times g \times h \times w$，再压缩维度得到打乱后的输出结果$b\times ng \times h \times w$。

作者从**ResNet**中的瓶颈模块(下图a)出发，设计了**ShuffleNet**的基本模块(下图b)。将$1\times 1$卷积替换为组卷积，并在第一个组卷积后使用通道打乱；深度卷积后没有使用激活函数。对于输入尺寸与输出尺寸不相等的情况(下图c)，在跳跃连接上使用平均池化调整尺寸，并将特征融合操作替换成连接操作。

![](https://pic.imgdb.cn/item/617112562ab3f51d918d3de7.jpg)

通过组卷积，网络可以处理具有更多通道数的特征，这为小型网络提供更多信息获取的机会。网络的深度卷积只作用于瓶颈层特征(通道数量为输入的$1/4$)，这是因为深度卷积在低功耗设备上实现困难，对内存的访问率更低。**ShuffleNet**网络的整体参数如下：

![](https://pic.imgdb.cn/item/6171205c2ab3f51d9197d471.jpg)

实验结果表明，在计算复杂度相当的情况下，**ShuffleNet**的表现超越了**MobileNet**。

![](https://pic.imgdb.cn/item/617122032ab3f51d91991ddf.jpg)