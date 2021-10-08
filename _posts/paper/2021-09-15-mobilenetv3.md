---
layout: post
title: 'Searching for MobileNetV3'
date: 2021-09-15
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/613b11fe44eaada739cd1a26.jpg'
tags: 论文阅读
---

> 使用神经结构搜索寻找MobileNet V3.

- paper：Searching for MobileNetV3
- arXiv：[link](https://arxiv.org/abs/1905.02244)

**MobileNet V3**网络的结构是通过神经结构搜索确定的。相较于**MobileNet V2**，**MobileNet V3**网络的模块中额外引入了通道注意力机制(**SENet**)。

![](https://pic.imgdb.cn/item/613b1f0144eaada739dfa032.jpg)

作者使用资源受限(**platform-aware**)的**NAS**与**NetAdapt**搜索了两个不同大小的网络，分别称为**Large**和**Small**。

![](https://pic.imgdb.cn/item/613b1a9544eaada739d98765.jpg)

资源受限的**NAS**是在限制计算量和参数量的前提下搜索网络模块的结构，是一种模块级搜索(**block-wise search**)。**NetAdapt**搜索用于模块确定后对网络层的参数(如卷积核数量)进行微调，是一种层级搜索(**layer-wise search**)。

在此基础上，**MobileNet V3**使用了**HardSwish**激活函数与改进的输出层。

**Swish**激活函数表达式为$\text{Swish}(x)=x \cdot \sigma (x)$，实验表明在深层模型中，**Swish**激活函数的表现超过**ReLU**。然而**Swish**中的指数运算在嵌入式环境中成本较高，为此作者提出了一种**hard Swish**激活函数，近似代替**Swish**。

**hard Swish**激活函数的表达式如下：

$$ \text{HardSwish} = x \cdot \frac{\text{ReLU6}(x+3)}{6} = \begin{cases} x , & x \geq 3 \\ \frac{x(x+3)}{6} , & -3 \leq x <3 \\ 0, & x < -3 \end{cases} $$

![](https://pic.imgdb.cn/item/613b22bd44eaada739e48d53.jpg)

作者通过实验发现**Swish**激活函数在更深的层中表现更好，因此只在深层网络中应用**hard Swish**。

作者发现原有输出层的计算代价比较大，于是对输出层进行了修改。具体地，将最后一个深度可分离卷积模块与输出卷积层合并，将平均池化提前以减少特征尺寸，并删除了不改变特征维度的$3\times 3$深度卷积层。

![](https://pic.imgdb.cn/item/613b248544eaada739e6f5a5.jpg)

实验表明，在相同的计算量时，**MobileNet V3**取得最好的表现。

![](https://pic.imgdb.cn/item/613b25a544eaada739e869f2.jpg)