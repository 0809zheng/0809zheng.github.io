---
layout: post
title: 'Training Deeper Convolutional Networks with Deep Supervision'
date: 2021-08-27
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6126273144eaada73944f2cb.jpg'
tags: 论文阅读
---

> 使用深度监督训练更深的卷积网络.

- paper：Training Deeper Convolutional Networks with Deep Supervision
- arXiv：[link](https://arxiv.org/abs/1505.02496v1)

通常而言，增加神经网络的深度能够提高网络的特征表示能力。但是增加深度会导致梯度消失和爆炸等现象，使得神经网络的训练变得困难。深度监督通过为神经网络的中间层增加辅助的分类器来减缓训练困难，其中的辅助分类器能够判断隐藏层特征的判别性好坏。

本文主要讨论了辅助分类器应该加到网络的什么位置上，作者通过实验展开讨论。首先使用标准的输出监督训练一个$8$层的卷积网络，绘制出迭代过程中中间层的平均梯度数值变化。辅助分支应该添加到平均梯度消失(小于$10^{-7}$)的层上。如下图所示，**conv1-4**层出现了梯度消失现象。

![](https://pic.imgdb.cn/item/61274a5144eaada7397f65c8.jpg)

下图是在第$4$层增加深度监督的例子。通过引入额外的分类分支构造了辅助损失函数。

![](https://pic.imgdb.cn/item/61274a2b44eaada7397f0256.jpg)

若记$$W=\{W_1,...,W_{11}\}$$为主干网络权重，$$W_s=\{W_{s5},...,W_{s8}\}$$为分支网络权重。则输出层的**softmax**分类结果表示为：

$$ p_k=\frac{\exp{(X_{11(k)})}}{\sum_{k}^{}\exp{(X_{11(k)})}} $$

主干网络的损失函数为：

$$ \mathcal{L}_0(W)=-\sum_{k=1}^{K}y_k \ln p_k $$

第$4$层的深度监督分支的**softmax**分类结果表示为：

$$ p_{sk}=\frac{\exp{(S_{8(k)})}}{\sum_{k}^{}\exp{(S_{8(k)})}} $$

深度监督分支的损失函数为：

$$ \mathcal{L}_s(W,W_s)=-\sum_{k=1}^{K}y_k \ln p_{sk} $$

则总损失函数表示为：

$$ \mathcal{L}(W,W_s)=\mathcal{L}_0(W)+\alpha_t\mathcal{L}_s(W,W_s) $$

其中深度监督分支的损失随训练轮数$t$衰减：

$$ \alpha_t \gets \alpha_t * (1-\frac{t}{N}) $$

实验观察到在**conv2-4**层增加深度监督能够有效地缓解梯度消失问题：

![](https://pic.imgdb.cn/item/61274a8f44eaada739803e04.jpg)

也可以在单个网络的多个隐藏层增加深度监督。下图表示在一个$16$层的卷积网络的第$4,7,10$个隐藏层上增加深度监督：

![](https://pic.imgdb.cn/item/61274a3b44eaada7397f2a39.jpg)

