---
layout: post
title: 'Deep image reconstruction from human brain activity'
date: 2020-07-10
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f07e3b814195aa594be2121.jpg'
tags: 论文阅读
---

> 从人类大脑活动中重构图像.

- paper：Deep image reconstruction from human brain activity
- arXiv：[link](https://www.biorxiv.org/content/biorxiv/early/2017/12/30/240317.full.pdf)

作者设计了一种根据人脑活动重构图像的方法。

![](https://pic.downk.cc/item/5f07e65814195aa594bed0cc.jpg)

- 作者首先训练了一个特征解码器，将图像分别通过人的观察和喂入深度神经网络，将人脑中的**fMRI activity**信号和网络中的特征建立联系；
- 对于一张给定的或想象中的图像，通过人脑读取**fMRI activity**信号，使用训练好的特征解码器将其映射为深度神经网络中的特征，通过迭代优化输入网络的图像使其特征接近解码特征，便可以重构人脑中的图像；
- 为使重构图像更接近自然图像，作者引入了一个深度生成网络作为先验。

随着迭代次数的增加，模型可以重构出人类所“看到”的图像：

![](https://pic.downk.cc/item/5f07e93714195aa594bf8d3c.jpg)

下面是针对深度生成网络的对比试验，使用深度生成网络的实验结果更自然；尽管像素相似度略低，但是人类更容易分辨：

![](https://pic.downk.cc/item/5f07e8f814195aa594bf7c9d.jpg)

作者通过实验发现，使用更多的神经网络中间层特征，能够重构出更好的效果：

![](https://pic.downk.cc/item/5f07e97614195aa594bf9e0a.jpg)

对人类观察到的非自然图像（如符号的图片），也可以进行重构：

![](https://pic.downk.cc/item/5f07e9d714195aa594bfb834.jpg)

进一步让人类想象这些符号图片，也可以进行重构：

![](https://pic.downk.cc/item/5f07ea1814195aa594bfca0f.jpg)