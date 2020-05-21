---
layout: post
title: '图像分类'
date: 2020-05-06
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eba62edc2a9a83be59d07b6.jpg'
tags: 深度学习
---

> Image Classifications.

**图像分类**是**计算机视觉**的基本任务，旨在根据每张图像内出现的物体进行分类。

用计算机求解图像分类的主要问题是**语义鸿沟（semantic gap）**，表现在：
- 不同图像低级的视觉特征**low-level visual features**相似，但高级的语义**high-level concepts**差别很大；
- 不同图像低级的视觉特征**low-level visual features**差距很大，但高级的语义**high-level concepts**相同。

传统的图像处理是先对图像手工提取特征，再根据特征对图像进行分类。

深度学习的方法不需要手工提取特征，使用卷积神经网络自动提取特征并分类。

**本文目录**：
1. 图像分类数据集
2. 应用于图像分类的卷积神经网络

# 1. 图像分类数据集
常见的图像分类数据集：
- MNIST
- CIFAR
- Places2
- Cats vs Dogs
- ImageNet
- PASCAL VOC

### (1)MNIST（Mixed National Institute of Standards and Technology）
[MNIST](http://yann.lecun.com/exdb/mnist/)手写数字识别数据集，由纽约大学的Yann LeCun整理。

$MNIST$训练集包含$60000$张图像，测试集包含$10000$张图像，每张图像都进行了尺度归一化和数字居中处理，固定尺寸大小为$28×28$。

![](https://pic.downk.cc/item/5eba522dc2a9a83be58c565d.jpg)

### (2)CIFAR（Canada Institute For Advanced Research）
[CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)是由加拿大先进技术研究院的AlexKrizhevsky, Vinod Nair和Geoffrey Hinton收集而成的小图片数据集，包含$CIFAR-10$和$CIFAR-100$两个数据集。

- $CIFAR-10$：由$60000$张$32*32$的$RGB$彩色图片构成，共$10$个分类。包含$50000$张训练图像，$10000$张测试图像。
![](https://pic.downk.cc/item/5eba5322c2a9a83be58d5328.jpg)

- $CIFAR-100$：由$60000$张图像构成，包含$100$个类别，每个类别$600$张图像，其中$500$张用于训练，$100$张用于测试。其中这$100$个类别又组成了$20$个大的类别，每个图像包含小类别和大类别两个标签。
![](https://pic.downk.cc/item/5eba5344c2a9a83be58d75ff.jpg)

### (3)Places2
- [paper](http://places2.csail.mit.edu/PAMI_places.pdf)

[Places2](http://places2.csail.mit.edu/index.html)是由MIT开发的一个场景图像数据集，可用于以场景和环境为应用内容的视觉认知任务。

包含1千万张图片，400多个不同类型的场景环境，每一类有5000-30000张图片。

![](https://pic.downk.cc/item/5eba53ecc2a9a83be58e4228.jpg)

### (4)Cats vs Dogs
猫狗分类数据集，共25000张图片，猫、狗各12500张。
- 下载链接：[kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

### (5)ImageNet
[ImageNet](http://www.image-net.org/)是斯坦福的计算机科学家李飞飞建立的，目前世界上图像识别最大的数据库，目前已经包含14197122张图像，21841个类别。

举办多届**ILSVRC（ImageNet Large-Scale Visual Recognition Challenge）**比赛。

使用的数据集是$ImageNet$的一个子集，总共有$1000$类，每类大约有$1000$张图像。具体地，有大约$1.2$ million的训练集，5万验证集，15万测试集。

### (6)PASCAL VOC
[PASCAL VOC](http://pjreddie.com/projects/pascal-voc-dataset-mirror/)数据集是视觉对象的分类识别和检测的一个基准测试集，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。

图像包含$VOC2007$（$430M$），$VOC2012$（$1.9G$）两个下载版本。


# 2. 应用于图像分类的卷积神经网络
见[卷积神经网络](https://0809zheng.github.io/2020/01/01/CNN-paper.html)