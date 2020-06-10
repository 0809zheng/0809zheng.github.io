---
layout: post
title: '行人属性识别'
date: 2020-05-12
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ed8a773c2a9a83be5131f5b.jpg'
tags: 深度学习
---

> Pedestrian Attribute Recognition.

**行人属性识别（Pedestrian Attribute Recognition，PAR）**是从行人图像中挖掘属性信息，属性信息是行人的高层语义信息，将属性信息集成到行人检测、行人重识别等任务中可以获得更好的性能。

本文目录：
1. Benchmarks
2. DeepSAR & DeepMAR
3. HydraPlus-Net

# 1. Benchmarks
行人属性识别常用的数据集包括：
- **PETA**：19000张图像，分辨率从17 × 39 到 169 × 365，来自于8705个人，共61个二分类属性和4个多分类属性。
- **RAP**:   41585张图像，分辨率从36 × 92 到 344 × 554，共72个属性(69个二分类属性，3个多分类属性)
- **PA-100K**:  来自于598个室外监控视频，100000幅图像，分辨率50×100到758 × 454，目前最大的PAR数据集，共26个属性。


# 2. DeepSAR & DeepMAR
- paper：[Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios](https://www.researchgate.net/publication/283484818_Multi-attribute_Learning_for_Pedestrian_Attribute_Recognition_in_Surveillance_Scenarios)

**DeepSAR**是对行人的单一属性进行识别，**DeepMAR**是对行人的多项属性进行识别；即通过卷积网络后分别进行二分类和多个二分类：

![](https://pic.downk.cc/item/5ed8a876c2a9a83be51486e0.jpg)

# 3. HydraPlus-Net
- paper：[HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis](https://arxiv.org/abs/1709.09930)

**HP-Net**将注意力机制引入行人属性识别中。

网络分为**Main Net(M-Net)**和**Attentive Feature Net(AF-Net)**。网络的四条主路径是共享参数的。

![](https://pic.downk.cc/item/5ed8ad56c2a9a83be51be388.jpg)

在$AF-Net$中，分别对三个$Inception$模块使用了**multi-directional attention (MDA)**：

![](https://pic.downk.cc/item/5ed8ad68c2a9a83be51c09b2.jpg)