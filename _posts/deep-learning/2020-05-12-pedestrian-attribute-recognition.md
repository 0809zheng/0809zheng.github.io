---
layout: post
title: '行人检测与属性识别(Pedestrian Detection and Attribute Recognition)'
date: 2020-05-12
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ed8a773c2a9a83be5131f5b.jpg'
tags: 深度学习
---

> Pedestrian Detection and Attribute Recognition.

# 1. 行人检测 Pedestrian Detection

**行人检测(Pedestrian Detection)**要解决的问题是：找出图像或视频帧中所有的行人，包括位置和大小，一般用矩形框表示，是典型的目标检测问题。

行人检测的主要困难：
- 外观差异大：外观差异主要体现在视角，姿态，服饰和附着物，光照，成像距离等。
- 遮挡问题：在很多应用场景中，行人非常密集，存在严重的遮挡，这对检测算法带来了严重的挑战。
- 背景复杂：无论是室内还是室外，行人检测一般面临的背景都非常复杂，有些物体的外观和形状、颜色、纹理很像人体。
- 检测速度：行人检测一般采用了复杂的模型，运算量相当大，要达到实时非常困难，一般需要大量的优化。

行人检测常用的数据集包括：

![](https://pic.downk.cc/item/5ed88636c2a9a83be5e5e0bc.jpg)

用于评估检测器性能的指标通常有：对数平均漏检率（**LAMR**）、每秒帧数(帧率）（**FPS**）、准确率（**AP**）和召回率（**recall**）。帧率表示模型的效率，准确率、对数平均漏检率和召回率反映了模型的准确性。

### ⚪ [DeepParts：Deep Learning Strong Parts for Pedestrian Detection](https://www.researchgate.net/publication/300412405_Deep_Learning_Strong_Parts_for_Pedestrian_Detection)

**DeepParts**基于多个局部检测器完成行人检测；使用对身体的局部检测来降低对行人检测的丢失率。进行局部检测的目的主要是为了处理行人身体被遮挡的问题。

算法流程：
1. 构造**part pool**；
2. 对于每一个**part**，训练一个单独的检测器；
3. 使用互补的检测器推断目标。

### ① Part Pool

![](https://pic.downk.cc/item/5ed8a28ec2a9a83be50c4851.jpg)

使用$2m×m$的网格定义一个人的全身，将身体的一部分（**Part prototype**）定义为$P=(x,y,w,h,i)$，$x,y$为中心点坐标，$w,h$分别为网格的宽高,i表示第几个**Part prototype**。

其中，为了避免可能有太细小的部分生成，使$w$和$h$的最小值为$2$。在具体的实现中定义了$45$个**Part prototype**。

### ② Part Detector
对于每个Part，都单独训练了卷积网络分类器。
- 负样本为对应区域不是行人的候选区域；
- 正样本是当**ground truth**中可见得部分能覆盖**Part prototype**时，提取对应部分为作为正样本。

如下图是**head-shoulder part**的分类器：

![](https://pic.downk.cc/item/5ed8a3aec2a9a83be50dc14b.jpg)

### ③ Bbox Shifting
检测得到的边界框存在漂移问题。

若在单方向上相对**ground truth**偏移$10\%$，就会使$IoU$为$0.9$，但是当在两个方向同时偏移$10\%$时，$IoU$就成了$0.68$，这就可能导致人体重要部分丢失，所以就要解决偏移问题。

![](https://pic.downk.cc/item/5ed8a459c2a9a83be50ec041.jpg)

在候选区域周围抖动的裁剪多个图像块，将这些块进行检测，选择得分最高的区域作为最终的结果。

为减少计算量，使用全卷积网络实现。将每个候选区域送给全卷积神经网络之前，重调它的大小为$(227+32n)×(227+32n)$，最后就可以得到对应于每个$227×227$区域的$(1+n)×(1+n)$个得分映射。取其中得分最大的区域作为候选区域。。

![](https://pic.downk.cc/item/5ed8a548c2a9a83be5100384.jpg)

# 2. 行人属性识别 Pedestrian Attribute Recognition

**行人属性识别（Pedestrian Attribute Recognition，PAR）**是从行人图像中挖掘属性信息，属性信息是行人的高层语义信息，将属性信息集成到行人检测、行人重识别等任务中可以获得更好的性能。

行人属性识别常用的数据集包括：
- **PETA**：19000张图像，分辨率从17 × 39 到 169 × 365，来自于8705个人，共61个二分类属性和4个多分类属性。
- **RAP**:   41585张图像，分辨率从36 × 92 到 344 × 554，共72个属性(69个二分类属性，3个多分类属性)
- **PA-100K**:  来自于598个室外监控视频，100000幅图像，分辨率50×100到758 × 454，目前最大的PAR数据集，共26个属性。

### ⚪ [DeepSAR & DeepMAR：Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios](https://www.researchgate.net/publication/283484818_Multi-attribute_Learning_for_Pedestrian_Attribute_Recognition_in_Surveillance_Scenarios)

**DeepSAR**是对行人的单一属性进行识别，**DeepMAR**是对行人的多项属性进行识别；即通过卷积网络后分别进行二分类和多个二分类：

![](https://pic.downk.cc/item/5ed8a876c2a9a83be51486e0.jpg)

### ⚪ [HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis](https://arxiv.org/abs/1709.09930)

**HP-Net**将注意力机制引入行人属性识别中。

网络分为**Main Net(M-Net)**和**Attentive Feature Net(AF-Net)**。网络的四条主路径是共享参数的。

![](https://pic.downk.cc/item/5ed8ad56c2a9a83be51be388.jpg)

在**AF-Net**中，分别对三个**Inception**模块使用了**multi-directional attention (MDA)**：

![](https://pic.downk.cc/item/5ed8ad68c2a9a83be51c09b2.jpg)