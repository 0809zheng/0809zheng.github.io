---
layout: post
title: '目标检测'
date: 2020-05-31
author: 郑之杰
cover: 'https://img2018.cnblogs.com/blog/1423648/201903/1423648-20190316200041489-44690273.png'
tags: 深度学习
---

> Object Detection.

**目标检测(Object Detection)**问题就是在图像中检测出可能存在的目标，包括**定位**和**分类**两个子任务。定位是指确定目标在图像中的具体位置，分类是确定目标的具体类别。

传统的目标检测算法主要有三个步骤，
1. 在图像中生成候选的区域，通常是由滑动窗口实现的；结合不同尺度的图像以及不同尺度的窗口生成大量的候选区域;
2. 对每个候选区域提取特征向量，这一步通常是用人工精心设计的特征描述子提取图像的特征;
3. 对每个候选区域提取的特征进行分类，确定对应的类别。

在传统的方法中，经常会使用集成、串联学习、梯度提升等方法来提高目标检测的准确率。但是传统的方法逐渐暴露出很多问题。近些年来深度学习的引入使得目标检测的精度和速度有了很大的提升，卷积神经网络能够提取图像的深层语义特征，省去了人工设计和提取特征的步骤。

目前主流的目标检测算法分成两类。**两阶段（Two-Stage）**的目标检测算法首先在图像中生成可能存在目标的候选区域，然后对这些候选区域进行预测。这些方法精度高，速度相对慢一些；**单阶段（One-Stage）**的目标检测算法把图像中的每一个位置看作潜在的候选区域，直接进行预测。这些方法速度快，精度相对低一些。

![](https://pic.downk.cc/item/5facf3b81cd1bbb86b4e1145.jpg)

上图是目前大部分目标检测算法的主要流程图。一个目标检测系统主要分成三部分，如图中的**backbone**、**neck**和**regression**部分。
1. **backbone**部分通常是一个卷积网络，把图像转化成对应的**特征映射(feature map)**；
2. **neck**部分通常是对特征映射做进一步的处理；
3. **regression**部分通常把提取的特征映射转换成**边界框(bounding box)**和**类别(class)**信息。

单阶段的目标检测算法直接在最后的特征映射上进行预测；而两阶段的方法先在特征映射上生成若干候选区域，再对候选区域进行预测。由此可以看出单阶段的方法所处理的候选区域是**密集**的，而两阶段的方法由于预先筛选了候选区域，最终处理的候选区域相对来说是**稀疏**的。

# 一些主流的两阶段目标检测算法

### 1. R-CNN

![](https://pic.downk.cc/item/5facf5e01cd1bbb86b4eb930.jpg)

**R-CNN**是深度学习应用于目标检测领域最早的模型之一。

该网络首先使用传统的区域**选择搜索(Selective Search)**算法在图像中提取大约$2000$个**候选区域(Region Proposal)**，对每一个候选区域通过卷积网络提取一个特征映射，把特征映射通过各向异性的放缩变换到预定的尺寸，再通过预测网络得到一个特征向量，之后使用支持向量机进行二分类，即获得每一个类别的可能性。为进一步提高精度，训练一个线性模型对目标框矩形位置进行微调，减小标注框与预测框的中心位置偏移和长宽尺寸比例。

该方法在选择候选区域时耗时长，对每个候选区域都要进行一次卷积网络的前向计算，对内存要求大。

### 2. SPPNet

![](https://pic.downk.cc/item/5facf6701cd1bbb86b4ef017.jpg)

由于预测的卷积网络需要固定尺寸的输入，**R-CNN**处理的方法是直接对特征映射进行各向异性的缩放，这一过程会引入失真和畸变，损失了图像的部分文本信息。

**SPPNet**提出了一种**空间金字塔池化(Spatial Pyramid Pooling)**层，对于任意大小的输入尺寸，均可形成相同长度的输出特征向量，且这个长度是预先定义的，与输入图像的尺寸无关。当给定一个输入特征映射时，把这个映射划分成$s×s$个子区域，如果不能整除就做近似。对每一个子区域应用最大池化可以得到一个标量，那么最终就能把输入转化成一个向量。当选择的$s$不同时，所得向量的长度也不同。在这篇文章中$s$取$1$、$2$、$4$，最终得到$21$维的向量。

### 3. Fast RCNN

![](https://pic.downk.cc/item/5facf6ff1cd1bbb86b4f1871.jpg)

**Fast R-CNN**是对**R-CNN**的改进，在原始的**R-CNN**网络中，先对图像提取候选区域，再把每个候选区域喂入卷积网络；而**Fast R-CNN**是先将图像图像喂入卷积网络得到特征映射，再在原始图像中生成候选区域，通过投影得到候选区域在特征映射上的位置，这样就减少了运算量。

**Fast R-CNN**还把特征的各向异性放缩换成了**RoI Pooling**，这可以看作一个特殊的空间金字塔池化层，即划分子区域的参数$s$只选一个特定的值，原文中选择$s=7$，即把候选区域的特征映射转换成$7×7$的子区域。该模型还把用于目标分类的分类损失和用于目标边界框坐标修正的回归损失结合起来一起训练，减小了训练复杂度。

### 4. Faster RCNN

![](https://pic.downk.cc/item/5facf7c81cd1bbb86b4f54e0.jpg)

**Faster R-CNN**是对**Fast R-CNN**的改进，主要是对于候选区域生成方法的改进。

之前的方法都是采用传统的选择搜索算法获得候选区域，这种方法速度慢，并且不能和后面的网络一起优化，从而不能得到整个系统的全局最优解。**Faster R-CNN**提出了**区域提议网络(Region Proposal Network, RPN)**，并引入了**anchor**的概念。

**RPN**网络的思想如上图右所示，将输入图像通过**backbone**卷积网络得到特征映射后，采用一个$3×3$的滑动窗口在特征映射进行移动。在窗口的每一个位置，相当于特征映射上一个个$3×3$的子区域，假设这个区域对应的原始图像位置上存在目标，并设置了$k$个**anchor**作为先验知识，即预先选定一些尺寸的边界框，用来检测这些目标。对该子区域提取特征向量后分别喂入一个分类和回归网络，分类网络用来判断这些边界框是否真的有目标，回归网络用来对这些人为给定的边界框尺寸进行修正。

### 5. FPN

![](https://pic.downk.cc/item/5facf8ae1cd1bbb86b4f97df.jpg)

之前的几个网络都是将图像喂入网络得到对应的特征向量。卷积网络的浅层映射特征通常具有较强的空间信息，据具有较高的分辨率，适合检测小目标；而深层的特征映射具有较强的语义信息，具有范围较大的感受野，适合检测较大的目标。

**特征金字塔网络(Feature Pyramid Network, FPN)**就是将卷积网络的浅层映射和深层映射结合起来，首先通过前向传播得到由浅入深的特征映射，再将深层的特征映射通过转置卷积增加尺寸，并结合每一层特征映射的信息，最终可以得到一个空间信息和语义信息都很丰富的特征映射。将该特征映射用于下游的任务，最终能够提高目标检测的准确率。

### 6. Cascade R-CNN

![](https://pic.downk.cc/item/5facf96f1cd1bbb86b4fcd81.jpg)

网络在检测目标时会设置一个**交并比阈值(IoU Threshold)**，当预测的边界框和真实目标框的交并比超过该阈值时，才认为边界框检测到了目标。通常该阈值设置越高，能够检测出的边界框越准确，但是由于提高了阈值，导致正样本的数量呈指数级降低，容易过拟合。在预测时，该阈值选取得不同会导致候选区域的样本分布发生变化，从而影响最终的结果。

**Cascade R-CNN**是一种串联的两阶段目标检测算法，串联网络的思想是使用不同交并比的阈值训练多个级联的检测器，在原文中作者使用了四个网络，第一个网络提取特征映射，之后分别使用阈值为$0.5$、$0.6$、$0.7$的检测网络，通过这样一种串联的学习获得了较高的目标检测精度。

# 一些主流的单阶段目标检测算法

### 1. YOLO系列

![](https://pic.downk.cc/item/5facfa001cd1bbb86b50019d.jpg)

**YOLO**算法可能是目前最知名的单阶段目标检测算法，它的基本思想是用卷积网络实现滑动窗口。

当一张图像喂入卷积网络后，可以得到尺寸缩小的特征映射，比如$7×7$的映射。映射的每一个子区域都能对应到原图像中的一个子区域，假设原图像的这个子区域内含有目标，则通过网络把相关信息编码到特征映射的对应区域上。

在原始的网络中，每一个子区域预设一些边界框用来检测该区域可能出现的目标，由此可以看出，单阶段的检测方法在每个子区域都会预测很多边界框，因此所处理的候选区域是非常密集的，所以会出现大量的负样本，造成目标检测中正负样本的比例极其不均衡，这也是影响单阶段目标检测算法的主要问题。

**YOLOv2**相比于**YOLO**进行了很多改进，包括引入了**BatchNorm**、新的网络结构和**anchor**机制。作者还引入了许多提高检测精度的训练和测试方法。后来的**YOLOv3**等也主要是使用了大量同时期的模型训练技巧，通过对照和消融实现选择能够最大程度提升检测性能的一些方法。

### 2. SSD

![](https://pic.downk.cc/item/5facfa9c1cd1bbb86b503ae1.jpg)

**SSD**网络也是一种单阶段的目标检测器。之前提到卷积网络的不同层次的特征映射具有不同的空间和语义信息，**SSD**网络考虑使用网络中的多层特征映射，在每一层映射上设置不同尺寸的**anchor**用来检测不同尺度的目标，取得了很好的检测效果。


# 目标检测领域的论文清单

## 综述 Survey

### Deep Learning for Generic Object Detection: A Survey
- arXiv:[https://arxiv.org/abs/1809.02165](https://arxiv.org/abs/1809.02165)

### Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks
- arXiv:[https://arxiv.org/abs/1809.03193](https://arxiv.org/abs/1809.03193)

### Object Detection in 20 Years: A Survey
- arXiv:[https://arxiv.org/abs/1905.05055](https://arxiv.org/abs/1905.05055)

### A Survey of Deep Learning-based Object Detection
- arXiv:[https://arxiv.org/abs/1907.09408](https://arxiv.org/abs/1907.09408)

### Recent Advances in Deep Learning for Object Detection
- arXiv:[https://arxiv.org/abs/1908.03673](https://arxiv.org/abs/1908.03673)

### Imbalance Problems in Object Detection: A Review
- arXiv:[https://arxiv.org/abs/1909.00169](https://arxiv.org/abs/1909.00169)

## 传统检测方法 Traditional CV Methods

### Selective Search for Object Recognition
- intro:Selective Search
- paper:[https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)

### Discriminatively Trained Deformable Part Models
- intro:DPM
- paper:[http://www.computervisiononline.com/software/discriminatively-trained-deformable-part-models](http://www.computervisiononline.com/software/discriminatively-trained-deformable-part-models)

## 两阶段的检测器 Two-Stage Detectors

### Rich feature hierarchies for accurate object detection and semantic segmentation
- intro:R-CNN
- arXiv:[http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- github(official):[https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)

### Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
- intro:SPP-net
- arXiv:[http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github(official):[https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)

### Fast R-CNN
- intro:Fast R-CNN
- arXiv:[http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- github(official):[https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- intro:Faster R-CNN
- arXiv:[http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- github(official, Matlab):[https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)

### R-FCN: Object Detection via Region-based Fully Convolutional Networks
- intro:R-FCN
- arXiv:[http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)

### Feature Pyramid Networks for Object Detection
- intro:FPN
- arXiv:[https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)

### Mask R-CNN
- intro:Mask R-CNN
- arXiv:[http://arxiv.org/abs/1703.06870](http://arxiv.org/abs/1703.06870)

### A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection
- intro:
- arXiv:[https://arxiv.org/abs/1704.03414](https://arxiv.org/abs/1704.03414)

### Light-Head R-CNN: In Defense of Two-Stage Object Detector
- intro:Light-Head R-CNN
- arXiv:[https://arxiv.org/abs/1711.07264](https://arxiv.org/abs/1711.07264)
- github(official):[https://github.com/zengarden/light_head_rcnn](https://github.com/zengarden/light_head_rcnn)

### Cascade R-CNN: Delving into High Quality Object Detection
- intro:Cascade R-CNN
- arXiv:[https://arxiv.org/abs/1712.00726](https://arxiv.org/abs/1712.00726)

## 单阶段的检测器 One-Stage Detectors

### Scalable Object Detection using Deep Neural Networks
- intro:MultiBox
- arXiv:[https://arxiv.org/abs/1312.2249](https://arxiv.org/abs/1312.2249)

### OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks
- intro:OverFeat
- arXiv:[https://arxiv.org/abs/1312.6229](https://arxiv.org/abs/1312.6229)

### You Only Look Once: Unified, Real-Time Object Detection
- intro:YOLO
- arXiv:[http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code:[https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)

### SSD: Single Shot MultiBox Detector
- intro:SSD
- arXiv:[http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- github(official):[https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)

### YOLO9000: Better, Faster, Stronger
- intro:YOLOv2
- arXiv:[https://arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- code:[https://pjreddie.com/darknet/yolov2/](https://pjreddie.com/darknet/yolov2/)

### DSSD : Deconvolutional Single Shot Detector
- intro:DSSD
- arXiv:[https://arxiv.org/abs/1701.06659](https://arxiv.org/abs/1701.06659)

### DSOD: Learning Deeply Supervised Object Detectors from Scratch
- intro:DSOD
- arXiv:[https://arxiv.org/abs/1708.01241](https://arxiv.org/abs/1708.01241)

### Focal Loss for Dense Object Detection
- intro:RetinaNet
- arXiv:[https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

### Single-Shot Refinement Neural Network for Object Detection
- intro:RefineNet
- arXiv:[https://arxiv.org/abs/1711.06897](https://arxiv.org/abs/1711.06897)

### MegDet: A Large Mini-Batch Object Detector
- intro:MegDet
- arXiv:[https://arxiv.org/abs/1711.07240](https://arxiv.org/abs/1711.07240)

### FSSD: Feature Fusion Single Shot Multibox Detector
- intro:FSSD
- arXiv:[https://arxiv.org/abs/1712.00960](https://arxiv.org/abs/1712.00960)

### Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network
- intro:ESSD
- arXiv:[https://arxiv.org/abs/1801.05918](https://arxiv.org/abs/1801.05918)

### YOLOv3: An Incremental Improvement
- intro:YOLOv3
- arXiv:[https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- github(official):[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

### DetNet: A Backbone network for Object Detection
- intro:DetNet
- arXiv:[https://arxiv.org/abs/1804.06215](https://arxiv.org/abs/1804.06215)

### Pelee: A Real-Time Object Detection System on Mobile Devices
- intro:Pelee
- arXiv:[https://arxiv.org/abs/1804.06882](https://arxiv.org/abs/1804.06882)

### MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects
- intro:MDSSD
- arXiv:[https://arxiv.org/abs/1805.07009](https://arxiv.org/abs/1805.07009)

### You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery
- intro:YOLT
- arXiv:[https://arxiv.org/abs/1805.09512](https://arxiv.org/abs/1805.09512)

### Fire SSD: Wide Fire Modules based Single Shot Detector on Edge Device
- intro:Fire SSD
- arXiv:[https://arxiv.org/abs/1806.05363](https://arxiv.org/abs/1806.05363)

### CornerNet: Detecting Objects as Paired Keypoints
- intro:CornerNet
- arXiv:[https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244)

### M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network
- intro:M2Det
- arXiv:[https://arxiv.org/abs/1811.04533](https://arxiv.org/abs/1811.04533)

### EfficientDet: Scalable and Efficient Object Detection
- intro:EfficientDet
- arXiv:[https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)
