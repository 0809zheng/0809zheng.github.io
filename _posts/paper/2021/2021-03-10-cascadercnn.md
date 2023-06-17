---
layout: post
title: 'Cascade R-CNN: Delving into High Quality Object Detection'
date: 2021-03-10
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/648a68801ddac507cc863c31.jpg'
tags: 论文阅读
---

> Cascade R-CNN：高质量目标检测研究.

- paper：[Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

在[<font color=blue>Faster R-CNN</font>](https://0809zheng.github.io/2021/03/09/fasterrcnn.html)等两阶段目标检测网络中，会在两个位置设置**交并比阈值(IoU Threshold)**：在训练时为**RPN**网络设置**IoU**阈值，区分**proposal**是否包含目标（**positive/negative**）；以及在推理时为预测边界框设置**IoU**阈值，区分**TP**和**FP**样本以计算**mAP**。
- 在训练阶段，**RPN**网络从所有**anchor**中提取约$2000$个置信度较高的**proposal**，这些**proposal**被送入**Faster R-CNN**结构中。在**Faster R-CNN**结构中，首先计算每个**proposal**和**Ground Truth**之间的**IoU**，通过人为设定一个**IoU**阈值（通常为$0.5$），把这些**Proposals**分为正样本（前景）和负样本（背景），并对正负样本采样（比例尽量满足$1:3$，总数量通常为$128$）。这些**proposals**被送入到**Roi Pooling**后进行类别分类和边界框回归。
- 在推理阶段，**RPN**网络从所有**anchor**中提取约$300$个置信度较高的**proposal**，这些**proposal**被送入**Faster R-CNN**结构中，直接通过**Roi Pooling**后进行类别分类和边界框回归。

## 1. RPN中的回归不匹配问题

在为**RPN**网络设置**IoU**阈值时，只有当**proposal**边界框和真实目标框的**IoU**超过该阈值时，才认为**proposal**检测到了目标，对这些目标进行边界框回归。通常该阈值设置越高，生成的**proposal**越准确；但是由于提高了阈值，导致正样本的数量呈指数级降低，容易过拟合。而在网络预测时，由于真实目标框是未知的，因此所有**proposal**都被视为正样本用于边界框回归。这导致了在训练和测试阶段中，**RPN**网络边界框回归的不匹配问题：训练阶段的输入**proposal**质量更高，测试阶段的输入**proposal**质量相对较差。

为了提高检测的精度，需要产生更高质量的**proposal**，因此可以考虑在训练时增大**RPN**网络设置的**IoU**阈值。但是直接提高**IoU**阈值会产生以下问题：
- 提高**IoU**阈值会导致满足阈值条件的**proposals**减少，容易导致过拟合。
- 导致在训练和测试阶段中，**RPN**网络边界框回归的不匹配问题。

下图分别表示**RPN**网络输出**proposal**的**IoU**分布以及改变**IoU**阈值对回归和检测精度的影响。

![](https://pic.imgdb.cn/item/648a6b091ddac507cc894f28.jpg)


## 2. Cascade R-CNN

上述分析表明单一**IoU**阈值训练出的检测器精度有限，不能对所有的**proposals**进行很好的优化。本文作者设计了**Cascade R-CNN**，通过多阶段结构串联网络，使用不同**IoU**阈值训练多个级联的检测器。具体地，**Cascade R-CNN**使用了四个网络，第一个网络提取特征映射，之后分别使用阈值为$0.5$、$0.6$、$0.7$的检测网络，通过串联的学习获得较高的目标检测精度。

![](https://pic.imgdb.cn/item/648a6d5a1ddac507cc8cd09c.jpg)

下图给出了在三个阶段，边界框回归的样本分布变化。每经过一次回归，样本都更靠近真实边界框一些，质量也就更高一些，样本的分布也在逐渐变化。每个阶段设置不同的**IoU**阈值，可以更好地去除离群点（红色点），适应新的**proposal**分布。

![](https://pic.imgdb.cn/item/648a6f331ddac507cc8f53a0.jpg)

下图给出了在三个阶段，所提**proposal**的**IoU**分布。每个阶段高阈值**proposals**数量逐渐增加，保证具有足够的正样本，不会容易过拟合。

![](https://pic.imgdb.cn/item/648a710a1ddac507cc924aaa.jpg)