---
layout: post
title: 'Region Proposal by Guided Anchoring'
date: 2021-06-18
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/65388472c458853aef09c07d.jpg'
tags: 论文阅读
---

> 通过Anchor引导实现区域提议.

- paper：[Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278)

**anchor-based**的目标检测方法需要预设**anchor**，**anchor**设置的好坏对结果影响很大，因为**anchor**本身不会改变，所有的预测值都是基于**anchor**进行回归；一旦**anchor**设置不太好，那么效果肯定影响很大。

不管是**one stage**还是**two-stage**，都是基于语义信息来预测的，在**bbox**内部的区域激活值较大，这种语义信息正好可以指导**anchor**的生成。本文通过图像特征来指导 **anchor** 的生成。通过预测 **anchor** 的位置和形状，来生成稀疏而且形状任意的动态**anchor**。

作者采用了一种可以直接插入当前**anchor-based**网络中进行**anchor**动态调整的做法，而不是替换掉原始网络结构。以**retinanet**为例，在预测**xywh**的同时，新增两条预测分支，一条分支是**loc (batch,anchor_num * 1,h,w)**，用于区分前后景，目标是预测哪些区域应该作为中心点来生成 **anchor**，是二分类问题；另一条分支是**shape (batch,anchor_num * 2,h,w)**, 用于预测**anchor**的形状。

![](https://pic.imgdb.cn/item/65378c74c458853aef90bc41.jpg)

一旦训练完成，学习到的**anchor**会和语义特征紧密联系：

![](https://pic.imgdb.cn/item/65378c9dc458853aef9119c6.jpg)

**Guided Anchoring**的测试流程为：
- 对于任何一层特征层，都会输出**4**条分支，分别是**anchor**的**loc_preds**，**anchor**的**shape_preds**，原始**retinanet**分支的**cls_scores**和**bbox_preds**
- 使用阈值将**loc_preds**预测值切分出前景区域，然后提取前景区域的**shape_preds**，然后结合特征图位置，**concat**得到**4**维的**guided_anchors** $(x,y,w,h)$
- 此时的**guided_anchors**就相当于**retinanet**里面的固定**anchor**了，然后和原始**retinanet**流程完全相同，基于**guided_anchors**和**cls_scores**、**bbox_preds**分支就可以得到最终的**bbox**预测值。

**anchor**的定位模块**loc_preds**是个二分类问题，希望学习出前景区域，采用**focal loss**进行训练。这个分支的设定和大部分**anchor-free**的做法一致：
- 首先对每个**gt**,利用**FPN**中提到的**roi**重映射规则，将**gt**映射到不同的特征图层上
- 定义中心区域和忽略区域比例，将**gt**落在中心区域的位置认为是正样本，忽略区域是忽略样本(模糊样本)，其余区域是背景负样本

![](https://pic.imgdb.cn/item/65378dc6c458853aef93bbf2.jpg)

**anchor**的形状预测模块**loc_shape**的目标是给定 **anchor** 中心点，预测最佳的长和宽，这是一个回归问题。
- 如何确定特征图的哪些位置是正样本区域？采用**ApproxMaxIoUAssigner**，其核心思想是：利用原始**retinanet**的每个位置**9**个**anchor**设定，计算**9**个**anchor**和**gt**的**iou**，然后选出每个位置**9**个**iou**中最高的**iou**值，利用该**iou**值计算后续的**MaxIoUAssigner**，此时就可以得到每个特征图位置上哪些位置是正样本了。
- 正样本位置对应的**shape label**是什么？得到每个位置匹配的**gt**，那么对应的**target**肯定就是**Gt**值了。该分支的**loss**是**bounded iou loss**：

$$
\mathcal{L}_{\text {shape }}=\mathcal{L}_1\left(1-\min \left(\frac{w}{w_g}, \frac{w_g}{w}\right)\right)+\mathcal{L}_1\left(1-\min \left(\frac{h}{h_g}, \frac{h_g}{h}\right)\right)
$$
