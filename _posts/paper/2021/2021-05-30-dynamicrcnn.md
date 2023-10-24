---
layout: post
title: 'Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training'
date: 2021-05-30
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6534d72bc458853aefb01e8f.jpg'
tags: 论文阅读
---

> Dynamic R-CNN：通过动态训练实现高质量目标检测.

- paper：[Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training](https://arxiv.org/abs/2004.06002)

本文针对**faster rcnn**的训练过程进行动态分析，提出了动态标签分配策略和动态回归损失函数。前者随着训练进行不断自适应增加**rcnn**正样本阈值，后者针对回归分支预测**bbox**的方差减少特点自适应修改**SmoothL1 Loss**参数。通过这两个自适应操作可以进一步提高目标检测的精度。

![](https://pic.imgdb.cn/item/6534e3a2c458853aefd94c7a.jpg)

## 1. 训练过程分析

**faster rcnn**的核心超参数主要是正样本**iou**阈值，一般都是设置成固定值如$0.5$。不同的固定阈值设置导致的检测精度**AP**是不一样的，特定的**iou**阈值对于分布相同的**roi**增益最大，其余增益比较少。也就是说如果实际项目倾向于无漏报，则**iou**阈值可以适当降低，反之则可以提高**iou**阈值。

![](https://pic.imgdb.cn/item/6534d93bc458853aefb8206f.jpg)

从图中可以看出，随着训练进行，在固定**iou**阈值下正样本个数是不断增加的，同时回归误差的方差是不断变小的。这说明随着训练过程进行，预测的**bbox**质量不断提高，正样本数也是不断增加的，此时采用固定的**iou**阈值肯定是不完美的；同时随着训练进行，回归误差的方差是不断变小的，采用**smooth l1**的固定$\beta$也是不完美的，这两个核心参数应该动态变化。

## 2. 训练过程改进

### (1) Dynamic Label Assignment

**rcnn**的默认设置是正负样本区分阈值是**0.5**，没有忽略样本。本文动态的思想是：在训练前期，高质量正样本很少，所以应该采用相对低的**iou**阈值来补偿；随着训练进行，高质量正样本不断增加，故可以慢慢增加**iou**阈值。其具体操作如下：

$$
label = \begin{cases}
1, & \max IoU(b,G) \geq T_{now} \\
0, & \max IoU(b,G) < T_{now}
\end{cases}
$$

$T_{now}$是要动态调整的参数。做法是首先计算每个**ROI**和所有**gt**的最大**iou**值，在每张图片上选取第$K_{I-th}$个最大值，遍历所有图片求均值作为$T_{now}$，并且每隔$C$个迭代更新一次该参数。

![](https://pic.imgdb.cn/item/6534dcddc458853aefc3f6ff.jpg)

绿色区域是正样本**anchor**，红色是负样本**anchor**。随着训练进行，**iou**阈值在动态增加，一些原本是相对高质量的预测慢慢变成了负样本，整个**mAP**在提高。

### (2) Dynamic SmoothL1 Loss

随着训练的进行，**bbox**预测值方差会不断变小，反映出高质量预测框不断增加。

![](https://pic.imgdb.cn/item/6534dec9c458853aefc9ef4c.jpg)

可以看出随着训练进行，回归误差的方差在变小，并且越来越集中(反映出高质量样本在增加)，并且不同**iou**阈值下，分布差异比较大。既然**iou**的阈值设置会影响输出分布，作者采用带参数的**smooth L1 loss**来自适应。

$$ \text{DSL}(x, \beta_{now}) = \begin{cases} |x|-0.5\beta_{now}, & |x| ≥ \beta_{now} \\ 0.5x^2/\beta_{now}, &|x| < \beta_{now} \end{cases} $$

$\beta_{now}$是需要动态确定的。其确定规则是先计算预测值和**GT**的回归误差，然后选择第$K_{\beta-th}$个最小值，然后在达到迭代次数后，采用中位数作为设置值。

![](https://pic.imgdb.cn/item/6534dff1c458853aefcda182.jpg)

随着训练进行，高质量样本越来越多，回归误差会越来越小，并且高质量的预测框其误差会更小。引入动态$\beta_{now}$减少来修正，来增加高质量部分样本的梯度，可以不断突出高质量预测框的回归误差。