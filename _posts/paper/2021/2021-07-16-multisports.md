---
layout: post
title: 'MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions'
date: 2021-07-16
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60ef8bee5132923bf8569298.jpg'
tags: 论文阅读
---

> MultiSports：基于篮球,足球,体操,排球赛事的大型时空动作检测数据集.

- paper：MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions
- arXiv：[link](https://arxiv.org/abs/2105.07404)
- web：[link](https://deeperaction.github.io/multisports/)

作者认为，一个高质量的时空动作检测数据集数据集应该具有以下特点：
1. 应该弱化背景信息以及人物姿态对判断人类动态行为的影响，即只通过单张图像较难判断动作。
1. 行为应该有良好的时间边界。
2. 行为类别应具有一定的细粒度。

作者提出了一个大型、高质量的时空动作检测数据集**MultiSports**，有$25$**fps**的逐帧标签，且每个动作具有清晰的时间边界。具体地，选择了四种多人运动场景(篮球、足球、排球、体操)。选择运动场景是因为动作具有明确的定义，且有大量不同动作同时进行。在标注时先让专业运动员标注每个动作的起始帧与结尾帧，并标注起始帧中的对应人物的边界框，然后通过众包进行标注。没有标注一些日常行为(如站,坐)，不考虑犯规动作。

![](https://pic.imgdb.cn/item/60efe0b55132923bf87a8c32.jpg)

下图从动作数量、实例个数、动作平均持续时间和边界框数量等因素对比不同的时空动作检测数据集：

![](https://pic.imgdb.cn/item/60efdfa65132923bf87213ba.jpg)

通过统计不同动作类别的实例数，发现数据服从长尾分布：

![](https://pic.imgdb.cn/item/60efdffd5132923bf874c7b4.jpg)

通过统计每个实例的长度(帧数)分布，发现大部分数据属于短视频：

![](https://pic.imgdb.cn/item/60efe0315132923bf876669d.jpg)

作者使用一些**SOTA**模型在**MultiSports**上进行训练，分析其测试结果：

![](https://pic.imgdb.cn/item/60efe0d85132923bf87ba574.jpg)

作者针对不同的错误情况进行定义，并统计这些错误实际出现的比例：

![](https://pic.imgdb.cn/item/60efe1295132923bf87e1f8d.jpg)

![](https://pic.imgdb.cn/item/60efe0ef5132923bf87c5496.jpg)

通过混淆矩阵，也可以分析不同运动中对不同类别的预测情况：

![](https://pic.imgdb.cn/item/60efe13c5132923bf87eb93d.jpg)