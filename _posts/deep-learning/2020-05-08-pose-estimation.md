---
layout: post
title: '人体姿态估计'
date: 2020-05-08
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ebab802101ccd402bd8d03b.jpg'
tags: 深度学习
---

> Person Pose Estimation.

**本文目录**：
1. 人体姿态估计
2. 单人姿态估计
3. 多人姿态估计
4. 姿态估计数据集

# 1. 人体姿态估计
**姿态估计（pose estimation）**，就是估计图像或视频中人体关节位置（也称为**关键点**，如手肘、膝盖、肩膀），又称人体**关键点检测**。

人体姿态估计可以分为：
- **单人姿态估计（single person pose estimation， SPPE）**
- **多人姿态估计（multi person pose estimation， MPPE）**

# 2. 单人姿态估计

### DeepPose
- paper:[DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/abs/1312.4659)

**DeepPose**是深度学习应用于人体姿态估计的开山之作，将姿态估计问题转化为图像的**特征提取**和关键点的**坐标回归**问题。

使用$AlexNet$卷积神经网络提取特征，输出维度$2k$表示对人体的$k$对关键点坐标进行回归预测。

模型使用了**级联回归器(cascaded regressor)**，基于前一阶段预测坐标位置对图像进行局部裁剪作为现阶段的输入，因此现阶段的输入有着更高的分辨率，从而能学习到更为精细的尺度特征，以此来对前一阶段的预测结果进行细化。

![](https://pic.downk.cc/item/5ec5e490c2a9a83be52357e5.jpg)

### Stacked Hourglass
- paper:[Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937)

该论文提出了**Hourglass**模块，不仅应用于姿态估计，后续也被用于目标检测、分割。

**Hourglass**模块包含重复的**降采样**（高分辨率到低分辨率）和**上采样**（低分辨率到高分辨率），此外还使用了**残差连接**保存不同分辨率下的空间信息。

与**DeepPose**采用级联回归器目的相似，**Hourglass**模块可以捕捉利用多个尺度上的信息，例如局部特征信息对于识别脸部、手部等特征十分重要，但人体最终的姿态估计也需要图像的全局特征信息。

![](https://pic.downk.cc/item/5ebab71e101ccd402bd776a8.jpg)

直接对$k$对关键点坐标进行回归比较困难，将回归转换为预测$k$个**热图heatmap**，第$i$个热图表示第$i$对关键点的位置置信度，用于预测每个像素点是关键点的概率。

$Ground Truth$热图为以实际关节点位置为中心的标准正态分布，采用均方误差损失。

![](https://pic.downk.cc/item/5ebab780101ccd402bd807c9.jpg)

# 3. 多人姿态估计
多人姿态估计的方法：
- **Top-Down方法**：先用目标检测的方法检测图像中的人，再用单人姿态估计对每个人进行姿态估计。
1. 优点：精度高，容易实现；
2. 缺点：依赖于目标检测，速度随场景中人数增加而变慢，无法解决遮挡问题。
- **Bottom-Up方法**：首先检测出图像中所有关键点位置，再将属于不同人的关键点进行关联和组合。
1. 优点：速度不受图像中人数变化影响，实时性好；
2. 缺点：精度相对低，无法解决遮挡问题。

![](https://pic.downk.cc/item/5ebaa9c2101ccd402bc250e9.jpg)

### Alpha Pose
- paper：[RMPE: Regional Multi-person Pose Estimation](https://arxiv.org/abs/1612.00137v3)

**Alpha Pose**是一种**Top-Down**的多人姿态估计方法，使用**Faster RCNN**进行人体检测，使用**Stacked Hourglass**进行单人姿态估计。

该方法的主要问题在于人体检测**位置不准确**和**检测冗余**导致的单人姿态估计失效。

提出了新的**区域多人姿态估计（Region Multi-Person Pose Estimation，RMPE）**框架，包含三部分：
1. **Symmetric Spatial Transformer Network（SSTN）**：对称空间变换网络，用于在不准确的人体检测框中提取准确的单人区域；
2. **Parametric Pose Non-Maximum-Suppression（p-Pose NMS）**：参数化姿态的非最大值抑制，用于解决人体检测框冗余问题；
3 .**Pose-Guided Proposals Generator（PGPG）**：姿态引导区域框生成器，用于数据增强。


# 4. 姿态估计数据集

### (1)LSP
- 地址：[http://sam.johnson.io/research/lsp.html](http://sam.johnson.io/research/lsp.html)
- 样本数：2000
- 关节点个数：14
- 全身，单人

### (2)FLIC
- 地址：[https://bensapp.github.io/flic-dataset.html](https://bensapp.github.io/flic-dataset.html)
- 样本数：20000
- 关节点个数：9
- 全身，单人

### (3)MPII
- 地址：[http://human-pose.mpi-inf.mpg.de/](http://human-pose.mpi-inf.mpg.de/)
- 样本数：25000
- 关节点个数：16
- 全身，单人/多人，40000人，410种人类活动

### (4)MSCOCO
- 地址：[http://cocodataset.org/#download](http://cocodataset.org/#download)
- 样本数：300000
- 关节点个数：18
- 全身，多人，100000人

![](https://pic.downk.cc/item/5ebaa357101ccd402bb8c7c6.jpg)

### (5)AI Challenge
- 地址：[https://challenger.ai/competition/keypoint/subject](https://challenger.ai/competition/keypoint/subject)
- 样本数：210000训练集，30000验证集，30000测试集
- 关节点个数：14
- 全身，多人，380000人