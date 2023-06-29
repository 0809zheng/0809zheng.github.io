---
layout: post
title: '人体姿态估计(Human Pose Estimation)'
date: 2020-05-31
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ebab802101ccd402bd8d03b.jpg'
tags: 深度学习
---

> Person Pose Estimation.

**人体姿态估计 (Human Pose Estimation, HPE)**是指从图像、视频等输入信号中估计人体的姿态信息。姿态通常以关键点（**keypoint**，也称为关节点 **joint**，比如手肘、膝盖、肩膀）组成的人体骨骼（**skeleton**）表示。

人体姿态估计与关键点检测的区别：



**本文目录**：
1. **2D**单人姿态估计
2. **2D**多人姿态估计
3. **3D**单人姿态估计
4. **3D**多人姿态估计
5. 人体姿态估计的评估指标
6. 人体姿态估计数据集

### ⭐ 扩展阅读：
- [<font color=blue>Monocular Human Pose Estimation: A Survey of Deep Learning-based Methods</font>](https://0809zheng.github.io/2020/11/17/pose.html)：(arXiv2006)单目人体姿态估计的深度学习方法综述。

# 1. 2D单人姿态估计 2D Single Human Pose Estimation

**2D**单人人体姿态估计通常是从已完成定位的人体图像中计算人体关节点的位置，并进一步生成**2D**人体骨架。这些方法可以进一步分为**基于回归(regression-based)**的方法与**基于检测(detection-based)**的方法。
- 基于回归的方法：直接将输入图像映射为人体关节的**坐标**或人体模型的**参数**，如**DeepPose**。
- 基于检测的方法：将输入图像映射为**图像块(patch)**或人体关节位置的**热图(heatmap)**，从而将身体部位作为检测目标；如**CPM**, **Hourglass**, **Chained**, **MCA**, **FPM**。

## （1）基于回归的2D单人姿态估计 Regression-based 2D SHPE

基于回归的方法直接预测人体各关节点的**联合坐标**。这类方法可以端到端的训练，速度更快，并且可以得到子像素级(**sub-pixel**)的精度；但由于映射是高度非线性的，学习较为困难，且缺乏鲁棒性。

### ⚪ DeepPose

- paper：[<font color=blue>DeepPose: Human Pose Estimation via Deep Neural Networks</font>](https://0809zheng.github.io/2021/04/01/deeppose.html)

**DeepPose**使用预训练卷积神经网络提取特征，直接对人体的$k$对关键点坐标进行回归预测。模型使用了级联回归器，对前一阶段的预测结果进行细化。

![](https://pic.imgdb.cn/item/649a3fac1ddac507cc46d279.jpg)



## （2）基于检测的2D单人姿态估计 Detection-based 2D SHPE

基于检测的方法使用**热图**来指示关节的真实位置。如下图所示，每个关键点占据一个热图通道，表示为以目标关节位置为中心的二维高斯分布。

![](https://pic.downk.cc/item/5fb37ddeb18d62711306f2a6.jpg)

由于热图能够保存空间位置信息，这类方法鲁棒性更好；但从中估计关节点坐标的准确性较差（热图大小往往是原图的等比例缩放，通过在输出热图上按通道找最大的响应位置，精度是**pixel**级别），并且阻碍了端到端的训练。


### ⚪ Convolutional Pose Machine (CPM)

- paper：[<font color=blue>Convolutional Pose Machines</font>](https://0809zheng.github.io/2021/04/04/cpm.html)

卷积姿态机使用顺序化的卷积网络来进行图像特征提取，以置信度热图的形式表示预测结果，在全卷积的结构下使用中间监督进行端到端的训练。

顺序化的卷积架构表现在网络分为多个阶段，每一个阶段都有监督训练的部分。前面的阶段使用原始图片作为输入，后面阶段使用之前阶段的特征图作为输入。

![](https://pic.imgdb.cn/item/649a887b1ddac507ccbf6247.jpg)


### ⚪ Stacked Hourglass Network

- paper：[<font color=blue>Stacked Hourglass Networks for Human Pose Estimation</font>](https://0809zheng.github.io/2021/04/03/hourglass.html)

堆叠沙漏网络是由若干个**Hourglass**模块堆叠而成，**Hourglass**模块是由若干次下采样+上采样和残差连接组成。此外还在每个**Hourglass**模块中引入了中间监督。

![](https://pic.imgdb.cn/item/649a44321ddac507cc4e7389.jpg)

### ⚪ Chained Prediction

- paper：[<font color=blue>Chained Predictions Using Convolutional Neural Networks</font>](https://0809zheng.github.io/2021/04/05/chain.html)

**Chained Prediction**是指按照关节链模型的顺序输出关节热图，每一步的输出取决于输入图像和先前预测的热图。

![](https://pic.imgdb.cn/item/649b8a9a1ddac507cc19d7e9.jpg)

### ⚪ Multi-Context Attention (MCA)

- paper：[<font color=blue>Multi-Context Attention for Human Pose Estimation</font>](https://0809zheng.github.io/2021/04/06/multicontext.html)

本文为堆叠沙漏网络引入了沙漏残差单元(**HRUs**)以学习不同尺度的特征，并在浅层引入多分辨率注意力，在深层引入由粗到细的部位注意力。

![](https://pic.imgdb.cn/item/649b95661ddac507cc2a4b4f.jpg)

### ⚪ Feature Pyramid Module (FPM)

- paper：[<font color=blue>Learning Feature Pyramids for Human Pose Estimation</font>](https://0809zheng.github.io/2021/04/07/prm.html)

特征金字塔模块能够提取不同尺度的特征，可以替换**Hourglass**中的残差模块。

![](https://pic.imgdb.cn/item/649ba4961ddac507cc44f04b.jpg)

# 2. 2D多人姿态估计 2D Multiple Human Pose Estimation

与单人姿态估计相比，多人姿态估计需要同时完成**检测**和**估计**任务。根据完成任务的顺序不同，多人姿态估计方法分为**自上而下(top-down)**的方法和**自下而上(bottom-up)**的方法。
- 自上而下的方法先做**检测**再做**估计**。即先通过目标检测的方法在输入图像中检测出不同的人体，再使用单人姿态估计方法对每个人进行姿态估计。
- 自下而上的方法先做**估计**再做**检测**。即先在图像中估计出所有人体关节点，再将属于不同人的关节点进行关联和组合。

![](https://pic.imgdb.cn/item/649bd8b31ddac507cc9c3cc5.jpg)

## （1）自上而下的2D多人姿态估计 Top-down 2D MHPE

自上而下的方法中两个最重要的组成部分是**人体区域检测器**和**单人姿态估计器**。大多数研究基于现有的人体目标检测器进行估计，如**Faster R-CNN**、**Mask R-CNN**和**FPN**。

通过将现有的人体检测网络和单人姿态估计网络结合起来，可以轻松实现自上而下的多人姿态估计。这类方法几乎在所有**Benchmarks**上取得了最先进的表现，但这种方法的处理速度受到检测人数的限制，当检测人数增加时运行时间成倍地增加。

## （2）自下而上的2D多人姿态估计 Bottom-up 2D MHPE

自下而上的人体姿态估计方法的主要组成部分包括**人体关节检测**和**候选关节分组**。大多数算法分别处理这两个组件，也可以在**单阶段**进行预测。

这类方法的关键在于正确组合关节点，速度不受图像中人数变化影响，实时性好；但是性能会受到复杂背景和人为遮挡的影响，精度相对低，当不同人体之间有较大遮挡时，估计效果会进一步下降。


### ⚪ OpenPose

- paper：[<font color=blue>Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields</font>](https://0809zheng.github.io/2021/04/08/openpose.html)

**OpenPose**使用卷积神经网络从输入图像中提取部位置信图(**PCM**)与部位亲和场(**PAF**)：部位置信图是指人体关节点的热力图，用于表征人体关节点的位置；部位亲和场是用于编码肢体支撑区域的位置和方向信息的**2D**向量场。然后通过二分匹配（边权基于**PAF**计算）进行关节分组。



![](https://pic.imgdb.cn/item/649bde451ddac507cca66b23.jpg)

# 3. 3D单人姿态估计 3D Single Human Pose Estimation

与**2D**人体姿态估计相比，**3D**人体姿态估计需要估计**深度(depth)**信息。根据是否引入**人体模型(human body model)**，**3D**单人姿态估计可以分为**不用模型(model-free)**的方法和**基于模型(model-based)**的方法。


## （1）不用模型的3D单人姿态估计 Model-free 3D SHPE

不用模型的**3D**单人姿态估计方法可以分成两类。第一类是直接把图像映射成**3D**关节点，这通常是严重的欠定问题；第二类是从**2D**姿态估计的结果中估计深度信息，再生成**3D**姿态估计，这类方法可以很容易地利用**2D**姿态数据集，并且具有**2D**姿态估计的优点。


## （2）基于模型的3D单人姿态估计 Model-based 3D SHPE

基于模型的方法通常采用**人体参数模型**从图像中估计人的姿态和形状。一些工作采用了**SMPL**人体模型，从图像中估计三维参数。**运动学模型(kinematic model)**广泛应用于$3D$人体姿态估计中。


# 4. 3D多人姿态估计 3D Multi Human Pose Estimation

$3D$多人姿态估计方法基于$3D$单人姿态估计方法。




# 3. 多人姿态估计


### Alpha Pose
- paper：[RMPE: Regional Multi-person Pose Estimation](https://arxiv.org/abs/1612.00137v3)

**Alpha Pose**是一种**Top-Down**的多人姿态估计方法，使用**Faster RCNN**进行人体检测，使用**Stacked Hourglass**进行单人姿态估计。

该方法的主要问题在于人体检测**位置不准确**和**检测冗余**导致的单人姿态估计失效。

提出了新的**区域多人姿态估计（Region Multi-Person Pose Estimation，RMPE）**框架，包含三部分：
1. **Symmetric Spatial Transformer Network（SSTN）**：对称空间变换网络，用于在不准确的人体检测框中提取准确的单人区域；
2. **Parametric Pose Non-Maximum-Suppression（p-Pose NMS）**：参数化姿态的非最大值抑制，用于解决人体检测框冗余问题；
3. **Pose-Guided Proposals Generator（PGPG）**：姿态引导区域框生成器，用于数据增强。


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


## ⚪ 姿态估计

- [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://0809zheng.github.io/2021/01/13/openpose.html)：(arXiv1607)SMPLify：从单张图像中建立三维SMPL模型。

- [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://0809zheng.github.io/2020/11/09/openpose.html)：(arXiv1611)OpenPose：实时多人体2D姿态估计。

- [A simple yet effective baseline for 3d human pose estimation](https://0809zheng.github.io/2020/11/19/3dpose.html)：(arXiv1705)通过回归2D关节坐标进行3D人体姿态估计。

- [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://0809zheng.github.io/2022/04/06/ssmp.html)：(arXiv1712)使用遮挡鲁棒姿态图从单目相机中重构三维姿态。


- [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://0809zheng.github.io/2021/02/02/smplifyx.html)：(arXiv1904)SMPLify-X：从单张图像重建3D人体、手部和表情。

- [AMASS: Archive of Motion Capture as Surface Shapes](https://0809zheng.github.io/2021/03/29/amass.html)：(arXiv1904)AMASS：经过SMPL参数标准化的三维人体动作捕捉数据集合。

- [TFPose: Direct Human Pose Estimation with Transformers](https://0809zheng.github.io/2022/05/09/tfpose.html)：(arXiv2103)TFPose: 基于Transformer的2D人体关节点回归。
