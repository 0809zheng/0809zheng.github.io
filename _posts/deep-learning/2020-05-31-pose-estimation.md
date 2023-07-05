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
3. **3D**人体姿态估计
5. 人体姿态估计的评估指标
6. 人体姿态估计数据集

### ⭐ 扩展阅读：
- [<font color=blue>Monocular Human Pose Estimation: A Survey of Deep Learning-based Methods</font>](https://0809zheng.github.io/2020/11/17/pose.html)：(arXiv2006)单目人体姿态估计的深度学习方法综述。

# 1. 2D单人姿态估计 2D Single Human Pose Estimation

**2D**单人人体姿态估计通常是从已完成定位的人体图像中计算人体关节点的位置，并进一步生成**2D**人体骨架。这些方法可以进一步分为**基于回归(regression-based)**的方法与**基于检测(detection-based)**的方法。
- 基于回归的方法：直接将输入图像映射为人体关节的**坐标**或人体模型的**参数**，如**DeepPose**, **TFPose**。
- 基于检测的方法：将输入图像映射为**图像块(patch)**或人体关节位置的**热图(heatmap)**，从而将身体部位作为检测目标；如**CPM**, **Hourglass**, **Chained**, **MCA**, **FPM**, **HRNet**, **ViTPose**。

## （1）基于回归的2D单人姿态估计 Regression-based 2D SHPE

基于回归的方法直接预测人体各关节点的**联合坐标**。这类方法可以端到端的训练，速度更快，并且可以得到子像素级(**sub-pixel**)的精度；但由于映射是高度非线性的，学习较为困难，且缺乏鲁棒性。

### ⚪ DeepPose

- paper：[<font color=blue>DeepPose: Human Pose Estimation via Deep Neural Networks</font>](https://0809zheng.github.io/2021/04/01/deeppose.html)

**DeepPose**使用预训练卷积神经网络提取特征，直接对人体的$k$对关键点坐标进行回归预测。模型使用了级联回归器，对前一阶段的预测结果进行细化。

![](https://pic.imgdb.cn/item/649a3fac1ddac507cc46d279.jpg)

### ⚪ TFPose

- paper：[<font color=blue>TFPose: Direct Human Pose Estimation with Transformers</font>](https://0809zheng.github.io/2022/05/09/tfpose.html)

**TFPose**通过将卷积神经网络与**Transformer**结构相结合，直接并行地预测所有关键点坐标序列。**Transformer**解码器将一定数量的关键点查询向量和编码器输出特征作为输入，并通过一个多层前馈网络预测最终的关键点坐标。

![](https://pic.imgdb.cn/item/629df817094754312978e61a.jpg)

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

### ⚪ HRNet

- paper：[<font color=blue>Deep High-Resolution Representation Learning for Human Pose Estimation</font>](https://0809zheng.github.io/2021/04/14/hrnet.html)

**HRNet**不断地去融合不同尺度上的信息，其整体结构分成多个层级，但是始终保留着最精细的空间层级信息，通过融合下采样然后做上采样的层，来获得更多的上下文以及语义层面的信息。

![](https://pic.imgdb.cn/item/64a510fb1ddac507cc221282.jpg)

### ⚪ ViTPose

- paper：[<font color=blue>ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation</font>](https://0809zheng.github.io/2022/07/07/vitpose.html)

**ViTPose**使用**ViT**结构作为**Backbone**，结合一个轻量级的**Decoder**解码关节点热图。

![](https://pic.imgdb.cn/item/64a38a991ddac507cc5f767b.jpg)

# 2. 2D多人姿态估计 2D Multiple Human Pose Estimation

与单人姿态估计相比，多人姿态估计需要同时完成**检测**和**估计**任务。根据完成任务的顺序不同，多人姿态估计方法分为**自上而下(top-down)**的方法和**自下而上(bottom-up)**的方法。
- 自上而下的方法先做**检测**再做**估计**。即先通过目标检测的方法在输入图像中检测出不同的人体，再使用单人姿态估计方法对每个人进行姿态估计；如**CPN**, 。
- 自下而上的方法先做**估计**再做**检测**。即先在图像中估计出所有人体关节点，再将属于不同人的关节点进行关联和组合；如**DeepCut**, **DeeperCut**, **Associative Embedding**, **OpenPose**。

![](https://pic.imgdb.cn/item/649bd8b31ddac507cc9c3cc5.jpg)

## （1）自上而下的2D多人姿态估计 Top-down 2D MHPE

自上而下的方法中两个最重要的组成部分是**人体区域检测器**和**单人姿态估计器**。大多数研究基于现有的人体目标检测器进行估计，如**Faster R-CNN**、**Mask R-CNN**和**FPN**。

通过将现有的人体检测网络和单人姿态估计网络结合起来，可以轻松实现自上而下的多人姿态估计。这类方法检测人体目标的召回率较高，关节点的定位精度较高，几乎在所有**Benchmarks**上取得了最先进的表现，但这种方法的处理速度受到检测人数的限制，当检测人数增加时运行时间成倍地增加。

### ⚪ CPN

- paper：[<font color=blue>Cascaded Pyramid Network for Multi-Person Pose Estimation</font>](https://0809zheng.github.io/2021/04/12/cpn.html)

**CPN**使用**FPN**进行人体目标检测，在关节点检测时采用**GlobalNet**和**RefineNet**。**GlobalNet**对关键点进行粗提取，**RefineNet**对不同层信息进行融合，更好地综合特征定位关键点。

![](https://pic.imgdb.cn/item/64a4db461ddac507ccb950e1.jpg)

### ⚪ MSPN

- paper：[<font color=blue>Rethinking on Multi-Stage Networks for Human Pose Estimation</font>](https://0809zheng.github.io/2021/04/13/mspn.html)

**MSPN**使用**CPN**的**GlobalNet**进行多阶段的堆叠，并引入了跨阶段的特征聚合（融合不同阶段同一层的特征）与由粗到细的监督（中间监督热图随着阶段的增加越来越精细）。

![](https://pic.imgdb.cn/item/64a50a0a1ddac507cc132f0a.jpg)


## （2）自下而上的2D多人姿态估计 Bottom-up 2D MHPE

自下而上的人体姿态估计方法的主要组成部分包括**人体关节检测**和**候选关节分组**。大多数算法分别处理这两个组件，也可以在**单阶段**进行预测。

这类方法的关键在于正确组合关节点，速度不受图像中人数变化影响，实时性好；但是性能会受到复杂背景和人为遮挡的影响，精度相对低，当不同人体之间有较大遮挡时，估计效果会进一步下降。

### ⚪ DeepCut

- paper：[<font color=blue>DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation</font>](https://0809zheng.github.io/2021/04/11/deepcut.html)

**DeepCut**首先从图像中提取关节点候选集合$D$，并将其划分到$C$个关节类别中。定义三元组$(x,y,z)$：

$$
x \in \{0,1\}^{D\times C},y \in \{0,1\}^{\begin{pmatrix}D \\ 2\end{pmatrix}},z \in \{0,1\}^{\begin{pmatrix}D \\ 2\end{pmatrix}\times C^2}
$$

其中$x_{dc} = 1$表示关节点$d$属于类别$c$，$y_{dd'}=1$表示关节点$d$和$d'$属于同一个人，$z_{dd'cc'}=x_{dc}x_{d'c'}y_{dd'}$是一个辅助变量。进一步建立$(x,y,z)$的线性方程，构造整数线性规划问题。

![](https://pic.imgdb.cn/item/64a4d4901ddac507ccacd73b.jpg)

### ⚪ DeeperCut

- paper：[DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model](https://arxiv.org/abs/1605.03170)

**DeeperCut**相比于**DeepCut**有三点改进：
- 使用**ResNet**提高关节点检测的准确率；
- 改进优化策略更有效地搜索空间，获得更好的性能和显著速度提升；
- 通过图像条件成对项(**image-conditioned pairwise terms**)进行关节点的非极大值抑制，即通过候选关节点之间的距离来判断是否为不同的重要关节点。

![](https://pic.imgdb.cn/item/64a4d5a31ddac507ccaedcf9.jpg)

### ⚪ Associative Embedding

- paper：[<font color=blue>Associative Embedding: End-to-End Learning for Joint Detection and Grouping</font>](https://0809zheng.github.io/2021/04/10/associative.html)

**Associative Embedding**为每个关节点输出一个**embedding**，使得同一个人的**embedding**尽可能相近，不同人的**embedding**尽可能不一样。

![](https://pic.imgdb.cn/item/64a4c1c81ddac507cc8b09b6.jpg)

### ⚪ OpenPose

- paper：[<font color=blue>Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields</font>](https://0809zheng.github.io/2021/04/08/openpose.html)

**OpenPose**使用卷积神经网络从输入图像中提取部位置信图(**PCM**)与部位亲和场(**PAF**)：部位置信图是指人体关节点的热力图，用于表征人体关节点的位置；部位亲和场是用于编码肢体支撑区域的位置和方向信息的**2D**向量场。然后通过二分匹配（边权基于**PAF**计算）进行关节分组。



![](https://pic.imgdb.cn/item/649bde451ddac507cca66b23.jpg)

# 3. 3D人体姿态估计 3D Human Pose Estimation

**3D**人体姿态估计是从图片或视频中估计出关节点的三维坐标$(x, y, z)$，本质上是一个回归问题。与**2D**人体姿态估计相比，**3D**人体姿态估计需要估计**深度(depth)**信息。**3D**人体姿态估计的主要挑战：
- 单视角下**2D**到**3D**映射中固有的深度模糊性与不适定性：因为一个**2D**骨架可以对应多个**3D**骨架。
- 缺少大型的室外数据集和特殊姿态数据集：**3D**姿态数据集是依靠适合室内环境的动作捕捉（**MOCAP**）系统构建的，而**MOCAP**系统需要佩戴有多个传感器的复杂装置，在室外环境使用是不切实际的。因此数据集大多是在实验室环境下建立的，模型的泛化能力也比较差。

**3D**人体姿态估计方法可以分为：
- 直接回归的方法：直接把图像映射为**3D**关节点，如**DconvMP**
- **2D→3D**的方法：从**2D**姿态估计结果中估计深度信息，如
- 基于模型的方法：引入人体模型，如

## （1）直接回归的3D人体姿态估计 Regression-based 3D HPE

直接回归的**3D**人体姿态估计直接把图像映射成**3D**关节点，这通常是严重的欠定问题，因此需要引入额外的约束条件。

### ⚪ DconvMP

- paper：[<font color=blue>3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network</font>](https://0809zheng.github.io/2021/04/15/dconvmp.html)

把**3D**人体姿态估计任务建模为关节点回归任务+关节点检测任务。关节点回归任务估计关节点相对于根关节点的位置；关节点检测任务检测每个局部窗口中是否存在关节点。

![](https://pic.imgdb.cn/item/64a531751ddac507cc6d7c6d.jpg)


## （2）2D→3D的3D人体姿态估计 2D→3D 3D HPE

**2D→3D**的**3D**人体姿态估计从**2D**姿态估计的结果中估计深度信息，再生成**3D**姿态估计，这类方法可以很容易地利用**2D**姿态数据集，并且具有**2D**姿态估计的优点。


- [A simple yet effective baseline for 3d human pose estimation](https://0809zheng.github.io/2020/11/19/3dpose.html)：(arXiv1705)通过回归2D关节坐标进行3D人体姿态估计。

- [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://0809zheng.github.io/2022/04/06/ssmp.html)：(arXiv1712)使用遮挡鲁棒姿态图从单目相机中重构三维姿态。


## （3）基于模型的3D人体姿态估计 Model-based 3D HPE

基于模型的方法通常采用**人体参数模型**从图像中估计人的姿态和形状。一些工作采用了**SMPL**人体模型，从图像中估计三维参数。**运动学模型(kinematic model)**广泛应用于$3D$人体姿态估计中。


- [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://0809zheng.github.io/2021/01/13/openpose.html)：(arXiv1607)SMPLify：从单张图像中建立三维SMPL模型。

- [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://0809zheng.github.io/2021/02/02/smplifyx.html)：(arXiv1904)SMPLify-X：从单张图像重建3D人体、手部和表情。


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



- [AMASS: Archive of Motion Capture as Surface Shapes](https://0809zheng.github.io/2021/03/29/amass.html)：(arXiv1904)AMASS：经过SMPL参数标准化的三维人体动作捕捉数据集合。


