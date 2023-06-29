---
layout: post
title: '点云分类(Point Cloud Classification)'
date: 2023-04-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/649ce3a01ddac507ccad0c9f.jpg'
tags: 深度学习
---

> Point Cloud Classification.

**点云（Point Cloud）**是一种三维数据的非结构化表示格式，能够保留三维空间中的原始几何信息，没有任何离散化。点云具有以下性质：
- **无序性(unordered)**：与像素点阵或三维点阵下排布的网格状数据不同，点云是无序的。因此要求以$N$个$3D$点（形成$N\times 3$矩阵）为输入的网络需要对数据的$N!$排布具有不变性；
- **交互性(interaction)**：点集中的大部分点不是孤立的，邻域点一定属于一个有意义的子集。因此网络需要从近邻点学习到局部结构，以及局部结构的相互关系；
- **变换不变性(transformation invariance)**：学习的点特征应该对特定的变换具有不变性，比如旋转和平移变换不会改变全局点云的类别以及每个点的分割结果。

**点云分类(Point Cloud Classification)**即点云形状分类，是一种重要的点云理解任务。该任务的方法通常首先学习每个点的嵌入，然后使用聚合方法从整个点云中提取全局形状嵌入，并通过分类器进行分类。根据神经网络输入的数据格式，三维点云分类方法可分为：
- 基于多视图(**Multi-view based**)的方法：将点云投影为多个二维图像，如**MVCNN**, **MHBN**
- 基于体素(**Voxel-based**)的方法：将点云转换为三维体素表示，如**VoxNet**, **OctNet**
- 基于点(**Point-based**)的方法：直接处理原始点云，如**PointNet**, **PointNet++**

### ⭐ 扩展阅读：
- [Deep Learning for 3D Point Clouds: A Survey](https://arxiv.org/abs/1912.12033)



# 1. 基于多视图的点云分类

基于**多视图 (Multi-view)**的点云分类首先把三维非结构化点云投影到多个二维视图中并提取视图特征，然后将这些特征融合为具有判别性的全局表示以进行精确的分类。

### ⚪ MVCNN

- paper：[<font color=blue>Multi-view Convolutional Neural Networks for 3D Shape Recognition</font>](https://0809zheng.github.io/2023/04/02/mvcnn.html)

**MVCNN**使用参数共享的卷积神经网络从多视角**2D**图像中提取特征，并通过**View-Pooling**层将特征最大化为全局描述符。

![](https://pic.imgdb.cn/item/649ce58a1ddac507ccafdd99.jpg)

### ⚪ MHBN

- paper：[Multi-view Harmonized Bilinear Network for 3D Object Recognition](https://cse.buffalo.edu/~jsyuan/papers/2018/Multi-view%20Harmonized%20Bilinear%20Network%20for%203D%20Object%20Recognition.pdf)

**MHBN**通过协调双线性池化来集成局部卷积特征，以产生紧凑的全局描述符。

![](https://pic.imgdb.cn/item/649cefc91ddac507ccc2ced0.jpg)



# 2. 基于体素的点云分类

基于**体素(Voxel)**的点云分类通常将点云体素化为三维网格，然后在体积表示上应用三维卷积神经网络进行形状分类。

### ⚪ VoxNet

- paper：[<font color=blue>VoxNet: A 3D Convolutional Neural Network for real-time object recognition</font>](https://0809zheng.github.io/2023/04/03/voxnet.html)

**VoxNet**把点云转换为体积占用网格（**Volumetric Occupancy Grid**），并通过三维卷积神经网络进行分类。

![](https://pic.imgdb.cn/item/649cf8b11ddac507ccd27e3c.jpg)

### ⚪ OctNet

- paper：[<font color=blue>OctNet: Learning Deep 3D Representations at High Resolutions</font>](https://0809zheng.github.io/2023/04/04/octnet.html)

**OctNet**把点云转换为混合网格八叉树（**Hybrid Grid-Octree**）数据结构，八叉树中的每个叶子节点都存储着体素里包含的所有特征的**pooled summary**。

![](https://pic.imgdb.cn/item/649d1cc31ddac507cc0c130c.jpg)

# 3. 基于点的点云分类

基于**点(point)**的点云分类使用神经网络学习每个点的特征。

### ⚪ PointNet

- paper：[<font color=blue>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</font>](https://0809zheng.github.io/2023/04/05/pointnet.html)

**PointNet**使用网络学习数据和特征的变换矩阵以对齐点云的变换不变性，引入最大池化以适应点云的无序性，并通过将点的特征与全局特征连接以感知全局语义信息。

![](https://pic.imgdb.cn/item/649d318c1ddac507cc304a37.jpg)


### ⚪ PointNet++

- paper：[<font color=blue>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</font>](https://0809zheng.github.io/2023/04/06/pointnetpp.html)

**PointNet++**引入了**Set Abstracion**来进行局部信息聚合。该结构通过最远点采样选择部分点，通过**Ball Query**把点云划分成若干局部区域，并通过**PointNet**提取局部区域的点云特征。

![](https://pic.imgdb.cn/item/649d3eb91ddac507cc47010a.jpg)