---
layout: post
title: '语义分割'
date: 2020-05-07
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ebb67aac2a9a83be59d878a.jpg'
tags: 深度学习
---

> Semantic Segmentation.

**语义分割（semantic segmentation）**是对图像的每个像素进行分类，注重类别之间的区分，而不区分同一类别的不同个体。

**本文目录**：
1. 语义分割数据集
2. FCN
3. SegNet
4. UNet
5. DeepLab
6. RefineNet

# 1. 语义分割数据集
除常用的**PASCAL VOC**数据集和**MS COCO**数据集外，语义分割数据集还有：
- APSIS
- SYNTHIA

### APSIS
- 主页：[APSIS](http://xiaoyongshen.me/webpage_portrait/index.html)

**人体肖像分割数据库(Automatic Portrait Segmentation for Image Stylization, APSIS)**

![](https://pic.downk.cc/item/5ebb5e07c2a9a83be58e8e68.jpg)

### SYNTHIA
- 主页：[SYNTHIA](http://synthia-dataset.net)

计算机合成的城市道路驾驶环境的像素级标注的数据集。

是为了在自动驾驶或城市场景规划等研究领域中的场景理解而提出的。

提供了**11**个类别物体（分别为天空、建筑、道路、人行道、栅栏、植被、杆、车、信号标志、行人、骑自行车的人）细粒度的像素级别的标注。

![](https://pic.downk.cc/item/5ebb5eb7c2a9a83be58f1d5e.jpg)

# 2. FCN
- paper：[Fully convolutional networks for semantic segmentation](https://arxiv.org/abs/1411.4038)

全卷积网络通过一系列下采样和上采样，输出特征图像对每个像素进行分类。

上采样使用**转置卷积 transpose conv**。

1. 先进行5次下采样得到尺寸为输入图像$\frac{1}{32}$的特征图像；
2. 对上述特征图像进行32倍上采样得到第一张输出特征图像$$FCN-32s$$；
3. 结合第4次和第5次下采样的特征映射进行16倍上采样得到第二张输出特征图像$$FCN-16s$$；
3. 结合第3次、第4次和第5次下采样的特征映射进行8倍上采样得到第三张输出特征图像$$FCN-8s$$。

![](https://pic.downk.cc/item/5ebb6115c2a9a83be592ef3d.jpg)

特征图像$$FCN-8s$$相对于特征图像$$FCN-32s$$和特征图像$$FCN-16s$$，既含有丰富的语义信息，又含有丰富的空间信息，分割效果最好：

![](https://pic.downk.cc/item/5ebcd2b6c2a9a83be51bdab1.jpg)

# 3. SegNet
- paper：[Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://arxiv.org/abs/1511.00561?context=cs)
- demo：[demo](http://mi.eng.cam.ac.uk/projects/segnet/)

![](https://pic.downk.cc/item/5ebb64bcc2a9a83be59a49f5.jpg)

**SegNet**设计了对称的**Encoder-Decoder**结构。

- **Encoder**：一系列卷积层和下采样的池化层
- **Decoder**：一系列卷积层和上采样的池化层

上采样使用**反池化 unpooling**。

下采样时，存储对应的最大池化索引位置；上采样时，用存储的索引进行上采样最大池化。
![](https://pic.downk.cc/item/5ebcd3ffc2a9a83be51d0f2c.jpg)

# 4. UNet
- paper：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

**UNet**主要应用于生物医学图像分割。

![](https://pic.downk.cc/item/5ebb673ec2a9a83be59d14c2.jpg)

**UNet**的**Encoder**进行4次下采样，**Decoder**进行4次上采样。

上采样使用**转置卷积 transpose conv**。

在同一个stage使用了**skip connection**，保证了最后恢复出来的特征图融合了更多的low-level的features，也使得不同尺度（scale） 的feature得到了融合，从而可以进行多尺度预测，4次上采样也使得分割图恢复边缘等信息更加精细。

# 5. DeepLab
**DeepLab**是谷歌提出的一系列图像分割网络，至今共有四个版本。

### (1)Deeplab v1
- paper：[Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)

**DeepLab v1**网络在**VGG-16**的基础上修改：
- 把全连接层转为卷积层；
- 下采样只保留前三个最大池化，使得输出特征映射为输入的$$\frac{1}{8}$$；
- 池化层之后的卷积采用空洞卷积。

创新点：
1. 使用**空洞卷积 Atrous conv**在不增加参数的情况下增加了感受野；
2. 使用**双线性插值 bi-linear interpolation**把特征映射恢复原始分辨率；
3. 使用全连接的**条件随机场 CRF**精细化分割结果。

![](https://pic.downk.cc/item/5ebcd9d0c2a9a83be5220bbb.jpg)

### (2)Deeplab v2
- paper：[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

提出**空洞空间金字塔池化 Atrous Spatial Pyramid Pooling(ASPP)**，使用不同扩张率的空洞卷积，能够提取多尺度特征：

![](https://pic.downk.cc/item/5ebcdcdcc2a9a83be5246ac8.jpg)

![](https://pic.downk.cc/item/5ebcdd28c2a9a83be524a0fa.jpg)

相比于**DeepLab v1**，**DeepLab v2**的改进在于：
1. 使用**ASPP**实现多尺度的目标分割；
2. 使用**ResNet**网络；
3. 改进了学习率策略。

### (3)Deeplab v3
- paper：[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

**DeepLab v3**去除了条件随机场，改进了**ASPP**层，增加了$1×1$卷积和全局平均池化：

![](https://pic.downk.cc/item/5ebcde6bc2a9a83be525b262.jpg)

另一个改进方法是串联使用不同扩张率的空洞卷积：

![](https://pic.downk.cc/item/5ebcdf3ac2a9a83be52684bd.jpg)

效果不如前者。

### (4)Deeplab v3+
- paper：[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611v1)

**DeepLab v3+**采用了两种卷积网络结构，分别是**Resnet 101**和**Xception**，后者效果更好。

下图是**（a）DeepLab v3**和**（c）DeepLab v3+**的对比：

![](https://pic.downk.cc/item/5ebce009c2a9a83be5274019.jpg)

**DeepLab v3+**的Decoder部分使用了卷积网络中间层的特征映射：

![](https://pic.downk.cc/item/5ebce077c2a9a83be527a780.jpg)

# 6. RefineNet
- paper：[RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)

**RefineNet**的创新点在于$decoder$的方式:

不同于$UNet$在上采样后直接和$encoder$的$feature$ $map$进行级联，本文通过$RefineNet$进行上采样，把$encoder$产生的$feature$和上一阶段$decoder$的输出同时作为输入，在$RefineNet$中进行一系列卷积、融合、池化，使得多尺度特征的融合更加深入。

![](https://pic.downk.cc/item/5ebcea7ac2a9a83be531a81b.jpg)

**RefineNet**模块包括：
1. 残差卷积单元**RCU：Residual Conv Unit**
2. 多分辨率融合单元**Muitl-resolution Fusion**
3. 链式残差池化单元**Chained Residual Pooling**

![](https://pic.downk.cc/item/5ebceacbc2a9a83be5320358.jpg)