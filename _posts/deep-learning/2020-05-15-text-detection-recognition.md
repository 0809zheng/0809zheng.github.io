---
layout: post
title: '文本检测与识别(Text Detection and Recognition)'
date: 2020-05-15
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ee1d7f1c2a9a83be5506347.jpg'
tags: 深度学习
---

> Text Detection and Recognition.

文本检测与识别根据环境场合不同，可分为：
- **光学字符识别（Optical Character Recognition, OCR）**，传统上指对输入扫描文档图像进行分析处理，识别出图像中文字信息。
- **场景文字识别（Scene Text Recognition，STR）**，指识别自然场景图片中的文字信息。

本文主要关注场景文字识别。

- **文本检测（Text Detection）**解决的问题是哪里有文字，文字的范围有多少；
- **文本识别（Text Recognition）**对定位好的文字区域进行识别，主要解决的问题是每个文字是什么，将图像中的文字区域进转化为字符信息。

本文目录：
1. Benchmarks
2. （文本检测）EAST
3. （文本识别）CRNN
4. （文本检测与识别）Mask TextSpotter

# 1. Benchmarks
文本检测与识别常用的数据集包括：
- SynthText：合成数据集，有约80万张图像，包含多方向文本
- ICDAR2003：ICDAR2003国际文档分析与识别竞赛数据集，忽略数字字符或少于三个字符的图像后，得到860个文本图像的测试集
- ICDAR2013：ICDAR2013国际文档分析与识别竞赛数据集，主要为自然场景下的水平文本，有229张训练图像，233张测试图像
- ICDAR2015：主要为自然场景下多方向文本，有1000张训练图像，500张测试图像
- Total-Text：除了水平文本和定向文本外，Total-Text还包含许多弯曲文本，有1255张训练图像，300张测试图像。
- Street View Text：由Google街景收集的249张街景图像组成，从中裁剪出647张词图像
- IIIT 5k-word：从互联网收集的3000张裁剪的词图像
- COCO-Text：训练集43686，测试集20000


# 2. EAST
- paper：[EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2)

**EAST**是一种文本检测的方法，可以检测出图像中的文本区域。

### 网络结构
**EAST**的网络结构总共包含三个部分：
- **feature extractor stem（特征提取分支）**
- **feature-merging branch（特征合并分支）**
- **output layer（输出层）**

![](https://pic.downk.cc/item/5ee1a55bc2a9a83be5ffe725.jpg)

在**特征提取分支**部分，使用**PVANet**作为backbone卷积网络，主要由四层卷积层组成，并使用每一层得到的feature map。

在**特征合并分支**部分，借鉴了**U-net**的思想，只是U-net采用的是反卷积的操作，而这里采用的是反池化的操作。

在**输出层**部分，主要有两部分，一部分是用单个通道的卷积得到**score map**；另一部分是多个通道的卷积得到**geometry map**，在这一部分，几何形状可以是**RBOX（旋转盒子）**或者**QUAD（四边形）**。对于**RBOX**，主要有5个通道，其中四个通道表示每一个像素点与文本线上、右、下、左边界距离（**axis-aligned bounding box，AABB**），另一个通道表示该四边形的旋转角度。对于**QUAD**，则采用四边形四个顶点的坐标表示，共有8个通道。

![](https://pic.downk.cc/item/5ee1a6d1c2a9a83be502aeec.jpg)

### ground truth
很多数据集(如ICDAR2015)是用**QUAD**的方式标注的，需要生成**score map**和**geometry map**的真实标签。

对于**score map**：对标注框进行**缩放（shrink）**，缩放框内的像素标注为1，外的像素标注为0。得到**score map**的标签：

![](https://pic.downk.cc/item/5ee1bbd8c2a9a83be5293574.jpg)

对于**geometry map**：选择**score map**为正的像素点，其**QUAD**对应的标签直接是他们与四个顶点的偏移坐标，即顶点的差值；而对于**RBOX**，则首先会选择一个最小的外接矩形框住真实的四边形，然后计算每个正例像素点与该矩形四条边界的距离。

![](https://pic.downk.cc/item/5ee1bc82c2a9a83be529d862.jpg)

### Local-Aware NMS
该算法得到的检测框太多，直接使用非极大值抑制计算量大，作者提出了一种**Local-Aware NMS**。

假设来自附近相邻像素的几何形状往往高度相关，对这些像素的候选框进行逐步的合并，之后对合并的候选框进行常规的NMS。这里合并的四边形坐标是通过两个给定四边形的得分进行加权平均的。


# 3. CRNN
- paper：[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

**CRNN(卷积循环神经网络)**是一种文本识别的方法，可以从含有文本的区域中识别出具体的文字。

![](https://pic.downk.cc/item/5ee1be88c2a9a83be52c587f.jpg)

**CRNN**的网络结构总共包含三个部分：
- **卷积层（convolutional layer）**：提取图像卷积特征，输入图像大小为$(32,w,3)$，输出特征大小为$(1,\frac{w}{4},512)$。
- **循环层（recurrent layer）**：提取图像卷积特征中的序列特征，采用双向$LSTM$网络，时间步长$T=\frac{w}{4}$，每个输入向量长度$D=512$。
- **转录层（transcription layer）**：采用[**CTC**](https://0809zheng.github.io/2020/06/11/ctc.html)进行训练，得到最终的序列输出。

**CRNN**在训练之前，先把输入图像按比例缩放到相同高度，论文中使用的高度值是32。提取的特征序列向量是从卷积特征图上从左到右按照顺序生成的，每个特征向量表示了图像上一定宽度上的特征。双向**LSTM**把特征序列向量转化成输出的标签分布。转录层将**LSTM**网络预测的特征序列的所有可能的结果进行整合，经过去除空格（blank）和去重操作，就可以得到最终的序列标签。


# 4. Mask TextSpotter
- paper：[Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes](https://arxiv.org/abs/1807.02242)

**Mask TextSpotter**是一种端到端同时实现文本检测和识别的方法，可以检测任意方向的文本。

### 网络结构

![](https://pic.downk.cc/item/5ee1ccaec2a9a83be5400015.jpg)

**Mask TextSpotter**的网络结构基于**Mask R-CNN**，同时使用目标检测和实例分割的方法。

**Mask R-CNN**中的**mask branch**主要是做实例分割以及辅助目标检测，会将两者的结果结合起来；

而**Mask TextSpotter**中的**mask branch**是主要的网络结构，而目标检测部分则是做到一个辅助结果，网络是将**fast rcnn**检测出的结果输入**mask branch**中，做到快速和高精度的分割。

### Mask branch

![](https://pic.downk.cc/item/5ee1d0abc2a9a83be5454acf.jpg)

经过卷积层和转置卷积层，最终输出通道数为$38$层，包括：
- **Globa word map**：$1$层，对文本类别的分割；可以对文本区域进行精确定位，而不管文本实例的形状如何。
- **Character map**：$36$层，代表$10$个数字和$26$个大写字母的概率。
- **Background map**：$1$层，不包括字符区域。

### Pixel Voting

![](https://pic.downk.cc/item/5ee1d2dec2a9a83be5489a00.jpg)

测试时使用**Pixel Voting**，即对**Background map**进行二值化，再对其中属于文本的区域对应的**Character map**取最大概率，对应某个字符。

### Weighted Edit Distance
使用**Weighted Edit Distance**衡量两个字符串的相似程度，作为损失函数，其中$p$表示该字符对应的预测概率：

![](https://pic.downk.cc/item/5ee1d519c2a9a83be54c0c5c.jpg)

该方法的主要限制：受**Character map**数量限制，不能检测字符很多的文本（如中文）。
