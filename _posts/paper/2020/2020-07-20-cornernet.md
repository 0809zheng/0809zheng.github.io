---
layout: post
title: 'CornerNet: Detecting Objects as Paired Keypoints'
date: 2020-07-20
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f15ac1a14195aa59450bfc6.jpg'
tags: 论文阅读
---

> CornerNet：一种anchor-free的目标检测算法.

- paper：CornerNet: Detecting Objects as Paired Keypoints
- arXiv：[link](https://arxiv.org/abs/1808.01244)

# 模型介绍
作者提出了一个anchor-free的目标检测算法：**CornerNet**。

![](https://pic.downk.cc/item/5f15ac4a14195aa59450d7aa.jpg)

**CornerNet**由**backbone network**和**prediction module**组成。
- **backbone network**选用**Hourglass Network**，将输入图像提取成特征映射；
- **prediction module**使用了**corner pooling**，将特征映射转化成三个输出，分别为：
1. **heatmaps**：分别预测每一个像素位置是某一类边界框的左上角坐标、右下角坐标的概率；
2. **embeddings**：对每一个可能为边界框左上角坐标、右下角坐标的位置进行编码，属于同一个边界框的编码应尽可能接近；否则编码尽可能相差较大；
3. **offsets**：把边界框的预测角点回归到像素点上。

# corner pooling
预测边界框的左上角坐标、右下角坐标，在预测点附近并没有出现物体，因此不能仅依赖预测点附近的感受野。作者提出了**corner pooling**，可以获取目标的边界信息。

以**top-left corner pooling**为例，该点的结果由该点右边和下边所有特征点数值最大者决定：

![](https://pic.downk.cc/item/5f15b29a14195aa59453ff03.jpg)

**corner pooling**在网络中应用如下，网络还使用了残差连接：

![](https://pic.downk.cc/item/5f15b32714195aa594543ce4.jpg)

# 损失函数
数据集在标注**“Ground truth**时并不是仅把目标边界框的左上角和右下角像素位置标注为1，而是使用一个二维高斯分布作为软标注：

![](https://pic.downk.cc/item/5f15b49e14195aa59454f5de.jpg)

网络的损失函数为：

$$ L = L_{det} + αL_{pull} + βL_{push} + γL_{off} $$

- $L_{det}$是目标损失，使用交叉熵损失：

![](https://pic.downk.cc/item/5f15b4d714195aa59455146d.jpg)

- $L_{pull}$用于使同一个边界框的左上角和右下角编码尽可能接近，具体地使其接近二者的平均值：

$$ L_{pull} = \frac{1}{N} \sum_{k=1}^{N} {[(e_{t_k}-e_k)^2+(e_{b_k}-e_k)^2]} $$

- $L_{push}$用于使不同边界框的角点编码尽可能远离：

$$ L_{push} = \frac{1}{N(N-1)} \sum_{k=1}^{N} {\sum_{j=1,j≠k}^{N} {max(0,Δ-\mid e_k-e_j \mid)}} $$

- $L_{off}$用于对坐标位置进行修正：

$$ L_{off} = \frac{1}{N} \sum_{k=1}^{N} {\text{SmoothL1Loss}(o_k,\hat{o}_k)} $$
