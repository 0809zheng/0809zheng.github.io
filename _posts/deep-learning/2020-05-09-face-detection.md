---
layout: post
title: '人脸检测'
date: 2020-05-09
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ecf4fc7c2a9a83be5ede3fb.jpg'
tags: 深度学习
---

> Face Detection.

**人脸检测(Face Detection)**：指对于任意一幅给定的图像，采用一定的策略对其进行搜索以确定其中是否含有人脸，如果是则返回人脸的位置、大小和姿态，是人脸验证、识别、表情分析等各种问题的关键步骤。

本文目录：
1. **Benchmarks**
2. **Eigenface**
3. **SSH**

# 1. Benchmarks
人脸检测常用的数据集包括：
- **WIDER FACE**：人脸检测的一个基准数据集，包含32,203张图像，393703个标注人脸，其中158,989张在训练集中，39,496张在验证集中，其余的在测试集中。验证集和测试集包含 “easy”“medium”, “hard” 三个子集，且依次包含，即“hard”数据集包含所有“easy”和“medium”的图像。
- **FDDB**：FDDB是全世界最具权威的人脸检测数据集之一，包含2845张图片，共有5171个人脸作为测试集。
- **Pascal Faces**：PASCAL VOC为图像识别和分类提供了一整套标准化的优秀的数据集，Pascal Faces是PASCAL VOC的子集，共851张包含已标注人脸的图像，在本论文中仅用于评估模型性能。

# 2. Eigenface
**Eigenface**是一种基于机器学习的人脸检测方法。**Eigenface**的思路是把人脸图像从像素空间变换到特征空间，在特征空间中进行相似度计算。具体地，使用**主成分分析**得到人脸数据矩阵的特征向量，这些特征向量可以看作组成人脸的特征，每个人脸都可以表示成这些特征向量的线性组合。**Eigenface**算法流程如下：
1. 将训练集表示成二维张量$A \in \Bbb{R}^{d \times N}$，其中$d$表示每张人脸图像的像素数，$N$表示训练集大小；
2. 求训练集中所有图像的平均人脸$\overline{A} = A.sum(dim=1)$，训练集减去该平均后得到差值图像的数据矩阵$\Phi = A - \overline{A}$；
3. 计算数据矩阵的协方差矩阵$C = \Phi \Phi^T$，对其进行特征值分解，得到特征向量$v$；
4. 对于测试图像$T$，将其投影到这些特征向量上，计算特征向量$v_k$相对于测试图像的权重$\omega_k = v_k^T(T-\overline{A})$，取前$M$个特征向量，得到权重矩阵$\omega_T = \[ \omega_1,\omega_2,...,\omega_M \]$；
5. 对训练集中的每一张人脸图像计算权重矩阵$\omega$，比较测试图像的权重矩阵与其之间的欧式距离$$\epsilon_T = \| \omega_T - \omega \|^2$$，若距离小于阈值，则认为测试图像和该训练图像对应同一个人脸。若测试图像与训练集的所有图像距离都大于阈值，则认为测试图像不包含人脸。

在实际中，通常人脸图像的像素数较大，而训练集的样本总数较少，即$d>>N$。在计算协方差矩阵$C = \Phi \Phi^T \in \Bbb{R}^{d \times d}$时会得到较大的矩阵，不利于后续的特征值分解。此时可以先计算另一个协方差矩阵，$C' = \Phi^T \Phi \in \Bbb{R}^{N \times N}$，计算其特征值为$u$，则矩阵$C$的特征值计算为$v=\Phi u$，推导如下：

$$ C' \cdot u = \lambda \cdot u \\ \Phi^T \Phi \cdot u = \lambda \cdot u \\ \Phi \Phi^T \Phi \cdot u = \lambda \cdot \Phi u \\ C \Phi \cdot u = \lambda \cdot \Phi u \\ C \cdot v = \lambda \cdot v $$

# 3. SSH
- paper：[SSH: Single Stage Headless Face Detector](https://arxiv.org/abs/1708.03979)

**SSH**是一个快速、轻量级的人脸检测器，直接从分类网络中的早期卷积层以单阶段方式检测人脸。

网络结构如图所示：

![](https://pic.downk.cc/item/5ecf4f8ac2a9a83be5eda2df.jpg)

网络使用**VGG16**作为**backbone**，分成$3$路进行不同尺度的检测，使得模型对于图像中不同尺寸大小脸的检测均具有良好的鲁棒性。

**检测模块（detection module）**分为**M1**、**M2**和**M3**，分别检测小、中、大尺寸的人脸，如下图所示，其中又使用了**上下文模块（context module）**：

![](https://pic.downk.cc/item/5ecf519bc2a9a83be5f009fd.jpg)

**上下文模块（context module）**如下图所示，通过$2$个$3 \times 3$的卷积层和$3$个$3 \times 3$的卷积层并联，用来读取图像的文本信息。

![](https://pic.downk.cc/item/5ecf52c5c2a9a83be5f159e4.jpg)