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
1. Benchmarks
2. SSH

# 1. Benchmarks
人脸检测常用的数据集包括：
- **WIDER FACE**：人脸检测的一个基准数据集，包含32,203张图像，393703个标注人脸，其中158,989张在训练集中，39,496张在验证集中，其余的在测试集中。验证集和测试集包含 “easy”“medium”, “hard” 三个子集，且依次包含，即“hard”数据集包含所有“easy”和“medium”的图像。
- **FDDB**：FDDB是全世界最具权威的人脸检测数据集之一，包含2845张图片，共有5171个人脸作为测试集。
- **Pascal Faces**：PASCAL VOC为图像识别和分类提供了一整套标准化的优秀的数据集，Pascal Faces是PASCAL VOC的子集，共851张包含已标注人脸的图像，在本论文中仅用于评估模型性能。

# 2. SSH
- paper：[SSH: Single Stage Headless Face Detector](https://arxiv.org/abs/1708.03979)

SSH是一个快速、轻量级的人脸检测器，直接从分类网络中的早期卷积层以单阶段方式检测人脸。

网络结构如图所示：

![](https://pic.downk.cc/item/5ecf4f8ac2a9a83be5eda2df.jpg)

网络使用VGG16作为backbone，分成3路进行不同尺度的检测，使得模型对于图像中不同尺寸大小脸的检测均具有良好的鲁棒性。

**检测模块（detection module）**分为M1、M2和M3，分别检测小、中、大尺寸的人脸，如下图所示，其中又使用了**文本模块（context module）**：

![](https://pic.downk.cc/item/5ecf519bc2a9a83be5f009fd.jpg)

**文本模块（context module）**如下图所示，通过2个3x3的卷积层和3个3x3的卷积层并联，用来读取图像的文本信息。

![](https://pic.downk.cc/item/5ecf52c5c2a9a83be5f159e4.jpg)