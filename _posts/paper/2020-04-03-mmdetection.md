---
layout: post
title: 'MMDetection: Open MMLab Detection Toolbox and Benchmark'
date: 2020-04-03
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e8722ef504f4bcb04dd0317.jpg'
tags: 论文阅读
---

> 商汤科技和香港中文大学开源的基于Pytorch实现的深度学习目标检测工具箱。

- github:[官方github](https://github.com/open-mmlab/mmdetection)
- docs:[官方文档](https://mmdetection.readthedocs.io/en/latest/index.html)
- paper:[MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/abs/1906.07155v1)

MMDetection的特点：
1. Mudule design
2. Support of multiple frameworks out of box
3. High efficiency
4. State of the art

目录：
1. Architecture
2. Hyper-parameters
3. Supported Frameworks

# 1. Architecture

### (1). Model Representation
目标检测模型通常由不同的组件构成。MMDetection中的基本组件包括：
- **Backbone**：把image转换成feature map，如ResNet-50；
- **Neck**：连接backbone和head，对raw feature map进行refine和reconfigure，如FPN；
- **DenseHead (AnchorHead/AnchorFreeHead)**：进行dense location，包括AnchorHead和AnchorFreeHead，如RPN-Head、RetinaHead、FCOSHead;
- **RoIExtractor**：提取RoI(Region of Interests)，如RoIPooling、RoIAlign；
- **RoIHead (BBoxHead/MaskHead)**：使用RoI进行预测，如bbox分类/回归、mask预测。

用基本组件组成的Single-stage和Two-stage的模型框架：
![](https://pic.downk.cc/item/5e872d4c504f4bcb04e828a4.jpg)

### (2). Training Pipeline
训练（training epochs）和验证（validation epochs）时设计了hooking mechanism。训练时的pipeline如下，验证类似。
![](https://pic.downk.cc/item/5e872e71504f4bcb04e94666.jpg)

# 2. Hyper-parameters

### (1). Regression Losses
目标检测的损失包括边界框回归损失和分类损失。可选的回归损失如下：
- Smooth L1 Loss
- L1 Loss
- Balanced L1 Loss
- Bounded IoU Loss
- IoU Loss
- GIoU Loss

不同的回归损失和损失系数搜索如下：
![](https://pic.downk.cc/item/5e872f49504f4bcb04ea1564.jpg)

### (2). Normalization Layers
目标检测训练时batch较小，使用Batch Normalization时设置如下：
- ```eval = True```：评估时冻结统计量
- ```requires_grad = True```：允许rescale和reshift的学习

可选的Normalization：
- FrozenBN
- Synchronized BN
- Group Normalization(G=32)

不同的Normalization及其应用的位置搜索如下：
![](https://pic.downk.cc/item/5e8730c0504f4bcb04eb7a6f.jpg)

### (3). Training Scales
目标检测默认的训练scale为1333×800。

MMDetection允许multi-scale训练，即每次训练时选择一个输入image的scale，提供了两种scale选择方法：
- **“value” mode**：shorter edge从列表中选择
- **“range” mode**：shorter edge在选择

不同的training scale搜索如下：
![](https://pic.downk.cc/item/5e8731f0504f4bcb04ec5570.jpg)

### (4). Other Hyper-parameters
其他超参数包括：
- **smoothl1 beta**：Smooth L1 Loss的参数```torch.where(x < beta, 0.5·x^2/beta, x - 0.5·beta)```
- **allowed border**：边界框超出原图像的允许上界
- **neg pos ub**：negative和positive样本比例

这些超参数搜索如下：
![](https://pic.downk.cc/item/5e873323504f4bcb04ed09e2.jpg)

# 3. Supported Frameworks

### (1). Single-stage Methods
- **SSD**： a classic and widely used single-stage detector with simple model architecture
- **RetinaNet**： a high-performance single-stage detector with Focal Loss
- **GHM**: a gradient harmonizing mechanism to improve single-stage detectors
- **FCOS**: a fully convolutional anchor-free singlestage detector
- **FSAF**: a feature selective anchor-free module for single-stage detectors

### (2). Two-stage Methods
- **Fast R-CNN**: a classic object detector which requires pre-computed proposals
- **Faster R-CNN**: a classic and widely used twostage object detector which can be trained end-to-end
- **R-FCN**: a fully convolutional object detector with faster speed than Faster R-CNN
- **Mask R-CNN**: a classic and widely used object detection and instance segmentation method
- **Grid R-CNN**: a grid guided localization mechanism as an alternative to bounding box regression
- **Mask Scoring R-CNN**: an improvement over Mask R-CNN by predicting the mask IoU
- **Double-Head R-CNN**: different heads for classi-fication and localization

### (3). Multi-stage Methods
- **Cascade R-CNN**: a powerful multi-stage object detection method
- **Hybrid Task Cascade**: a multi-stage multi-branch object detection and instance segmentation method

### (4). General Modules and Methods
- **Mixed Precision Training**: train deep neural networks using half precision floating point (FP16) numbers
- **Soft NMS**: an alternative to NMS
- **OHEM**: an online sampling method that mines hard samples for training
- **DCN**: deformable convolution and deformable RoI pooling
- **DCNv2**: modulated deformable operators
- **Train from Scratch**: training from random initialization instead of ImageNet pretraining
- **ScratchDet**: another exploration on training from scratch
- **M2Det**: a new feature pyramid network to construct more effective feature pyramids
- **GCNet**: global context block that can efficiently model the global context
- **Generalized Attention**: a generalized attention formulation
- **SyncBN**: synchronized batch normalization across GPUs
- **Group Normalization**: a simple alternative to BN
- **Weight Standardization**: standardizing the weights in the convolutional layers for micro-batch training
- **HRNet**: a new backbone with a focus on learning reliable high-resolution representations
- **Guided Anchoring**: a new anchoring scheme that predicts sparse and arbitrary-shaped anchors
- **Libra R-CNN**: a new framework towards balanced learning for object detection

一些frameworks的表现如下：
![](https://pic.downk.cc/item/5e873649504f4bcb04ef44c0.jpg)