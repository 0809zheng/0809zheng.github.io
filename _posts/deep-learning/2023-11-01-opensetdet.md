---
layout: post
title: '开放集合目标检测(Open-Set Object Detection)'
date: 2023-11-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63f48f89f144a0100755912f.jpg'
tags: 深度学习
---

> Open-Set Object Detection.

**开集目标检测(Open-Set Object Detection, OSOD)**也被称为**Open-World**目标检测或**Open-Vocabulary**目标检测，是指目标检测模型能够检测任意给定的目标类别（这些类别可能不是预定义的训练集类别）。

注意：早期的**Open-Set**目标检测与**Open-Vocabulary**目标检测所指代含义是不同的。前者倾向于检测出未知目标即可，不再判别目标的具体类别；后者则需要根据指定的新类别进行检测。本文则不再进行区分。

# 1. 基于无监督学习的开集检测器

### ⚪ OSODD
- paper：[<font color=blue>Towards Open-Set Object Detection and Discovery</font>](https://0809zheng.github.io/2023/11/04/osodd.html)


**OSODD**采用阶段检测方式：**Object Detection and Retrieval (ODR)** 和 **Object Category Discovery (OCD)**。
- **ODR**对已知物体和未知物体进行检测（通过对类别无感知的**RPN**实现）：已知物体预测位置信息和类别；未知物体只预测其位置信息。
- **OCD**对已知物体和未知物体的区域特征进行编码，并用 **constrained k-means**对未知类别进行聚类。

![](https://pic.imgdb.cn/item/655c6f42c458853aef59a09d.jpg)

# 2. 基于多模态学习的开集检测器

开集目标检测。开集目标检测是利用已有的边界框标注进行训练，目的是借助语言泛化来检测任意类。DetCLIP[53]涉及大规模图像字幕数据集，并使用生成的伪标签扩展知识库。生成的伪标签有效地扩展了检测器的泛化能力。


## （1）基于Referring的开集检测器

预训练的开放词汇图像分类模型（如**CLIP**）是一种在大量图像和文本对上训练的多模态模型。这类模型可用于查找最能代表图像的文本片段，或查找给定文本查询的最合适图像。

基于**Referring**的开集检测器是指将开放词汇图像分类模型中视觉端的能力迁移至检测模型中，再利用文本编码器完成检测模型的识别工作。


### ⚪ ViLD
- paper：[<font color=blue>Open-vocabulary Object Detection via Vision and Language Knowledge Distillation</font>](https://0809zheng.github.io/2023/11/07/vild.html)

**ViLD**通过**CLIP**的视觉编码器蒸馏训练**R-CNN**中的区域提议网络，再使用**CLIP**的文本编码器对目标区域进行分类。

![](https://pic.imgdb.cn/item/65701bf4c458853aef0a6a30.jpg)

## （2）基于Grounding的开集检测器

基于**Grounding**的开集检测器是指把开集目标检测任务建模为边界框提取+短语对齐（**phrase grounding**）任务。

### ⚪ OVR-CNN
- paper：[<font color=blue>Open-Vocabulary Object Detection Using Captions</font>](https://0809zheng.github.io/2023/11/06/ovrcnn.html)

**OVR-CNN**使用一个 **Vision to Language（V2L）**映射层将视觉特征变换到文本嵌入空间，使得两个不同模态的特征能在同一空间来衡量相似性。**V2L**通过**grounding**, **masked language modeling (MLM)**和**image-text matching (ITM)**任务预训练后，把预测边界框的特征嵌入后与预先给定类别标签的文本特征计算相似度来进行分类。

![](https://pic.imgdb.cn/item/655f16d0c458853aef4de228.jpg)

### ⚪ MDETR
- paper：[<font color=blue>MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding</font>](https://0809zheng.github.io/2023/11/05/mdetr.html)

**MDETR**采用**CNN**提取视觉特征，采用语言模型提取文本特征，将两者**concat**后通过**DETR**预测目标框。

![](https://pic.imgdb.cn/item/655d9d3ac458853aefcca84d.jpg)

### ⚪ GLIP
- paper：[<font color=blue>Grounded Language-Image Pre-training</font>](https://0809zheng.github.io/2023/11/03/glip.html)

**GLIP**把图像特征与文本特征分别由单独的编码器编码，然后通过跨模态多头注意力模块（**X-MHA**）进行深度融合。

![](https://pic.imgdb.cn/item/655c6111c458853aef2d9091.jpg)

### ⚪ Grounding DINO
- paper：[<font color=blue>Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection</font>](https://0809zheng.github.io/2023/11/02/groundingdino.html)

**Grounding DINO**采用双编码器、单解码器结构。对于输入的每个(图像，文本)对，首先分别使用图像主干网络和文本主干网络提取图像特征和文本特征；两个特征被送入特征增强模块，用于跨模态特征融合；获得跨模态文本和图像特征后，使用语言引导的查询选择模块从图像特征中选择跨模态的目标查询；这些跨模态查询被送到跨模态解码器中，用于预测目标框并提取相应的短语。

![](https://pic.imgdb.cn/item/6555809dc458853aef89eec8.jpg)

### ⚪ Co-DETR
- paper：[<font color=blue>DETRs with Collaborative Hybrid Assignments Training</font>](https://0809zheng.github.io/2023/11/02/groundingdino.html)