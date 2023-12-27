---
layout: post
title: '开放集合目标检测(Open-Set Object Detection)'
date: 2023-11-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/658aa19ac458853aef584c92.jpg'
tags: 深度学习
---

> Open-Set Object Detection.

**开集目标检测(Open-Set Object Detection, OSOD)**也被称为**Open-World**目标检测或**Open-Vocabulary**目标检测，是指在可见类（**base class**）的数据上进行训练，然后完成对不可见类（**unseen/target**）数据的定位与识别。

开集目标检测的出发点是制定一种更加通用的目标检测模型来覆盖更多的**object concept**，使得目标检测不再受限于带标注数据的少数类别，从而实现更加泛化的目标检测，识别出更多**novel**的物体类别。

注意：早期的**Open-Set**目标检测与**Open-Vocabulary**目标检测所指代含义是不同的。前者倾向于检测出未知目标即可，不再判别目标的具体类别；后者则需要根据指定的新类别进行检测。本文则不再进行区分。

一些常见的开集目标检测方法包括：
- 基于无监督学习的开集检测器：通过聚类、弱监督等手段实现开集检测，如**OSODD**, **Detic**, **VLDet**
- 基于多模态学习的开集检测器：
1. 基于**Referring**的开集检测器：借助多模态视觉-语言模型实现检测，如**ViLD**, **RegionCLIP**, **VL-PLM**, **Grad-OVD**
2. 基于**Grounding**的开集检测器：把开集检测任务建模为边界框提取+短语定位任务，如**OVR-CNN**, **MDETR**, **GLIP**, **DetCLIP**, **DetCLIPv2**, **Grounding DINO**

# 1. 基于无监督学习的开集检测器

### ⚪ OSODD
- paper：[<font color=blue>Towards Open-Set Object Detection and Discovery</font>](https://0809zheng.github.io/2023/11/04/osodd.html)


**OSODD**采用阶段检测方式：**Object Detection and Retrieval (ODR)** 和 **Object Category Discovery (OCD)**。
- **ODR**对已知物体和未知物体进行检测（通过对类别无感知的**RPN**实现）：已知物体预测位置信息和类别；未知物体只预测其位置信息。
- **OCD**对已知物体和未知物体的区域特征进行编码，并用 **constrained k-means**对未知类别进行聚类。

![](https://pic.imgdb.cn/item/655c6f42c458853aef59a09d.jpg)

### ⚪ Detic
- paper：[<font color=blue>Detecting Twenty-thousand Classes using Image-level Supervision</font>](https://0809zheng.github.io/2023/11/09/detic.html)

**Detic**使用图像分类数据集和目标检测数据集一起联合训练。对于分类图像，由**RPN**获取**proposal**，选取最大面积的**proposal**，这个**proposal**对应的**label**就是图像层面的类别。

![](https://pic.imgdb.cn/item/658933bec458853aef946211.jpg)

### ⚪ VLDet
- paper：[<font color=blue>Learning Object-Language Alignments for Open-Vocabulary Object Detection</font>](https://0809zheng.github.io/2023/11/13/vldet.html)

**VLDet**直接从**image-text pairs**训练目标检测器，主要出发点是从**image-text pairs**中提取**region-word pairs**可以表述为两个集合的元素匹配问题，该问题可以通过找到区域和单词之间具有最小全局匹配成本的二分匹配来有效解决。

![](https://pic.imgdb.cn/item/658b9483c458853aef13b6c9.jpg)


# 2. 基于多模态学习的开集检测器

当人类捕捉到视觉信息后，会将它们与语言文字联系起来，从而产生丰富的视觉和语义词汇，这些词汇可以用于检测物体，并拓展模型的表达能力。因此通过多模态学习技术借助大量的图像与语义词汇，可以使得目标检测器在未知类别也能进行识别与定位。


## （1）基于Referring的开集检测器

预训练的开放词汇图像分类模型（如**CLIP**）是一种在大量图像和文本对上训练的多模态模型。这类模型可用于查找最能代表图像的文本片段，或查找给定文本查询的最合适图像。

基于**Referring**的开集检测器是指将开放词汇图像分类模型中视觉端的能力迁移至检测模型中，再利用文本编码器完成检测模型的识别工作。


### ⚪ ViLD
- paper：[<font color=blue>Open-vocabulary Object Detection via Vision and Language Knowledge Distillation</font>](https://0809zheng.github.io/2023/11/07/vild.html)

**ViLD**通过**CLIP**的视觉编码器蒸馏训练**R-CNN**中的区域提议网络，再使用**CLIP**的文本编码器对目标区域进行分类。

![](https://pic.imgdb.cn/item/65701bf4c458853aef0a6a30.jpg)

### ⚪ RegionCLIP
- paper：[<font color=blue>RegionCLIP: Region-based Language-Image Pretraining</font>](https://0809zheng.github.io/2023/11/12/regionclip.html)

**CLIP**在**Region**区域上的识别很差，这是由于**CLIP**是在**Image-Language level**上进行的预训练导致的。**RegionCLIP**将**CLIP**在**region**图像和单词层面进行预训练，提高了区域级别的检测能力。

![](https://pic.imgdb.cn/item/658a7ae7c458853aefdff48d.jpg)

### ⚪ VL-PLM
- paper：[<font color=blue>Exploiting Unlabeled Data with Vision and Language Models for Object Detection</font>](https://0809zheng.github.io/2023/11/11/vlplm.html)

**VL-PLM**利用**CLIP**对无标签数据进行伪标签标注，然后混合伪标签数据和真实标注数据一起训练。注意到**CLIP**对于区域级别的图像识别能力不足，**VL-PLM**反复将**ROI**输入**ROI head**精调，最后**RPN**的分数将和**CLIP**的分类分数作平均。

![](https://pic.imgdb.cn/item/658a947bc458853aef34c850.jpg)

### ⚪ Grad-OVD
- paper：[<font color=blue>Open Vocabulary Object Detection with Pseudo Bounding-Box Labels</font>](https://0809zheng.github.io/2023/11/08/gradovd.html)

给定一个预训练的视觉语言模型和一个图像-描述样本对，**Grad-OVD**在图像中计算**Grad-CAM**激活图，对应于描述中感兴趣的目标。然后将激活映射转换为对应类别的伪标签框。开放词汇检测器可以直接在这些伪标签的监督下进行训练。

![](https://pic.imgdb.cn/item/6585330dc458853aef21b989.jpg)

## （2）基于Grounding的开集检测器

短语定位（**phrase grounding**）任务是指同时提供一张图像和一段描述图像的文本（**captions**），根据文本的描述信息从图像中找到对应的物体。

基于**Grounding**的开集检测器是指把开集目标检测任务建模为边界框提取+短语定位任务。该过程引入大规模**caption**数据集完成**region-word**级别的视觉模型和语言模型预训练。

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

### ⚪ DetCLIP
- paper：[<font color=blue>DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection</font>](https://0809zheng.github.io/2023/11/14/detclip.html)

**GLIP** 在 **text encoder** 中会学习所有类别之间的 **attention**，**DetCLIP**将每个 **concept** 分别送入不同的 **text encoder**。这样能够避免模型受到不相关类别无效关联，并且能给每个 **concept** 都产生一个长描述。

![](https://pic.imgdb.cn/item/658bcb3cc458853aefce62d3.jpg)

### ⚪ DetCLIPv2
- paper：[<font color=blue>DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment</font>](https://0809zheng.github.io/2023/11/15/detclipv2.html)

**DetCLIPv2**是一个面向开放词汇目标检测的统一端到端预训练框架，通过使用区域和单词之间的最佳匹配集相似性来引导对比目标，以端到端的方式执行检测、定位和图像对数据的联合训练。

![](https://pic.imgdb.cn/item/658be095c458853aef1b3bf3.jpg)

### ⚪ Grounding DINO
- paper：[<font color=blue>Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection</font>](https://0809zheng.github.io/2023/11/02/groundingdino.html)

**Grounding DINO**采用双编码器、单解码器结构。对于输入的每个(图像，文本)对，首先分别使用图像主干网络和文本主干网络提取图像特征和文本特征；两个特征被送入特征增强模块，用于跨模态特征融合；获得跨模态文本和图像特征后，使用语言引导的查询选择模块从图像特征中选择跨模态的目标查询；这些跨模态查询被送到跨模态解码器中，用于预测目标框并提取相应的短语。

![](https://pic.imgdb.cn/item/6555809dc458853aef89eec8.jpg)

