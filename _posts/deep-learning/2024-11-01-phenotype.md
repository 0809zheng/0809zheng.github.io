---
layout: post
title: '表型图像分析(Phenotypic Image Analysis)'
date: 2024-11-01
author: 郑之杰
cover: ''
tags: 深度学习
---

> Phenotypic Image Analysis.

表型(**Phenotype**)是指生物体的外部特征和性状（比如形态、结构、颜色、纹理等）。作为生物信息和深度学习的交叉领域，表型图像分析旨在利用计算机视觉技术对生物体的表型特征进行可视化和量化，进一步自动捕获和分析与生物体的形态、结构和生长模式相关的数据。

表型图像分析广泛应用于植物学、医学、遗传学等领域，以帮助研究人员理解生物体的特征与其基因型和环境之间的关系。通过分析这些图像数据，研究人员可以进行更详细的表型评估，有助于加速生物研究和育种等应用。

本文前序章节记录了一些**前沿的**表型图像分析相关工作，并按照顶会的投递顺序进行整理。（值得一提的是，相关工作均是通过[**OpenReview**](https://openreview.net/)在顶会出分的时候进行整理的，并不代表该工作最后**被接收**。检索关键词包括：**phenotype, plant, argi, bio**）本文最后一章汇总了表型图像分析领域的**视觉基础模型**相关工作。

- **ICLR 2025 Submissions**
1. CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale
2. DODA: Diffusion for Object-detection Domain Adaptation in Agriculture
- **Vision Foundation Model of Plant Phenotyping**
1. Adapting the Segment Anything Model for Plant Recognition and Automated Phenotypic Parameter Measurement
2. Adapting Vision Foundation Models for Plant Phenotyping



## 👉 ICLR 2025 Submissions

### ⚪ [<font color=blue>CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale</font>](https://0809zheng.github.io/2024/11/15/clibd.html)

**CLIBD**旨在融合生物图像、分类类别和DNA数据，以构建一个强大的多模态表示空间，以提高生物多样性监测的准确性。该模型包括一个图像编码器、一个DNA编码器和一个文本编码器。

**CLIBD**在训练时采用对比学习策略，通过最大化相同物种的图像、DNA和文本特征之间的相似性，同时最小化不同物种之间的相似性，来学习一个有效的多模态表示空间。

**CLIBD**在推理时可以在零样本设置下对新物种进行评估。为了预测分类标签，计算输入图像与从可用物种中采样的 DNA 嵌入之间的余弦相似度，使用与最接近的键匹配的分类标签（目、科、属、种）作为预测。

![](https://pic.imgdb.cn/item/67371781d29ded1a8c856edd.png)

### ⚪ [<font color=blue>DODA: Diffusion for Object-detection Domain Adaptation in Agriculture</font>](https://0809zheng.github.io/2024/11/22/doda.html)

本文提出了一个具有领域特征自适应和布局图像条件生成的框架**DODA**，以增强扩散模型生成新的农业领域检测数据的能力。只需从目标域获取少量参考图像，**DODA**就可以为其生成数据，而无需额外的训练。

**DODA**通过将领域信息集成到**L2I**扩散中来实现领域感知图像的生成。首先将布局表示为图像，然后使用预训练的布局编码器提取特征作为域嵌入，最后通过特征加法融合整合到**L2I**扩散中。

![](https://pic.imgdb.cn/item/674051e4d29ded1a8cda4c4a.png)

## 👉 Vision Foundation Model of Plant Phenotyping

### ⚪ [<font color=blue>Adapting the Segment Anything Model for Plant Recognition and Automated Phenotypic Parameter Measurement</font>](https://0809zheng.github.io/2024/11/03/eclip.html) (Horticulturae 2024)

本文开发了一套零样本植物成分识别和表型性状测量框架：
1. 预处理：颜色校准和图像对齐，计算坐标转换的比例因子；
2. 无标签分割：使用**ECLIP**通过植物部位的文本描述在图像上生成前景和背景点位置；通过这些点引导**SAM**识别和分割植物成分；最后消除和合并错误分割的区域。
3. 表型性状测量：对输出掩码进行骨架化并通过b样条曲线拟合，可以精确测量植物的各种表型性状，如宽度和长度。

![](https://pic.imgdb.cn/item/673ee546d29ded1a8ccc6f68.png)

### ⚪ [<font color=blue>Adapting Vision Foundation Models for Plant Phenotyping</font>](https://0809zheng.github.io/2024/11/02/avfm.html) (ICCVW 2023)

本文研究了视觉基础模型对植物表型任务的适应性。作者对三个植物表型任务（叶片计数、实例分割和疾病分类）进行了实验，对**MAE**、**DINO**和**DINOv2**三个基础模型进行了微调，同时评估两种不同的微调方法：**LoRA**和解码器微调。实验结果表明，视觉基础模型可以有效地适应多种植物表型任务，其性能与专门为每种任务设计**SoTA**模型相似。但是在某些情况下，微调的基础模型比特定任务的**SoTA**模型执行得稍微差一些。

![](https://pic.imgdb.cn/item/673ece96d29ded1a8cba1a40.png)
