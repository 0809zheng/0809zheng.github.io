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

本文主要记录了笔者阅读的表型图像分析相关工作，并按照顶会的投递顺序进行整理。值得一提的是，相关工作均是通过[**OpenReview**](https://openreview.net/)在顶会出分的时候进行整理的，并不代表该工作最后**被接收**。检索关键词包括：**phenotype, plant, argi, bio**。

- **ICLR 2025 Submissions**
1. CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale




## 👉 ICLR 2025 Submissions

### ⚪ [<font color=blue>CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale</font>](https://0809zheng.github.io/2024/11/15/clibd.html)

**CLIBD**旨在融合生物图像、分类类别和DNA数据，以构建一个强大的多模态表示空间，以提高生物多样性监测的准确性。该模型包括一个图像编码器、一个DNA编码器和一个文本编码器。

**CLIBD**在训练时采用对比学习策略，通过最大化相同物种的图像、DNA和文本特征之间的相似性，同时最小化不同物种之间的相似性，来学习一个有效的多模态表示空间。

**CLIBD**在推理时可以在零样本设置下对新物种进行评估。为了预测分类标签，计算输入图像与从可用物种中采样的 DNA 嵌入之间的余弦相似度，使用与最接近的键匹配的分类标签（目、科、属、种）作为预测。

![](https://pic.imgdb.cn/item/67371781d29ded1a8c856edd.png)
