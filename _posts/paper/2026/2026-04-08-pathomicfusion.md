---
layout: post
title: 'Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis'
date: 2026-04-08
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/69d5c562cfae5fcb267c22a9.png'
tags: 论文阅读
---

> Pathomic Fusion: 融合病理与基因组学特征进行癌症诊断与预后的集成框架.

- paper：[Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis](https://ieeexplore.ieee.org/document/9186053)

# 0. TL; DR

**Pathomic Fusion**是一个用于端到端多模态融合的框架。该框架能够整合**histology**图像和**genomic**数据（包括突变、**CNV**、**RNA-Seq**），用于生存结局预测。模型通过对单模态特征表示进行**Kronecker**积，显式地捕捉不同模态之间的成对特征交互。通过一个基于门控的注意力机制来控制每种模态表示的表达强度。

作者使用**The Cancer Genome Atlas** (**TCGA**) 的**glioma**（胶质瘤）和**clear cell renal cell carcinoma**（透明细胞肾细胞癌）数据集对该方法进行了验证。在15折交叉验证中，**Pathomic Fusion**在预后预测方面的表现优于基于人工分级和分子分型的传统方法，也优于仅使用**histology**或**genomic**数据训练的单模态深度网络。

# 1. 背景介绍

癌症的诊断、预后和治疗反应预测通常依赖于异构的数据来源，包括**histology**幻灯片、分子谱以及患者年龄等临床数据。对肿瘤微环境进行基于**histology**的主观定性分析，再结合基因组检测的定量检查，是目前大多数癌症的临床诊疗标准。随着病理学从玻璃切片向数字化**whole slide images** (**WSI**) 迁移，开发能够以综合方式利用表型和基因型信息的算法迎来了关键机遇。

然而，传统的诊疗范式存在一些局限：
*   主观性与变异性：病理学家对**histology**的主观解读存在巨大的观察者间和观察者内差异。
*   信息利用不充分：主观判读未能充分利用**histology**图像中丰富的、具有预后相关性的表型信息。
*   基因组分析的局限：基因组分析虽然提供了定量的分子信息，但无法精确地将肿瘤诱导的基因型变化与非肿瘤实体（如正常细胞）区分开。

尽管单细胞测序和空间转录组学等新兴技术能够更高分辨率地解析这些信息，但它们尚未在临床普及。因此，肿瘤科医生仍然依赖于定性的**histology**信息和定量的基因组数据来预测临床结局。

现有的大多数**histology**分析范式没有整合基因组信息，也未能明确地利用细胞空间组织和群落结构中已知的诊断和预后信息。将**histology**的形态学信息与**genomics**的分子信息融合，为更准确地量化肿瘤微环境、开发用于早期诊断、预后、患者分层和治疗反应预测的“影像-组学”检测方法提供了激动人心的可能性。

# 2. Pathomic Fusion 框架

**Pathomic Fusion**框架的核心思想是：首先为每种数据模态（**histology image**, **cell graph**, **genomic data**）学习一个独立的特征表示，然后通过一个精巧的融合机制将它们结合起来，用于最终的预测任务。

## 各数据模态的特征表示学习

### A. 从H&E组织学图像中学习 (CNN)

为了捕捉**histology**图像中的表型异质性，如高细胞密度和微血管增生等，作者使用了一个**CNN**。从**WSI**中提取的512x512大小的**ROIs**。使用在**ImageNet**上预训练的**VGG19**网络，并进行微调。从网络的最后一个隐藏层提取一个$h_i \in \mathbb{R}^{32 \times 1}$的嵌入向量，作为图像特征表示。对于生存预测，使用**Cox partial likelihood loss**；对于分级分类，使用交叉熵损失。

### B. 从形态学细胞和图特征中学习 (GCN)

与**CNN**直接处理像素不同，**GCNs**将**histology**图像显式地表示为一个细胞图，从而直接建模细胞间的相互作用和空间组织。

作者使用了一个**conditional generative adversarial network** (**cGAN**) 来进行精准的细胞核分割，从而定义图中的节点集合$V$。使用**K-Nearest Neighbors** (**KNN**) 算法（K=5）来构建图的边集合$E$和邻接矩阵$A$。

![](https://pic1.imgdb.cn/item/69d5c715cfae5fcb267c25fb.png)

为每个细胞计算8个轮廓特征（如长轴、短轴、圆度）和4个纹理特征（来自**GLCM**）。使用**contrastive predictive coding** (**CPC**) 从以每个细胞为中心的64x64图像块中提取1024维的深度特征。**CPC**通过最大化一个信号不同部分之间的互信息来学习丰富的特征表示。

作者采用**GraphSAGE**架构中的邻域聚合策略。在第$k$次迭代中，节点$v$的新特征$h_v^{(k)}$由其自身前一层的特征$h_v^{(k-1)}$和其邻居特征的聚合$a_v^{(k)}$组合而成：

$$
a_v^{(k)} = \text{AGGREGATE}^{(k)}\left( \left\{ h_u^{(k-1)} : u \in \mathcal{N}(v) \right\} \right)\\
h_v^{(k)} = \text{COMBINE}^{(k)}\left( h_v^{(k-1)}, a_v^{(k)} \right)
$$

其中，**AGGREGATE**函数是一个最大池化操作，**COMBINE**函数是一个拼接和线性变换。为了编码细胞图的层次结构，作者采用了**SAGPOOL**自注意力池化策略。该策略通过一个注意力机制自适应地学习每个节点在池化中的贡献，从而实现对图的局部池化。在**GCN**的最后一层，通过池化所有节点特征得到一个$h_g \in \mathbb{R}^{32 \times 1}$的图特征表示。

### C. 从分子谱中预测患者结局 (SNN)

对于高维、小样本的基因组数据（如**RNA-Seq**, **CNV**, **mutation**），传统的全连接网络容易过拟合。因此，作者采用了**Self-Normalizing Networks** (**SNNs**)。

**SNN**使用**scaled exponential linear units** (**SeLU**) 作为激活函数，而不是**ReLU**。这使得每一层的输出都能自动趋向于零均值和单位方差。结合一种名为**Alpha Dropout**的改进正则化技术，**SNN**能够被稳定地训练。

网络由四个全连接层组成，最后一个全连接层输出一个$h_n \in \mathbb{R}^{32 \times 1}$的基因组特征表示。

## Pathomic Fusion: 多模态融合策略

作者提出了一种新颖的多模态融合策略，它通过**Kronecker Product**（克罗内克积）和**Gating-Based Attention**（基于门控的注意力机制）来显式地建模不同模态间的交互。

![](https://pic1.imgdb.cn/item/69d5c7c0cfae5fcb267c2786.png)

### 1. 基于门控的注意力机制

在进行融合之前，为了减少冗余和噪声特征的影响，作者首先采用了一个**gating-based attention**机制来控制每种模态特征的表达强度。对于一个模态$m$的特征表示$h_m$，模型会学习一个注意力权重向量$z_m$，然后通过元素乘积得到门控后的特征表示$h_{m, \text{gated}}$。

$$
h_{m, \text{gated}} = z_m \odot h_m
$$

其中，$z_m = \sigma(W_{ign \to m} \cdot [h_i, h_g, h_n])$。这里的注意力权重$z_m$是由所有模态的特征共同决定的，可以理解为其他模态在“关注”模态$m$中的每一个特征。

### 2. 通过克罗内克积进行多模态张量融合

为了显式地捕捉不同模态特征之间的交互，作者使用了**Kronecker Product**（在这里等价于**outer product**，外积）。

将三个门控后的单模态特征向量（$h_{i,\text{gated}}$, $h_{g,\text{gated}}$, $h_{n,\text{gated}}$）进行外积，可以构建一个三阶的多模态张量$h_{\text{fusion}}$。

$$
h_{\text{fusion}} = \left[ \begin{matrix} h_{i,\text{gated}} \\ 1 \end{matrix} \right] \otimes \left[ \begin{matrix} h_{g,\text{gated}} \\ 1 \end{matrix} \right] \otimes \left[ \begin{matrix} h_{n,\text{gated}} \\ 1 \end{matrix} \right]
$$

通过在每个特征向量后拼接一个1，外积的结果张量$h_{\text{fusion}}$将不仅包含三模态交互项($h_i \otimes h_g \otimes h_n$)，还会保留所有的单模态特征($h_i, h_g, h_n$)和双模态交互项($h_i \otimes h_g, h_g \otimes h_n, h_i \otimes h_n$)。这个$33 \times 33 \times 33$的张量捕捉了所有可能的特征交互空间，然后被送入一个最终的全连接网络进行预测。

### 3. 多模态可解释性

为了理解模型的决策过程，作者修改了**Grad-CAM**和**Integrated Gradients** (**IG**) 两种方法。
*   **Grad-CAM**：用于生成**histology**图像的视觉解释，高亮出与高风险预测相关的像素区域。
*   **Integrated Gradients (IG)**：一种基于梯度的特征归因方法，用于量化**GCN**中的每个细胞和**SNN**中的每个基因特征对于最终预测的贡献度。

# 3. 实验分析

作者在**TCGA**的**glioma** (**TCGA-GBMLGG**) 和**clear cell renal cell carcinoma** (**CCRCC**, **TCGA-KIRC**) 数据集上进行了严格的15折交叉验证。

## 3.1 Pathomic Fusion在生存预测中的性能

在**glioma**和**CCRCC**两个数据集上，**Pathomic Fusion**的性能（**C-Index**分别为0.826和0.720）都显著优于所有单模态网络（**CNN**, **GCN**, **SNN**）、传统的临床标准（**WHO Paradigm**, **Fuhrman Grade**）以及简单的拼接融合方法（**Concatenation Fusion**）。

多模态模型始终优于其单模态基线，其中三模态融合（**CNN⊗GCN⊗SNN**）取得了最佳性能。这证明了融合**histology**和**genomic**信息的有效性。

在**glioma**中，虽然单独使用**GCN**的性能不如**CNN**，但将其整合到三模态融合中后，模型在区分低风险和中风险患者方面的能力得到了显著提升（p值从0.103降至2.68e-03）。这表明**GCN**捕捉到的细胞图谱特征为**CNN**提供了有价值的补充信息。

作者通过将相同模态输入融合网络（如**CNN⊗CNN**）的实验，证明了性能的提升并非源于简单的网络**ensembling**，因为这种操作反而导致了过拟合。

![](https://pic1.imgdb.cn/item/69d5c884cfae5fcb267c2902.png)

## 3.2 Pathomic Fusion改善患者分层

作者通过**Kaplan-Meier** (**KM**) 曲线进一步分析了**Pathomic Fusion**在患者风险分层上的能力。

对**Glioma**的分析：
*   **KM曲线 (B)**：与**Histology CNN**相比，**Pathomic Fusion**能够更好地辨别中风险和高风险患者。与分子分型相比，它能更好地辨别低风险和中风险患者。
*   **风险分布 (A)**：**Pathomic Fusion**预测的风险值形成了三个清晰的高密度聚类，这三个聚类与**WHO**范式中的三个主要分子亚型（**IDHwt ATC**, **IDHmut ATC**, **ODG**）高度吻合。而**Histology CNN**的风险预测则较为分散，未能清晰地区分这些亚型。

![](https://pic1.imgdb.cn/item/69d5c8c7cfae5fcb267c299e.png)

对**CCRCC**的分析：**Pathomic Fusion**能够很好地区分生存期较长和较短的患者，其风险预测呈现出明显的双峰分布。而**Histology CNN**的风险预测几乎是均匀分布的，表明单独的组织学信息在**CCRCC**中预后能力较差。

![](https://pic1.imgdb.cn/item/69d5c958cfae5fcb267c2b71.png)

## 3.3 多模态可解释性分析

作者使用**Grad-CAM**和**IG**来解释模型的预测依据。

对**Glioma**的分析：
*   **局部解释 (A)**：
    *   在**IDHwt ATC**中，模型关注**histology**图像中的**microvascular proliferation**（微血管增生）和**cell graph**中的微血管之间的胶质细胞。
    *   在**IDHmut ATC**中，两种模态都关注肿瘤细胞密度，但**IDH**突变对生存的贡献方向（正向）被正确识别。
    *   在**ODG**中，两种模态都定位到了具有“**fried egg cells**”（煎蛋样细胞）特征的区域。
*   **全局解释 (B)**：模型识别出了许多已知的**glioma**致癌基因（如**IDH, PTEN, MYC, CDKN2A**）作为重要的风险预测因子。有趣的是，在多模态条件下，一些基因的重要性发生了变化，例如**EGFR**在**IDHwt ATC**中的重要性下降，这与**EGFR**在胶质母细胞瘤治疗中并非强效靶点的证据相符。

![](https://pic1.imgdb.cn/item/69d5c99dcfae5fcb267c2c1f.png)

对**CCRCC**的分析：
*   **局部解释 (A)**：
    *   对于生存期较长的患者，模型关注那些细胞核仁不明显的细胞。
    *   对于生存期较短的患者，模型关注那些具有明显核仁的大细胞以及特征性的“**chicken-wire**”血管模式。
*   **全局解释 (B)**：模型发现了**CYP3A7**低表达和**PITX2**, **DDX43**, **XIST**高表达与高风险相关，这些基因已被证实与多种癌症的 **predisposition** 和 **progression** 有关。

![](https://pic1.imgdb.cn/item/69d5c9d7cfae5fcb267c2c98.png)

这些可解释性分析不仅验证了模型学习到了有生物学意义的特征，还揭示了新的、潜在的预后生物标志物。