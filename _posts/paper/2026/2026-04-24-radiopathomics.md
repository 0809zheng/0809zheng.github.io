---
layout: post
title: 'RadioPathomics: Multimodal Learning in Non-Small Cell Lung Cancer for Adaptive Radiotherapy'
date: 2026-04-24
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/69d895cbe80dabf2eaccd492.png'
tags: 论文阅读
---

> RadioPathomics：非小细胞肺癌的自适应放疗的多模态学习.

- paper：[RadioPathomics: Multimodal Learning in Non-Small Cell Lung Cancer for Adaptive Radiotherapy](https://ieeexplore.ieee.org/document/10122541)

# 0. TL; DR

本研究开发了一种多模态晚期融合方法，结合了从**radiomics**, **pathomics**和**clinical data**（临床数据）中计算出的手工特征，以预测**non-small-cell lung cancer (NSCLC)**（非小细胞肺癌）患者的放射治疗结果。作者深入研究了八种不同的晚期融合规则和两种患者级聚合规则，以充分利用**CT**图像、全切片扫描和临床数据所提供的丰富信息。

在一个包含33名患者的内部队列上进行的实验显示：提出的基于融合的多模态范式，其**AUC**达到了**90.9%**，显著优于任何单一模态的方法，这表明数据整合可以推动精准医疗的发展。**late fusion**的表现优于另一种常用的多模态方法**early fusion**。在一个由不同模态和有限数据量构成的场景中（这在许多癌症研究领域很常见），结果表明，与**deep learning**框架相比，**hand-crafted features**仍然是提取相关信息的一种可行且有效的选择。

# 1. 背景介绍

肺癌是全球最常见的癌症类型之一，也是肿瘤死亡的主要原因。**NSCLC**占所有新发肺癌病例的约80-85%。当前的临床决策过程依赖于多种数据源，如放射学数据、数字病理学切片、基因组谱和临床数据，以改善肿瘤的检测、分类和预后。从这些多样化的模态中提取的互补性定量生物标志物，可以带来更准确的诊断和更有效的治疗方案。

近年来，**artificial intelligence (AI)** 社区在利用单一模态进行肿瘤检测、分类和预后预测方面付出了巨大努力。这催生了两个新兴的研究领域：
*   **Radiomics**：从放射学图像中提取定量特征，以预测临床结局或指导临床决策。
*   **Pathomics**：结合数字病理学、组学科学和**AI**，从高分辨率的组织活检切片**WSI**中提取嵌入信息，以获得定量生物标志物。

尽管**genomics**, **radiomics**, **pathomics**等单模态分析取得了显著进展，但将这些异构数据源融合到一个统一的机器学习框架中的研究仍然较少。特别是，尽管**radiomics**和**pathomics**各自的重要性已得到认可，但将它们结合起来的研究屈指可数。

因此，本研究提出了一个**multimodal late fusion**方案，旨在将**radiomics**, **pathomics**和**clinical data**结合起来，以预测**NSCLC**患者的放射治疗结果。作者选择**late fusion**是因为它特别适合处理那些在维度和采样率上差异显著、且相关性不强的异构数据流。

# 2. 方法介绍

本研究提出的融合框架包含四个主要模块：预处理、特征提取、患者聚合和晚期融合。

![](https://pic1.imgdb.cn/item/69d896e6e80dabf2eaccdf9d.png)

## 2.1 实验材料与数据
研究对象是一个包含33名局部晚期（III期）**NSCLC**患者的内部队列。所有患者均接受了同步放化疗，并根据肿瘤体积的变化被分为**adaptive**（自适应，即肿瘤体积缩小）和**not-adaptive**（非自适应）两组。

数据模态：
*   **Pathomics modality**：包含从**H&E**染色的肺癌组织活检切片中手动勾画出的1113个肿瘤区域（**crops**）。
*   **Radiomics modality**：包含从治疗前的**CT**扫描中手动勾画出的39个**Clinical Target Volume (CTV)**（临床靶区）。
*   **Semantic modality**：包含由两位经验丰富的放射肿瘤学家独立评估的临床数据，包括肿瘤分期评分（**T, N, stage**）、组织学评估、年龄和性别。

## 2.2 预处理 (Pre-processing)
*   **Pathomics**：为了增加样本量并统一尺寸，作者对原始的**crops**进行**patch**（图像块）提取，使用100x100的滑动窗口，步长为60。最终获得了53550个**patches**。
*   **Radiomics**：为了增加维度，作者将3D的**CTV**分解为其组成的2D切片，总共得到928个切片。
*   **Semantic**：对分类特征（如T, N, stage）进行序数编码，对无序分类特征（如性别）进行**one-hot**编码。

## 2.3 特征提取 (Features Extraction)
作者为**pathomics**和**radiomics**模态提取了**hand-crafted features**。
*   **Pathomics特征**：对每个**patch**的S通道（HSV颜色模型）计算其**Grey Level Co-Occurrence Matrix (GLCM)**（灰度共生矩阵）。从每个**GLCM**中提取6个**Haralick**描述子（对比度、相异性、同质性、能量、相关性、角二阶矩）。最终，每个**patch**由24个纹理描述子表示。
*   **Radiomics特征**：（统计特征）计算每个**CT**切片的一阶直方图的12个统计特征（如均值、标准差、偏度、峰度等）。（纹理特征）从**GLCM**中提取6个二阶统计特征。从**Local Binary Pattern (LBP)**（局部二值模式）的直方图中提取12个统计特征。总计，每个**CT**切片由116个特征表示。

## 2.4 患者聚合规则 (Patient Aggregation Rule)
由于一个患者可能对应多个**pathomics**的**patches**或**radiomics**的切片，为了进行患者级别的融合，需要一个聚合步骤。作者提出了两种聚合规则：
*   **A1 (Feature Mean)**：在分类前进行聚合。将属于同一患者的所有样本的特征向量取平均，得到一个患者级的特征向量$x_p$。
*   **A2 (Score Mean)**：在分类后进行聚合。首先为每个样本计算出其属于各个类别的软标签（概率），然后将属于同一患者的所有样本的软标签取平均，得到一个患者级的平均软标签$dp_{i,j}$。

## 2.5 晚期融合规则 (Late Fusion Rules)
在获得了每个模态对每个患者的预测后，需要一个融合规则来组合这些信息。作者探究了八种不同的晚期融合技术。这些技术都基于一个**decision profile (DP)** 矩阵，该矩阵组织了$L$个单模态分类器的输出。

基于独立类别支持度的规则:
- **Product rule (LF1)**: $\chi_j = \prod_{i=1}^{L} \mu_{i,j}$
- **Max rule (LF2)**: $\chi_j = \max_i \mu_{i,j}$
- **Min rule (LF3)**: $\chi_j = \min_i \mu_{i,j}$
- **Mean rule (LF4)**: $\chi_j = \frac{1}{L} \sum_{i=1}^{L} \mu_{i,j}$

基于决策模板(**DTs**)的规则:
- **Decision Templates (LF5)**：首先为每个类别计算一个**Decision Template (DT)**，即该类别所有训练样本**DP**矩阵的质心。然后，通过计算当前患者的**DP**与每个类别的**DT**之间的相似度来确定支持度。
- **Dempster-Shafer rule (LF6)**：同样基于**DTs**，但使用一种更复杂的公式来计算每个分类器输出与**DT**之间的接近度，并将其组合成最终的支持度。

其他范式的规则:
- **Majority voting rule (LF7)**：基于每个单模态分类器的硬标签（**crisp label**）进行投票。
- **Confidence rule (LF8)**：选择所有单模态预测中具有最高置信度（即最大软标签值）的那个预测作为最终结果。

# 3. 实验分析

## 3.1 实验设置

所有实验中，单模态分类器均使用**Random Forest**。采用**leave-one-patient-out (LOPO)** 交叉验证。使用**Area under the ROC curve (AUC)** 作为主要性能指标。

比较实验：
*   晚期融合：测试了上述所有16种融合规则组合（8种融合规则 x 2种聚合规则）在不同模态组合（**P+R+S**, **P+S**, **R+S**, **P+R**）下的性能。
*   早期融合：与两种早期融合规则（简单拼接和**Kronecker**积）进行比较。
*   手工特征 vs. 深度特征：将使用手工特征的结果与使用**deep features**（从预训练的**ResNet-18**和**GoogLeNet**中提取）的结果进行比较。

## 3.2 实验结果与讨论

### 1. 晚期融合结果分析

在单模态中，**Radiomics**表现最好（AUC=0.870），其次是**Pathomics**（AUC=0.814）。最佳性能由三模态组合**P+R+S**在**A1+LF6**规则下取得，**AUC**达到了**0.909**。

雷达图清晰地显示，多模态方法（特别是三模态组合**P+R+S**）的性能普遍优于单模态方法。**P+R+S**组合获得了最高的排名，并且其性能与所有单模态方法以及**P+S**组合相比，具有统计学上的显著差异。多模态整合显著提升了预测性能。在单模态中，**Radiomics**是信息量最大的模态。

![](https://pic1.imgdb.cn/item/69d897f4e80dabf2eacce1ea.png)

在分类前进行特征平均的**A1**规则，其性能普遍优于在分类后进行分数平均的**A2**规则。不同的融合规则在不同的模态组合下表现各异，没有一个规则在所有情况下都是最优的。**A1+LF4**（特征平均+均值规则）和**A1+LF7**（特征平均+多数投票）在所有模态组合中平均排名最高，表明它们具有较好的泛化能力。

![](https://pic1.imgdb.cn/item/69d8981fe80dabf2eacce23e.png)

### 2. 晚期融合 vs. 早期融合

该表展示了晚期融合与两种早期融合规则（简单拼接和**Kronecker**积）的胜-平-负次数。结果显示，晚期融合范式几乎总是优于早期融合范式，并且在大多数情况下，这种优势是统计显著的（灰色单元格）。这表明，当处理像本研究中这样性质和维度差异巨大且不相关的异构数据时，在数据层面进行融合（早期融合）是困难且效果不佳的。

![](https://pic1.imgdb.cn/item/69d8983ee80dabf2eacce285.png)

### 3. 手工特征 vs. 深度特征

该表比较了使用手工特征和深度特征的性能。单元格中的数字代表“手工特征的胜-平-负次数”。从对角线可以看出，对于相同的模态组合，使用手工特征的性能优于使用深度特征。例如，在**P+R+S**组合中，手工特征在16次比较中胜出10次。在这个数据量有限的特定场景下，手工特征的表现优于深度特征。作者认为，这是因为数据集的低维度限制了**deep neural networks**充分发挥其抽象、泛化和判别能力。

![](https://pic1.imgdb.cn/item/69d89852e80dabf2eacce2b8.png)