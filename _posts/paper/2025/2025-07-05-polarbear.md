---
layout: post
title: 'Multimodal Single-Cell Translation and Alignment with Semi-Supervised Learning'
date: 2025-07-05
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/69984fb6bb93d236450b63a6.png'
tags: 论文阅读
---

> 通过半监督学习进行多模态单细胞翻译与对齐.

- paper：[Multimodal Single-Cell Translation and Alignment with Semi-Supervised Learning](https://www.liebertpub.com/doi/full/10.1089/cmb.2022.0264)

# 0. TL; DR

**Polarbear**是一个半监督机器学习框架，能够预测缺失的模态信息，并实现单细胞的跨模态对齐。该框架分为两个阶段：
1.  预训练阶段：利用**co-assay**和**single-assay**数据，为每种模态分别训练一个**beta-**变分自编码器（**beta-VAE**），以无监督的方式学习单个细胞的稳健低维表示。
2.  翻译阶段：将一个模态的编码器与另一个模态的解码器连接起来并冻结参数，中间插入一个翻译层，仅使用成对的**co-assay**数据以有监督的方式训练翻译层。

实验证明，与仅使用**co-assay**数据的全监督方法相比，这种半监督框架能够更准确地预测缺失的模态谱，并更精确地匹配跨模态的单个细胞，从而有力地促进了多模态数据的整合。

# 1. 背景介绍

单细胞组学，包括表观基因组学、转录组学、蛋白质组学等，对于研究细胞间的变异非常有价值，因为每种检测都提供了关于细胞调控的独特视角。然而，在大多数情况下，每种类型的测量都是在不同的细胞集上进行的，任何单个细胞只捕获一种类型的活动。

近年来出现的单细胞共检测（**co-assay**）技术，能够在同一个细胞内进行多种类型的测量，使我们能够直接测量每个细胞内的多种分子活动。然而，**co-assay**测量通常比标准的单细胞测量通量更低，并且通常比单检测数据更难生成。

多种机器学习方法已被提出，利用**co-assay**数据在不同单细胞组学测量之间进行翻译。在大多数这些研究中，模型是基于**co-assay**数据提供的跨模态细胞匹配关系以全监督方式学习的。因此，这些模型的性能必然受到**co-assay**数据的稀疏性和有限数量的限制。

与此同时，公共数据库中存有数量级更多的单细胞数据。作者假设，与仅使用**co-assay**数据相比，同时使用**co-assay**数据（有标签数据）和来自独立研究的**single-assay**数据（无标签数据）来训练翻译模型，将会提高跨模态翻译的性能。

因此，作者提出了一个名为**Polarbear**的半监督框架，它通过同时使用**co-assay**和**single-assay**数据来学习在单细胞测量之间进行翻译。本文重点关注**scRNA-seq**和**scATAC-seq**之间的翻译。

# 2. Polarbear框架

**Polarbear**框架通过两个阶段实现半监督学习，以在**scRNA-seq**和**scATAC-seq**数据域之间进行翻译。

![](https://pic1.imgdb.cn/item/6998508abb93d236450b63af.png)

## 2.1 第一阶段：各模态的beta-VAE预训练

在第一阶段，**Polarbear**为**scRNA-seq**和**scATAC-seq**分别构建独立的**beta-VAE**。与传统的**VAE**相比，该模型借鉴了**scVI**和**peakVI**的思想，校正了细胞间的批次效应和测序深度偏差。

### scRNA-seq VAE
  
输入基因表达的原始计数（$x$）。模型将批次因子（$b$）进行**one-hot**编码并与输入和嵌入层拼接。每个细胞的测序深度因子（$d_x$）则用于后续的重建损失计算。

模型假设基因表达计数服从零膨胀负二项分布（**ZINB**）。解码器的输出是该分布的均值、方差和零膨胀（**dropout**）概率。经过归一化的期望频率作为预测的**scRNA-seq**表达谱。

### scATAC-seq VAE

输入二值化的染色质开放峰计数（$y$）。为了节省内存以处理大量的**peaks**，模型在编码器的前两层和解码器的后两层中只允许染色体内部的连接（**within-chromosome connections**）。

模型假设每个**peak**的开放状态服从伯努利分布（**Bernoulli distribution**）。解码器学习一个经过测序深度归一化的概率作为输出。该概率与测序深度因子（$d_y$）结合，用于最大化二元交叉熵。

## 2.2 第二阶段：翻译层的监督学习

当两个模态的自编码器都优化完毕后，作者引入一个线性的翻译层，在**scRNA-seq**和**scATAC-seq**的嵌入层之间建立连接。这一步仅使用成对的**co-assay**数据进行监督学习，并最小化翻译损失。

## 2.3 模型的四步优化过程

**Polarbear**模型通过四个独立的步骤进行优化：

![](https://pic1.imgdb.cn/item/69985197bb93d236450b63bc.png)

### 2.3.1 优化scATAC-seq VAE

优化**scATAC-seq**的编码器$q_y(z_y\|y, b_y, d_y)$和解码器$p_y(y\|z_y, b_y, d_y)$。损失函数是重建损失和**KL**散度项的加权和：

$$
loss_{ATACreconst} = - E_{z_y \sim q_y(z_y|y, b_y, d_y)} \log p_y(y|z_y, b_y, d_y) + \beta D_{KL}(q_y(z_y|y, b_y, d_y) \| p_y(z_y))
$$

### 2.3.2 优化scRNA-seq VAE

类似地，通过最小化$loss_{RNAreconst}$来优化**scRNA-seq**的编码器$q_x(z_x\|x, b_x, d_x)$和解码器$p_x(x\|z_x, b_x, d_x)$：

$$
loss_{RNAreconst} = - E_{z_x \sim q_x(z_x|x, b_x, d_x)} \log p_x(x|z_x, b_x, d_x) + \beta D_{KL}(q_x(z_x|x, b_x, d_x) \| p_x(z_x))
$$

### 2.3.3 优化scATAC-seq到scRNA-seq的翻译

冻结两个**VAE**的参数，仅优化从**scATAC-seq**到**scRNA-seq**的线性翻译层$t_{yx}$。此过程最小化从输入的**scATAC-seq**谱（$y$）估计出的**scRNA-seq**谱（$x$）的负对数似然：

$$
loss_{RNAtranslat} = - E_{z_y \sim q_y(z_y|y, b_y, d_y)} \log p_x(x|t_{yx}(z_y), b_x, d_x)
$$

### 2.3.4 优化scRNA-seq到scATAC-seq的翻译

同理，通过最小化$loss_{ATACtranslat}$来优化从**scRNA-seq**到**scATAC-seq**的翻译层$t_{xy}$：

$$
loss_{ATACtranslat} = - E_{z_x \sim q_x(z_x|x, b_x, d_x)} \log p_y(y|t_{xy}(z_x), b_y, d_y)
$$

通过这种分步优化的方式，**Polarbear**避免了多任务联合优化时超参数选择的困难，使得模型不易偏向于某个特定任务的优化。

# 3. 实验分析

作者在一系列实验中验证了**Polarbear**的性能，并与当时的**state-of-the-art**翻译模型**BABEL**以及一个仅使用**co-assay**数据训练的**Polarbear**变体（称为**Polarbear co-assay**）进行了比较。

## 3.1 Polarbear在随机测试集上的翻译准确性

在**SNARE-seq**数据集上，随机划分80%的细胞用于训练，20%用于测试。
-   **图A, B (scATAC-seq -> scRNA-seq翻译)**：在预测差异表达基因时，**Polarbear**的基因级相关性显著优于**BABEL**。采用半监督学习的**Polarbear**显著优于仅使用**co-assay**数据的**Polarbear co-assay**版本。这直接证明了引入大量**single-assay**数据能够有效提升翻译性能。
-   **图C, D, E, F (scRNA-seq -> scATAC-seq翻译)**：在预测差异开放**peak**时，**Polarbear**的**peak**级**AUROC**和**AUPRnorm**均优于**BABEL**。**Polarbear**再次显著优于**Polarbear co-assay**。

![](https://pic1.imgdb.cn/item/699851dfbb93d236450b63bf.png)

这些结果表明，**Polarbear**的半监督框架能够准确地在两种模态间进行翻译，并且其性能的提升很大程度上归功于对海量**single-assay**数据的利用。

## 3.2 Polarbear能够重现并预测细胞类型特异性标志

利用翻译生成的谱图来预测细胞类型的特异性标志物（**marker**）。
-   **图A**: 在预测已知的细胞类型**marker**基因时，**Polarbear**的预测准确性（中位**AUROC**=0.933）显著高于**BABEL**（中位**AUROC**=0.869）。
-   **图B**: **Polarbear**同样优于**Polarbear co-assay**（中位**AUROC**=0.914），尤其是在识别稀有细胞类型时，其优势更为明显。这表明，**single-assay**数据对于提升稀有细胞类型的表征学习至关重要。

**Polarbear**根据**scATAC-seq**谱预测出**Sall1**基因在小胶质细胞（**microglia**）中特异性高表达（**AUROC**=0.800），而实验观测的**scRNA-seq**数据由于稀疏性未能显示这一特征（**AUROC**=0.498）。**Sall1**已被证实是维持小胶质细胞身份的关键转录因子。这表明**Polarbear**的预测甚至可以揭示被实验噪声掩盖的生物学信号。

![](https://pic1.imgdb.cn/item/699851fbbb93d236450b63c2.png)

## 3.3 Polarbear能够预测新细胞类型中的内外部变异

为了模拟真实场景中可能遇到的新细胞类型，作者将**SNARE-seq**数据集中一个完整的细胞簇作为测试集（**unseen cell type**），其余细胞用于训练。
-   **图A-F (细胞内变异的捕捉)**：在预测 **unseen** 细胞类型内部的基因表达时，**Polarbear**的基因级相关性优于**BABEL**和**Polarbear co-assay**。在预测**peak**开放性时，**Polarbear**的**peak**级**AUROC**和**AUPRnorm**也表现出同样的优势。这表明，即使对于训练中未见过的细胞类型，**Polarbear**依然能够捕捉到有意义的细胞间细微差异。
-   **图G (细胞间变异的捕捉)**：作者基于翻译后的基因表达谱，对**unseen**细胞类型进行差异表达分析。**Precision-Recall**曲线显示，**Polarbear**能够最准确地识别出该细胞类型真正的差异表达基因，显著优于其他方法。

![](https://pic1.imgdb.cn/item/69985226bb93d236450b63c7.png)

这些结果表明**Polarbear**具有强大的泛化能力，能够准确预测新细胞类型的内部（**intra-cell type**）和外部（**inter-cell type**）变异。

## 3.4 Polarbear能够实现跨模态的细胞匹配

在潜在空间中，基于欧氏距离以贪心的方式匹配来自不同模态的细胞。使用**FOSCTTM**（**Fraction of Samples Closer Than The True Match**，越低越好）作为评估指标。
-   **图A, B (随机测试集)**：在随机测试集上，无论是用**scRNA-seq**查询**scATAC-seq**（图6A）还是反之（图6B），**Polarbear**的**FOSCTTM**分数都最低，表明其匹配性能优于**BABEL**和**Polarbear co-assay**。
-   **图C, D (unseen细胞类型)**：在**unseen**细胞类型中，**Polarbear**同样表现出最佳的匹配性能。这再次证实了**single-assay**数据对于学习更具对齐能力的细胞表示至关重要。

![](https://pic1.imgdb.cn/item/6998523fbb93d236450b63cb.png)

## 3.5 特征预过滤对细胞匹配性能的影响

比较使用全部特征的**Polarbear-full**与经过特征筛选的两个变体：**Polarbear-exp**（保留高表达特征）和**Polarbear-var**（保留高变异特征）在细胞匹配任务上的性能。
-   **图A, B (随机测试集)**：使用全特征的**Polarbear-full**表现最好。
-   **图C, D (unseen细胞类型)**：**Polarbear-full**与**Polarbear-exp**表现相似，均优于**Polarbear-var**。

![](https://pic1.imgdb.cn/item/6998525bbb93d236450b63d2.png)

该分析表明，为了达到最优的跨模态对齐性能，保留那些相对稀疏或变异性较低的特征是重要的。简单的特征过滤可能会损害模型的对齐能力。
