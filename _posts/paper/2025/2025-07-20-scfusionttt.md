---
layout: post
title: 'scFusionTTT: Single-cell transcriptomics and proteomics fusion with Test-Time Training layers'
date: 2025-07-20
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6997cd1ed2628f800ee111f4.png'
tags: 论文阅读
---

> scFusionTTT：通过测试时训练层进行单细胞转录组与蛋白质组融合.

- paper：[scFusionTTT: Single-cell transcriptomics and proteomics fusion with Test-Time Training layers](https://arxiv.org/abs/2410.13257)

# 0. TL; DR

**scFusionTTT** 是一个用于单细胞多组学融合的方法，将**TTT**（测试时训练）层应用于单细胞多组学分析的模型。模型的核心是一个基于**TTT**的掩码自编码器**（masked autoencoder）**，采用了一个精心设计的三阶段训练策略：1) 多组学预训练；2) 多组学微调；3) 将从多组学数据中学到的知识迁移到单组学（仅**scRNA-seq**）数据的分析中，以增强其性能。

在四个多组学（**CITE-seq**）数据集和四个单组学（**scRNA-seq**）数据集上的实验表明，**scFusionTTT**在大多数评估指标上都取得了最佳性能，证明了其优越性。

# 1. 背景介绍

单细胞多组学技术，特别是**CITE-seq**，让我们能够同时检测同一个细胞的基因表达（**transcriptomics**）和表面蛋白（**proteomics**）。整合这两种信息，对于精准地识别细胞身份、理解疾病机制至关重要。

近年来，基于深度学习的方法，特别是利用注意力机制**（attention mechanism）**的方法，在多组学整合中展现了巨大的潜力。然而，当我们将这些方法直接应用于基因组数据时，会遇到困难。
1.  计算复杂度：一个细胞的转录组通常包含数万个基因。对于自注意力**（self-attention）**机制来说，其计算复杂度是输入序列长度的二次方 ($O(N^2)$)。这意味着，当基因数量从一千增加到一万时，计算量会增加一百倍。这种指数级的增长使得传统的注意力模型在处理全基因组尺度的序列数据时，变得不切实际。
2.  序列信息：基因在染色体上的排列并非随机，其线性顺序**（sequential order）**蕴含着重要的调控信息（例如邻近基因的共表达）。大多数现有的整合方法，如**VAE**或简单的**MLP**，都将基因视为一个无序的特征集合，完全忽略了这种宝贵的空间序列关系。

作者将目光投向了一种全新的序列建模范式：**测试时训练（Test-Time Training, TTT）**层。**TTT**层是一种新颖的**RNN**变体，它通过在推理时**（test-time）**利用自监督学习来动态更新其隐藏状态**（hidden state）**，从而实现了线性计算复杂度 ($O(N)$)，并且在长上下文建模任务中表现出色。

作者认为，**TTT**层的这些特性，使其成为处理和建模具有内在序列关系的高维基因组数据的理想工具。基于这一洞见，作者开发了**scFusionTTT**，旨在利用**TTT**层的优势，来革新单细胞多组学数据的融合方式。

# 2. scFusionTTT 方法

**scFusionTTT**的核心是一个基于**TTT**层的掩码自编码器**（masked autoencoder）**，它通过一个精巧的三阶段训练策略，实现了从多组学融合到单组学增强的完整流程。

![](https://pic1.imgdb.cn/item/6997cfb3d2628f800ee11206.png)

## 2.1 TTT层与TTT块

**TTT**层的核心思想是其隐藏状态 $W_t$ 不再是像传统**RNN**那样通过固定的门控机制更新，而是在处理每个输入**token**（如一个基因）$x_t$ 时，通过一个自监督学习任务，进行梯度下降式的更新。

$$
W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)
$$

这里的自监督损失 $\ell$ 是一个重构损失，它鼓励模型用当前的隐藏状态，去预测输入**token**的某个变换。

$$
\ell(W; x_t) = \| f(\theta_K x_t; W) - \theta_V x_t \|^2
$$

通过这种方式，**TTT**层能够将序列信息动态地压缩并存储到其隐藏状态（即网络权重）中。

作者将**TTT**层与**MLP**和**RMSNorm**层组合，构建了一个类似于**Transformer Block**的**TTT块**。这是构成**scFusionTTT**所有编码器和解码器的基本单元。

$$
X'_l = \text{TTTLayer}(\psi(X_{l-1})) + X_{l-1}\\
X_l = \text{MLP}(\psi(X'_l)) + X'_l
$$

![](https://pic1.imgdb.cn/item/6997d105d2628f800ee1121f.png)

## 2.2 模型架构与三阶段训练

### 第一阶段：多组学预训练 (Multi-omics Pre-training)

在一个无监督的自监督任务中，让模型学习**RNA**和**ADT**（蛋白质）数据的基本表达模式和序列关系。

输入包含基因和蛋白质的顺序信息。作者首先根据基因在人类基因组中的物理位置进行排序，蛋白质则根据其编码基因的顺序排序。

对输入的**RNA**和**ADT**序列进行随机掩码**（random masking）**。使用两个独立的、基于**TTT块**的编码器，分别对**RNA**和**ADT**的未掩码部分进行编码，得到潜在表示 $E_{RNA}^{out}$ 和 $E_{ADT}^{out}$。

通过两个融合**TTT**模块**（FusionTTT modules）**进行信息交换。例如，为了更新**RNA**的表示，作者将**ADT**的表示与**RNA**的表示拼接后，输入到一个**FusionTTT**模块中，并将其输出与原始**RNA**表示进行残差连接。

$$
FT_{RNA}^{in} = E_{RNA}^{out} + \lambda \cdot \text{FusionTTT}(E_{ADT}^{out}, E_{RNA}^{out})
$$

最后，使用两个独立的解码器，从融合后的表示中重构出被掩码的原始**RNA**和**ADT**数据。最小化**RNA**和**ADT**的重构均方误差。

$$
L_1 = \alpha L_{MSE}^{RNA} + \beta L_{MSE}^{ADT}
$$

### 第二阶段：多组学微调 (Multi-omics Fine-tuning)

在预训练好的模型基础上，利用少量的带标签**CITE-seq**数据，进行有监督的微调，以学习区分不同细胞类型。

移除解码器，将**FusionTTT**模块的输出作为最终的细胞嵌入，并添加一个分类头。最小化交叉熵损失**（cross-entropy loss）**。

$$
L_2 = \text{CrossEntropyLoss}(y, \hat{y})
$$

### 第三阶段：知识迁移与单组学预测 (Unimodal Omics Predicting)

将从多组学数据中学到的知识，迁移到只需要单组学（如**scRNA-seq**）输入的任务中，以增强其性能。

固定预训练好的**RNA**编码器和**FusionTTT**模块的参数。当输入一个**scRNA-seq**样本时，模型会通过一个投影层**（projection layer）**来模拟出其缺失的**ADT**模态的潜在表示。然后，将真实的**RNA**表示和模拟的**ADT**表示一起送入**FusionTTT**模块，进行与第二阶段相同的分类预测。损失函数同样是交叉熵损失。


# 3. 实验分析

作者在四个公开的**CITE-seq**数据集和四个公开的**scRNA-seq**数据集上，对**scFusionTTT**的性能进行了全面的评估。

## 3.1 实验一：在CITE-seq多组学数据集上的聚类性能

作者将**scFusionTTT**与**CiteFuse**, **TotalVI**, **SCOIT**等六种SOTA的多组学整合方法进行了比较，使用了多达10种不同的聚类评估指标。

在所有四个**CITE-seq**数据集（**SPL111**, **PBMC5K**, **PBMC10K**, **MALT10K**）上，**scFusionTTT**在绝大多数评估指标上都取得了最佳性能。例如，在**PBMC10K**数据集上，**scFusionTTT**的**ARI**达到了**0.90**，而次优的**CiteFuse**仅为0.79。在**MALT10K**上，**DBI**（越低越好）达到了**0.38**，远低于其他方法。

![](https://pic1.imgdb.cn/item/6997d17cd2628f800ee11228.png)

**UMAP**可视化结果也直观地显示，**scFusionTTT**学习到的细胞嵌入，其簇结构最清晰，与真实标签的吻合度最高。

![](https://pic1.imgdb.cn/item/6997d1a7d2628f800ee1122b.png)

这些结果强有力地证明了**scFusionTTT**在整合**CITE-seq**数据方面的卓越性能。


## 3.2 实验二：在scRNA-seq单组学数据集上的聚类性能

作者进一步评估了**scFusionTTT**通过知识迁移来增强单组学分析的能力，并与**Scanpy**, **Seurat**, **scVI**等四种经典的**scRNA-seq**分析工具进行了比较。

在所有四个**scRNA-seq**数据集（**IFNB**, **PBMC3K**, **CBMC**, **BMCITE**）上，经过知识迁移的**scFusionTTT**，其聚类性能同样在绝大多数指标上都取得了最佳性能。例如，在**CBMC**数据集上，其**ARI**达到了**0.76**，而表现最好的基线方法**scVI**仅为0.53。

![](https://pic1.imgdb.cn/item/6997d1dcd2628f800ee1122c.png)

这证明了作者提出的三阶段训练策略是成功的：从多组学数据中学到的知识，确实可以被有效迁移，并显著提升单组学分析的准确性。

## 3.3 实验三：消融研究与参数分析

作者比较了**TTT**融合模块、注意力融合模块和简单的逐元素相加三种融合方式。结果清晰地显示，基于**TTT**的融合模块性能最佳，显著优于其他两种方式。这直接证明了**TTT**层在信息融合中的核心贡献。

作者测试了三种不同的基因/蛋白质输入顺序：正常顺序、反向顺序和随机打乱顺序。结果显示，随机打乱顺序后，模型的性能出现了显著的下降。这无可辩驳地证明了，基因和蛋白质的线性顺序信息对于模型学习是至关重要的，而**scFusionTTT**的**TTT**层架构能够有效地利用这种序列信息。

![](https://pic1.imgdb.cn/item/6997d22bd2628f800ee11231.png)

这些实验全面地证明，**scFusionTTT**通过其创新的**TTT**层架构和精巧的三阶段训练策略，无论是在多组学整合还是单组学增强任务中，都展现了SOTA的性能，为处理高维序列化的组学数据提供了全新的范式。