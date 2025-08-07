---
layout: post
title: 'GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies'
date: 2025-04-23
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/689417a358cb8da5c80c5a28.png'
tags: 论文阅读
---

> GenomeOcean：基于大规模宏基因组组装训练的高效基因组基础模型.

- paper：[GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies](https://www.biorxiv.org/content/10.1101/2025.01.30.635558)


# 0. TL; DR

本文介绍了一种名为 **GenomeOcean** 的高效基因组基础模型，它在大规模宏基因组数据集上进行了训练，包含超过 **600 Gbp** 的高质量序列片段，涵盖了来自地球不同生态系统的微生物多样性。**GenomeOcean** 采用字节对编码（**BPE**）分词策略，结合架构优化，实现了高达 150 倍的序列生成速度提升，同时保持高生物保真度。该模型在表示微生物物种和生成受进化原则约束的蛋白质编码基因方面表现出色，能够在没有明确提示的情况下生成新的生物合成基因簇（**BGCs**），并在多个基因组功能模块中展现出强大的建模能力，为宏基因组研究、天然产物发现和合成生物学等领域树立了新的标杆。

# 1. 背景介绍

基因组基础模型在精准医学、药物发现和复杂生物系统理解方面具有变革潜力。然而，现有模型通常效率低下，受限于次优的分词策略和架构设计，并且对参考基因组存在偏差，限制了其对稀有生物圈中低丰度、未培养微生物的表示能力。

现有的基因组基础模型在分词策略上存在选择，如基因密码子、固定大小的 **k-mer** 或单核苷酸等，各有优缺点。在训练集选择上，预测基因关注编码区域，参考基因组提供高质量数据但存在偏差，宏基因组测序提供全面的基因组多样性表示，但面临装配质量、潜在噪声和计算资源需求等挑战。

为解决现有模型的局限性，作者提出了 **GenomeOcean**，一个基于 **BPE** 的生成式基因组基础模型。它直接在宏基因组装配上进行训练，通过大规模共装配宏基因组样本增强对稀有微生物物种的表示，并提高泛化能力。

# 2. GenomeOcean 模型

**GenomeOcean** 的训练数据集由六个宏基因组数据集组成，覆盖水生、陆生和宿主相关群落，总共有约 **645 Gbp** 的高质量序列片段，来自超过 **219 TB** 的原始宏基因组数据。

通过比较宏基因组装配数据集的四核苷酸频率（**TNF**）分布的香农熵，发现 **GenomeOcean** 的数据集在物种多样性方面与 **Genome Taxonomy Database（GTDB）**相当或更高。

![](https://pic1.imgdb.cn/item/68946c6d58cb8da5c80e8175.png)
![](https://pic1.imgdb.cn/item/68946c8358cb8da5c80e83f9.png)

**GenomeOcean** 采用基于 **SentencePiece** 的 **BPE** 分词器，将输入的 **DNA** 序列转换为一组非重叠的 **token**。词汇表通过迭代选择预训练语料库中频繁出现的序列 **k-mer**（k 从 1 到 12）来确定，直到达到所需的大小（4096）。

**GenomeOcean** 使用 **Transformer** 解码器架构，并采用 **BPE** 分词器高效表示 **DNA** 序列。模型在训练时采用下一个 **token** 预测任务，优化了 **Transformer** 解码器架构，包括 **FlashAttention-2、Group Query Attention（GQA）、RMS** 层归一化、**SiLU** 激活函数和旋转向量位置嵌入（**RoPE**）等技术。

**GenomeOcean**的下游应用包括：
- 序列嵌入功能：将 **DNA** 序列转换为高维特征空间中的表示，用于微生物物种分类等任务。
- 生成新的 **DNA** 序列：通过给定提示序列生成后续序列，在生成蛋白质编码基因和 BGCs 等方面表现出色。

![](https://pic1.imgdb.cn/item/68946e5158cb8da5c80e93fd.png)

实验了不同参数规模的 **GenomeOcean** 模型（**100M、500M** 和 **4B**），发现随着参数数量的增加，训练损失持续下降，**4B** 模型表现出更强的基因组复杂性捕捉能力。

# 3. 实验分析

**GenomeOcean** 在内存使用和推理速度方面均优于其他模型。在序列嵌入任务中，**GenomeOcean** 在处理 32 kb 序列时所需的内存仅为 **Evo** 的六分之一，且处理速度更快。在序列生成任务中，**GenomeOcean** 的生成速度比 **Evo** 快 150 倍，比 **GenSLMs** 快 87 倍。

![](https://pic1.imgdb.cn/item/68946ebc58cb8da5c80e97e5.png)

在合成宏基因组数据集上，**GenomeOcean** 的嵌入表示在物种分类任务中表现出色，其 **UMAP** 投影形成的簇更忠实地反映了底层物种身份，调整兰德指数（**ARI**）达到 0.92，优于 **TNF**（0.81）、**GenSLM**（0.064）和 **Evo**（0.521）。

![](https://pic1.imgdb.cn/item/68946edf58cb8da5c80e992b.png)

**GenomeOcean** 生成的合成数据集涵盖了多种蛋白质功能类别，能够根据部分提示序列生成完整的蛋白质序列，并且生成的蛋白质序列在三维结构上与原始序列具有相似的折叠结构，表明其学习了蛋白质编码的基本规则。

![](https://pic1.imgdb.cn/item/68946f1e58cb8da5c80e9b80.png)

在区分自然和人工序列的任务中，**GenomeOcean** 的 **F1** 分数超过 99%，显示出其能够有效识别自身生成的人工序列，降低了人工序列污染公共数据库的风险。

![](https://pic1.imgdb.cn/item/68946fb158cb8da5c80ea1c6.png)

**GenomeOcean** 在建模 **BGCs** 等高阶基因组功能模块方面表现出色。微调后的 **bgcFM** 模型能够生成新的、长的 **BGC** 序列，并在扫描微生物基因组时识别出候选 **BGC** 区域，包括与已知 **BGCs** 重叠的区域和可能的新 **BGCs**。

![](https://pic1.imgdb.cn/item/68946f8558cb8da5c80e9ee8.png)