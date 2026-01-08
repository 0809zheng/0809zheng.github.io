---
layout: post
title: 'MatchCLOT: Single-Cell Modality Matching with Contrastive Learning and Optimal Transport'
date: 2025-08-06
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/692d531f11af9ce9c3e888f8.png'
tags: 论文阅读
---

> MatchCLOT: 通过对比学习和最优传输实现单细胞模态匹配.

- (NeurIPS 2022 Workshop LMRL version)：[MatchCLOT: Single-Cell Modality Matching with Contrastive Learning and Optimal Transport](https://openreview.net/forum?id=PfdWl0H0Zq)
- (Briefings in Bioinformatics version)：[Matching single cells across modalities with contrastive learning and optimal transport](https://academic.oup.com/bib/article/24/3/bbad130/7147026)

# 0. TL; DR

本文提出了 **MatchCLOT**，是一个融合了**对比学习 (Contrastive Learning)**、**最优传输 (Optimal Transport, OT)** 和**直推式学习 (Transductive Learning)** 的单细胞多组学**模态匹配 (Modality Matching)**方法。

**MatchCLOT**使用对比学习来训练两个模态的编码器，目标是在共享的嵌入空间中，让来自同一个细胞的两种模态的表征尽可能相似，而来自不同细胞的表征则相互排斥。引入了**熵正则化的最优传输（Entropic Optimal Transport）**。它将匹配问题转化为一个寻找最优“运输方案”的数学问题，从而生成一个“软匹配”概率矩阵，不仅性能更优，而且计算速度更快、内存占用更低。

**MatchCLOT**在极具挑战性的**NeurIPS 2021**多模态单细胞数据整合竞赛的模态匹配任务上，取得了新的**SOTA**成绩，比之前的最佳方法（**scMoGNN**）的总匹配得分提升了**28.9%**，**Top-5**匹配准确率提升了**209%**。


# 1. 背景介绍

单细胞多组学技术正在以前所未有的分辨率揭示细胞的复杂性。然而，同时测量多种组学的技术（如**SHARE-seq**）通常成本高昂、通量有限。在实际研究中，我们更常遇到的情况是：对同一份生物样本（如肿瘤组织），分别用**scRNA-seq**和**scATAC-seq**等技术测了两批不同的细胞。

这就引出了模态匹配问题：我们能否通过计算，将被分开测量的**RNA**谱和**ATAC**谱重新“配对”，找到它们在原始样本中一一对应的关系？**NeurIPS 2021**为此专门举办了多模态单细胞数据整合竞赛，模态匹配是其中的核心赛道之一。竞赛中的**Team Novel**使用了**对比学习**和**最大权重二分图匹配**的方法，取得了不错的成绩。其思路是：
1.  用对比学习训练模型，得到一个$N \times N$的相似度矩阵。
2.  将这个矩阵看作一个二分图的邻接矩阵，然后寻找一个完美匹配，使得匹配边的总权重最大。

然而，这种方法存在两个问题：
*   硬匹配的局限性：二分图匹配给出的是一个非0即1的“硬匹配”结果，无法表达匹配的不确定性。
*   计算开销大：求解大规模图上的最大权重匹配问题，计算复杂度和内存开销都非常大。

受到单细胞领域中**最优传输（Optimal Transport, OT）**应用的启发，本文用**OT**来替代二分图匹配，实现一个更高效、更精准的“软匹配”方案。

# 2. MatchCLOT 模型

**MatchCLOT**的框架清晰地分为三个模块：预处理、训练和测试（匹配）。

![](https://pic1.imgdb.cn/item/692d54b311af9ce9c3e889e2.png)

### 2.1 模块一：预处理 (Preprocessing)

**MatchCLOT**采用了**直推式学习（Transductive Learning）**的设定：在训练开始前，就将训练集和（无标签的）测试集数据合并在一起进行预处理。具体流程如下：
1.  合并数据：将训练集和测试集按模态（如所有**RNA**数据、所有**ATAC**数据）分别合并。
2.  降维：对每个模态的合并后数据，采用**LSI（Latent Semantic Indexing）**进行标准化和降维。**LSI**是处理**scATAC-seq**数据的常用方法，它包括**TF-IDF**变换、**L1**归一化、**log**变换，最后通过截断**SVD**实现降维。
3.  批次校正：使用成熟的批次校正工具**Harmony**，对**LSI**降维后的数据进行处理，以消除不同实验批次带来的技术差异。

在竞赛数据集中，训练集和测试集来自不同的批次。如果只在训练集上训练，模型将难以泛化到分布有显著差异的测试集上。通过直推式学习，模型在训练前就“看到”了测试集的数据分布，**Harmony**也能够将训练集和测试集的批次效应拉到同一个基准上，从而极大地缓解了**分布偏移（distribution shift）**问题。此过程仅使用了测试集的特征数据，没有使用任何标签信息。

### 2.2 模块二：训练 (Training)

**MatchCLOT**的训练主干借鉴了**CLIP**模型，采用对比学习来对齐两个模态。对于每种模态都使用一个简单的多层感知机**（MLP）**作为编码器。编码器的结构（层数、维度）和训练超参数都通过贝叶斯优化（Wandb）进行了精细的调整。

将一个**batch**的配对数据送入模型，得到两组嵌入向量（**RNA**嵌入和**ATAC**嵌入）。计算两种嵌入两两之间的余弦相似度，得到一个$N \times N$的相似度矩阵。训练的目标（基于**InfoNCE Loss**）是最大化对角线上的相似度（即正确配对的样本），同时最小化非对角线上的相似度（即错误配对的样本）。

通过这个过程，模型学会了如何为来自同一个细胞的不同模态数据生成相似的嵌入向量，为后续的匹配奠定了基础。

### 2.3 模块三：测试与匹配

在测试阶段得到了一个$N \times N$的测试集余弦相似度矩阵。作者不把它当作图来求解匹配，而是将其视为一个最优传输问题。

最大权重二分图完美匹配问题，可以被写成一个整数线性规划（**ILP**）：
    
$$
\begin{aligned}
\max_x& \sum_{(a,b) \in A \times B} w(a,b)x(a,b) \\
\text{s.t.}& \sum_b x(a,b)=1, \sum_a x(a,b)=1, x(a,b) \in \{0, 1\}
\end{aligned}
$$

其中 $w(a,b)$ 是相似度。如果放宽 $x(a,b)$ 必须是整数的约束，它就变成了一个线性规划（**LP**）问题。这个**LP**问题可以被完美地等价为一个最优传输（**OT**）问题：
- 代价矩阵 **(Cost Matrix) $C$:** 将相似度矩阵取负，即 $c(a,b) = -w(a,b)$。目标是最小化“运输”总成本。
- 运输方案 **(Transport Plan) $Γ$:** 这就是要求的解，一个$N \times N$的矩阵，$\Gamma(a,b)$ 表示从源a“运输”多少“货物”到目的地b。
- 边缘分布 **(Marginal Distributions):** 假设源和目的地都是均匀分布，即每个点不多不少，刚好有 $1/N$ 的“货物”。

直接求解**OT**问题仍然很慢。作者引入了熵正则化，它在原目标函数上增加了一个熵项：
    
$$
\min_{\Gamma} \underbrace{\sum_{(a,b)} c(a,b)\Gamma(a,b)}_{\text{运输成本}} + \varepsilon \underbrace{\sum_{(a,b)} \Gamma(a,b) \log \Gamma(a,b)}_{\text{熵正则项}}
$$

其中 $\varepsilon$ 是正则化强度。这个正则项使得问题可以通过非常高效的**Sinkhorn**算法来迭代求解。熵正则化会使得最优的运输方案$\Gamma$不再是稀疏的（像硬匹配那样），而是一个“模糊”的概率矩阵。$\Gamma(a,b)$ 的值可以被解释为细胞a和细胞b配对的概率。这比硬匹配提供了更丰富的信息。

最后，作者利用已知的批次标签进一步优化匹配过程：不直接在整个$N \times N$的测试集上求解**OT**，而是将细胞按批次分开，在每个批次内部独立地计算相似度矩阵并求解**OT**。最后再将各个批次的匹配矩阵拼接起来。这极大地减小了每次**OT**求解的规模，进一步提升了效率和准确性。


# 3. 实验分析

作者在**NeurIPS 2021**竞赛数据集上对**MatchCLOT**进行了详尽的评估和分析。

作者比较了**MatchCLOT**与竞赛中的**Top 2**队伍（**CLUE, Novel**）以及一个后来的挑战者（**scMoGNN**）的性能。在所有五个子任务（**ATAC-GEX，GEX-ATAC，ADT-GEX，GEX-ADT**及**Overall**）上，**MatchCLOT**的匹配概率得分均取得了第一名。

为了搞清楚**MatchCLOT**的各个模块分别贡献了多少性能，作者进行了一系列增量式的消融实验：从基线方法（**Team Novel**的方法）开始，逐步加入**MatchCLOT**的各个创新点。
*   超参数优化：仅通过精细的超参数调优，性能就比基线提升了18.1%。
*   批次标签：引入批次标签来缩小搜索空间，带来了最大的性能飞跃，使得分提升了**76.9%**。这凸显了在多批次数据整合中，合理利用批次信息的重要性。
*   熵正则化**OT**和直推式学习分别带来了8.6%和2.8%的性能增益，进一步巩固了领先优势。
*   尽管OT对匹配得分的直接贡献不大，但它在计算效率上带来了巨大收益。实验显示，在内存受限的情况下，OT可以在处理100%数据的情况下，比需要丢弃95%数据的二分图匹配方法更快。

![](https://pic1.imgdb.cn/item/692d5758c2ca2fe15cef19bd.png)