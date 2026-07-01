---
layout: post
title: 'MethylBERT enables read-level DNA methylation pattern identification and tumour deconvolution using a Transformer-based model'
date: 2026-02-07
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/698ae90887851c417e21fb37.png'
tags: 论文阅读
---

> MethylBERT通过基于Transformer的模型实现了read水平的DNA甲基化模式识别和肿瘤解卷积.

- paper：[MethylBERT enables read-level DNA methylation pattern identification and tumour deconvolution using a Transformer-based model](https://www.nature.com/articles/s41467-025-55920-z)

# 0. TL; DR

**MethylBERT**是一个用于读段级别的甲基化模式分类的基于 **Transformer** 的模型，能够根据单个测序读段（**sequence reads**）的甲基化模式和局部基因组序列，识别其是否来源于肿瘤，并以此估算大块样本（**bulk samples**）中的肿瘤细胞比例。

在评估中，**MethylBERT** 的性能优于现有的解卷积方法，并且在不同甲基化模式复杂度、读段长度和覆盖度的条件下都表现出高准确性。此外，作者还展示了它在细胞类型解卷积以及利用液体活检样本进行非侵入性早期癌症诊断方面的应用潜力。

# 1. 背景介绍

**DNA** 甲基化（**DNAm**）是一种重要的表观遗传修饰，在人类肿瘤中表现出显著的异质性。因此，**DNAm** 数据被广泛用于估算**肿瘤纯度 (tumour purity)**，这对于理解与临床结局、肿瘤诊断和表型特征相关的肿瘤表观遗传图谱至关重要。

**DNAm** 可以通过多种技术进行分析，包括全基因组亚硫酸氢盐测序 **(WGBS)**、简化代表性亚硫酸氢盐测序 **(RRBS)**、长读长测序（如 **Oxford Nanopore**）以及微阵列方法（如 **Infinium 450K/EPIC**）。高质量的测序数据能够产生覆盖广泛基因组区域且具有足够深度的读段，从而保留了稀有细胞群体的单分子信号。这在**循环肿瘤DNA (ctDNA)** 分析中尤为关键，因为它有助于癌症患者的非侵入性早期诊断、预后评估和治疗反应监测。

然而，大多数现有的纯度估算或细胞类型解卷积方法都是为基于芯片的 **DNAm** 数据设计的，这些方法通常处理的是平均甲基化水平（**beta-values**）的矩阵。之前的研究表明，现有的基于测序的解卷积方法性能并不优于基于芯片的方法，这意味着它们未能充分利用测序数据在准确推断方面的优势。

为了克服这些局限性，作者提出了 **MethylBERT**，这是一种基于双向编码器表示的**Transformer (BERT)** 的深度学习方法，专门用于读段级别的甲基化模式识别和肿瘤纯度估算。**MethylBERT** 使用一个修改过的 **BERT** 模型来编码读段级别的甲基化组，并将测序读段分为肿瘤或正常细胞类型。由此产生的后验概率被用于通过贝叶斯概率反演和最大似然估计来推导肿瘤纯度。**MethylBERT** 的应用代表了在测序数据分析中利用 **Transformer** 的一种新途径。

# 2. MethylBERT 方法

**MethylBERT** 的工作流程包含三个主要步骤：模型预训练、读段分类微调和肿瘤纯度估算。

![](https://pic1.imgdb.cn/item/698ae9e087851c417e21fcf1.png)

## 2.1 MethylBERT 模型架构

**MethylBERT** 的核心是 **BERT** 模型，它通过**多头自注意力 (multi-head attention)** 机制来捕捉输入序列中的复杂依赖关系。

注意力机制的计算公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中，$Q$ (查询), $K$ (键), $V$ (值) 是输入序列的三个不同表示，$d_k$ 是键向量的维度。通过缩放因子 $\frac{1}{\sqrt{d_k}}$ 可以防止点积结果过大。

多头注意力通过并行计算多个独立的注意力头，并将它们的结果拼接起来，从而使模型能够从不同的表示子空间中学习信息：

$$
\text{Multi-head attention}(Q, K, V) = \text{Concatenation}(A_1, ..., A_H)W^O
$$

其中，$A_i = \text{Attention}_i(QW_i^Q, KW_i^K, VW_i^V)$ 是第 $i$ 个注意力头。

为了适应 **DNA** 甲基化数据，作者对标准的 **BERT** 输入进行了修改，使其包含三种类型的嵌入信息：
1.  **DNA** 令牌嵌入 **(Token Embeddings)**：将 **DNA** 序列片段（**3-mers**）映射到嵌入向量。
2.  甲基化嵌入 **(Methylation Embeddings)**：编码每个 **CpG** 位点的甲基化状态（0: 未甲基化, 1: 甲基化, 2: 非 **CpG** 位点）。
3.  位置嵌入 **(Position Embeddings)**：提供序列中每个令牌的位置信息。

## 2.2 模型预训练 (Pre-training)

预训练是 **BERT** 的一个关键步骤，它使模型能够学习输入的通用上下文，而无需针对每个特定任务进行大量的架构工程。作者采用了**掩码语言模型 (Masked Language Model, MLM)** 进行预训练，具体步骤如下：
1.  将人类参考基因组（**hg19**）分割成长度为510 bp的片段，并将其转换为3-mer序列（**token**）。
2.  在预训练过程中，随机掩盖15%的3-mer令牌。其中，80%被替换为 `[MASK]` 标记，10%被替换为随机的其他令牌，10%保持不变。
3.  模型的目标是预测这些被掩盖的令牌。损失函数采用分类交叉熵损失：

$$
\mathcal{L}_{\text{pre-training}} = -\sum_{t=1}^{T}\sum_{l=1}^{69} m_t y_{tl} \log(\hat{y}_{tl})
$$

其中，$y_{tl}$ 是令牌 $t$ 的真实标签（**one-hot**编码），$\hat{y}_{tl}$ 是模型的预测概率，$m_t$ 指示令牌 $t$ 是否被掩盖。

## 2.3 读段分类微调 (Fine-tuning)

在预训练之后，模型被微调以执行读段级别的甲基化模式分类任务。输入除了包含 **DNA** 序列、甲基化模式和位置信息外，还额外加入了差异甲基化区域 **(DMR)** 的信息，以帮助模型学习区域特异性的肿瘤甲基化谱。

**MethylBERT** 的编码器部分将这些输入信息编码成一个高维向量。一个细胞类型分类器（**cell-type classifier**）接收该向量，并使用 **softmax** 函数计算出每个读段属于“肿瘤”或“正常”细胞的后验概率 $P(c_j\|r_i)$。

微调阶段的损失函数同样采用交叉熵损失：

$$
\mathcal{L}_{\text{fine-tuning}} = -\sum_{r=1}^{R}\sum_{c \in \{T,N\}} m_{r,c} \log\left(\frac{\exp(x_r^c)}{\sum_{c' \in \{T,N\}} \exp(x_r^{c'})}\right)
$$

其中，$x_r^c$ 是读段 $r$ 属于细胞类型 $c$ 的最终激活值。

## 2.4 肿瘤纯度/比例估算 (Tumour Purity/Fraction Estimation)

利用微调后模型计算出的后验概率，作者通过最大似然估计来估算肿瘤纯度 $\delta$。

首先，应用贝叶斯定理将后验概率 $P(\text{cell type}=T\|r_i)$ 转换为似然度 $P(r_i\|\text{cell type}=T)$：

$$
P(r_i|\text{cell type}=T) \propto P(\text{cell type}=T|r_i)P(\text{cell type}=T)^{-1}
$$

构建似然函数 $L(\delta)$：

$$
L(\delta) = \prod_{i=1}^{N} [\delta P(r_i|\text{cell type=Tumour}) + (1-\delta)P(r_i|\text{cell type=Normal})]
$$

通过网格搜索找到使似然函数最大化的 $\hat{\delta}$ 值：

$$
\hat{\delta} = \text{argmax}_{\delta} L(\delta)
$$

此外，作者还提出了一种**估算调整 (estimation adjustment)** 策略，通过最小化区域间肿瘤纯度分布的**偏度 (skewness)** 来校正估算结果，使其更加稳健。作为模型输出质量的衡量标准，**MethylBERT** 还提供了基于**Fisher信息** 的模型精度评估。

# 3. 实验分析

作者进行了一系列详尽的实验来评估 **MethylBERT** 在不同场景下的性能。

## 3.1 MethylBERT能够准确分类复杂的读段级别甲基化模式

作者通过模拟不同复杂度的甲基化模式、不同的读段长度和覆盖度，系统地评估了 **MethylBERT** 的读段分类能力。
- **图 A, B**: 对于150 bp和500 bp的模拟读段，**MethylBERT** 在所有 **methylation** 模式复杂度下，分类准确性均显著优于 **CancerDetector**、**DISMIR** 和一个基于 **HMM** 的基准方法。虽然所有方法的准确性都随着模式复杂度的增加而下降，但 **MethylBERT** 始终保持领先。
- **图 C**: 在一个极端场景下，即肿瘤和正常细胞的平均甲基化水平几乎相同，但 **CpG** 位点特异性模式不同，**MethylBERT** 依然能够有效地进行分类，而 **CancerDetector** 和 **HMM** 则完全失效。这表明 **MethylBERT** 不会被平均甲基化水平误导。
- **图 D, E**: 在不同读段覆盖度（**read coverage**）的测试中，**MethylBERT** 再次表现最佳，尤其是在低覆盖度（例如10x）的情况下，其准确率仍能保持在0.95以上。

![](https://pic1.imgdb.cn/item/698aeb2787851c417e21ffa1.png)

## 3.2 预训练使MethylBERT能够理解序列特征

作者深入探究了预训练过程对模型性能的影响。

- **图 A**: **UMAP** 可视化显示，经过预训练后，**BERT** 模型能够自发地学习到 **DNA 3-mer**之间的相互关系。例如，包含“CG”的令牌被聚为一类，并且模型还能识别出互补的核苷酸对（C-G, T-A）。
- **图 C, D, E**: 在**弥漫性大B细胞淋巴瘤 (DLBCL)** 的真实数据上，与未经预训练的模型相比，预训练后的 **MethylBERT** 表现出天壤之别。未经预训练的模型在微调过程中损失值先降后升，最终分类准确率仅在0.5左右徘徊。而预训练后的模型能够持续优化，最终实现高度准确的分类。微调过程中，预训练模型能够逐渐将肿瘤和正常读段的概率分布清晰地分离开，并且这种分类能力不依赖于简单的平均甲基化水平。
- **图 B**: 跨物种预训练适用性分析表明，使用小鼠（**mm10**）基因组预训练的模型在分析人类癌症数据时，其性能与使用人类（**hg19**）基因组预训练的模型几乎没有差异。这极大地扩展了 **MethylBERT** 的应用范围。

![](https://pic1.imgdb.cn/item/698aeb9287851c417e22009b.png)

## 3.3 MethylBERT准确估算体外混合样本的肿瘤纯度

作者使用 **DLBCL** 和正常B细胞的读段混合成的体外模拟大块样本（**pseudo-bulk samples**）来评估肿瘤纯度估算性能。

- **图 A**: 在不同肿瘤纯度下，**MethylBERT** 的估算误差（中位绝对误差）显著低于 **CancerDetector**、**DISMIR_dmr** 和经典的 **Houseman** 方法。**MethylBERT** 在低纯度和高纯度下均能保持高准确性。
- **图 B**: 作者提出的**估算调整**策略优于 **CancerDetector** 的“去除混杂因素”方法，能够更有效地校正估算结果。
- **图 C**: **Fisher信息** 作为模型精度的度量，与估算误差呈现出强烈的负相关。这意味着费雪信息值越高，估算结果越可靠，这为在没有金标准的情况下评估模型结果质量提供了重要指导。
- **图 D, E**: **MethylBERT** 不仅能估算纯度，还能准确地从混合样本中重构出肿瘤和正常细胞各自的甲基化水平，其重构值与参考细胞类型的真实甲基化水平高度相关。

![](https://pic1.imgdb.cn/item/698aebd387851c417e2200fc.png)

## 3.4 MethylBERT在真实临床样本和细胞类型解卷积中的应用

- **图 F, G**: 在分析来自前列腺癌患者的淋巴结样本时，即使没有肿瘤参考数据，**MethylBERT** 也能利用正常细胞甲基化图谱，准确估算出前列腺上皮细胞的比例，该比例与使用肿瘤参考数据估算出的肿瘤纯度高度相关（相关系数 $r > 0.8$）。
- **图 H**: **MethylBERT** 成功地应用于非肿瘤样本的多细胞类型解卷积任务。在对23个白细胞混合样本进行解卷积时，其对五种主要细胞类型（B细胞、**NK**细胞、粒细胞、T细胞和单核/巨噬细胞）的比例估算结果与**图谱研究 (atlas study)**中的方法高度一致。

![](https://pic1.imgdb.cn/item/698aebf287851c417e220136.png)

## 3.5 MethylBERT能够准确检测癌症患者液体活检样本中的稀有肿瘤信号

- **图 A, B**: 在肿瘤比例低于10%的体外模拟样本中，**MethylBERT** 估算的肿瘤纯度与真实值之间的相关性最高（**Spearman** 相关系数 $\rho = 0.98$），中位绝对误差也最低，证明其在检测稀有信号方面具有足够的灵敏度。
- **图 C**: 在真实的**结直肠癌 (CRC)** 患者血浆 **ctDNA** 样本分析中，**MethylBERT** 估算的肿瘤比例在II期及以上的患者中显著高于健康捐赠者，展示了其在早期（II-III期）癌症诊断中的潜力。
- **图 D**: 在对**胰腺导管腺癌 (PDAC)**——一种极难早期发现的癌症——的分析中，**MethylBERT** 同样能够在IIB期患者中检测到与健康人有统计学差异的肿瘤信号。

![](https://pic1.imgdb.cn/item/698aec0f87851c417e22016b.png)