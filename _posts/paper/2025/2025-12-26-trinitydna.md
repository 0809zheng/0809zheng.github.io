---
layout: post
title: 'TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling'
date: 2025-12-26
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/699aaf00ba83e2a3a482183f.png'
tags: 论文阅读
---

> TrinityDNA：用于高效长序列DNA建模的生物启发基础模型.

- paper：[TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling](https://arxiv.org/abs/2507.19229)

# 0. TL; DR

**TrinityDNA** 模型是一个整合了生物学启发的组件的**DNA**基础模型，包括：
-   **Groove Fusion**：用于捕捉**DNA**的大沟（**major groove**）和小沟（**minor groove**）结构特征。
-   门控反向互补**（Gated Reverse Complement, GRC）**：用于处理**DNA**序列固有的对称性。
-   多尺度注意力机制：使模型能够关注不同层次的序列依赖关系；
-   进化训练策略：使模型能够逐步适应从原核生物到真核生物的基因组。

**TrinityDNA**为基因组序列建模提供了一种更准确、更高效的方法，在基因功能预测、调控机制发现和其他基因组学应用中取得了显著的改进。此外，作者还引入了一个新的**DNA**长序列**CDS**注释基准，以使评估更全面、更面向实际应用。

# 1. 背景介绍

大规模长序列建模的快速发展，特别是在自然语言处理（**NLP**）领域的突破，从根本上改变了我们处理复杂数据的方式。深度学习模型，如**Transformers**，在从语言翻译到文本生成的任务中取得了前所未有的成功。这些进展为将序列建模的强大能力扩展到基因组学带来了令人兴奋的机会，因为基因组数据与自然语言在某些关键方面具有相似性，例如其序列性。

然而，直接将传统的**NLP**模型应用于基因组序列是困难的。与**NLP**中常见的高度密集和结构化的数据不同，基因组序列本质上是稀疏的，包含大量重复和变异。现有的模型常常难以识别长程依赖关系并解释潜在的生物学结构。此外，当前模型中缺乏生物学启发的特征，限制了它们在基因组环境中的有效性。

为了解决这些问题，作者引入了**TrinityDNA**，一个专门为克服当前基因组序列建模局限性而设计的新型**DNA**基础模型。**TrinityDNA**利用深度学习的最新进展，创建了一个针对**DNA**序列独特挑战进行优化，同时融入关键生物学洞察的模型。

# 2. TrinityDNA 模型

**TrinityDNA**旨在通过创新的架构和训练方法，全面地对**DNA**进行建模。

![](https://pic1.imgdb.cn/item/699ab056ba83e2a3a4821860.png)

## 2.1 捕捉DNA的物理与对称特性

###  Groove Fusion模块

**DNA**的双螺旋结构具有大沟（**major groove**）和小沟（**minor groove**）两种不同的凹槽，它们在蛋白质结合和分子相互作用中扮演着不同的角色。为了模拟这些结构差异，作者提出了**Groove Fusion**模块。

该模块使用三种不同大小的卷积核（3、5和7）对**DNA**序列进行分词和卷积操作。这三种尺度的卷积可以分别捕捉与小沟和中等、较大结构（如大沟）相关的空间特征。

$$
\text{GrooveFusion}(S) = \sum_{k \in \{3,5,7\}} \text{GELU}(\text{Conv}_k(S))
$$

其中$S$是输入的**DNA**序列，$\text{Conv}_k$是核大小为$k$的卷积操作。通过融合不同尺度的卷积输出，模型能够捕捉到对解释**DNA**结构变异至关重要的多尺度上下文信息。

### 门控反向互补（Gated Reverse Complement, GRC）

**DNA**的一个基本特性是其双链的反向互补性（**reverse complementarity**）。这意味着正向链和反向互补链在生物学上通常是等价的。为了利用这一先验知识，作者设计了**GRC**机制。

**GRC**机制使用一个共享的**Transformer**模块，并行处理正向序列（$S$）及其反向互补序列（$S^R$）。然后，通过一个门控机制将两个表示进行有效结合。

$$
\text{GRC}(S, S^R) = f_\theta(S) + \sigma(W_G \cdot f_\theta(\text{Flip}(S^R)))
$$

其中$f_\theta$是共享的**Transformer**网络，$\sigma$是激活函数，$\text{Flip}$操作将反向互补序列的输出翻转回原始顺序。这种设计使模型能够同时学习两条链的表示，利用了**DNA**的对称性。

## 2.2 克服长程依赖建模的挑战

传统的全注意力机制在处理长序列时会遇到计算瓶颈和“过平滑”（**oversmoothing**）问题，即注意力分数趋于均匀，丢失了关键信息。为了解决这个问题，作者提出了滑动多窗口注意力**（Sliding Multi-Window Attention, SMWA）**。

**SMWA**将不同的注意力头（**attention heads**）分配给不同大小的注意力窗口（例如，128, 512, 2048, 8192个词元）。一些头关注局部细节（小窗口），而另一些头则捕捉全局长程依赖（大窗口）。

对于第$i$个位置，第$h$个注意力头的计算被限制在其大小为$L_h$的滑动窗口内：

$$
\text{Attn}_h(S_i) = \text{Softmax}\left(\frac{Q_h(i)K_h(i+[-\frac{L_h}{2}, \frac{L_h}{2}])^T}{\sqrt{d_k}}\right)V_h(i+[-\frac{L_h}{2}, \frac{L_h}{2}])
$$

这种多尺度机制使得模型能够同时捕捉**DNA**序列中的局部基序和远距离调控关系。

## 2.3 进化训练策略

为了让模型能够适应从简单到复杂的不同基因组结构，作者设计了一种“进化训练策略”（**Evolutionary Training Strategy, ETS**）。
-   第一阶段：首先在相对简单的原核生物基因组上进行训练。原核生物基因组较短，调控结构直接，这使得模型能够快速学习到**DNA**序列的基本基序和组织模式。
-   第二阶段：接着，在更复杂的真核生物基因组上继续训练。真核生物基因组具有内含子-外显子结构，基因长度可达数万个碱基。同时，模型的上下文窗口也从8k扩展到100k，以适应更复杂的调控元件。

通过这种从简单到复杂的渐进式学习，模型获得了跨越不同物种和基因组复杂度的强大泛化能力。

# 3. 实验分析

作者在一系列实验中系统地评估了**TrinityDNA**的性能。

## 3.1 预训练与消融实验

比较不同模型组件（**GRC**, **GFM**, **SMWA**）对模型预训练性能（以困惑度**PPL**衡量）的影响，并评估模型的计算效率和训练策略的有效性。

**TrinityDNA**在不同参数规模下，其性能（困惑度**PPL**）都随着计算量的增加而稳步提升，并且始终优于基线模型。此外，增加上下文窗口长度能显著降低**PPL**，证明了长程建模的优势。

![](https://pic1.imgdb.cn/item/699ab0f8ba83e2a3a482186b.png)

**Groove Fusion (GFM)**和**Gated Reverse Complement (GRC)**模块都显著降低了模型的困惑度，证明了融入**DNA**结构和对称性先验知识的有效性。

![](https://pic1.imgdb.cn/item/699ab116ba83e2a3a482186e.png)

在处理不同长度的序列时，**TrinityDNA**的吞吐量远高于其他模型（如**DNABERT-2**），即使在处理64k长度的序列时仍能保持高效率。进化训练策略（先在原核生物上预训练，再在多物种上继续训练）的性能，优于直接在混合数据集上从头训练的模型。这验证了分阶段、由简到繁的训练策略的优越性。

![](https://pic1.imgdb.cn/item/699ab12aba83e2a3a482186f.png)

## 3.2 GUE基准测试

在**Genomic Understanding Evaluation (GUE)**基准测试上评估**TrinityDNA**的性能，该基准包含调控元件分类、组蛋白标记预测、剪接位点注释等多种任务。

**TrinityDNA**在多个**GUE**任务上取得了**state-of-the-art**的性能，其平均性能显著优于**DNABERT**、**NT**和**Caduceus**等模型。特别是在需要识别长程依赖的任务中（如人类转录因子结合位点预测），**TrinityDNA**的优势尤为明显，这得益于其多尺度注意力和**GRC**等设计。

![](https://pic1.imgdb.cn/item/699ab144ba83e2a3a4821875.png)

## 3.3 零样本性能评估

在不进行任何微调的情况下，直接评估预训练好的**TrinityDNA**在19个下游任务上的性能，包括**DNA**致病性预测、**RNA**和蛋白质的深度突变扫描（**DMS**）等。

作者展示了两个模型：**TrinityMicroDNA**（仅在原核生物上训练）和**TrinityDNA**（经过进化训练）。**TrinityMicroDNA**在13个原核生物任务中的8个上取得了最佳成绩，平均性能最高。**TrinityDNA**则在真核生物的蛋白质适应性预测任务上表现出色，平均性能甚至超过了规模大得多的**EVO2-40B**模型。这些结果凸显了进化阶段感知预训练的好处，即针对不同生物域训练专门的模型可以获得更好的性能。

![](https://pic1.imgdb.cn/item/699ab15dba83e2a3a4821876.png)

## 3.4 CDS注释基准测试

作者引入了一个新的编码序列（**Coding Sequence, CDS**）注释基准，用于评估模型在真实基因组中识别基因结构的能力。

在这个更贴近实际应用的基准上，**TrinityMicroDNA-1B**在精确匹配（**Exact Match**）任务中取得了最高的精确率（**Precision**）和**F1**分数。虽然传统的基因预测工具**Prodigal**在召回率（**Recall**）上表现出色，但**TrinityDNA**的整体性能（**F1**分数）更优，显示了其在不同数据集上的强大泛化能力。

![](https://pic1.imgdb.cn/item/699ab173ba83e2a3a482187a.png)