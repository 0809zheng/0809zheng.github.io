---
layout: post
title: 'iDNA-OpenPrompt: OpenPrompt learning model for identifying DNA methylation'
date: 2025-12-12
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6996c3d3d2628f800ee0f44d.png'
tags: 论文阅读
---

> iDNA-OpenPrompt: 识别DNA甲基化的开发提示学习模型.

- paper：[iDNA-OpenPrompt: OpenPrompt learning model for identifying DNA methylation](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1377285/full)

# 0. TL; DR

**iDNA-OpenPrompt** 利用了 **OpenPrompt** 学习框架，结合了**提示模板 (prompt template)**、**提示描述器 (prompt verbalizer)** 和**预训练语言模型 (Pre-trained Language Model, PLM)**，为 **DNA** 甲基化序列构建了一个**提示学习 (prompt-learning)** 框架。

在包含17个不同物种和三种甲基化类型（**4mC, 5hmC, 6mA**）的大规模基准数据集上进行的广泛分析表明，**iDNA-OpenPrompt** 模型的性能和鲁棒性均超越了现有的杰出方法（如 **iDNA-ABT, iDNA-ABF**）。

# 1. 背景介绍

**DNA** 甲基化对于众多生物学过程至关重要，并与癌症等多种疾病相关。准确识别 **DNA** 甲基化位点对于理解基因调控和疾病机制是必要的。近年来，深度学习方法已成为识别 **DNA** 甲基化位点的重要工具，并取得了令人鼓舞的成果。

目前，针对三种主要的 **DNA** 甲基化类型（**4mC, 5hmC, 6mA**），已有多种计算方法被开发出来。
- **4mC** 预测：**4mCCNN** 和 **DeepTorrent** 等模型利用 **CNN** 或 **CNN+BiLSTM** 混合架构进行预测。
- **5hmC** 预测：相关研究相对较少，但已有方法开始尝试使用 **k-mer** 嵌入和 **BiLSTM** 等技术。
- **6mA** 预测：**sNNRice6mA** 使用 **CNN**，而 **BERT6mA** 等模型则引入了 **NLP** 领域的 **BERT** 模型，并取得了优异的性能。

尽管这些方法取得了不同程度的进展，但它们大多是为单一类型的 **DNA** 甲基化设计的。只有少数技术（如 **iDNA-ABT, iDNA-ABF, iDNA-MS**）能够处理所有三种甲基化类型。此外，现有的深度学习模型在特征学习方面仍有潜力可挖。

受自然语言处理 **(NLP)** 领域最新范式提示学习 **(prompt learning)** 的启发，作者将基因组序列视为“生物文本”，将序列中的碱基视为“生物单词”。基于此，作者提出了 **iDNA-OpenPrompt** 模型，一个用于 **DNA** 甲基化序列的 **OpenPrompt** 学习方法。该模型通过结合提示模板、提示描述器和预训练语言模型 **(PLM)**，为 **DNA** 甲基化位点的识别构建了一个全新的框架。

# 2. iDNA-OpenPrompt方法

## 2.1 数据集

作者使用了从 **iDNA-MS** 服务器获取的17个基准数据集，这些数据集覆盖了 **4mC, 5hmC, 6mA** 三种甲基化类型和多个物种。所有序列的长度均为41 bp。

## 2.2 提示学习 (Prompt Learning)

提示学习是一种新的 **NLP** 范式，它通过将下游任务（如文本分类）重新构建为类似于预训练语言模型在预训练阶段所见的“完形填空”式任务，从而更有效地利用 **PLM** 的能力。

对于一个输入句子 $x$，通过一个提示模板将其转换为 $x_p = \text{[CLS]} x, \text{a [MASK] question.}$。

模型需要预测 `[MASK]` 位置最有可能填入的词。通过一个提示描述器，将类别标签（如“商业”、“体育”）映射到词汇表中的具体单词（如 “business”, “sports”）。

通过比较不同类别对应的标签词被预测出的概率大小，来确定输入句子的类别。

## 2.3 iDNA-OpenPrompt 模型

作者将提示学习的思想巧妙地应用于 **DNA** 甲基化位点的识别，并基于开源框架 **OpenPrompt** 构建了 **iDNA-OpenPrompt** 模型。

**iDNA-OpenPrompt** 的整体架构如图所示。其核心是一个提示模型 **(prompt model)**，主要由三个部分组成：提示模板、提示描述器和预训练语言模型 **(PLM)**。

![](https://pic1.imgdb.cn/item/6996c4cdd2628f800ee0f464.png)


### 2.3.1 提示模板 (Prompt Template)

提示模板用于将输入的 **DNA** 序列构建成一个适合 **PLM** 处理的格式。作者采用了一种可训练的手动模板。

与传统 **NLP** 不同，单个或几个核苷酸并不能决定甲基化位点。作者构建了一个包含不同长度（**k-mer** 从1到6）的所有可能核苷酸组合的词汇库，总共包含5,460个“生物单词”。使用 **BERT** 的分词器处理这个 **DNA** 词汇库，以生成 **iDNA-OpenPrompt** 模型的分词器。

### 2.3.2 提示描述器 (Prompt Verbalizer)

提示描述器用于将模型的输出映射到任务特定的标签上（在这里是“甲基化”或“非甲基化”）。作者采用了一种手动描述器。

作者提出了一种为 **DNA** 序列构建特定标签词的方法。以序列的第21个核苷酸（中心位点）为中心，在其两侧使用 **k-mer=6** 的方式进行编码，生成一系列6个碱基长度的“单词”。所有从阳性样本中生成的“单词”被作为“阳性样本标签词”。所有从阴性样本中生成的“单词”被作为“阴性样本标签词”。模型最终需要预测 `[MASK]` 位置是更可能填入“阳性标签词”还是“阴性标签词”。

![](https://pic1.imgdb.cn/item/6996c527d2628f800ee0f46e.png)

### 2.3.3 PLM：BERT 模型

**iDNA-OpenPrompt** 的 **PLM** 是 **BERT** 模型。**BERT** 的核心是 **Transformer** 的编码器部分，它由多个包含多头自注意力 **(multi-head attention)** 机制的编码器层组成。

自注意力通过计算输入序列中每个元素与其他所有元素之间的相关性（“注意力”），来动态地更新每个元素的表示。其计算公式为：

$$
\text{Self-attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

通过并行地运行多个独立的自注意力头，模型能够从不同的表示子空间中学习信息，从而增强学习能力。

# 3. 实验分析

作者对 **iDNA-OpenPrompt** 模型的性能进行了全面的评估，包括其特征表示能力、与现有方法的比较以及跨物种的泛化能力。

## 3.1 iDNA-OpenPrompt 样本的可视化

作者使用 **UMAP** 对经过 **iDNA-OpenPrompt** 模型处理前后的样本分布进行了可视化。在模型处理之前，阳性（红色）和阴性（蓝色）样本是混合在一起的，难以区分。经过 **iDNA-OpenPrompt** 模型处理后，在所有六个代表性的物种（包括 **5hmC**, **4mC**, **6mA** 三种类型）上，阳性和阴性样本都被清晰地分离成了两个独立的簇。这直观地证明了 **iDNA-OpenPrompt** 模型强大的特征学习和类别区分能力。

![](https://pic1.imgdb.cn/item/6996c596d2628f800ee0f47a.png)

## 3.2 iDNA-OpenPrompt 与现有优秀方法的性能比较

作者在17个基准数据集上，将 **iDNA-OpenPrompt** 与四种优秀的预测器（**iDNA-ABT, iDNA-ABF, iDNA-MS, MM-6mAPred**）在五个评估指标（**ACC, SN, SP, AUC, MCC**）上进行了全面比较。

结果清晰地显示，在所有17个数据集上，**iDNA-OpenPrompt** 在所有五个评估指标上的性能都一致地优于其他四种方法。这表明，作者提出的 **OpenPrompt** 学习框架，以及为 **DNA** 甲基化序列专门设计的提示模板和提示描述器，是极其有效的。

![](https://pic1.imgdb.cn/item/6996c5c6d2628f800ee0f47e.png)

## 3.3 成功的跨物种验证结果

为了评估模型的跨物种泛化能力，作者进行了交叉验证。即在一个物种上训练模型，然后在其他物种上进行测试。在 **5hmC** 的两个物种之间进行交叉测试，准确率均达到了98%以上。在 **4mC** 的四个物种之间进行交叉测试，准确率也普遍很高。在 **6mA** 的十一个物种之间进行交叉测试，除了少数几个物种外，模型的准确率都令人满意。**iDNA-OpenPrompt** 表现出强大的跨物种泛化能力，表明该模型能够学习到不同物种间共通的甲基化模式。

![](https://pic1.imgdb.cn/item/6996c5e1d2628f800ee0f482.png)

## 3.4 DNA 词汇库和标签词对模型准确率的影响

作者还探究了提示模板中 **DNA** 词汇库的最大 **k-mer** 长度，以及提示描述器中标签词的 **k-mer** 长度，对模型性能的影响。当固定标签词长度为6时，**DNA** 词汇库的最大 **k-mer** 长度为6时模型准确率最高。当固定 **DNA** 词汇库长度为6时，标签词的长度为6时模型准确率最高。当同时改变两者长度时，同样是在长度都为6时，模型准确率达到峰值。**k-mer=6** 是为 **iDNA-OpenPrompt** 模型构建 **DNA** 词汇库和标签词的最佳长度。

![](https://pic1.imgdb.cn/item/6996c61ed2628f800ee0f48a.png)

综上所述，**iDNA-OpenPrompt** 通过将新颖的 **OpenPrompt** 学习框架应用于 **DNA** 甲基化识别任务，并为其设计了领域特异性的提示模板和提示描述器，成功地构建了一个在性能和泛化能力上均超越现有方法的通用预测模型。