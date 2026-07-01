---
layout: post
title: 'MultiScale-CNN-4mCPred: a multi-scale CNN and adaptive embedding-based method for mouse genome DNA N4-methylcytosine prediction'
date: 2026-03-20
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6995a45e556e27f1c93a2f04.png'
tags: 论文阅读
---

> MultiScale-CNN-4mCPred: 小鼠基因组DNA N4甲基胞嘧啶预测的多尺度CNN与自适应嵌入方法.

- paper：[MultiScale-CNN-4mCPred: a multi-scale CNN and adaptive embedding-based method for mouse genome DNA N4-methylcytosine prediction](https://link.springer.com/article/10.1186/s12859-023-05135-0)

# 0. TL; DR

**MultiScale-CNN-4mCPred** 是基于多尺度卷积神经网络 **(multi-scale convolution neural network, CNN)** 和自适应嵌入 **(adaptive embedding)** 的计算方法，用于预测小鼠基因组中的 **4mC** 位点。

该模型是一个端到端的学习方法，可以直接将原始 **DNA** 序列作为输入，无需复杂的手动特征设计。在10折交叉验证中，**MultiScale-CNN-4mCPred** 的准确率达到了81.66%；在独立测试中，准确率达到了84.69%，其性能优于现有的先进方法。

# 1. 背景介绍

**DNA** 修饰在发育、衰老、癌症和基因表达调控等生物学过程和疾病中扮演着关键角色。**DNA** 甲基化是其中一种重要的表观遗传修饰，主要类型包括 **5-甲基胞嘧啶 (5mC)**、**N6-甲基腺嘌呤 (6mA)** 和 **N4-甲基胞嘧啶 (4mC)**。

其中，**4mC** 参与了宿主防御、转录调控、基因表达和 **DNA** 复制等生物学功能。与 **5mC** 和 **6mA** 相比，我们对 **4mC** 修饰及其功能的了解还很有限。

检测 **4mC** 位点是探索其功能的关键。尽管存在单分子实时 **(SMRT)** 测序和亚硫酸氢盐测序 **(bisulfite sequencing)** 等实验方法，但它们通常耗时且费力。因此，生物信息学计算工具的出现为大规模识别 **4mC** 位点提供了可能。

近年来，已经有许多计算工具被开发出来用于预测 **4mC** 位点，包括传统的机器学习方法和新兴的深度学习方法。
- 传统方法：如 **iDNA4mC**, **4mCPred** 等，它们依赖于手工设计的特征，虽然简单易用，但其性能受限于特征的质量。
- 深度学习方法：如 **4mCPred-CNN** 和 **Mouse4mC-BGRU**，它们是端到端的学习方法，能够自动从序列中学习特征，并展现出比传统方法更优的性能。

然而，现有的深度学习方法仍有其局限性。例如，**4mCPred-CNN** 未能利用 **LSTM** 或 **GRU** 来捕捉序列的上下文语义，而 **Mouse4mC-BGRU** 则没有利用 **CNN** 来表征局部属性。更重要的是，它们都没有利用多尺度 **(multi-scale)** 的信息。

为了克服上述缺点，作者提出了一个名为 **MultiScale-CNN-4mCPred** 的、基于深度神经网络的计算方法，用于预测小鼠的 **4mC** 位点。该方法结合了多尺度 **CNN** 和 **Bi-LSTM**，旨在同时捕捉序列的尺度表示和语义信息，从而提升预测的准确性。

# 2. MultiScale-CNN-4mCPred方法

**MultiScale-CNN-4mCPred** 是一个基于深度神经网络的端到端学习模型。其整体框架如图所示，主要包含自适应嵌入、**Bi-LSTM**、多尺度 **CNN** 和全连接层四个部分。

![](https://pic1.imgdb.cn/item/6995a4ec556e27f1c93a2f0e.png)

## 2.1 数据集

为了与现有先进方法进行公平比较，作者使用了与 **Mouse4mC-BGRU**, **4mCPred-CNN** 等研究相同的基准数据集。

该数据集来源于 **MethSMRT** 数据库。**DNA** 序列被分割成长度为41 bp的窗口，中心为待预测的胞嘧啶。使用 **CD-HIT** (70%) 去除序列同源性后，将样本随机划分为训练集（746个正样本，746个负样本）和测试集（160个正样本，160个负样本）。

## 2.2 MultiScale-CNN-4mCPred 架构

### 2.2.1 字符编码与自适应嵌入

首先，将 **DNA** 序列中的 A, G, C, T 四个字符分别映射为整数 0, 1, 2, 3。然后，使用一个嵌入 **(embedding)** 层将这些整数序列映射为连续的向量。与固定的 **one-hot** 或预训练的 **Word2Vec** 不同，自适应嵌入层的权重是在模型训练过程中动态学习和调整的，这使得模型能够学习到对当前任务最优的序列表示。

### 2.2.2 双向长短期记忆网络 (Bi-LSTM)

自适应嵌入层的输出被送入一个 **Bi-LSTM** 层，用于提取序列的上下文语义信息。

**LSTM** 是一种擅长处理序列数据的循环神经网络，它通过精巧的“门”结构（输入门、遗忘门、输出门）解决了传统 **RNN** 的长程依赖问题。**Bi-LSTM** 则通过从前向和后向两个方向处理序列，能够更全面地捕捉序列的上下文语义。

### 2.2.3 多尺度 CNN (Multi-scale CNN)

**CNN** 擅长提取局部特征。不同大小的卷积核（即不同的“尺度”）能够捕捉到不同范围的局部信息。

作者并行地使用了三个具有不同尺度（即不同大小的卷积核）的 **CNN** 层。**Bi-LSTM** 的输出被同时送入这三个 **CNN** 层，分别提取不同尺度的局部特征。

### 2.2.4 特征融合与分类

每个 **CNN** 层的输出都经过一个 **dropout** 层以防止过拟合。然后，这三个不同尺度的特征表示被**拼接 (concatenate)** 在一起。

拼接后的特征被送入一个包含三个全连接（**dense**）层的网络。最终的输出层返回一个概率值，表示该位点是 **4mC** 的可能性。

# 3. 实验分析

作者通过一系列实验，对 **MultiScale-CNN-4mCPred** 的不同组成部分进行了评估，并与现有的先进方法进行了比较。

## 3.1 不同尺度 CNN 组合的优化

作者首先探究了不同尺度 **CNN** 组合对模型性能的影响。测试了五种不同的三尺度组合（如 1,3,5；3,5,7 等）。

结果显示，当使用卷积核大小为3、5、7的组合时，模型在所有四个性能指标（**Sn, Sp, Acc, MCC**）上都取得了最优或接近最优的性能。因此，作者在后续的实验中选择了 3, 5, 7 这三种尺度的组合。

![](https://pic1.imgdb.cn/item/6995a58c556e27f1c93a2f12.png)

## 3.2 与不同嵌入方法的比较

作者比较了自适应嵌入与传统的 **one-hot** 编码以及不同 **k-mer** (k=1,2,3) 的 **Word2Vec** 编码的性能。

结果显示，自适应嵌入在 **Sn, Acc, MCC** 三个指标上都取得了最佳性能。这表明，让模型在训练过程中动态地学习序列表示，比使用固定的编码方法效果更好。

![](https://pic1.imgdb.cn/item/6995a5a4556e27f1c93a2f13.png)

## 3.3 与现有先进方法的比较

作者将最终确定的 **MultiScale-CNN-4mCPred** 模型与四种现有的先进方法（**Mouse4mC-BGRU**, **i4mC-Mouse**, **4mCpred-EL**, **4mCPred-CNN**）进行了比较。

在10折交叉验证中，**MultiScale-CNN-4mCPred** 取得了最高的**准确率 (Acc)** (81.66%)，以及次优的 **MCC** 和 **Sn**。与其他方法相比，该模型在**敏感性 (Sn)** 和**特异性 (Sp)** 之间取得了更好的平衡（两者均超过0.8）。

![](https://pic1.imgdb.cn/item/6995a5c1556e27f1c93a2f15.png)

在独立测试集上，**MultiScale-CNN-4mCPred** 的优势更加明显。除了 **Sp** 之外，它的所有指标（**Sn, Acc, MCC**）均为最高。与次优的 **Mouse4mC-BGRU** 相比，其 **Sn** 提升了0.0563，**Acc** 提升了0.0219，**MCC** 提升了0.0429。

![](https://pic1.imgdb.cn/item/6995a5ce556e27f1c93a2f17.png)

作者还将模型与 **4mCPred-CNN** 在不同预测阈值下进行了比较。结果显示，在不同的阈值下，**MultiScale-CNN-4mCPred** 的 **Acc** 和 **MCC** 值均优于 **4mCPred-CNN**。

![](https://pic1.imgdb.cn/item/6995a5db556e27f1c93a2f1a.png)

这些结果表明，**MultiScale-CNN-4mCPred** 是一个具有竞争力的、先进的 **DNA 4mC** 位点预测计算方法。

## 3.4 CNNs 对 4mC 预测的贡献

作者通过每次移除一个尺度的 **CNN**，来研究不同尺度 **CNN** 的贡献。移除大小为3的 **CNN** 会导致 **Sn** 下降。移除大小为5的 **CNN** 会导致 **Sp** 下降。移除大小为7的 **CNN** 会导致 **Sn** 和 **Sp** 都下降。

![](https://pic1.imgdb.cn/item/6995a5f6556e27f1c93a2f21.png)

作者还评估了只使用单一尺度 **CNN** 时的性能。大小为3的 **CNN** 主要贡献于 **Sn**。大小为5的 **CNN** 主要贡献于 **Sp**。大小为7的 **CNN** 对 **Sp** 的贡献大于对 **Sn** 的贡献。

![](https://pic1.imgdb.cn/item/6995a600556e27f1c93a2f24.png)

这些结果表明，不同尺度的 **CNN** 确实捕捉了不同的信息，它们共同对 **4mC** 的预测做出了贡献。

## 3.5 泛化能力

作者还在另外六个不同物种的数据集上测试了 **MultiScale-CNN-4mCPred** 的泛化能力，并与 **DeepTorrent** 和 **4mCPred** 进行了比较。

结果显示，在三种真核生物（**A. thaliana, C. elegans, D. melanogaster**）上，**MultiScale-CNN-4mCPred** 的性能优于另外两种方法。而在三种原核生物（**E. coli, G. pickeringii, G. subterraneus**）上，其性能则稍逊于 **DeepTorrent**。

![](https://pic1.imgdb.cn/item/6995a616556e27f1c93a2f25.png)

这表明，**MultiScale-CNN-4mCPred** 可能更适合于预测真核生物中的 **4mC** 位点。

综上所述，**MultiScale-CNN-4mCPred** 通过其独特的多尺度 **CNN** 和自适应嵌入架构，在小鼠基因组的 **4mC** 位点预测任务上取得了最先进的性能，并显示出良好的跨物种泛化潜力。