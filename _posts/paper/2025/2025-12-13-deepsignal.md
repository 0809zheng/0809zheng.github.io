---
layout: post
title: 'DeepSignal: detecting DNA methylation state from Nanopore sequencing reads using deep-learning'
date: 2025-12-13
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6996c7f1d2628f800ee0f4ac.png'
tags: 论文阅读
---

> DeepSignal：使用深度学习从纳米孔测序中检测DNA甲基化状态.

- paper：[DeepSignal: detecting DNA methylation state from Nanopore sequencing reads using deep-learning](https://academic.oup.com/bioinformatics/article/35/22/4586/5474907)

# 0. TL; DR

**DeepSignal** 是一种利用深度学习从纳米孔测序 **reads** 中检测 **DNA** 甲基化状态的方法。核心是一个双模块架构，**信号特征模块 (signal feature module)** 使用 **CNN** 直接从原始电信号中构建特征；**序列特征模块 (sequence feature module)** 使用双向循环神经网络 **(BRNN)** 从信号信息的序列中构建特征。这种设计能够同时捕捉信号的底层物理特性和序列的上下文信息。

在对**智人 (H. sapiens)**、**大肠杆菌 (E. coli)** 和 **pUC19** 的纳米孔 **reads** 进行测试时，**DeepSignal** 在检测 **6mA** 和 **5mC** 甲基化状态方面，其读长级别和基因组级别的性能均优于先前的方法。**DeepSignal** 所需的覆盖度远低于基于 **HMM** 和统计学的方法。

# 1. 背景介绍

**DNA** 甲基化作为一种关键的表观遗传标记，在许多关键的生物学过程中扮演着重要角色。**N6-甲基腺嘌呤 (6mA)** 和 **5-甲基胞嘧啶 (5mC)** 是两种最普遍、研究最深入的 **DNA** 碱基甲基化。

近年来，**PacBio** 单分子实时 **(SMRT)** 测序和纳米孔 **(Nanopore)** 测序等单分子测序技术已被证明能够直接检测 **DNA** 甲基化标记。这两种技术都通过其独特的信号来区分修饰碱基和标准核苷酸碱基。特别是纳米孔测序，其电信号对核苷酸的表观遗传变化非常敏感。

目前，已有多种基于模型或基于统计学的方法被开发出来，用于从纳米孔测序 **reads** 中识别 **DNA** 甲基化。
- 基于模型的方法：如 **nanopolish** 和 **signalAlign**，它们大多基于隐马尔可夫模型 **(HMM)**。这些方法首先在**read**级别上预测每个目标碱基的甲基化状态，然后将所有覆盖该位点的 **reads** 的预测结果进行汇总，得到基因组级别的甲基化状态。
- 基于统计学的方法：如 **nanoraw** 和 **NanoMod**。这些方法需要两组 **reads**：一组来自天然 **DNA**，另一组来自匹配的扩增 **DNA**（作为对照）。通过检验两组 **reads** 信号的显著性差异来预测甲基化状态。

尽管这些方法取得了一定的成功，但它们仍然存在一些局限性。例如，基于 **HMM** 的方法在处理包含混合甲基化状态的 **k-mers** 时表现不佳；而基于统计学的方法不仅需要额外的对照样本，而且准确性通常低于基于模型的方法。

为此，作者提出了一种名为 **DeepSignal** 的深度学习方法。该方法旨在通过一个结合了卷积神经网络 **(CNN)** 和双向循环神经网络 **(BRNN)** 的双模块架构，直接从原始的纳米孔电信号中学习特征，从而实现对 **DNA** 碱基甲基化状态更准确、更鲁棒的预测。

# 2. DeepSignal方法

## 2.1 数据

作者使用了多种公开可用的纳米孔测序数据和亚硫酸氢盐测序数据进行模型训练和评估，覆盖了**智人 (H. sapiens)**、**大肠杆菌 (E. coli)** 和 **pUC19** 质粒，以及 **5mC** 和 **6mA** 两种修饰类型。

## 2.2 DeepSignal 模型

**DeepSignal** 的整体架构如图所示，主要由四个模块组成：信号提取、信号特征模块、序列特征模块和分类模块。

![](https://pic1.imgdb.cn/item/6996c908d2628f800ee0f4e1.png)

### 2.2.1 纳米孔读数的信号提取

在提取信号特征之前，需要进行两个预处理步骤：
1.  碱基检出 **(Basecall)**：使用 **ONT** 官方的碱基检出软件 **(basecaller)**（如 **Albacore**）从原始信号中预测出碱基序列。
2.  重摆动 **(Re-squiggle)**：使用 **nanoraw/tombo** 工具，将原始的电信号值与基因组参考序列上的连续碱基进行对齐。这一步可以校正碱基检出中的插入/删除错误。

对每个 **read** 的原始信号进行归一化。

$$
\text{signal}_{\text{norm}} = \frac{\text{signal}_{\text{raw}} - \text{median}(\text{signals})}{\text{MAD}(\text{signals})}
$$

其中，**MAD** 是中位数绝对偏差 **(median absolute deviation)**。

完成预处理后，对于每个待预测的甲基化位点，作者以其为中心提取一个17 bp核苷酸序列的信号。

### 2.2.2 序列特征模块 (Sequence Feature Module)

该模块使用双向循环神经网络 **(BRNN)** 从信号信息的序列中构建特征。

对于17 bp窗口中的每一个碱基，提取四个特征：核苷酸类型、映射到该碱基的信号值的均值、标准差和数量。这构成了一个长度为17的序列，每个时间步有4个特征。

使用长短期记忆 **(LSTM)** 作为 **RNN** 的单元。**LSTM** 通过其输入门 $i_t$、遗忘门 $f_t$ 和输出门 $o_t$ 来控制信息流，能够有效地捕捉序列中的长程依赖关系。

$$
\begin{align*}
i_t &= \text{sigmoid}(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci} \cdot c_{t-1} + b_i) \\
f_t &= \text{sigmoid}(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf} \cdot c_{t-1} + b_f) \\
c_t &= f_t \cdot c_{t-1} + i_t \cdot \tanh(W_{xc}x_t + W_{hc} \cdot h_{t-1} + b_c) \\
o_t &= \text{sigmoid}(W_{xo}x_t + W_{ho}h_{t-1} + W_{co} \cdot c_t + b_o) \\
h_t &= o_t \cdot \tanh(c_t)
\end{align*}
$$

双向 **(Bidirectional)** 结构意味着模型会同时从前向和后向处理序列，以捕捉更全面的上下文信息。

该模块最终为每个样本生成一个表示向量 $z$。

### 2.2.3 信号特征模块 (Signal Feature Module)

该模块使用深度 **CNN** 直接从原始电信号中构建特征。

对于每个待预测位点，提取其17-mer窗口中心区域的360个原始信号值。

**CNN** 架构是一个 **GoogLeNet** 的变体，由11个堆叠的初始模块 **(inception blocks)** 组成。每个初始模块并行地包含 $1 \times 1$, $1 \times 3$, $1 \times 5$ 的卷积，一个残差连接的 $1 \times 3$ 卷积，以及一个 $1 \times 3$ 的最大池化 **(maxpool)**。

### 2.2.4 分类模块

来自两个特征模块的特征向量被拼接起来，并送入一个包含两个隐藏层的全连接神经网络。最终，一个 **sigmoid** 激活函数输出目标位点的甲基化概率。

# 3. 实验分析

作者在多种数据集上对 **DeepSignal** 的性能进行了评估，并与现有的基于 **HMM** 的方法（如 **nanopolish** 和 **signalAlign**）进行了比较。

## 3.1 在 E. coli 和 H. sapiens 的 CpG 甲基化数据上的评估

作者首先在 **E. coli** 和 **H. sapiens** 的 **5mC** 数据上，对 **DeepSignal** 和 **nanopolish** 进行了读长级别 **(read level)** 的性能评估。

在 **E. coli** 数据集内验证: **DeepSignal** 在所有评估指标上都优于 **nanopolish**。在 **H. sapiens** 数据集内验证: **DeepSignal** 同样表现更优，尽管在**特异性 (specificity)** 和**精确率 (precision)** 上的优势较小。跨基因组验证 **(E. coli 训练, H. sapiens 测试)**: **DeepSignal** 的性能（准确率 0.938）显著高于 **nanopolish** (0.894)，证明其具有更强的泛化能力。

![](https://pic1.imgdb.cn/item/6996cb67d2628f800ee0f534.png)

## 3.2 在 pUC19 质粒的 GATC 和 CCWGG 甲基化数据上的评估

作者接着在 **pUC19** 质粒的 **6mA (GATC)** 和 **5mC (CCWGG)** 数据上，对 **DeepSignal** 和 **signalAlign** 进行了基因组级别 **(genome level)** 的性能评估。

**GATC (6mA) 预测**: **DeepSignal** 的准确率达到了0.999，远高于 **signalAlign** 的0.908。**CCWGG (5mC) 预测**: **DeepSignal** 的准确率达到了0.997，同样高于 **signalAlign** 的0.962。**DeepSignal** 在不同甲基化碱基和不同甲基化基序上都表现出了一致的高性能。

![](https://pic1.imgdb.cn/item/6996cba0d2628f800ee0f536.png)

## 3.3 与亚硫酸氢盐测序在人类 DNA CpG 甲基化状态检测上的比较

作者将 **DeepSignal** 的预测结果与被视为“金标准”的亚硫酸氢盐测序结果进行了比较。

在高置信度的 **DNA CpG** 位点上，对单个 **read** 进行评估。**单例 CpG (singleton CpGs) 预测**: **DeepSignal** 在所有指标上都显著优于 **nanopolish**。**混合 CpG (mixed CpGs) 预测**: 对于周围10 bp内存在不同甲基化状态 **CpG** 的复杂情况，**DeepSignal** 仍然能保持很高的性能（准确率约0.84-0.86），而 **nanopolish** 的性能则急剧下降（准确率约0.53），几乎相当于随机猜测。这证明了 **DeepSignal** 能够更好地捕捉 **CpG** 的甲基化特征。

![](https://pic1.imgdb.cn/item/6996cbc4d2628f800ee0f537.png)

评估了 **DeepSignal** 预测的甲基化频率与亚硫酸氢盐测序结果之间的皮尔逊相关性 **(Pearson correlation)**。在 **HX1** 样本上，**DeepSignal** 的相关性（0.923）高于 **nanopolish** (0.886)。在 **NA12878** 样本上，**DeepSignal** 的相关性（0.920）同样高于 **nanopolish** (0.892)。

![](https://pic1.imgdb.cn/item/6996cbe7d2628f800ee0f538.png)

**DeepSignal** 能够比亚硫酸氢盐测序多预测5-6%的 **CpG** 位点，这些位点大多位于**着丝粒 (Centromere)**、**CpG 岛**和**内含子 (Intron)** 等区域。

![](https://pic1.imgdb.cn/item/6996cbfcd2628f800ee0f539.png)

## 3.4 数据覆盖度对 DeepSignal 性能的影响

作者评估了不同 **read** 覆盖度对预测性能的影响。
- **CpG 位点预测准确率 (左侧)**: 仅需 1X 的覆盖度，**DeepSignal** 在 **HX1** 和 **NA12878** 上的准确率就分别达到了0.924和0.923，显著高于 **nanopolish**。
- **与亚硫酸氢盐测序的相关性 (右侧)**: 仅需 20X 的覆盖度，**DeepSignal** 与亚硫酸氢盐测序的相关性就可以达到0.9以上，而 **nanopolish** 在40X覆盖度下也未能达到这一水平。

**DeepSignal** 能够在远低于现有方法所需的覆盖度下，实现高精度的甲基化状态预测。

![](https://pic1.imgdb.cn/item/6996cc23d2628f800ee0f53f.png)

综上所述，**DeepSignal** 通过其创新的双模块深度学习架构，实现了在低覆盖度下对多种 **DNA** 甲基化类型的高精度、高鲁棒性预测，其性能全面超越了现有的基于 **HMM** 的方法，并与“金标准”亚硫酸氢盐测序高度相关。