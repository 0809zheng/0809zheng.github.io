---
layout: post
title: 'Scaling Laws for Neural Language Models'
date: 2020-05-30
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/67fca85588c538a9b5d0373d.png'
tags: 论文阅读
---

> 神经语言模型中的尺度定律.

- paper：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

# 0.  TL; DR
本文研究了神经语言模型在交叉熵损失上的经验性尺度定律。研究发现，模型性能与模型大小、数据集大小和训练计算量之间存在幂律关系，这些趋势跨越了超过七个数量级。在合理范围内，性能对其他架构超参数（如深度与宽度）的依赖性非常弱。此外，研究还发现，随着模型规模的增加，模型变得更加样本高效，最优的计算高效训练涉及在相对较少的数据上训练非常大的模型，并在收敛之前显著提前停止训练。

# 1. 背景介绍

语言是研究人工智能的自然领域，因为大多数推理任务都可以通过语言高效地表达和评估，同时全球的文本数据为无监督学习提供了丰富的资源。近年来，深度学习在语言建模方面取得了快速进展，最先进的模型在许多特定任务上接近人类水平的表现。

这些模型通常基于**Transformer**架构，其性能受到模型架构、模型大小、训练计算量和训练数据量等因素的影响。本文旨在通过实证研究，探讨语言建模损失与这些因素的依赖关系。

# 2. 方法介绍

**Transformer**架构通过以下超参数进行参数化：
- $n_{\text{layer}}$：层数
- $d_{\text{model}}$：残差流的维度
- $d_{\text{ff}}$：前馈层中间维度
- $d_{\text{attn}}$：注意力输出维度
- $n_{\text{heads}}$：每层的注意力头数

模型大小 $N$（非嵌入参数数量）可以通过以下公式近似计算：

$$
N \approx 2d_{\text{model}}n_{\text{layer}}(2d_{\text{attn}} + d_{\text{ff}}) = 12n_{\text{layer}}d_{\text{model}}^2
$$

其中忽略了偏差和其他次要项，并遵循$d_{\text{attn}}=d_{\text{ff}}/4=d_{\text{model}}$。

模型的前向传播计算量 $C_{\text{forward}}$ （定义为每个**token**的**FLOPs**）可以通过以下公式估算：

$$
C_{\text{forward}} \approx 2N + 2n_{\text{layer}}n_{\text{ctx}}d_{\text{model}}
$$

其中 $n_{\text{ctx}}$ 是输入上下文的长度。

![](https://pic1.imgdb.cn/item/67fcabfa88c538a9b5d04831.png)

由于本文主要研究 $d_{\text{model}} \gg n_{\text{ctx}}/12$ 的模型，上下文相关的计算成本在总计算量中占比很小，因此在训练计算量估计中可以忽略上下文相关的项。考虑到反向传播的计算量大约是前向传播的两倍，因此定义每个**token**的非嵌入训练计算量为$C \approx 6N$。


除非另有说明，作者使用**Adam**优化器进行训练，固定训练步数为 $2.5 \times 10^5$ 步，每步的批量大小为 512 个序列，每个序列包含 1024 个 **token**。由于内存限制，作者对超过 10 亿参数的模型使用 **Adafactor** 进行训练。作者尝试了多种学习率和学习率调度方案，发现收敛时的结果与学习率调度基本无关。除非另有说明，所有训练运行都使用了 3000 步的线性预热，随后是余弦衰减至零的学习率调度。

作者使用 **WebText2** 数据集进行训练，这是 **WebText** 数据集的扩展版本。**WebText2** 包含 2030 万篇文档，总文本量为 96 GB，包含 16.2 亿个单词。经过可逆分词处理后，总 **token** 数量为 22.9 亿。作者保留了 6.6 亿个 **token** 作为测试集，并在类似准备的书籍语料库、**Common Crawl**、英文维基百科和公开的互联网书籍样本上进行测试。

# 3. 实验分析

## 3.1 Transformer中形状超参数的独立性

**Transformer**的性能在固定非嵌入参数总数 $N$ 时，对形状参数 $n_{\text{layer}}$、$n_{\text{heads}}$ 和 $d_{\text{ff}}$ 的依赖性非常弱。作者通过固定模型大小并改变单个超参数进行训练来验证这一点。例如，当改变 $n_{\text{layer}}$ 时，作者同时调整 $d_{\text{model}}$ 以保持 $N \approx 12n_{\text{layer}}d_{\text{model}}^2$ 不变。实验结果表明，性能对这些形状参数的变化非常不敏感，损失仅在几个百分点内变化。即使在参数计数略有差异的情况下，通过使用 $L(N)$ 的拟合作为基线，也可以补偿这些差异。例如，当 $n_{\text{layer}}$ 与 $d_{\text{model}}$ 的比例从 6:4288 变化到 48:1600 时，性能仅下降 3%。

![](https://pic1.imgdb.cn/item/67fcad7e88c538a9b5d04ebd.png)

## 3.2 模型大小与性能

作者训练了多种不同大小的模型，这些模型在 **WebText2** 数据集上训练至近乎收敛，且未出现过拟合现象（除了可能的最大模型）。作者发现，随着非嵌入参数数量 $N$ 的增加，模型性能呈现出明显的幂律趋势，可以用以下公式拟合：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

其中 $N_c \approx 8.8 \times 10^{13}$（非嵌入参数），$\alpha_N \approx 0.076$。

![](https://pic1.imgdb.cn/item/67fcae3288c538a9b5d051b7.png)

值得注意的是，当使用包含嵌入参数的总参数数量进行趋势拟合时，这一趋势会变得不那么明显。这表明，即使将嵌入矩阵的大小减小，也不会对性能产生影响。


## 3.3 数据集大小与性能

为了研究数据集大小 $D$ 对性能的影响，作者在 **WebText2** 数据集的不同子集上训练了一个具有 $n_{\text{layer}} = 36$ 和 $d_{\text{model}} = 1280$ 的模型。训练过程中，作者一直训练到测试损失不再下降为止。

实验结果表明，测试损失与数据集大小 $D$ 之间存在幂律关系，可以用以下公式拟合：

$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

其中 $D_c \approx 5.4 \times 10^{13}$（**token**），$\alpha_D \approx 0.095$。

## 3.4 计算量与性能

作者还研究了训练计算量 $C$ 对性能的影响。训练计算量可以通过 $C = 6NBS$ 估算，其中 $B$ 是批量大小，$S$ 是参数更新的步数。对于给定的计算量 $C$，作者可以在不同的模型大小 $N$ 之间进行扫描，以找到在步数 $S = \frac{C}{6BS}$ 时性能最佳的模型。

实验结果表明，测试损失与训练计算量 $C$ 之间也存在幂律关系，可以用以下公式拟合：

$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

其中 $C_c \approx 1.6 \times 10^7$（**PF-days**），$\alpha_C \approx 0.057$。

![](https://pic1.imgdb.cn/item/67fcaf7888c538a9b5d0576d.png)

## 3.5 过拟合与数据需求

作者进一步研究了在同时改变模型大小 $N$ 和数据集大小 $D$ 时的性能。作者提出了一个关于 $L(N, D)$ 的公式，用于描述在不同 $N$ 和 $D$ 下的最优测试损失：

$$
L(N, D) = \left(\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right)^{\alpha_D}
$$

其中 $N_c \approx 6.4 \times 10^{13}$（非嵌入参数），$\alpha_N \approx 0.076$，$D_c \approx 1.8 \times 10^{13}$（**token**），$\alpha_D \approx 0.103$。

通过实验验证，作者发现这个公式能够很好地拟合数据。实验结果表明，过拟合的程度主要取决于 $N^{\frac{\alpha_N}{\alpha_D}}/D$ 的比值。具体来说，当模型大小 $N$ 增加 8 倍时，作者只需要将数据集大小 $D$ 增加大约 5 倍，就可以避免性能下降。

![](https://pic1.imgdb.cn/item/67fcb09c88c538a9b5d05c09.png)

## 3.6 样本效率与训练时间

作者还研究了模型大小 $N$ 和训练时间（以步数 $S$ 表示）对性能的影响。作者发现，当在无限数据限制下训练时，模型大小 $N$ 和训练步数 $S$ 之间的性能关系可以用以下公式描述：

$$
L(N, S_{\text{min}}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\text{min}}}\right)^{\alpha_S}
$$

其中 $N_c \approx 6.5 \times 10^{13}$（非嵌入参数），$\alpha_N \approx 0.077$，$S_c \approx 2.1 \times 10^3$，$\alpha_S \approx 0.76$。这个公式表明，随着模型大小的增加，模型变得更加样本高效，即在达到相同性能水平时所需的优化步数更少。

![](https://pic1.imgdb.cn/item/67fcb19088c538a9b5d05fad.png)

## 3.7 最优计算预算分配

最后，作者研究了在固定计算预算 $C$ 下，如何最优地分配计算资源以实现最佳性能。作者发现，随着计算预算 $C$ 的增加，最优模型大小 $N$、最优批量大小 $B$ 和最优训练步数 $S$ 都会相应地增加。具体来说，有以下关系：

$$
N \propto C^{\alpha_{\text{min}}^C / \alpha_N} \\
B \propto C^{\alpha_{\text{min}}^C / \alpha_B} \\
S \propto C^{\alpha_{\text{min}}^C / \alpha_S}
$$

其中 $\alpha_{\text{min}}^C = 1 / (1/\alpha_S + 1/\alpha_B + 1/\alpha_N) \approx 0.054$。这些关系与实验结果非常吻合，表明在计算预算增加时，作者应该主要将计算资源用于增加模型大小，而对训练时间和数据集大小的增加相对较少。

![](https://pic1.imgdb.cn/item/67fcb3a488c538a9b5d066d0.png)