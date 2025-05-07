---
layout: post
title: 'FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores'
date: 2023-11-16
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6804b33058cb8da5c8b904d7.png'
tags: 论文阅读
---

> FlashFFTConv: 使用张量核实现长序列的高效卷积.

- paper：[FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores](https://arxiv.org/abs/2311.05908)
- code：[flashfftconv](https://github.com/HazyResearch/flash-fft-conv/tree/main)

# 0. TL; DR

本文介绍了一种名为 **FlashFFTConv** 的高效卷积算法，专门针对长序列任务进行优化。传统的卷积模型在处理长序列时由于硬件效率低下而受到限制，尤其是快速傅里叶变换（**FFT**）的低效实现。**FlashFFTConv** 通过矩阵分解技术将 **FFT** 转换为矩阵乘法运算，充分利用现代硬件（如 **GPU** 的 **Tensor Cores**）的计算能力，并通过内核融合减少内存 **I/O** 成本。此外，**FlashFFTConv** 还引入了部分卷积和频率稀疏卷积两种稀疏卷积算法，进一步节省内存和计算资源。实验表明，**FlashFFTConv** 在长序列任务中显著提升了卷积的效率，使模型能够处理更长的序列，并在语言建模、图像分类和 **DNA** 建模等任务中取得了更好的性能。

# 1. 背景介绍
在机器学习中，高效处理长序列是一个关键挑战。卷积模型因其在语言建模、时间序列分析、计算机视觉和 **DNA** 建模等任务中的出色表现而受到关注。然而，尽管卷积模型在性能上表现出色，但在处理长序列时，其效率仍然落后于优化良好的 **Transformer** 模型。主要瓶颈在于快速傅里叶变换（**FFT**）的低效实现。**FFT** 虽然在理论上能够在 $O(N \log N)$ 的时间复杂度内完成长卷积运算，但在现代加速器上，**FFT** 的硬件利用率极低。

此外，长序列任务对内存和计算资源的需求极高。例如，在语言建模中，模型需要处理数百万个时间步长的序列；在 **DNA** 建模中，模型需要处理长达数百万个碱基对的基因序列。这些任务对卷积模型的效率和内存管理提出了极高的要求。

# 2. 方法介绍

**FlashFFTConv** 的核心思想是将 **FFT** 卷积转换为矩阵乘法运算，从而充分利用现代硬件（如 **GPU** 的 **Tensor Cores**）的计算能力。在现代硬件设备上，矩阵乘法（绿色和黄色）比一般的浮点运算（红色和蓝色）快得多，使得 **I/O** 成为计算速度的一个主要瓶颈。

![](https://pic1.imgdb.cn/item/681aca4858cb8da5c8e1cc21.png)

**FlashFFTConv** 使用了 [**Monarch 分解**](https://0809zheng.github.io/2025/04/18/fftconv.html#-baileys-fft%E7%AE%97%E6%B3%95)，将 **FFT** 分解为一系列矩阵乘法操作。这种分解不仅提高了计算效率，还减少了内存 **I/O** 成本。

**Monarch** 分解将 **FFT** 分解为 $p$ 个矩阵乘法操作。例如，对于一个长度为 $N = N_1 \times N_2$ 的序列，一个二阶 **Monarch** 分解可以表示为：

$$
F_N = P (I_{N_2} \otimes F_{N_1}) D P^{-1} (I_{N_1} \otimes F_{N_2}) P
$$

其中，$P$ 是一个排列矩阵，用于重塑输入序列；$D$ 是一个对角矩阵，包含 **Twiddle** 因子；$F_{N_1}$ 和 $F_{N_2}$ 是较小的 **FFT** 矩阵。通过这种分解，**FlashFFTConv** 能够将 **FFT** 运算高效地映射到硬件上。

![](https://pic1.imgdb.cn/item/6804ba1d58cb8da5c8b91dd3.png)

**FlashFFTConv** 还通过内核融合减少了内存 **I/O** 成本。内核融合允许将多个操作（如矩阵乘法和逐元素乘法）合并到一个内核中，从而减少数据在不同内存层次之间的传输。例如，对于长序列，**FlashFFTConv** 可以将内层矩阵操作和逐元素乘法融合在一起，仅在最外层矩阵操作时进行 **I/O** 操作。

**FlashFFTConv** 还利用了一些领域特定的优化技术。例如，对于实数到实数的 **FFT**，**FlashFFTConv** 使用了 **一阶段时间抽取** 算法，将长度为 $N$ 的 **FFT** 运算转换为长度为 $N/2$ 的复数 **FFT** 运算，从而将 **FFT** 的计算成本减半。此外，**FlashFFTConv** 还利用了输入和输出的零填充特性，进一步减少了矩阵乘法操作的数量。

**FlashFFTConv** 引入了两种稀疏卷积算法：部分卷积和频率稀疏卷积。
- **部分卷积 (partial convolution)**通过将卷积核的后半部分置零，类似于局部注意力机制。这种设计不仅减少了模型的内存占用，还允许将预训练的卷积模型扩展到更长的序列。例如，通过部分卷积，**FlashFFTConv** 成功将 **HyenaDNA** 模型扩展到 **4M** 序列长度，使其能够处理最长的人类基因（长达 **2.3M** 碱基对）。
- **频率稀疏卷积 (frequency-sparse convolution)**通过在频率空间中置零卷积核的某些部分来实现稀疏性。这种设计不仅减少了计算量，还可能通过去除高频噪声来提高模型的性能。例如，在实验中，**FlashFFTConv** 通过将卷积核的 75% 置零，将 **HyenaDNA** 模型的困惑度降低了 **0.01**。

# 3. 实验分析

**Monarch** 分解可以推广到$p>2$阶的形式，$p$阶分解需要的 **FLOPs** 总量为$O(N^{\frac{p+1}{p}})$。在现代 **GPU** 上，张量核心设计用于将两个 $16\times 16$ 矩阵相乘，因此如果矩阵太小，张量核心的效率就会很低。测试结果表明，对于短序列，二阶分解足够好；但对于较长的序列，二阶分解的开销会迅速增加。同时，对于短序列，高阶分解无法充分利用张量核，但对于较长的序列会显现更好的渐近效果。

![](https://pic1.imgdb.cn/item/681b3cc658cb8da5c8e333cd.png)

**FlashFFTConv** 在多个长序列任务中显著提升了卷积的效率。实验表明，**FlashFFTConv** 在短序列上实现了显著的速度提升，而在长序列上，速度提升虽然有所下降，但仍然显著优于 **PyTorch** 实现。

![](https://pic1.imgdb.cn/item/681b3a7258cb8da5c8e332cc.png)


**FlashFFTConv** 在多个任务中实现了性能提升。例如，在语言建模任务中，**FlashFFTConv** 使 **Hyena-s-155M** 模型的困惑度降低了 **2.3**，相当于将模型参数数量翻倍的效果。在图像分类任务中，**FlashFFTConv** 成功解决了 **Path-512** 任务，这是首次有模型在该任务上达到 96.1% 的准确率。

![](https://pic1.imgdb.cn/item/681b3b5a58cb8da5c8e33331.png)

**FlashFFTConv** 使长序列建模成为可能。例如，在 **DNA** 建模任务中，**FlashFFTConv** 将 **HyenaDNA** 模型扩展到 **4M** 序列长度，使其能够处理最长的人类基因（长达 **2.3M** 碱基对）。这是首次有模型能够在单核苷酸分辨率下嵌入如此长的基因序列。

![](https://pic1.imgdb.cn/item/681b3b0458cb8da5c8e33314.png)

稀疏卷积算法进一步提升了 **FlashFFTConv** 的效率。例如，频率稀疏卷积通过将卷积核的 75% 置零，不仅将卷积的计算速度提升了 **1.3×**，还略微提高了模型的性能。部分卷积则通过减少卷积核的长度，显著减少了模型的内存占用，并允许模型扩展到更长的序列。

![](https://pic1.imgdb.cn/item/681b3ae658cb8da5c8e33305.png)