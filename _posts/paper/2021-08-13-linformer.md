---
layout: post
title: 'Linformer: Self-Attention with Linear Complexity'
date: 2021-08-13
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6114c5e15132923bf88f2f23.jpg'
tags: 论文阅读
---

> Linformer: 线性复杂度的自注意力机制.

- paper：Linformer: Self-Attention with Linear Complexity
- arXiv：[link](https://arxiv.org/abs/2006.04768)

作者通过实验发现**Transformer**中的自注意力矩阵大多是低秩的。实验对**RoBERTa**预训练模型不同层的自注意力矩阵进行奇异值分解，并绘制奇异值的累积分布曲线，如下图所示。
- 图中奇异值分布呈长尾分布，即只有一小部分奇异值具有较大的值，大部分奇异值具有较小的数值；这表明自注意力矩阵的大部分信息都可以通过少量奇异值进行恢复。
- 下图中的奇异值热图显示了不同层的最大相对奇异值分布。图中表示更高的层中会有更多信息集中在少量最大的奇异值中，即自注意力矩阵的秩是更低的。

![](https://pic.imgdb.cn/item/6114c60c5132923bf88f8303.jpg)

本文中作者提出了**Linformer**，在自注意力的计算过程中引入低秩分解，从而实现近似线性的计算复杂度。

若记$Q,K,V \in \Bbb{R}^{n \times d}$，则标准的**Attention**计算为:

$$ \text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d}})V $$

其中矩阵乘法$QK^T$的计算复杂度为$O(n^2d)=O(n^2)$。作者为$K$和$V$引入了低秩映射$E,F \in \Bbb{R}^{k \times n}$，使得注意力计算变为：

$$ \text{Attention}(Q,K,V)=\text{softmax}(\frac{Q(EK)^T}{\sqrt{d}})(FV) $$

从而使得矩阵乘法的计算复杂度降低为$O(kn)$。

![](https://pic.imgdb.cn/item/6114c6365132923bf88fcfc9.jpg)

作者还为**Linformer**设计了一些参数共享方法以提升模型的效率和表现。实验设置了三种层级的参数共享：
1. **Headwise**：多头自注意力中每一层的所有自注意力头分别共享$E$和$F$的参数。
2. **Key-Value**：每一层的所有自注意力头的$E$和$F$共享同一参数$E=F$。
3. **Layerwise**：所有层共享投影矩阵$E$。

模型使用**RoBERTa**结构，训练使用**BookCorpus**和英文维基百科(共约$3300$M单词)作为预训练语料库，采用**mask**语言模型作为预训练任务。实验结果如下：

![](https://pic.imgdb.cn/item/6114c6b65132923bf890be77.jpg)

实验使用**困惑度**作为模型评估指标。困惑度越低，则模型表现越好。实验分析如下：
- 图(a)和(b)显示，随着投影维度$d$增加，模型的困惑度越低，模型表现越好。
- 图(c)显示，不同的参数共享结果接近，因此可以使用**Layerwise**的共享参数设置降低参数量。
- 图(d)显示，随着输入序列长度增大，训练前期困惑度较大，但收敛后不同序列长度的困惑度接近，说明模型对于输入序列长度近似线性复杂度。



![](https://pic.imgdb.cn/item/6114cecf5132923bf8a139ca.jpg)