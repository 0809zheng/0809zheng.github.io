---
layout: post
title: 'Efficient Attention: Attention with Linear Complexities'
date: 2021-08-15
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6117b6f25132923bf88a0953.jpg'
tags: 论文阅读
---

> 具有线性复杂度的高效自注意力机制.

- paper：Efficient Attention: Attention with Linear Complexities
- arXiv：[link](https://arxiv.org/abs/1812.01243)

通常的自注意力机制运算可以表示为：

$$ D(Q,K,V) = \rho(QK^T)V = \sigma_{\text{row}}(QK^T)V  $$

其中$Q,K \in \Bbb{R}^{n \times d_k},V \in \Bbb{R}^{n \times d_v}$$，\sigma_{\text{row}}$表示对矩阵的每一行应用**softmax**函数进行归一化，这首先需要计算$QK^T$，使得计算量为$O(n^2)$。

作者注意到，若矩阵$Q$的每一行、矩阵$K$的每一列是归一化的，则矩阵$QK^T$的每一行也是归一化的。证明如下：

$$ (QK^T)_{ij} = \sum_{k}^{} Q_{ik}{(K^T)}_{kj} = \sum_{k}^{} Q_{ik}{K}_{jk} $$

$$ \sum_{j}^{} (QK^T)_{ij} = \sum_{j}^{}\sum_{k}^{} Q_{ik}K_{jk} = \sum_{k}^{} Q_{ik} \sum_{j}^{}K_{jk} = \sum_{k}^{} Q_{ik} = 1 $$

因此对矩阵$Q$的每一行应用**softmax**函数进行归一化，对矩阵$K$的每一列应用**softmax**函数进行归一化，从而近似对矩阵$QK^T$的每一行应用**softmax**函数进行归一化：

$$ \sigma_{\text{row}}(QK^T)≈\sigma_{\text{row}}(Q)\sigma_{\text{col}}(K)^T=\rho_Q(Q)\rho_K(K)^T $$

因此自注意力机制可以被简化为：

$$ E(Q,K,V) = \rho_Q(Q)\rho_K(K)^TV = \rho_Q(Q)(\rho_K(K)^TV)  $$

通过矩阵乘法的结合律，优先计算$\rho_K(K)^TV$，可将计算复杂度降低为$O(n)$。

![](https://pic.imgdb.cn/item/611ca3a64907e2d39c95af86.jpg)

作者在**COCO2017**数据集上进行了目标检测和图像分割等实验，结果表明该方法能够有效地降低计算成本，且保持相当的性能。

![](https://pic.imgdb.cn/item/611ca4104907e2d39c99226f.jpg)