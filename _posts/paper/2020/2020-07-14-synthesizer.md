---
layout: post
title: 'Synthesizer: Rethinking Self-Attention in Transformer Models'
date: 2020-07-14
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f0d9e5014195aa594e1ad35.jpg'
tags: 论文阅读
---

> Synthesizer：使用合成注意力的Transformer模型.

- paper：Synthesizer: Rethinking Self-Attention in Transformer Models
- arXiv：[link](https://arxiv.org/abs/2005.00743v1)

**Transformer**模型中使用了自注意力机制机制，即对于输入$X$，分别使用**query**、**key**、**value**仿射变换得到$Q(X)$、$K(X)$、$G(X)$，再通过**query**和**key**的交互实现自注意力分布$B=K(X)Q(X)$，最后计算输出结果$Y=\text{Softmax}(B)G(X)$。

作者提出了**合成注意力(synthetic attention)**机制的想法，即不引入仿射变换$Q(X)$和$K(X)$，而是直接对输入$X$进行变换得到自注意力分布$B=F(X)$。

![](https://pic.downk.cc/item/5f0d990e14195aa594e01fe2.jpg)

**Synthesizer**包含以下几种形式：
- **Dense**：使用$2$层神经网络对输入进行拟合得到自注意力分布$B=W_2(σ(W_1(X)+b_1))+b_2$
- **Random**：随机初始化自注意力分布$B$，随训练更新
- **Fixed Random**：随机初始化自注意力分布$B$，不随训练更新
- **Factorized Dense**：通过低秩分解降低**Dense**参数量。首先通过神经网络生成矩阵$B_1 \in \Bbb{R}^{n \times k_1}$,$B_2 \in \Bbb{R}^{n \times k_2}$($k_1k_2=n$)，将$B_1$重复$k_2$次,$B_2$重复$k_1$次得到$\tilde{B}_1 \in \Bbb{R}^{n \times n}$,$\tilde{B}_2 \in \Bbb{R}^{n \times n}$，最后逐元素相乘$B=\tilde{B}_1 \otimes \tilde{B}_2$
- **Factorized Random**：通过低秩分解降低**Random**参数量。首先通随机生成矩阵$R_1 \in \Bbb{R}^{n \times k}$,$R_2 \in \Bbb{R}^{n \times k}$，再进行矩阵乘法$B=R_1R_2^T$

上述五种方法的对比如下：

![](https://pic.downk.cc/item/5f0d9cdd14195aa594e13f60.jpg)

作者在机器翻译任务上进行试验，证明在该任务上合成注意力机制可以取代传统的自注意力机制，并且达到类似的效果。将两者结合使用时效果会更好：

![](https://pic.downk.cc/item/5f0d9de114195aa594e18460.jpg)

作者在自动摘要和对话生成任务上进行试验，在自动摘要任务上标准注意力效果比较好，但是对话生成任务上标准注意力是最差的。这说明不同的注意力各有优势。

![](https://pic.imgdb.cn/item/60ed3dc35132923bf8320211.jpg)

作者还进行了预训练+微调的实验。微调后相比标准自注意力，**Dense**和**Random**效果差一些，这表明它们也许在单一任务上表现不错，但迁移能力比较弱。但是**Dense**和**Random**带来的计算效率是显著提升的。

![](https://pic.imgdb.cn/item/60ed3e5d5132923bf8376eb2.jpg)
