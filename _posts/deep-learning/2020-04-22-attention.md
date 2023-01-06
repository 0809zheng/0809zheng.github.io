---
layout: post
title: '序列到序列模型中的注意力机制(Attention Mechanism)'
date: 2020-04-22
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e9fbd69c2a9a83be53b80dd.jpg'
tags: 深度学习
---

> Attention Mechanism in Seq2Seq Models.

**注意力机制（Attention Mechanism）**最初用于[神经机器翻译](https://arxiv.org/abs/1409.0473)任务中记忆比较长的输入序列。

在[**Seq2Seq**模型](https://0809zheng.github.io/2020/04/21/sequence-2-sequence.html)中，将输入文本序列通过编码器转换为一个**上下文向量**$c$，再喂入解码器:

![](https://pic.downk.cc/item/5e9ed899c2a9a83be5966fe5.jpg)

在比较简单的任务中，比如文本分类，只需要编码一些对分类有用的信息，因此用一个上下文向量$c$来表示文本语义是可行的。

但是在复杂的任务中，比如阅读理解，给定的背景文档一般比较长，如果用循环神经网络来将其转换为上下文向量$c$，则该编码向量很难反映出输入文本的所有语义。

注意力机制并不是用编码器的最后一个隐状态$h_T$作为固定的上下文向量$c$；而是在解码器的每一步中，通过输入序列的所有隐状态$h_{1:T}$构造当前步的上下文向量$c$。

![](https://pic.imgdb.cn/item/63b5886fbe43e0d30e75c044.jpg)

注意力机制的实现过程：

**1**. 把解码器上一步的隐状态$s_{t-1}$作为查询向量，在输入序列的所有编码器隐状态$h_{1:T}$上计算注意力分布 $(α_1,...,α_t,...,α_T)$：

$$ α_t = \text{softmax}(\text{score}(s_{t-1},h_t)) = \frac{\exp(\text{score}(s_{t-1},h_t))}{\sum_{t=1}^{T} {\exp(\text{score}(s_{t-1},h_t))}} $$

**2**. 根据注意力分布对输入序列的编码器隐状态$h_{1:T}$进行加权平均，作为当前步的上下文向量：

$$ c = \sum_{t=1}^{T} {α_th_t} $$

其中$\text{score}(s,h)$是**注意力得分函数**，也称为**相似得分函数（alignment score function）**，常用的计算方式包括：


| 相似得分函数 | 表达式 |  说明 |
| :---: | :---:  | :---:  |
| [加性 Additive](https://arxiv.org/abs/1409.0473)  | $v^T \tanh(Ws+Uh)$ | $v$、$W$、$U$是可学习参数 |
| [点积 Dot-Product](https://arxiv.org/abs/1508.04025)  | $s^Th$ | 使用矩阵乘法提高计算效率 |
| [缩放点积 Scaled Dot-Product](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)  | $\frac{s^Th}{\sqrt{n}}$ | $n$是隐状态维度，通过缩放防止输入过大导致**softmax**函数具有极小的梯度 |
| [双线性 General](https://arxiv.org/abs/1508.04025)  | $s^TWh$ | $W$是可学习参数，引入了非对称性 |
| [基于位置 Location-based](https://arxiv.org/abs/1508.04025)  | $Ws$ | 简化了**softmax**对准，使其仅取决于目标位置 |
| [基于上下文 Context-based](https://arxiv.org/abs/1410.5401)  | $\cos (s,h)$ | 使用余弦相似度函数 |

### ⚪ 软性/全局注意力和硬性/局部注意力

上述这种同时考虑输入序列的所有隐状态的注意力机制被称为**软性注意力机制（Soft Attention Mechanism）**,也叫**全局(global)**注意力。这种注意力的优点是平滑且可微；缺点是当输入序列长度很大时计算成本较高。

与之对应的，考虑输入序列的一部分隐状态的注意力机制被称为**局部(local)**注意力，这种注意力在推断时需要更少的计算量。

![](https://pic.imgdb.cn/item/63b6b536be43e0d30e311774.jpg)

特别地，只考虑输入序列的某一个隐状态的注意力机制被称为**硬性注意力（Hard Attention）**：
$$ c = x_{\hat{n}}, \quad \hat{n} = \mathop{\arg\max}_{n}α_n $$

硬性注意力的缺点是最大采样不能使用反向传播，需要用方差缩减(**variance reduction**)或强化学习进行训练。

### ⚪ 注意力机制的正则化

当同时使用多个输入文本时，每一个文本样本的注意力得分总和应该相近，因此引入正则化：

$$ \sum_{i}^{} {(τ-\sum_{t}^{} {\text{score}(s_t,h)})} $$

其中$τ$是给定的常数。
