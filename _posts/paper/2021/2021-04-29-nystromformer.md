---
layout: post
title: 'Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention'
date: 2021-04-29
author: 郑之杰
cover: 'https://img.imgdb.cn/item/608a52c6d1a9ae528f8e6415.jpg'
tags: 论文阅读
---

> Nyströmformer：使用Nyström方法近似自注意力运算.

- paper：Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention
- arXiv：[link](https://arxiv.org/abs/2102.03902)

# 1. Nyström Method
**Nyström**方法最初是用来解决如下特征函数问题的数值方式：

$$ \int_{a}^{b} W(x,y) \phi(y) dy = \lambda \phi(x) $$

注意到直接求解该积分方程是困难的。在积分区间$\[a,b\]$内选择一系列点$\xi_1,\xi_2,...,\xi_n$，做如下近似计算：

$$ \frac{(b-a)}{n} \sum_{j=1}^{n} W(x,\xi_j)\hat{\phi}(\xi_j) = \lambda \hat{\phi}(x) $$

整理可得$\phi(x)$的近似表达式：

$$ \hat{\phi}(x) = \frac{(b-a)}{\lambda n} \sum_{j=1}^{n} W(x,\xi_j)\hat{\phi}(\xi_j) $$

通过上式可以用有限个点($\xi_1,\xi_2,...,\xi_n$)近似计算任意点的特征函数值。**Nyström**方法的特点便是使用有限的特征表示所有特征。

# 2. Nyström Completion
矩阵的**Nyström**分解是指在矩阵$W = \begin{bmatrix} A & B \\ C & D \\ \end{bmatrix}$中随机地选择一些列向量$(A,C)^T$和行向量$(A,B)$，从而用这些矩阵近似地替代原矩阵。分解形式为：

$$ W≈ \begin{bmatrix} A & B \\ C & C A^{-1} B \\ \end{bmatrix} = \begin{bmatrix} A  \\ C  \\ \end{bmatrix}A^{-1}\begin{bmatrix} A & B  \end{bmatrix} $$

**Nyström**方法常被用在**矩阵补全(Matrix Completion)**，如用来近似替代**亲和矩阵(Affinity Matrix)**。
亲和矩阵用于衡量空间中任意两点的距离或者相似度，因此是一个对称矩阵。亲和矩阵$W$的一个划分如下：

$$ W = \begin{bmatrix} A & B \\ B^T & C \\ \end{bmatrix} $$

其中$A$表示随机选择的一些样本点之间的相似度矩阵，$C$表示其余未选择的样本点之间的相似度矩阵，$B$表示这两组样本点之间的相似度。通常选择$A$的尺寸小，$C$的尺寸比较大。由于矩阵$W$本身比较大，因此该方法通过储存矩阵$A$和$B$来近似替代矩阵$W$。

由于矩阵$A$是对称矩阵，可以将其对角化$A=U \Lambda U^T$。记$\overline{U}$为矩阵$W$的特征向量，则其表示为：

$$ \overline{U} = \begin{bmatrix} U \\ B^T U \Lambda^{-1} \\ \end{bmatrix} $$

则矩阵$W$可以近似表示为：

$$ \hat{W} = \overline{U} \Lambda \overline{U}^T = \begin{bmatrix} U \\ B^T U \Lambda^{-1} \\ \end{bmatrix} \Lambda \begin{bmatrix} U^T & \Lambda^{-1} U^T B \\ \end{bmatrix} \\ = \begin{bmatrix} U \Lambda U^T & B \\ B^T & B^T U \Lambda^{-1} U^T B \\ \end{bmatrix} = \begin{bmatrix} A & B \\ B^T & B^T A^{-1} B \\ \end{bmatrix} $$



# 3. Nyströmformer
作者将**Nyström**方法引入**Transformer**中的自注意力矩阵计算中。标准的自注意力矩阵计算如下：

$$ Q=XW_Q, K=XW_K, V=XW_V $$

$$ S = \text{softmax}(\frac{QK^T}{\sqrt{d_q}}) $$

注意到直接计算$S$矩阵的计算量是比较大的，其中矩阵乘法$QK^T$的计算复杂度为$O(n^2)$。作者使用**Nyström**方法对$QK^T$进行近似分解：

$$ \hat{S} = \text{softmax}(\frac{1}{\sqrt{d_q}}\begin{bmatrix} A & B \\ C & C A^{-1} B \\ \end{bmatrix} ) = \text{softmax}(\frac{1}{\sqrt{d_q}} \begin{bmatrix} A  \\ C  \\ \end{bmatrix}A^{-1}\begin{bmatrix} A & B  \end{bmatrix}) \\ = \text{softmax}(\frac{1}{\sqrt{d_q}} \begin{bmatrix} A  \\ C  \\ \end{bmatrix})\times\text{softmax}(\frac{1}{\sqrt{d_q}} A^{-1})\times\text{softmax}(\frac{1}{\sqrt{d_q}} \begin{bmatrix} A & B  \end{bmatrix}) \\ = \tilde{F}\times\tilde{A}\times\tilde{B} $$

因此**Nyström**方法将自注意力矩阵近似成三个矩阵相乘的形式：

![](https://img.imgdb.cn/item/608b73d7d1a9ae528f8a4e84.jpg)

其中矩阵$\begin{bmatrix} A  \\ C  \\ \end{bmatrix}$是从矩阵$QK^T$中随机选择的$k$列，在实践中采用**Segment-Means(sMEANS)**的方法，即先通过平均池化将矩阵$K$下采样为具有$k$列的矩阵$\tilde{K}$，再用矩阵$Q\tilde{K}^T$近似替代矩阵$\begin{bmatrix} A  \\ C  \\ \end{bmatrix}$，则有：

$$ \tilde{F} = \text{softmax}(\frac{1}{\sqrt{d_q}} \begin{bmatrix} A  \\ C  \\ \end{bmatrix}) = \text{softmax}(\frac{Q\tilde{K}^T}{\sqrt{d_q}} ) $$

矩阵$\begin{bmatrix} A & B  \end{bmatrix}$是从矩阵$QK^T$中随机选择的$k$行，采用**sMEANS**的方法，即先通过平均池化将矩阵$Q$下采样为具有$k$行的矩阵$\tilde{Q}$，再用矩阵$\tilde{Q}K^T$近似替代矩阵$\begin{bmatrix} A & B  \end{bmatrix}$，则有：

$$ \tilde{B} = \text{softmax}(\frac{1}{\sqrt{d_q}} \begin{bmatrix} A & B  \end{bmatrix}) = \text{softmax}(\frac{\tilde{Q}K^T}{\sqrt{d_q}} ) $$

矩阵$A$是从矩阵$QK^T$中随机选择的$k$行$k$列交汇处，采用**sMEANS**的方法，即根据平均池化获得的矩阵$\tilde{K}$和矩阵$\tilde{Q}$，用矩阵$\tilde{Q}\tilde{K}^T$近似替代矩阵$A$，则有：

$$ \tilde{A} = \text{softmax}(\frac{1}{\sqrt{d_q}} A^{-1}) = \text{softmax}(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_q}} )^{-1} $$

则使用**Nyström**方法近似自注意力矩阵的算法流程如下：

![](https://img.imgdb.cn/item/608b7410d1a9ae528f8c7654.jpg)

完整的**Nyström**方法进行自注意力运算的结构图如下：

![](https://img.imgdb.cn/item/608b7421d1a9ae528f8d21c2.jpg)

# 4. 实验分析
下图是遮挡语言模型**MLM**和句子顺序预测**SOP**这两个预训练任务上模型的表现。实验表明模型的表现和标准的自注意力机制表现相差不大，在**MLM**任务上甚至超过了标准的表现。

![](https://pic.imgdb.cn/item/611797ea5132923bf8fae87d.jpg)

模型计算复杂度下降带来的内存占用减少和推理时间加快也比较明显：

![](https://pic.imgdb.cn/item/611798b55132923bf8fe75f3.jpg)