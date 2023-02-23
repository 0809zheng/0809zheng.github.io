---
layout: post
title: 'On the Relationship between Self-Attention and Convolutional Layers'
date: 2021-01-04
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ff2715e3ffa7d37b38d69a9.jpg'
tags: 论文阅读
---

> 理解自注意力和卷积层的关系.

- paper：On the Relationship between Self-Attention and Convolutional Layers
- arXiv：[link](https://arxiv.org/abs/1911.03584)

# 1. 自注意力机制
假设$X \in \Bbb{R}^{T \times D_{in}}$表示包含$T$个维度为$D_{in}$的**token**的数据矩阵，其物理意义是长度为$T$的离散序列，每一个**token**为该序列的一个元素（如**NLP**中句子中的一个单词，**CV**中图像的一部分像素）。自注意力层将其映射为维度为$D_{out}$的输出：

$$ Self \text{-} Attention(X) = softmax(A)XW_{val} = softmax(XW_{qry}W_{key}^TX^T)XW_{val} $$

其中$softmax(A)$表示**注意力概率(attention probability)**，是由**注意力得分(attention score)** $A$计算得到的。该层通过**query**矩阵$W_{qry} \in \Bbb{R}^{D_{in} \times D_{out}}$、**key**矩阵$W_{key} \in \Bbb{R}^{D_{in} \times D_{out}}$和**value**矩阵$W_{val} \in \Bbb{R}^{D_{in} \times D_{out}}$参数化。

上式的计算是无序的，即将$T$个**token**打乱顺序后不会影响输出结果。因此引入**位置编码(position encoding)** $P$学习每个**token**在序列中的位置，将其应用到注意力得分的计算中：

$$ A = (X+P)W_{qry}W_{key}^T(X+P)^T $$

其中$P \in \Bbb{R}^{T \times D_{in}}$代表位置编码向量，可以由任意函数表示。

实践发现为自注意力机制引入**multi head**能够提高表现。具体地，使用$N_h$个独立的自注意力模块得到$D_{h} = \frac{D_{out}}{N_h}$维中间结果，将其连接后映射到$D_{out}$维输出：

$$ MHSA(X) = concat[Self \text{-} Attention_h(X)]W_{out} + b_{out} $$

上式引入两个新的参数：$W_{out} \in \Bbb{R}^{N_hD_{h} \times D_{out}}$和$b_{out} \in \Bbb{R}^{D_{out}}$。

# 2. 图像的注意力
给定图像张量$X \in \Bbb{R}^{W \times H \times D_{in}}$，卷积层输出的$(i,j)$位置可以表示为：

$$ Conv(X)_{i,j,:} = \sum_{(\delta_1,\delta_2 \in \Delta_K)}^{} X_{i+\delta_1,j+\delta_2,:} W_{\delta_1,\delta_2,:,:} + b $$

其中$W \in \Bbb{R}^{K \times K \times D_{in} \times D_{out}}$代表权重张量，$b \in \Bbb{R}^{D_{out}}$代表偏置向量。$\Delta_K$代表$K \times K$的卷积核可取的所有(相对中心)位置集合。

将图像的**pixel**看作**token**，将图像按空间$W \times H$展开成$1D$张量，则像素$p$的输出计算为：

$$ Self \text{-} Attention(X)_p = \sum_{k}^{} softmax(A_{p,:})_kX_{k,:}W_{val} $$

位置编码有两种形式，**绝对(absolute)**位置编码和**相对(relative)**位置编码。

绝对位置编码对每个像素$p$附加一个固定或可学习的向量$P_{p,:}$，则像素$q$和像素$k$的注意力得分计算为：

$$ A_{q,k}^{abs} = (X_{q,:}+P_{q,:})W_{qry}W_{key}^T(X_{k,:}+P_{k,:})^T \\ = X_{q,:}W_{qry}W_{key}^TX_{k,:}^T + X_{q,:}W_{qry}W_{key}^TP_{k,:}^T + P_{q,:}W_{qry}W_{key}^TX_{k,:}^T + P_{q,:}W_{qry}W_{key}^TP_{k,:}^T $$

相对位置编码只考虑**query**像素和**key**像素的位置差异，而不是表示它们的绝对位置。此时像素$q$和像素$k$的注意力得分由它们的相对位置$\delta = k - q$决定：

$$ A_{q,k}^{rel} = X_{q,:}^TW_{qry}^TW_{key}X_{k,:} + X_{q,:}^TW_{qry}^T \hat{W}_{key}r_{\delta} + u^TW_{key}X_{k,:} + v^T \hat{W}_{key}r_{\delta} $$

其中可学习向量$u$和$v$对每个**head**是独立的，相对位置编码$r_{\delta} \in \Bbb{R}^{D_p}$被所有层和所有**head**共享。**key**的权重也被划分为两部分：由输入决定的$W_{key}$和由像素相对位置决定的$\hat{W}_{key}$。

# 3. 将自注意力表示为卷积层
- **定理**：给定包含$N_h$个维度为$D_{h}$的**head**的自注意力层，其输出维度是$D_{out}$，相对位置编码的维度$D_p ≥ 3$。则其可以表示任意卷积核大小为$\sqrt{N_h} \times \sqrt{N_h}$、输出通道维度为$min(D_h,D_{out})$的卷积层。

证明如下：

将**multi head**自注意力计算重写为：

$$ MHSA(X) = concat[Self \text{-} Attention_h(X)]W_{out} + b_{out} \\ = concat[softmax(A^{(h)})XW_{val}^{(h)}]W_{out} + b_{out} \\ = \sum_{h \in [N_h]}^{} softmax(A^{(h)})XW_{val}^{(h)}W_{out}[(h-1)D_h+1:h-D_h+1] + b_{out} \\ := \sum_{h \in [N_h]}^{} softmax(A^{(h)})XW^{(h)} + b_{out} $$

则其每一个输出像素$q$表示为：

$$ MHSA(X)_{q,:} = \sum_{h \in [N_h]}^{} (\sum_{k}^{} softmax(A_{q,:}^{(h)})_kX_{k,:})W^{(h)} + b_{out} $$

对于**head** $h$，假设每个像素$q$只关注像素$k$(相对位置为$q-k$)，注意力得分简化为：

$$ softmax(A_{q,:}^{(h)})_k = \begin{cases} 1 \quad \text{if } f(h) = q-k \\ 0 \quad \text{otherwise} \end{cases} $$

则自注意力计算简化为：

$$ MHSA(X)_{q,:} = \sum_{h \in [N_h]}^{} X_{q-f(h),:}W^{(h)} + b_{out} $$

对比卷积层的计算：

$$ Conv(X)_{i,j,:} = \sum_{(\delta_1,\delta_2 \in \Delta_K)}^{} X_{i+\delta_1,j+\delta_2,:} W_{\delta_1,\delta_2,:,:} + b $$

可以得出，当每个**head**的注意力计算只关注某一具体位置的像素时，$N_h$个**head**的计算与$\sqrt{N_h} \times \sqrt{N_h}$的卷积核计算是等价的。

![](https://pic.downk.cc/item/5ff52ab03ffa7d37b3642d3f.jpg)

# 4. 可视化自注意力head
作者可视化$6$层的$9$个**head**，具体地展示了中心某像素位置的注意力图，通过可视化发现不同层、不同**head**学习到类似于卷积层的特征模式。

![](https://pic.downk.cc/item/5ff52ae33ffa7d37b3645390.jpg)