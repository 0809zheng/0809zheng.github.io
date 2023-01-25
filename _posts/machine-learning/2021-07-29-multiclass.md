---
layout: post
title: '多类别(Multiclass)与多标签(Multilabel)分类'
date: 2021-07-29
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63d095b4588a5d166cd6a692.jpg'
tags: 机器学习
---

> Multiclass and Multilabel Classification.

# 1. 多类别分类

**多类别分类(multiclass classification)**也称单标签分类，是指从$n$个候选类别中选出$1$个目标类别。

### ⚪ One-versus-All
**One-versus-All（OVA）**是指把多类别分类问题拆分成若干二分类问题，在每一次分类中使用所有类别的样本，选择其中某一类样本为正样本，其余类的样本为负样本。

![](https://pic.downk.cc/item/5ed11069c2a9a83be50c76d3.jpg)

- 优点：实现简单；
- 缺点：对于二分类器来说正负样本不平衡。

### ⚪ One-versus-One
**One-versus-One（OVO）**是指把多类别分类问题拆分成若干二分类问题，在每一次分类中使用其中两类样本，选择一类为正样本，另一类为负样本。

![](https://pic.downk.cc/item/5ed10fbdc2a9a83be50b55da.jpg)

- 优点：每一次二分类的样本基本上是平衡的；
- 缺点：需要训练的二分类器数量为组合数$C_n^2$，增加训练复杂度。

### ⚪ Softmax分类

在处理常规的多分类问题时，通过神经网络输出每个类的分数$s_1,s_2,...,s_n$，然后选取最大分数对应的类别作为目标类别，该过程可以被光滑化为[**softmax**函数](https://0809zheng.github.io/2021/11/16/mollifier.html#-textonehotargmaxxtextsoftmaxx_1x_2x_n)：

$$ p_i = \frac{e^{s_i}}{\sum_j e^{s_j}} $$

从而可以把多类别分类问题建模为一个多项(**multinomial**)分布，类别标签$y$用**one-hot**向量$(y_1,...,y_n)=(0,...,1,...,0)$表示，则概率表达式为：

$$ \Pi(p_1,p_2,...,p_n) = \prod_{i=1}^n p_i^{y_i} $$

损失函数构造为最小化概率的负对数似然：

$$ \begin{aligned} \mathcal{L}(x,y=y_t)&=-\log \prod_{i=1}^n p_i^{y_i} = -\sum_{i=1}^n y_i\log p_i \\ &= -\log p_t =  -\log \frac{e^{s_t}}{\sum_j e^{s_j}}  \end{aligned} $$

上式被称为**交叉熵(cross-entropy)**损失函数，根据[**max**函数的光滑化](https://0809zheng.github.io/2021/11/16/mollifier.html#-maxx_1x_2x_ntextlogsumexpx_1x_2x_n)，该函数的目标是使得目标类的得分$s_t$变为$s_1,s_2,...,s_n$中的最大值，即目标类得分大于每个非目标类的得分：

$$ -\log \frac{e^{s_t}}{\sum_j e^{s_j}} = \log \sum_j e^{s_j-s_t} = \log (1+ \sum_{j \neq t} e^{s_j-s_t}) ≈ \max \begin{pmatrix} s_0-s_t \\ s_1-s_t \\ \cdots \\ s_{t-1}-s_t \\ 0 \\ s_{t+1} - s_t \\ \cdots \\ s_n-s_t \end{pmatrix} $$

# 2. 多标签分类

**多标签分类(multilabel classification)**是指从$n$个候选类别中选出$k>1$个目标类别。

### ⚪ Sigmoid分类

可以把多标签分类问题拆分成$n$个二分类问题，对每个类别使用**Sigmoid**激活，然后构造二元交叉熵损失；则多标签分类的总损失为$n$个二分类的交叉熵之和：

$$ \sum_{i=1}^n -y_i \log p_i - (1-y_i) \log (1-p_i) $$

这种做法会面临着严重的类别不均衡问题，需要使用一些平衡策略，比如手动调整正负样本的权重等。训练完成之后，还需要根据验证集来进一步确定最优的分类阈值。

### ⚪ Softmax分类

多类别分类任务中的**Softmax**分类旨在使得目标类得分大于每个非目标类的得分，可以将其扩展到多标签分类任务中，即使得**每个目标类得分都不小于每个非目标类的得分**。

记$$\Omega_p,\Omega_n$$分别是样本的正负类别集合，则希望任意目标类的分数$s_i$都大于任意非目标类的分数$s_j$，并且目标类的分数$s_i$大于$s_0$，非目标类的分数$s_j$小于$s_0$，则仿照交叉熵损失构造一个损失函数：

$$ \begin{aligned} &\log(1+\sum_{i \in \Omega_p,j \in \Omega_n}e^{s_i-s_j}+\sum_{i \in \Omega_p}e^{s_i-s_0}+\sum_{j \in \Omega_n}e^{s_0-s_j}) \\ = &\log(e^{s_0}+\sum_{i \in \Omega_p}e^{s_i}) + \log(e^{-s_0}+\sum_{j \in \Omega_n}e^{-s_j}) \end{aligned} $$

不失一般性地假设$s_0=0$，则多标签分类的“**Softmax**+交叉熵”形式为：

$$ \log(1+\sum_{i \in \Omega_p}e^{s_i}) + \log(1+\sum_{j \in \Omega_n}e^{-s_j}) $$

```python
def multilabel_categorical_crossentropy(y_true, y_pred, num_classes):
    """
    y_true为整型，y_pred为网络输出logits（无激活函数）
    """
    # 生成one-hot标签
    labels = torch.FloatTensor(y_true.shape[0], num_classes).zero_()
    y_true = labels.scatter_(1, y_true.data, 1)

    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.stack([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.stack([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss  
```