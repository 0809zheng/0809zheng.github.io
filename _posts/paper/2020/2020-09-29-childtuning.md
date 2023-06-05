---
layout: post
title: 'Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning'
date: 2020-09-29
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/647ae6d4f024cca173cba164.jpg'
tags: 论文阅读
---

> 在大型语言模型中培养孩子：面向有效和泛化的微调.

- paper：[Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning](https://arxiv.org/abs/2109.05687)

本文作者设计了**ChildTuning**方法，来提高预训练模型在微调时的效果，该方法每次从预训练模型中选择一个子网络进行优化，缓解优化整个模型所带来的过拟合风险。其中，在子网络的选择上，又分为两种方式：**ChildTuning-D**和**ChildTuning-F**。

![](https://pic.imgdb.cn/item/647af86ef024cca173e0f099.jpg)

# 1. ChildTuning-D

**ChildTuning-D（Task-Dependent）**是任务相关的选择方式，它需要下游任务的训练数据来参与计算。具体来说，假设训练数据为$(x_1,y_1),...,(x_n,y_n)$，模型建模为$p(y\|x;\theta)$。记$\theta_i$是模型的第$i$个参数，计算如下形式的**Fisher**信息作为该参数的重要性：

$$ F_i = \frac{1}{n} \sum_{j=1}^n \left( \frac{\partial \log p(y_j|x_j;\theta)}{\partial \theta_i} \right)^2 $$

计算出重要性指标后，按照重要性对每个参数进行排序，然后选择最重要的**top**-$p$个参数，在模型更新过程中只优化这些参数。由于优化的总参数减少了，所以过拟合的风险也降低了。

在实际实现时，可能一个参数矩阵里面只有一部分参数被选中，所以要通过构建对应的**0/1**矩阵$M$来将对应的梯度**mask**掉，即:

$$ g \leftarrow \frac{g \otimes M}{p} $$

其中$p$是被选中更新的参数占比，除以$p$是保持梯度更新的量级不变。虽然**ChildTuning-D**理论上更新的参数减少了，但它不能节约计算量。

# 2. ChildTuning-F

**ChildTuning-F（Task-Free）**是任务无关的选择方式。**ChildTuning-D**是根据任务数据来构建了固定的**0/1**矩阵$M$，然后将梯度进行**mask**并除以$p$；而**ChildTuning-F**希望与任务无关，因此每步更新时随机构建一个**0/1**矩阵$M$，其中$1$的比例为$p$，然后将梯度修改为:

$$ g \leftarrow \frac{g \otimes M}{p} $$

值得一提的是，即使某个参数当前的梯度为$0$，也不代表该参数当前的更新量为$0$。这是因为通常使用带有动量的优化器，对于此类优化器，更新量是正比于动量，而动量是历史梯度滑动平均计算的，所以如果该参数的历史梯度不为$0$，那么即便当前梯度为$0$，动量依然很可能不会为$0$，所以更新量也不为$0$。

作者针对**ChildTuning-F**给出一个基于**SGD**的理解，指出其能扩大更新过程中的方差，从而有助于模型逃离较差的局部最优点。

**SGD**每步所计算的梯度有一定的随机性，假设它服从均值为$μ$、方差为$σ^2$的高斯分布；对于**ChildTuning-F**来说，引入一个随机变量$ε$
，有$p$的概率为$1$，$1−p$的概率为$0$。那么有：

$$
\begin{aligned}
\mathbb{E} \left[\frac{g\epsilon}{p}\right] & = \frac{\mathbb{E}[g]\mathbb{E}[\epsilon]}{p} = μ \\
Var\left[\frac{g\epsilon}{p}\right] & = \mathbb{E} \left[\left(\frac{g\epsilon}{p}\right)^2\right] - \left(\mathbb{E} \left[\frac{g\epsilon}{p}\right]\right)^2 \\
& = \frac{\mathbb{E}[g^2]\mathbb{E}[\epsilon^2]}{p^2} - μ^2 \\
& = \frac{(\sigma^2+\mu^2)p}{p^2} - μ^2 \\
& = \sigma^2 + \frac{1-p}{p}(\sigma^2+\mu^2) \geq \sigma^2 \\
\end{aligned}
$$

因此**ChildTuning-F**能保持梯度的均值不变，但能扩大梯度的方差；而在**SGD**中，更新量正比于梯度，因此扩大了更新量的方差有助于模型达到更好的收敛结果。

