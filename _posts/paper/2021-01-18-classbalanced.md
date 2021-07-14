---
layout: post
title: 'Class-Balanced Loss Based on Effective Number of Samples'
date: 2021-01-18
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60ee896a5132923bf8057881.jpg'
tags: 论文阅读
---

> Class-balanced Loss：基于有效样本数的类别平衡损失.

- paper：[Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)
- code：[github](https://github.com/vandit15/Class-balanced-loss-pytorch)

**Class-Balanced Loss**可以用于改进任何传统的损失函数，从而提高在类别不平衡数据集上的表现。通过数据增强等方法，每一个样本可以产生许多与自身相似的样本，这些样本分布在空间中的某个邻域内，即占据一定的空间“体积”。**Class-Balanced Loss**关注每一类样本所能覆盖的空间体积，而不是单纯的关注每一类样本的数量，并引入**有效样本数量(effective number of samples)**的概念。

将数据采样看作一个**随机覆盖(random covering)**问题，则某一类别的$n$个样本对应的有效样本数量为$E_n$，则其计算表达式为：

$$ E_n=\frac{1-\beta^n}{1-\beta}, \quad \beta = \frac{N-1}{N} $$

其中$N$表示该类别所有样本占有的空间体积。下面使用数学归纳法证明该结论。

当$n=1$时$E_n=1$成立，假设我们已经采样了$n-1$个样本，准备采样第$n$个样本。该类别所有样本占有的空间体积为$N$，已经采集样本的有效样本数量为$E_{n-1}=\frac{1-\beta^{n-1}}{1-\beta}$，则新的样本有$p=\frac{E_{n-1}}{N}$的概率落入已采集样本内；则采集新样本后的有效样本数量的数学期望计算为：

$$ E_n = pE_{n-1}+(1-p)(E_{n-1}+1) = E_{n-1}+1-p \\ =E_{n-1}+1-\frac{E_{n-1}}{N} = 1+\frac{N-1}{N}E_{n-1} \\ = 1+\frac{N-1}{N}\frac{1-\beta^{n-1}}{1-\beta} = 1+\beta\frac{1-\beta^{n-1}}{1-\beta} = \frac{1-\beta^{n}}{1-\beta} $$

![](https://pic.imgdb.cn/item/60a5be226ae4f77d3502dee4.jpg)

由此使用不同类别的有效样本数量$E_n$(而不是样本数量$n$)对不同类别的损失函数进行加权：

$$ \mathcal{L}_{\text{CB}}(\hat{y},y) = \frac{1}{E_{n_y}} \mathcal{L}(\hat{y},y) = \frac{1-\beta}{1-\beta^{n_y}} \mathcal{L}(\hat{y},y) $$

- 当样本空间大小$n=1$时，$\beta = 0$，意味着每个类别只需要$1$个样本便可以完全表示该类别的分布，增加样本数量对应的边际效应很强(增加样本得到的收益越来越少)，对应的策略为**no weighting**(即下图中蓝色曲线)；
- 当样本空间大小$n→∞$时，$\beta = 1$，意味着每个类别无法通过有限的样本量表示该类别的分布，增加样本数量对应的边际效应不存在(增加样本得到的收益恒定为$1$)，即每个类别完全由其含有的样本数量决定，需要按照样本数量加权，对应的策略为**inverse class frequency weighting**(即下图中紫色曲线)；
- 当$0 < \beta < 1$时，策略为上述两种方法的权衡。

![](https://pic.imgdb.cn/item/60a5bc996ae4f77d35f7d597.jpg)

