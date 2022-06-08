---
layout: post
title: 'Disentangling by Factorising'
date: 2022-04-15
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/628dc6be09475431292effbe.jpg'
tags: 论文阅读
---

> FactorVAE：通过分解特征表示的分布进行解耦.

- paper：[Disentangling by Factorising](https://arxiv.org/abs/1802.05983)


# 1. 分解ELBO中的KL散度项

**VAE**优化**对数似然的变分下界**:

$$ \begin{aligned} \log p(x)  &= \log \Bbb{E}_{z \text{~} q(z|x)}[\frac{p(x,z)}{q(z|x)}] \geq \Bbb{E}_{z \text{~} q(z|x)}[\log \frac{p(x,z)}{q(z|x)}] \\ \text{ELBO} &= - KL[q(z|x)||p(z)]+\mathbb{E}_{z \text{~} q(z|x)} [\log p(x | z)] \end{aligned} $$

作者对**ELBO**中的**KL**散度进行分解。首先对每个训练样本指定唯一的索引$n$，并且定义一个在$[1,N]$上均匀的随机变量$p(n)$与训练样本相关联，表示每个样本被选择的概率相同。分解过程如下：

$$ \begin{aligned} KL[q(z|x)||p(z)] &= \Bbb{E}_{q(z|x)}[\log \frac{q(z|x)}{p(z)}] = \Bbb{E}_{p(x)} [\Bbb{E}_{q(z|x)}[\log \frac{q(z|x)}{p(z)}]] \\ &= \sum_{x}p(x)\sum_{z} q(z|x) \log \frac{q(z|x)}{p(z)} = \sum_{x}p(x)\sum_{z} \frac{q(z,x)}{p(x)} \log \frac{q(z|x)}{q(z)}\frac{q(z)}{p(z)} \\ &= \sum_{x}\sum_{z} q(z,x) \log \frac{q(z|x)}{q(z)} + \sum_{x}\sum_{z} q(z,x) \log \frac{q(z)}{p(z)} \\ &= \sum_{x}\sum_{z} q(z,x) \log \frac{q(z,x)}{q(z)p(x)} + \sum_{z} q(z) \log \frac{q(z)}{p(z)} \\ &= I(x;z) + KL(q(z)||p(z)) \end{aligned} $$

分解式的第一项$I(x;z)$表示随机变量$x$和$z$的互信息，即知道随机变量$x$的信息后，随机变量$z$的不确定性的减少量。对该项进行惩罚将会导致随机变量$z$中包含$x$的信息减少，通过$z$重构出$x$的难度增大，即降低了模型的重构能力。

分解式的第二项$KL(q(z)\|\|p(z)$是隐变量的**KL**散度，迫使隐变量接近标准正态分布。对该项进行惩罚能够提高模型的解耦能力。

在[<font color=Blue>β-VAE</font>](https://0809zheng.github.io/2020/12/02/bvae.html)等模型中，加重对**ELBO**中的**KL**散度的惩罚会导致模型的解耦效果好，但是重构效果较差，作者设计了**FactorVAE**解决这个问题。

# 2. FactorVAE

**FactorVAE**的基本思路不是增加**ELBO**中的**KL**散度的权重，而是在原始**ELBO**后增加一个类似[<font color=Blue>β-TCVAE</font>](https://0809zheng.github.io/2022/04/05/btcvae.html)中的**全相关(Total Correlation)**项$KL(q(z)\|\|\prod_{j}q(z_j))$：

$$ \mathbb{E}_{z \text{~} q(z|x)} [-\log p(x|z)]+KL[q(z\|x)||p(z)] + \gamma \cdot KL(q(z)||\prod_{j}q(z_j)) $$

![](https://pic.imgdb.cn/item/628dd2210947543129410488.jpg)

在实际实现$KL(q(z)\|\|\prod_{j}q(z_j))$时，采用**density ratio trick**技巧，将**KL**散度的计算转化成交叉熵损失。对采样的隐变量$z$随机交换特征维度构造$z'$，额外引入一个判别器$D$用于区分$z$和$z'$：

$$ KL(q(z)||\prod_{j}q(z_j)) = \Bbb{E}_{q(z)}[\log \frac{q(z)}{\prod_{j}q(z_j)}] ≈ \Bbb{E}_{q(z)}[\log \frac{D(z)}{1-D(z)}] $$

![](https://pic.imgdb.cn/item/628dd62d0947543129471cf9.jpg)

# 3. 解耦能力的评估

作者改进了[<font color=Blue>β-VAE</font>](https://0809zheng.github.io/2020/12/02/bvae.html)中的解耦能力评估方法，从而不需要引入有参数的线性分类器，而是通过无参数的投票器实现的：
1. 随机选择一个解耦因子$f_k$（如尺寸）；
2. 采样$L$张图像$x^{(l)}$，将它们的因子$f_k$设置为固定值，其他因子随机；
3. 使用编码器$q(z\|x)$构造图像的隐变量$z^{(l)}$并进行归一化；
4. 计算所有归一化图像隐变量$z^{(l)}$的方差$Var[z^{(l)}/s]$；
5. 选择方差最小的维度$d=\mathop{\arg \max}_{d}Var[z^{(l)}_d/s_d]$；
6. 采用多数投票$d=f_k$作为最终的解耦评估得分。

![](https://pic.imgdb.cn/item/628dd4ac094754312944d649.jpg)