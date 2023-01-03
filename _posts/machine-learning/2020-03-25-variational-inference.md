---
layout: post
title: '变分推断(Variational Inference)'
date: 2020-03-25
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63b40d63be43e0d30e2f4cdf.jpg'
tags: 机器学习
---

> Variational Inference.

**隐变量模型(latent variable model)**是一类强大的生成模型，其主要思想是在已知的**观测数据(observed data)**$x_i$后存在未观测到的**隐变量(latent variable)**$z_i$，其图模型如下：

![](http://adamlineberry.ai/images/vae/graphical-model.png)

隐变量模型的概率分布表示如下：

$$ p_{\theta}(x,z) = p_{\theta}(x | z)p_{\theta}(z) $$

通常求解概率模型$p_{\theta}(x)$的参数$\theta$采用极大似然估计，即在训练时最大化观测数据的**边际似然(marginal likelihood)**，也等价于最大化**边际对数似然(marginal log likelihood)**：

$$ \theta = \mathop{\arg \max}_{\theta} p_{\theta}(x) = \mathop{\arg \max}_{\theta} \log p_{\theta}(x) $$

记训练集$X$包含$N$个数据点$$\{x_1,x_2,...,x_N\}$$，则隐变量模型的边际对数似然表示为：

$$ \begin{aligned} \log(p_{\theta}(X))& = \log \prod_{i=1}^{N} {p_{\theta}(x_i)} = \sum_{i=1}^{N} {\log p_{\theta}(x_i)} \\ &= \sum_{i=1}^{N} {\log \sum_{z} {p_{\theta}(x_i,z_i) }} \end{aligned} $$

理论上需要将上式最大化，但是其积分通常是**不可解(intractable)**的，尤其是隐变量$z_i$维度较高时，求和对应为多重求和(连续形式则为多重积分)。

当隐变量$z$的**后验分布(posterior)** $p_{\theta}(z \| x) = \frac{p_{\theta}(x \| z)p_{\theta}(z)}{p_{\theta}(x)}$可解时，可通过[**期望最大算法**](https://0809zheng.github.io/2020/03/26/expectation-maximization.html)对对数似然进行交替求解。

当隐变量$z$的后验分布$p_{\theta}(z \| x)$不可解时，通常有两种方法解决这个问题：**蒙特卡罗方法**和**变分推断**。

**变分推断**是指寻找对数似然的一个**变分下界(variational lower bound)**，也称为**置信下界(evidence lower bound, ELBO)**，从而用最大化**ELBO**代替最大化边际对数似然。

## ① 用KL散度推导ELBO
引入一个新的分布$q_{\phi}(z)$近似隐变量$z$的后验分布$p_{\theta}(z \| x)$，用[KL散度](https://0809zheng.github.io/2020/02/03/kld.html)衡量两者的相似程度：

$$ \begin{aligned} KL[q_{\phi}(z)||p_{\theta}(z | x)] &= -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(z | x)}{q_{\phi}(z)}} \\ &= -\sum_{z}^{} {q_{\phi}(z) \log (\frac{p_{\theta}(x,z)}{q_{\phi}(z)} \cdot \frac{1}{p_{\theta}(x)})} \\ &= -\sum_{z}^{} {q_{\phi}(z) (\log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} - \log p_{\theta}(x))} \\ &= -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \sum_{z}^{} {q_{\phi}(z) \log p_{\theta}(x)} \\ &= -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \log p_{\theta}(x) \cdot \sum_{z}^{} {q_{\phi}(z)} \\ &= -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \log p_{\theta}(x) \end{aligned} $$

将上式整理得：

$$ \log p_{\theta}(x) = KL[q_{\phi}(z)||p_{\theta}(z | x)] + \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} $$

上式左边即为边际对数似然$\log p_{\theta}(x)$，右边第一项为两个概率分布的**KL**散度，第二项被称作**变分下界(variational lower bound)**或**置信下界(evidence lower bound, ELBO)**，记作$\mathcal{L}$。

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} $$

注意到**KL**散度是非负的，所以$\mathcal{L}$是边际对数似然$\log p_{\theta}(x)$的一个下界：$\mathcal{L} ≤ \log p_{\theta}(x)$，因此可以通过最大化**ELBO** $\mathcal{L}$代替最大化边际对数似然。

## ② 用期望推导ELBO
引入一个新的分布$q_{\phi}(z)$作为隐变量$z$的后验分布$p_{\theta}(z \| x)$的近似。则边际对数似然$\log p_{\theta}(x)$可以表示成：

$$ \log p_{\theta}(x) = \log \frac{p_{\theta}(x,z)}{p_{\theta}(z | x)} = \log p_{\theta}(x,z) - \log p_{\theta}(z | x) \\ = \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} - \log \frac{p_{\theta}(z | x)}{q_{\phi}(z)} \\ = \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} + \log \frac{q_{\phi}(z)}{p_{\theta}(z | x)} $$

对上式两端求$q_{\phi}(z)$的期望，得：

$$ \begin{aligned} \text{左端} &= \sum_{z}^{} {q_{\phi}(z)\log p_{\theta}(x)} = \log p_{\theta}(x) \cdot \sum_{z}^{} {q_{\phi}(z)} = \log p_{\theta}(x) \\ \text{右端} &= \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \sum_{z}^{} {q_{\phi}(z) \log \frac{q_{\phi}(z)}{p_{\theta}(z | x)}} \\ &= \mathcal{L}  +  KL[q_{\phi}(z)||p_{\theta}(z | x)] \end{aligned} $$


上式左边即为边际对数似然$\log p_{\theta}(x)$，右边第一项被称作**变分下界(variational lower bound)**或**置信下界(evidence lower bound, ELBO)**，记作$\mathcal{L}$，第二项为两个概率分布的KL散度。

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} $$

注意到右边第二项的两个概率分布的KL散度是非负的，所以$\mathcal{L}$是边际对数似然$\log p_{\theta}(x)$的一个下界：$\mathcal{L} ≤ \log p_{\theta}(x)$，因此可以通过最大化**ELBO** $\mathcal{L}$代替最大化边际对数似然。

## ③ 用Jensen不等式推导ELBO
采用[Jensen不等式](https://0809zheng.github.io/2022/07/20/jenson.html)可以快速推导出**ELBO**的表达。同样引入一个新的分布$q_{\phi}(z)$作为隐变量$z$的后验分布$p_{\theta}(z \| x)$的近似，则边际对数似然$\log p_{\theta}(x)$可以表示成：

$$ \begin{aligned} \log p_{\theta}(x) &= \log (\sum_{z}^{} {p_{\theta}(x,z)}) \\ &= \log (\sum_{z}^{} {\frac{p_{\theta}(x,z)}{q_{\phi}(z)}q_{\phi}(z)}) \\& = \log \mathbb{E}_{z \text{~} q_{\phi}(z)} [\frac{p_{\theta}(x,z)}{q_{\phi}(z)}] \\ &≥ \mathbb{E}_{z \text{~} q_{\phi}(z)} [\log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}] \\& = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} \end{aligned} $$

上式被称作**变分下界(variational lower bound)**或**置信下界(evidence lower bound, ELBO)**，记作$\mathcal{L}$。

$$ \mathcal{L} = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} $$

$\mathcal{L}$是边际对数似然$\log p_{\theta}(x)$的一个下界：$\mathcal{L} ≤ \log p_{\theta}(x)$，因此可以通过最大化**ELBO** $\mathcal{L}$代替最大化边际对数似然。
