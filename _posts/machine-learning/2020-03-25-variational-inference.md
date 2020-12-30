---
layout: post
title: '变分推断'
date: 2020-03-25
author: 郑之杰
cover: ''
tags: 机器学习
---

> Variational Inference.

**隐变量模型(latent variable model)**是一类强大的生成模型，其主要思想是在已知的**观测数据(observed data)**$x_i$后存在未观测到的**隐变量(latent variable)**$z_i$，其图模型如下：

![](http://adamlineberry.ai/images/vae/graphical-model.png)

隐变量模型的概率分布表示如下：

$$ p_{\theta}(x,z) = p_{\theta}(x | z)p_{\theta}(z) $$

求解模型参数$\theta$，即在训练时最大化观测数据的**边际似然(marginal likelihood)**，等价于最大化**边际对数似然(marginal log likelihood)**：

$$ \theta = \mathop{\arg \max}_{\theta} p_{\theta}(x) = \mathop{\arg \max}_{\theta} \log p_{\theta}(x) $$

记训练集$X$包含$N$个数据点$$\{x_1,x_2,...,x_N\}$$，其边际对数似然表示为：

$$ \log(p_{\theta}(X)) = \log \prod_{i=1}^{N} {p_{\theta}(x_i)} = \sum_{i=1}^{N} {\log p_{\theta}(x_i)} \\ = \sum_{i=1}^{N} {\log \int_{}^{} {p_{\theta}(x_i,z_i) dz}} \\ = \sum_{i=1}^{N} {\log \int_{}^{} {p_{\theta}(x_i | z_i)p_{\theta}(z_i) dz}} $$

理论上需要将上式最大化，但是其积分通常是**不可解(intractable)**的，尤其是隐变量$z_i$维度较高时，积分对应为多重积分。

为解决上述问题，首先考虑隐变量$z$的**后验分布(posterior)**：

$$ p_{\theta}(z | x) = \frac{p_{\theta}(x | z)p_{\theta}(z)}{p_{\theta}(x)} $$

由于观测变量的分布$p_{\theta}(x)$是未知的，隐变量$z$的后验分布$p_{\theta}(z \| x)$是不可解的，通常有两种方法解决这个问题：**蒙特卡罗方法**和**变分推断**。本节介绍变分推断。特别地，若$p_{\theta}(z \| x)$是可解的，可以使用**期望最大算法**。以下推导均假设随机变量为离散型。

## ① 用KL散度推导ELBO
由于隐变量$z$的后验分布$p_{\theta}(z \| x)$是不可解的，引入一个新的分布$q_{\phi}(z)$作为近似。用[KL散度](https://0809zheng.github.io/2020/02/03/kld.html)衡量两者的相似程度：

$$ KL[q_{\phi}(z)||p_{\theta}(z | x)] = -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(z | x)}{q_{\phi}(z)}} \\ = -\sum_{z}^{} {q_{\phi}(z) \log (\frac{p_{\theta}(x,z)}{q_{\phi}(z)} \cdot \frac{1}{p_{\theta}(x)})} \\ = -\sum_{z}^{} {q_{\phi}(z) (\log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} - \log p_{\theta}(x))} \\ = -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \sum_{z}^{} {q_{\phi}(z) \log p_{\theta}(x)} \\ = -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \log p_{\theta}(x) \cdot \sum_{z}^{} {q_{\phi}(z)} \\ = -\sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \log p_{\theta}(x) $$

将上式整理得：

$$ \log p_{\theta}(x) = KL[q_{\phi}(z)||p_{\theta}(z | x)] + \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} $$

上式左边即为边际对数似然$\log p_{\theta}(x)$，右边第一项为两个概率分布的KL散度，第二项被称作**变分下界(variational lower bound)**或**置信下界(evidence lower bound, ELBO)**，记作$\mathcal{L}$。

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} $$

注意到右边第一项的两个概率分布的KL散度是非负的，所以$\mathcal{L}$是边际对数似然$\log p_{\theta}(x)$的一个下界：$\mathcal{L} ≤ \log p_{\theta}(x)$。变分推断方法用最大化**ELBO**代替最大化边际对数似然。

## ② 用期望推导ELBO
与上一小节相同，引入一个新的分布$q_{\phi}(z)$作为隐变量$z$的后验分布$p_{\theta}(z \| x)$的近似。则边际对数似然$\log p_{\theta}(x)$可以表示成：

$$ \log p_{\theta}(x) = \log \frac{p_{\theta}(x,z)}{p_{\theta}(z | x)} = \log p_{\theta}(x,z) - \log p_{\theta}(z | x) \\ = \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} - \log \frac{p_{\theta}(z | x)}{q_{\phi}(z)} \\ = \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} + \log \frac{q_{\phi}(z)}{p_{\theta}(z | x)} $$

对上式两端求$q_{\phi}(z)$的期望，得：

$$ \sum_{z}^{} {q_{\phi}(z)\log p_{\theta}(x)} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \sum_{z}^{} {q_{\phi}(z) \log \frac{q_{\phi}(z)}{p_{\theta}(z | x)}} $$

$$ \log p_{\theta}(x) \cdot \sum_{z}^{} {q_{\phi}(z)} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \sum_{z}^{} {q_{\phi}(z) \log \frac{q_{\phi}(z)}{p_{\theta}(z | x)}} $$

整理得：

$$ \log p_{\theta}(x) = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} + \sum_{z}^{} {q_{\phi}(z) \log \frac{q_{\phi}(z)}{p_{\theta}(z | x)}} \\ = \mathcal{L}  +  KL[q_{\phi}(z)||p_{\theta}(z | x)] $$

上式左边即为边际对数似然$\log p_{\theta}(x)$，右边第一项被称作**变分下界(variational lower bound)**或**置信下界(evidence lower bound, ELBO)**，记作$\mathcal{L}$，第二项为两个概率分布的KL散度。

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} $$

注意到右边第二项的两个概率分布的KL散度是非负的，所以$\mathcal{L}$是边际对数似然$\log p_{\theta}(x)$的一个下界：$\mathcal{L} ≤ \log p_{\theta}(x)$。变分推断方法用最大化**ELBO**代替最大化边际对数似然。

## ③ 用Jensen不等式推导ELBO
采用[Jensen不等式]()可以快速推导出**ELBO**的表达。同样引入一个新的分布$q_{\phi}(z)$作为隐变量$z$的后验分布$p_{\theta}(z \| x)$的近似，则边际对数似然$\log p_{\theta}(x)$可以表示成：

$$ \log p_{\theta}(x) = \log (\sum_{z}^{} {p_{\theta}(x,z)}) \\ = \log (\sum_{z}^{} {\frac{p_{\theta}(x,z)}{q_{\phi}(z)}q_{\phi}(z)}) \\ = \log \mathbb{E}_{z \text{~} q_{\phi}(z)} [\frac{p_{\theta}(x,z)}{q_{\phi}(z)}] \\ ≥ \mathbb{E}_{z \text{~} q_{\phi}(z)} [\log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}] \\ = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} $$

上式被称作**变分下界(variational lower bound)**或**置信下界(evidence lower bound, ELBO)**，记作$\mathcal{L}$。

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} $$

$\mathcal{L}$是边际对数似然$\log p_{\theta}(x)$的一个下界：$\mathcal{L} ≤ \log p_{\theta}(x)$。变分推断方法用最大化**ELBO**代替最大化边际对数似然。
