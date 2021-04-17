---
layout: post
title: '高斯混合模型'
date: 2020-05-29
author: 郑之杰
cover: 'https://img.imgdb.cn/item/60763e328322e6675c8cb89f.jpg'
tags: 机器学习
---

> Gaussian Mixture Model.

**高斯混合模型(Gaussian Mixture Model,GMM)**是一类概率生成模型，表示为$K$个高斯分布的加权平均。其**隐变量(latent variable)**是服从多项分布的离散型随机变量，表示属于每个高斯分布的概率，记作$z$，表示如下：

$$ P(z=k) = p^k, \quad k=1,2,...,K. \quad \sum_{k=1}^{K}p^k = 1 $$

记**观测变量(observable variable)**为$x$，则高斯混合模型可表示为：

$$ P(x) = \sum_{z}^{} P(x,z) = \sum_{z}^{} P(z) P(x|z) \\ = \sum_{k=1}^{K} P(z=k) P(x|z=k) \\ = \sum_{k=1}^{K} p^k \mathcal{N}(x; \mu^k,\sigma^k) $$

假设观测数据为$$\{x_1,x_2,...,x_n\}$$，高斯混合模型的模型参数为$$\theta = \{ p^1,p^2,...,p^K, \mu^1,\mu^2,...,\mu^K,\sigma^1,\sigma^2,...,\sigma^K\}$$。

尝试直接用极大似然估计方法求解该问题。求解如下：

$$ \hat{\theta} = \mathop{\arg \max}_{\theta} logp(x;\theta) = \mathop{\arg \max}_{\theta} log \prod_{i=1}^{n} P(x_i;\theta) = \mathop{\arg \max}_{\theta} \sum_{i=1}^{n} log P(x_i;\theta) \\ \mathop{\arg \max}_{\theta} \sum_{i=1}^{n} log \sum_{k=1}^{K} P^k \mathcal{N}(x_i; \mu^k,\sigma^k) $$

上式在对数函数中存在求和项，直接求解是**intractable**的。采用[期望最大算法](https://0809zheng.github.io/2020/03/26/expectation-maximization.html)求解该问题。

## E-step
**EM**算法的**E-step**计算如下：

$$ \text{E-step：} P(z|x ; \theta^{(t)}) → \Bbb{E}_{P(z|x ; \theta^{(t)})}[logP(x,z ; \theta)] $$

期望计算如下：

$$ \Bbb{E}_{P(z|x ; \theta^{(t)})}[logP(x,z ; \theta)] = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz \\ = \sum_{z_1}^{}...\sum_{z_n}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) \\ = \sum_{z_1}^{}...\sum_{z_n}^{} \prod_{i=1}^{n} P(z_i|x_i ; \theta^{(t)}) log \prod_{j=1}^{n} P(x_j,z_j ; \theta) \\ = \sum_{z_1}^{}...\sum_{z_n}^{} \prod_{i=1}^{n} P(z_i|x_i ; \theta^{(t)}) \sum_{j=1}^{n} log P(x_j,z_j ; \theta) \\ = \sum_{z_1}^{}...\sum_{z_n}^{} \prod_{i=1}^{n} P(z_i|x_i ; \theta^{(t)}) log P(x_1,z_1 ; \theta) + ... \\ + \sum_{z_1}^{}...\sum_{z_n}^{} \prod_{i=1}^{n} P(z_i|x_i ; \theta^{(t)}) log P(x_n,z_n ; \theta) \\ = \sum_{z_1}^{} P(z_1|x_1 ; \theta^{(t)}) log P(x_1,z_1 ; \theta) \prod_{j=2}^{n} \sum_{z_j}^{} P(z_j|x_j ; \theta^{(t)}) + ... \\ + \sum_{z_n}^{} P(z_n|x_n ; \theta^{(t)}) log P(x_n,z_n ; \theta) \prod_{j=1}^{n-1} \sum_{z_j}^{} P(z_j|x_j ; \theta^{(t)}) \\ (\text{由于}\sum_{z_j}^{} P(z_j|x_j ; \theta^{(t)})=1) \\ = \sum_{i=1}^{n} \sum_{z_i}^{} P(z_i|x_i ; \theta^{(t)}) log P(x_i,z_i ; \theta) \\ = \sum_{i=1}^{n} \sum_{k=1}^{K} P(z_i=k|x_i ; \theta^{(t)}) log P(x_i,z_i=k ; \theta) $$

联合概率$P(x,z)$计算如下：

$$ P(x,z) = P(z)P(x|z) = P(z)\mathcal{N}(x; \mu_z,\sigma_z) $$

条件概率$P(z\|x)$计算如下：

$$ P(z|x) = \frac{P(x,z)}{P(x)} = \frac{P(z)\mathcal{N}(x; \mu_z,\sigma_z)}{\sum_{k=1}^{K} p^k\mathcal{N}(x; \mu^k,\sigma^k)} $$

记在当前模型参数$\theta^{(t)}$下第$i$个观测数据来自第$k$个分高斯模型的概率为$\gamma_{ik}$，称为分模型$k$对观测数据$x_i$的**响应度**。即：

$$ \gamma_{ik} = P(z_i=k|x_i ; \theta^{(t)}) = \frac{p^k\mathcal{N}(x_i; \mu^k,\sigma^k)}{\sum_{k=1}^{K} p^k\mathcal{N}(x_i; \mu^k,\sigma^k)} $$

则原期望计算公式可以表示为：

$$ \sum_{i=1}^{n} \sum_{k=1}^{K} P(z_i=k|x_i ; \theta^{(t)}) log P(x_i,z_i=k ; \theta) \\ = \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} log P(z_i=k)P(x_i|z_i=k) = \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} log p^k\mathcal{N}(x_i; \mu^k,\sigma^k) \\ = \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [log p^k+log\mathcal{N}(x_i; \mu^k,\sigma^k)] $$

## M-step
**EM**算法的**M-step**计算如下：

$$ \text{M-step：} \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \Bbb{E}_{P(z|x ; \theta^{(t)})}[logP(x,z ; \theta)] $$

### 计算${p^k}^{(t+1)}$

$$ {p^k}^{(t+1)} = \mathop{\arg \max}_{p^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [log p^k+log\mathcal{N}(x_i; \mu^k,\sigma^k)] \\ \text{s.t. } \sum_{k=1}^{K} p^k=1 $$

采用拉格朗日乘子法解决上述约束最优化问题。建立拉格朗日函数：

$$ \mathop{L}(p^k,\lambda) = \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [log p^k+log\mathcal{N}(x_i; \mu^k,\sigma^k)] + \lambda (\sum_{k=1}^{K} p^k-1) $$

令拉格朗日函数关于参数$p^k$的导数为零：

$$ \frac{\partial \mathop{L}(p^k,\lambda)}{\partial p^k} = \sum_{i=1}^{n} \gamma_{ik} \cdot \frac{1}{p^k} + \lambda = 0 $$

解得$p^k = -\frac{\sum_{i=1}^{n} \gamma_{ik}}{\lambda}$。又由$\sum_{k=1}^{K} p^k=1$，可得$\lambda = \sum_{i=1}^{n}\sum_{k=1}^{K} \gamma_{ik} = n $，因此参数$p^k$的估计值为：

$$ {p^k}^{(t+1)} = \frac{\sum_{i=1}^{n} \gamma_{ik}}{n} $$

### 计算${\mu^k}^{(t+1)}$

$$ {\mu^k}^{(t+1)} = \mathop{\arg \max}_{\mu^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [log p^k+log\mathcal{N}(x_i; \mu^k,\sigma^k)] \\  = \mathop{\arg \max}_{\mu^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} log\mathcal{N}(x_i; \mu^k,\sigma^k) \\ = \mathop{\arg \max}_{\mu^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} log \frac{1}{\sqrt{2\pi}\sigma^k} exp \{ -\frac{(x_i-\mu^k)^2}{ 2(\sigma^k)^2} \} \\ = \mathop{\arg \max}_{\mu^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [ log\frac{1}{\sqrt{2\pi}} - log\sigma^k -\frac{(x_i-\mu^k)^2}{ 2(\sigma^k)^2} ] $$

令上述表达式关于参数$\mu^k$的导数为零：

$$ \frac{\partial}{\partial \mu^k} \sum_{i=1}^{n} \sum_{k=1}^{K} - \gamma_{ik} \frac{(x_i-\mu^k)^2}{ 2(\sigma^k)^2} = \sum_{i=1}^{n} \gamma_{ik} \frac{2(x_i-\mu^k)}{ 2(\sigma^k)^2} = 0 $$

解得：

$$ {\mu^k}^{(t+1)} = \frac{\sum_{i=1}^{n} \gamma_{ik} x_i}{\sum_{i=1}^{n} \gamma_{ik}} $$

### 计算${\sigma^k}^{(t+1)}$

$$ {\sigma^k}^{(t+1)} = \mathop{\arg \max}_{\sigma^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [ log\frac{1}{\sqrt{2\pi}} - log\sigma^k -\frac{(x_i-\mu^k)^2}{ 2(\sigma^k)^2} ] $$

令上述表达式关于参数$\sigma^k$的导数为零：

$$ \frac{\partial}{\partial \sigma^k} \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} [ - log\sigma^k -\frac{(x_i-\mu^k)^2}{ 2(\sigma^k)^2} ] = \sum_{i=1}^{n} \gamma_{ik} [-\frac{1}{\sigma^k} + \cdot \frac{(x_i-\mu^k)^2}{ (\sigma^k)^3}] = 0 $$

解得：

$$ ({\sigma^k}^{(t+1)})^2 = \frac{\sum_{i=1}^{n} \gamma_{ik}(x_i-{\mu^k}^{(t+1)})^2}{\sum_{i=1}^{n} \gamma_{ik}} $$

## 算法总结
数据样本集为$$\{x_1,x_2,...,x_n\}$$，建立高斯混合模型：

$$ P(x) = \sum_{k=1}^{K} p^k \mathcal{N}(x; \mu^k,\sigma^k) $$

待求模型参数为$$\theta = \{ p^1,p^2,...,p^K, \mu^1,\mu^2,...,\mu^K,\sigma^1,\sigma^2,...,\sigma^K\}$$，随机取参数的初始值开始迭代。

根据当前模型参数，计算分模型$k$对观测数据$x_i$的响应度：

$$ \gamma_{ik} = P(z_i=k|x_i ; \theta^{(t)}) = \frac{p^k\mathcal{N}(x_i; \mu^k,\sigma^k)}{\sum_{k=1}^{K} p^k\mathcal{N}(x_i; \mu^k,\sigma^k)} $$

计算新一轮迭代的模型参数：

$$ {p^k}^{(t+1)} = \frac{\sum_{i=1}^{n} \gamma_{ik}}{n} $$

$$ {\mu^k}^{(t+1)} = \frac{\sum_{i=1}^{n} \gamma_{ik} x_i}{\sum_{i=1}^{n} \gamma_{ik}} $$

$$ ({\sigma^k}^{(t+1)})^2 = \frac{\sum_{i=1}^{n} \gamma_{ik}(x_i-{\mu^k}^{(t+1)})^2}{\sum_{i=1}^{n} \gamma_{ik}} $$

重复迭代直至收敛。

