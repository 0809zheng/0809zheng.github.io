---
layout: post
title: 'Restricted Boltzmann Machine：受限玻尔兹曼机'
date: 2020-04-15
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e9676a8c2a9a83be55aca1e.jpg'
tags: 机器学习
---

> Restricted Boltzmann Machine.

**本文目录**：
1. 模型介绍
2. 推断
3. 学习
4. 深度玻尔兹曼机


# 1. 模型介绍
全连接的[玻尔兹曼机](https://0809zheng.github.io/2020/04/14/boltzmann-machine.html)由于其复杂性，目前为止并没有被广泛使用。

**受限玻尔兹曼机(Restricted Boltzmann Machine，RBM)**的神经元也可以分为**可观测变量**$v$和**隐变量**$h$，下图给出了包含4个可观测变量和3个隐变量的受限玻尔兹曼机：

![](https://pic.downk.cc/item/5e967762c2a9a83be55b6723.jpg)

受限玻尔兹曼机的可观测变量和隐变量之间是全连接的，分别用**可观测层**和**隐藏层**表示；但是同一层内的节点没有连接。

一个受限玻尔兹曼机由$K_v$个可观测变量和$K_h$个隐变量组成，模型参数包括：
1. 可观测的随机向量$$v \in \Bbb{R}^{K_v}$$
2. 隐藏的随机向量$$h \in \Bbb{R}^{K_h}$$
3. 权重矩阵$$W \in \Bbb{R}^{K_v×K_h}$$
4. 可观测层偏置$$a \in \Bbb{R}^{K_v}$$
5. 隐藏层偏置$$b \in \Bbb{R}^{K_h}$$

**能量函数**$E(v,h)$定义为：

$$ E(v,h) = -(\sum_{i}^{} {a_iv_i} + \sum_{j}^{} {b_jh_j} + \sum_{i,j}^{} {v_iw_{ij}h_j}) = -a^Tv-b^Th-v^TWh $$

**联合概率分布**$p(v,h)$定义为：

$$ p(v,h) = \frac{1}{Z} exp(-E(v,h)) = \frac{1}{Z} exp(a^Tv) exp(b^Th) exp(v^TWh) $$

其中$Z$是配分函数(Partition Function)。

常见的受限玻尔兹曼机有以下三种：
1. **“伯努利-伯努利”受限玻尔兹曼机（Bernoulli-Bernoulli RBM, BB-RBM）**：上面介绍的可观测变量和隐变量都为二值类型的受限玻尔兹曼机;
2. **“高斯-伯努利”受限玻尔兹曼机（Gaussian-Bernoulli RBM, GB-RBM）**：可观测变量为高斯分布，隐变量为伯努利分布;
3. **“伯努利-高斯”受限玻尔兹曼机（Bernoulli-Gaussian RBM, BG-RBM）**：可观测变量为伯努利分布，隐变量为高斯分布。

# 2. 推断
**推断(Inference)**问题是指当给定受限玻尔兹曼机的模型参数时，由观测值生成服从联合概率分布$p(v,h)$的样本。

受限玻尔兹曼机的联合概率分布$$p(v,h)$$一般通过MCMC方法（如Gibbs采样）来近似。

在受限玻尔兹曼机的全条件概率中，可观测变量之间互相**条件独立**，隐变量之间也互相**条件独立**：

$$ p(v_i \mid v_{-i},h) = p(v_i \mid h) $$

$$ p(h_j \mid v,h_{-j}) = p(h_j \mid v) $$

因此，受限玻尔兹曼机可以并行地对所有的可观测变量（或所有的隐变量）同时进行采样，从而可以更快地达到热平衡状态。

**全条件概率**计算如下：

$$ p(v_i = 1 \mid h) = sigmoid(a_i + \sum_{j}^{} {w_{ij}h_j}) $$

$$ p(h_j = 1 \mid v) = sigmoid(b_j + \sum_{i}^{} {w_{ij}v_i}) $$

也可以写为向量形式，即:

$$ p(v = 1 \mid h) = sigmoid(a+Wh) $$

$$ p(h = 1 \mid v) = sigmoid(b+W^Tv) $$

受限玻尔兹曼机的**Gibbs采样**过程为：
1. 给定或随机初始化一个可观测的向量$v_0$,计算隐变量的概率，并从中采样一个隐向量$h_0$；
2. 基于$h_0$计算可观测向量的概率，并从中采样一个可观测的向量$v_1$；
3. 重复$t$次，获得$v_t$、$h_t$;
4. 当$$t → ∞$$时，$$(v_t,h_t)$$的采样分布服从$p(v,h)$分布。


# 3. 学习
**学习(Learning)**问题是指当给定变量的多组观测值时，学习模型的最优参数。

受限玻尔兹曼机通过最大化似然函数来找到最优的参数$W$,$a$,$b$。

对数似然函数：

$$ L(D;W,a,b) = \frac{1}{N} \sum_{n=1}^{N} {logp(\hat{v}^n \mid W,a,b)} $$

计算得对数似然函数对参数的梯度为：

$$ \frac{\partial L(D;W,a,b)}{\partial w_{ij}} = E_{\hat{p}(v)}E_{p(h \mid v)}(v_ih_j)-E_{p(v,h)}(v_ih_j) $$

$$ \frac{\partial L(D;W,a,b)}{\partial a_{i}} = E_{\hat{p}(v)}E_{p(h \mid v)}(v_i)-E_{p(v,h)}(v_i) $$

$$ \frac{\partial L(D;W,a,b)}{\partial b_{j}} = E_{\hat{p}(v)}E_{p(h \mid v)}(h_j)-E_{p(v,h)}(h_j) $$

### (1). Gibbs采样
上述计算涉及配分函数和期望，很难精确计算。一般通过MCMC方法（如Gibbs采样）来进行近似求解。

以参数$w_{ij}$的梯度为例:
1. 对于第一项，固定可观测变量$$v$$，只对隐变量$$h$$进行Gibbs采样，在训练集上所有的训练样本上重复此过程，得到$v_ih_j$的近似期望$$(v_ih_j)_{data}$$;
2. 对于第二项，对所有变量进行Gibbs采样。当达到热平衡状态时，采样$v_ih_j$的值，得到近似期望$$(v_ih_j)_{model}$$。

采用**梯度上升法**更新权重：

$$ w_{ij} = w_{ij} + α((v_ih_j)_{data} - (v_ih_j)_{model}) $$

根据受限玻尔兹曼机的条件独立性，可以对可观测变量和隐变量进行分组轮流采样。

受限玻尔兹曼机的采样效率会比玻尔兹曼机有很大提高，但一般还是需要通过很多步采样才可以采集到符合真实分布的样本。

### (2). 对比散度
**[对比散度（Contrastive Divergence）](https://www.researchgate.net/publication/11207765_ARTICLE_Training_Products_of_Experts_by_Minimizing_Contrastive_Divergence)**算法使得受限玻尔兹曼机的训练非常高效。

对比散度算法用一个训练样本作为可观测向量的初始值。然后交替对可观测向量和隐向量进行Gibbs采样，不需要等到收敛，只需要$K$步就足够了。

算法的流程如下：
1. 初始化可观测的向量$v_0$,计算隐变量的概率，并从中采样一个隐向量$h_0$；
2. 基于$h_0$计算可观测向量的概率，并从中采样一个可观测的向量$v_1$；
3. 基于$v_1$计算隐变量的概率，并从中采样一个隐向量$h_1$；
4. 更新参数：

$$ W = W + α(v_0h_0^T - v_1h_1^T) $$

$$ a = a + α(v_0 - v_1) $$

$$ b = b + α(h_0 - h_1) $$

重复上述步骤，直到满足终止判断条件。

# 4. 深度玻尔兹曼机
**深度玻尔兹曼机**堆叠多层隐藏层，每两层可以看作一组受限玻尔兹曼机。
![](https://pic.downk.cc/item/5e97db41c2a9a83be538112a.jpg)
