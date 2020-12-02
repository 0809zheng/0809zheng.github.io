---
layout: post
title: '变分自编码器'
date: 2020-05-30
author: 郑之杰
cover: 'https://pic.downk.cc/item/5fb715b8b18d627113d7ae88.jpg'
tags: 机器学习
---

> Variational Autoencoder.

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

1. 模型简介
2. 模型细节
3. 模型实现

# 1. 模型简介
**变分自编码器(Variational Autoencoder,VAE)**是第一个建立起**深度学习**和**Bayes统计**的隐变量模型，即一种能够显式表示隐空间概率分布的生成模型。**VAE**希望能够学习数据集的有效特征表示（即编码），并能够从这种特征表示中采样生成新的数据。

与自编码器类似，**VAE**把输入的**观测数据(observed data)**编码到**隐空间(latent space)**，这种编码称为**隐变量(latent variable)**或**隐向量(latent vector)**，并解码重构输入数据。从给定数据集学习隐空间的分布的过程称为**推断(inference)**，从隐空间采样生成新的数据的过程称为**生成(generation)**。

记观测数据为$x$，隐变量为$z$，由**Bayes**定理可得：

$$ p(x | z) = \frac{p(z | x)p(x)}{p(z)} $$

在自编码器中，$p(z \| x)$为编码器，$p(x \| z)$为解码器，两者都由神经网络拟合。但在自编码器中，输入数据的分布$p(x)$和隐变量的分布$p(z)$是**不可解(intractable)**的。对于变分自编码器，通过强制使隐变量的先验分布$p(z)$为标准正态分布，引入分布$q(z)$近似后验分布$p(z \| x)$，并假设分布$q(z)$也是正态分布，最终获得对输入数据的分布估计。

# 2. 模型细节
**VAE**的训练是通过最大化观测数据的对数似然$\log p_{\theta}(x)$实现的。直接解该问题是不可解的，需要使用**变分推断**求解。

由[变分推断](https://0809zheng.github.io/2020/03/25/variational-inference.html)理论，引入一个新的分布$q_{\phi}(z)$作为隐变量$z$的后验分布$p_{\theta}(z \| x)$的近似，并最大化变分下界$\mathcal{L}$（**ELBO**）来代替最大化观测数据的对数似然。

**ELBO**可以表示为：

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} $$

为了将上述**ELBO**适用于**VAE**，将其改写为期望的形式，并用**Bayes**定理整理如下：

$$ \mathcal{L} = \sum_{z}^{} {q_{\phi}(z) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)}} \\ = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z)} \\ = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(x | z)p_{\theta}(z)}{q_{\phi}(z)} \\ = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log p_{\theta}(x | z) + \mathbb{E}_{z \text{~} q_{\phi}(z)} \log \frac{p_{\theta}(z)}{q_{\phi}(z)} \\ = \mathbb{E}_{z \text{~} q_{\phi}(z)} \log p_{\theta}(x | z) - KL[q_{\phi}(z)||p_{\theta}(z)] $$

由于$q_{\phi}(z)$用来近似$p_{\theta}(z \| x)$，可以选择$q_{\phi}(z) = q_{\phi}(z\|x)$，则**VAE**的目标函数可以表示为：

$$ \mathcal{L} = \mathbb{E}_{z \text{~} q_{\phi}(z|x)} \log p_{\theta}(x | z) - KL[q_{\phi}(z|x)||p_{\theta}(z)] $$

目标函数的第一项$$\mathbb{E}_{z \text{~} q_{\phi}(z \| x)} \log p_{\theta}(x \| z)$$表示**重构误差**，即把观测数据映射到隐空间中，再还原为原数据的过程。在一定条件下，该项等价于均方误差(**MSE**)。

**VAE**假设隐变量$z$的先验分布为**标准正态分布**$p_{\theta}(z) = \mathcal{N}(0,I)$，目标函数的第二项$$- KL[q_{\phi}(z\|x)\|\|p_{\theta}(z)]$$使得$q_{\phi}(z\|x)$趋近标准正态分布，可以看作一种**正则化**。

$q_{\phi}(z\|x)$优化的目标是趋近标准正态分布，但是其形式仍需要人为给定。选择**多维对角正态分布**作为其分布，即表达如下：

$$ q_{\phi}(z|x) = \mathcal{N}(\mu_{\phi}(x),diag(\sigma_{\phi}^{2}(x))) $$

其中正态分布的参数$\mu$和$\sigma^{2}$看作输入数据$x$的函数，使用神经网络拟合。分布$p_{\theta}(x \| z)$也使用神经网络拟合，则**VAE**的整体结构如下：

![](http://adamlineberry.ai/images/vae/vae-architecture.png)

其中$q_{\phi}(z\|x)$也被称作**概率编码器(probabilistic encoder)**，$p_{\theta}(x \| z)$也被称作**概率解码器(probabilistic decoder)**。

由于直接从$q_{\phi}(z\|x)$中采样$z$的过程无法进行反向传播，因此使用**重参数化(reparameterization)**的技巧。把隐变量$z_i$表示成输入数据$x_i$与随机噪声$\epsilon_i$的的函数：

$$ z_i = g_{\phi}(x_i,\epsilon_i) = \mu_{\phi}(x_i) + diag(\sigma_{\phi}^{2}(x_i)) \cdot \epsilon_i $$

$$ \epsilon_i \text{~} \mathcal{N}(0,I) $$

![](http://adamlineberry.ai/images/vae/architecture-with-reparam.png)

# 3. 模型实现
从第二节可知**VAE**的优化目标函数如下：

$$ \mathcal{L} = \mathbb{E}_{z \text{~} q_{\phi}(z|x)} \log p_{\theta}(x | z) - KL[q_{\phi}(z|x)||p_{\theta}(z)] $$

### 第一项：重构损失
重构损失可以用**均方误差(MSE)**表示，Pytorch实现如下：

```
recons_loss = F.mse_loss(recons, input, reduction = 'sum')
```

注意`reduction`参数可选`'sum'`和`'mean'`，应该使用`'sum'`，这使得损失函数计算与原式保持一致。笔者在实现时曾选用`'mean'`，导致即使训练损失有下降，也只能生成噪声图片，推测是因为取平均使重构损失误差占比过小，无法正常训练。

### 第二项：KL散度
目标第二项是最小化**变分后验(variational posterior)**和**先验(prior)**之间的KL散度。由于两个分布都是正态分布，KL散度有闭式解(**closed-form solution**)，计算如下：

$$ KL[q_{\phi}(z|x)||p_{\theta}(z)] = \frac{1}{2} \sum_{j=1}^{J} (\mu_{j}^{2} + \sigma_{j}^{2} - \log \sigma_{j}^{2} - 1) $$

其中$J$表示隐空间的维数，$\mu$和$\sigma^{2}$是概率编码器的输出。Pytorch实现如下：

```
kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
```

**VAE**的总损失函数表达如下：

```
loss = recons_loss + kld_weight * kld_loss
```

### 从编码器中采样
使用重参数化的技巧从编码器输出中采样：

```
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
```