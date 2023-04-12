---
layout: post
title: 'Diffusion Models Beat GANs on Image Synthesis'
date: 2022-06-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/643651800d2dde577700c957.jpg'
tags: 论文阅读
---

> 在图像合成任务上扩散模型超越了生成对抗网络.

- paper：[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)


本文提出了一种实现条件扩散模型的**事后修改（Classifier-Guidance）**方法。**事后修改**是指在已经训练好的无条件扩散模型的基础上引入一个可训练的分类器（**Classifier**），用分类器来调整生成过程以实现控制生成。这类模型的训练成本比较低，但是采样成本会高一些，而且难以控制生成图像的细节。

实现扩散模型的一般思路：
1. 定义前向扩散过程：$$q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$$
2. 解析地推导：$$q\left(\mathbf{x}_t \mid \mathbf{x}_{0}\right)$$
3. 解析地推导：$$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{x}_{0}\right)$$
4. 近似反向扩散过程：$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$

条件扩散模型是指在采样过程$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$中引入输入条件$$\mathbf{y}$$，则采样过程变为$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$。为了重用训练好的模型$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$，根据贝叶斯定理：

$$
\begin{aligned}
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right) &= \frac{p_{\theta}\left(\mathbf{x}_{t-1} , \mathbf{x}_t,\mathbf{y}\right)}{p_{\theta}\left(\mathbf{x}_t,\mathbf{y}\right)} \\
&= \frac{p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_{t-1} , \mathbf{x}_t\right)p_{\theta}\left(\mathbf{x}_{t-1} , \mathbf{x}_t\right)}{p_{\theta}\left(\mathbf{y}\mid \mathbf{x}_t\right)p_{\theta}\left(\mathbf{x}_t\right)} \\
&= \frac{p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_{t-1} , \mathbf{x}_t\right)p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}{p_{\theta}\left(\mathbf{y}\mid \mathbf{x}_t\right)} \\
&= \frac{p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_{t-1}\right)p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}{p_{\theta}\left(\mathbf{y}\mid \mathbf{x}_t\right)} \\
&= p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) e^{\log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_{t-1}\right) - \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)}
\end{aligned}
$$

为了进一步得到可采样的近似结果，在$$\mathbf{x}_{t-1}=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)$$处考虑泰勒展开：

$$
\begin{aligned}
\log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_{t-1}\right) \approx \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right) + (\mathbf{x}_{t-1}-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)) \cdot \nabla_{\mathbf{x}_t} \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)} + \mathcal{O}(\mathbf{x}_t) 
\end{aligned}
$$

并注意到反向扩散过程的建模：

$$
\begin{aligned}
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)&=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \sigma_t^2 \mathbf{I}\right) \\
\end{aligned}
$$

因此有：

$$
\begin{aligned}
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right) &=p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) e^{\log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_{t-1}\right) - \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)} \\
&\propto \exp\left(-\frac{\left\| \mathbf{x}_{t-1} -\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right) \right\|^2}{2\sigma_t^2}+(\mathbf{x}_{t-1}-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)) \cdot \nabla_{\mathbf{x}_t} \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)} + \mathcal{O}(\mathbf{x}_t) \right) \\ 
&\propto \exp\left(-\frac{\left\| \mathbf{x}_{t-1} -\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)-\sigma_t^2 \nabla_{\mathbf{x}_t} \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)}\right\|^2}{2\sigma_t^2} + \mathcal{O}(\mathbf{x}_t) \right) \\ 
\end{aligned}
$$

则$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$近似服从正态分布：

$$
\begin{aligned}
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)+\sigma_t^2 \nabla_{\mathbf{x}_t} \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)}, \sigma_t^2 \mathbf{I}\right) \\
\end{aligned}
$$

因此条件扩散模型$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$的采样过程为：

$$
\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)+\underbrace{\sigma_t^2 \nabla_{\mathbf{x}_t} \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)}}_{\text{新增项}}+ \sigma_t \mathbf{z},\mathbf{z} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)
$$

向分类器的梯度中引入一个缩放参数$γ$，可以更好地调节生成效果：

$$
\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)+\sigma_t^2 \gamma \nabla_{\mathbf{x}_t} \log p_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)}+ \sigma_t \mathbf{z},\mathbf{z} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)
$$

当$γ>1$时，生成过程将使用更多的分类器信号，结果将会提高生成结果与输入信号$y$的相关性，但是会相应地降低生成结果的多样性；反之，则会降低生成结果与输入信号之间的相关性，但增加了多样性。

缩放参数$γ$相当于通过幂操作来提高分布的聚焦程度，即定义：

$$ \tilde{p}_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right) = \frac{p_{\theta}^{\gamma}\left(\mathbf{y} \mid \mathbf{x}_t\right)}{\sum_{\mathbf{y}} p_{\theta}^{\gamma}\left(\mathbf{y} \mid \mathbf{x}_t\right)} $$

随着$γ$的增大，$$\tilde{p}_{\theta}\left(\mathbf{y} \mid \mathbf{x}_t\right)$$的预测会越来越接近**one hot**分布，生成过程会倾向于挑出分类置信度很高的样本。