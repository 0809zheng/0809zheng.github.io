---
layout: post
title: 'Classifier-Free Diffusion Guidance'
date: 2022-06-10
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/643669bd0d2dde57773345a9.jpg'
tags: 论文阅读
---

> 基于事前训练的条件生成扩散模型.

- paper：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)


本文提出了一种实现条件扩散模型的事前训练方法。实现扩散模型的一般思路：
1. 定义前向扩散过程：$$q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$$
2. 解析地推导：$$q\left(\mathbf{x}_t \mid \mathbf{x}_{0}\right)$$
3. 解析地推导：$$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{x}_{0}\right)$$
4. 近似反向扩散过程：$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$

条件扩散模型是指在采样过程$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$中引入输入条件$$\mathbf{y}$$，则采样过程变为$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$。

注意到反向扩散过程的建模：

$$
\begin{aligned}
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)&=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \sigma_t^2 \mathbf{I}\right) \\
\end{aligned}
$$

引入输入条件$$\mathbf{y}$$后，定义反向扩散过程：

$$
\begin{aligned}
p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)&=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right), \sigma_t^2 \mathbf{I}\right) \\
\end{aligned}
$$

一般把$$\boldsymbol{\mu}_\theta$$表示为$$\boldsymbol{\epsilon}_\theta$$的函数：

$$
\boldsymbol{\mu}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right)\right)
$$

对应训练的损失函数：

$$ \begin{aligned} \mathbb{E}_{t \sim[1, T], \mathbf{x}_0, \mathbf{y},\epsilon_t}\left[\left\|\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t,\mathbf{y}, t\right)\right\|^2\right]\end{aligned} $$ 

可以引入$w$参数控制生成结果的相关项与多样性。用：

$$ \tilde{\boldsymbol{\epsilon}}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right) = (1+w) \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right) - w\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) $$

代替$$\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right)$$来做生成。则当$w$越大时，生成过程将使用更多的条件信号，结果将会提高生成结果与输入信号$y$的相关性，但是会相应地降低生成结果的多样性；反之，则会降低生成结果与输入信号之间的相关性，但增加了多样性。

$$\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)$$可以通过引入一个特定的输入类别$\phi$实现，它对应的目标图像为全体图像，因此把$$\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) = \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, \phi, t\right)$$加入到模型训练中。

噪声预测器$$\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t,\mathbf{y}, t\right)$$通过条件**UNet**模型实现，$$\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)$$则通过丢弃条件输入实现：

```python
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return scaled_logits
```

基于事前训练的条件扩散模型的完整实现可参考[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py)。