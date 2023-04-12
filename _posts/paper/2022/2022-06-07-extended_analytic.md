---
layout: post
title: 'Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models'
date: 2022-06-07
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6435207a0d2dde57777a7f09.jpg'
tags: 论文阅读
---

> 扩散概率模型中具有不准确均值的最优协方差估计.

- paper：[Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models](https://arxiv.org/abs/2206.07309)


[<font color=Blue>Analytic-DPM</font>](https://0809zheng.github.io/2022/06/06/analytic.html)给出了已经训练好的生成扩散模型的最优方差的一个解析估计，实验显示该估计结果确实能有效提高扩散模型的生成质量。

**扩散模型 (Diffusion Model)**是一类深度生成模型。这类模型首先定义前向扩散过程，向数据中逐渐地添加随机噪声；然后学习反向扩散过程，从噪声中构造所需的数据样本。

对于一个训练完成的扩散模型，在采样（反向扩散）过程中，我们希望求解$$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)$$。一种常见的求解方法是首先通过$$\mathbf{x}_{t}$$构造$$\mathbf{x}_{0}$$，然后计算：

$$
\begin{aligned}
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t},\mathbf{x}_0\right) \approx q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t},\mathbf{x}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}\right)
\end{aligned}
$$

然而从$$\mathbf{x}_{t}$$构造$$\mathbf{x}_{0}$$并不是完全准确的，因此应该用概率分布而非确定性的函数来描述它。事实上，严格地有：

$$
\begin{aligned}
q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right) = \int q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t},\mathbf{x}_0\right) q\left(\mathbf{x}_0 \mid \mathbf{x}_{t}\right) d \mathbf{x}_0
\end{aligned}
$$

注意到$$q\left(\mathbf{x}_0 \mid \mathbf{x}_{t}\right)$$是未知的，因此用正态分布$$\mathcal{N}\left(\mathbf{x}_0 ; \bar{\mu}(\mathbf{x}_{t}),\bar{\sigma}_t^2 \mathbf{I}\right)$$进行近似。

用正态分布$$\mathcal{N}\left(\mathbf{x}_0 ; \bar{\mu}(\mathbf{x}_{t}),\bar{\sigma}_t^2 \mathbf{I}\right)$$近似$$q\left(\mathbf{x}_0 \mid \mathbf{x}_{t}\right)$$，落脚于分别近似$$q\left(\mathbf{x}_0 \mid \mathbf{x}_{t}\right)$$的均值和方差：

$$
\begin{aligned}
\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right) &= \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{\bar{\alpha}_t}} \\
\hat{\sigma}_t^2 &= \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t} \left(1-\frac{1}{d}\mathbb{E}_{\mathbf{x}_t \sim q\left(\mathbf{x}_{t}\right)} \left[ ||\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) ||^2 \right] \right)
\end{aligned}
$$

## ⚪ Extended-Analytic-DPM

值得一提的是，方差估计$$\hat{\sigma}_t^2$$是建立在均值$$\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)$$精确估计的基础上。然而$$\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)$$是通过网络学习得到的（更具体地，$$\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)$$是由神经网络近似的），因此均值是一种**Imperfect Mean**。本文作者讨论如何在**Imperfect Mean**下改进估计结果。

假设均值$$\bar{\mu}(\mathbf{x}_{t})$$已经训练完成，则分布$$\mathcal{N}\left(\mathbf{x}_0 ; \bar{\mu}(\mathbf{x}_{t}),\bar{\sigma}_t^2 \mathbf{I}\right)$$的未知参数还有$$\bar{\sigma}_t^2$$。考虑其负对数似然：

$$
\begin{aligned}
&\mathbb{E}_{\mathbf{x}_t \sim q\left(\mathbf{x}_{t}\right)}\mathbb{E}_{\mathbf{x}_0 \sim q\left(\mathbf{x}_0 \mid \mathbf{x}_{t}\right)} \left[ -\log \mathcal{N}\left(\mathbf{x}_0 ; \bar{\mu}(\mathbf{x}_{t}),\bar{\sigma}_t^2 \mathbf{I}\right) \right] \\
= &\frac{\mathbb{E}_{\mathbf{x}_t,\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}\right)q\left(\mathbf{x}_t \mid \mathbf{x}_{0}\right)} \left[ ||\mathbf{x}_0  -\bar{\mu}(\mathbf{x}_{t})||^2\right]}{2\bar{\sigma}_t^2} + \frac{1}{2} \log \bar{\sigma}_t^2 + \frac{1}{2} \log 2\pi
\end{aligned}
$$

上式取得最小值对应：

$$
\begin{aligned}
\bar{\sigma}_t^2 &= \mathbb{E}_{\mathbf{x}_t,\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}\right)q\left(\mathbf{x}_t \mid \mathbf{x}_{0}\right)} \left[ ||\mathbf{x}_0  -\bar{\mu}(\mathbf{x}_{t})||^2\right] \\
&= \mathbb{E}_{\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}\right), \boldsymbol{\epsilon}\sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)}\left[\left\| \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}} - \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)}{\sqrt{\bar{\alpha}_t}} \right\|^2\right] \\
&= \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\mathbb{E}_{\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}\right), \boldsymbol{\epsilon}\sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)}\left[\left\| \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \right\|^2\right] \\
\end{aligned}
$$

如果把方差$$\bar{\sigma}_t^2$$也建模为$$\mathbf{x}_t$$的函数$$\bar{\sigma}_t^2(\mathbf{x}_t)$$，则可以构建目标函数：

$$
\begin{aligned}
\bar{\sigma}_t^2(\mathbf{x}_t) &= \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\mathbb{E}_{\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}| \mathbf{x}_t\right)}\left[\left( \boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \right)^2\right] \\
&= \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\mathop{\arg\min}_{\mathbb{g}}\mathbb{E}_{\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}| \mathbf{x}_t\right)}\left[\left\| \left(\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \right)^2-\mathbb{g}\right\|^2\right] \\
&= \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\mathop{\arg\min}_{\mathbb{g}(\mathbf{x}_t)}\mathbb{E}_{\mathbf{x}_t \sim q\left(\mathbf{x}_{t}\right)}\mathbb{E}_{\mathbf{x}_0 \sim q\left(\mathbf{x}_{0}| \mathbf{x}_t\right)}\left[\left\| \left(\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \right)^2-\mathbb{g}(\mathbf{x}_t)\right\|^2\right] \\
&= \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}\mathop{\arg\min}_{\mathbb{g}(\mathbf{x}_t)}\mathbb{E}_{\mathbf{x}_t ,\mathbf{x}_0 \sim q\left(\mathbf{x}_{t}| \mathbf{x}_0\right)q\left(\mathbf{x}_{0}\right)}\left[\left\| \left(\boldsymbol{\epsilon}_t-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) \right)^2-\mathbb{g}(\mathbf{x}_t)\right\|^2\right] \\
\end{aligned}
$$

**Extended-Analytic-DPM**提出了两阶段的训练方案，即用原始固定方差的测试训练好均值模型$$\bar{\mu}(\mathbf{x}_{t})$$，然后固定该模型，并重用该模型的大部分参数来学一个方差模型$$\bar{\sigma}_t^2(\mathbf{x}_t)$$。该方法降低了参数量和训练成本，允许重用已经训练好的均值模型，训练过程更加稳定。
