---
layout: post
title: 'More Control for Free! Image Synthesis with Semantic Diffusion Guidance'
date: 2022-06-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6436624f0d2dde5777239cfd.jpg'
tags: 论文阅读
---

> 基于语义扩散引导的图像合成.

- paper：[More Control for Free! Image Synthesis with Semantic Diffusion Guidance](https://arxiv.org/abs/2112.05744)


本文提出了一种实现条件扩散模型的**语义扩散引导(Semantic Diffusion Guidance)**方法。实现扩散模型的一般思路：
1. 定义前向扩散过程：$$q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)$$
2. 解析地推导：$$q\left(\mathbf{x}_t \mid \mathbf{x}_{0}\right)$$
3. 解析地推导：$$q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{x}_{0}\right)$$
4. 近似反向扩散过程：$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$

条件扩散模型是指在采样过程$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$中引入输入条件$$\mathbf{y}$$，则采样过程变为$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$。为了重用训练好的模型$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$$，定义：

$$
\begin{aligned}
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right) &= \frac{p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)e^{\gamma \cdot \text{sim}(\mathbf{x}_{t-1},\mathbf{y})}}{Z(\mathbf{x}_t,\mathbf{y})} \\
Z(\mathbf{x}_t,\mathbf{y}) &= \sum_{\mathbf{x}_{t-1}} p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)e^{\gamma \cdot \text{sim}(\mathbf{x}_{t-1},\mathbf{y})}
\end{aligned}
$$

其中$$\text{sim}(\mathbf{x}_{t-1},\mathbf{y})$$是生成结果$$\mathbf{x}_{t-1}$$和输入条件$$\mathbf{y}$$之间的相似性度量，$\gamma$控制结果与条件的相关性。

为了进一步得到可采样的近似结果，在$$\mathbf{x}_{t-1}=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)$$处考虑泰勒展开：

$$
\begin{aligned}
e^{\gamma \cdot \text{sim}(\mathbf{x}_{t-1},\mathbf{y})} \approx e^{\gamma \cdot \text{sim}(\mathbf{x}_t,\mathbf{y})+ \gamma(\mathbf{x}_{t-1}-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)) \cdot \nabla_{\mathbf{x}_t} \text{sim}(\mathbf{x}_t,\mathbf{y}) + \mathcal{O}(\mathbf{x}_t) } 
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
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right) &\propto p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)e^{\gamma \cdot \text{sim}(\mathbf{x}_{t-1},\mathbf{y})} \\
&\propto \exp\left(-\frac{\left\| \mathbf{x}_{t-1} -\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right) \right\|^2}{2\sigma_t^2}+\gamma(\mathbf{x}_{t-1}-\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)) \cdot \nabla_{\mathbf{x}_t} \text{sim}(\mathbf{x}_t,\mathbf{y}) + \mathcal{O}(\mathbf{x}_t) + \mathcal{O}(\mathbf{x}_t) \right) \\ 
&\propto \exp\left(-\frac{\left\| \mathbf{x}_{t-1} -\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)-\sigma_t^2 \gamma \nabla_{\mathbf{x}_t} \text{sim}(\mathbf{x}_t,\mathbf{y})\right\|^2}{2\sigma_t^2} + \mathcal{O}(\mathbf{x}_t) \right) \\ 
\end{aligned}
$$

则$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$近似服从正态分布：

$$
\begin{aligned}
p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)+\sigma_t^2 \gamma \nabla_{\mathbf{x}_t} \text{sim}(\mathbf{x}_t,\mathbf{y}), \sigma_t^2 \mathbf{I}\right) \\
\end{aligned}
$$

因此条件扩散模型$$p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t,\mathbf{y}\right)$$的采样过程为：

$$
\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)+\underbrace{\sigma_t^2 \gamma \nabla_{\mathbf{x}_t} \text{sim}(\mathbf{x}_t,\mathbf{y})|_{\mathbf{x}_t=\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)}}_{\text{新增项}}+ \sigma_t \mathbf{z},\mathbf{z} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)
$$

因此实现条件扩散模型，只需要直接定义度量函数$$\text{sim}(\mathbf{x}_t,\mathbf{y})$$即可。通常的处理方式是用各自的编码器（$$\mathbf{x}_t$$:图像；$$\mathbf{y}$$:图像、文本、类别等）将其编码为特征向量，然后用余弦相似度计算：

$$ \text{sim}(\mathbf{x}_t,\mathbf{y}) = \frac{E_1(\mathbf{x}_t)\cdot E_2(\mathbf{y})}{\left\|E_1(\mathbf{x}_t)\right\|\left\|E_2(\mathbf{y})\right\|} $$

值得一提的是，由于$$\mathbf{x}_t$$是带高斯噪声的，所以编码器$E1$一般不能直接调用干净数据训练的编码器，而是要用加噪声后的数据对它进行微调才比较好。