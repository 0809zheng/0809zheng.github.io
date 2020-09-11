---
layout: post
title: '图像分类中的对抗攻击'
date: 2020-07-26
author: 郑之杰
cover: ''
tags: 深度学习
---

> 一些图像分类中的对抗攻击方法.

# 攻击方法
- **白盒攻击（white-box attacks）**：在已经获取机器学习模型内部的所有信息和参数上进行攻击。已知给定模型的梯度信息生成对抗样本。
- **黑盒攻击（black-box attacks）**：在神经网络结构为黑箱时，仅通过模型的输入和输出，生成对抗样本。
- **跨模型可转移性（cross-model transferability）**：对一个模型制作的对抗样本在很大概率下会欺骗其他模型。可转移性使得黑盒攻击能够应用于实际，并引发严重的安全问题（自动驾驶、医疗）。
- **单步攻击**：仅进行一次更新，容易**underfit**，针对白盒攻击效果差，针对黑盒攻击效果好（转移性强）；
- **多步攻击**：迭代地更新，容易**overfit**，针对白盒攻击效果好，针对黑盒攻击效果差（转移性差）。

常用的对抗攻击方法：
- GFSM
- I-FGSM（SIM）
- MI-FGSM
- NI-FGSM
- DIM
- TIM

### ⚪ FGSM（Fast Gradient Sign Method）
- paper：[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)

**FGSM**通过寻找一个对抗样本$x^{adv}$来使损失函数$J(x^{adv},y^{true})$最大化。这个算法的假设是决策边界周围的数据点是线性的；

$$ x^{adv}=x+{\epsilon} \cdot sign({\nabla}_xJ(x,y^{true})) $$

**FGSM**算法只涉及单次梯度更新，沿梯度方向移动步长$\epsilon$。

### ⚪ I-FGSM（Iterative FGSM） 也叫 BIM（Basic Iterative Method）
- paper：[Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533v4)

**I-GFSM**相比于**FGSM**，实现迭代更新，

$$ x_0=x \\ x_{t+1}^{adv}=Clip_{x,\epsilon}\{x_{t}^{adv}+{\alpha} \cdot sign({\nabla}_xJ(x_t^{adv},y^{true}))\} $$

步长$\alpha=1$，将溢出的数值用边界值$\epsilon$代替。

上述方法是一种无目标的攻击；作者还提出了一种有目标的攻击方法：**iterative least-likely class method（LLC）**：将输入图像分类成原本最不可能分到的类别$y^{LL}$。

$$ x_0=x \\ x_{t+1}^{adv}=Clip_{x,\epsilon}\{x_{t}^{adv}-{\alpha} \cdot sign({\nabla}_xJ(x_t^{adv},y^{LL}))\} $$

### ⚪ MI-FGSM（Momentum Iterative FGSM）
- paper：[Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081)

**MI-GFSM**在梯度上升中引入了动量方法，稳定更新方向，避免局部极值：

$$ g_{t+1}={\mu}g_{t} + \frac{ {\nabla}_x J(x_t^{adv},y^{true}) } {\mid\mid {\nabla}_x J(x_t^{adv},y^{true}) \mid\mid_1} $$

$$ x_{t+1}^{adv} = Clip_{x,\epsilon}\{x_{t}^{adv}+{\alpha} \cdot sign(g_{t+1})\} $$

作者还提出了攻击的**集成方法（ensemble attack）**，即攻击集成模型：

$$ l(x) = \sum_{k=1}^{K} w_k l_k(x) $$

其中$w_k$是第$k$个模型对应的权重，$l_k(x)$表示的是第$k$个模型输入$softmax$的$logits$。作者通过实验发现这比集成模型$softmax$输出的$prediciton$或集成模型计算得到的$loss$效果更好。

### ⚪ NI-FGSM（Nesterov Iterative FGSM）
- paper：[Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attack](https://arxiv.org/abs/1908.06281v2)

将基于梯度的对抗样本生成过程看作一种训练过程，训练的对象是对抗扰动，训练用的数据是被白盒攻击的模型。作者从优化方法和模型增强两个角度出发，提高对抗样本的效果。

一方面，从优化方法的角度出发，将**Nesterov Accelerated Gradient（NAG）**应用到对抗攻击中。相比于**Momentum**，**NAG**除了具有稳定梯度更新方向的作用之外，还具有向前看的性质，可以有效加速对抗样本的生成和收敛效果。

$$ x_t^{nes} = x_{t}^{adv} + \alpha \cdot \mu \cdot g_{t} $$

$$ g_{t+1}={\mu}g_{t} + \frac{ {\nabla}_x J(x_t^{nes},y^{true}) } {\mid\mid {\nabla}_x J(x_t^{nes},y^{true}) \mid\mid_1} $$

$$ x_{t+1}^{adv} = Clip_{x,\epsilon}\{x_{t}^{adv}+{\alpha} \cdot sign(g_{t+1})\} $$

另一方面，从模型增强的角度出发，通过攻击不同放缩大小的图片，变相实现对被攻击的白盒模型的模型增强，从而提高生成的对抗样本的泛化能力。


### ⚪ DIM（Diverse Input Method）
- paper：[Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978)

作者对每次攻击图像引入了一个随机移植函数$T$，每轮更新有概率$p$会对图像进行**resize**和**padding**操作：

![](https://pic.downk.cc/item/5f02e07114195aa594ecaa17.jpg)

$$ x_0=x \\ x_{t+1}^{adv}=Clip_{x,\epsilon}\{x_{t}^{adv}+{\alpha} \cdot sign({\nabla}_xJ(T(x_t^{adv};p),y^{true}))\} $$

### ⚪ TIM（Translation-Invariant Method）
- paper：[Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/abs/1904.02884)

对于一般的对抗样本生成问题，目标函数是最大化对抗样本和真实样本对应的标签间的损失函数$J(x^{adv},y^{true})$，所以优化函数为：

$$ \mathop{\arg \max}_{x^{adv}} J(x^{adv},y^{true}) \\ \text{s.t. } \mid\mid x^{adv} - x \mid\mid_∞ ≤ \epsilon $$

为了生成对白盒模型的识别区域不敏感的对抗样本，作者采用的方法是用一系列平移后的图像来优化对抗样本：

$$ \mathop{\arg \max}_{x^{adv}} \sum_{i,j}^{} {w_{ij}J(T_{ij}(x^{adv}),y^{true})} \\ \text{s.t. } \mid\mid x^{adv} - x \mid\mid_∞ ≤ \epsilon $$

其中，$T_{ij}(x)$是**平移函数（translation operation）**，将图像$x$在对应维度平移$i$、$j$个像素点，设置$$i,j∈\{-k,…,0,…,k\}$$，$k$为平移的最大像素值。这样，生成的对抗样本将减弱对被攻击的白盒模型的识别区域的敏感，这能够帮助其转移到其他模型。

对于上述优化算法，需要计算$(2k+1)^2$张图像的梯度。经过推导：

$$ \nabla_x \sum_{i,j}^{} {w_{ij}J(T_{ij}(x^{adv}),y^{true})} ≈W \cdot \nabla_x J(x^{adv},y^{true}) $$

因此，不需要求得所有图像的梯度，而是求未平移图像的梯度，然后对平移的梯度求平均，也等价于对梯度和由权值$w_{ij}$组成的核做卷积。下面是一些核矩阵的选取方法：
- **uniform kernel**：

$$W_{ij}=\frac{1}{(2k+1)^2}$$

- **linear kernel**：

$$\hat{W}_{ij}=(1-\frac{\mid i \mid}{k+1})\cdot(1-\frac{\mid j \mid}{k+1}) \\ W_{ij}=\frac{\hat{W}_{ij}}{\sum_{i,j}^{} {\hat{W}_{ij}}}$$

- **Gaussian kernel**：

$$\hat{W}_{ij}=\frac{1}{2\pi σ^2}exp(-\frac{i^2+j^2}{2σ^2}), σ=\frac{k}{\sqrt{3}} \\ W_{ij}=\frac{\hat{W}_{ij}}{\sum_{i,j}^{} {\hat{W}_{ij}}}$$
