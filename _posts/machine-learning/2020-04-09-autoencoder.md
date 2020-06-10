---
layout: post
title: '自编码器'
date: 2020-04-09
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e8eb7ff504f4bcb040d0c17.jpg'
tags: 机器学习
---

> Auto Encoder.

**本文目录**：
1. 自编码器
2. 稀疏自编码器
3. 栈式自编码器
4. 降噪自编码器

# 1. 自编码器
**自编码器(Auto Encoder)**是通过无监督的方式来学习一组数据的有效编码（或表示）。

假设有一组$D$维的样本$$x^{(n)} \in \Bbb{R}^D,1≤n≤N$$，自编码器将这组数据映射到特征空间，得到每个样本的编码$$z^{(n)} \in \Bbb{R}^M,1≤n≤N$$。

自编码器的结构可以分成两部分：

**编码器**：$$f : \Bbb{R}^D → \Bbb{R}^M$$

**解码器**：$$g : \Bbb{R}^M → \Bbb{R}^D$$

自编码器的学习目标是最小化**重构误差(Reconstruction error)**:

$$ L = \sum_{n=1}^{N} {\mid\mid x^{(n)}-g(f(x^{(n)})) \mid\mid^2} $$

使用神经网络实现自编码器：
![](https://pic.downk.cc/item/5e8e9ba9504f4bcb04f2e20e.jpg)

**编码器**：$$z = f(W^{(1)}x+b^{(1)})$$

**解码器**：$$x' = f(W^{(2)}z+b^{(2)})$$

**重构误差**:

$$ L = \sum_{n=1}^{N} {\mid\mid x^{(n)}-x'^{(n)} \mid\mid^2} + λ \mid\mid W \mid\mid_F^2 $$

其中λ是正则化系数。

如果$W^{(2)} = {W^{(1)}}^T$，称为**捆绑权重(Tied weight)**。捆绑权重自编码器的参数更少，因此更容易学习。此外，捆绑权重还在一定程度上起到正则化的作用。

如果特征空间的维度$M$小于原始空间的维度$D$，自编码器相当于是一种降维或特征抽取方法。

如果编码只能取$K$个不同的值$$(𝐾<𝑁)$$，那么自编码器就可以转换为一个$K$类的聚类（Clustering）问题。

使用自编码器是为了得到有效的数据表示，因此在训练结束后，一般会去掉解码器，只保留编码器。编码器的输出可以直接作为后续机器学习模型的输入。

自编码器也可以用于深层神经网络的逐层预训练：

![](https://pic.downk.cc/item/5ee0d720c2a9a83be5d3fe2f.jpg)

# 2. 稀疏自编码器
自编码器除了可以学习低维编码之外，也能够学习高维的稀疏编码。

假设中间隐藏层$z$的维度$M$大于输入样本$x$的维度$D$，并让$z$尽量稀疏，这就是**稀疏自编码器（Sparse Auto-Encoder）**。

稀疏自编码器的优点是有很高的可解释性，并同时进行了隐式的特征选择。

稀疏自编码器的目标函数是：

$$ L = \sum_{n=1}^{N} {\mid\mid x^{(n)}-x'^{(n)} \mid\mid^2} + λ \mid\mid W \mid\mid_F^2 + ηρ(z) $$

其中λ是正则化系数,$ρ$是一个衡量稀疏性的函数。

$ρ$通常使用**$l_1$**范数:

$$ ρ(z) = \sum_{m=1}^{M} {\mid z_m \mid} $$

或者事先给定隐藏层第$m$个神经元激活的目标概率$ρ^*$，由样本计算第$m$个神经元的**平均激活度**$ρ_m$：

$$ ρ_m = \frac{1}{N} \sum_{n=1}^{N} {z_m^{(n)}} $$

用KL散度衡量$ρ^*$和$ρ_m$的差异：

$$ ρ(z) = \sum_{m=1}^{M} {KL(ρ^* \mid\mid ρ_m)} $$

$$ KL(ρ^* \mid\mid ρ_m) = ρ^*log(\frac{ρ^*}{ρ_m}) + (1-ρ^*)log(\frac{1-ρ^*}{1-ρ_m}) $$

平均激活度$ρ_m$是根据所有样本计算出来的，实际中可以只计算Mini-Batch中包含的样本的平均激活度，然后用滑动平均求近似值：

$$ ρ_m^{(t)} = βρ_m^{(t-1)} + (1-β)ρ_m^{(t)} $$

# 3. 栈式自编码器
**栈式自编码器（Stacked Auto-Encoder，SAE）**是指使用逐层堆叠的方式来训练一个深层的自编码器。

栈式自编码器一般可以采用[逐层训练（Layer-Wise Training）](https://www.researchgate.net/publication/200744514_Greedy_layer-wise_training_of_deep_networks)来学习网络参数：

1. 首先训练第一个自编码器，然后保留第一个自编码器的编码器部分；
2. 把第一个自编码器的中间层作为第二个自编码器的输入层进行训练；
3. 反复地把前一个自编码器的中间层作为后一个编码器的输入层，进行迭代训练。

# 4. 降噪自编码器
[**降噪自编码器（Denoising Auto-Encoder）**](https://www.researchgate.net/publication/221346269_Extracting_and_composing_robust_features_with_denoising_autoencoders)是一种通过引入噪声来增加编码鲁棒性的自编码器。

对于一个向量$x$，根据损坏比例$μ$随机将$x$的一些维度的值设置为$0$，得到带有噪声的向量$\tilde{x}$。然后将$\tilde{x}$输入给自编码器得到编码$z$，并重构原始的无损输入$x$。

向样本中加入噪声也可以引入Gaussian噪声，即$$\tilde{x} = x + ε$$，噪声$ε$~$$N(0,σ^2)$$。

![](https://pic.downk.cc/item/5e8ea202504f4bcb04f97a19.jpg)
