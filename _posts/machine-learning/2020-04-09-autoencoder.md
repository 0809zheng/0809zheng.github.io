---
layout: post
title: 'Autoencoder: 自编码器'
date: 2020-04-09
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e8eb7ff504f4bcb040d0c17.jpg'
tags: 机器学习
---

> Auto Encoder.

**自编码器(AutoEncoder)**是一种具有**瓶颈层(bottleneck layer)**的神经网络模型，被用于重建高维数据，并通过无监督的方式学习数据的有效编码（或表示）。

**本文目录**：
1. 自编码器 **Autoencoder**
2. 稀疏自编码器 **Sparse Autoencoder**
3. 压缩自编码器 **Contractive Autoencoder**
4. 降噪自编码器 **Denoising Autoencoder**
5. 栈式自编码器 **Stacked Autoencoder**
6. 变分自编码器 **Variational Autoencoder**

# 1. 自编码器
**自编码器(AutoEncoder)**以无监督的方式重构原始输入，同时压缩输入数据，从而发现更高效的压缩表示。

假设有一组$D$维的样本$$x^{(n)} \in \Bbb{R}^D,1≤n≤N$$，自编码器将这组数据映射到隐空间，得到每个样本的编码$$z^{(n)} \in \Bbb{R}^M,1≤n≤N$$。自编码器由两个网络组成：
- **编码器** $$g : \Bbb{R}^D → \Bbb{R}^M$$：将原始的高维输入转换为低维的特征编码$z=g(x)$。
- **解码器** $$f : \Bbb{R}^M → \Bbb{R}^D$$：从编码中恢复数据$x'=f(g(x))$。

![](https://pic.imgdb.cn/item/62a41c0c094754312953248f.jpg)

自编码器旨在学习一个恒等函数$x=f(g(x))$。自编码器的目标函数可以选择任意量化两个张量之间的差异的度量，如**重构误差(Reconstruction error)**:

$$ L = \sum_{n=1}^{N} {|| x^{(n)}-f(g(x^{(n)})) ||^2} $$

### ⚪ 使用多层感知机实现自编码器

![](https://pic.downk.cc/item/5e8e9ba9504f4bcb04f2e20e.jpg)

使用多层感知机对自编码器的编码器和解码器进行建模：

$$ \begin{aligned} z &= g(W^{(1)}x+b^{(1)}), x' = f(W^{(2)}z+b^{(2)}) \\ L &= \sum_{n=1}^{N} {|| x^{(n)}-x'^{(n)} ||^2} + λ || W ||_F^2 \end{aligned} $$


如果$W^{(2)} = {W^{(1)}}^T$，称为**捆绑权重(Tied weight)**。捆绑权重自编码器的参数更少，因此更容易学习。此外，捆绑权重还在一定程度上起到正则化的作用，其中λ是正则化系数。

如果特征空间的维度$M$小于原始空间的维度$D$，自编码器相当于是一种降维或特征抽取方法。

如果编码只能取$K$个不同的值$$(𝐾<𝑁)$$，那么自编码器就可以转换为一个$K$类的聚类（Clustering）问题。

使用自编码器是为了得到有效的数据表示，因此在训练结束后，一般会去掉解码器，只保留编码器。编码器的输出可以直接作为后续机器学习模型的输入。

# 2. 稀疏自编码器
**稀疏自编码器（Sparse Auto-Encoder）**是指对隐藏层单元$z$增加稀疏性约束，使得模型在任意时间只能激活少量的隐藏单元。稀疏性能够避免过拟合并提高模型鲁棒性，同时进行了隐式的特征选择，具有很高的可解释性。

稀疏自编码器的目标函数是：

$$ L = \sum_{n=1}^{N} {|| x^{(n)}-x'^{(n)} ||^2} + λ || W ||_F^2 + ηρ(z) $$

其中λ是正则化系数，$ρ(z)$是一个衡量**稀疏性**的函数。

$ρ(z)$通常使用$l_1$范数:

$$ ρ(z) = \sum_{m=1}^{M} {| z_m |} $$

或者事先给定隐藏层第$m$个神经元激活的目标概率$ρ^*$ (常取$0.05$)，由样本计算第$m$个神经元的**平均激活度**$ρ_m= \frac{1}{N} \sum_{n=1}^{N} {z_m^{(n)}}$，并用KL散度衡量$ρ^*$和$ρ_m$的差异：

$$ \begin{aligned} ρ(z) &= \sum_{m=1}^{M} {KL(ρ^* || ρ_m)} \\ & = \sum_{m=1}^{M}  ρ^*log(\frac{ρ^*}{ρ_m}) + (1-ρ^*)log(\frac{1-ρ^*}{1-ρ_m}) \end{aligned} $$

平均激活度$ρ_m$理论上应根据所有样本计算出来，实际中可以只计算Mini-Batch中包含的样本的平均激活度，然后用滑动平均迭代计算近似值：

$$ ρ_m^{(t)} = βρ_m^{(t-1)} + (1-β)ρ_m^{(t)} $$

### ⚪ k-Sparse Autoencoder

- paper：[k-Sparse Autoencoders](https://arxiv.org/abs/1312.5663)

**k稀疏自编码器**只保留隐藏层单元$z$中前$k$个最大的值，从而实现稀疏性。首先通过编码器网络获得压缩编码$z=g(x)$，对编码向量中的值进行排序，只保留前$k$个最大的值，其他神经元设置为$0$，从而构造稀疏的编码$z'$。损失函数计算为$L=\|x-f(z')\|^2_2$。反向传播时梯度只通过被激活的$k$个神经元。


# 3. 压缩自编码器
- paper：[Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](http://www.icml-2011.org/papers/455_icmlpaper.pdf)

与稀疏自编码器类似，**压缩自编码器（Contractive Autoencoder）**学习到的特征表示存在一个压缩的空间中，以增强模型的鲁棒性。

压缩自编码器在损失函数中增加一个惩罚项，以防止学习到的表示对于输入过于敏感，以此提高模型对训练数据附近的微小扰动的鲁棒性。敏感性是通过衡量编码器的输出表示$z=f(x)$相对于输入的**Jacobian**矩阵的**Frobenius**范数实现的：

$$ ||J_f(x)||_F^2 = \sum_{ij} (\frac{\partial f_j(x)}{\partial x_i})^2 $$

该惩罚项是学习到的编码相对于每个输入维度的偏导数的平方和，经验表明该惩罚项倾向于使得编码表示落入低维非线性流形中，对于正交于该流形的一些主方向保持较强的不变性。

# 4. 降噪自编码器

- paper：[Extracting and composing robust features with denoising autoencoders](https://www.researchgate.net/publication/221346269_Extracting_and_composing_robust_features_with_denoising_autoencoders)

由于自编码器旨在学习恒等函数，因此当模型的参数数量多于输入样本的数据量时，存在**过拟合**的风险，模型可能会“记住”所有样本。

为了避免过拟合并增加模型的鲁棒性，**降噪自编码器（Denoising Auto-Encoder）**在输入数据中添加随机噪声，然后训练模型以恢复原始输入（而不是有噪声的输入）。

![](https://pic.imgdb.cn/item/62a4349709475431297147ee.jpg)

对于输入样本$x$，可以根据损坏比例$μ$随机将$x$的一些维度的值设置为$0$，得到带有噪声的向量$\tilde{x}$。也可以引入Gaussian噪声，即$$\tilde{x} = x + ε$$，噪声$ε$~$$N(0,σ^2)$$。然后将$\tilde{x}$输入给自编码器得到编码$z$，并重构原始的无损输入$x'$。若用$$\mathcal{M}_D$$表示真实数据样本到噪声数据的映射，则降噪自编码器建模为：

$$ \begin{aligned} \tilde{x} &= \mathcal{M}_D(\tilde{x} | x) \\ L_{DAE} &= \sum_{n=1}^{N} {|| x^{(n)}-f(g(\tilde{x}^{(n)})) ||^2} \end{aligned} $$


# 5. 栈式自编码器
**栈式自编码器（Stacked Auto-Encoder，SAE）**是指使用逐层堆叠的方式来训练一个深层的自编码器。

栈式自编码器一般可以采用[逐层训练（Layer-Wise Training）](https://www.researchgate.net/publication/200744514_Greedy_layer-wise_training_of_deep_networks)来学习网络参数：

1. 首先训练第一个自编码器，然后保留第一个自编码器的编码器部分；
2. 把第一个自编码器的中间层作为第二个自编码器的输入层进行训练；
3. 反复地把前一个自编码器的中间层作为后一个编码器的输入层，进行迭代训练。

![](https://pic.downk.cc/item/5ee0d720c2a9a83be5d3fe2f.jpg)

# 6. 变分自编码器
上述讨论的自编码器模型是将观测数据$x$编码为固定的特征向量$z$，每一个特征向量对应特征空间中的一个离散点。**变分自编码器（Variational Auto-Encoder，VAE）**是自编码器的**Bayesian**形式，将特征向量$z$看作随机变量，使其能够覆盖特征空间中的一片区域。

若随机变量的概率分布$p$被$\theta$参数化，则可以定义先验(**prior**)概率$p_{\theta}(z)$、似然(**likelihood**)概率$p_{\theta}(x\|z)$和后验概率(**prior**)概率$p_{\theta}(z\|x)$。**VAE**的目标是构造真实数据分布$p_{\theta}(x)$，这是通过极大化对数似然实现的：

$$ \theta^* = \mathop{\arg \max}_{\theta} \sum_{n} \log p_{\theta}(x_n) $$

在实践中计算$p_{\theta}(x_n)$是相当困难的，因此引入通过$\phi$参数化的函数$q_{\phi}(z\|x)$来近似后验分布$p_{\theta}(z\|x)$。则整体结构近似于自编码器：
- 条件概率$p_{\theta}(x\|z)$定义了一个生成模型，被称作概率编码器。
- 近似函数$q_{\phi}(z\|x)$被称作概率解码器。

![](https://pic.imgdb.cn/item/62ac64e909475431296e555c.jpg)

关于变分自编码器的细节请参考[<font color=blue>VAE专题</font>](https://0809zheng.github.io/2022/04/01/vae.html)。