---
layout: post
title: '迁移学习(Transfer Learning)'
date: 2020-05-22
author: 郑之杰
cover: 'https://pic.downk.cc/item/5efc584614195aa5940f63f1.jpg'
tags: 深度学习
---

> Transfer Learning.

**迁移学习（Transfer Learning）**是指将解决某个问题时获取的知识应用在另一个不同但相关的问题中。

在迁移学习中，称模型最初解决的问题领域为**源域 (source domain)**，其中的数据集为**源域数据 (source data)**；需要迁移解决的相关问题领域为**目标域 (target domain)**，其中的数据集为**目标域数据 (target data)**。

通常源域数据数量大，而目标域数据数量较少，两者的数据分布是不同的。根据源域数据和目标域数据的标签存在情况，迁移学习可以细分为：
- 源域数据有标签，目标域数据有标签：微调(**Fine Tuning**)
- 源域数据有标签，目标域数据无标签：领域自适应(**Domain Adaptation**)、零样本学习(**Zero-Shot Learning**)
- 源域数据无标签，目标域数据有标签：**self-taught learning**
- 源域数据无标签，目标域数据无标签：**self-taught clustering**

![](https://pic.downk.cc/item/5efc584614195aa5940f63f1.jpg)

**本文目录**：
1. 微调 **Fine-tuning**
2. 领域自适应 **Domain Adaptation**
3. 零样本学习 **Zero-shot learning**

# 1. 微调 Fine-Tuning
模型**微调（Fine-tuning）**是指用带有标签的源域数据预训练模型后，再用带有标签的目标域数据微调模型。

由于目标域数据通常少于源域数据，当模型参数量较多时容易过拟合。因此微调往往只训练网络的一部分层（**Layer Transfer**），冻结其他层的参数。或者采用**保守学习 (Conservative Training)**的方式，在微调网络时，限制网络参数和预训练参数足够接近。
- 对于图像任务，浅层网络提取任务无关的低级语义信息，这部分信息可以在不同任务中共享；而深层网络提取任务相关的高级语义信息，这部分信息是不同任务所独有的。因此在微调卷积神经网络时，通常固定浅层网络，训练深层网络。
- 对于文本任务，浅层网络提取与特定文本输入有关的信息，这部分信息通常是与输入相关的；而深层网络提取与输入无关的高级语义信息，这部分信息可以被不同文本任务共享。因此在微调循环神经网络时，通常固定深层网络，训练浅层网络。


目前，在下游任务上微调**大规模预训练**模型已经成为大量 **NLP** 和 **CV** 任务常用的训练模式。然而，随着模型尺寸和任务数量越来越多，微调整个模型的方法消耗大量的储存空间，并且耗费大量训练时间。因此在微调大模型时通常采用[**参数高效 (Parameter-Efficient)**的微调方法](https://0809zheng.github.io/2023/02/02/peft.html)，即只微调部分参数，或者向预训练模型中加入少量额外的可训练参数。

# 2. 领域自适应 Domain Adaptation
**领域自适应 (Domain Adaptation)**是指通过构造合适的特征提取模型，使得源域数据和目标域数据的特征落入相同或相似的特征空间中，再用这些特征解决下游任务。本文主要讨论**homogeneous**的**Domain Adaptation**问题，即原问题和目标问题属于同一类问题（比如图像分类任务）。

![](https://pic.downk.cc/item/5efd6eec14195aa59476829f.jpg)


常用的领域自适应方法包括：
- 基于差异的方法：直接计算和减小源域和目标域数据特征向量的差异，如**Deep Domain Confusion**, **Deep Adaptation Network**, **CORAL**, **CMD**。
- 基于对抗的方法：引入域判别器并进行对抗训练，如**DANN**, **SDT**, **PixelDA**。
- 基于重构的方法：引入解码器重构输入样本，如**Domain Separation Network**。

## （1）基于差异的方法 Discrepancy-based methods
基于**差异**的方法是指，通过直接训练模型使得源域和目标域数据的特征向量足够接近:

![](https://pic.imgdb.cn/item/64bf99a21ddac507cc093894.jpg)


### ⚪ Deep Domain Confusion
- paper：[Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/abs/1412.3474)

**Deep Domain Confusion**通过训练减小**source data**的分类误差，以及**source data**和**target data**的特征向量之间的差别：

![](https://pic.imgdb.cn/item/64bf9cb91ddac507cc0e94e3.jpg)

特征向量的差距也称为**Maximum Mean Discrepancy（MMD）**，通过一阶矩（绝对值）衡量：

$$
\begin{aligned}
& \operatorname{MMD}\left(X_S, X_T\right)=  \left\|\frac{1}{\left|X_S\right|} \sum_{x_s \in X_S} \phi\left(x_s\right)-\frac{1}{\left|X_T\right|} \sum_{x_t \in X_T} \phi\left(x_t\right)\right\|
\end{aligned}
$$

### ⚪ Deep Adaptation Network
- paper：[Learning Transferrable Features with Deep Adaptation Networks](https://arxiv.org/abs/1502.02791)

**Deep Adaptation Network**在计算特征差别时，使用了网络中的多层特征：

![](https://pic.imgdb.cn/item/64bf9d661ddac507cc0ff26c.jpg)

### ⚪ Correlation Alignment (CORAL)
- paper：[Return of Frustratingly Easy Domain Adaptation](https://arxiv.org/abs/1511.05547)

**CORAL**使用二阶矩对齐源域和目标域特征向量的协方差：

![](https://pic.imgdb.cn/item/64bfa0781ddac507cc164bed.jpg)

### ⚪ Central Moment Discrepancy (CMD)
- paper：[Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning](https://arxiv.org/abs/1702.08811)

**CMD**对齐两个域分布的高阶矩，高阶矩通过样本的中心距估计：

$$
\begin{aligned}
& C M D_K(X, Y)=\frac{1}{|b-a|}\|\mathbf{E}(X)-\mathbf{E}(Y)\|_2+\sum_{k=2}^K \frac{1}{|b-a|^k}\left\|C_k(X)-C_k(Y)\right\|_2 \\
& C_k(X)=\mathbf{E}\left((x-\mathbf{E}(X))^k\right)
\end{aligned}
$$

## （2）Adversarial-based methods
基于**对抗**的方法是指，训练模型（特征提取器）得到**source domain**和**target domain**的特征向量后，再训练一个域判别器区分特征属于哪个**domain**，并采用对抗训练的方法训练整个模型：

![](https://pic.imgdb.cn/item/64bfb2851ddac507cc317fe9.jpg)


### ⚪ DANN
- paper：[Domain Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)

模型包括三部分：
1. **feature extractor**：目标是最大化label分类精度，最小化domain分类精度；
2. **label predictor**：目标是最大化label分类精度
3. **domain classifier**：目标是最大化domain分类精度。

模型在训练时，循环进行：
1. 从**source data**中抽样，通过分类损失训练**feature extractor**和**label predictor**，通过域损失训练**domain classifier**；
2. 从**target data**中抽样，通过域损失训练**domain classifier**；
3. 对**domain classifier**的梯度进行**梯度反转（gradient reversal）**，更新**feature extractor**。
   
![](https://pic.downk.cc/item/5efd750814195aa59478e6a0.jpg)

### ⚪ Simultaneous Deep Transfer (SDT)
- paper：[Simultaneous Deep Transfer Across Domains and Tasks](https://arxiv.org/abs/1510.02192)

该方法通过对抗学习最小化两个域之间特征的距离；同时考虑了类别之间的关联：使用**source domain**生成每个类别的**soft label**。该模型的损失函数包括：
1. **Classification Loss**：最终的分类损失；
2. **Domain Confusion Loss**：交替训练，一方面希望训练一个好的**domain**分类器，另一方面希望特征骗过**domain**分类器；
3. **Label Correlation Loss**：希望**target domain**的特征含有更多信息，采用引入温度**T**的**soft**分类。

![](https://pic.imgdb.cn/item/64bfb4051ddac507cc3397db.jpg)

### ⚪ PixelDA
- paper：[<font color=blue>Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks</font>](https://0809zheng.github.io/2022/03/15/pixelda.html)

**PixelDA**的想法是先训练**GAN**，喂入**source data**生成**target data**；再用**source data**和其对应生成的**target data**作为同一类训练分类器。

![](https://pic.imgdb.cn/item/64bfb9461ddac507cc3b6859.jpg)

## （3）Reconstruction-based methods
基于**重构**的方法是指，要求训练模型得到**source domain**和**target domain**的特征向量足够接近，并且通过解码器能够恢复各自的图像：

![](https://pic.imgdb.cn/item/64bfb9961ddac507cc3bdcb1.jpg)

### ⚪ Domain Separation Networks

模型训练两个私有的编码器和一个共享参数的编码器，**Private encoder**提取**domain-specific**特征；**Shared encoder**提取**domain-invariant**特征。

对于每个**domain**，将提取的两种特征结合起来通过解码器恢复图像；用**domain-invariant**特征解决下游任务。

![](https://pic.downk.cc/item/5efd793614195aa5947a72eb.jpg)



# 3. 零样本学习 Zero-shot Learning

## ⚪ Attribute Embedding

人为构造一系列**属性attribute**，将**source data**的每一个类别标记为一个属性向量：

![](https://pic.downk.cc/item/5efc62c614195aa594130c93.jpg)

属性向量生成可以用一个神经网络实现，即把数据$x_n$的类别标签$y_n$喂入网络得到属性向量$g(y_n)$。

通过训练网络把**source data**转化成对应的属性向量，假设网络的预测结果为$f(x_n)$，对应的属性向量为$g(y_n)$，希望样本$x_n$经过网络得到的输出和其类别对应的属性向量为$g(y_n)$足够接近，和其余类别的属性向量相差很大，相差一个超参数$k$：

$$ f(x_n)g(y_n) - \mathop{\max}_{m≠n} f(x_n)g(y_m) > k $$

由此定义**zero loss**：

$$ loss = max(0,k-f(x_n)g(y_n) + \mathop{\max}_{m≠n} f(x_n)g(y_m)) $$

两个网络$f$和$g$可以一起训练：

$$ f^*,g^* = \mathop{\arg \min}_{f,g} \sum_{n}^{} {max(0,k-f(x_n)g(y_n) + \mathop{\max}_{m≠n} f(x_n)g(y_m))} $$

训练好网络，进行迁移时，将没有标签的**target data**喂入网络得到其对应的属性向量，与已有类别的属性向量进行比较，按照最相近的结果进行分类。
