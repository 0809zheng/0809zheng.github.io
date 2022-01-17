---
layout: post
title: 'Transfer Learning：迁移学习'
date: 2020-05-22
author: 郑之杰
cover: 'https://pic.downk.cc/item/5efc584614195aa5940f63f1.jpg'
tags: 机器学习
---

> Transfer Learning.

- **迁移学习（Transfer Learning）**是指将解决某个问题获取的知识应用在另一个不同但相关的问题中。
- Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a diferent but related problem.

在迁移学习中，称模型最初解决的问题领域为**source domain**，其中的数据集为**source data**；需要迁移解决的相关问题领域为**target domain**，其中的数据集为**target data**。

通常**source data**数量大，而**target data**数量较少，两者的数据分布是不同的。

迁移学习的不同分类：
- **source data**有标签，**target data**有标签：**fine tuning**、**multitask learning**
- **source data**有标签，**target data**无标签：**Domain Adaptation**、**zero-shot learning**
- **source data**无标签，**target data**有标签：**self-taught learning**
- **source data**无标签，**target data**无标签：**self-taught clustering**

![](https://pic.downk.cc/item/5efc584614195aa5940f63f1.jpg)

**本文目录**：
1. Fine-tuning
2. Multitask Learning
3. Domain Adaptation
4. Zero-shot learning

# 1. Fine-Tuning
**模型微调（Fine-tuning）**是指用**source data**训练模型，再用**target data**微调模型。

当**target data**只有几个样本时，也叫**one-shot learning**。

缺点：**target data**数量少，容易过拟合。

一些解决措施：
- **保守学习 Conservative Training**：在微调网络时，限制网络参数和预训练参数足够接近；
- **Layer Transfer**：只微调网络的一部分层，冻结其他层的参数。对于图像任务，固定浅层网络，训练深层网络；对于语音任务，固定深层网络，训练浅层网络。

# 2. Multitask Learning
**多任务学习（Multitask Learning）**是指同时学习**source domain**和**target domain**的任务，不同的任务可以共享一部分模型结构。

![](https://pic.downk.cc/item/5efc5e3014195aa594117f34.jpg)

模型也可采用**渐进神经网络（Progressive Neural Networks）**结构，每次训练一个新的模型解决新的问题，也会用到之前训练好的模型结构：

![](https://pic.downk.cc/item/5efc5ec614195aa59411b022.jpg)

# 3. Domain Adaptation
**Domain Adaptation**是指通过构造合适的特征提取模型，使得**source domain**和**target domain**的特征在相同的特征空间中，再用这些特征解决下游任务。

![](https://pic.downk.cc/item/5efd6eec14195aa59476829f.jpg)

本文主要讨论**homogeneous**的**Domain Adaptation**问题，即原问题和目标问题属于同一类问题（以图像分类为例）。

常用方法：
1. Discrepancy-based methods
2. Adversarial-based methods
3. Reconstruction-based methods

## （1）Discrepancy-based methods
基于差异的方法是指，通过直接训练模型使得**source domain**和**target domain**特征向量足够接近:

![](https://pic.downk.cc/item/5efd6f3414195aa59476a72f.jpg)

常用的方法包括：
1. Deep Domain Confusion (MMD) 
2. Deep Adaptation Networks 
3. CORAL, CMD

### Deep Domain Confusion
- paper：Deep Domain Confusion: Maximizing for Domain Invariance

通过训练，减小**source data**的分类误差，以及**source data**和**target data**的特征向量之间的差别：

![](https://pic.downk.cc/item/5efd709c14195aa5947736c9.jpg)

特征向量的差距也称为**Maximum Mean Discrepancy（MMD）**。

### Deep Adaptation Networks
- paper：Learning Transferrable Features with Deep Adaptation Networks

**Deep Adaptation Networks**在计算特征差别时，使用了多层特征：

![](https://pic.downk.cc/item/5efd712e14195aa594777261.jpg)

### CORAL, CMD
之前计算特征向量的差别使用的是一阶矩（绝对值），**CORAL**使用二阶矩，**CMD**使用高阶矩。

## （2）Adversarial-based methods
基于对抗的方法是指，训练模型得到**source domain**和**target domain**的特征向量，再训练一个**domain classifier**区分特征属于哪个domain，采用对抗的方法训练整个模型：

![](https://pic.downk.cc/item/5efd724914195aa59477e8fd.jpg)

常用的方法包括：
1. Simultaneous Deep Transfer Across Domains and Tasks 
2. DANN
3. PixelDA

### Simultaneous Deep Transfer Across Domains and Tasks
- paper：Simultaneous Deep Transfer Across Domains and Tasks

![](https://pic.downk.cc/item/5efd732314195aa594783be6.jpg)

该模型的损失函数包括：
1. Classification Loss：最终的分类损失；
2. Domain Confusion Loss：交替训练，一方面希望训练一个好的domain分类器，另一方面希望特征骗过domain分类器；
3. Label Correlation Loss：希望**target domain**的特征含有更多信息，采用引入温度T的soft分类。

### DANN
- paper：Domain Adversarial Training of Neural Networks

![](https://pic.downk.cc/item/5efd750814195aa59478e6a0.jpg)

模型包括三部分：
1. **feature extractor**：目标是最大化label分类精度，最小化domain分类精度；
2. **label predictor**：目标是最大化label分类精度
3. **domain classifier**：目标是最大化domain分类精度。

模型在训练时，循环进行：
1. 从**source data**中抽样，训练**label predictor**和**domain classifier**；
2. 从**target data**中抽样，训练**domain classifier**；
3. 对**domain classifier**的梯度进行**梯度反转（gradient reversal）**，更新**feature extractor**。

### PixelDA
- paper：Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks

**PixelDA**的想法是先训练GAN，喂入**source data**生成**target data**，再用**source data**和其对应生成的**target data**作为同一类训练分类器。

![](https://pic.downk.cc/item/5efd780214195aa5947a0498.jpg)

## （3）Reconstruction-based methods
基于重构的方法是指，要求训练模型得到**source domain**和**target domain**的特征向量足够接近，并且通过解码器能够恢复各自的图像：

![](https://pic.downk.cc/item/5efd78ea14195aa5947a5779.jpg)

常用的方法包括：
1. Domain Separation Networks

### Domain Separation Networks

![](https://pic.downk.cc/item/5efd793614195aa5947a72eb.jpg)

模型训练两个私有的编码器和两个共享参数的编码器，**Private encoder**提取**domain-specific**特征；**Shared encoder**提取**domain-invariant**特征。

对于每个domain，将提取的两种特征结合起来通过解码器恢复图像；用**domain-invariant**特征解决下游任务。


# 4. Zero-shot Learning
下面以图像分类任务为例介绍**Zero-shot Learning**方法。

### Attribute embedding

人为构造一系列**属性attribute**，将**source data**的每一个类别标记为一个属性向量：

![](https://pic.downk.cc/item/5efc62c614195aa594130c93.jpg)

属性向量生成可以用一个神经网络实现，即把数据$x_n$的类别标签$y_n$喂入网络得到属性向量$g(y_n)$。

### zero loss

通过训练网络把**source data**转化成对应的属性向量，假设网络的预测结果为$f(x_n)$，对应的属性向量为$g(y_n)$，希望样本$x_n$经过网络得到的输出和其类别对应的属性向量为$g(y_n)$足够接近，和其余类别的属性向量相差很大，相差一个超参数$k$：

$$ f(x_n)g(y_n) - \mathop{\max}_{m≠n} f(x_n)g(y_m) > k $$

由此定义**zero loss**：

$$ loss = max(0,k-f(x_n)g(y_n) + \mathop{\max}_{m≠n} f(x_n)g(y_m)) $$

两个网络$f$和$g$可以一起训练：

$$ f^*,g^* = \mathop{\arg \min}_{f,g} \sum_{n}^{} {max(0,k-f(x_n)g(y_n) + \mathop{\max}_{m≠n} f(x_n)g(y_m))} $$

训练好网络，进行迁移时，将没有标签的**target data**喂入网络得到其对应的属性向量，与已有类别的属性向量进行比较，按照最相近的结果进行分类。
