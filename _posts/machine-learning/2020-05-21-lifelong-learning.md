---
layout: post
title: '终身学习(Lifelong Learning)'
date: 2020-05-21
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eec25c514195aa59496e904.jpg'
tags: 机器学习
---

> Lifelong Learning.

**终身学习（lifelong learning）**也叫**持续学习（Continuous Learning, Never Ending Learning）**或**增量学习（Incremental Learning）**；是指把之前任务训练的模型应用到新任务中，在这些任务上都能获得不错的表现。

终身学习与迁移学习的区别在于：
- **迁移学习**是指在一个任务上训练好的模型应用到新任务上，微调后不能保证模型在之前的任务上还有较好的表现；
- **终身学习**是指把模型应用到新任务后，对之前的任务和新任务都能有较好的表现。

# 1. 终身学习算法的性能评估
在评估终身学习算法的性能时，通常的做法是：
- 对于$T$个任务，模型随机初始化后依次测试在$T$个任务上的表现；
- 在任务$1$上进行训练，之后依次测试在$T$个任务上的表现；
- 在任务$2$上进行训练，之后依次测试在$T$个任务上的表现；
- ......
- 在任务$T$上进行训练，之后依次测试在$T$个任务上的表现。

记录上述表现，可以列出矩阵：

![](https://pic.downk.cc/item/5eec25e714195aa594970359.jpg)

其中$R_{i,j}$表示模型依次在前$i$个任务上训练后在第$j$个任务上的表现。
- 当$i>j$时，衡量当用前$i$个任务训练后，对前面的第$j$个任务是否会变差
- 当$i<j$时，衡量当用前$i$个任务训练后，对后面的第$j$个任务是否有帮助

衡量算法性能的指标包括：
- **accuracy**：模型经过$T$个任务训练后，在这些任务上的平均表现（如下图中红色框所示）

$$ accuracy = \frac{1}{T}\sum_{i=1}^{T} {R_{T,i}} $$

- **backward transfer（BWT）**：模型经过$T$个任务训练后在第$i$个任务上的表现减去刚在第$i$个任务上训练后的表现，通常是**负值**（如下图中蓝色框所示）

$$ BWT = \frac{1}{T-1}\sum_{i=1}^{T-1} {R_{T,i}-R_{i,i}} $$

- **forward transfer（FWT）**：模型经过前$i-1$个任务训练后在第$i$个任务上的表现减去未训练时在第$i$个任务上的表现，通常是**正值**（如下图中绿色框所示）

$$ FWT = \frac{1}{T-1}\sum_{i=2}^{T} {R_{i-1,i}-R_{0,i}} $$

![](https://pic.downk.cc/item/5eec2ac014195aa5949b77b2.jpg)

# 2. 一些终身学习算法

## ⚪ Multi-task training
[**多任务学习（Multi-task training）**](https://0809zheng.github.io/2021/08/28/MTL.html)可以用来解决终身学习问题。每当要解决一个新任务时，模型在所有之前的任务数据和新任务数据上进行学习；多任务学习的结果常作为终身学习的**上界（upper bound）**。

![](https://pic.downk.cc/item/5eec2b6f14195aa5949c2194.jpg)

多任务学习的主要问题在于：
- 使用所有累计的数据训练，计算量大；
- 需要存储之前所有的数据，占用内存。

一些解决方法：
- [Generating Data](https://arxiv.org/abs/1705.08690)：训练一个数据生成器（如GAN），保存生成器而不是原数据
- Adding New Classes：[Learning without forgetting](https://arxiv.org/abs/1606.09282)、[iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)

## ⚪ Elastic Weight Consolidation
- paper：[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)

**Elastic Weight Consolidation (EWC)**是指在一个任务上训练得到参数$θ^b$后，对于一个新的任务，在损失函数上加上正则化项：

$$ L'(θ) = L(θ) + λ\sum_{i}^{} {b_i(θ_i-θ_i^b)^2} $$

其中参数$b_i$用来衡量$θ_i^b$对上一个任务的重要性，
- 若$b_i=0$，则对$θ_i$没有约束；
- 若$b_i→∞$，则$θ_i$应尽可能保持与$θ_i^b相等。

参数$b$的选择可以使用上一个任务损失函数的二阶微分。
- 当其二阶微分较小时，表示参数的变换对损失影响不大，因此对应的可选小一些；
- 当其二阶微分较大时，表示参数的变换对损失影响较大，因此对应的可选大一些。

![](https://pic.downk.cc/item/5eec302a14195aa594a0c98d.jpg)

![](https://pic.downk.cc/item/5eec308514195aa594a1359d.jpg)

相关阅读：
- [Synaptic Intelligence](https://arxiv.org/abs/1703.04200)
- [Memory Aware Synapses](https://arxiv.org/abs/1711.09601)：不需要标签数据

## ⚪ Gradient Episodic Memory
- paper：[Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)

**Gradient Episodic Memory (GEM)**是指每次训练一个新的任务时，用之前的任务在模型当前参数下的梯度$g^1$、$g^2$...来修正参数$θ$的梯度更新方向$g$:

![](https://pic.downk.cc/item/5eec5ceb14195aa594cce215.jpg)

- 若当前任务的梯度方向与之前任务的梯度方向都指向同一侧（表明参数更新对之前的任务也是有帮助的），则直接使用当前梯度更新参数；
- 否则将梯度$g$修正到$g'$，且满足$g'$与之前任务的梯度方向指向同一侧并且$g'$与$g$尽可能接近。

该方法也需要存储之前的数据。

## ⚪ Progressive Neural Networks
- paper：[Progressive Neural Networks](https://arxiv.org/abs/1606.04671)

**Progressive Neural Networks**的思想是，每处理一个新的任务，就训练一个新的神经网络，且固定之前的神经网络模型使其可以处理之前的任务；新的神经网络使用之前的网络特征进行训练。

![](https://pic.downk.cc/item/5eec5d5414195aa594cd878a.jpg)

## ⚪ Net2Net
- paper：[Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641)

**Net2Net**是训练一个神经网络；每当处理一个新的任务，当前网络表现不够好时，就为网络增加神经元，使其当前参数等效于之前的网络，再在这个更大的模型上训练。

如下图，左边隐藏层具有两个神经元，右图为隐藏层增加了一个神经元，通过分配参数使其与左边等效。为参数加上一些噪声，再进行训练。

![](https://pic.downk.cc/item/5eec5eb314195aa594cfb073.jpg)

## ⚪ Curriculum Learning
**Curriculum Learning**旨在为终身学习选择一个合适的学习顺序。[taskonomy](http://taskonomy.stanford.edu/#abstract)为计算机视觉的各项任务分析了合适的学习顺序。

## ⚪ SupSup
- paper：[<font color=Blue>Supermasks in Superposition</font>](https://0809zheng.github.io/2021/06/30/supsup.html)

训练过程**supermask**：随机初始化一个网络，网络的参数值$W$在训练中并不改变。引入一个**mask**矩阵$M$，随机对网络的每个连接赋予$0/1$值，从而构造一个子网络$W \otimes M$。模型在训练时针对某一特定任务上的数据集训练一个对应的**mask**。因此对于所有任务共享一个主网络，在解决每一个任务时根据其相应的**mask**使用一个子网络。

![](https://pic.imgdb.cn/item/60dc104f5132923bf8a3b49d.jpg)

推断过程**superposition**：若已知测试图像$x$所属的数据集$i$，便可以直接选用对应任务的子网络$W \otimes M^i$进行测试:

$$ p=f(x,W \otimes M^i) $$

当测试图像所述任务未知时，为每一个子网络$i$引入一个系数$\alpha_i \in \[0，1\]$，表示该网络对于该图像的适合程度。若共有$k$个任务，则初始化为$\alpha_i =\frac{1}{k}$。则此时模型的预测结果为：

$$ p(\alpha)=f(x,W \otimes (\sum_{i=1}^{k} \alpha_i M^i)) $$

期望预测结果的熵$\mathcal{H}$最小，通过梯度下降算法实现：

$$ \alpha ← \alpha - \eta \nabla_{\alpha} \mathcal{H}(p(\alpha)) $$

![](https://pic.imgdb.cn/item/60dc106c5132923bf8a4b5f6.jpg)
