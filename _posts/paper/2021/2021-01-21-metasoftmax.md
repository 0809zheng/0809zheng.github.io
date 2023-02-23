---
layout: post
title: 'Balanced Meta-Softmax for Long-Tailed Visual Recognition'
date: 2021-01-21
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/62611470239250f7c50a364f.jpg'
tags: 论文阅读
---

> BALMS: 长尾图像分类中的平衡元Softmax函数.

- paper：[Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://arxiv.org/abs/2007.10740v3)

长尾分布面临的主要问题：
1. **标签分布偏移(Label Distribution Shift)**：长尾问题中的训练集通常是类别不平衡的，而此时及是类别平衡的，造成类别不平衡的训练数据集分布与类别平衡的测试优化指标之间的不匹配性。比如平均精度**mAP**鼓励模型在测试集上所有类别上的平均精度最小，导致训练与测试时标签分布不同。
2. **过平衡(Over-balance)**：由于尾部类别在训练过程中出现频率较低，因此无法提供足够的优化梯度。因此即使设置一个较好的优化目标，也很难保证模型收敛到全局最优。如果同时使用类别平衡采样与优化目标则会导致过平衡问题。

本文作者针对长尾分布问题提出了**Balanced Meta-Softmax (BALMS)**方法。针对**Softmax**函数处理标签分布偏移问题时存在估计偏差，提出**Balanced Softmax**避免偏差；针对类别平衡采样在与**Balanced Softmax**一起使用时导致过平衡问题，提出元采样器**Meta Sampler**通过元学习方法学习最优重采样策略。

# 1. 平衡Softmax损失 Balanced Softmax
$k$类别分类的输出通过**Softmax**函数建模为多项分布：

$$ \phi=\phi_1^{\Bbb{I}\{y=1\}}\phi_2^{\Bbb{I}\{y=2\}} \cdots \phi_k^{\Bbb{I}\{y=k\}} \\ \phi_j = \frac{e^{\eta_j}}{\sum_{i=1}^{k}e^{\eta_i}},  \sum_{j=1}^{k}\phi_j=1 $$

根据贝叶斯推断，第$j$个类别的条件概率$\phi_j$可以写作：

$$ \phi_j = p(y=j|x) = \frac{p(x,y=j)}{p(x)} = \frac{p(x|y=j)p(y=j)}{p(x)} $$

训练集和测试集的标签分布$p(y=j)$存在差异，因此**Softmax**函数构造的条件概率$\phi$是真实条件概率的有偏估计。

为了消除训练和测试后验分布的差异，引入**Balanced Softmax**，使用模型的**logits**输出分别参数化训练概率$\hat{\phi}$和测试概率$\phi$。

若测试时类别分布是平衡的，即$p(y=j)=\frac{1}{k}$，则测试概率$\phi$表示为：

$$ \phi_j =\frac{p(x|y=j)}{p(x)}\frac{1}{k} $$

而训练时类别分布是不平衡的，记第$j$个类别的样本数量为$n_j$，则训练概率$\hat{\phi}$表示为：

$$ \hat{\phi}_j =\frac{p(x|y=j)}{p(x)}\frac{n_j}{\sum_{i=1}^{k}n_i} $$

如果$\phi$是通过**logits**输出$\eta$和**Softmax**函数表示的，则$\hat{\phi}$应表示为：

$$ \hat{\phi}_j = \frac{\eta_je^{\eta_j}}{\sum_{i=1}^{k}\eta_ie^{\eta_i}} $$

上式称为**Balanced Softmax**，对应的损失函数为：

$$ l = -\log \hat{\phi}_y = -\log  \frac{\eta_ye^{\eta_y}}{\sum_{i=1}^{k}\eta_ie^{\eta_i}} $$

# 2. 元采样器 Meta Sampler

最近的一些工作表明使用**Softmax**函数学习的全局最小值与小批量采样过程无关，因此采用合适的重采样策略可以模拟类别平衡的数据分布。作者通过实验发现同时使用类别平衡采样**CBS**与**Balanced Softmax**使得表现变差。

在理想的优化过程中，每个类别的梯度的权重应和该类别样本数量成反比，然而应用**Balanced Softmax**后权重和该类别样本数量成平方反比，造成过平衡现象。

下图展示了一个类别不平衡的二维数据三分类问题的可视化，同时应用类别平衡采样**CBS**与**Balanced Softmax**后，优化过程将过度地被尾部类别主导。

![](https://pic.imgdb.cn/item/626135e4239250f7c55882f7.jpg)

为了解决过平衡问题，引入**元采样器Meta Sampler**，一种基于元学习的类别平衡采样方法。**Meta Sampler**自动学习不同类别样本的最佳采样率，从而更好地配合**Balanced Softmax**的使用。

**Meta Sampler**采用双级元学习策略：在内循环中更新采样分布$\pi_{\psi}$的参数$\psi$，在外循环中更新网络参数$\theta$。从训练集$D_{train}$中采样一个类别平衡的元数据集$D_{meta}$，通过内循环寻找使得网络$\theta$在$D_{meta}$上表现最好的采样分布$\pi_{\psi}$。问题建模如下：

$$ \pi^*_{\psi} = \mathop{\arg \min}_{\psi} L_{D_{meta}}(\theta^*(\pi_{\psi})) \\ \text{s.t.  } \theta^*(\pi_{\psi})=\mathop{\arg \min}_{\theta} \hat{L}_{D_{q(x,y;\pi_{\psi})}}(\theta) $$

其中$L$是标准的**Softmax**函数，$\hat{L}$是**Balanced Softmax**函数，$D_{q(x,y;\pi_{\psi})}$是对训练集根据类别采样率$\pi_{\psi}$进行采样。

元学习优化过程如下：
1. 根据类别采样率$\pi_{\psi}$采样一个小批量$B_{\psi}$，通过梯度下降更新代理模型的参数：$$\tilde{\theta} \gets \theta - \nabla_{\theta}\hat{L}_{B_{\psi}}(\theta)$$
2. 计算元数据集$D_{meta}$上代理模型的损失$L_{D_{meta}}(\tilde{\theta})$，通过梯度下降更新采样率参数：$$\psi \gets \psi - \nabla_{\psi}L_{D_{meta}}(\tilde{\theta})$$
3. 更新模型参数：$$\theta \gets \theta - \nabla_{\theta}\hat{L}_{B_{\psi}}(\theta)$$

从离散分布中采样是不可导的，为了保证采样过程能够端到端训练，在构造小批量$B_{\psi}$时应用**Gumbel-Softmax**重参数化技巧。

# 3. 实验分析

下图展示了模型在测试集中每个类别的累加预测分数，即模型预测的标签分布。理想情况下，模型在类别平衡的测试集上预测的标签分布应该是平衡的。图中类别按照出现频率降序排列。**Softmax**函数明显地偏向于头部类别，而**Balanced Softmax**实现了相对平衡的预测类别分布。

![](https://pic.imgdb.cn/item/62614259239250f7c578df75.jpg)

当直接结合**Balanced Softmax**与**CBS**时产生过平衡问题，预测类别分布明显地偏向于尾部类别。而**Balanced Softmax**与**Meta Sampler**的组合取得了较为均衡的标签分布。

![](https://pic.imgdb.cn/item/626142c4239250f7c579c703.jpg)

作者在图像分类（**CIFAR-10/100-LT，ImageNet-LT，Places-LT**）与实例分割（**LVIS-v0.5**）任务上分别进行了实验验证。**BALMS**方法取得最好的表现。

![](https://pic.imgdb.cn/item/6261432c239250f7c57aa8d1.jpg)
