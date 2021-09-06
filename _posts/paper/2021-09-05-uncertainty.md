---
layout: post
title: 'Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics'
date: 2021-09-05
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6131858844eaada73984a589.jpg'
tags: 论文阅读
---

> 使用同方差不确定性调整多任务损失权重.

- paper：Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
- arXiv：[link](https://arxiv.org/abs/1705.07115v3)

多任务学习的优化目标是多个任务目标的组合，通常使用对各个目标损失加权求和的方式构造总损失：

$$ L_{total} = \sum_{i}^{} w_iL_i $$

然而任务权重$w_i$通常需要人工选择，且不同权重的选择对结果的影响是显著的。下图展示了作者同时学习一个分类任务和一个回归任务的结果。实验表明同时学习两个任务有时能获得比单独学习某个任务更好的表现，但是合适的权重很难人工选择。

![](https://pic.imgdb.cn/item/61318aae44eaada7398e6ccc.jpg)

作者从**不确定性**(**uncertainty**)的角度出发，设计了一种自动设置任务权重的方法。在深度学习建模中，通常会引入两种形式的不确定性：
- **认知不确定性**(**epistemic uncertainty**)：模型本身的不确定性，表示由于缺乏训练数据导致模型认知不足，可以通过增加训练数据来降低。
- **偶然不确定性**(**aleatoric uncertainty**)：表示由数据无法解释的信息导致的不确定性，可以通过增强观察所有解释变量的能力来降低。

偶然不确定性又可以分成两个子类：
- **异方差不确定性**(**heteroscedastic uncertainty**)：又叫**数据依赖**(**data-dependent**)的不确定性，依赖于输入数据的不确定性，表现在模型的输出上。
- **同方差不确定性**(**homoscedastic uncertainty**)：又叫**任务依赖**(**task-dependent**)的不确定性，不依赖于输入数据，而是依赖于任务的不确定性，该不确定性对所有输入数据保持不变，但在不同任务之间变化。

对于多任务学习，通常是对于同样的输入数据集，产生不同任务的输出结果。如本文中作者研究的是给定一张单目图像，同时进行语义分割、实例分割和深度估计任务。因此可以用同方差不确定性作为多任务学习中不同人物的权重指标。

![](https://pic.imgdb.cn/item/6131e4c244eaada7397b379e.jpg)

对于具有参数$W$的模型$f^W(x)$，下面从回归任务和分类任务的角度分别推导考虑同方差不确定性时的损失函数。

对于回归任务，将模型输出建模为具有观测噪声$\sigma$的**Gaussian**分布：

$$ p(y|f^W(x)) = \mathcal{N}(f^W(x),\sigma^2) $$

在极大似然推断中，最大化概率模型的对数似然，写作：

$$ \log p(y|f^W(x)) = \log \mathcal{N}(f^W(x),\sigma^2) \\ = \log (\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{||y-f^W(x)||^2}{2\sigma^2}} ) \\ ∝ -\frac{1}{2\sigma^2}||y-f^W(x)||^2-\log \sigma $$

假设模型同时预测两个回归任务，则可以建模为：

$$ p(y_1,y_2|f^W(x)) = p(y_1|f^W(x))\cdot p(y_2|f^W(x)) \\ =  \mathcal{N}(y_1;f^W(x),\sigma_1^2) \cdot \mathcal{N}(y_2;f^W(x),\sigma_2^2) $$

最大化上述概率模型的对数似然，等价于最小化以下目标函数：

$$ \mathcal{L}(W,\sigma_1,\sigma_2) = -\log(p(y_1,y_2|f^W(x))) \\ ∝ \frac{1}{2\sigma_1^2}||y-f^W(x)||^2+\log \sigma_1+\frac{1}{2\sigma_2^2}||y-f^W(x)||^2+\log \sigma_2 \\= \frac{1}{2\sigma_1^2}\mathcal{L}_1(W) +\log \sigma_1+\frac{1}{2\sigma_2^2}\mathcal{L}_2(W) +\log \sigma_2 $$

注意到上式中$\sigma$相当于任务损失的相对权重。噪声$\sigma$越小，表明同方差不确定度越小，则任务损失权重较高。$\log \sigma$相当于正则化项，防止噪声$\sigma$过大。

对于分类任务，输出通过经过**softmax**函数进行归一化：

$$ p(y|f^W(x)) = \text{softmax}(f^W(x))) $$

为了便于讨论，采用一种缩放的形式，引入温度系数$\sigma$用于衡量分布的平坦程度，将分类模型建模成**Boltzmann**分布(也称**Gibbs**分布)：

$$ p(y|f^W(x)) = \text{softmax}(\frac{1}{\sigma^2}f^W(x))) $$

上述分类模型的对数似然表示为：

$$ \log p(y|f^W(x)) = \log \text{softmax}(\frac{1}{\sigma^2}f^W(x))) \\ = \log \frac{\exp(\frac{1}{\sigma^2}f_c^W(x))}{\sum_{c}^{}\exp(\frac{1}{\sigma^2}f_c^W(x))} \\ = \frac{1}{\sigma^2}f_c^W(x)-\log \sum_{c}^{}\exp(\frac{1}{\sigma^2}f_c^W(x)) \\ = \frac{1}{\sigma^2}(f_c^W(x)-\log \sum_{c}^{}\exp(f_c^W(x))) + \frac{1}{\sigma^2}\log \sum_{c}^{}\exp(f_c^W(x))-\log \sum_{c}^{}\exp(\frac{1}{\sigma^2}f_c^W(x)) \\ = \frac{1}{\sigma^2}(\log \frac{\exp(f_c^W(x))}{\sum_{c}^{}\exp(f_c^W(x))}) + \log (\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma^2}}-\log \sum_{c}^{}\exp(\frac{1}{\sigma^2}f_c^W(x))  \\ = \frac{1}{\sigma^2}(\log \frac{\exp(f_c^W(x))}{\sum_{c}^{}\exp(f_c^W(x))}) + \log (\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma^2}}-\log \sum_{c}^{}\exp(\frac{1}{\sigma^2}f_c^W(x)) \\ = \frac{1}{\sigma^2}(\log \text{softmax}(f^W(x))) + \log \frac{(\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma^2}}}{ \sum_{c}^{}\exp(\frac{1}{\sigma^2}f_c^W(x))}  $$


如果模型同时执行回归($y_1$是连续输出)和分类($y_2$是离散输出)任务，则目标函数可以表示为：

$$ \mathcal{L}(W,\sigma_1,\sigma_2) = -\log(p(y_1,y_2=c|f^W(x))) \\ = -\log \mathcal{N}(y_1;f^W(x),\sigma_1^2)\cdot \text{softmax}(y_2=c;f^W(x),\sigma_2^2) \\ =  \frac{1}{2\sigma_1^2}||y-f^W(x)||^2+\log \sigma_1-\log p(y_2=c;f^W(x),\sigma_2^2) \\ = \frac{1}{2\sigma_1^2}||y-f^W(x)||^2+\log \sigma_1 - \frac{1}{\sigma_2^2}(\log \text{softmax}(f^W(x))) - \log \frac{(\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma_2^2}}}{ \sum_{c}^{}\exp(\frac{1}{\sigma_2^2}f_c^W(x))} \\ = \frac{1}{2\sigma_1^2}||y-f^W(x)||^2+\log \sigma_1 + \frac{1}{\sigma_2^2}(-\log \text{softmax}(f^W(x))) + \log \frac{ \sum_{c}^{}\exp(\frac{1}{\sigma_2^2}f_c^W(x))}{(\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma_2^2}}} $$

若记回归损失$$\mathcal{L}_1(W)=\|y-f^W(x)\|^2$$，分类损失$\mathcal{L}_2(W)=-\log \text{softmax}(f^W(x))$，则总目标函数表示为：

$$ \mathcal{L}(W,\sigma_1,\sigma_2) = \frac{1}{2\sigma_1^2}\mathcal{L}_1(W)+\log \sigma_1+\frac{1}{\sigma_2^2}\mathcal{L}_2(W)+\log \frac{\sum_{c}^{}\exp(\frac{1}{\sigma_2^2}f_c^W(x))}{(\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma_2^2}}} $$

做近似$\frac{1}{\sigma_2}\sum_{c}^{}\exp(\frac{1}{\sigma_2^2}f_c^W(x))≈(\sum_{c}^{}\exp(f_c^W(x)))^{\frac{1}{\sigma_2^2}}$，则目标函数近似为：

$$ \mathcal{L}(W,\sigma_1,\sigma_2) = \frac{1}{2\sigma_1^2}\mathcal{L}_1(W)+\log \sigma_1+\frac{1}{\sigma_2^2}\mathcal{L}_2(W)+\log \sigma_2 $$

综上所述，可以将多任务损失建模为以下形式，对于第$k$个任务，引入观测噪声$\sigma_k$，则损失函数表现为：

$$ \mathcal{L}(W,\sigma_1,...,\sigma_K) = \sum_{k=1}^{K}\frac{1}{2\sigma_k^2}\mathcal{L}_k(W)+\log \sigma_k $$

在实际实现时，为了提升数值稳定性，回归方差的对数$s=\log \sigma^2$，一方面是为了防止分母为$0$，另一方面是为了回归无范围约束的结果。

作者通过实验验证，所提自动设置损失权重的方法能够使单个任务上具有最好的表现：

![](https://pic.imgdb.cn/item/61320e4644eaada739c071ef.jpg)