---
layout: post
title: '深度学习中的Optimization'
date: 2020-03-02
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e7ee115504f4bcb04298afb.png'
tags: 深度学习
---

> Optimization in Deep Learning.

1. Background
2. Gradient Descent
3. Momentum
4. Nesterov Momentum
5. AdaGrad
6. RMSprop
7. AdaDelta
8. Adam
9. Nadam
10. Gradient Clipping
11. Learning Rate Strategy
12. Hyperparameter Optimization


# 1. Background
机器学习（尤其是监督学习）的问题包括**优化**和**泛化**问题。

**优化**是指在已有的数据集上实现最小的训练误差（training error），**泛化**是指在未训练的数据集（通常假设与训练集同分布）上实现最小的泛化误差（generalize error）。深度神经网络具有很强的拟合能力，在训练数据集上错误率通常较低，但是容易**过拟合（overfitting）**。

深度神经网络是一个高度非线性模型，网络优化存在以下困难：

**（1）网络结构多样性带来的困难**

由于网络结构的多样性，很难找到通用的优化方法；网络含有的参数和超参数比较多，优化困难。

**（2）非凸优化带来的困难**

深度网络的优化函数是高维空间的非凸函数，目标是**全局最小点（global minima）**，但存在梯度为零的**局部极小点（local minima）**和鞍点**（saddle point）**。在高维空间中，局部极小点要求在每一维度上都是极小点，这种概率比较低。[大部分梯度为零的点都是鞍点](https://arxiv.org/abs/1406.2572)，如下图所示。

注：鞍点是指梯度为0但是Hessian矩阵不是半正定的点。
![](https://pic.downk.cc/item/5e7ee39e504f4bcb042baa2a.png)

[最近的研究表明](https://arxiv.org/abs/1412.0233v3)，在非常大的神经网络中，大部分局部极小点和全局最小点是近似的；在训练神经网络时，通常没有必要找全局最小点，这反而可能过拟合。

有研究者分析了损失函数的[landspace](https://arxiv.org/abs/1712.09913)，发现由于深度网络的参数非常多，且有一定的冗余性，使得每个参数对最终损失的影响非常小。这使得损失函数在局部极小点附近通常是一个平坦的区域，即平坦最小值（flat minima）而不是尖锐最小值（sharp minima），如下图所示。这使得模型收敛于局部极小值时更加robust，即微小的参数变动不会剧烈影响模型能力。
![](https://pic.downk.cc/item/5e7ee7a0504f4bcb042f42da.png)

# 2. Gradient Descent
**梯度下降（Gradient Descent，GD）**是神经网络优化最常用的方法，它是一种一阶近似方法。

若神经网络的参数为$θ$，将其初始化为$θ_0$，待优化的损失函数为$l(θ)$，在$θ_0$处对$l(θ)$进行Taylor展开：

$$ l(θ)=l(θ_0)+l'(θ_0)(θ-θ_0)+o(θ-θ_0)^2 $$

如果更新$θ$使损失函数减小，需要满足：

$$ l'(θ_0)(θ-θ_0)<0 $$

即应该沿梯度$l'(θ_0)$的负方向更新参数。

每次更新使用批量大小（batch size）为N的数据，并设置学习率（learning rate）$α$，若记第t次更新时的梯度为$g^t=l'(θ^{t-1})$，则更新公式：

$$ θ^t=θ^{t-1}-αg^t $$

**（1）batch size的选择**

通常训练数据的规模比较大，如果梯度下降在整个训练集上进行，需要比较多的计算资源；此外大规模训练集中的数据通常非常冗余。实际中把训练集分成若干mini batch，此时的方法叫做小批量梯度下降（mini batch gradient descent）。

mini batch的数据分布和总体的分布有差异，故batch size越小梯度计算的方差越大，引入的噪声就越大，可能导致训练不稳定。

当batch size=1时，也叫做**随机梯度下降（stochastic gradient descent，SGD）**。

当batch size为整个训练集时，也叫做**批量梯度下降（batch gradient descent）**。

Keskar通过[实验](https://arxiv.org/abs/1609.04836v1)发现，batch size越小，越有可能收敛到flat minima。

**（2）learning rate的选择**

学习率决定梯度下降的更新步长。学习率太大，可能会使更新不收敛甚至发散（diverge）；学习率太小，导致收敛速度慢。

Goyal提出了选择batch size和learning rate的[linear scaling rule](https://arxiv.org/abs/1706.02677)：batch size乘以k倍时，learning rate也乘以k倍。

更多学习率的选择方法参考“Learning Rate Strategy”一节。

梯度下降方法的**缺点**：
1. 不同参数方向梯度大小不同，更新时会振荡（解决措施：AdaGrad、RMSprop、AdaDelta、Adam）；
2. 在局部极小值和鞍点处梯度为零，无法更新（解决措施：Momentum、Nesterov Momentum）；
3. 批量越小，对噪声越敏感。

# 3. Momentum
梯度更新在局部极小值和鞍点处梯度为零，可以引入**动量(Momentum)**优化更新。

动量记录了最近一段时间内梯度的加权平均值：

$$ M^t = ρM^{t-1} + g^t $$

$$ θ^t=θ^{t-1}-αM^t $$

其中$ρ$为衰减率，一般取0.9。

# 4. Nesterov Momentum
Momentum更新方向是当前动量方向和当前梯度方向的矢量和；而[Nesterov Momentum](https://www.researchgate.net/publication/23629541_Gradient_methods_for_minimizing_composite_functions)更新方向是当前动量方向和沿动量方向的下一次梯度方向的矢量和：

$$ M^t = ρM^{t-1} + g(x^{t-1}+ρM^{t-1}) $$

$$ θ^t=θ^{t-1}-αM^t $$

Momentum和Nesterov Momentum对比如下：

![](https://pic.downk.cc/item/5e90327d504f4bcb047deaef.jpg)

# 5. AdaGrad
梯度下降时不同参数的梯度范围不同，导致参数更新时在梯度大的方向震荡，在梯度小的方向收敛较慢：

![](https://pic.downk.cc/item/5e902a62504f4bcb04758232.jpg)

损失函数的条件数越大，这一现象越严重。**条件数(Condition number)**是指损失函数的Hessian矩阵最大奇异值与最小奇异值之比。

[**AdaGrad（Adaptive Gradient）**](https://www.researchgate.net/publication/220320677_Adaptive_Subgradient_Methods_for_Online_Learning_and_Stochastic_Optimization)累积了之前的梯度平方，对梯度大小进行修正：

$$ G^t = G^{t-1} + g^t·g^t $$

$$ θ^t=θ^{t-1}-α\frac{g^t}{\sqrt{G^t}+ε} $$

AdaGrad的**缺点**是梯度平方的累计会越来越大，使得更新的实际梯度值越来越小。

# 6. RMSprop
[**RMSprop**](https://www.scirp.org/reference/ReferencesPapers.aspx?ReferenceID=1887533)计算梯度平方累积的指数衰减滑动平均，避免了实际梯度值越来越小：

$$ G^t = βG^{t-1} + (1-β)g^t·g^t $$

$$ θ^t=θ^{t-1}-α\frac{g^t}{\sqrt{G^t}+ε} $$

其中$β$为衰减率，一般取0.9。

# 7. AdaDelta
[**AdaDelta**](https://arxiv.org/abs/1212.5701)也采用了计算梯度平方累积的指数衰减滑动平均：

$$ G^t = βG^{t-1} + (1-β)g^t·g^t $$

此外，用参数更新差值$Δθ$的平方的指数衰减滑动平均代替学习率$α$：

$$ ΔX^{t-1} = β_2ΔX^{t-2} + (1-β_2)Δθ^{t-1}·Δθ^{t-1} $$

其中$β_2$为衰减率。

则$t$时刻参数更新差值为：

$$ Δθ^t=- \sqrt{ΔX^{t-1}+ε} \frac{g^t}{\sqrt{G^t}+ε} $$

# 8. Adam
[**Adam(自适应矩估计,Adaptive Moment Estimation)**](https://arxiv.org/abs/1412.6980v8)可以看作Momentum和RMSprop的结合。

一方面，计算梯度的加权平均值：

$$ M^t = β_1M^{t-1} + (1-β_1)g^t $$

另一方面，计算梯度平方的加权平均值：

$$ G^t = β_2G^{t-1} + (1-β_2)g^t·g^t $$

其中$β_1$,$β_2$为衰减率，一般取0.9,0.999。

迭代初期$M$和$G$很小，而$β_1$,$β_2$接近1，会使更新步长很小；引入偏差修正：

$$ \hat{M^t} = \frac{M^t}{1-β_1^t} $$

$$ \hat{G^t} = \frac{G^t}{1-β_2^t} $$

$$ θ^t=θ^{t-1}-α\frac{\hat{M^t}}{\sqrt{\hat{G^t}}+ε} $$

# 9. Nadam
Adam算法是Momentum和RMSprop的结合，[**Nadam**](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ&noteId=OM0jvwB8jIp57ZJjtNEZ)是Nesterov Momentum和RMSprop的结合。

# 10. Gradient Clipping
在循环神经网络中，Exploding Gradient是影响优化的重要因素。

[**梯度截断（Gradient Clipping）**](https://arxiv.org/abs/1211.5063)，是指当梯度的模大于一定阈值时，就对梯度进行截断，避免Exploding Gradient。

**按值截断**：给定区间$\[a,b\]$，如果梯度超过该区间就进行截断：

$$ g = max(min(g,b),a) $$

**按模截断**：将梯度的模截断到一个给定的截断阈值$b$:

$$ g= \begin{cases} g, & \mid\mid g \mid\mid^2 ≤ b \\ b, & \mid\mid g \mid\mid^2 > b \end{cases} $$

截断阈值$b$是一个超参数，[实验](https://arxiv.org/abs/1211.5063)发现，训练过程对阈值$b$并不十分敏感，通常一个小的阈值就可以得到很好的结果。

# 11. Learning Rate Strategy
学习率是模型训练时最重要的超参数之一。常用的学习率设置策略如下。

### （1）Learning Rate Decay

**学习率衰减（Learning Rate Decay）**也叫**学习率退火（Learning Rate Annealing）**，是指在训练初期设置大一些，再逐渐调小一些。

学习率衰减通常按每次迭代（iteration）进行，也可以按每个epoch进行。

注：在每一个mini batch上更新一次叫做一次iteration，在整个训练集上更新一次叫做一次epoch。

假设初始学习率为$α_0$，在第t次迭代时的学习率为$α_t$，常见的衰减方法：

**1.分段常数衰减（piecewise constant decay）**

每经过$T_1$,...,$T_m$次迭代，将学习率衰减为原来的$β_1$,...,$β_m$倍。

其中$β_1$,...,$β_m<1$为衰减率。

**2.逆时衰减（inverse time decay）**

$$ α_t=α_0\frac{1}{1+βt} $$

其中$β$为衰减率。

**3.指数衰减（exponential decay）**

$$ α_t=α_0β^t $$

其中$β<1$为衰减率。

**4.自然指数衰减（natural exponential decay）**

$$ α_t=α_0e^{-βt} $$

其中$β$为衰减率。

**5.余弦衰减（cosine decay）**

$$ α_t=\frac{1}{2}α_0(1+cos(\frac{t\pi}{T})) $$

其中T为总迭代次数。

下图给出了不同学习率衰减的示例（假设初始学习率为1）。
![](https://pic.downk.cc/item/5e7efc9f504f4bcb044157cf.png)

### （2）Warming up

训练初期需要较大的学习率，但刚开始训练时由于参数是随机初始化的，梯度也比较大，再加上较大的学习率，容易训练不稳定。

为了提高训练稳定性，可以在最初几轮迭代使用较小的学习率，等梯度下降到一定程度后再恢复到初始的学习率，这种方法叫做**学习率预热（warming up）**。

[Gradual Warmup](https://arxiv.org/abs/1706.02677)假设预热的迭代次数为$T'$，初始学习率为$α_0$，在预热时每次更新的学习率为：

$$ α_t=\frac{t}{T'}α_0, \quad 1≤t≤T' $$

预热结束后再选择学习率衰减方法逐渐降低学习率。

### （3）Cyclic Learning Rate

**循环学习率（Cyclic Learning Rate）**可以使梯度下降法逃离鞍点或局部极小点。让学习率在一个区间内周期性的增大和缩小，虽然这么做短期内损害优化过程，但从长期来看有助于找到更好的局部最优点。

1.[**三角循环学习率（Triangular Cyclic Learning Rate）**](https://arxiv.org/abs/1706.02677)

在一个周期内采用线性缩放调整学习率。

2.[**带热重启的随机梯度下降（Stochastic Gradient Descent with Warm Restarts，SGDR）**](https://arxiv.org/abs/1608.03983?context=math.OC)

学习率每间隔一定周期后重新初始化为某个预先设定值，然后逐渐衰减。

下图给出了两种周期性学习率调整的示例（假设初始学习率为1），每个周期中学习率的上界也逐步衰减。
![](https://pic.downk.cc/item/5e7f0191504f4bcb0445f81f.png)

# 12. Hyperparameter Optimization
在优化过程中，除了可学习的参数，还有许多**超参数(Hyperparameter)**。

常见的超参数有以下三类：
1. 网络结构参数，如层数、每层神经元数、激活函数类型；
2. 优化参数，如学习率、批量；
3. 正则化系数。

**超参数优化（Hyperparameter Optimization）**也叫超参数搜索，是一个组合优化问题。

### （1）Grid Search
**网格搜索(Grid Search)**通过尝试所有超参数的组合来寻找一组合适的超参数。

如果超参数是连续的，可以将超参数离散化；对于连续的超参数，不能按等间隔的方式进行离散化，需要根据超参数自身的特点进行离散化（如对数间隔）。

### （2）Random Search
不同超参数对模型性能的影响有很大差异，采用网格搜索会在不重要的超参数上进行不必要的尝试。

[**随机搜索(Random Search)**](https://www.researchgate.net/publication/262395872_Random_Search_for_Hyper-Parameter_Optimization)是对超参数进行随机组合，然后选取一个性能最好的配置。

### （3）Bayesian optimization
**贝叶斯优化（Bayesian optimization）**是一种自适应的超参数优化方法，根据当前已经试验的超参数组合，来预测下一个可能带来最大收益的组合。

一种比较常用的贝叶斯优化方法为[**时序模型优化（Sequential Model-Based Optimization，SMBO）**](https://link.springer.com/chapter/10.1007%2F978-3-642-25566-3_40)。

### （4）Dynamic Resource Allocation
**动态资源分配(Dynamic Resource Allocation)**的关键是将有限的资源分配给更有可能带来收益的超参数组合。

一种有效方法是[**逐次减半（Successive Halving）**](https://www.researchgate.net/publication/273005206_Non-stochastic_Best_Arm_Identification_and_Hyperparameter_Optimization)。

### （5）Neural Architecture Search
[**神经结构搜索(Neural Architecture Search，NAS)**]()是通过神经网络来自动实现网络结构的设计。

利用元学习的思想，神经结构搜索利用一个控制器来生成另一个子网络的结构描述。

控制器可以由一个循环神经网络来实现。

控制器的训练可以通过强化学习来完成，其奖励信号为生成的子网络在开发集上的准确率。