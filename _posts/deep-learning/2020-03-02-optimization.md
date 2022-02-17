---
layout: post
title: '深度学习中的优化算法(Optimization)'
date: 2020-03-02
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e7ee115504f4bcb04298afb.png'
tags: 深度学习
---

> Optimization in Deep Learning.

本文目录：
1. 深度学习中的优化问题
2. 基于梯度的优化算法
3. 常用的梯度下降算法

# 1. 深度学习中的优化问题
深度学习中的**优化**(**optimization**)问题通常是指在已有的数据集上实现最小的训练误差(**training error**)。记深度网络的待优化参数为$\theta$，包含$N$个训练样本$x$的数据集为$X$，损失度量为$l(x;\theta)$；则损失函数定义为：

$$ L(\theta)= \frac{1}{N}\sum_{x \in X}^{}l(x;\theta) $$

优化问题建模为寻找使得损失函数为**全局最小(global minima)**的参数$\theta^*$:

$$ \theta^*=\mathcal{\arg \min}_{\theta} L(\theta) $$

在实践中，优化深度网络存在以下困难：
1. **网络结构多样性**：深度网络是高度非线性的模型，网络结构的多样性阻碍了构造通用的优化方法；网络通常含有比较多的参数，难以优化。
2. **非凸优化**：深度网络的损失函数是高维空间的非凸函数，其损失曲面存在大量**局部极小(local minima)**和**鞍点(saddle point)**，这些点也满足梯度为$0$。
目前常用的优化方法大多数是基于梯度的，因此在寻找损失函数的全局最小值点的过程中，有可能会落入局部极小值点或鞍点上。

注：鞍点是指梯度为$0$但是**Hessian**矩阵不是半正定的点。
![](https://pic.imgdb.cn/item/61ef7aec2ab3f51d912a465c.jpg)

目前深度学习中的优化问题尚没有通用的解决方法，有许多工作尝试给出优化过程和损失函数的直觉解释：

- [<font color=Blue>Deep Ensembles: A Loss Landscape Perspective</font>](https://0809zheng.github.io/2020/07/12/deep-ensemble.html)：通过随机初始化训练一系列模型，使每个模型都收敛到不同的局部极小值，将这些模型集成起来对最终的结果有很大的提升。
- [<font color=Blue>Every Model Learned by Gradient Descent Is Approximately a Kernel Machine</font>](https://0809zheng.github.io/2021/05/19/kernalmachine.html)：使用梯度下降优化的深度学习模型近似于使用路径核的核方法构造的模型。
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572)：极小值点要求在特征的每个维度上都是极小点，这种情况概率比较低。在实践中，损失曲面上大部分梯度为零的点都是鞍点。
- [The Loss Surfaces of Multilayer Networks](https://arxiv.org/abs/1412.0233v3)：在非常大的神经网络中，大部分局部极小点和全局最小点是近似的；因此在训练神经网络时，通常没有必要找全局最小点，这反而可能过拟合。
- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)：由于深度网络的参数非常多且有一定的冗余性，每个参数对总损失的影响非常小。这使得损失函数在局部极小点附近通常是一个平坦的区域，即**平坦最小值(flat minima)**而不是**尖锐最小值(sharp minima)**，如下图所示。这使得模型收敛于局部极小值时更加**robust**，即微小的参数变动不会剧烈地影响模型能力。
  
![](https://pic.imgdb.cn/item/61ef7c112ab3f51d912b2ade.jpg)

# 2. 基于梯度的优化算法
深度学习中的优化问题通常用**梯度下降(Gradient Descent, GD)**算法求解，它是一种一阶近似方法。将神经网络的参数为$θ$初始化为$θ_0$，对损失函数$l(θ)$在$θ_0$处进行一阶**Taylor**展开：

$$ l(θ)=l(θ_0)+\nabla_{\theta} l(θ_0)(θ-θ_0)+o(θ-θ_0)^2 $$

注意到若使损失函数$l(θ)$减小，需要满足：

$$ \nabla_{\theta} l(θ_0)(θ-θ_0)<0 $$

当梯度$\nabla_{\theta} l(θ_0)$大于零时，应该减小$\theta$；当梯度$\nabla_{\theta} l(θ_0)$小于零时，应该增大$\theta$。综上所述，应该沿梯度$\nabla_{\theta}$的负方向更新参数。

在执行梯度下降时需要指定每次计算梯度时所使用数据的**批量(batch)** $\mathcal{B}$ 和**学习率(learning rate)** $\gamma$。若记第$t$次参数更新时的梯度为$g_t=\frac{1}{\|\mathcal{B}\|}\sum_{x \in \mathcal{B}}^{}\nabla_{\theta} l(θ_{t-1})$，则参数更新公式：

$$ θ_t=θ_{t-1}-\gamma g_t $$

## (1) 超参数的选择
### ① 超参数 batch size

损失函数$L(\theta)$是在整个训练集上定义的，因此计算更新梯度时需要考虑所有训练数据。这种方法称为**全量梯度下降**，此时**batch size**为整个训练集大小：$\|\mathcal{B}\|=N$。

通常训练数据的规模比较大，如果梯度下降在整个训练集上进行，需要比较多的计算资源；此外大规模训练集中的数据通常非常冗余。因此在实践中，把训练集分成若干**批次**(**batch**)的互补的子集，每次在一个子集上计算梯度并更新参数。这种方法称为**小批量梯度下降**(**mini batch gradient descent**)，此时$\|\mathcal{B}\|<N$。

特别地，把每个数据看作一个批次，对应**batch size=1**时，称为**随机梯度下降(stochastic gradient descent, SGD)**，此时$\|\mathcal{B}\|=1$。

每一批量数据的梯度都是总体数据梯度的近似，由于**mini batch**的数据分布和总体的分布有所差异，并且**batch size**越小这种差异越大；因此**batch size**越小，则梯度近似误差的方差越大，引入的噪声就越大，可能导致训练不稳定。

### ② 超参数 learning rate

学习率$\gamma$决定了梯度下降时的更新步长。学习率太大，可能会使梯度更新不收敛甚至发散(**diverge**)；学习率太小，导致收敛速度慢。

### ③ 选择超参数

学习率$\gamma$和批量大小$\|\mathcal{B}\|$的选择可以参考以下工作：
- [<font color=Blue>Don't Decay the Learning Rate, Increase the Batch Size</font>](https://0809zheng.github.io/2020/12/05/increasebatch.html)：在训练模型时通过增加批量大小替代学习率衰减，在相同的训练轮数下能够取得相似的测试精度，但前者所进行的参数更新次数更少，并行性更好，缩短了训练时间。

也有[实验](https://arxiv.org/abs/1609.04836v1)表明**batch size**越小，越有可能收敛到**flat minima**。

**Goyal**提出了选择**batch size**和**learning rate**的[linear scaling rule](https://arxiv.org/abs/1706.02677)：**batch size**增加$k$倍时，**learning rate**也增加$k$倍。

## (2) 从不同角度理解梯度下降

### ① 动力学角度

把神经网络建模为一个动力系统(**dynamical system**)，则梯度下降算法描述了参数$\theta$随时间(即梯度更新轮数)的演化情况。该动力系统的规则是参数$\theta$的变化率为损失函数的梯度$g=\nabla_{\theta}l(θ)$的负值：

$$ \dot θ =-g $$

该动力系统是一个**保守**动力系统，因此它最终可以收敛到一个不动点($\dot \theta = 0$)，并可以进一步证明该稳定的不动点是一个极小值点。求解上述常微分方程**ODE**可以采用**欧拉解法**（该方法是指将方程$dy/dx=f(x,y)$转化为$y_{n+1}-y_n≈f(x_n,y_n)h$）。因此有：

$$ θ_t-θ_{t-1}=-\gamma g $$

上式即为梯度下降算法的更新公式。对于小批量梯度下降，其每一批量上损失函数的梯度是总损失函数的梯度的近似估计，假设梯度估计的误差服从方差为$\sigma^2$的正态分布，则相当于在动力系统中引入高斯噪声：

$$ \dot θ =-g+\sigma \xi $$

该系统用随机微分方程**SDE**描述，称为**朗之万方程**。该方程的解可以用平衡状态下的概率分布描述：

$$ P(\theta) \text{ ~ } \exp(-\frac{l(\theta)}{\sigma^2}) $$

从上式中可以看出，当$\sigma^2$越大时，$P(\theta)$越接近均匀分布，此时参数$\theta$可能的取值范围也越大，允许探索更大比例的参数空间。当$\sigma^2$越小时，$P(\theta)$的极大值点(对应$l(\theta)$的极小值点)附近的区域越突出，则参数$\theta$落入极值点附近。在实践中，**batch size**越小，则参数$\sigma^2$越大。

![](https://pic.imgdb.cn/item/61f0098c2ab3f51d91b25d4d.jpg)

### ② 近似曲线逼近角度

### ③ 概率角度
从概率视角建模优化问题，记模型当前参数为$\theta$，优化目标为$l(\theta)$，将下一步的更新量$\Delta \theta$看作随机变量。则使得$l(\theta+\Delta \theta)$的数值越小的$\Delta \theta$出现的概率越大，用下面的分布表示：

$$ p(\Delta \theta | \theta) =\frac{e^{-[l(\theta+\Delta \theta)-l(\theta)] / \alpha}}{Z(\theta)}, Z(\theta)=\int_{}^{} e^{-[l(\theta+\Delta \theta)-l(\theta)] / \alpha}d(\Delta \theta) $$

式中$Z(\theta)$是归一化因子。参数$\alpha>0$调整分布的形状；当$\alpha \to ∞$时，$p(\Delta \theta \| \theta)$趋近于均匀分布，参数可以向任意方向变化；当$\alpha \to 0$时，只有使得$l(\theta+\Delta \theta)$最小的$\Delta \theta$对应的概率$p(\Delta \theta \| \theta)$不为$0$，因此$\Delta \theta$将选择损失下降最快的方向。

下一步的实际更新量可以选择上式的期望：

$$ \Delta \theta^* = \Bbb{E}_{\Delta \theta\text{~}p(\Delta \theta | \theta)}[\Delta \theta] = \int_{}^{} p(\Delta \theta | \theta) \Delta \theta d (\Delta \theta) $$

通常$p(\Delta \theta \| \theta)$很难直接求得。假设$l(\theta)$是一阶可导的，由**Taylor**展开得：

$$ l(\theta+\Delta \theta) - l(\theta) ≈ \Delta \theta^Tg $$

其中梯度$g=\nabla_\theta l(\theta)$。若约束参数更新的步长不超过$\epsilon$，即$\|\|\Delta \theta\|\| \leq \epsilon$，则概率$p(\Delta \theta \| \theta)$表示为：

$$ p(\Delta \theta | \theta) =\frac{e^{-\Delta \theta^Tg / \alpha}}{Z(g)}, Z(g)=\int_{||\Delta \theta|| \leq \epsilon}^{} e^{-\Delta \theta^Tg / \alpha}d(\Delta \theta) $$

概率的期望表示为：

$$ \Delta \theta^* = \int_{||\Delta \theta|| \leq \epsilon}^{} \frac{e^{-\Delta \theta^Tg / \alpha}}{Z(g)} \Delta \theta d (\Delta \theta) = -\nabla_g \ln Z(g) $$

假设更新量$\Delta \theta$与梯度$g$的夹角为$\eta$，则$Z(g)$表示为：

$$ Z(g)=\int_{||\Delta \theta|| \leq \epsilon}^{} e^{-||\Delta \theta|| \times ||g|| \times \cos \eta / \alpha}d(\Delta \theta) $$

上式表示一个高维球体内的积分，由于积分空间具有各向同性，因此该积分只和梯度的模长$\|\|g\|\|$有关，因此将积分$Z(g)$记为$Z(\|\|g\|\|)$，则有：

$$ \Delta \theta^* = -\nabla_g \ln Z(||g||) = -\frac{Z'(||g||)}{Z(||g||)} \nabla_g ||g|| = -\frac{Z'(||g||)}{Z(||g||)} \frac{g}{||g||} $$

上式表示参数更新的最佳方向与梯度方向相反，即为梯度下降算法。

# 3. 常用的梯度下降算法

标准的批量梯度下降方法存在一些缺陷：
- 更新过程中容易陷入局部极小值或鞍点(这些点处的梯度也为$0$)；常见解决措施是在梯度更新中引入**动量**(如**momentum**,**Nesterov**)。
- 参数的不同维度的梯度大小不同，导致参数更新时在梯度大的方向震荡，在梯度小的方向收敛较慢；损失函数的**条件数(Condition number**，指损失函数的**Hessian**矩阵最大奇异值与最小奇异值之比)越大，这一现象越严重。常见解决措施是为每个特征设置**自适应**学习率(如**AdaGrad**, **RMSprop**, **AdaDelta**)。这类算法的缺点是改变了梯度更新的方向，一定程度上造成精度损失。![](https://pic.downk.cc/item/5e902a62504f4bcb04758232.jpg)
- 批量大小难以选择。批量较小时，引入较大的梯度噪声；批量较大时，内存负担较大。在分布式训练大规模神经网络时，整体批量通常较大，训练的模型精度会剧烈降低。这是因为总训练轮数保持不变时，批量增大意味着权重更新的次数减少。常见解决措施是为网络每一层参数设置**层级自适应**学习率(如**LARS**, **LAMB**, **NovoGrad**)。

在实际应用梯度下降算法时，可以根据截止到当前步$t$的历史梯度信息$$\{g_{1},...,g_{t}\}$$计算修正的参数更新量$h_t$（比如累积动量、累积二阶矩校正学习率等），从而弥补上述缺陷。因此梯度下降算法的一般形式可以表示为：

$$ \begin{align} g_t&=\frac{1}{\|\mathcal{B}\|}\sum_{x \in \mathcal{B}}^{}\nabla_{\theta} l(θ_{t-1}) \\ h_t &= f(g_{1},...,g_{t}) \\ θ_t&=θ_{t-1}-\gamma h_t \end{align} $$

下面介绍一些常见的基于梯度的优化算法，部分算法的代码实现可参考[Pytorch文档](https://pytorch.org/docs/stable/optim.html#algorithms)。

| 优化算法 | 参数定义(缺省值/初始值) |  更新形式 |
| ---- | ---- |   ---- |
| GD | $$g_t\text{: 梯度} \\ \gamma\text{: 学习率}$$ | $$\begin{align} g_t&=\nabla_{\theta} l(θ_{t-1}) \\ θ_t&=θ_{t-1}-\gamma g_t \end{align}$$   |
| [<font color=Blue>RProp</font>](https://0809zheng.github.io/2020/12/07/rprop.html)：根据梯度符号更新参数 | $$\eta_t\text{: 弹性步长} \\ \eta_+\text{: 步长增加值}(1.2) \\ \eta_{\_}\text{: 步长减小值}(0.5) \\ \Gamma_{max} \text{: 最大步长} \\ \Gamma_{min} \text{: 最小步长}$$ | $$\begin{align} \eta_t &= \begin{cases} \min(\eta_{t-1}\eta_+,\Gamma_{max}) & \text{if }g_{t-1}g_t>0 \\ \max(\eta_{t-1}\eta_{\_},\Gamma_{min}) & \text{if }g_{t-1}g_t<0  \\ \eta_{t-1}& \text{else} \end{cases} \\θ_{t} &= θ_{t-1} -\eta_t \text{sign}(g_t) \end{align}$$   |
| Momentum：引入动量(梯度累计值) | $$M_t\text{: 动量}(0) \\ \gamma\text{: 学习率} \\ \mu \text{: 衰减率}(0.9)$$ | $$\begin{align} M_t &= \mu M_{t-1} + g_t \\ θ_t&=θ_{t-1}-\gamma M_t \end{align}$$   |
| [<font color=Blue>Nesterov Momentum</font>](https://0809zheng.github.io/2020/12/08/nesterov.html)：引入Nesterov动量 | $$M_t\text{: 动量}(0) \\ \gamma\text{: 学习率} \\ \mu \text{: 衰减率}(0.9)$$ | $$\begin{align}  M_t &= \mu M_{t-1} + \nabla_{\theta} l(θ_{t-1}-\mu M_{t-1}) \\ θ_t&=θ_{t-1}-\gamma M_t \\-&-------------\\ M_t & = \mu M_{t-1} +  g_t \\ θ_t & =θ_{t-1}-\gamma(\mu M_t + g_t) \end{align}$$   |
| [AdaGrad](http://jmlr.org/papers/v12/duchi11a.html)：使用平方梯度累计调整学习率 | $$G_t\text{: 平方梯度}(0) \\ \gamma\text{: 学习率} \\ \epsilon \text{: 小值}(1e-10)$$ | $$\begin{align} G_t &=  G_{t-1} + g_t^2 \\ θ_{t} &= θ_{t-1} -\gamma \frac{g_t}{\sqrt{G_t}+\epsilon} \end{align}$$   |
| [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)：使用指数滑动平均调整学习率 | $$G_t\text{: 平方梯度}(0) \\ \gamma\text{: 学习率} \\ \rho \text{: 衰减率}(0.99) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} G_t &= \rho G_{t-1} + (1-\rho)g_t^2 \\θ_{t} &= θ_{t-1} -\gamma \frac{g_t}{\sqrt{G_t}+\epsilon} \end{align}$$   |
| [<font color=Blue>Adadelta</font>](https://0809zheng.github.io/2020/12/06/adadelta.html)：校正RMSProp的参数更新单位 | $$G_t\text{: 平方梯度}(0) \\ X_t\text{: 平方参数更新量}(0) \\ \rho \text{: 衰减率}(0.9) \\ \epsilon \text{: 小值}(1e-6)$$ | $$\begin{align} G_t &= \rho G_{t-1} + (1-\rho)g_t^2 \\ X_{t-1} &= \rho X_{t-2} + (1-\rho)\Delta θ_{t-1}^2 \\ Δθ_t &= -\frac{\sqrt{X_{t-1}+\epsilon}}{\sqrt{G_t+\epsilon}} g_t \\θ_{t} &= θ_{t-1} +Δθ_t \end{align}$$   |
| [<font color=Blue>Adam</font>](https://0809zheng.github.io/2020/12/09/adam.html)：结合动量与RMSProp | $$M_t\text{: 动量}(0) \\ G_t\text{: 平方梯度}(0) \\ \gamma\text{: 学习率} \\ \beta_1 \text{: 衰减率}(0.9)  \\ \beta_2 \text{: 衰减率}(0.999) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} M_t &= β_1M_{t-1} + (1-β_1)g_t \\ G_t &= β_2G_{t-1} + (1-β_2)g_t^2 \\\hat{M}_t &= \frac{M_t}{1-β_1^t} \\ \hat{G}_t &= \frac{G_t}{1-β_2^t} \\ θ_t&=θ_{t-1}-\gamma \frac{\hat{M}_t}{\sqrt{\hat{G}_t}+ε} \end{align}$$   |
| [<font color=Blue>Adamax</font>](https://0809zheng.github.io/2020/12/09/adam.html)：使用L-∞范数调整学习率 | $$M_t\text{: 动量}(0) \\ G_t\text{: L-∞范数梯度}(0) \\ \gamma\text{: 学习率} \\ \beta_1 \text{: 衰减率}(0.9)  \\ \beta_2 \text{: 衰减率}(0.999) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} M_t &= β_1M_{t-1} + (1-β_1)g_t \\ G_t &= \max(\beta_2 G_{t-1}, \|g_t\|+\epsilon) \\\hat{M}_t &= \frac{M_t}{1-β_1^t}  \\ θ_t&=θ_{t-1}-\gamma \frac{\hat{M}_t}{G_t} \end{align}$$   |
| [<font color=Blue>Nadam</font>](https://0809zheng.github.io/2020/12/12/nadam.html)：将Nesterov动量引入Adam | $$M_t\text{: 动量}(0) \\ G_t\text{: 平方梯度}(0) \\ \gamma\text{: 学习率} \\ \beta_1 \text{: 衰减率}(0.9)  \\ \beta_2 \text{: 衰减率}(0.999) \\ \psi \text{: 衰减率}(0.004) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} \mu_t &= \beta_1(1-\frac{1}{2}0.96^{t\psi}) \\ \mu_{t+1} &= \beta_1(1-\frac{1}{2}0.96^{(t+1)\psi}) \\ M_t &= β_1M_{t-1} + (1-β_1)g_t \\ G_t &= β_2G_{t-1} + (1-β_2)g_t^2 \\\hat{M}_t &= \frac{\mu_{t+1} M_{t}}{1-\prod_{i=1}^{t}\mu_i}+\frac{(1-\mu_t) g_t}{1-\prod_{i=1}^{t}\mu_i} \\ \hat{G}_t &= \frac{G_t}{1-β_2^t} \\ θ_t&=θ_{t-1}-\gamma \frac{\hat{M}_t}{\sqrt{\hat{G}_t}+ε} \end{align}$$   |
| [<font color=Blue>AMSGrad</font>](https://0809zheng.github.io/2020/12/10/amsgrad.html)：改善Adam的收敛性 | $$M_t\text{: 动量}(0) \\ G_t\text{: 平方梯度}(0) \\ \gamma\text{: 学习率} \\ \beta_1 \text{: 衰减率}(0.9)  \\ \beta_2 \text{: 衰减率}(0.999) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} M_t &= β_1M_{t-1} + (1-β_1)g_t \\ G_t &= β_2G_{t-1} + (1-β_2)g_t^2 \\\hat{M}_t &= \frac{M_t}{1-β_1^t} \\ \hat{G}_t &= \frac{G_t}{1-β_2^t} \\ \hat{G}_t^{max} &= \max(\hat{G}_{t-1}^{max},\hat{G}_t) \\ θ_t&=θ_{t-1}-\gamma \frac{\hat{M}_t}{\sqrt{\hat{G}_t^{max}}+ε} \end{align}$$   |
| [<font color=Blue>Radam</font>](https://0809zheng.github.io/2020/12/13/radam.html)：减小Adam中自适应学习率的早期方差 | $$M_t\text{: 动量}(0) \\ G_t\text{: 平方梯度}(0) \\ \gamma\text{: 学习率} \\ \beta_1 \text{: 衰减率}(0.9)  \\ \beta_2 \text{: 衰减率}(0.999) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} \rho_∞ &= \frac{2}{1-\beta_2} - 1 \\ M_t &= β_1M_{t-1} + (1-β_1)g_t \\ G_t &= β_2G_{t-1} + (1-β_2)g_t^2 \\ \hat{M}_t &= \frac{M_t}{1-β_1^t} \\ \rho_t &= \rho_∞ -  \frac{2t\beta_2^t}{1-\beta_2^t} \\ \text{if } \rho_t &> 4: \\ \hat{G}_t &= \frac{G_t}{1-β_2^t} \\ r_t &= \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_∞}{(\rho_∞-4)(\rho_∞-2)\rho_t}} \\ θ_t&=θ_{t-1}-\gamma r_t \frac{\hat{M}_t}{\sqrt{\hat{G}_t}+ε} \\ \text{else} & :\\ θ_t&=θ_{t-1}-\gamma \hat{M}_t \end{align}$$   |
| [<font color=Blue>LARS</font>](https://0809zheng.github.io/2020/12/15/lars.html)：层级自适应学习率+Momentum | $$M_t\text{: 动量}(0) \\ \gamma\text{: 全局学习率} \\ \mu \text{: 衰减率}(0.9)$$ | $$\begin{align}  M_t &= \mu M_{t-1} + g_t \\ θ_t^{(i)}&=θ_{t-1}^{(i)}-\gamma \frac{\| θ_{t-1}^{(i)} \|}{\| M_t^{(i)} \|} M_t^{(i)}, \text{for all }i \in [L] \end{align}$$   |
| [<font color=Blue>LAMB</font>](https://0809zheng.github.io/2020/12/17/lamb.html)：层级自适应学习率+Adam | $$M_t\text{: 动量}(0) \\ G_t\text{: 平方梯度}(0) \\ \gamma\text{: 全局学习率} \\ \beta_1 \text{: 衰减率}(0.9)  \\ \beta_2 \text{: 衰减率}(0.999) \\ \epsilon \text{: 小值}(1e-8)$$ | $$\begin{align} M_t &= β_1M_{t-1} + (1-β_1)g_t \\ G_t &= β_2G_{t-1} + (1-β_2)g_t^2 \\\hat{M}_t &= \frac{M_t}{1-β_1^t} \\ \hat{G}_t &= \frac{G_t}{1-β_2^t} \\ θ_t^{(i)}&=θ_{t-1}^{(i)}-\gamma \frac{\| θ_{t-1}^{(i)} \|}{\|\frac{\hat{M}_t^{(i)}}{\sqrt{\hat{G}_t^{(i)}}+ε}\|} \frac{\hat{M}_t^{(i)}}{\sqrt{\hat{G}_t^{(i)}}+ε} , \text{for all }i \in [L]  \end{align}$$   |
| [<font color=Blue>Lookahead</font>](https://0809zheng.github.io/2020/12/14/lookahead.html)：快权重更新k次,慢权重更新1次 | $$\phi_t\text{: 慢权重} \\ k\text{: 快权重更新次数}(5) \\ \alpha \text{: 慢权重更新步长}(0.5)  \\ \gamma\text{: 学习率}$$ | $$\begin{align} h_t &= f(g_{1},...,g_{t}) \\ θ_t&=θ_{t-1}-\gamma h_t \\ \text{if } t \text{ mod }k &=0: \\ \phi_{t} &= \phi_{t-1} +\alpha(\theta_{t}-\phi_{t-1}) \\ \theta_t &= \phi_t \end{align}$$   |





