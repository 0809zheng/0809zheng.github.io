---
layout: post
title: '生成对抗网络'
date: 2020-05-18
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ebd0c69c2a9a83be54ed281.jpg'
tags: 深度学习
---

> Generative Adversarial Networks.

**生成对抗网络(Generative Adversarial Network,GAN)**是一种生成模型，可用来生成图像、文本等结构数据（**structured data**）。

各式各样的$GAN$：[GAN zoo](https://github.com/hindupuravinash/the-gan-zoo)

近些年来$GAN$的发展：

![](https://pic.downk.cc/item/5ed86506c2a9a83be5b4e88c.jpg)

**GAN**的演化：
- 选用不同的损失以减小优化困难：vanilla GAN、fGAN、LSGAN、WGAN
- 加入条件控制生成样本：conditional GAN、InfoGAN
- 领域迁移：cycleGAN、GauGAN、GANILLA
- 与自编码器一起训练：EBGAN、VAE-GAN、BiGAN
- 图像生成：DCGAN、LAPGAN、SAGAN、BigGAN
- 特殊的模型：SinGAN（使用单张图像训练）、NICE-GAN（生成器是判别器的一部分）、StackGAN（文本生成图像）

**本文目录**：
1. GAN的基本原理
2. vanilla GAN
3. GAN训练时的问题
4. conditional GAN
5. cycleGAN
6. fGAN
7. Least Square GAN (LSGAN)
8. Wasserstein GAN (WGAN)
9. Energy-based GAN (EBGAN)
10. InfoGAN
11. VAE-GAN
12. BiGAN
13. DCGAN
14. Self-attention GAN (SAGAN)
15. BigGAN
16. SinGAN
17. GauGAN
18. GANILLA
19. NICE-GAN
20. GAN的评估指标
21. LAPGAN
22. StackGAN


# 1. GAN的基本原理
假设已知数据具有概率分布$$P_{data}(x)$$，$GAN$的目的是拟合一个近似的分布$$P_G(x;θ_g)$$;

从已知数据集$$P_{data}(x)$$中采样出输入样本$$\{x^1,x^2,...,x^m\}$$，

可以采用**极大似然估计（maximum likelihood estimation）**优化参数$θ_g$：

$$ θ_g^* = argmax_{(θ_g)} \prod_{i=1}^{m} {P_G(x^i;θ_g)} = argmax_{(θ_g)} log(\prod_{i=1}^{m} {P_G(x^i;θ_g)}) \\ = argmax_{(θ_g)} \sum_{i=1}^{m} {log(P_G(x^i;θ_g))} = argmax_{(θ_g)} E_{x^i \text{~} P_{data}}[log(P_G(x^i;θ_g))] $$

上式等价于**最小化KL散度（KL divergence）**:

$$ θ_g^* = argmax_{(θ_g)} E_{x^i \text{~} P_{data}}[log(P_G(x^i;θ_g))] \\ = argmax_{(θ_g)} \int_{x}^{} {P_{data}log(P_G(x;θ_g))dx} - \int_{x}^{} {P_{data}log(P_{data}(x))dx} \\ = argmin_{(θ_g)} KL(P_{data} \mid\mid P_G) $$

如何定义分布$$P_G(x;θ_g)$$？

# 2. vanilla GAN
$GAN$没有显式的定义$$P_G(x;θ_g)$$，而是使用生成器（通常是神经网络）拟合一个数据分布；衡量生成分布和数据的实际分布之间的差异时，使用判别器（通常是神经网络）定义损失函数。

$GAN$包括**生成器generator**和**判别器discriminator**。

- **生成器generator**：生成器是一个神经网络，从一个简单的概率分布（噪声）中采样得到$z$，经过生成器$G$得到输入数据概率分布的估计$$P_G(x;θ_g)=G(z)$$，其中$$θ_g$$表示生成器的参数；
![](https://pic.downk.cc/item/5eb555dcc2a9a83be5b2b166.jpg)

$$ G^* = argmin_{(G)} Div(P_G,P_{data}) $$

- **判别器discriminator**：判别器是一个二分类器，判断并区分从输入数据分布$$P_{data}$$和生成器分布$$P_G$$中采样得到的数据，并输出一个标量。
![](https://pic.downk.cc/item/5eb55bb7c2a9a83be5b7fced.jpg)

$$ D^* = argmax_{(D)} E_{x \text{~} P_{data}}[logD(x)] + E_{x \text{~} P_{G}}[log(1-D(x))] $$

若记$V$：

$$ V = E_{x \text{~} P_{data}}[logD(x)] + E_{x \text{~} P_{G}}[log(1-D(x))] \\ = \int_{x}^{} {[P_{data}(x)logD(x)+P_{G}(x)log(1-D(x))]dx} $$

令$$ \frac{\partial V}{\partial D} = 0 $$，得：

$$ D^*(x) = \frac{P_{data}(x)}{P_{data}(x)+P_{G}(x)} $$

代回原优化问题得：

$$ max_DV(G,D) = V(G,D^*) \\ = \int_{x}^{} {[P_{data}(x)\frac{P_{data}(x)}{P_{data}(x)+P_{G}(x)}+P_{G}(x)\frac{P_{G}(x)}{P_{data}(x)+P_{G}(x)}]dx} $$

引入**JS散度（Jensen-Shannon divergence）**：

$$ JS(P \mid\mid Q) = \frac{1}{2}KL(P \mid\mid M) + \frac{1}{2}KL(Q \mid\mid M), \quad M = \frac{P+Q}{2} $$

则**判别器**的优化问题是最大化两个分布的**JS散度**：

$$ max_DV(G,D) = \int_{x}^{} {[P_{data}(x)\frac{P_{data}(x)}{\frac{(P_{data}(x)+P_{G}(x))}{2}}+P_{G}(x)\frac{P_{G}(x)}{\frac{(P_{data}(x)+P_{G}(x))}{2}}]dx} - 2log2 \\ = 2JS(P_{data} \mid\mid P_G) - 2log2 $$

$$ D^* = argmax_{(D)}V(G,D) $$

**生成器**的优化问题：

$$ G^* = argmin_{(G)} max_{(D)}V(G,D) $$

这是一个**极小极大博弈**：

![](https://pic.downk.cc/item/5eb5623fc2a9a83be5bbec03.jpg)

这也是一个**零和博弈**，参与博弈的各方，在严格竞争下，一方的收益必然意味着另一方的损失，博弈各方的收益和损失相加总和永远为“零”。

交替迭代，在这个过程中，双方都极力优化自己的网络，从而形成竞争对抗，直到双方达到一个动态的平衡（**纳什均衡**）。

### Algorithm
初始化生成器和判别器，在训练的每一次迭代中，先更新判别器$D$，再更新生成器$G$：
- **固定生成器$G$，更新判别器$D$**，重复$k$次：
1. 从训练数据集中采样$$\{x^1,x^2,...,x^m\}$$；
2. 从随机噪声中采样$$\{z^1,z^2,...,z^m\}$$；
3. 根据生成器$G$获得生成数据$$\{\tilde{x}^1,\tilde{x}^2,...,\tilde{x}^m\}$$；

$$ θ_d = argmax_{(θ_d)} \frac{1}{m} \sum_{i=1}^{m} {logD(x^i)} + \frac{1}{m} \sum_{i=1}^{m} {log(1-D(\tilde{x}^i))} $$

- **固定判别器$D$，更新生成器$G$**，进行$1$次：
1. 从随机噪声中采样$$\{z^1,z^2,...,z^m\}$$；

$$ θ_g = argmin_{(θ_g)} \frac{1}{m} \sum_{i=1}^{m} {log(1-D(G(z^i)))} $$

使用上式的$GAN$被称作**Minimax GAN (MMGAN)**。

由于上式在计算$θ_g$时，函数$log(1-D(x))$在$D(x)$接近$0$（即还没优化好）时梯度较小，在$D(x)$接近$1$（即已经优化好）时梯度较大，会使优化困难，因此用下式代替：

$$ θ_g = argmax_{(θ_g)} \frac{1}{m} \sum_{i=1}^{m} {log(D(G(x^i)))} $$

使用上式的$GAN$被称作**Non-saturating GAN (NSGAN)**。

![](https://pic.downk.cc/item/5eb5666fc2a9a83be5bfcf31.jpg)

# 3. GAN训练时的问题
### Mode Collapse
**Mode Collapse**是指生成的分布集中在真实数据分布的某一部分，缺乏多样性。
![](https://pic.downk.cc/item/5eb61acac2a9a83be55203a9.jpg)

### Mode Dropping
**Mode Dropping**是指真实数据分布有多个簇，而生成迭代每一次只生成其中某一个簇。
![](https://pic.downk.cc/item/5eb61b14c2a9a83be5527b4d.jpg)

**解决措施**：$Ensemble$，训练多个生成器，每次随机选择一个生成器生成样本。

# 4. conditional GAN
$GAN$无法控制生成的图像，**conditional GAN**可以生成给定条件的图像。

- **Generator**：在输入随机噪声$z$的同时还输入了条件$c$：

![](https://pic.downk.cc/item/5ebba5c9c2a9a83be51ed12f.jpg)

- **Discriminator**：输入图像和条件，判断图像是否是真实的并且是否与条件匹配：

![](https://pic.downk.cc/item/5ebba610c2a9a83be51f0865.jpg)

也有论文在设计时把判断是否真实和判断条件匹配分开：

![](https://pic.downk.cc/item/5ebba65cc2a9a83be51f6912.jpg)

# 5. cycleGAN
**cycleGAN**可以实现domain迁移，即从一种风格的图像转变成另一种风格的图像。

假设有两类domain的图像$X$和$Y$，给定$X$的图像，希望能转换成$Y$的类型；或给定$Y$的图像转换成$X$的类型。

训练两个**Generator**，$$G_{X→Y}$$实现从类型$X$转换成类型$Y$，$$G_{Y→X}$$实现从类型$Y$转换成类型$X$；

训练两个**Discriminator**，$$D_{X}$$判断图像是否属于类型$X$；$$D_{Y}$$判断图像是否属于类型$Y$；

为保证转换后的图像仍具有转换前的信息，引入**Cycle Consistency Loss**，保持循环转换后的结果尽可能相似：

![](https://pic.downk.cc/item/5ebba8ebc2a9a83be521d466.jpg)

# 6. fGAN
- paper：[f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709v1)

$GAN$的生成器在优化时试图减小$$P_{data}$$和$$P_G$$之间的**KL散度**。

$KL$散度是两个概率分布的不对称性度量，若$P$表示随机变量的真实分布，$Q$表示理论或拟合分布，则$$KL(P \mid\mid Q)$$被称为**前向KL散度（forward KL divergence）**；$$KL(Q \mid\mid P)$$被称为**后向KL散度（backward KL divergence）**，也叫**reverse KL散度**。

前向散度中拟合分布是$KL$散度公式的分母，因此若在随机变量的某个取值范围中，拟合分布的取值趋于$0$，则此时$KL$散度的取值趋于无穷。因此使用前向$KL$散度最小化拟合分布和真实分布的距离时，拟合分布趋向于覆盖理论分布的所有范围。前向KL散度的上述性质被称为“**0避免（zero avoiding）**”。

相反地，当使用后向$KL$散度求解拟合分布时，由于拟合分布是分子，其$0$值不影响$KL$散度的积分，反而是有利的，因此后项$KL$散度是“**0趋近（zero forcing）**”的。

![](https://pic.downk.cc/item/5ebcf68cc2a9a83be53cf38d.jpg)

**KL散度**并不是唯一的选择。一般地，可以优化$$p(x)$$和$$q(x)$之间的$f$**散度**：

$$ D_f(P \mid\mid Q) = \int_{x}^{} {q(x)f(\frac{p(x)}{q(x)})dx} $$

函数$f$的性质：
- $$f(1)=0$$：当$$p(x)=q(x)$$时，散度为$0$;
- $f$是凸函数：该性质使散度恒大于等于零，$$ D_f(P \mid\mid Q) = \int_{x}^{} {q(x)f(\frac{p(x)}{q(x)})dx} ≥ f(\int_{x}^{} {q(x)\frac{p(x)}{q(x)}dx}) = f(1) = 0 $$

当函数$f$不同时，对应的散度也不同：
- **KL散度**：$$f(x) = xlogx$$

$$ D_f(P \mid\mid Q) = \int_{x}^{} {q(x) \frac{p(x)}{q(x)} log(\frac{p(x)}{q(x)})dx} = \int_{x}^{} {p(x) log(\frac{p(x)}{q(x)})dx} $$

- **Reverse KL散度**：$$f(x) = -logx$$

$$ D_f(P \mid\mid Q) = \int_{x}^{} {q(x) (-log(\frac{p(x)}{q(x)}))dx} = \int_{x}^{} {q(x) log(\frac{q(x)}{p(x)})dx} $$

- $$\chi^2$$**散度**：$$f(x) = (x-1)^2$$

$$ D_f(P \mid\mid Q) = \int_{x}^{} {q(x) (\frac{p(x)}{q(x)}-1)^2dx} = \int_{x}^{} {\frac{(p(x)-q(x))^2}{q(x)}dx} $$

### Fenchel Conjugate
每一个凸函数$f$都有一个共轭函数（conjugate function）$f^*$：

$$ f^*(t) = max_{(x \in dom(f))} \{ xt - f(x) \} $$

![](https://pic.downk.cc/item/5ebcfd25c2a9a83be542cee6.jpg)

$f^*$的性质：
- $f^*$也是凸函数；
- $$(f^*)^*$$=$$f$$

例题：**KL散度**的$$f(x) = xlogx$$对应$$ f^*(t) = e^{t-1} $$

求解下式：

$$ f^*(t) = max_{(x \in dom(f))} \{ xt - xlogx \} $$

令$$\frac{\partial(xt-xlogx)}{\partial x}=0$$得$$t-logx-1=0$$，即$$x=e^{t-1}$$,代入得：

$$ f^*(t) = e^{t-1}t - e^{t-1}(t-1) = e^{t-1} $$

同理$f$可表达为：

$$ f(x) = max_{(t \in dom(f^*))} \{ xt - f^*(t) \} $$

则$f$**散度**可以进行化简：

$$ D_f(P \mid\mid Q) = \int_{x}^{} {q(x)f(\frac{p(x)}{q(x)})dx} \\ = \int_{x}^{} {q(x)(max_{(t \in dom(f^*))} \{ \frac{p(x)}{q(x)} t - f^*(t) \})dx} \\ ≈ max_D \int_{x}^{} {p(x)D(x)dx} - \int_{x}^{} {q(x)f^*(D(x))dx} \\ = max_D \{ E_{x \text{~} P}[D(x)]- E_{x \text{~} Q}[f^*(D(x))] \} $$

其中$$D(x)$$是一个函数，拟合从$x$到$t$的映射。

则网络的生成器优化目标为：

$$ G^* = argmin_G D_f(P_{data} \mid\mid P_G) \\ = argmin_G max_D \{ E_{x \text{~} P}[D(x)]- E_{x \text{~} Q}[f^*(D(x))] \} \\ = argmin_G max_D V(G,D) $$

在实践中可以选择不同的散度：
![](https://pic.downk.cc/item/5ebd05ccc2a9a83be549a221.jpg)

# 7. Least Square GAN (LSGAN)
GAN的优化通常考虑$$P_{data}$$和$$P_G$$之间的散度，但散度并不一定是最好的选择。

在实践中，$$P_{data}$$和$$P_G$$之间几乎没有重叠，原因如下：
1. 这些概率通常是高维空间中的低维流形，重合可以被忽略；
2. 即使两个概率会有重合，由于实践中是对概率进行采样，也很难采集到重合的样本。

对于散度，当两个概率分布没有重合时，无论两个概率分布差距如何，散度的值都是恒定的(如对于JS散度，恒为$log2$)，从而优化的梯度为0，不利于优化。

![](https://pic.downk.cc/item/5ebe8dbbc2a9a83be5d06ebb.jpg)

**LSGAN**对判别器不再使用Sigmoid分类，而是使用线性回归代替：

![](https://pic.downk.cc/item/5ebe8e6cc2a9a83be5d133ab.jpg)

# 8. Wasserstein GAN (WGAN)

### (1)Wasserstein Distance
**Wasserstein距离**又叫**推土机距离(Earth Mover’s Distance)**，是指把一个概率分布$P$变成另一个概率分布$Q$时所需要的最小变换距离。$变成另一个概率分布$Q$时所需要的最小变换距离。

![](https://pic.downk.cc/item/5ebe8f4dc2a9a83be5d214e2.jpg)

记从概率分布$P$变成概率分布$Q$的任意一个变换过程为**moving plan γ**，对应一个矩阵，

- 每一行代表概率分布$P$的某个取值$x_p$要分配到概率分布$Q$的值（黑色为0）；
- 每一列代表概率分布$Q$的某个取值$x_q$接收到概率分布$P$分配的值。

则这一个**moving plan γ**的平均距离定义为：

$$ B(γ) = \sum_{x_p,x_q}^{} {γ(x_p,x_q) \mid\mid x_p-x_q \mid\mid} $$

Wasserstein距离定义为：

$$ W(P,Q) = min_{γ}B(γ) $$

Wasserstein距离可以用来衡量两个概率分布的差距。

### (2)WGAN
WGAN的目标函数：

$$ V(G,D) = max_{(D \in 1-Lipschitz)} \{ E_{x \in P_{data}}[D(x)]-E_{x \in P_{G}}[D(x)] \} $$

其中要求$D$是$1$阶**Lipschitz**函数。$k$阶**Lipschitz**函数的定义如下：

$$ \mid\mid f(x_1)-f(x_2) \mid\mid ≤ k \mid\mid x_1-x_2 \mid\mid $$

**Lipschitz**函数的输出变化相对输入变化是缓慢的。若没有该限制，优化会使D趋向正负无穷。

在实践中，通过**weight clipping**实现这个约束：

- 把参数$w$的取值限制在$$[-c,c]$$之间。

## (3)Improved WGAN (WGAN-GP)
提出了一种**Improved WGAN**，即对梯度施加惩罚项（**gradient penalty**），因此该方法又称**WGAN-GP**。

$D$是$1$阶**Lipschitz**函数，等价于$D$在任意位置的梯度的模小于等于$1$。

在原目标函数的基础上，加上一个惩罚项：

$$ V(G,D) ≈ max_{D} \{ E_{x \in P_{data}}[D(x)]-E_{x \in P_{G}}[D(x)] - λ \int_{x}^{} {max(0, \mid\mid ▽_xD(x) \mid\mid -1)dx} \} $$

积分运算较为复杂，在实践中改用抽样：

$$ V(G,D) ≈ max_{D} \{ E_{x \in P_{data}}[D(x)]-E_{x \in P_{G}}[D(x)] - λ E_{x \in P_{penalty}}[max(0, \mid\mid ▽_xD(x) \mid\mid -1)] \} $$

![](https://pic.downk.cc/item/5ebe966fc2a9a83be5da9598.jpg)

P_{penalty}定义为从$$P_{data}$$和$$P_G$$中各抽取一个样本，再在其连线上抽取的样本。这样的操作是合理的，因为直观上，优化过程是使$$P_G$$靠近$$P_{data}$$，样本点大多从这两个分布之间选取，而不是整个空间。

在实践中，通常使$D$在任意位置的梯度的模近似等于$1$：

$$ V(G,D) ≈ max_{D} \{ E_{x \in P_{data}}[D(x)]-E_{x \in P_{G}}[D(x)] - λ E_{x \in P_{penalty}}[(\mid\mid ▽_xD(x) \mid\mid -1)^2] \} $$

# 9. Energy-based GAN (EBGAN)
**EBGAN**的生成器和之前的GAN类似，判别器用一个自编码器代替：

如果一幅图像经过自编码器可以被很好的还原，则判别器认为其是正样本。用负的重构误差作为判别得分。

![](https://pic.downk.cc/item/5ebe9985c2a9a83be5de5db6.jpg)

- 优点：判别器可以使用实际图像独立训练，不需要借助生成器。

在实践中，通常限制判别器对于正负样本的得分差值不超过$m$:

![](https://pic.downk.cc/item/5ebe9a22c2a9a83be5df14d8.jpg)

# 10. InfoGAN
**conditional GAN**通过输入条件$c$可以生成给定条件的图像。但是如果我们想要改变条件生成不同的图像是非常困难的，因为条件控制的输出空间是不规则的，当我们改变条件中的某一个维度，很难有确切的含义。

**InfoGAN**把生成器的输入$z$拆分成条件$c$和噪声$z'$，通过生成器获得人造样本$x$；

一方面，该样本通过一个分类器解码，用来预测之前输入的条件$c$；

另一方面，该样本通过一个判别器，用来区分是否为人造样本。

分类器和判别器的大部分参数是共享的，只有最后一层不同。

![](https://pic.downk.cc/item/5ed0a85cc2a9a83be56f2e44.jpg)

通过这种方法人为的为条件$c$对样本$x$生成设置确切的影响。

# 11. VAE-GAN
**VAE-GAN**是一种用$GAN$训练$VAE$（或用$VAE$训练$GAN$）的方法。

![](https://pic.downk.cc/item/5ed0ab11c2a9a83be571db25.jpg)

该模型包括编码器、解码器（生成器）、判别器三部分。
- **编码器**：把真实图像编码成正态分布$z$；训练目标：最小化重构误差，最小化$z$与正态分布的散度；
- **解码器（生成器）**：从$z$中抽样生成重构图像；训练目标：最小化重构误差，同时重构图像骗过判别器；
- **判别器**：训练目标：区分真实图像和重构图像。

# 12. BiGAN
**BiGAN**是一种用$GAN$训练自编码器的方法。

![](https://pic.downk.cc/item/5ed0ac9dc2a9a83be573ee7a.jpg)

该模型包括编码器、解码器、判别器三部分。
- **编码器**：把真实图像$x$编码成$z$；
- **解码器**：把$z$解码成重构图像；
- **判别器**：给定图像$x$和编码$z$，区分是编码器还是解码器提供的。

# 13. DCGAN
**DCGAN（Deep Convolutional GAN）**是一种用于图像生成的网络。

其判别器是常规的卷积网络，主要特点有：
- 去掉pooling层，使用Strided convolution (步幅卷积)
- 使用RReLU激活函数

其生成器是经过上采样的卷积网络，把噪声向量转化成一张图像，主要特点有：
- 去掉pooling层，使用Fractional-strided convolutions（分散步幅卷积）
- 输出层使用Tanh激活函数，其他层使用ReLU激活函数

![](https://pic.downk.cc/item/5ee72f5ac2a9a83be50bcecc.jpg)

# 14. SAGAN
**SAGAN(self-attention GAN)**是一种应用于图像生成的$GAN$，主要特点有：
- 使用**自注意力（self-attention）**
- 对$G$和$D$都使用**Spectral Norm（SN）**
- 训练$G$和$D$的学习率不相同（**TTUR**）

### （1）self-attention

![](https://pic.downk.cc/item/5ed864eac2a9a83be5b4cee2.jpg)

### （2）Spectral Norm
在$SNGAN$中，**Spectral Norm**被用于判别器$D$；而在$SAGAN$中，**Spectral Norm**同时用于判别器$D$和生成器$G$。

### （3）TTUR
在$SAGAN$的训练中，判别器$D$和生成器$G$的学习率是不平衡的，这种方法被称作**Two-Timescale Update Rule（TTUR）**。
- 判别器$D$的学习率设置为$0.0004$;
- 生成器$G$的学习率设置为$0.0001$.

# 15. BigGAN
**BigGAN**是在$SAGAN$的基础上使用更大的模型。包括：
- 两到四倍的参数量
- 八倍的$batch$ $size$

$BigGAN$还提出了一种**truncation trick**，用于平衡模型生成某一具体的图像或生成不同类型的图像：

具体地，在对噪声采样时进行截断，若噪声$z$的取值范围越小则生成图像越相近；反之生成的图像越丰富：

![](https://pic.downk.cc/item/5ed868d3c2a9a83be5ba48c5.jpg)

# 16. SinGAN
**SinGAN**通过使用**单张**图像来训练$GAN$。

具体的做法是将单张图像进行不同尺度的裁剪，用来生成大量的子图像，

并在每次训练时采用**progressively training**：

![](https://pic.downk.cc/item/5ed86a5ac2a9a83be5bcfa93.jpg)

# 17. GauGAN
**GauGAN**给定一张绘制图像和一张实际图像，希望能够生成具有真实图像风格的绘制图像：

![](https://pic.downk.cc/item/5ed86e88c2a9a83be5c47fae.jpg)

网络结构如下图所示，其主要步骤如下：
1. 对输入的实际图像进行编码，产生均值和方差，用来构建噪声分别；
2. 生成器接收输入的绘制图像和采样噪声，用来生成图像；
3. 判别器判断实际图像与生成图像，以及生成图像是否与绘制图像相似。

![](https://pic.downk.cc/item/5ed86cd4c2a9a83be5c1735e.jpg)

网络还提出了一种新的Normalization方法：**SPADE**，在标准化过程中引入绘制图像：

![](https://pic.downk.cc/item/5ed86c8fc2a9a83be5c0f2eb.jpg)

# 18. GANNILLA
**GANNILLA**实现了图像的**domain transform**，即把一张现实图像转换成具有儿童绘本风格的插图。

该论文提出了一个儿童绘本图像数据集，网络结构如下：

![](https://pic.downk.cc/item/5ed870b3c2a9a83be5c7d6e8.jpg)

该网络加入了一些残差连接、实例标准化和局部级联。

# 19. NICE-GAN
**NICE-GAN**提出了一种不显式的使用生成器的方法，而是使用判别器的前半部分作为生成器：

![](https://pic.downk.cc/item/5ed8710fc2a9a83be5c85b3e.jpg)

# 20. GAN的评估指标
几种评估$GAN$模型好坏的评估指标：
- Kernel Density Estimation
- Inception Score

### Kernel Density Estimation
从生成器$G$中随机采样一些样本，使用高斯混合模型拟合这些样本，得到生成样本概率分布的估计值$P_G$，

![](https://pic.downk.cc/item/5ed9e0abc2a9a83be5e94672.jpg)

选择一些真实样本$x^i$，计算其在估计的生成样本分布中的极大似然概率：

$$ L = \frac{1}{N} \sum_{i=1}^{N} {log(P_G(x^i))} $$

将这个似然值作为$GAN$模型的好坏。这种评估方法存在一些问题：
- 一些高质量样本的似然值较低：模型可能产生不同于已知样本集的高质量样本；
- 一些低质量样本的似然值较高

### Inception Score
把生成图像作为图像分类任务的样本集，进行进一步的实验；分成两步：
1. 对于一张生成的图像，将其喂入训练好的分类网络（如Inception），若分类置信度越高则图像质量越好；
![](https://pic.downk.cc/item/5ed9e362c2a9a83be5ecc47e.jpg)
2. 对于不同的生成图像，若分类网络分类后结果越平均（意味着$GAN$不会只生成某几个类）则图像质量越好；
![](https://pic.downk.cc/item/5ed9e388c2a9a83be5ecea9c.jpg)

**Inception Score**的定义是对某一具体的生成图像，分类网络的输出分布越集中越好（熵越小越好）；对于生成的所有图像，分类网络的输出分布越平均越好（熵越大越好）：

$$ \sum_{x}^{} {\sum_{y}^{} {P(y \mid x)logP(y \mid x)}} - \sum_{y}^{} {P(y)logP(y)} $$

![](https://pic.downk.cc/item/5ed9e3e9c2a9a83be5ed87a3.jpg)

# 21. LAPGAN
- LAPGAN：Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks.

在原始 GAN和CGAN中，还只能生成$16×16$,$32×32$这种低像素小尺寸的图片。

而**LAPGAN**首次实现$64×64$的图像生成。采用**coarse-to-fine**的思路，与其一下子生成这么大的（包含信息量这么多），不如一步步由小到大，这样每一步生成的时候，可以基于上一步的结果，而且还只需要“填充”和“补全”新图片所需要的那些信息。

![](https://pic.downk.cc/item/5ee7304bc2a9a83be50d3720.jpg)

# 22. StackGAN
解决文本到图像生成分辨率不高的问题，采用与LAPGAN相似的思路，采用coarse-to-fine的思路，构建两个GAN。
- 第一个GAN用于根据文本描述生成一张低分辨率的图像。
- 第二个GAN将低分辨率图像和文本作为输入，修正之前生成的图像，添加细节纹理，生成高分辨率图像。

![](https://pic.downk.cc/item/5ee7314fc2a9a83be50ed01c.jpg)
