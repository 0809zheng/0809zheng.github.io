---
layout: post
title: '深度学习中的Activation Function'
date: 2020-03-01
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e7b4db6504f4bcb040071f1.png'
tags: 深度学习
---

> Activation Functions in Deep Learning.

1. Background
2. Step
3. Sigmoid
4. Tanh
5. Hard Sigmoid & Hard Tanh
6. Softplus
7. ReLU
8. Leaky ReLU
9. PReLU
10. RReLU
11. Maxout
12. ELU
13. CELU
14. SELU
15. GELU
16. Swish
17. Mish

# 1. Background
封面图是[神经网络中一个神经元的简单建模](http://cs231n.github.io/neural-networks-1/)。

大脑中的**神经元（neuron）**通过**树突（dendrites）**接收输入信号，在胞体中进行信号的处理，通过**轴突（axon）**分发信号。当神经元中的信号累积达到一定阈值时产生电脉冲将信号输出，这个阈值称为**点火率（firing rate）**。

在神经网络中，使用**激活函数（activation function）**对点火率进行建模。激活函数能够为网络引入非线性表示；当不使用激活函数时（或激活函数为**恒等函数 identity function**），多层神经网络实质相当于单层：

$$ W_2(W_1X+b_1)+b_2=W_2W_1X+W_2b_1+b_2=(W_2W_1)X+(W_2b_1+b_2) $$

值得一提的是，这种对神经元的建模是**coarse**的。真实神经元有很多不同的种类；突触是一个复杂的的非线性动态系统，树突进行的是复杂的非线性运算，轴突的输出时刻也很重要。近些年来神经网络中神经元的**生物可解释性（Biological Plausibility）**被逐渐弱化。

# 2. Step
最早作为激活函数被使用的是[**阶跃函数**](https://en.wikipedia.org/wiki/Heaviside_step_function)。1943年，Warren McCulloch和Walter Pitts将其应用在[**MP神经元模型**](https://books.google.com/books?id=qOy4yLBqhFcC&pg=PA3)中。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/1280px-Dirac_distribution_CDF.svg.png)

# 3. Sigmoid
**Sigmoid函数**也叫**Logistic函数**，函数及其导数表达式如下：

$$ sigmoid(x)= \frac{1}{1+e^{-x}} $$

$$ sigmoid'(x)= sigmoid(x)(1-sigmoid(x)) $$

![](https://pytorch.org/docs/stable/_images/Sigmoid.png)

优点：
1. 连续可导；
2. 输出为范围$[0,1]$的归一化结果，可以作为概率分布。

缺点：
1. 存在饱和区，当$ \| x \| >> 0 $时梯度接近0，产生**梯度消失 vanishing gradient**；
2. 输出不是**zero-centered**，使得后一层神经元的输入发生**偏置偏移(bias shift)**，减慢梯度下降的收敛速度;
3. 存在**指数**运算，计算量大。

# 4. Tanh
**Tanh**全称是**双曲正切函数（hyperbolic tangent function）**，函数及其导数表达式如下：

$$ tanh(x)= \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=2sigmoid(2x)-1 $$

$$ tanh'(x)= 1-tanh^2(x) $$

![](https://pytorch.org/docs/stable/_images/Tanh.png)

**Tanh**函数是**zero-centered**，但仍然存在**vanishing gradient**和**指数**运算带来的问题。

# 5. Hard-Sigmoid & Hard-Tanh
**Hard-Sigmoid**和**Hard-Tanh**是对**Sigmoid**和**Tanh**函数的分段线性表示，以减少计算开销。

**Sigmoid**函数在x=0附近的一阶**Taylor**展开：

$$ sigmoid(x)≈ \frac{1}{2}+ \frac{x}{4} $$

**Tanh**函数在x=0附近的一阶**Taylor**展开：

$$ tanh(x)≈ x $$

可将**Sigmoid**和**Tanh**函数分段表示为：
![](https://pic.downk.cc/item/5e7b5a94504f4bcb0408006f.png)

# 6. Softplus
- paper：[Incorporating second-order functional knowledge for better option pricing](https://www.researchgate.net/publication/4933639_Incorporating_Second-Order_Functional_Knowledge_for_Better_Option_Pricing)

**Softplus**函数及其导数表达式如下：

$$ softplus(x)=log(1+e^x) $$

$$ softplus'(x)=sigmoid(x) $$

![](https://pytorch.org/docs/stable/_images/Softplus.png)

**Softplus**具有**单侧抑制**、**宽兴奋边界**($0$,$+∞$)的特点，但没有稀疏激活性。

# 7. ReLU
- paper：[Rectified linear units improve restricted boltzmann machines](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

**ReLU**全称是**rectified linear unit**，表达式如下：

$$
        relu(x) =
        \begin{cases}
        x,  & \text{if $x≥0$} \\
        0, & \text{if $x<0$}
        \end{cases}
$$

![](https://pytorch.org/docs/stable/_images/ReLU.png)

优点：
1. 计算简单，只存在线性运算；
2. 左饱和函数，x>0时梯度为1，一定程度上缓解了**vanishing gradient**；
3. 具有一定的**生物解释性**，如单侧抑制、宽兴奋边界（兴奋程度可以很高）；
4. 具有**稀疏性**，大约激活50%的神经元。

缺点：
1. 输出不是**zero-centered**，引入偏置偏移；
2. 容易产生**dead ReLU**，在反向传播过程中，如果输入负数，则梯度将完全为零，即有些神经元永远不会被激活。

# 8. Leaky ReLU
- paper：[Rectifier nonlinearities improve neural network acoustic models](https://www.mendeley.com/catalogue/a4a3dd28-b56b-3e0c-ac53-2817625a2215/)

**Leaky ReLU**的表达式如下：

$$
        leakyrelu(x) =
        \begin{cases}
        x,  & \text{if $x≥0$} \\
        0.1x, & \text{if $x<0$}
        \end{cases}
$$

![](https://pytorch.org/docs/stable/_images/LeakyReLU.png)

**Leaky ReLU**在x<0时也有一定的梯度，是专门设计用于解决**dead ReLU**问题的激活函数。

# 9. PReLU
- paper：Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
- arXiv：[https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)

**PReLU**全称是**parametric ReLU**，表达式如下：

$$
        prelu(x) =
        \begin{cases}
        x,  & \text{if $x≥0$} \\
        γx, & \text{if $x<0$}
        \end{cases}
$$

![](https://pytorch.org/docs/stable/_images/PReLU.png)

$γ$是可学习的参数，通常取$0$到$1$；**PReLU**允许不同神经元具有不同的参数。

# 10. RReLU
- paper：Empirical Evaluation of Rectified Activations in Convolutional Network
- arXiv：[https://arxiv.org/abs/1505.00853](https://arxiv.org/abs/1505.00853)

**RReLU**全称是**Randomized Leaky ReLU**，表达式如下：

$$
        rrelu(x) =
        \begin{cases}
        x,  & \text{if $x≥0$} \\
        ax, & \text{if $x<0$}
        \end{cases}
$$

参数$a$是从均匀分布$U(l,u)$中抽样得到的，$0≤l<u<1$。
![](https://pic.downk.cc/item/5e7defb3504f4bcb047b21b5.png)

测试时：

$$ y=\frac{l+u}{2} x $$

# 11. Maxout
- paper：Maxout networks
- arXiv：[https://arxiv.org/abs/1302.4389](https://arxiv.org/abs/1302.4389)

**Maxout**是一种分段线性单元。**ReLU**的输入是隐藏层的单个神经元，**Maxout**的输入是隐藏层的$k$个神经元：

$$ maxout(x)=max_k(x_k) $$

**Maxout**节点可以看作是任意凸函数的分段线性近似：
![](http://img.mp.sohu.com/upload/20170604/b24a3622a6bb4b349fa90833f4eaaac6_th.png)

# 12. ELU
- paper：Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
- arXiv：[https://arxiv.org/abs/1511.07289](https://arxiv.org/abs/1511.07289)

**ELU**全称是**exponential linear unit**，表达式如下：

$$
        elu(x) =
        \begin{cases}
        x,  & \text{if $x≥0$} \\
        α(e^x-1), & \text{if $x<0$}
        \end{cases}
$$

![](https://pytorch.org/docs/stable/_images/ELU.png)

$α≥0$是可学习的参数；**ELU**是近似**zero-centered**的，对噪声**robust**(函数是连续的)，但是引入了指数运算。

# 13. CELU
- paper：Continuously Differentiable Exponential Linear Units
- arXiv：[https://arxiv.org/abs/1704.07483](https://arxiv.org/abs/1704.07483)

**CELU**全称是**continuously differentiable exponential linear unit**，表达式如下：

$$
        celu(x) =
        \begin{cases}
        x,  & \text{if $x≥0$} \\
        α(e^{\frac{x}{α}}-1), & \text{if $x<0$}
        \end{cases}
$$

![](https://pytorch.org/docs/stable/_images/CELU.png)

# 14. SELU
- paper：Self-Normalizing Neural Networks
- arXiv：[https://arxiv.org/abs/1706.02515](https://arxiv.org/abs/1706.02515)

**SELU**全称是**scaled exponential linear unit**，表达式如下：

$$
        selu(x) =
        \begin{cases}
        scale*x,  & \text{if $x≥0$} \\
        scale*α(e^x-1), & \text{if $x<0$}
        \end{cases}
$$

其中：

$$ α=1.6732632423543772848170429916717 $$

$$ scale=1.0507009873554804934193349852946 $$

$scale$和$α$的计算参考[原作者](https://github.com/bioinf-jku/SNNs/blob/master/Calculations/SELU_calculations.pdf)。主要思想是一组均值0、方差1的i.i.d.随机变量通过SELU后仍然为均值0、方差1。

![](https://pytorch.org/docs/stable/_images/SELU.png)

# 15. GELU
- paper：Gaussian Error Linear Units (GELUs)
- arXiv：[https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)

**GELU**全称是**Gaussian error linear unit**，表达式如下：

$$ gelu(x)=xP(X≤x) $$

其中P(X≤x)是高斯分布$ N(μ,σ^2) $的累积分布函数（CDF）。

GELU是一种通过门控机制调整其输出值的激活函数。
![](https://pytorch.org/docs/stable/_images/GELU.png)

**GELU**可用Tanh函数或Logistic函数近似：

$$ gelu(x)≈0.5x(1+tanh(\sqrt{\frac{2}{\pi}(x+0.044715x^3)})) $$

$$ gelu(x)≈xσ(1.702x) $$

# 16. Swish
- paper：Searching for Activation Functions
- arXiv：[https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)

**Swish**函数表达式如下，其中$β$是可学习的参数：

$$ switsh(x)=x·sigmoid(βx) $$

**Swish**函数可以看作一种软性的门控机制，当$sigmoid(βx)$接近于$1$时门“开”；当$sigmoid(βx)$接近于$0$时门“关”。

$β=0$时，**Switsh**函数退化成线性函数；$β->∞$时，**Switsh**函数退化成**ReLU**。
![](https://pic.downk.cc/item/5e7df942504f4bcb048376a9.png)

**Swish**函数受**LSTM**中**门控(gating)**机制的启发，使用自身的值作为门控，即**self-gating**。

# 17. Mish
- paper：A Self Regularized Non-Monotonic Neural Activation Function
- arXiv：[https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)

**Mish**函数表达式如下：

$$ mish(x)=x·tanh(softplus(x)) $$

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/KYSDTmOVZvoq7ibpibADnKetBktT61NOiaj7icib3n2mQwNeJcS2D1EHkBAmZUDicK7ib0PiaGriaUxj7Bxxw1A9SHJibntA/640?wx_fmt=jpeg)

下面是一个随机网络使用**ReLU**、**Swish**和**Mish**后的输出图，可以看出后两个比**ReLU**更加平滑。
![](https://pic.downk.cc/item/5e7edcce504f4bcb0425d2fe.png)