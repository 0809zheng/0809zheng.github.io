---
layout: post
title: 'Gradients without Backpropagation'
date: 2022-02-19
author: 郑之杰
cofer: 'https://pic.imgdb.cn/item/6210ddf92ab3f51d91a169da.jpg'
tags: 论文阅读
---

> 使用前向梯度代替反向传播.

- paper：[Gradients without Backpropagation](https://arxiv.org/abs/2202.08587)

反向传播算法是优化神经网络模型的主流算法。反向传播算法，即反向模式微分，是自动微分算法的一种实现形式。本文作者提出了一种基于方向导数的梯度计算方法，通过正向模式微分实现梯度计算。该方法称为**前向梯度(forward gradient)**，是一种梯度的无偏估计。前向梯度可以从单次前向传播中计算得到，从而节省了反向传播的计算和内存，并提高了训练速度。

# 1. 自动微分

神经网络的优化通常采用基于一阶导数(梯度)的优化方法(如梯度下降法)，其中待优化变量的一阶导数通常是用**自动微分(Automatic Differentiation, AD)**实现的。

自动微分是指自动地计算函数在某点的导数值，主要原理是利用求导的链式法则。若多层神经网络表示为$y=f_n(f_{n-1}(...(f_1(x))))$，，则$y$关于$x$的导数：

$$ \frac{dy}{dx} = \frac{dy}{df_n}\frac{df_n}{dx}= \frac{dy}{df_n}\frac{df_n}{df_{n-1}}\frac{df_{n-1}}{dx} \\ =  \frac{dy}{df_n}\frac{df_n}{df_{n-1}}\cdot\cdot\cdot\frac{df_2}{df_1}\frac{df_1}{dx} $$

根据链式法则，为计算$\frac{dy}{dx}$，可以依次求出上式中的每一项微分表达式，然后将它们乘起来得到最后的结果。在具体实现时有两种实现形式：
- **前向模式(forward/tangent-linear mode)**：先计算$\frac{df_1}{dx}$，然后沿着上式乘积的反向依次求导。计算顺序与函数求值的顺序相同，可以同时计算函数值与微分。
- **反向模式(reverse/adjoint mode)**：先计算$\frac{dy}{df_n}$，然后沿着上式乘积的正向依次求导。计算顺序与函数求值的顺序相反，需要分别计算函数值与微分。

若$x \in \Bbb{R}^{n}$,$y \in \Bbb{R}^{m}$，则函数$y=f(x):\Bbb{R}^{n} \to \Bbb{R}^{m}$的导数是一个$n\times m$的线性算符，称为**Jacobian**矩阵$J_f$。若输出为标量($m=1$)，**Jacobian**矩阵退化为梯度$\nabla f(\theta)$。

对于前向模式自动微分，给定函数参数$\theta \in \Bbb{R}^{n}$和扰动(**perturbation**)向量$v \in \Bbb{R}^{n}$，可以在一次前向传播过程中同时计算函数值$f(\theta)$和**Jacobian**矢量积$J_f(\theta)v$。**Jacobian**矢量积$J_f(\theta)v$可以看作输入参数的微小扰动导致输出的变化。特别地，当$m=1$时，**Jacobian**矢量积对应方向导数$\nabla f(\theta)\cdot v$，表示参数$\theta$沿方向$v$的变化率。

对于后向模式自动微分，给定函数参数$\theta \in \Bbb{R}^{n}$和伴随(**adjoint**)向量$v \in \Bbb{R}^{m}$，通过一次前向传播过程计算函数值$f(\theta)$，再通过一次反向传播过程计算矢量**Jacobian**积$v^TJ_f(\theta)$。特别地，当$m=1$,$v=1$时，矢量**Jacobian**积对应梯度$\nabla f(\theta)=[\frac{\partial f}{\partial \theta_1},...,\frac{\partial f}{\partial \theta_n}]^T$。

![](https://pic.imgdb.cn/item/6210966c2ab3f51d91526247.jpg)

两种自动微分模式各有不同的适用范围。
- 前向模式假设待求导函数的自变量的导数$\frac{df_1}{dx}$是已知的，每次自动微分可以求出所有相关节点关于一个自变量分量$x_i$的导数$\frac{d \cdot}{dx_i}$，若$x \in \Bbb{R}^{n}$，则需要执行$n$次自动求导以获得完整的**Jacobian**矩阵。
- 反向模式假设待求导函数的因变量的导数$\frac{dy}{df_n}$是已知的，每次自动微分可以求出函数$y_i(x)$关于所有相关节点的导数$\frac{dy_i}{d \cdot}$，若$y \in \Bbb{R}^{m}$，则需要执行$m$次自动求导以获得完整的**Jacobian**矩阵。

在实际问题中，输入特征维度$n$通常远大于输出维度$m$；特别地，大部分神经网络中输出通常是标量值($m=1$)，而输入维度特别高。因此采用反向模式进行自动微分，以降低微分过程的时间复杂度，此即反向传播算法。

由于反向模式需要正向求函数值再反向求导数，这一过程通过计算图实现：将函数求值过程中的所有中间节点按求值顺序存储到栈中，得到表达式对应的计算图；然后再依次弹出栈中的元素，求相应的导数。因此反向模式具有较高的空间复杂度。

# 2. 前向梯度

给定函数$f:\Bbb{R}^{n} \to \Bbb{R}$，**前向梯度(forward gradient)** $g:\Bbb{R}^{n} \to \Bbb{R}^{n}$定义为：

$$ g(\theta) = (\nabla f(\theta)\cdot v) v $$

其中$\theta \in \Bbb{R}^{n}$是计算梯度的参数点，$v \in \Bbb{R}^{n}$是一个用多元随机变量表示的干扰向量，$v$的每一个元素具有零均值和单位方差（通常把$v$建模为标准正态分布$v\text{~}\mathcal{N}(0,I)$）。$\nabla f(\theta)\cdot v \in \Bbb{R}$表示在$\theta$点指向$v$的方向导数。

前向模式自动微分可以得到方向导数$\nabla f(\theta)\cdot v = \sum_{i}^{}\frac{\partial f}{\partial \theta_i}v_i$。注意到如果$v$取一组单位正交基，则可以分别得到$f$对于每个分量$\theta_i$的梯度$\frac{\partial f}{\partial \theta_i}$，从而进一步得到总梯度值$\nabla f$，这需要$n$次前向传播过程。

如果仅进行一次前向传播，则$v$可以看作是对每个分量$\theta_i$的梯度$\frac{\partial f}{\partial \theta_i}$进行加权的权重向量。因此可以将梯度加权值$\sum_{i}^{}\frac{\partial f}{\partial \theta_i}v_i$按照权重向量$v$归还给每一个参数分量$\theta_i$，参数$\theta_i$获得的梯度估计值与其权重$v_i$成比例：$(\sum_{i}^{}\frac{\partial f}{\partial \theta_i}v_i) v_i$。因此前向梯度是对梯度的一种估计。

前向梯度的计算步骤如下：
- 从随机干扰向量中采样$v\text{~}p(v)$；
- 通过单次前向微分计算$f(\theta)$和$\nabla f(\theta)\cdot v$；
- 将标量的方向导数$\nabla f(\theta)\cdot v$和向量$f(\theta)$相乘得到前向梯度。

下图展示了扰动向量(橙色)转变成前向梯度(蓝色)的过程，结果显示平均前向梯度(绿色)与真实梯度非常接近。

![](https://pic.imgdb.cn/item/62109f5f2ab3f51d915a89f2.jpg)

事实上，前向梯度$g(\theta)$是真实梯度$\nabla f(\theta)$的**无偏估计(unbiased estimator)**。下面给出证明。

将方向导数写作：

$$ d(\theta,v)=\nabla f(\theta)\cdot v = \sum_{i}^{}\frac{\partial f}{\partial \theta_i}v_i \\ = \frac{\partial f}{\partial \theta_1}v_1+\frac{\partial f}{\partial \theta_2}v_2+...+\frac{\partial f}{\partial \theta_n}v_n $$

则前向梯度写作：

$$ g(\theta) = d(\theta,v) v \\ = \begin{bmatrix} \frac{\partial f}{\partial \theta_1}v_1^2+\frac{\partial f}{\partial \theta_2}v_1v_2+...+\frac{\partial f}{\partial \theta_n}v_1v_n \\ \frac{\partial f}{\partial \theta_1}v_1v_2+\frac{\partial f}{\partial \theta_2}v_2^2+...+\frac{\partial f}{\partial \theta_n}v_2v_n \\ \cdots\\ \frac{\partial f}{\partial \theta_1}v_1v_n+\frac{\partial f}{\partial \theta_2}v_2v_n+...+\frac{\partial f}{\partial \theta_n}v_n^2 \end{bmatrix} $$

记其第$i$行元素为：

$$ g_i(\theta) = \frac{\partial f}{\partial \theta_i}v_i^2 + \sum_{j≠i}^{}\frac{\partial f}{\partial \theta_j}v_iv_j $$

前向梯度第$i$行元素的期望为：

$$ \Bbb{E}[g_i(\theta)] = \Bbb{E}[\frac{\partial f}{\partial \theta_i}v_i^2 + \sum_{j≠i}^{}\frac{\partial f}{\partial \theta_j}v_iv_j] \\ = \Bbb{E}[\frac{\partial f}{\partial \theta_i}v_i^2] + \Bbb{E}[\sum_{j≠i}^{}\frac{\partial f}{\partial \theta_j}v_iv_j]\\ = \Bbb{E}[\frac{\partial f}{\partial \theta_i}v_i^2] + \sum_{j≠i}^{}\Bbb{E}[\frac{\partial f}{\partial \theta_j}v_iv_j] \\ = \frac{\partial f}{\partial \theta_i}\Bbb{E}[v_i^2] + \frac{\partial f}{\partial \theta_j}\sum_{j≠i}^{}\Bbb{E}[v_iv_j] $$

注意到随机变量$v$的分量独立，且满足$\Bbb{E}[v]=0,\text{Var}[v]=1$，则有：
- $\Bbb{E}[v_i^2]=\text{Var}[v_i]+\Bbb{E}[v_i]^2=1$
- $\Bbb{E}[v_iv_j]=\Bbb{E}[v_i]\Bbb{E}[v_j]=0$

因此：

$$ \Bbb{E}[g_i(\theta)]  = \frac{\partial f}{\partial \theta_i} \\ \Bbb{E}[g(\theta)]  = \nabla f(\theta) $$

# 3. 前向梯度下降

仿照标准的梯度下降算法，作者构造了基于前向梯度的**前向梯度下降(forward gradient descent, FGD)**算法。该算法使用前向梯度代替了反向传播计算的梯度，从而省去反向传播的计算过程。

![](https://pic.imgdb.cn/item/6210a6302ab3f51d9160d8c5.jpg)

# 4. 实验分析

## (1) 优化轨迹分析

为了可视化分析优化过程的轨迹，作者选取了两个二维目标函数：
- **Beale**函数：$f(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2$
- **Rosenbrock**函数：$f(x,y)=(a-x)^2+b(y-x^2)^2,a=1,b=100$

图中$10$次优化轨迹表明前向梯度的整体优化行为与反向传播是类似的，前向梯度在每次迭代过程中速度更快。

![](https://pic.imgdb.cn/item/6210a7e02ab3f51d9162305e.jpg)

## (2) 运行时间与损失分析

作者定义$R_f$和$R_b$分别为使用基于前向梯度和反向传播的算法训练相同的网络结构所需要的时间，这些时间包括梯度计算和参数更新等过程。
作者还定义$T_f$和$T_b$分别为使用基于前向梯度和反向传播的算法实现最低的验证集损失所需要的时间。

在训练逻辑回归模型时，$R_f/R_b$和$T_f/T_b$是一致的，均表明前向梯度的方法的速度大约是反向传播方法的两倍。且两个比例的一致性表明几乎每次更新都对损失变化产生有用的影响。

![](https://pic.imgdb.cn/item/6210aa832ab3f51d91644fb5.jpg)

在训练多层感知机时，当学习率较小($2\times 10^{-5}$，下图第一行)时，其表现与逻辑回归模型类似，表明前向梯度的速度大约是反向传播的两倍。当学习率较大($2\times 10^{-4}$，下图第二行)时，前向梯度实现更快的下降，大约是反向传播的$4$倍。作者认为前向梯度引入的噪声可能有助于探索损失曲面。

![](https://pic.imgdb.cn/item/6210ab5e2ab3f51d91650a59.jpg)

在训练卷积神经网络时，结果也表明前向梯度的速度大约是反向传播的两倍。

![](https://pic.imgdb.cn/item/6210abf12ab3f51d91657eeb.jpg)

下图表明当网络层数加深时，前向梯度的速度相比于反向传播的速度仍然具有优势，尽管两者的内存消耗几乎是一样的。

![](https://pic.imgdb.cn/item/6210ac9e2ab3f51d9166030d.jpg)