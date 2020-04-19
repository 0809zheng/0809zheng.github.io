---
layout: post
title: '前馈神经网络'
date: 2020-04-17
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e95b7cac2a9a83be5deb0bd.jpg'
tags: 机器学习
---

> Feedforward Neural Networks，亦称多层感知机（Multi-Layer Perceptron，MLP）.

**本文目录**：
1. 模型介绍
2. 前向传播
3. 反向传播
4. 自动求导


# 1. 模型介绍
**前馈神经网络（Feedforward Neural Networks，FNN）**，亦称**多层感知机（Multi-Layer Perceptron，MLP）**。

前馈神经网络包含**输入层(input layer)**、**隐藏层(hidden layer)**和**输出层(output layer)**。计算网络层数时通常不考虑输入层。

**[通用近似定理(Universal Approximation Theorem)](https://www.researchgate.net/publication/245468707_Approximations_by_superpositions_of_a_sigmoidal_function)**:

具有非线性激活函数的单隐藏层前馈神经网络，只要其隐藏层神经元的数量足够，它可以以任意的精度来近似任何一个定义在实数空间$\Bbb{R}^D$中的有界闭集函数。

通用近似定理只是说明了神经网络的计算能力可以去近似一个给定的连续函数，但并没有给出如何找到这样一个网络，以及是否是最优的。因为神经网络的强大能力，反而容易在训练集上过拟合。

# 2. 前向传播

对于一个$L$层的前馈神经网络，引入以下记号：
1. $$M_l$$:第$l$层神经元的个数($$0≤l≤L$$);
2. $$W^{(l)} \in \Bbb{R}^{M_l×M_{l-1}}$$:第$l-1$层到第$l$层的权重矩阵;
3. $$b^{(l)} \in \Bbb{R}^{M_l}$$:第$l$层的偏置;
4. $$z^{(l)} \in \Bbb{R}^{M_l}$$:第$l$层激活函数之前的值(仿射变换的输出);
5. $$a^{(l)} \in \Bbb{R}^{M_l}$$:第$l$层激活函数之后的值(神经元的激活值),且$$x = a^{(0)}$$

则神经网络第$l$层的**前向传播(forward propagation)**过程：

$$ z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)} $$

$$ a^{(l)} = f(z^{(l)}) $$

其中$f$表示第$l$层的激活函数。

特别地，线性回归、Logistic回归或Softmax回归也可以看作只有一层的神经网络。


# 3. 反向传播

记神经网络的损失函数为$$L(y,\hat{y})$$，在神经网络的训练中经常使用**反向传播算法(back propagation, BP)**来高效地计算梯度。

记第$l$层的误差项为$$δ^{(l)}$$:

$$δ^{(l)} = \frac{\partial L(y,\hat{y})}{\partial z^{(l)}}$$

网络的输出为：

$$ \hat{y} = f^{(L)}(z^{(L)}) $$

其中$$f^{(L)}$$表示输出层的激活函数。

则反向传播算法可表示为：

$$ δ^{(L)} = \frac{\partial L(y,\hat{y})}{\partial z^{(L)}} = \frac{\partial L(y,\hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z^{(L)}} $$

$$ δ^{(l)} = \frac{\partial L(y,\hat{y})}{\partial z^{(l)}} = \frac{\partial L(y,\hat{y})}{\partial z^{(l+1)}} \frac{\partial z^{(l+1)}}{\partial z^{(l)}} = δ^{(l+1)} \frac{\partial z^{(l+1)}}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}} = {W^{(l+1)}}^T δ^{(l+1)} \ast f'(z^{(l)}) $$

$$ \frac{\partial L(y,\hat{y})}{\partial W^{(l)}} = \frac{\partial L(y,\hat{y})}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial W^{(l)}} = δ^{(l)}{a^{(l-1)}}^T $$

$$ \frac{\partial L(y,\hat{y})}{\partial b^{(l)}} = \frac{\partial L(y,\hat{y})}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial b^{(l)}} = δ^{(l)} $$

对于矩阵微积分，常用**维度检验(dimensionality check)**的方法判断公式的正确性。

注意到在误差$$δ^{(l)}$$的反向传播过程中，在每一层都要乘以该层的激活函数的导数，当激活函数存在饱和区时，容易产生**梯度消失（Vanishing Gradient）**问题。

# 4. 自动求导
目前，主流的深度学习框架都包含了自动梯度计算的功能，自动计算梯度的方法可以分为以下三类：
1. 数值微分
2. 符号微分
3. 自动微分

### (1). 数值微分
**数值微分（Numerical Differentiation）**是用数值方法来计算函数**f(x)**的导数。

$$ f'(x) = \frac{f(x+Δx)-f(x)}{Δx} $$

对$$f(x+Δx)$$在点$$x$$进行Taylor展开，可得上述计算的误差是$$O(Δx^2)$$。

在实际应用，经常使用下面公式来计算梯度:

$$ f'(x) = \frac{f(x+Δx)-f(x-Δx)}{2Δx} $$

对$$f(x+Δx)$$、$$f(x-Δx)$$在点$$x$$进行Taylor展开，可得上述计算的误差是$$O(Δx^4)$$。

数值微分由于计算复杂度高，仅用来进行**梯度检查(gradient check)**。

### (2). 符号微分
**符号微分（Symbolic Differentiation）**是一种基于**符号计算**的自动求导方法。

符号计算一般来讲是对输入的表达式，通过迭代或递归使用一些事先定义的规则进行转换。当转换结果不能再继续使用变换规则时，便停止计算。

符号计算的一个**优点**是符号计算和平台无关，可以在CPU或GPU上运行。

符号微分的**不足**：
1. 编译时间较长，特别是对于循环，需要很长时间进行编译；
2. 为了进行符号微分，一般需要设计一种专门的语言来表示数学表达式，并且要对变量（符号）进行预先声明；
3. 很难对程序进行调试。

### (3). 自动微分
**自动微分（Automatic Differentiation，AD）**是目前大多数深度学习框架的首选。

自动微分的基本原理是所有的数值计算可以分解为一些基本操作，构成一个**计算图(computational graph)**，然后利用链式法则来自动计算一个复合函数的梯度。

按照计算导数的顺序，自动微分可以分为两种模式：**前向模式**和**反向模式**。
1. **前向模式**是按计算图中计算方向的相同方向来递归地计算梯度,前向模式需要对每一个输入变量都进行一遍遍历;
2. **反向模式**是按计算图中计算方向的相反方向来递归地计算梯度,反向模式需要对每一个输出都进行一个遍历。

在前馈神经网络的参数学习中，$$\Bbb{R}^N → \Bbb{R}$$，输出为标量，因此采用反向模式为最有效的计算方式，只需要一遍计算。

计算图按构建方式可以分为**静态计算图**和**动态计算图**。

**1.静态计算图**

静态计算图是在编译时构建计算图，计算图构建好之后在程序运行时不能改变；

优点：
- 构建时可以优化；
- 并行能力强。

缺点：
- 灵活性差。

如：Theano、Tensorflow。

**2.动态计算图**是在程序运行时动态构建。

优点：
- 灵活性高。

缺点：
- 不容易优化；
- 难以并行计算。

如：PyTorch、Tensorflow 2.0。