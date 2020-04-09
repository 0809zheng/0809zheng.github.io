---
layout: post
title: '深度学习中的Initialization'
date: 2020-03-05
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e8ed4c7504f4bcb0429f47f.jpg'
tags: 深度学习
---

> Initialization in Deep Learning.

对神经网络进行训练时，需要对神经网络的参数进行初始化。

常见的初始化方法：
1. Zero Initialization
2. Random Initialization
3. Xavier Initialization
4. He Initialization
5. Orthogonal Initialization

# 1. Zero Initialization
在传统的机器学习算法（比如感知器和 Logistic 回归）中，一般将参数全部初始化为0。但是这在神经网络的训练中会存在一些问题。

如果参数都为0，在第一遍前向计算时，所有的隐层神经元的激活值都相同；在反向传播时，所有权重的更新也都相同，这样会导致隐藏层神经元没有区分性。这种现象称为**对称权重**现象。

对于一些特殊的参数，我们可以根据经验用一个特殊的固定值来进行初始化：
- 偏置（Bias）通常用0来初始化；
- 在LSTM网络的遗忘门中，偏置通常初始化为1或2，使得时序上的梯度变大；
- 对于使用ReLU的神经元，有时也可以将偏置设为0.01，使得ReLU神经元在训练初期更容易激活。

对于神经网络的权重矩阵，选用一些随机的初始化方法**打破对称性(Symmetry breaking)**。

# 2. Random Initialization
**随机初始化(Random Initialization)**是从一个固定均值（通常为0）和方差$σ^2$的分布中采样来生成参数的初始值。

(1) 高斯分布初始化

使用高斯分布$N(0,σ^2)$对参数进行随机初始化。

(2) 均匀分布初始化

使用均匀分布$U(a,b)$对参数进行随机初始化，且方差$σ^2$满足：

$$ σ^2 = \frac{(b-a)^2}{12} $$

随机初始化的关键是设置方差$σ^2$的大小。
- 如果方差过小，会导致神经元的输出过小，经过多层之后信号慢慢消失了；还会使Sigmoid型激活函数丢失非线性能力；
- 如果方差过大，会导致神经元的输出过大，还会使Sigmoid型激活函数进入饱和区，产生vanishing gradient。

# 3. Xavier Initialization
初始化一个深度网络时，为了缓解梯度消失或爆炸问题，尽可能保持每个神经元的输入和输出的方差一致。

[**Xavier初始化**](https://www.researchgate.net/publication/215616968_Understanding_the_difficulty_of_training_deep_feedforward_neural_networks)根据每层的神经元数量来自动计算初始化参数的方差。

假设第$l$层的一个神经元$a^{(l)}$，接收前一层的$M_{l-1}$个神经元的输出$$a^{(l-1)}$$，

$$ a^{(l)} = f(\sum_{i=1}^{M_{l-1}} {w_i^{(l)}a_i^{(l-1)}}) $$

此处假设偏置$b$初始化为0，$f$为激活函数。

假设$f$为恒等函数，且$w_i^{(l)}$和$a_i^{(l-1)}$均值为0，互相独立，则$a^{(l)}$的均值为：

$$ E(a^{(l)}) = E(\sum_{i=1}^{M_{l-1}} {w_i^{(l)}a_i^{(l-1)}}) = \sum_{i=1}^{M_{l-1}} {E(w_i^{(l)})E(a_i^{(l-1)})} = 0 $$

$a^{(l)}$的方差为：

$$ Var(a^{(l)}) = Var(\sum_{i=1}^{M_{l-1}} {w_i^{(l)}a_i^{(l-1)}}) = \sum_{i=1}^{M_{l-1}} {Var(w_i^{(l)})Var(a_i^{(l-1)})} = M_{l-1}Var(w_i^{(l)})Var(a_i^{(l-1)}) $$

输入信号的方差在经过该神经元后被缩放了$$M_{l-1}Var(w_i^{(l)})$$倍。

为了使得在经过多层网络后，信号不被过分放大或过分减弱，尽可能保持每个神经元的输入和输出的方差一致，则有：

$$ Var(w_i^{(l)}) = \frac{1}{M_{l-1}} $$

同理，为了使得在反向传播中，误差信号也不被放大或缩小，需要将$w_i^{(l)}$的方差保持为：

$$ Var(w_i^{(l)}) = \frac{1}{M_{l}} $$

作为折中，同时考虑信号在前向和反向传播中都不被放大或缩小，可以设置：

$$ Var(w_i^{(l)}) = \frac{2}{M_{l-1}+M_{l}} $$

在上述推导中假设激活函数为恒等函数，Xavier初始化也适用于Sigmoid函数和Tanh函数，这是因为神经元的参数和输入的绝对值通常比较小，处于激活函数的线性区间。

由于Sigmoid函数在线性区的斜率约为$\frac{1}{4}$，因此其参数初始化的方差为$$ Var(w_i^{(l)}) = 16 × \frac{2}{M_{l-1}+M_{l}} $$。

# 4. He Initialization
[**He初始化**](https://arxiv.org/abs/1502.01852)也称为Kaiming初始化。

当第$l$层神经元使用ReLU激活函数时，通常有一半的神经元输出为0，因此其分布的方差也近似为使用恒等函数时的一半。

只考虑前向传播时，参数$w_i^{(l)}$的理想方差为：

$$ Var(w_i^{(l)}) = \frac{2}{M_{l-1}} $$

Xavier初始化和He初始化的具体设置情况如下表：

![](https://pic.downk.cc/item/5e8ecce6504f4bcb0421c0af.jpg)

# 5. Orthogonal Initialization
[**正交初始化（Orthogonal Initialization）**]()是将$W^{(l)}$初始化为正交矩阵，即：

$$ W^{(l)}{W^{(l)}}^T = I $$

实现过程：
1. 用均值为0、方差为1的高斯分布初始化一个矩阵;
2. 将这个矩阵用奇异值分解得到两个正交矩阵，并使用其中之一作为权重矩阵。

正交初始化使误差项在反向传播中具有**范数保持性(Norm-Preserving)**。对于误差项$$δ^{(l-1)} = {W^{(l)}}^T δ^{(l)}$$，满足：

$$ \mid\mid δ^{(l-1)} \mid\mid^2 = \mid\mid {W^{(l)}}^T δ^{(l)} \mid\mid^2 = \mid\mid δ^{(l)} \mid\mid^2 $$

当在非线性神经网络中应用正交初始化时，通常需要将正交矩阵乘以一个缩放系数$ρ$。

比如当激活函数为ReLU时，激活函数在0附近的平均梯度可以近似为0.5。为了保持范数不变，缩放系数$ρ$可以设置为$\sqrt{2}$。

正交初始化通常用在循环神经网络中循环边上的权重矩阵上。