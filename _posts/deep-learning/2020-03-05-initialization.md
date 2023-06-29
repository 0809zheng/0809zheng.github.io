---
layout: post
title: '深度学习中的初始化方法(Initialization)'
date: 2020-03-05
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e8ed4c7504f4bcb0429f47f.jpg'
tags: 深度学习
---

> Initialization in Deep Learning.

对神经网络进行训练时，需要对神经网络的参数进行初始化。对于深度网络来说，参数的初始化显得尤为重要。糟糕的初始化不仅会使模型效果变差，还有可能使得模型根本训练不动或者不收敛。

在网络中，如果参数的初始化数值过小，则随着网络层数加深，输出激活值$h^l = Wh^{l-1}$趋近于$0$，反向传播的梯度（$\propto h^l$）也趋近于$0$，使得网络无法学习。如果参数的初始化数值过大，经过一些带有饱和区的激活函数（如**sigmoid**、**tanh**）后，会使网络激活值趋近于饱和（$\Delta h \to 0$），从而产生**vanishing gradient**。

![](https://pic.downk.cc/item/5e8ed4c7504f4bcb0429f47f.jpg)

在**PyTorch**中，可以在定义网络时为其每个模块（如卷积层、**BatchNorm**）指定初始化类型：

```python
class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        # 定义模型

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 前向传播
```

也可以在实例化网络后，对其中的模块进行初始化：

```python
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

model = Model()
weights_init(model)
```

本文介绍一些常见的初始化方法，包括零初始化、随机初始化、恒等初始化、**Xavier**初始化、**Kaiming**初始化、正交初始化、稀疏初始化。


# 1. 零初始化 Zero Initialization
在传统的机器学习算法（比如感知机和**Logistic**回归）中，一般将参数全部初始化为$0$。但是这在神经网络的训练中会存在一些问题。

如果参数都为$0$，在第一遍前向计算时，所有的隐藏层神经元的激活值都相同（不一定为$0$，取决于激活函数在$0$处的值）；在反向传播时，所有权重的更新也都相同，这样会导致隐藏层神经元没有区分性。这种现象称为**对称权重**。

对于网络中的一些特殊参数，我们可以根据经验用一个特殊的固定值来进行初始化：
- 偏置（**Bias**）通常用$0$来初始化；
- 在**LSTM**网络的遗忘门中，偏置通常初始化为$1$或$2$，使得时序上的梯度变大；
- 对于使用**ReLU**的神经元，也可以将偏置设为$0.01$，使得**ReLU**神经元在训练初期更容易激活。

```python
torch.nn.init.zeros_(tensor)         # 初始化为0
torch.nn.init.ones_(tensor)          # 初始化为1
torch.nn.init.constant_(tensor, val) # 初始化为常数val
```

对于神经网络的权重（**Weight**）矩阵，选用一些随机的初始化方法**打破对称性(Symmetry breaking)**。

# 2. 随机初始化 Random Initialization
**随机初始化**是指从一个固定均值（通常为$0$）和方差$σ^2$的分布中随机采样来生成参数的初始值。

### (1) 高斯分布初始化

使用高斯分布$N(0,σ^2)$对参数进行随机初始化。

```python
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
```

### (2) 均匀分布初始化

使用均匀分布$U(a,b)$对参数进行随机初始化，且方差$σ^2$满足：

$$ σ^2 = \frac{(b-a)^2}{12} $$

```python
torch.nn.init.uniform_(tensor, a=0.0, b=1.0)
```

随机初始化的关键是设置方差$σ^2$的大小。
- 如果方差过小，会导致神经元的输出过小，经过多层之后信号慢慢消失了；还会使**Sigmoid**型激活函数丢失非线性能力；
- 如果方差过大，会导致神经元的输出过大，还会使**Sigmoid**型激活函数进入饱和区，产生**vanishing gradient**。

### (3) 截断的高斯分布初始化

使用高斯分布$N(\mu,σ^2)$对参数进行随机初始化，并将数值截断在$[a,b]之间$。

```python
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=- 2.0, b=2.0)
```

# 3. 恒等初始化 Identity Initialization

**恒等初始化**是指通过把参数设置为恒等变换，使得网络层的输出值与输出值相等。

对于二维参数（如全连接层的权重参数），通过单位矩阵进行恒等初始化：

```python
torch.nn.init.eye_(tensor)
```

对于高维参数（如卷积层的权重矩阵），通过**Dirac-delta**函数进行恒等初始化：

```python
torch.nn.init.dirac_(tensor, groups=1)
```

# 4. Xavier初始化 Xavier Initialization

- paper：[Understanding the difficulty of training deep feedforward neural networks](https://www.researchgate.net/publication/215616968_Understanding_the_difficulty_of_training_deep_feedforward_neural_networks)

初始化一个神经网络时，为了缓解梯度消失或爆炸问题，应尽可能保持每个神经元输入和输出的方差一致。标准的初始化策略是基于**概率统计**的，即假设输入数据分布的均值为$0$、方差为$1$，期望输出数据分布也保持均值为$0$、方差为$1$，然后推导初始变换应满足的均值与方差条件。

**Xavier初始化**是由**Xavier Glorot**提出的，因此也称为**Glorot**初始化。该初始化方法假设参数的初始化均值为$0$，根据每层的神经元数量来自动计算初始化参数的方差。

假设第$l$层的一个神经元$z^{(l)}$，接收前一层的$d_{l-1}$个神经元的输出$z^{(l-1)}$，

$$ z^{(l)} = \sum_{i=1}^{d_{l-1}} {w_i^{(l)}a_i^{(l-1)} } = \sum_{i=1}^{d_{l-1}} {w_i^{(l)}f\left(z_i^{(l-1)}\right) }$$

此处假设偏置$b$初始化为$0$，$f(\cdot)$为激活函数。

## （1）无激活函数

假设$f(\cdot)$为恒等函数(即无激活函数)，且各$w_i^{(l)}$和$z_i^{(l-1)}$均值为$0$、互相独立，则$z^{(l)}$的均值为：

$$ \mathbb{E}[z^{(l)}] = \mathbb{E}[\sum_{i=1}^{d_{l-1}} {w_i^{(l)}z_i^{(l-1)}}] = \sum_{i=1}^{d_{l-1}} {\mathbb{E}[w_i^{(l)}]\mathbb{E}[z_i^{(l-1)}]} = 0 $$

$z^{(l)}$的方差为：

$$
\begin{aligned}
Var[z^{(l)}] &= Var[\sum_{i=1}^{d_{l-1}} {w_i^{(l)}z_i^{(l-1)}}] = \sum_{i=1}^{d_{l-1}} Var[{w_i^{(l)}z_i^{(l-1)}}]\\
&= \sum_{i=1}^{d_{l-1}} \mathbb{E}[({w_i^{(l)}z_i^{(l-1)}})^2]-(\mathbb{E}[{w_i^{(l)}z_i^{(l-1)}}])^2 \\
&= \sum_{i=1}^{d_{l-1}} \mathbb{E}[(w_i^{(l)})^2]\mathbb{E}[(z_i^{(l-1)})^2]-(\mathbb{E}[{w_i^{(l)}z_i^{(l-1)}}])^2 \\
&= \sum_{i=1}^{d_{l-1}} \left(Var[w_i^{(l)}] +(\mathbb{E}[w_i^{(l)}])^2\right)\left(Var[z_i^{(l-1)}] +(\mathbb{E}[z_i^{(l-1)}])^2\right)-(\mathbb{E}[{w_i^{(l)}z_i^{(l-1)}}])^2 \\
&= \sum_{i=1}^{d_{l-1}} {Var[w_i^{(l)}]Var[z_i^{(l-1)}]} + Var[w_i^{(l)}](\mathbb{E}[z_i^{(l-1)}])^2+Var[z_i^{(l-1)}] (\mathbb{E}[w_i^{(l)}])^2\\
&= d_{l-1}Var[w_i^{(l)}]Var[z_i^{(l-1)}]
\end{aligned}
$$

即输入信号的方差在经过该神经元后被缩放了$d_{l-1}Var[w_i^{(l)}]$倍。

为了使得在经过多层网络后，信号不被过分放大或过分减弱，尽可能保持每个神经元的输入和输出的方差一致，则有：

$$ Var[w_i^{(l)}] = \frac{1}{d_{l-1}} $$

同理，为了使得在[反向传播](https://0809zheng.github.io/2020/04/17/feedforward-neural-network.html#3-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD)中，误差信号$$\delta^{(l-1)} \propto \sum_{j=1}^{d_l}w_j^{(l)} \cdot \delta_j^{(l)}$$也不被放大或缩小，需要将$w_i^{(l)}$的方差保持为：

$$ Var[w_i^{(l)}] = \frac{1}{d_{l}} $$

作为折中，同时考虑信号在前向和反向传播中都不被放大或缩小，可以设置：

$$ Var[w_i^{(l)}] = \frac{2}{d_{l-1}+d_{l}} $$

因此把参数初始化为均值为$0$、方差为$\frac{2}{d_{l-1}+d_{l}}$，其中$d_{l-1}$和$d_l$分别为前一层和当前层神经元的数量。

## （2）有激活函数

在上述推导中假设激活函数为恒等函数。对于一般的激活函数$f(\cdot)$，会改变输入数据$z^{(l-1)}$的分布形式。当激活函数能够近似线性化时，不会改变输入分布的均值(仍然为$0$)；此时只需考虑方差的变化。

$$
\begin{aligned}
Var[z^{(l)}] &= Var[\sum_{i=1}^{d_{l-1}} {w_i^{(l)}f\left(z_i^{(l-1)}\right)}] = \sum_{i=1}^{d_{l-1}} Var[{w_i^{(l)}f\left(z_i^{(l-1)}\right)}]\\
&= \sum_{i=1}^{d_{l-1}} {Var[w_i^{(l)}]Var[f\left(z_i^{(l-1)}\right)]} + Var[w_i^{(l)}](\mathbb{E}[f\left(z_i^{(l-1)}\right)])^2 \\
&\approx d_{l-1}Var[w_i^{(l)}]Var[f\left(z_i^{(l-1)}\right)]
\end{aligned}
$$

参考[**SELU**](https://0809zheng.github.io/2021/09/02/selu.html)提出的自标准化方法，假设输入数据$x$方差为$1$，经过激活函数$f(x)$后仍然希望方差为$1$，则可以为激活函数$f(\cdot)$引入一个增益值(**gain value**) $\lambda$，使得输出满足二阶统计量(方差$=1$)对应的积分方程：

$$ \int_{-∞}^{+∞} \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}} \cdot (\lambda f(x))^2dx = 1  $$

使用[sympy](https://0809zheng.github.io/2021/09/01/solve.html)库可以快速求解上述方程：

```python
import sympy
from sympy import Symbol, nsolve, integrate

x = Symbol('x')
l = Symbol('l')
integal = integrate(sympy.exp(-x**2/2)*(f(x))**2, (x,-sympy.oo,sympy.oo))
fn = l**2/sympy.sqrt(2*sympy.pi)*integal - 0
ans = nsolve(fn, l, 1)
```

为激活函数引入增益值$\lambda$等价于为权重引入增益值$1/\lambda$：

$$
\begin{aligned}
Var[z^{(l)}] 
&\approx d_{l-1}Var[\frac{1}{\lambda}w_i^{(l)}]Var[\lambda f\left(z_i^{(l-1)}\right)]
\end{aligned}
$$

因此对于一般的激活函数$f(\cdot)$，网络权重初始化时会额外引入一个增益值，对于常见的激活函数，可以查询该增益值：

```python
gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
```

## （3）Xavier初始化的例子

**Xavier**初始化通常用于无激活函数的场合，此时把参数初始化为均值为$0$、方差为$\frac{2}{d_{l-1}+d_{l}}$的常见分布形式，其中$d_{l-1}$和$d_l$分别为前一层和当前层神经元的数量。

**Xavier**初始化也适用于激活函数为**Sigmoid**函数或**Tanh**函数的场合，这是因为神经元的参数和输入的绝对值通常比较小，处于激活函数的线性区间。例如**Sigmoid**函数在线性区的斜率约为$\frac{1}{4}$，因此其参数初始化的方差应调整为：

$$ Var[w_i^{(l)}] =  16 \times \frac{2}{d_{l-1}+d_{l}} $$

### ⚪ 高斯分布的Xavier初始化

若采用高斯分布$N(0,σ^2)$对参数进行**Xavier**初始化，则有：

$$
\sigma = gain \cdot \sqrt{\frac{2}{d_{l-1}+d_{l}}}
$$

```python
torch.nn.init.xavier_normal_(tensor, gain=1.0)
```

### ⚪ 均匀分布的Xavier初始化

若采用均匀分布$U(-a,a)$对参数进行**Xavier**初始化，则有：

$$
\sigma = \sqrt{\frac{(a-(-a))^2}{12}} = gain \cdot \sqrt{\frac{2}{d_{l-1}+d_{l}}} \\
\downarrow \\
a = gain \cdot \sqrt{\frac{6}{d_{l-1}+d_{l}}}
$$

```python
torch.nn.init.xavier_uniform_(tensor, gain=1.0)
```

# 5. Kaiming初始化 Kaiming Initialization

- paper：[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

**Xavier**初始化仅适用于线性激活函数或在零值附近具有线性区域的激活函数。对于更常用的**ReLU**族等激活函数，会使输入数据分布的均值不再为$0$，因此不再满足**Xavier**初始化的条件。此时可使用**Kaiming初始化**。

**Kaiming**初始化是由**Kaiming He**提出的，因此也称为**He**初始化。假设第$l$层的一个神经元$z^{(l)}$，接收前一层的$d_{l-1}$个神经元的输出$z^{(l-1)}$，

$$ z^{(l)} = \sum_{i=1}^{d_{l-1}} {w_i^{(l)}a_i^{(l-1)} } = \sum_{i=1}^{d_{l-1}} {w_i^{(l)}f\left(z_i^{(l-1)}\right) }$$

此处假设偏置$b$初始化为$0$，$f(\cdot)$为任意激活函数。$z^{(l)}$的方差为：

$$
\begin{aligned}
Var[z^{(l)}] &= Var[\sum_{i=1}^{d_{l-1}} {w_i^{(l)}f\left(z_i^{(l-1)}\right)}] = \sum_{i=1}^{d_{l-1}} Var[{w_i^{(l)}f\left(z_i^{(l-1)}\right)}]\\
&= \sum_{i=1}^{d_{l-1}} {Var[w_i^{(l)}]Var[f\left(z_i^{(l-1)}\right)]} + Var[w_i^{(l)}](\mathbb{E}[f\left(z_i^{(l-1)}\right)])^2 \\
&= \sum_{i=1}^{d_{l-1}} Var[w_i^{(l)}]\left(Var[f\left(z_i^{(l-1)}\right)] + (\mathbb{E}[f\left(z_i^{(l-1)}\right)])^2\right) \\
&= \sum_{i=1}^{d_{l-1}} Var[w_i^{(l)}]\mathbb{E}[f^2\left(z_i^{(l-1)}\right)] \\
\end{aligned}
$$

## （1）ReLU激活函数

以**ReLU**激活函数$f(x)=\max(0,x)$为例，则有：

$$
\begin{aligned}
\mathbb{E}[f^2\left(x\right)] &= \int_{-\infty}^{+\infty} (\max(0,x))^2p(x)dx \\
&= \int_{0}^{+\infty} x^2p(x)dx = \frac{1}{2}\int_{-\infty}^{+\infty} x^2p(x)dx \\
&= \frac{1}{2}\mathbb{E}[x^2] = \frac{1}{2}\left(Var[x]+(\mathbb{E}[x])^2\right) \\
\end{aligned}
$$

代回$z^{(l)}$的方差表达式：

$$
\begin{aligned}
Var[z^{(l)}] &=  \sum_{i=1}^{d_{l-1}} Var[w_i^{(l)}]\mathbb{E}[f^2\left(z_i^{(l-1)}\right)] \\
&=  \sum_{i=1}^{d_{l-1}}  \frac{1}{2}Var[w_i^{(l)}]\left(Var[z_i^{(l-1)}]+(\mathbb{E}[z_i^{(l-1)}])^2\right) \\
&=  \sum_{i=1}^{d_{l-1}}  \frac{1}{2}Var[w_i^{(l)}]Var[z_i^{(l-1)}] \\
&=   \frac{d_{l-1}}{2}Var[w_i^{(l)}]Var[z_i^{(l-1)}] \\
\end{aligned}
$$

即输入信号的方差在经过该神经元后被缩放了$\frac{d_{l-1}}{2}Var[w_i^{(l)}]$倍。为了使得在经过多层网络后，信号不被过分放大或过分减弱，尽可能保持每个神经元的输入和输出的方差一致，则有：

$$ Var[w_i^{(l)}] = \frac{2}{d_{l-1}} $$

**Kaiming**初始化默认使用输入神经元的个数$d_{l-1}$。若采用高斯分布$N(0,σ^2)$对参数进行**Kaiming**初始化，则有：

$$
\sigma = gain \cdot \sqrt{\frac{2}{d_{l-1}}}
$$

```python
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
# a：leaky ReLU的负斜率
# mode：fan_in考虑前向传播，fan_out考虑反向传播
# nonlinearity：relu或leaky_relu
```

若采用均匀分布$U(-a,a)$对参数进行**Kaiming**初始化，则有：

$$
\sigma = \sqrt{\frac{(a-(-a))^2}{12}} = gain \cdot \sqrt{\frac{2}{d_{l-1}}} \\
\downarrow \\
a = gain \cdot \sqrt{\frac{6}{d_{l-1}}}
$$

```python
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

## （2）Leaky ReLU激活函数

若激活函数选用**Leaky ReLU** $f(x)=\max(\alpha x,x)$为例，则有：

$$
\begin{aligned}
\mathbb{E}[f^2\left(x\right)] &= \int_{-\infty}^{+\infty} (\max(\alpha x,x))^2p(x)dx \\
&= \int_{-\infty}^{0} \alpha^2x^2p(x)dx+ \int_{0}^{+\infty} x^2p(x)dx \\
&= \frac{\alpha^2+1}{2}\int_{-\infty}^{+\infty} x^2p(x)dx \\
&= \frac{\alpha^2+1}{2}\mathbb{E}[x^2] = \frac{\alpha^2+1}{2}\left(Var[x]+(\mathbb{E}[x])^2\right) \\
\end{aligned}
$$

代回$z^{(l)}$的方差表达式：

$$
\begin{aligned}
Var[z^{(l)}] &=  \sum_{i=1}^{d_{l-1}} Var[w_i^{(l)}]\mathbb{E}[f^2\left(z_i^{(l-1)}\right)] \\
&=  \sum_{i=1}^{d_{l-1}}  \frac{\alpha^2+1}{2}Var[w_i^{(l)}]\left(Var[z_i^{(l-1)}]+(\mathbb{E}[z_i^{(l-1)}])^2\right) \\
&=  \sum_{i=1}^{d_{l-1}}  \frac{\alpha^2+1}{2}Var[w_i^{(l)}]Var[z_i^{(l-1)}] \\
&=   \frac{(\alpha^2+1)d_{l-1}}{2}Var[w_i^{(l)}]Var[z_i^{(l-1)}] \\
\end{aligned}
$$

即输入信号的方差在经过该神经元后被缩放了$\frac{(\alpha^2+1)d_{l-1}}{2}Var[w_i^{(l)}]$倍。为了使得在经过多层网络后，信号不被过分放大或过分减弱，尽可能保持每个神经元的输入和输出的方差一致，则有：

$$ Var[w_i^{(l)}] = \frac{2}{(\alpha^2+1)d_{l-1}} $$

# 6. 正交初始化 Orthogonal Initialization

- paper：[Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)

**正交初始化（Orthogonal Initialization）**是指将参数矩阵$W^{(l)}$初始化为正交矩阵，即：

$$ W^{(l)}{W^{(l)}}^T = I $$

实现过程：
1. 用标准高斯分布$N(0,1)$初始化一个矩阵;
2. 将这个矩阵用奇异值分解得到两个正交矩阵，并使用其中之一作为权重矩阵。

```python
torch.nn.init.orthogonal_(tensor, gain=1)
```

正交初始化使误差项在反向传播中具有**范数保持性(Norm-Preserving)**。对于误差项$δ^{(l-1)} = {W^{(l)}}^T δ^{(l)}$，满足：

$$ \mid\mid δ^{(l-1)} \mid\mid^2 = \mid\mid {W^{(l)}}^T δ^{(l)} \mid\mid^2 = \mid\mid δ^{(l)} \mid\mid^2 $$

当在非线性神经网络中应用正交初始化时，通常需要将正交矩阵乘以一个缩放系数$ρ$。比如当激活函数为**ReLU**时，激活函数在$0$附近的平均梯度可以近似为$0.5$。为了保持范数不变，缩放系数$ρ$可以设置为$\sqrt{2}$。

正交初始化通常用在循环神经网络中循环边上的权重矩阵上。

# 7. 稀疏初始化 Sparse Initialization

稀疏初始化是指将权重矩阵中的大部分元素设置为零，从而实现稀疏性。稀疏初始化的公式为：

$$
W_{ij} \sim \mathcal{N}(0, \sigma^2) * \textbf{B}
$$

其中$W_{ij}$是连接第$i$个输入神经元和第$j$个输出神经元的权重，$$\mathcal{N}(0, \sigma^2)$$表示均值为$0$, 方差为$\sigma^2$的高斯分布，$$\textbf{B}$$是大小为$m \times n$的二元矩阵，其中$m$是输入神经元的数量，$n$是输出神经元的数量。$$\textbf{B}$$中的每个元素都是$0$或$1$，其中$1$的数量为$\rho mn$，$\rho$是一个控制稀疏度的参数。

```python
torch.nn.init.sparse_(tensor, sparsity, std=0.01)
```

在实践中，通常将$\rho$设置为$0.1$或$0.01$，从而将权重矩阵中的大部分元素设置为零。这种稀疏性可以减少神经网络中的冗余性和过拟合，提高网络的泛化能力和性能。

