---
layout: post
title: '利普希茨连续条件(Lipschitz Continuity Condition)'
date: 2022-10-11
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/63468bf216f2c2beb1846342.jpg'
tags: 数学
---

> Lipschitz Continuity Condition.

1. Lipschitz连续条件的定义
2. 神经网络中的Lipschitz约束
3. 实现Lipschitz约束的方法：权重裁剪、梯度惩罚、谱归一化、梯度归一化

# 1. 利普希茨连续条件的定义

**利普希茨连续条件(Lipschitz Continuity Condition)**是一个比**一致连续**更强的函数光滑性条件。该条件限制了函数改变的速度，即符合**Lipschitz**连续条件的函数的斜率必小于一个依函数而定的**Lipschitz**常数。

一般地，一个实值函数$f(x)$是$K$阶**Lipschitz**连续的(记作$\|f\|_L = K$)，是指存在一个实数$K\geq 0$，使得对$$\forall x_1,x_2 \in \Bbb{R}$$，有：

$$ || f(x_1)-f(x_2) || \leq K || x_1-x_2 || $$

通常一个连续可微函数满足**Lipschitz**连续，这是因为其微分(用$\frac{\|f(x_1)-f(x_2)\|}{\|x_1-x_2\|}$近似)总是有界的。但是一个**Lipschitz**连续函数不一定是处处可微的，比如$f(x) = \|x\|$。

**Lipschitz**连续性保证了函数对于**输入扰动的稳定性**，即函数的输出变化相对输入变化是缓慢的。

# 2. 神经网络中的利普希茨约束

若神经网络具有**Lipschitz**连续性，意味着该网络对输入扰动不敏感，具有更好的泛化性。下面讨论如何对神经网络$f(x)$施加**Lipschitz**约束。

假设神经网络具有参数$W$，则**Lipschitz**常数$K$通常是由参数$W$决定的，此时**Lipschitz**约束为：

$$ || f_W(x_1)-f_W(x_2) || \leq K(W) || x_1-x_2 || $$

不失一般性地考虑全连接层$f_W(x)=\sigma(Wx)$，其中$\sigma$是激活函数，对应**Lipschitz**约束：

$$ || \sigma(Wx_1)-\sigma(Wx_2) || \leq K(W) || x_1-x_2 || $$

对$\sigma(Wx)$进行[Taylor展开](https://0809zheng.github.io/2021/08/20/taylor.html)并取一阶近似可得：

$$ ||  \frac{\partial \sigma}{\partial Wx} W(x_1-x_2) || \leq K(W) || x_1-x_2 || $$

其中$\frac{\partial \sigma}{\partial Wx}$表示激活函数的导数。上式表明**Lipschitz**约束分别对激活函数的导数和网络权重进行了约束。

通常激活函数的导数是有界的，比如**ReLU**函数的导数范围是$[0,1]$；因此这一项可以被忽略。则全连接层的**Lipschitz**约束进一步写作：

$$ ||  W(x_1-x_2) || \leq K(W) || x_1-x_2 || $$

上式表示若对权重参数$W$进行约束后，全连接层将会满足**Lipschitz**约束。在实践中全连接网络是由全连接层组合而成，而卷积网络、循环网络等也可以表示为特殊的全连接网络，因此上述分析具有一般性。

其中$K(W)$的最小值被称为**Lipschitz**常数，通常希望其取值尽可能小，在实践中常约束$K(W)=1$。

# 3. 实现Lipschitz约束的方法

为判别器引入**Lipschitz**约束的方法主要有两种。第一种是施加**硬约束**，即通过约束参数使得网络每一层的**Lipschitz**常数都是有界的，则总**Lipschitz**常数也是有界的，这类方法包括权重裁剪、谱归一化。

这些方法强制网络的每一层都满足**Lipschiitz**约束，从而把网络限制为所有满足**Lipschiitz**约束的函数中的一小簇函数。事实上考虑到如果网络有些层不满足**Lipschiitz**约束，另一些层满足更强的**Lipschiitz**约束，则网络整体仍然满足**Lipschiitz**约束。这类方法无法顾及这种情况。

第二种是施加**软约束**，即选择**Lipschitz**约束的一个充分条件(通常是网络对输入的梯度)，并在目标函数中添加相关的惩罚项。这类方法包括梯度惩罚、梯度归一化。

## （1）权重裁剪 weight clipping

- paper: [<font color=Blue>Wasserstein GAN</font>](https://0809zheng.github.io/2022/02/04/wgan.html)

既然**Lipschitz**约束对网络权重$W$进行了约束，在实践中可以通过**weight clipping**实现该约束：在每次梯度更新后，把网络权重$W$的取值限制在$[-c,c]$之间。

$$ \begin{aligned}  W &\leftarrow\text{clip}(W,-c,c)  \end{aligned} $$

```python
for p in model.parameters():
    p.data.clamp_(-clip_value, clip_value)
```

然而该做法也有一些问题。若$c$值取得太大，则模型训练容易不稳定，收敛速度慢；若$c$值取得太小，则容易造成梯度消失。

## （2）梯度惩罚 gradient penalty

- paper: [<font color=Blue>Improved Training of Wasserstein GANs</font>](https://0809zheng.github.io/2022/02/06/wgangp.html)

注意到**Lipschitz**约束是一种差分约束：

$$ \frac{|| f_W(x_1)-f_W(x_2) ||}{|| x_1-x_2 ||} \leq K(W)  $$

上式的一个充分条件是：

$$ ||\frac{\partial f_W(x)}{\partial x}|| \leq K(W) $$

在实践中可以向目标函数中引入**梯度惩罚**实现该约束，即约束$f_W(x)$在任意位置的梯度的模小于等于$1$：

$$ \begin{aligned} W \leftarrow \mathop{\arg \min}_{W}& \mathcal{L}(x;W)+ \lambda \max(||\frac{\partial f_W(x)}{\partial x}||,1) \end{aligned} $$

或：

$$ \begin{aligned}  W \leftarrow \mathop{\arg \min}_{W}& \mathcal{L}(x;W)+ \lambda (||\frac{\partial f_W(x)}{\partial x}||-1)^2 \end{aligned} $$

```python
def compute_gradient_penalty(model, data):
    """Calculates the gradient penalty loss"""
    preds = model(data)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=preds,
        inputs=data,
        grad_outputs=torch.ones_like(preds).requires_grad_(False),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

## （3）谱归一化

- paper: [<font color=Blue>Spectral Normalization for Generative Adversarial Networks</font>](https://0809zheng.github.io/2022/02/08/sngan.html)

定义参数矩阵的**谱范数(spectral norm)**：

$$ ||W||_2 = \mathop{\max}_{x \neq 0} \frac{||Wx||}{||x||} $$

谱范数是一种由向量范数诱导出来的矩阵范数，作用相当于向量的模长。则全连接层的**Lipschitz**约束可以转化为一个矩阵范数问题：

$$ ||  W(x_1-x_2) || \leq ||W||_2 \cdot || x_1-x_2 || $$

谱范数$\|\|W\|\|_2$等于$W^TW$的最大特征值(主特征值)的平方根；若$W$为方阵，则$\|\|W\|\|_2$等于$W$的最大特征值的绝对值。

谱范数可以通过**幂迭代(power iteration)**方法求解（迭代的收敛性证明可参考原论文超链接）：

$$ v \leftarrow \frac{W^Tu}{||W^Tu||},u \leftarrow \frac{Wv}{||Wv||}, ||W||_2 ≈ u^TWv $$

**谱归一化(Spectral Normalization)**是指使用谱范数对网络参数进行归一化：

$$ W \leftarrow \frac{W}{||W||_2^2} $$

根据前述分析，如果激活函数的导数是有界的，应用谱归一化约束参数后，可以精确地使网络满足**Lipschiitz**连续性。

```python
model = model()

def add_sn(m):
        for name, layer in m.named_children():
             m.add_module(name, add_sn(layer))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
             return nn.utils.spectral_norm(m)
        else:
             return m

model = add_sn(model)
```

## （4）梯度归一化

- paper：[<font color=Blue>Gradient Normalization for Generative Adversarial Networks</font>](https://0809zheng.github.io/2022/02/10/gngan.html)、[<font color=Blue>GraN-GAN: Piecewise Gradient Normalization for Generative Adversarial Networks</font>](https://0809zheng.github.io/2022/02/11/grangan.html)

**Lipschitz-1**约束的一个充分条件是：

$$ ||\nabla_x f(x)|| \leq 1 $$

如果将函数$f$变换为$\hat{f}$，使得其自动满足$$\|\nabla_x \hat{f}(x)\| \leq 1$$，则实现了**Lipschitz**约束的引入。不妨取：

$$ \hat{f}(x) = \frac{f(x)}{||\nabla_x f(x)||} $$

注意到网络通常用**ReLU**或**LeakyReLU**作为激活函数，此时$f(x)$实际上是一个“分段线性函数”，除边界之外$f(x)$在局部的连续区域内是一个线性函数，因此$\nabla_x f(x)$是一个常向量。此时有：

$$ ||\nabla_x \hat{f}(x)|| = ||\nabla_x \frac{f(x)}{||\nabla_x f(x)||}|| = ||\frac{\nabla_x f(x)}{||\nabla_x f(x)||}|| = 1 $$

上式可能会出现分母为零的情况，[<font color=Blue>GN-GAN</font>](https://0809zheng.github.io/2022/02/10/gngan.html)将$\|f(x)\|$引入分母，同时也保证了函数的有界性：

$$ \hat{f}(x) = \frac{f(x)}{||\nabla_x f(x)||+|f(x)|} \in [-1,1] $$

```python
def grad_normalize(f, x):
    """Calculates the gradient normalization"""
    x.requires_grad_(True)
    out = f(x)
    grad_out=torch.ones_like(out).requires_grad_(False),
    # Get gradient w.r.t. x
    gradients = autograd.grad(
        outputs=out, inputs=x, grad_outputs=grad_out,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_norm = gradients.view(gradients.size(0), -1).pow(2).sum(1) ** (1/2)
    return out / (grad_norm + torch.abs(out))
```

而[<font color=Blue>GraN-GAN</font>](https://0809zheng.github.io/2022/02/11/grangan.html)设计了如下变换：

$$ \hat{f}(x) = \frac{f(x) \cdot ||\nabla_x f(x)||}{||\nabla_x f(x)||^2+\epsilon}  $$

```python
def grad_normalize(f, x):
    """Calculates the gradient normalization"""
    x.requires_grad_(True)
    out = f(x)
    grad_out=torch.ones_like(out).requires_grad_(False),
    # Get gradient w.r.t. x
    gradients = autograd.grad(
        outputs=out, inputs=x, grad_outputs=grad_out,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_norm = gradients.view(gradients.size(0), -1).pow(2).sum(1) ** (1/2)
    return (out * grad_norm) / (grad_norm**2 + epsilon)
```