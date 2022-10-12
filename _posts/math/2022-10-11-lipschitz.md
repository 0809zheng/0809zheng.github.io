---
layout: post
title: '利普希茨连续条件(Lipschitz Continuity Condition)'
date: 2022-10-11
author: 郑之杰
cover: ''
tags: 数学
---

> Lipschitz Continuity Condition.

1. Lipschitz连续条件的定义
2. 神经网络中的Lipschitz约束
3. 实现Lipschitz约束的方法：参数裁剪、梯度惩罚、谱归一化、

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

$\frac{\partial \sigma}{\partial Wx}$表示激活函数的导数。通常激活函数的导数是有界的，比如**ReLU**函数的导数范围是$[0,1]$；因此这一项可以被忽略。则全连接层的**Lipschitz**约束为：

$$ ||  W(x_1-x_2) || \leq K(W) || x_1-x_2 || $$

上式对全连接层的参数$W$进行了约束。在实践中全连接网络是由全连接层组合而成，而卷积网络、循环网络等也可以表示为特殊的全连接网络，因此上述分析具有一般性。

# 3. 实现Lipschitz约束的方法

## （1）权重裁剪 weight clipping

- paper: [Wasserstein GAN](https://0809zheng.github.io/2022/02/04/wgan.html)

既然**Lipschitz**约束对网络参数$W$进行了约束，在实践中可以通过**weight clipping**实现该约束：在每次梯度更新后，把网络参数$W$的取值限制在$[-c,c]$之间。

$$ \begin{aligned}  W &\leftarrow\text{clip}(W,-c,c)  \end{aligned} $$

```python
for p in model.parameters():
    p.data.clamp_(-clip_value, clip_value)
```

然而该做法也有一些问题。若$c$值取得太大，则模型训练容易不稳定，收敛速度慢；若$c$值取得太小，则容易造成梯度消失。

## （2）梯度惩罚 gradient penalty

- paper: [Improved Training of Wasserstein GANs](https://0809zheng.github.io/2022/02/06/wgangp.html)

注意到**Lipschitz**约束是一种差分约束：

$$ \frac{|| f_W(x_1)-f_W(x_2) ||}{|| x_1-x_2 ||} \leq K(W)  $$

将差分形式用梯度形式近似：

$$ \frac{|| f_W(x_1)-f_W(x_2) ||}{|| x_1-x_2 ||} ≈ ||\frac{\partial f_W(x)}{\partial x}|| $$

在实践中可以向目标函数中引入**梯度惩罚**实现该约束，即约束$f_W(x)$在任意位置的梯度的模小于等于$K$：

$$ \begin{aligned} W \leftarrow \mathop{\arg \min}_{W}& \mathcal{L}(x;W)+ \lambda \max(||\frac{\partial f_W(x)}{\partial x}||,K) \end{aligned} $$

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

- paper: [Spectral Normalization for Generative Adversarial Networks](https://0809zheng.github.io/2022/02/08/sngan.html)

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