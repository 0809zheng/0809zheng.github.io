---
layout: post
title: 'Squareplus: A Softplus-Like Algebraic Rectifier'
date: 2021-11-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61ebf3f12ab3f51d911accbc.jpg'
tags: 论文阅读
---

> Squareplus：一种类似softplus的代数整流器.

- paper：[Squareplus: A Softplus-Like Algebraic Rectifier](https://arxiv.org/abs/2112.11687)

**ReLU**激活函数在$x=0$处不可导，它的一种连续的近似替代是**Softplus**，定义如下：

$$ \text{softplus}(x) =\log(1+e^x) $$

然而这类近似替代引入了指数运算，增加了激活函数的计算复杂度。本文提出了一种与**Softplus**相似的激活函数，只使用代数运算(加法、乘法和平方根)实现，称之为**Squareplus**，其表达式如下：

$$ \text{squareplus}(x,b) =\frac{1}{2}(x+\sqrt{x^2+b}) $$

![](https://pic.imgdb.cn/item/61ec00502ab3f51d91273cdf.jpg)

超参数$b$定义了函数在$x=0$处的弯曲大小。**Squareplus**的一阶导数和二阶导数如下：

$$ \frac{d}{dx}\text{squareplus}(x,b) = \frac{1}{2}(1+\frac{x}{\sqrt{x^2+b}}) $$

$$ \frac{d^2}{dx^2}\text{squareplus}(x,b) = \frac{1}{2}(\frac{b}{(x^2+b)^{3/2}}) $$

**Softplus**的一阶导数是**S**型曲线，而**Squareplus**的一阶导数是代数**S**型曲线；**Softplus**的二阶导数是**logistic**分布的概率密度函数，而**Squareplus**的二阶导数是**student-t**分布的概率密度函数。

![](https://pic.imgdb.cn/item/61ec00662ab3f51d9127508f.jpg)

![](https://pic.imgdb.cn/item/61ec007a2ab3f51d912763b4.jpg)

当$b=0$时，**Squareplus**退化为**ReLU**：

$$ \text{squareplus}(x,0) =\frac{x+|x|}{2} = \text{relu}(x) $$

下面计算使得**Squareplus**和**Softplus**。将问题建模成**min-max**形式，即希望函数$\text{softplus}(x)$和$\text{squareplus}(x,b)$在全局的最大差异尽可能小，表示为下述问题：

$$ \mathop{\min}_{b} \mathop{\max}_{x} \quad |\frac{1}{2}(x+\sqrt{x^2+b})-\log(1+e^x)| $$

上式可以通过[非线性规划](https://0809zheng.github.io/2021/08/23/minimize.html)求解：

```python
import numpy as np
from scipy.optimize import minimize

def f(x, b):
    a = np.sqrt(2 / np.pi)
    return np.abs(0.5*(x+np.sqrt(x**2+b))-np.log(1+np.exp(x)))

def g(b):
    return np.max([f(x, b) for x in np.arange(-2, 4, 0.001)])

options = {'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 100000}
result = minimize(g, 0, method='Powell', options=options)
print(result.x) # [1.52382104]
```

注意到**Squareplus**和**Softplus**都是**ReLU**的上界，如果希望**Squareplus**是**Softplus**的上界，则应有：

$$ \frac{1}{2}(x+\sqrt{x^2+b}) \geq \log(1+e^x) \\ b \geq 4\log(1+e^x)[\log(1+e^x)-x] $$

可以证明，右式在$x=0$处取极大值，因此应有$b\geq 4 \log^2 2$。尽管**Squareplus**和**Softplus**都是**ReLU**的近似，当输入值较大时，**Softplus**的数值不稳定，将会偏离**ReLU**：

![](https://pic.imgdb.cn/item/61ec078a2ab3f51d912e775e.jpg)

在**CPU**上，**Squareplus**的计算速度比**Softplus**快$6$倍，与**ReLU**相当。在**GPU**上，**Squareplus**的加速并不明显，这是因为带宽的限制。这表明**Squareplus**在计算资源有限的情况是**Softplus**的理想替代品。

![](https://pic.imgdb.cn/item/61ec022d2ab3f51d91292543.jpg)