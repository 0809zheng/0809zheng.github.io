---
layout: post
title: '深度学习中的激活函数(Activation Function)'
date: 2020-03-01
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-001-neuron.png'
tags: 深度学习
---

> Activation Functions in Deep Learning.

**激活函数(activation function)**是神经网络中重要的非线性来源。它的形式看起来微不足道（通常只是一个作用在标量上的一元函数），但它决定了网络能否被优化、梯度能否传播、表示能力有多强，甚至决定了模型能否部署到低精度硬件上。

本文首先讨论激活函数的意义与设计准则，然后按设计思路把主流激活函数组织成不同族系统梳理，并在最后给出一份速查表和实践选型建议。
1. 激活函数的意义
2. 激活函数的设计准则
3. 常见的激活函数
   - 3.1 **S**型激活函数
   - 3.2 **ReLU**族激活函数
   - 3.3 自动搜索的激活函数
   - 3.4 周期性激活函数
   - 3.5 通用近似激活函数
   - 3.6 上下文相关的激活函数
   - 3.7 门控激活函数
4. 速查表与选型建议

**符号约定**：全文用$\sigma(\cdot)$专门表示**Sigmoid**函数$\sigma(x)=1/(1+e^{-x})$；用$s$表示光滑化核的宽度；用$\mathbb{E}[\cdot]$与$\text{Var}[\cdot]$表示期望与方差。

较为全面的综述可参考[Activation Functions: Comparison of trends in Practice and Research for Deep Learning](https://arxiv.org/abs/1811.03378)，以及一份收录了$400$余个激活函数的详尽调研[Three Decades of Activations: A Comprehensive Survey of 400 Activation Functions for Neural Networks](https://arxiv.org/abs/2402.09092)。

# 1. 激活函数的意义

## (1) 从生物学的角度理解激活函数

早期激活函数的设计受到生物神经网络中**神经元**的启发，即对神经元进行简单的建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-001-neuron.png)

大脑中的**神经元(neuron)**通过**树突(dendrites)**接收其他神经元的输入信号，在胞体中进行信号的处理，通过**轴突(axon)**分发信号。当神经元中的信号累积达到一定阈值时产生电脉冲将信号输出，这个阈值称为**点火率(firing rate)**。

- 其他神经元的输入信号建模为$x_i$；
- 树突的信号接收过程建模为$w_ix_i$；
- 胞体的信号处理过程建模为$\sum_i w_ix_i$；
- 点火率建模为$-b$；
- 信号累积与阈值的比较建模为$\sum_i w_ix_i+b$；
- 产生电脉冲建模为$f(\cdot)=\text{Step}(\cdot)$；
- 轴突的输出信号建模为$f\left(\sum_i w_ix_i+b\right)$。

因此激活函数最初的作用是模拟“信号累积达到阈值并产生电脉冲”这一过程。

值得一提的是，这种对神经元的建模是非常粗糙(**coarse**)的：真实神经元有很多不同的种类；突触是一个复杂的非线性动态系统；树突进行的是复杂的非线性运算；轴突的输出时刻(而非仅仅输出强度)也携带信息。因此近些年来神经网络中神经元的**生物可解释性(biological plausibility)**被逐渐弱化，激活函数的设计更多地由优化性质、表示能力和计算效率驱动。

## (2) 从非线性的角度理解激活函数

在神经网络中，使用激活函数能够为网络引入非线性，增强网络的表示能力。当不使用激活函数时（或激活函数为**恒等函数 identity function**），多层神经网络会退化为单层网络：

$$
\begin{aligned}
W_2\left(W_1x+b_1\right)+b_2 &= W_2W_1x+W_2b_1+b_2 \\
&= \left(W_2W_1\right)x+\left(W_2b_1+b_2\right) \\
&= W'x+b'
\end{aligned}
$$

即无论堆叠多少层，整个网络仍然只能表达输入的仿射变换。通过引入非线性的激活函数，能够为神经网络带来非线性表示能力。

## (3) 从通用近似的角度理解激活函数

**通用近似定理(universal approximation theorem)**指出：只要激活函数是非常数、有界、单调递增的连续函数（后续工作把条件放宽到非多项式函数），具有一个隐藏层的前馈网络就可以在紧集上以任意精度逼近任意连续函数。

这个定理说明了两件事：其一，激活函数的“非线性”本身就是网络表示能力的来源；其二，几乎任何非线性函数都可以作为激活函数，因此激活函数的设计空间极大。这也解释了为什么激活函数的研究会分化出如此多的路线：真正的约束不来自表示能力，而来自**可优化性**与**计算效率**。


# 2. 激活函数的设计准则

既然理论上任何非线性函数都可以作为激活函数，那么在实践中选择或设计激活函数时，需要考虑以下性质。这七条准则也构成了后文各族激活函数的设计动机。

### ⚪ 准则1：连续可导

激活函数需要参与反向传播过程，因此需要计算激活函数的导数，这就要求激活函数**连续可导**。

例如**ReLU**族激活函数在$x=0$处不可导。处理方式有两种：既可以人工指定该点处的梯度（工程实现中的常见做法），也可以选用形状接近的连续函数进行近似，如用**Softplus**替代**ReLU**、用**CELU**替代**ELU**。

事实上，“为非光滑激活函数寻找光滑近似”后来发展成了一条独立的研究路线（见3.5节的**ACON**、**SMU**、**SAU**）。更多关于不可导函数光滑化的内容可参考[<font color=Blue>光滑化方法</font>](https://0809zheng.github.io/2021/11/16/mollifier.html)。

### ⚪ 准则2：计算量小

激活函数会作用在网络的每一个激活值上，调用次数极多，因此应具有尽可能小的**计算量**。通常线性运算（如**ReLU**族）比指数运算（如**S**型曲线）具有更低的计算量；而在指数运算与平方根运算之间，平方根通常更快（**ISRLU**的论文报告在**Intel Xeon Platinum 8160**上逆平方根比指数运算快约$2.2$倍）。

降低计算量的常见手段包括：
- **分段线性近似**：对指数函数进行[<font color=Blue>Taylor展开</font>](https://0809zheng.github.io/2021/08/20/taylor.html#3-%E6%B3%B0%E5%8B%92%E5%85%AC%E5%BC%8F%E7%9A%84%E5%BA%94%E7%94%A8hard-sigmoid%E4%B8%8Ehard-tanh)后截断，得到**HardSigmoid**、**HardTanh**、**HardSwish**等；
- **代数近似**：用加法、乘法和平方根替代指数运算，得到**ISRU**、**ISRLU**、**Squareplus**。

在低精度部署场景下，还需要考虑激活函数的**值域**是否适合量化（比如**ReLU6**）。

### ⚪ 准则3：没有饱和区

**饱和**是指导数很接近$0$。若激活函数存在饱和区，则会使反向传播的梯度接近$0$，从而导致**梯度消失(gradient vanishing)**现象。

早期的激活函数通常使用**S**型函数，如**Sigmoid**、**Tanh**；这类函数会把输出挤压到一个有界区域内，两端不可避免地产生饱和区，因此也被称为**squashing function**。

**ReLU**等无上界、有下界的激活函数，在正半轴没有饱和区，显著减缓了梯度消失现象；但在负半轴会置零（产生**dead ReLU**现象，即梯度恒为$0$阻断了反向传播）或趋于饱和。后续的**LeakyReLU**、**PReLU**、**ELU**等改进都是围绕负半轴展开的。

### ⚪ 准则4：没有偏置偏移

若激活函数的输出不是**zero-centered**的，会使得后一层神经元的输入产生**偏置偏移(bias shift)**，从而减慢梯度下降的收敛速度。

对于某一层神经元的计算，假设具有两个参数$w_1,w_2$，则$y=f(w_1x_1+w_2x_2+b)$，反向传播时两个参数的梯度为：

$$
\begin{aligned}
\nabla_{w_1}L &= \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_1} = \nabla_yL \cdot f' \cdot x_1 \\
\nabla_{w_2}L &= \frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial w_2} = \nabla_yL \cdot f' \cdot x_2
\end{aligned}
$$

若上一层的激活函数使得该层神经元的输入值恒大于$0$，则$\text{sign}(\nabla_{w_1}L)=\text{sign}(\nabla_{w_2}L)$，梯度只能沿着$w_1,w_2$同时增大或同时减小的方向更新，参数更新路径呈**Z**字形，从而减慢收敛速度。

当激活函数的值域同时包含正值和负值时，能够有效缓解偏置偏移现象。这是**Tanh**优于**Sigmoid**、**ELU**族优于**ReLU**的重要原因之一。

### ⚪ 准则5：具有生物可解释性

生物神经元通常具有**单侧抑制**（即输入大于阈值才会被激活）、**宽兴奋边界**（即输出范围较宽，如$[0,+\infty)$）、**稀疏激活**（即同时被激活的神经元较少）等特性。**ReLU**恰好同时满足这三条，这也是它在提出时被广泛接受的原因之一。

不过后续研究（如**RReLU**）通过实验表明，*稀疏性并不是激活函数性能的决定性因素*。因此**ReLU**之后的激活函数在设计时逐渐淡化了生物可解释性。

### ⚪ 准则6：能够提取上下文信息

通常的激活函数是标量函数，如**ReLU**对神经元输入的每一个标量值分别独立计算。如果能够将激活函数拓展为多输入函数，则能够捕捉输入的上下文信息，增强神经元的表达能力。

某个特征位置的上下文信息既可以从**所有输入特征**中获取（如**maxout**、**Dynamic ReLU**），也可以在该特征的**一个邻域**上获取（如**Dynamic Shift-Max**、**FReLU**）。详见3.6节。

### ⚪ 准则7：具有通用近似性

直观上，神经网络每一层的每个神经元都应具有不同的激活曲线。可以设计一些由参数控制的通用近似激活函数，使得每个神经元学习不同的激活曲线，这些激活参数与网络权重一起参与反向传播。

设计通用近似激活函数主要有两种思路，详见3.5节：
- 使用通用的函数逼近方法，如**分段线性近似**（**APL**、**PWLU**）、**Padé近似**（**PAU**、**OPAU**）；
- 寻找现有激活函数的**光滑逼近**，如手工设计近似（**ACON**、**SMU**）、使用**Dirac**函数构造近似（**SAU**）。


# 3. 常见的激活函数

本节把主流激活函数分成七个族。前两族（**S**型、**ReLU**族）是手工设计的固定形状函数，构成了深度学习早期实践中的主流；后五族分别代表五条不同的改进思路：用搜索代替手工设计、换用周期性的函数基、让激活曲线可学习、让激活函数看到上下文、以及把激活函数升级为一种网络结构。

## 3.1 S型激活函数

**S**型激活函数是形如**S**型曲线（**sigmoidal curve**）的一类激活函数，其特点是单调有界、处处光滑，因此天然满足准则1；但由于有界，两端必然饱和，违背准则3。这类函数是神经网络早期的主流选择，如今主要用于**门控机制**和**概率输出**。

### ⚪ Step：阶跃函数

$$ \text{Step}(x) = \begin{cases} 1, & x\geq 0 \\ 0, & x<0 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-002-step.png)

阶跃函数（**Heaviside**函数）直接对应生物神经元“达到阈值即点火”的建模，是[**感知机(perceptron)**](https://0809zheng.github.io/2020/03/11/perceptron.html)使用的激活函数。但它在$x=0$处不连续、其余各处导数恒为$0$，无法用于反向传播，因此只有历史意义。后续的**S**型函数都可以看作阶跃函数的光滑近似。

### ⚪ Sigmoid

$$
\begin{aligned}
\text{Sigmoid}(x)&=\sigma(x)=\frac{1}{1+e^{-x}} \\
\sigma'(x)&=\sigma(x)\left(1-\sigma(x)\right)
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-003-sigmoid.png)

**Sigmoid**（**logistic**函数）把实数域压缩到$(0,1)$，因此可以解释为概率，至今仍是二分类输出层和门控单元的标准选择。它的导数具有优美的自表达形式，这也是它在早期被广泛使用的原因之一。

**Sigmoid**的主要缺点：
- **两端饱和**：$|x|$较大时导数趋于$0$；
- **导数上界过小**：$\sigma'(x)\leq \sigma'(0)=1/4$，即使在非饱和区，每经过一层梯度也至少衰减$4$倍，这是深层网络中梯度消失的直接原因；
- **非zero-centered**：输出恒为正，产生偏置偏移（准则4）；
- **计算量大**：包含指数运算。

### ⚪ Tanh

$$
\begin{aligned}
\tanh(x)&=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = 2\sigma(2x)-1 \\
\tanh'(x)&=1-\tanh^2(x)
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-004-tanh.png)

**Tanh**是**Sigmoid**的缩放平移版本，值域为$(-1,1)$，因此是**zero-centered**的，缓解了偏置偏移；其导数在原点处取到$1$，梯度衰减问题也弱于**Sigmoid**。但它仍然两端饱和、仍然包含指数运算。在**LSTM/GRU**中，**Tanh**用于生成候选状态，**Sigmoid**用于生成门控信号。

### ⚪ HardSigmoid 与 HardTanh：分段线性近似

$$
\begin{aligned}
\text{HardSigmoid}(x)&=\begin{cases} 1, & x\geq 1 \\ (x+1)/2, & -1<x<1 \\ 0, & x\leq -1 \end{cases} \\
\text{HardTanh}(x)&=\begin{cases} 1, & x>1 \\ x, & -1\leq x\leq 1 \\ -1, & x<-1 \end{cases}
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-005-hardsigmoid-hardtanh.png)

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-006-hardsigmoid-hardtanh.png)

对**Sigmoid**和**Tanh**在原点处作一阶[<font color=Blue>Taylor展开</font>](https://0809zheng.github.io/2021/08/20/taylor.html#3-%E6%B3%B0%E5%8B%92%E5%85%AC%E5%BC%8F%E7%9A%84%E5%BA%94%E7%94%A8hard-sigmoid%E4%B8%8Ehard-tanh)并把结果截断到值域内，即可得到分段线性的**HardSigmoid**与**HardTanh**。它们只需加法、乘法和截断，完全避免了指数运算，是移动端和量化部署的常见替代品；代价是在$x=\pm 1$处不可导，且在$|x|>1$时梯度严格为$0$（饱和区比原函数更“硬”）。

### ⚪ ISRU：逆平方根单元

- paper：[Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)](https://arxiv.org/abs/1710.09967)

$$
\begin{aligned}
\text{ISRU}(x)&= \frac{x}{\sqrt{1 + \alpha x^2}} \\
\text{ISRU}'(x)&=\left(\frac{1}{\sqrt{1 + \alpha x^2}}\right)^3
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-007-isru.png)

**ISRU(inverse square root unit)**用**代数运算**近似**Tanh**：它的曲线形状与**Tanh**、**Sigmoid**相似，但只需一次逆平方根运算。相比**HardTanh**这种分段线性近似，**ISRU**保持了处处光滑；相比**Tanh**，它的计算显著更快。作者建议在**LSTM/GRU**中用**ISRU**替换**Tanh**与**Sigmoid**以加速循环网络。超参数$\alpha$控制曲线的饱和速度。

### ⚪ Softsign：多项式饱和的Tanh替代

$$
\begin{aligned}
\text{Softsign}(x)&= \frac{x}{1 + |x|} \\
\text{Softsign}'(x)&=\frac{1}{(1+|x|)^2}
\end{aligned}
$$

**Softsign**是一个更早的代数型**S**型函数（**PyTorch**中实现为`torch.nn.Softsign`）。它与**ISRU**属于同一思路：用有理式代替指数式。

**Softsign**与**Tanh**最本质的差别在**饱和速度**：**Tanh**的尾部以$e^{-2x}$**指数**衰减，而**Softsign**以$1/x$**多项式**衰减，因此其导数在远处按$1/x^2$而非指数级衰减：尾部“更肥”，梯度消失来得更晚。两者在原点处的导数都等于$1$，但**Softsign**因含$|x|$而在原点处二阶不可导（一阶导数连续，二阶导数有跳变）。它在现代网络中很少使用，主要作为**Tanh**的廉价替代出现在早期工作和一些序列模型中。

## 3.2 ReLU族激活函数

**ReLU**族是当前最主流的激活函数家族。它们的共同点是：正半轴保持（近似）恒等映射，从而在正半轴不饱和、梯度不衰减；差别主要在于*如何处理负半轴*，以及*如何处理原点处的不可导*。近年还出现了一条相反的思路：让正半轴增长得比线性更快。

### (1) ReLU 及其光滑与有界变体

#### ⚪ ReLU

- paper：[Rectified Linear Units Improve Restricted Boltzmann Machines](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

$$ \text{ReLU}(x)=\max(x,0)=\begin{cases} x, & x\geq 0 \\ 0, & x<0 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-008-relu.png)

**ReLU(rectified linear unit)**是深度学习中最重要的激活函数。它的优点几乎覆盖了准则2、3、5：
- **计算量极小**：只需一次比较；
- **正半轴无饱和**：导数恒为$1$，梯度可以无衰减地穿过任意多层，这是深层网络得以训练的关键；
- **生物可解释性好**：同时具有单侧抑制、宽兴奋边界和稀疏激活。

它的缺点集中在负半轴与原点：
- **dead ReLU**：若某个神经元的输入长期为负，其梯度恒为$0$，参数将永久停止更新，该神经元“死亡”；
- **非zero-centered**：输出恒非负，产生偏置偏移；
- **$x=0$处不可导**。

#### ⚪ Softplus：ReLU的光滑近似

- paper：[Incorporating Second-Order Functional Knowledge for Better Option Pricing](https://www.researchgate.net/publication/4933639_Incorporating_Second-Order_Functional_Knowledge_for_Better_Option_Pricing)

$$ \text{Softplus}(x)=\ln\left(1+e^x\right) = \int_{-\infty}^{x}\sigma(t)dt $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-009-softplus.png)

**Softplus**是**ReLU**处处可导的光滑上界，其导数恰为**Sigmoid**。但它牺牲了**ReLU**的两大优势：输出恒大于$0$（失去了稀疏性），且引入了指数与对数运算（计算量大）。因此实践中**Softplus**很少直接用作激活函数，更多用于**约束参数为正**（如输出方差）。

#### ⚪ Squareplus：Softplus的代数近似

- paper：[Squareplus: A Softplus-Like Algebraic Rectifier](https://arxiv.org/abs/2112.11687)

$$ \text{Squareplus}(x;b)=\frac{1}{2}\left(x+\sqrt{x^2+b}\right) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-010-squareplus.jpg)

**Squareplus**只用加法、乘法和平方根实现了与**Softplus**几乎相同的曲线。超参数$b$控制函数在$x=0$处的弯曲程度，其一阶、二阶导数为：

$$
\begin{aligned}
\text{Squareplus}'(x;b) &= \frac{1}{2}\left(1+\frac{x}{\sqrt{x^2+b}}\right) \\
\text{Squareplus}''(x;b) &= \frac{b}{2\left(x^2+b\right)^{3/2}}
\end{aligned}
$$

两者的对应关系相当整齐：**Softplus**的一阶导数是**S**型曲线（**logistic**分布的**CDF**），二阶导数是**logistic**分布的概率密度；而**Squareplus**的一阶导数是代数**S**型曲线，二阶导数是**student-t**分布的概率密度。

当$b=0$时**Squareplus**精确退化为**ReLU**：

$$ \text{Squareplus}(x;0) =\frac{x+|x|}{2} = \text{ReLU}(x) $$

为了确定与**Softplus**最接近的$b$，可以把问题建模成**min-max**形式，即最小化两个函数在全局的最大差异：

$$ \mathop{\min}_{b} \mathop{\max}_{x} \left|\frac{1}{2}\left(x+\sqrt{x^2+b}\right)-\ln\left(1+e^x\right)\right| $$

上式可以通过[<font color=Blue>非线性规划</font>](https://0809zheng.github.io/2021/08/23/minimize.html)求解：

```python
import numpy as np
from scipy.optimize import minimize

def f(x, b):
    return np.abs(0.5*(x+np.sqrt(x**2+b))-np.log(1+np.exp(x)))

def g(b):
    return np.max([f(x, b) for x in np.arange(-2, 4, 0.001)])

options = {'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 100000}
result = minimize(g, 0, method='Powell', options=options)
print(result.x) # [1.52382104]
```

注意到**Squareplus**和**Softplus**都是**ReLU**的上界；如果进一步希望**Squareplus**是**Softplus**的上界，则应有：

$$ \frac{1}{2}\left(x+\sqrt{x^2+b}\right) \geq \ln\left(1+e^x\right) \Longrightarrow b \geq 4\ln\left(1+e^x\right)\left[\ln\left(1+e^x\right)-x\right] $$

右式在$x=0$处取极大值，因此应有$b\geq 4 \ln^2 2$。

此外，当输入值较大时**Softplus**的数值稳定性较差（$e^x$会溢出）而偏离**ReLU**，**Squareplus**则不存在这一问题：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-011-squareplus.jpg)

实测在**CPU**上**Squareplus**比**Softplus**快约$6$倍、与**ReLU**相当；在**GPU**上由于受显存带宽限制，加速并不明显。因此**Squareplus**是计算资源受限场景下**Softplus**的理想替代品。

#### ⚪ ReLU6：为低精度部署设计

- paper：[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

$$ \text{ReLU6}(x)=\min\left(\max(x,0),6\right) =\begin{cases} 6, & x\geq 6 \\ x, & 0\leq x<6 \\ 0, & x<0 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-012-relu6.png)

**ReLU6**（由**MobileNet**引入）把**ReLU**的最大输出限制为$6$。这样做并非为了优化性质，而是为了**量化**：在移动端使用低精度数据类型(**float16/int8**)时，无上界的激活值可能分布在一个很大的动态范围内，导致量化时精度损失严重；限定上界后可以获得更好的数值分辨率。上界取$6$是经验选择，它足够大以至于不损害表示能力，又足够小以适配定点表示。

### (2) 修正负半轴斜率

#### ⚪ LeakyReLU

- paper：[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

$$ \text{LeakyReLU}(x)=\max(x,0.01x)=\begin{cases} x, & x\geq 0 \\ 0.01x, & x<0 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-013-leakyrelu.png)

**LeakyReLU**为负半轴引入一个很小的固定斜率，使得负输入的梯度不再为$0$，从而**根治了dead ReLU问题**：即使神经元当前输出为负，它仍能继续接收梯度并有机会“复活”。

**LeakyReLU**的论文还给出了一组有价值的分析：作者定义激活的**稀疏性(sparsity)**为激活值为$0$的比例，**离散性(disperseness)**为不同神经元激活概率的差异程度。实验发现，使用**ReLU**或**LeakyReLU**时神经元激活概率的标准差约为$0.04$，而使用**Tanh**时约为$0.14$。也就是说，**ReLU**族激活函数使得各个神经元的激活频率更加**均匀**，避免了少数神经元长期主导表示，这被认为是**ReLU**族激活函数优于**S**型函数的一个重要原因。

#### ⚪ PReLU：可学习的负半轴斜率

- paper：[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

$$
\begin{aligned}
\text{PReLU}(x)&=\max(x,\alpha x) =\begin{cases} x, & x\geq 0 \\ \alpha x, & x<0 \end{cases} \\
\frac{\partial \text{PReLU}(x)}{\partial \alpha} &= \begin{cases} 0, & x\geq 0 \\ x, & x<0 \end{cases}
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-014-prelu.png)

**PReLU(parametric ReLU)**把**LeakyReLU**中固定的斜率$\alpha$变成可学习参数，与网络权重一起用反向传播更新。$\alpha$可以按层共享，也可以按通道独立（**channel-wise**，实践中更常用，额外参数量仅等于通道数，可忽略）。

从几何上看，**PReLU**可以写成$y=\max\left(\alpha(Wx+b),Wx+b\right)$的形式：网络学到的超平面把空间分成两半，$\alpha$控制了负侧空间被**压缩**的程度。$\alpha=0$对应**ReLU**（负侧完全压扁到$0$），$\alpha=1$对应线性映射（不压缩），因此$\alpha$实际上在“非线性强度”这一维度上给了网络自由度。

需要注意，**PReLU**的论文同时提出了配套的**Kaiming初始化**，两者是一起工作的：使用整流类激活函数时，为保持前向传播中激活值方差不变，第$l$层的权重方差应取$\text{Var}\left[w^{(l)}\right]=2/d_{l-1}$（$d_{l-1}$为输入神经元个数）；若使用**PReLU/LeakyReLU**，则应修正为$\text{Var}\left[w^{(l)}\right]=2/\left[\left(\alpha^2+1\right)d_{l-1}\right]$。推导详见[<font color=Blue>参数初始化</font>](https://0809zheng.github.io/2020/03/05/initialization.html)部分。

#### ⚪ RReLU：随机采样的负半轴斜率

- paper：[Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853)

$$ \text{RReLU}(x)=\max(x,\alpha x), \quad \begin{cases} \alpha \sim U(l,u), & \text{during training} \\ \alpha = \dfrac{l+u}{2}, & \text{during test} \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-015-rrelu.jpg)

**RReLU(randomized ReLU)**在训练阶段从均匀分布中随机采样负半轴斜率，测试阶段则使用其期望。这与**Dropout**的“训练加噪、测试取期望”思想完全一致，因此**RReLU**本质上是一种**正则化**手段，在小数据集上尤其有效。

这篇论文的另一项贡献是通过系统的对比实验说明：**LeakyReLU/PReLU/RReLU**普遍优于**ReLU**，因此*稀疏性并不是整流类激活函数性能的关键因素*；恰恰相反，允许负值通过反而更好。这一结论直接推动了后续**ELU**、**GELU**、**Swish**等允许负输出的激活函数。

#### ⚪ CReLU：拼接正负两路整流

- paper：[Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units](https://arxiv.org/abs/1603.05201)

$$ \text{CReLU}(x)= \left[\text{ReLU}(x), \text{ReLU}(-x)\right] $$

前面几种方法都试图给负半轴一个“小一点”的输出，**CReLU**则换了一个角度：既然负半轴的信息有用，为什么不干脆把它**完整保留**？

作者的出发点是一个实验观察：卷积网络（尤其是**低层**）的滤波器往往**成对出现且方向近似相反**——同一组卷积核中存在大量负相关的“相位对”$w$与$-w$。这意味着网络实际上在用两个滤波器学习同一个模式的正负两个相位，而**ReLU**又把其中一半的响应抹成$0$，造成了双重浪费。

**CReLU**把$\text{ReLU}(x)$与$\text{ReLU}(-x)$在通道维**拼接**输出。这一变换是**信息保持**的：由两路输出可以完整恢复$x$（也可以恢复$|x|$），没有任何信息损失，同时输出仍然是稀疏的（每个位置恰有一路为$0$）。代价是输出通道数翻倍，因此实践中通常把滤波器数量减半以保持计算量不变；这样反而用一半的参数达到了更好的效果。

#### ⚪ SReLU：双侧可学习阈值

- paper：[Deep Learning with S-shaped Rectified Linear Activation Units](https://arxiv.org/abs/1512.07030)

$$ \text{SReLU}(x)=\begin{cases} a^r\left(x-t^r\right)+t^r, & x\geq t^r \\ x, & t^l < x < t^r \\ a^l\left(x-t^l\right)+t^l, & x\leq t^l \end{cases} $$

**SReLU(S-shaped ReLU)**由**三段**线性函数拼接而成，四个参数$t^r,a^r,t^l,a^l$全部可学习（$t^l,t^r$为左右转折点，$a^l,a^r$为两端斜率）。

**SReLU**的意义在于突破了**凸性**限制：**ReLU/LeakyReLU/PReLU**无论参数取何值都是**凸函数**（准确地说是两段线性的凸组合），而**SReLU**通过三段的不同斜率可以表示**非凸**的**S**型映射，从而能拟合诸如**Webber**定律、**Stevens**幂律这类心理物理学中的非凸响应曲线。它是手工设计走向**可学习分段线性激活函数**（3.5节的**APL**、**PWLU**）的一个前身。

#### ⚪ SUGAR：只替换反向梯度

- paper：[The Resurrection of the ReLU](https://arxiv.org/abs/2505.22074)

前面所有方法为了修好负半轴的梯度，都**改变了前向输出**，因而牺牲了**ReLU**的稀疏性与低廉的推理成本。**SUGAR(surrogate gradient)**指出这两件事其实可以**解耦**：激活函数同时承担了“前向的函数形状”和“反向的梯度形状”两个角色，而它们不必由同一个函数提供。

具体做法是：**前向传播完全保留ReLU**，反向传播时把负半轴恒为$0$的梯度替换为某个光滑替代函数（作者提出的**B-SiLU**是其中效果最好的一个）的导数。这样既保住了**ReLU**的稀疏激活与硬件友好性，又消除了**dead ReLU**。这种“前向硬、反向软”的思路来自**脉冲神经网络(SNN)**中处理不可导脉冲函数的标准技巧。

$$ \text{B-SiLU}(x)= (x+\alpha)\sigma(x) - \frac{\alpha}{2}, \quad \alpha = 1.67 $$

### (3) 负半轴指数饱和

#### ⚪ ELU

- paper：[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

$$
\begin{aligned}
\text{ELU}(x) &=\begin{cases} x, & x\geq 0 \\ \alpha\left(e^x-1\right), & x<0 \end{cases} \\
\text{ELU}'(x) &=\begin{cases} 1, & x\geq 0 \\ \text{ELU}(x)+\alpha, & x<0 \end{cases}
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-016-elu.png)

**ELU(exponential linear unit)**的设计目标是准则4：使激活函数的输出**均值接近$0$**，从而消除偏置偏移。它在负半轴取负值，且饱和到$-\alpha$。

这里的“饱和”是刻意设计的：**ELU**认为负半轴的饱和是一个优点而非缺点。当输入是较大的负值（通常意味着该特征“不存在”）时，函数输出趋于常数$-\alpha$，导数趋于$0$，因此**对输入噪声不敏感**；这被称为对噪声的**鲁棒性**。相比之下，**LeakyReLU/PReLU**在负半轴无界，噪声会被线性地传递下去。

注意**ELU**的一阶导数仅在$\alpha=1$时于$x=0$处连续（函数值的左极限$\alpha\left(e^0-1\right)=0$恰好等于右极限$0$，但导数的左极限为$\alpha$、右极限为$1$），因此$\alpha=1$是默认取值。

#### ⚪ CELU：连续可微的ELU

- paper：[Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483)

$$ \text{CELU}(x) =\begin{cases} x, & x\geq 0 \\ \alpha\left(e^{x/\alpha}-1\right), & x<0 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-017-celu.png)

**ELU**的一阶导数在$x=0$处的左右极限分别是$\alpha$和$1$，只有$\alpha=1$时才连续，这限制了$\alpha$的取值。**CELU(continuously differentiable ELU)**把指数项的自变量缩放为$x/\alpha$，使得负半轴导数为$e^{x/\alpha}$，在$x=0$处恒为$1$，从而对**任意$\alpha>0$都连续可微**。这带来了三个良好性质：

- **导数有界**：$0<\text{CELU}'(x)\leq 1$，避免了梯度放大；
- **尺度相似性(scale-similar)**：$\text{CELU}(x;\alpha)=\frac{1}{c}\text{CELU}(cx;c\alpha)$，即对输入的缩放可以被$\alpha$吸收；
- **良好的极限行为**：$\alpha \to 0^+$时$\text{CELU}\to\text{ReLU}$，$\alpha \to +\infty$时$\text{CELU}\to$恒等映射。

因此$\alpha$在**CELU**中扮演了“从**ReLU**平滑过渡到线性函数”的插值参数。

#### ⚪ SELU：自标准化的ELU

- paper：[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

$$ \text{SELU}(x) =\begin{cases} \lambda x, & x\geq 0 \\ \lambda\alpha\left(e^x-1\right), & x<0 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-018-selu.png)

**SELU(scaled ELU)**的设计思路是让激活函数本身承担**归一化**的功能：若激活函数的输入是均值为$0$、方差为$1$的独立同分布随机变量，则希望通过激活函数后的输出仍然保持均值为$0$、方差为$1$。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-019-selu.jpg)

激活函数处的数据流如上图所示。记$a_1,\cdots,a_K$为网络上一层的输出，假设它们是独立同分布的随机变量（不必服从**Gaussian**），求和后得到$z=\sum_{k=1}^{K}a_kw_k$；由中心极限定理$z$近似服从**Gaussian**。若能通过设计激活函数的参数使得本层输出$a=f(z)$仍然服从$\mathcal{N}(0,1)$，则该性质会逐层自动传递下去，网络具有**自标准化(self-normalizing)**性质。

**① 让输出服从$\mathcal{N}(0,1)$**。写出一阶统计量（均值$=0$）和二阶统计量（方差$=1$）对应的积分方程：

$$
\begin{aligned}
\int_{-\infty}^{0} \frac{\lambda\alpha\left(e^x-1\right)}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx + \int_{0}^{+\infty} \frac{\lambda x}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx &= 0 \\
\int_{-\infty}^{0} \frac{\lambda^2\alpha^2\left(e^x-1\right)^2}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx + \int_{0}^{+\infty} \frac{\lambda^2x^2}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}dx &= 1
\end{aligned}
$$

使用[<font color=Blue>sympy</font>](https://0809zheng.github.io/2021/09/01/solve.html)可以快速求解上述方程组：

```python
import sympy
from sympy import Symbol, nsolve, integrate

x = Symbol('x')
a = Symbol('a')
l = Symbol('l')
int1 = integrate(sympy.exp(-x**2/2)*(sympy.exp(x)-1), (x,-sympy.oo,0))
int2 = integrate(sympy.exp(-x**2/2)*x, (x,0,sympy.oo))
fn1 = a*l/sympy.sqrt(2*sympy.pi)*int1 + l/sympy.sqrt(2*sympy.pi)*int2 - 0
int3 = integrate(sympy.exp(-x**2/2)*(sympy.exp(x)-1)**2, (x,-sympy.oo,0))
int4 = integrate(sympy.exp(-x**2/2)*x**2, (x,0,sympy.oo))
fn2 = a**2*l**2/sympy.sqrt(2*sympy.pi)*int3 + l**2/sympy.sqrt(2*sympy.pi)*int4 - 1
z = nsolve([fn1,fn2], [a,l], [1,1])
print(z) # Matrix([[1.67326324235438], [1.05070098735548]])
```

即$\alpha=1.6732632423543772$，$\lambda=1.0507009873554805$。

**② 让输入服从$\mathcal{N}(0,1)$**。激活函数的输入为$z=\sum_{k=1}^{K}a_kw_k$。假设上一层输出独立同分布且$\mathbb{E}[a_k]=0$、$\mathbb{E}\left[a_k^2\right]=1$，则：

$$
\begin{aligned}
\mathbb{E}[z] &= \sum_{k=1}^{K}w_k\mathbb{E}[a_k] = 0 \\
\text{Var}[z] &= \mathbb{E}\left[\left(\sum_{k=1}^{K}a_kw_k\right)^2\right] = \sum_{k=1}^{K}w_k^2\mathbb{E}\left[a_k^2\right] = \sum_{k=1}^{K}w_k^2 = K \cdot \text{Var}[w]
\end{aligned}
$$

因此只需取$\mathbb{E}[w]=0$、$\text{Var}[w]=1/K$（即**LeCun**初始化），就有$\text{Var}[z]=1$。

综上，若所有隐藏层都使用**SELU**，则隐藏层的输入自动近似服从$\mathcal{N}(0,1)$；对输入层只需对数据做标准化；每一层的参数从$\mathcal{N}(0,1/K)$中采样即可。实验表明**SELU**具有类似**BatchNorm**的效果，能在**不使用任何归一化层**的情况下训练很深的全连接网络。这也是**SELU**最主要的应用场景。注意它对初始化方式和网络结构的要求较严格，在卷积网络中通常不如**ReLU**族$+$**BatchNorm**的组合。

#### ⚪ ISRLU：ELU的代数近似

- paper：[Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)](https://arxiv.org/abs/1710.09967)

$$
\begin{aligned}
\text{ISRLU}(x) &= \begin{cases} x, & x \geq 0 \\ \dfrac{x}{\sqrt{1 + \alpha x^2}}, & x < 0 \end{cases} \\
\text{ISRLU}'(x) &= \begin{cases} 1, & x \geq 0 \\ \left(\dfrac{1}{\sqrt{1 + \alpha x^2}}\right)^3, & x < 0 \end{cases}
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-020-isrlu.png)

**ISRLU(inverse square root linear unit)**用逆平方根替代**ELU**中的指数运算。它与**ELU**的曲线非常相似（负半轴饱和、输出均值接近$0$），但有两个优势：

- **更光滑**：**ELU**只有一阶导数连续，而**ISRLU**的一阶和二阶导数都连续；
- **更快**：在现代**CPU**上逆平方根比指数运算快得多，实测**ISRLU**($\alpha=1$)比**ELU**快约$2.63$倍，其快速近似版本与**ReLU**的速度仅差$1\%$。

超参数$\alpha$控制负半轴的饱和值（$\alpha=1$时饱和值接近$-1$，$\alpha=3$时饱和值更小、向下层传递的误差信号更少），并且可以像**PReLU**一样通过反向传播学习。

#### ⚪ PELU：参数化的ELU

- paper：[Parametric Exponential Linear Unit for Deep Convolutional Neural Networks](https://arxiv.org/abs/1605.09332)

$$ \text{PELU}(x) =\begin{cases} \dfrac{a}{b}x, & x\geq 0 \\ a\left(e^{x/b}-1\right), & x<0 \end{cases}, \quad a,b>0 $$

**PReLU**对**ReLU**做的事情（把固定超参数变成可学习参数）同样可以对**ELU**做。但**PELU(parametric ELU)**比**PReLU**更彻底：它不只参数化负半轴，而是同时控制三件事：**正半轴的斜率**$a/b$、**负半轴的饱和值**$-a$、以及**负半轴的衰减速率**$1/b$。

正半轴的斜率之所以被写成$a/b$这个看似别扭的耦合形式，是为了保证函数在原点处**可导**：负半轴导数在$x\to 0^-$时为$(a/b)e^{0}=a/b$，与正半轴斜率恰好相等。若正半轴直接取$x$（斜率$1$），则只有$a=b$时才可导，参数就失去了自由度。

训练时需要保证$a,b>0$（原论文通过重参数化或截断实现）。其效果是把**ELU**中“偏置偏移修正的强度”从固定的$\alpha=1$交给网络自己决定，不同层可以学到差别很大的曲线形状。

### (4) 概率视角

#### ⚪ GELU

- paper：[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

$$ \text{GELU}(x)= x\Phi(x)=x\int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt = \frac{x}{2}\left(1+\text{erf}\left(\frac{x}{\sqrt{2}}\right)\right) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-021-gelu.png)

**GELU(Gaussian error linear unit)**是**Transformer**时代最重要的激活函数，其设计动机来自一个全新的视角：**把激活函数与随机正则化统一起来**。

考虑一个随机的**输入自适应**掩码：以概率$\Phi(x)=P(X\leq x),X\sim \mathcal{N}(0,1)$保留输入、以概率$1-\Phi(x)$置零，即$x\cdot \text{Bernoulli}(\Phi(x))$。这个掩码同时具有**Dropout**的随机性和**ReLU**的门控性，但与两者不同的是：**Dropout**的保留概率与输入无关，**ReLU**的门控是确定性的$\mathbb{1}[x>0]$，而这里的门控概率**由输入本身的大小决定**：输入越大越可能被保留。**GELU**就是这个随机变换的**期望**：

$$ \mathbb{E}\left[x\cdot \text{Bernoulli}(\Phi(x))\right] = x\Phi(x) $$

从这个视角看，**ReLU**是$x\mathbb{1}[x>0]$，即把$\Phi$替换成阶跃函数的极限情形；**GELU**则是它的“概率化”推广，因此天然带有正则化效果。**GELU**的曲线是非单调的（在负半轴有一个极小值），这与后文**Swish**、**Mish**的形状一致。

由于$\text{erf}$函数计算代价较高，实践中常用两种近似：

$$
\begin{aligned}
\text{GELU}(x) &\approx \frac{x}{2}\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right) \\
\text{GELU}(x) &\approx x\cdot \sigma(1.702x)
\end{aligned}
$$

其中的常数可以通过**min-max**拟合（最小化近似式与精确式的全局最大误差）确定，求解方式与前文**Squareplus**中的[<font color=Blue>非线性规划</font>](https://0809zheng.github.io/2021/08/23/minimize.html)完全相同。注意第二个近似恰好是$\beta=1.702$的**Swish**函数——这暗示了**GELU**与**Swish**之间的深刻联系，3.5节会从光滑化的角度给出解释。

### (5) 幂与多项式

到目前为止，所有整流类变体都在**负半轴**上做文章，而正半轴一律保持恒等映射。$2021$年之后，随着大语言模型的兴起，一条相反的路线浮现出来：**修改正半轴，让它比线性增长得更快**。这类激活函数几乎构成了近年大模型前馈层的全部进展。

#### ⚪ Squared ReLU：搜索出来的平方整流

- paper：[Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668)

$$ \text{ReLU}^2(x)= \left(\text{ReLU}(x)\right)^2 = \max(x,0)^2 $$

**Primer**用**进化式结构搜索**在**TensorFlow**原语组成的空间中搜索更高效的**Transformer**，搜出的若干改动中，**把前馈层的ReLU换成它的平方**是单项收益最大、也最容易迁移的一个。

从传统设计准则看，**ReLU**$^2$是相当反常的：它上无界，且导数$2\text{ReLU}(x)$随输入**线性增长**而非趋于常数或饱和，理论上有梯度爆炸的风险。但它在**Transformer**前馈层中表现稳定，且比**GELU**便宜得多（一次乘法，不含$\exp$或$\text{erf}$）。

它的另一个性质在大模型时代变得格外重要：平方放大了大激活值、压低了小激活值，使得前馈层的激活分布**远比GELU稀疏且集中**。[ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs](https://arxiv.org/abs/2402.03804) 据此论证**ReLU**$^2$是稀疏化推理的最佳选择：可以只计算少数被激活的神经元。目前**Nemotron-H**等生产级模型即使用**ReLU**$^2$作为前馈层激活。

#### ⚪ StarReLU：把平方整流标准化

- paper：[MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452)

$$ \text{StarReLU}(x)= s\cdot\left(\text{ReLU}(x)\right)^2 + b $$

**ReLU**$^2$有一个直接的问题：它的输出既非零均值也非单位方差（平方使分布严重右偏）。**StarReLU**用一个缩放$s$和一个偏移$b$把它**标准化**回来。

这两个常数可以精确算出。设$x\sim \mathcal{N}(0,1)$，则：

$$
\begin{aligned}
\mathbb{E}\left[\left(\text{ReLU}(x)\right)^2\right] &= \mathbb{E}\left[x^2\mathbb{1}[x>0]\right] = \frac{1}{2} \\
\mathbb{E}\left[\left(\text{ReLU}(x)\right)^4\right] &= \mathbb{E}\left[x^4\mathbb{1}[x>0]\right] = \frac{3}{2} \\
\text{Var}\left[\left(\text{ReLU}(x)\right)^2\right] &= \frac{3}{2}-\left(\frac{1}{2}\right)^2 = \frac{5}{4}
\end{aligned}
$$

要使输出零均值、单位方差，只需取：

$$ s = \frac{1}{\sqrt{5/4}} \approx 0.8944, \quad b = -\frac{1/2}{\sqrt{5/4}} \approx -0.4472 $$

$s$与$b$也可以设为**可学习**参数（按层或按通道），此时上述值作为初始化。

**StarReLU**的另一个优势是计算量。**GELU**的$\text{erf}$或$\tanh$形式需要多次超越函数运算，而**StarReLU**只需一次乘法和一次加法；论文统计其**激活函数部分的FLOPs比GELU减少约71%**。

#### ⚪ xIELU：正半轴二次、负半轴指数

- paper：[Deriving Activation Functions Using Integration](https://arxiv.org/abs/2411.13010)

$$ \text{xIELU}(x) =\begin{cases} \alpha_p x^2 + 0.5x, & x> 0 \\ \alpha_n\left(e^x-1\right)-\alpha_n x + 0.5x, & x\leq 0 \end{cases} $$

**xIELU**的推导方式本身就很有启发性：与其直接设计激活函数$f$，不如**设计它的导数**$f'$，因为我们真正关心的性质（梯度是否消失、是否饱和、是否有界）全都是关于导数的；然后对$f'$**积分**得到$f$。这样所需的梯度行为是构造性地保证的，而不是事后验证的。作者取$f'$为**ELU**的可训练仿射变换，积分后即得到上式。

从形式上看，**xIELU**把两条路线拼在一起：**正半轴借用ReLU**$^2$**的增长型导数**（$2\alpha_p x + 0.5$随输入增大），**负半轴借用ELU的饱和**（$\alpha_n\left(e^x-1\right)$），$\alpha_p$与$\alpha_n$均可学习。

式中的$0.5x$项与$-\alpha_n x$修正项不是随意添加的，它们的作用恰好是保证原点处的**连续可导**：$x\to 0^+$与$x\to 0^-$的函数值都是$0$；导数方面，右导数为$2\alpha_p\cdot 0+0.5=0.5$，左导数为$\alpha_n e^0-\alpha_n+0.5=0.5$，两者相等。因此无论$\alpha_p,\alpha_n$学成什么值，**xIELU**始终是$C^1$的。

**xIELU**已被用于**Apertus 8B/70B**开源模型（[Apertus: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233)）。

#### ⚪ PolyCom：多项式组合

- paper：[Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models](https://arxiv.org/abs/2411.03884)

既然二次项有效，为什么不用**任意阶多项式**？**PolyCom**给出两个变体：

$$
\begin{aligned}
\text{PolyReLU}(x) &= \sum_{i=0}^{r} a_i \left(\text{ReLU}(x)\right)^i \\
\text{PolyNorm}(x) &= \sum_{i=0}^{r} a_i \frac{x^i}{\left\| x^i \right\|_2}
\end{aligned}
$$

其中系数$a_i$可学习，实践中取$r=3$。

**表达能力**是这项工作的核心论证：$r$阶多项式组合可以达到**Sobolev**空间中的最优逼近速率$O\left(\epsilon^{-d/n}\right)$，而纯**ReLU**网络要达到同样的速率需要多得多的层数。也就是说，把非线性做得“更强”可以换取深度。

**PolyNorm**中的$\left\| x^i \right\|_2$是让多项式激活在大模型规模下真正可训练的关键：高次项的数值范围随$i$急剧膨胀（$x^3$的尺度远大于$x$），直接相加会让高次项完全主导并溢出；除以各自的$L_2$范数后，每一项都被归一化到可比的尺度，训练才能稳定。

#### ⭐ 讨论：为什么“不饱和”反而更好

这一族的共同特征是**正半轴的导数随输入增长而非饱和**，这与第2章设计准则中的一个隐含期待（导数有界）正好相反。

一个合理的解释是：这类激活函数几乎只出现在**Transformer**的前馈层中，而该模块的输入已经被**LayerNorm/RMSNorm**归一化、输出又紧接着残差连接与下一个归一化层。激活值的尺度问题在**块的层面**上已经由归一化解决了，因此激活函数本身可以“放开手脚”去追求更强的非线性与更稀疏的表示。换句话说，激活函数的设计准则并不是绝对的，它依赖于激活函数所处的结构上下文。

## 3.3 自动搜索的激活函数

手工设计激活函数依赖直觉和经验。一个自然的想法是：既然网络结构可以用**神经结构搜索(NAS)**自动设计，激活函数为什么不能？这条路线产生了目前实际效果最好的几个激活函数。

### ⚪ Swish：强化学习搜索

- paper：[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)

**Swish**是用自动搜索技术找到的激活函数。首先需要设计合适的**搜索空间**：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-022-swish.jpg)

该搜索空间由**core unit**递归地构造而成。**core unit**接收两个输入（其中一个可以是上一个**core unit**的输出），分别经过两次**一元(unary)**操作后使用一个**二元(binary)**操作进行组合，并得到输出。作者选用的一元操作和二元操作如下：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-023-swish.jpg)

搜索使用一个**RNN**控制器，在每个时间步预测激活函数的一个组成部分。当搜索出一个完整的激活函数后，构造对应的**ResNet-20**子网络，在**CIFAR-10**上训练$10K$步并记录验证准确率；该准确率作为**reward**通过强化学习更新控制器。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-024-swish.jpg)

搜索得到的表现较好的激活函数如上图所示，从中可以总结出几条**经验规律**（这些规律对手工设计激活函数同样有指导意义）：
- **复杂的激活函数表现不如简单的激活函数**，可能是因为复杂函数导致优化更加困难；表现最好的激活函数通常只有$1$~$2$个**core unit**；
- 表现较好的激活函数通常会**使用$x$作为输入的一部分**（即保留恒等路径）；
- 一些表现较好的激活函数使用了**周期函数**（如$\sin,\cos$），且以加减的形式出现，这类函数此前研究较少；
- **使用除法的激活函数通常表现较差**，因为分母接近$0$时数值爆炸；只有当分子分母同时接近$0$时才有较好的表现（如$\cosh$）。

搜索得到的最好的激活函数被命名为**Swish**（在**PyTorch**中称为**SiLU**）：

$$ \text{Swish}(x)=x\cdot\sigma(\beta x)=\frac{x}{1+e^{-\beta x}} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-025-swish.png)

**Swish**具有一系列良好性质。首先它是**ReLU**与线性函数之间的光滑插值：当$\beta=0$时退化为线性函数$x/2$；当$\beta\to +\infty$时$\sigma(\beta x)$趋于阶跃函数，**Swish**退化为**ReLU**。因此$\beta$控制了函数的非线性程度。

与**ReLU**相同，**Swish**无上界、有下界；不同于**ReLU**，**Swish**是**处处光滑**且**非单调**的（负半轴存在一个极小值）。其导数为：

$$
\begin{aligned}
\text{Swish}'(x) &= \sigma(\beta x) + \beta x \cdot \sigma(\beta x)\left(1-\sigma(\beta x)\right) \\
&= \beta\,\text{Swish}(x) + \sigma(\beta x)\left(1-\beta\,\text{Swish}(x)\right)
\end{aligned}
$$

不同于**ReLU**在$x>0$时导数恒为$1$，**Swish**的导数是随输入连续变化的。

受**LSTM**中**门控(gating)**机制的启发，**Swish**可以理解为一种**自门控(self-gating)**机制：使用自身的值作为门控信号，当$\sigma(\beta x)$接近$1$时门“开”，接近$0$时门“关”。

参数$\beta$也可以通过学习得到，此时网络中每个神经元的激活函数形状都可以不同。作者统计了训练后$\beta$的取值分布，大多数集中在$1$附近，因此实践中通常直接固定$\beta=1$：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-026-swish.jpg)

### ⚪ HardSwish：Swish的分段线性近似

- paper：[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

$$ \text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6} = \begin{cases} x, & x \geq 3 \\ \dfrac{x(x+3)}{6}, & -3 \leq x <3 \\ 0, & x < -3 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-027-hardswish.jpg)

**Swish**中的指数运算在嵌入式环境中成本较高，**MobileNet V3**因此提出用$\text{ReLU6}(x+3)/6$（即**HardSigmoid**的一种形式）替代**Sigmoid**，得到只含加法、乘法和截断的**HardSwish**。由于**Swish**在深层网络中收益更明显，而深层特征图的分辨率更低、激活函数调用次数更少，**MobileNet V3**只在网络的**深层**使用**HardSwish**，浅层仍使用**ReLU**。

### ⚪ Mish

- paper：[Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)

$$ \text{Mish}(x) = x\cdot \tanh\left(\text{Softplus}(x)\right) =x\cdot \tanh\left(\ln\left(1+e^x\right)\right) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-028-mish.png)

**Mish**是在**Swish**周边的函数空间中进一步手工搜索得到的：作者设计了一批与**Swish**形式相似的候选函数并逐一实验，最终选出表现最好、训练最稳定的一个。**Mish**具有四个特点：
- **连续可微**：避免了梯度优化时由于奇点引入的副作用；
- **无上界**：避免饱和导致的梯度衰减；
- **有下界**：下界约为$-0.30884$，具有隐式正则化的作用；
- **非单调**：保留了一定程度的负值信息，增强了表现力和信息流动。

**Mish**最有说服力的证据来自**损失平面(loss landscape)**的可视化。所谓**输出平面(output landscape)**是指随机初始化一个网络，将可视化空间的坐标输入网络并输出相应标量；**Mish**对应的输出平面比**ReLU**更加平滑，而更平滑的输出平面会产生更平滑的损失平面：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-029-mish.jpg)

上图对比了**ReLU**、**Mish**、**Swish**对应的损失平面。**Mish**的损失平面更平滑、具有更宽的最小值区域、取得的损失值也更小；而其余两个损失平面存在多个局部极小值。

**Mish**与**Swish**之间存在直接联系。对**Mish**求导：

$$
\begin{aligned}
\text{Mish}'(x) &= \tanh\left(\text{Softplus}(x)\right) +x\cdot \text{sech}^2\left(\text{Softplus}(x)\right)\cdot \sigma(x) \\
&= \text{sech}^2\left(\text{Softplus}(x)\right)\cdot\left(x\cdot \sigma(x)\right) + \tanh\left(\text{Softplus}(x)\right) \\
&= \Delta(x)\,\text{Swish}(x)+\frac{\text{Mish}(x)}{x}
\end{aligned}
$$

其中$\Delta(x)=\text{sech}^2\left(\text{Softplus}(x)\right)$相当于一个**预条件算子(preconditioner)**，使得梯度更加平滑并提供更强的正则化效果。这被认为是**Mish**在更深、更复杂的网络中表现优于**Swish**的原因。

### ⚪ ELiSH 与 HardELiSH：遗传算法搜索

- paper：[The Quest for the Golden Activation Function](https://arxiv.org/abs/1808.00783)

$$ \text{ELiSH}(x) =\text{ELU}(x)\cdot \sigma(x) = \begin{cases} \dfrac{x}{1+e^{-x}}, & x\geq 0 \\ \dfrac{e^x-1}{1+e^{-x}}, & x<0 \end{cases} $$

$$ \text{HardELiSH}(x) =\text{ELU}(x)\cdot \text{HardSigmoid}(x) = \begin{cases} x, & x\geq 1 \\ x(x+1)/2, & 0 \leq x<1 \\ \left(e^x-1\right)(x+1)/2, & -1\leq x<0 \\ 0, & x\leq -1 \end{cases} $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-030-elish-hardelish.jpg)

**ELiSH(exponential linear sigmoid squashing)**沿用了**Swish**的“自门控”结构，但把被门控的对象从$x$换成$\text{ELU}(x)$：当$x>0$时它与**Swish**完全相同；当$x<0$时它继承了**ELU**减少偏置偏移、对噪声鲁棒的特点。**HardELiSH**则进一步把**Sigmoid**替换为**HardSigmoid**以降低计算量。

这篇论文更重要的贡献是提出用**遗传算法(genetic algorithm)**为特定任务搜索激活函数。遗传算法维护一个**个体(individual)**总体，每个个体由若干**基因(gene)**描述，通过迭代执行以下操作进化：
- **选择(selection)**：选出**适应度(fitness)**最高的两个个体作为父代；
- **交叉(crossover)**：组合两个父代生成子代；
- **变异(mutation)**：随机修改子代的若干基因。

关键在于如何把激活函数编码为基因。作者观察到常用的激活函数大多是**分段(piece-wise)**的，即可以拆分为左右两部分（或两个因子的乘积），于是把每一部分指定为一个基因，并设计了两种交叉方式：**遗传(inheritance)**（随机取一个个体的左半与另一个体的右半）和**杂交(hybrid)**（对两个个体的左右部分分别做某种运算再组合）；**变异**则是随机替换其中一个部分。相比**Swish**使用的强化学习搜索，遗传算法的搜索成本更低，代价是搜索空间受编码方式的限制更强。

### ⚪ 手工设计的自门控变体：E-Swish、LiSHT、TanhExp

- paper：[E-swish: Adjusting Activations to Different Network Depths](https://arxiv.org/abs/1801.07145)
- paper：[LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks](https://arxiv.org/abs/1901.05894)
- paper：[TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks](https://arxiv.org/abs/2003.09855)

搜索给出了$x\cdot g(x)$这一**自门控**模板之后，手工设计的空间就变成了“换一个$g$”。下面三个是最常见的变体的：

$$
\begin{aligned}
\text{E-Swish}(x) &= \beta x\sigma(x), \quad \beta \in [1,2] \\
\text{LiSHT}(x) &= x\tanh(x) \\
\text{TanhExp}(x) &= x\tanh\left(e^x\right)
\end{aligned}
$$

**E-Swish**只是给**Swish**乘上一个常数$\beta$（推荐$\beta=1.75$），看起来微不足道，但它改变的是函数的**最大导数**：$\beta>1$使正半轴的梯度整体放大，等价于把该层的有效学习率调高。作者据此建议按网络深度选取不同的$\beta$：浅层用较大的$\beta$，深层用较小的$\beta$。

**LiSHT**是这一组中最特殊的：$x\tanh(x)$是一个**偶函数**且**恒非负**（可以写成$|x|\tanh|x|$），形状是一条**V**型曲线。它完全丢弃了输入的符号而只保留幅值，这与前面所有激活函数都不同。它以一种独特的方式满足准则4：输出虽然全为正，但函数关于$y$轴对称，正负输入被同等对待，因此不存在方向性的偏置偏移。代价是符号信息的丢失必须由后续层从其他通道恢复。

**TanhExp**面向**轻量级网络**设计。它在原点附近的过渡比**Swish**和**Mish**都更陡（因为$e^x$在$0$附近变化快，使$\tanh$迅速接近饱和），从而在小模型上收敛更快；同时它只需一次$\exp$和一次$\tanh$。

### ⚪ GELUSine：LLM搜索出来的激活函数

- paper：[Mining Generalizable Activation Functions](https://arxiv.org/abs/2602.05688)

$$ \text{GELUSine}(x) = \text{GELU}(x) + 0.1\sin(x) $$

激活函数搜索的第三代工具是**大语言模型驱动的进化式搜索智能体**：不再由人设计搜索空间或基因编码，而是让模型直接提出、评估并改写候选程序（该工作使用的是**AlphaEvolve**）。**GELUSine**是这类方法给出的结果之一。

有意思的是它恰好闭合了一个从$2017$年打开的循环：**Swish**的原始论文在报告搜索结果时就注意到，*含周期性成分的候选函数表现出乎意料地好*，但作者没有深究。**GELUSine**几乎就是这一观察的兑现：在标准激活函数上叠加一个**幅度很小的周期扰动**。周期性为什么有帮助，见下一节3.4。


## 3.4 周期性激活函数

前面所有激活函数都是**非周期**的、并且大体上是单调递增的。由它们堆叠出的网络有一个被反复观察到的固有偏好：**谱偏差(spectral bias)**，也称**频率原则(frequency principle)**：网络总是先拟合目标函数的**低频**成分，而高频细节要么学得极慢，要么根本学不出来。

对分类任务而言这甚至是好事（相当于一种隐式正则化），但对**隐式神经表示(implicit neural representation)**是致命的：这类任务要求一个小型**MLP**把**坐标**映射到信号值（图像的像素、音频的采样、形状的符号距离、场景的辐射亮度），而目标信号本身就充满高频结构。

周期性激活函数直接从根源上解决这个问题：让函数基本身就是**振荡**的。

### ⚪ SIREN：正弦激活的隐式神经表示

- paper：[Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)

$$ \Phi_i(x) = \sin\left(\omega_0\left(W_i x + b_i\right)\right) $$

**SIREN(sinusoidal representation network)**就是把**MLP**每一层的激活函数换成正弦。这个改动看似简单，却带来一个其他激活函数都没有的性质：

**SIREN的导数仍然是一个SIREN**。因为$\sin$的导数是相移后的$\sin$，逐层求导后网络的结构形式不变。这使得**SIREN可以被导数约束监督**：可以只用图像的**梯度**或**Laplacian**去拟合图像本身，可以求解符号距离场的**Eikonal**方程，也可以求解**Poisson**方程和**Helmholtz**方程。**ReLU**网络做不到这一点：**ReLU**网络的二阶导数几乎处处为$0$，任何涉及二阶导数的监督信号都会直接消失。

超参数$\omega_0$（论文取$30$）控制网络能够表示的**频带**：$\omega_0$越大，可表示的细节越精细，但也越容易过拟合噪声。它是**SIREN**中最重要的单个超参数。

**SIREN**能否训练起来强烈依赖于一套配套的**初始化方案**。隐藏层权重需要从下面的均匀分布中采样（$n$为该层输入维度）：

$$ w \sim U\left(-\sqrt{\frac{6}{n}}, \sqrt{\frac{6}{n}}\right) $$

这样做的目的是使**激活前的分布在深度方向上保持不变**（否则正弦的周期性会让分布迅速退化）；第一层则额外乘以$\omega_0$。缺少这套初始化，**SIREN**可能根本训练不出来。


### ⚪ Snake：周期性与单调性的折中

- paper：[Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195)

$$
\begin{aligned}
\text{Snake}_a(x) &= x + \frac{1}{a}\sin^2(ax) \\
\text{Snake}_a'(x) &= 1+\sin(2ax)
\end{aligned}
$$

纯正弦激活放弃了单调性，这让优化变得困难（存在大量局部极小）。**Snake**给出一个折中：**一个单调的恒等项加上一个有界的周期项**。

这个组合的好处从导数上看得最清楚：$\text{Snake}_a'(x) \in [0,2]$，**恒非负**，因此**Snake**是**单调不减**的，优化行为接近**ReLU**族；同时周期项又赋予它表示与**外推**周期结构的能力。系数$1/a$的作用是把周期项的**振幅与频率绑定**：频率$a$越高，振幅越小，从而保证无论$a$取何值，函数都不会偏离恒等映射太远。

这篇论文的核心观察是：标准网络在周期信号上的**插值**几乎完美，但**外推**（超出训练区间）会灾难性失败：它们学到的是训练区间内的一段曲线，而不是“周期”这个概念。**Snake**及其按通道可学习的变体**snakebeta**如今是现代**神经声码器**（如**BigVGAN**）的标准激活函数，原因很直接：音频波形本质上是周期的。

### ⚪ GCU：余弦门控单元

- paper：[Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks](https://arxiv.org/abs/2108.12943)

$$ \text{GCU}(x) = x\cos(x) $$

**GCU(growing cosine unit)**是自门控模板$x\cdot g(x)$的周期版本：把门控函数换成$\cos$。它是非单调的，并且有**无穷多个零点**。

作者对此的解释是：由于每个零点都对应一个决策边界，**单个GCU神经元就可以表示XOR这类非线性可分问题**；而任何单调激活函数的单个神经元只能给出一个超平面。

### ⭐ 讨论：周期性激活的适用范围

周期性激活函数基本上是**领域专用工具**。它们在**隐式神经表示**、**音频合成**、**物理信息神经网络(PINN)**中占据主导地位，但在图像分类和语言模型的骨干网络中很少使用；因为那里的输入不是坐标，目标函数也不具有真正的周期结构。**GELUSine**是一个有意思的例外：它不改变主干激活函数的形状，只是叠加一点**微量**的周期性（幅度$0.1$），试图在通用任务上取得谱偏差与优化难度之间的平衡。

## 3.5 通用近似激活函数

前面所有激活函数都是**固定形状**的（**PReLU**只有一个自由度）。一个更激进的想法是：**让每个神经元自己学出它需要的激活曲线**。这要求激活函数是一个具有足够表达能力的**参数化函数族**，且参数可以通过梯度更新。

这类方法有三条技术路线：**直接用通用逼近器参数化**（下面的(1)分段线性、(2)有理函数、(3)样条），**对已有的非光滑激活函数做参数化的光滑近似**（下面的(4)），以及**用一个参数族统一多个已知激活函数**（下面的(5)）。

### (1) 分段线性参数化

#### ⚪ Maxout

- paper：[Maxout Networks](https://arxiv.org/abs/1302.4389)

$$ h_i(x)=\mathop{\max}_{j\in [1,k]}\left(x^\top W_{i,j}+b_{i,j}\right) $$

**Maxout**是最早的可学习激活函数。它把$k$个线性函数取最大值作为激活输出，因此**maxout**单元本身就是一个**分段线性凸函数**，并且可以逼近任意凸函数（只要$k$足够大）。更进一步，作者证明**两个maxout单元的差可以逼近任意连续函数**，因此**maxout**网络具有通用近似性。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-031-maxout.jpg)

**ReLU**和**LeakyReLU**都是**maxout**的特例（$k=2$，其中一支为$0$或$\alpha x$），这也说明整流类激活函数本质上是分段线性的。**maxout**的代价与缺点是：
- **参数量翻$k$倍**（每个输出需要$k$套权重）；
- 输出**非稀疏**且**无上界**；
- 只能表示**凸函数**（这是取$\max$带来的固有限制）。

注意**maxout**同时也是第一个**多输入**的激活函数（它作用在整个输入向量上，而非单个标量），因此也可归入3.6节。

#### ⚪ APL：用ReLU拼出分段线性

- paper：[Learning Activation Functions to Improve Deep Neural Networks](https://arxiv.org/abs/1412.6830)

$$ \text{APL}(x)=\max(0,x)+\sum_{s=1}^{S}a^s\max\left(0,-x+b^s\right) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-032-apl.jpg)

**APL(adaptive piecewise linear unit)**换了一种参数化方式：以**ReLU**为主干，叠加$S$个带有可学习斜率$a^s$和折点$b^s$的**hinge**项。它相比**maxout**有两个优势：
- **参数量小**：每个神经元只需$2S$个额外参数，与输入维度无关；
- **可以表示非凸函数**：因为$a^s$可以为负（而**maxout**取$\max$只能得到凸函数）。

同时它是**单输入**的标量函数，可以直接替换任何现有网络中的**ReLU**，无需改动网络结构。

#### ⚪ PWLU：直接参数化分段线性函数

- paper：[Learning specialized activation functions with the Piecewise Linear Unit](https://arxiv.org/abs/2104.03693)

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-033-pwlu.jpg)

**PWLU(piecewise linear unit)**是分段线性路线的集大成者，它直接用“**区间$+$端点值**”来参数化函数。给定区间数量$N$、左右边界$B_L,B_R$、$N+1$个分界点的函数值$Y^0,\cdots,Y^N$、以及最左端与最右端的斜率$K_L,K_R$：

$$ \text{PWLU}_N(x) = \begin{cases} \left(x-B_L\right)K_L+Y^0, & x<B_L \\ \left(x-B_R\right)K_R+Y^N, & x\geq B_R \\ \left(x-B_i\right)K_i+Y^i, & B_L\leq x<B_R \end{cases} $$

区间$[B_L,B_R]$被**均分**为$N$份，因此子区间宽度为$d$，$x$所属子区间的索引$i$及该区间的左端点与斜率都可以$O(1)$地算出：

$$
\begin{aligned}
d &= \frac{B_R-B_L}{N}, \quad i = \left\lfloor \frac{x-B_L}{d} \right\rfloor \\
B_i &= B_L+i\cdot d, \quad K_i = \frac{Y^{i+1}-Y^{i}}{d}
\end{aligned}
$$

**PWLU**具有以下性质：
- 是通用近似器，可以近似任意连续有界标量函数（$N$越大拟合能力越强）；
- 随其参数连续变化，因此可以通过梯度优化；![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-034-pwlu.jpg)
- 自由度集中在一个**有界区间**内，可以充分利用可学习参数；
- 由于区间等分，索引计算和推理都非常高效。

**PWLU**的关键实践问题是**输入边界不对齐(input-boundary misalignment)**：函数的主要自由度分布在$[B_L,B_R]$内，如果输入数据的分布与该区间交集很小，则大部分参数对网络几乎没有贡献。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-035-pwlu.jpg)

作者提出了一种基于统计的**两阶段对齐方法**：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-036-pwlu.jpg)

首先把所有**PWLU**初始化为**ReLU**（取$N$为偶数，此时$K_L=0,K_R=1,B_L=-B_R,Y^i=\text{ReLU}(B_i)$）。第一阶段（前$T'$轮，实验中$T'=5$）不更新**PWLU**的参数，只统计输入数据的均值与标准差并滑动更新：

$$ \mu \leftarrow 0.9\mu+0.1\cdot\text{mean}(x), \quad s \leftarrow 0.9s+0.1\cdot\text{std}(x) $$

第二阶段把每个**PWLU**的参数按统计量重置，使有效区间覆盖输入分布的$\pm 3s$：

$$ B_L=\mu-3s, \quad B_R=\mu+3s, \quad K_L=0, \quad K_R=1, \quad Y^i=\text{ReLU}(B_i) $$

此后再通过梯度更新**PWLU**的参数。可视化学到的函数可以发现：浅层的激活函数接近线性函数，而深层的激活函数呈“**V**”型。

**PWLU**的论文也总结了自动搜索路线的局限性，这正是通用近似路线的动机所在：搜索空间是**受限且离散**的，组合的微小变化可能导致完全不同的结果；搜索需要评估大量候选函数，**计算成本高**；且搜索是在给定数据集和网络结构上进行的，**难以迁移**。相比之下，通用近似器的参数是连续的、可以随网络一起训练，天然适配不同的数据集和结构。

### (2) 有理函数参数化

#### ⚪ PAU：Padé近似

- paper：[Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks](https://arxiv.org/abs/1907.06732)

给定任意函数$f(x)$，**Padé近似**是指使用给定阶数的有理分式$F(x)$对其进行近似。给定分子$P$和分母$Q$的阶数$m$和$n$：

$$ f(x) \approx F(x) = \frac{P(x)}{Q(x)} = \frac{\sum_{j=0}^{m}a_jx^j}{1+\sum_{k=1}^{n}b_kx^k}= \frac{a_0+a_1x+\cdots+a_mx^m}{1+b_1x+\cdots+b_nx^n} $$

通常**Padé近似**能够给出比**Taylor近似**更好的结果，且在**Taylor**级数不收敛的情况下仍然有效。然而有理分式的灵活性也带来风险：它可能拟合出**极点**（分母为$0$），导致训练和推理时的数值不稳定。作者因此使用一种**安全的Padé近似**，对分母的每一项取绝对值，保证分母不小于$1$：

$$ F(x) = \frac{\sum_{j=0}^{m}a_jx^j}{1+\sum_{k=1}^{n}|b_k||x|^k}= \frac{a_0+a_1x+\cdots+a_mx^m}{1+|b_1||x|+\cdots+|b_n||x|^n} $$

**PAU(Padé activation unit)**把这个有理分式作为激活函数，通过梯度下降从数据中学习系数。所需的梯度为：

$$
\begin{aligned}
\frac{\partial F}{\partial x} &= \frac{1}{Q(x)}\frac{\partial P(x)}{\partial x}-\frac{P(x)}{Q(x)^2}\frac{\partial Q(x)}{\partial x} \\
\frac{\partial F}{\partial a_j} &= \frac{x^j}{Q(x)}, \qquad \frac{\partial F}{\partial b_k} = -\frac{P(x)}{Q(x)^2}\cdot\frac{b_k}{|b_k|}|x|^k
\end{aligned}
$$

其中分子与分母的导数为：

$$
\begin{aligned}
\frac{\partial P(x)}{\partial x} &= a_1+2a_2x+\cdots+ma_mx^{m-1} \\
\frac{\partial Q(x)}{\partial x} &= \frac{x}{|x|}\left(|b_1|+2|b_2||x|+\cdots+n|b_n||x|^{n-1}\right)
\end{aligned}
$$

为了减少参数量，**每一层的所有神经元共享一套PAU参数**。初始化时用标准的**Padé近似**去拟合一个已知的激活函数，下图展示了**Padé近似**（虚线）对常用激活函数的拟合效果：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-037-pau.jpg)

#### ⚪ OPAU：正交Padé近似

- paper：[Orthogonal-Padé Activation Functions: Trainable Activation functions for smooth and faster convergence in deep networks](https://arxiv.org/abs/2106.09693)

**OPAU(orthogonal-Padé activation unit)**在**PAU**的基础上，把分子和分母中的幂基$1,x,x^2,\cdots$替换为**正交多项式基**$f_1,f_2,\cdots,f_l$：

$$ G(x) = \frac{\sum_{i=0}^{k}c_if_i(x)}{1+\sum_{j=1}^{l}|d_j||f_j(x)|}= \frac{c_0+c_1f_1(x)+\cdots+c_kf_k(x)}{1+|d_1||f_1(x)|+\cdots+|d_l||f_l(x)|} $$

同样对分母取绝对值以避免极点。作者考察了六种常用的正交多项式基：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-038-opau.jpg)

使用正交基的好处是数值条件更好、达到同等拟合精度所需的阶数更低，因此**运行时间更短**。实践中使用**LeakyReLU**初始化**OPAU**。

### (3) 样条参数化

**样条(spline)**是与分段线性函数、有理函数并列的第三种经典通用逼近器。把样条用作激活函数，会导出一个与前面所有方法都不同量级的改动：它改变的不是激活函数，而是整个网络的组织方式。

#### ⚪ KAN：把激活函数搬到边上

- paper：[KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

$$ \phi(x) = w_b\cdot \text{silu}(x) + w_s\cdot \sum_i c_i B_i(x) $$

其中$B_i$是定义在某个网格上的**B样条**基函数，系数$c_i$可学习，$w_b,w_s$是两个可学习的缩放。

**KAN**与前面所有方法的差别是**性质上的**：**MLP**把**固定的激活函数放在节点上**、把**可学习的权重放在边上**；**KAN**则反过来，把**可学习的激活函数放在边上**，节点只做求和。至此激活函数不再是网络的一个组件，激活函数本身就是网络。

式中两项的分工也值得注意：**silu**项是一条**基础通路**（类似残差连接），保证优化行为良好；样条项提供**局部表达能力**。样条基是**局部支撑**的：修改一个系数$c_i$只影响一个区间，这既是**KAN**可解释性的来源，也使得**网格扩展(grid extension)**成为可能：可以逐步加密样条网格来提升容量，而不必从头重训。

**KAN**的理论依据是**柯尔莫哥洛夫-阿诺德表示定理**。代价也要如实说明：**KAN**的参数量远大于同规模**MLP**，且单个参数的计算效率低得多（样条求值无法像矩阵乘法那样被硬件充分优化），因此目前主要用于**科学计算**与**符号回归**这类小规模、重可解释性的问题。详见[<font color=Blue>Kolmogorov-Arnold网络</font>](https://0809zheng.github.io/2024/05/06/kan.html)。

### (4) 非光滑激活函数的光滑化：一个统一视角

回顾一下：**ReLU**、**LeakyReLU**、**maxout**都可以写成**最大值函数**的形式，它们的问题都出在$\max$带来的**折点不可导**（准则1）。因此一个自然的思路是：**为最大值函数寻找参数化的光滑近似**。不同的近似方式会导出不同的激活函数；而**Softplus**、**Swish**、**GELU**这些看似出身各异的函数，其实都落在这个统一框架内。

下面三种方法分别从**softmax**、**绝对值函数**和**Dirac函数**出发，得到了三条不同的光滑化路径。

#### ⚪ ACON：最大值函数的softmax近似

- paper：[Activate or Not: Learning Customized Activation](https://arxiv.org/abs/2009.04759)

最大值函数$\max(x_1,\cdots,x_n)$的一个经典可微近似是$\alpha$-**softmax**：

$$ \max(x_1,x_2,\cdots,x_n) \approx \frac{\sum_{i=1}^{n}x_ie^{\beta x_i}}{\sum_{i=1}^{n}e^{\beta x_i}} $$

其中$\beta$是**开关因子**：$\beta \to +\infty$时上式趋近于最大值函数，$\beta =0$时上式退化为算术平均。

当$n=2$时可以化简出一个非常简洁的形式：

$$
\begin{aligned}
\max(x_1,x_2) &\approx \frac{x_1e^{\beta x_1}+x_2e^{\beta x_2}}{e^{\beta x_1}+e^{\beta x_2}} \\
&= x_1\frac{1}{1+e^{-\beta(x_1-x_2)}}+x_2\frac{1}{1+e^{-\beta(x_2-x_1)}} \\
&= x_1\sigma\left(\beta(x_1-x_2)\right)+x_2\left[1-\sigma\left(\beta(x_1-x_2)\right)\right] \\
&= (x_1-x_2)\sigma\left(\beta(x_1-x_2)\right)+x_2
\end{aligned}
$$

若自变量为任意函数$\eta_{a}(x)$和$\eta_{b}(x)$，则整个**maxout**函数族$\max\left(\eta_{a}(x),\eta_{b}(x)\right)$的光滑近似为：

$$ \max\left(\eta_{a}(x),\eta_{b}(x)\right) \approx \left(\eta_{a}(x)-\eta_{b}(x)\right)\sigma\left(\beta\left(\eta_{a}(x)-\eta_{b}(x)\right)\right)+\eta_{b}(x) $$

对$\eta_a,\eta_b$赋予不同形式，就得到一系列光滑激活函数，作者称之为**ACON(activate or not)**族：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-039-acon.jpg)

- **ACON-A**：令$\eta_{a}=x,\eta_{b}=0$，对应$\text{ReLU}=\max(x,0)$；光滑形式为$x\sigma(\beta x)$，**恰好就是Swish**（因此**Swish**可以看作**ReLU**的$\alpha$-**softmax**光滑近似）；
- **ACON-B**：令$\eta_{a}=x,\eta_{b}=px$，对应$\text{PReLU}=\max(x,px)$；光滑形式为$(1-p)x\sigma\left(\beta (1-p)x\right)+px$；
- **ACON-C**：令$\eta_{a}=p_1x,\eta_{b}=p_2x$，对应$\max(p_1x,p_2x)$；光滑形式为$(p_1-p_2)x\sigma\left(\beta (p_1-p_2)x\right)+p_2x$。

论文主要讨论**ACON-C**，其函数曲线及一阶导数曲线如下：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-040-acon.jpg)

**ACON-C**相比**Swish**的核心优势是**梯度界可学习**。其梯度为：

$$
\begin{aligned}
\text{ACON-C}'(x) = &\left(p_1-p_2\right)\sigma\left(\beta (p_1-p_2)x\right) \\
&+ \beta\left(p_1-p_2\right)^2x\,\sigma\left(\beta (p_1-p_2)x\right)\left[1-\sigma\left(\beta (p_1-p_2)x\right)\right]+p_2
\end{aligned}
$$

注意到两端的极限：

$$ \lim_{x \to +\infty}\text{ACON-C}'(x) = p_1, \quad \lim_{x \to -\infty}\text{ACON-C}'(x) = p_2 $$

对其求二阶导数并令其为$0$，可得导数的极值：

$$
\begin{aligned}
\max\left(\text{ACON-C}'(x)\right) &\approx 1.0998p_1-0.0998p_2 \\
\min\left(\text{ACON-C}'(x)\right) &\approx 1.0998p_2-0.0998p_1
\end{aligned}
$$

在**Swish**中（对应$p_1=1,p_2=0$），梯度的上下界固定为$1.0998$和$-0.0998$，超参数$\beta$只决定梯度趋近上下界的速度；而**ACON-C**的梯度界由可学习的$p_1,p_2$决定。

开关因子$\beta$控制激活函数的非线性程度，也即“**是否激活**”：$\beta \to +\infty$时**ACON-C**趋近于$\max(p_1x,p_2x)$，$\beta =0$时退化为线性函数$(p_1-p_2)x/2$（此时该神经元实际上是“未被激活”的）。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-041-acon.jpg)

作者进一步提出**meta-ACON**，即把$\beta$表示为输入样本的函数$\beta=G(x)$，从而让网络**为每个样本自适应地决定每个神经元是否被激活**。对图像数据$x \in \mathbb{R}^{C\times H\times W}$，$\beta$可以有三种粒度：

$$
\begin{aligned}
\text{layer-wise}: \quad & \beta = \sigma\left(\sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{c,h,w}\right) \\
\text{channel-wise}: \quad & \beta_c = \sigma\left(W_1W_2 \sum_{h=1}^{H}\sum_{w=1}^{W}x_{c,h,w}\right) \\
\text{pixel-wise}: \quad & \beta_{c,h,w} = \sigma\left(x_{c,h,w}\right)
\end{aligned}
$$

作者选用**channel-wise**版本（形式上与**SE**模块一致）。注意**meta-ACON**已经跨入了3.6节“上下文相关激活函数”的范畴。

#### ⚪ SMU：最大值函数的绝对值近似

- paper：[SMU: smooth activation function for deep networks using smoothing maximum technique](https://arxiv.org/abs/2111.04682)

**SMU(smooth maximum unit)**从另一个方向出发。最大值函数可以精确地写成：

$$ \max(x_1,x_2) = \frac{x_1+x_2+|x_1-x_2|}{2} $$

因此**只需光滑化绝对值函数$|x|$**。$|x|$常用的两种光滑近似是$x\,\text{erf}(\mu x)$和$\sqrt{x^2+\mu^2}$：前者从下方逼近$|x|$（$\mu$越大越逼近），后者从上方逼近（$\mu$越小越逼近）。代入得到最大值函数的两种近似：

$$
\begin{aligned}
f_1(x_1,x_2;\mu) &= \frac{x_1+x_2+(x_1-x_2)\,\text{erf}\left(\mu (x_1-x_2)\right)}{2} \\
f_2(x_1,x_2;\mu) &= \frac{x_1+x_2+\sqrt{(x_1-x_2)^2+\mu^2}}{2}
\end{aligned}
$$

由此可以直接写出各类基于最大值的激活函数的光滑近似：

$$
\begin{aligned}
\text{maxout}: \quad & f_1(ax,bx;\mu) = \frac{(a+b)x+(a-b)x\,\text{erf}\left(\mu (a-b)x\right)}{2} \\
\text{ReLU}: \quad & f_1(x,0;\mu) = \frac{x+x\,\text{erf}(\mu x)}{2} \\
\text{LeakyReLU}: \quad & f_1(x,\alpha x;\mu) = \frac{(1+\alpha)x+(1-\alpha)x\,\text{erf}\left(\mu (1-\alpha)x\right)}{2}
\end{aligned}
$$

其中**ReLU**的近似在$\mu=1/\sqrt{2}$时**恰好就是GELU**。**这解释了GELU的另一重身份**：它既是随机正则化的期望（见3.2节），也是**ReLU**的$\text{erf}$型光滑近似。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-042-smu.jpg)

作者取**LeakyReLU**的光滑近似作为**SMU**，其中$\alpha$和$\mu$都通过梯度下降学习：

$$
\begin{aligned}
\frac{\partial f_1}{\partial \alpha} &= \frac{1}{2}\left[x-x\,\text{erf}\left(\mu (1-\alpha)x\right)-\frac{2\mu(1-\alpha) x^2}{\sqrt{\pi}}e^{-\left(\mu (1-\alpha)x\right)^2}\right] \\
\frac{\partial f_1}{\partial \mu} &= \frac{(1-\alpha)^2x^2}{\sqrt{\pi}}e^{-\left(\mu (1-\alpha)x\right)^2}
\end{aligned}
$$

其中用到$\text{erf}(x)=\frac{2}{\sqrt{\pi}}\int_{0}^{x}e^{-t^2}dt$及$\text{erf}'(x)=\frac{2}{\sqrt{\pi}}e^{-x^2}$。使用$\sqrt{x^2+\mu^2}$近似的版本称为**SMU-1**。实验中设$\alpha=0.25$，$\mu$为可训练参数。

#### ⚪ SAU：使用Dirac函数构造光滑近似

- paper：[SAU: Smooth activation function using convolution with approximate identities](https://arxiv.org/abs/2109.13210)

**SAU(smooth activation unit)**给出了最一般的光滑化框架。**Dirac**函数（单位冲激函数）定义为：

$$ \delta(x) = \begin{cases} +\infty, & x =0 \\ 0, & x\neq 0 \end{cases}, \quad \int_{-\infty}^{+\infty}\delta(x)dx = 1 $$

它可以看作一个仅在$x=0$处取值的概率密度函数。**Dirac**函数本身没有明显意义，但作用于其他函数时具有重要性质：

$$
\begin{aligned}
\int_{-\infty}^{+\infty}f(x)\delta(x)dx &= f(0), & \int_{-\infty}^{+\infty}f(y)\delta(x-y)dy &= f(x) \\
\int_{-\infty}^{+\infty}f(x)\delta'(x)dx &= -f'(0), & \int_{-\infty}^{+\infty}f(x)\delta^{(n)}(x)dx &= (-1)^{n}f^{(n)}(0)
\end{aligned}
$$

其中第一行右侧的**卷积**形式是构造光滑近似的关键。由于**Dirac**函数没有显式表达式，通常采用一些连续函数作为它的光滑近似（文中称**approximate identity**）。一种思路是构造“钟形曲线”并让其宽度趋于$0$而积分保持为$1$：

$$
\begin{aligned}
\text{近似①（Gauss核）}: \quad & \delta(x) = \lim_{s \to 0} \frac{1}{\sqrt{2\pi}s}e^{-\frac{x^2}{2s^2}} \\
\text{近似②（Cauchy核）}: \quad & \delta(x) = \frac{1}{\pi}\lim_{a \to 0} \frac{a}{x^2+a^2}
\end{aligned}
$$

另一种思路是注意到**Dirac**函数的积分是单位阶跃函数$\theta(x)$，因此$\theta(x)$的光滑近似（**S**型曲线）的导数即为**Dirac**函数的光滑近似：

$$ \text{近似③（Sigmoid导数）}: \quad \delta(x) = \lim_{t \to +\infty} \frac{d}{dx}\sigma(tx) = \lim_{t \to +\infty} \frac{te^{tx}}{\left(1+e^{tx}\right)^2} $$

于是，对任意具有可数个间断点的函数$f$，只要找到**Dirac**函数的光滑近似$\phi \approx \delta$，就可以构造$f$的光滑近似：

$$ g(x) \approx \int_{-\infty}^{+\infty}f(y)\phi(x-y)dy = (f* \phi)(x) $$

即**光滑化 $=$ 与一个近似单位元做卷积**。这个框架的威力在于：**替换$f$和$\phi$就能导出各种已知的激活函数**。

**① 光滑化ReLU（取$\phi$为近似③）**：

$$
\begin{aligned}
\max(x,0) &\approx \int_{0}^{+\infty}\frac{yte^{t(x-y)}}{\left(1+e^{t(x-y)}\right)^2}dy = \int_{0}^{+\infty}y\,d\left(\frac{1}{1+e^{t(x-y)}}\right) \\
&= \left.\frac{y}{1+e^{t(x-y)}}\right|_{0}^{+\infty}-\int_{0}^{+\infty}\frac{dy}{1+e^{t(x-y)}} = \frac{\ln\left(1+e^{tx}\right)}{t}
\end{aligned}
$$

当$t=1$时，上式**恰好就是Softplus**。

**② 光滑化阶跃函数（把ReLU写成$x\theta(x)$）**：若直接取$\theta(x)\approx\sigma(x)$，则$f(x)=x\sigma(x)$，**恰好就是Swish**；若用近似①作为$\theta(x)$的光滑近似，则

$$ \max(x,0) = x\theta(x) \approx x \int_{0}^{+\infty}\frac{1}{\sqrt{2\pi}s}e^{-\frac{(x-y)^2}{2s^2}}dy = \frac{x}{2}\left[1+\text{erf}\left(\frac{x}{\sqrt{2}s}\right)\right] $$

当$s=1$时，上式**恰好就是GELU**。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-043-sau.jpg)

**③ 光滑化LeakyReLU（取$\phi$为近似①）**，这就是**SAU**本身：

$$ \max(x,\alpha x) \approx \frac{\alpha}{\sqrt{2\pi}s} \int_{-\infty}^{0} ye^{-\frac{(x-y)^2}{2s^2}}dy + \frac{1}{\sqrt{2\pi}s} \int_{0}^{+\infty} ye^{-\frac{(x-y)^2}{2s^2}}dy $$

利用不定积分

$$
\begin{aligned}
\int ye^{-\frac{(x-y)^2}{2s^2}}dy &= \int (y-x)e^{-\frac{(y-x)^2}{2s^2}}dy + x\int e^{-\frac{(y-x)^2}{2s^2}}dy \\
&= -s^2 e^{-\frac{(y-x)^2}{2s^2}} + \frac{\sqrt{2\pi}sx}{2} \text{erf}\left(\frac{y-x}{\sqrt{2}s}\right)
\end{aligned}
$$

代入并整理可得：

$$ \text{SAU}(x) = \frac{(1-\alpha)s}{\sqrt{2\pi}} e^{-\frac{x^2}{2s^2}}+ \frac{x}{2} + \frac{(1-\alpha) x}{2}\text{erf}\left(\frac{x}{\sqrt{2}s}\right) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-044-sau.jpg)

其关于$x$和$\alpha$的梯度为：

$$
\begin{aligned}
\frac{\partial \text{SAU}(x)}{\partial x} &= \frac{1}{2} + \frac{(1-\alpha)}{2}\text{erf}\left(\frac{x}{\sqrt{2}s}\right)-\frac{x(1-\alpha)}{\sqrt{2\pi}s} e^{-\frac{x^2}{2s^2}}+ \frac{(1-\alpha) x}{\sqrt{2\pi}s}e^{-\frac{x^2}{2s^2}} \\
\frac{\partial \text{SAU}(x)}{\partial \alpha} &= -\frac{s}{\sqrt{2\pi}} e^{-\frac{x^2}{2s^2}}- \frac{x}{2}\text{erf}\left(\frac{x}{\sqrt{2}s}\right)
\end{aligned}
$$

实验中$\alpha$初值取$0.15$并通过梯度更新，$s$固定为$5\times 10^{-5}$。

#### ⭐ 讨论：光滑化视角下的激活函数家族

三条光滑化路径殊途同归地把**Softplus、Swish、GELU、SMU、SAU、ACON**串成了一个家族。它们的差异只在于“光滑化谁”和“用什么核去光滑化”：

| 被光滑化的函数 | 光滑化方式 | 得到的激活函数 |
| ---- | ---- | ---- |
| ReLU | 与$\sigma'(tx)$卷积（$t=1$） | Softplus |
| ReLU | 阶跃函数替换为Sigmoid | Swish / ACON-A |
| ReLU | 阶跃函数与Gauss核卷积（$s=1$） | GELU |
| ReLU | $\|x\|\approx x\,\text{erf}(\mu x)$，$\mu=1/\sqrt{2}$ | GELU |
| PReLU | $\alpha$-softmax近似 | ACON-B |
| LeakyReLU | 与Gauss核卷积 | SAU |
| LeakyReLU | $\|x\|\approx x\,\text{erf}(\mu x)$ | SMU |
| maxout | $\alpha$-softmax近似 | ACON-C |

### (5) 统一参数族

(4)中的光滑化视角在**概念上**统一了一批激活函数；另一条路线更彻底：把多个已知激活函数**字面地**写进同一个公式里，让网络通过少数几个可学习参数在它们之间**插值**，而不是由设计者事先挑定一个。

#### ⚪ AGLU：统一的门控激活

- paper：[Adaptive Parametric Activation](https://arxiv.org/abs/2407.08567)

作者首先把**Sigmoid**推广为**自适应参数化Sigmoid**：

$$ \eta(z;\kappa,\lambda) = \left(\lambda e^{-\kappa z}+1\right)^{-1/\lambda} $$

再套进自门控模板得到**AGLU(adaptive gated linear unit)**：

$$ \text{AGLU}(z) = z\cdot \eta(z;\kappa,\lambda) $$

这个两参数族包含了三个熟悉的特例：$\lambda=\kappa=1$时$\eta$退化为普通的**Sigmoid**；$\kappa\to\infty$时退化为**ReLU**；而在**AGLU**的形式下，$\lambda=\kappa=1$给出**SiLU/Swish**。也就是说，两个可学习标量就跨越了**Sigmoid**、**ReLU**、**SiLU**三者。

两个参数的分工是清晰的：$\kappa$控制过渡区的**陡峭程度**，$\lambda$控制曲线的**非对称性**。论文的动机来自**长尾/不平衡识别**：分类器上适合头部类别的激活形状与适合尾部类别的并不相同，因此让每个类别（或每个通道）各自学一组$(\kappa,\lambda)$能带来明显收益。且开销极小（每通道两个标量），该函数已集成进**Ultralytics YOLO**。

## 3.6 上下文相关的激活函数

前面所有激活函数（除**maxout**）都是**逐元素**的标量函数：网络中每个激活值的变换只依赖它自己。这构成了一个信息瓶颈：激活函数完全无法利用**上下文**。

本节的激活函数打破这一限制：它们是**多输入**函数，输出依赖于一组相关的输入。上下文的来源可以是**全部特征**（**Dynamic ReLU**）、**相邻通道**（**Dynamic Shift-Max**）或**空间邻域**（**FReLU**）。

注意这类方法与注意力机制（尤其是**SE**模块）的边界是模糊的；它们通常都用“全局池化$+$两层全连接”来生成参数。区别在于：注意力机制生成的是**乘性权重**，而这里生成的是**激活函数的形状参数**。

### ⚪ Dynamic ReLU

- paper：[Dynamic ReLU](https://arxiv.org/abs/2003.10027)

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-045-dynamic-relu.jpg)

**DY-ReLU**把所有输入元素$x$的**全局上下文信息**编码到一个**超函数(hyper function)** $\theta(x)$中，并用它决定激活函数$f_{\theta(x)}(x)$的形状。

**ReLU**可以推广为分段线性函数$y_c=\max_k\left(a_c^kx_c+b_c^k\right)$；**DY-ReLU**让其中的斜率和截距由输入自适应地计算：

$$ y_c = f_{\theta(x)}(x_c) = \mathop{\max}_{1\leq k \leq K} \left\{a_c^k(x)\,x_c+b_c^k(x)\right\} $$

超函数$\theta(x)$输出全部$2KC$个参数：

$$ \left[a_1^1,\cdots,a_C^1,\cdots,a_1^K,\cdots,a_C^K,b_1^1,\cdots,b_C^1,\cdots,b_1^K,\cdots,b_C^K\right]^\top=\theta(x) $$

$\theta(x)$的实现参考了**SENet**的**SE**模块：先沿空间维度全局平均池化，再通过两个全连接层输出$2KC$个残差量$\Delta a,\Delta b$，最后与固定初值相加：

$$ a_c^k(x)=\alpha^k+\lambda_a \Delta a_{c}^{k}(x), \quad b_c^k(x)=\beta^k+\lambda_b \Delta b_{c}^{k}(x) $$

实验中取$K=2$，$\alpha^1=1$，$\alpha^2=\beta^1=\beta^2=0$，即**初始化为ReLU**。

**DY-ReLU**是一个相当强的统一框架，许多已有的激活函数都是它的特例：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-046-dynamic-relu.jpg)

根据超函数作用粒度的不同，作者设计了三种形式：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-047-dynamic-relu.jpg)

- **DY-ReLU-A**：每一层的所有神经元共享激活函数，引入$2K$个参数；
- **DY-ReLU-B**：每层的每个通道共享激活函数，引入$2KC$个参数；
- **DY-ReLU-C**：每层的每个元素独立使用激活函数。若直接实现需要$2KCHW$个参数；作者把通道参数和空间参数分别计算再相乘，从而把参数量降低为$2KC+HW$。

### ⚪ Dynamic Shift-Max

- paper：[MicroNet: Towards Image Recognition with Extremely Low FLOPs](https://arxiv.org/abs/2011.12289)

**Dynamic Shift-Max**（由**MicroNet**提出）的动机是：当网络层数减少时性能会下降，而**改善每一层的非线性可以补偿网络深度的减少**。它引入的上下文是**相邻通道组**。

设输入向量$x$具有$C$个通道，划分为$G$组，每组$C/G$个通道。定义第$i$个通道的$N$通道循环移位，以及**组循环移位**（即取每个组的对应通道位置）：

$$
\begin{aligned}
x_N(i) &= x_{(i+N) \bmod C} \\
x_{C/G}(i,j) &= x_{\left(i+jC/G\right) \bmod C}, \quad j=0,1,\cdots,G-1
\end{aligned}
$$

**Dynamic Shift-Max**对多个组移位的加权和取最大值：

$$ y_i= \mathop{\max}_{1\leq k\leq K} \left\{\sum_{j=0}^{J-1} a_{i,j}^k(x)\,x_{C/G}(i,j)\right\} $$

其中系数$a_{i,j}^k(x)$同样通过类似**SE**的方式由输入计算。下图展示了只关注该组与下一组的特殊情况：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-048-dynamic-shift-max.jpg)

**Dynamic Shift-Max**共引入$CJK$个参数（实践中取$J=K=2$）。它的一个附带好处是：由于跨组混合了通道，它**提高了不同通道组之间的连通性**，可以作为逐点卷积的补充；这在极端轻量化的网络中尤其有价值。

### ⚪ FReLU：空间邻域上下文

- paper：[Funnel Activation for Visual Recognition](https://arxiv.org/abs/2007.11824)

$$ \text{FReLU}(x) = \max\left(x, T(x)\right) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-049-frelu.jpg)

**FReLU(funnel activation)**引入的上下文是**空间邻域**，其设计逻辑是一条清晰的推广链条：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-050-frelu.jpg)

- **ReLU** $=\max(x,0)$：与$0$比较；
- **PReLU** $=\max(x,px)$：引入**参数条件(parametric condition)**；
- **FReLU** $=\max\left(x,T(x)\right)$：引入**漏斗条件(funnel condition)**，其中$T(\cdot)$是一个可学习的$3\times 3$**深度卷积**（后接**BatchNorm**）。

这个改动看起来很小，但它使激活函数具备了**空间建模能力**。通常的卷积神经网络通过加深网络线性地增加感受野，且感受野形状始终是矩形（下图a）；引入**FReLU**后，每个位置会根据$\max$的结果隐式地在$1 \times 1$与$3 \times 3$感受野之间选择，使得整个网络的有效感受野不再是规则的矩形，从而能够适配物体的不规则形状（下图b、c）：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-activation-051-frelu.jpg)

**FReLU**因此特别适合视觉任务（分类、检测、分割），额外开销仅为一个深度卷积。

## 3.7 门控激活函数

门控激活函数与前面各族的思路都不同：它不是设计一个更好的一元函数，而是**改变前馈层的结构**；把一路线性变换的输出作为**门**去调制另一路线性变换的输出。

### ⚪ GLU 及其变体

- paper：[Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
- paper：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

**GLU(gated linear unit)**的基本形式为：

$$ \text{GLU}(x) = \sigma(xW + b) \otimes (xV + c) $$

其中$\otimes$表示逐元素相乘。**GLU**用一路的**Sigmoid**输出作为门控信号，控制另一路信息的通过量。相比逐元素激活函数，**GLU**引入了**乘性交互**，其表达能力更强，且梯度可以通过线性支路无衰减地传播。

把$\sigma$替换为其他激活函数，可以得到一系列变体（此外还有省略激活函数的**Bilinear**形式）：

$$
\begin{aligned}
\text{ReGLU}(x) &= \text{ReLU}(xW + b) \otimes (xV + c) \\
\text{GEGLU}(x) &= \text{GELU}(xW + b) \otimes (xV + c) \\
\text{SwiGLU}(x) &= \text{Swish}(xW + b) \otimes (xV + c) \\
\text{Bilinear}(x) &= (xW + b) \otimes (xV + c)
\end{aligned}
$$

在**Transformer**中，把传统的前馈层(**FFN**)替换为**GLU**族即得到：

$$
\begin{aligned}
\text{FFN}(x) &= \text{ReLU}(xW_1)W_2 \\
\text{FFN}_{\text{SwiGLU}}(x) &= \left(\text{Swish}(xW) \otimes xV\right)W_2
\end{aligned}
$$

注意这里有一个重要的工程细节：**GLU**族的**FFN**需要三个权重矩阵($W,V,W_2$)而非两个，因此为了保持参数量和计算量与原始模型一致，需要把隐藏维度$d_{ff}$**减少为原来的$2/3$**（原论文中直接减半）。

在**T5**的预训练与下游微调实验中，**GEGLU**和**SwiGLU**取得了最低的困惑度和最好的下游任务表现。这一结论影响深远：**SwiGLU**已成为当前主流大语言模型（**PaLM**、**LLaMA**系列等）前馈层的标准配置。

### ⚪ dReLU：为稀疏推理设计的门控

- paper：[Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters](https://arxiv.org/abs/2406.05955)

$$ \text{FFN}_{\text{dReLU}}(x) = \left(\text{ReLU}(xW_{\text{gate}}) \otimes \text{ReLU}(xW_{\text{up}})\right)W_{\text{down}} $$

**SwiGLU**虽然效果最好，却对**稀疏推理**极不友好：**Swish**永远不会精确等于$0$（只是趋近于$0$），因此前馈层中几乎**每一个神经元都是“激活”的**，推理时无法跳过任何计算。

**dReLU(dual ReLU)**的改动很简单：**对两路都施加ReLU**。这样一个神经元的输出只有在**两路同时为正**时才非零。稀疏度的提升是相乘的：若两路各有约一半概率为正且相关性不强，联合激活率会急剧下降，远低于只对一路施加**ReLU**的情形。论文报告在保持模型质量的前提下前馈层稀疏度可达约$90\%$，这直接转化为**CPU**和显存受限设备上的推理加速。

代价是：**dReLU**必须在预训练阶段（或大规模持续预训练中）引入，不能对已有的**SwiGLU**模型直接替换。

### ⚪ 扩展门控范围：xATLU、xGELU、xSiLU

- paper：[Expanded Gating Ranges Improve Activation Functions](https://arxiv.org/abs/2405.20768)

作者首先用$\arctan$构造了一个新的门控函数：

$$ \text{ATLU}(x) = x\cdot \frac{\arctan(x) + \pi/2}{\pi} $$

更有价值的是文中提出的**通用扩展技巧**：对任意值域为$(0,1)$的门控函数$g(x)$，定义

$$ x\left[g(x)(1+2\alpha) - \alpha\right], \quad \alpha \text{ 可学习，初始化为 } 0 $$

这个线性重映射把门控的值域从$(0,1)$拉伸为$(-\alpha, 1+\alpha)$。于是门控获得了两项新能力：可以取**负值**（不只是抑制输入，还能翻转输入的符号），也可以**大于1**（放大输入）。

$\alpha$初始化为$0$意味着训练从**未修改的原始激活函数**出发，因此这是一个严格安全的推广：最差情况下$\alpha$保持在$0$附近，退化为原函数。把该技巧分别作用于$\arctan$、**GELU**、**SiLU**的门控，即得到**xATLU**、**xGELU**、**xSiLU**。

### ⚪ PowLU：一个非常新的幂型门控

- paper：[PowLU: An Activation Function for Stable Pre-Training of LLMs](https://arxiv.org/abs/2605.25704)

$$ \text{PowLU}(x) =\begin{cases} x\cdot x^{\frac{m}{\sqrt{x}+1}}\cdot \sigma(x), & x> 0 \\ x^2\cdot \sigma(x), & x\leq 0 \end{cases}, \quad 0<m<10 $$

**PowLU**把3.2(5)节的幂型思路与自门控结合起来。正半轴的指数$\frac{m}{\sqrt{x}+1}$随$x$增大从$m$**衰减到$0$**，因此函数在原点附近是**超线性**增长的，在远处则趋于近似线性；相当于在幂型激活与普通自门控之间做插值。

有一点必须明确指出（这也是它与本文其他所有激活函数的显著差异）：负半轴的$x^2\sigma(x)$是**正的**而非负的，形成一个小的正向凸包，峰值约为$0.48$、位置在$x\approx -2.4$附近。这个行为相当反常，不过函数在原点处仍然连续且$C^1$。


# 4. 速查表与选型建议

## (1) 速查表

- Reference：[Pytorch中的激活函数层](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

<style>
table th:first-of-type {
    width: 30%;
}
table th:nth-of-type(2) {
    width: 70%;
}
</style>


| 激活函数 | 表达式 |
| ---- | ---- |
| Step | $$\begin{cases} 1, & x\geq 0 \\ 0, & x<0 \end{cases}$$ |
| Sigmoid | $$\sigma(x)=\frac{1}{1+e^{-x}}$$ |
| HardSigmoid：降低Sigmoid计算量 | $$\begin{cases} 1, & x\geq 1 \\ (x+1)/2, & -1<x<1 \\ 0, & x\leq -1 \end{cases}$$ |
| Tanh | $$\begin{aligned} &2\sigma(2x)-1 \\ &=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} \end{aligned}$$ |
| HardTanh：降低Tanh计算量 | $$\begin{cases} 1, & x>1 \\ x, & -1\leq x\leq 1 \\ -1, & x<-1 \end{cases}$$ |
| ISRU：使用逆平方根近似Tanh | $$\frac{x}{\sqrt{1 + \alpha x^2}}$$ |
| Softsign：多项式饱和的Tanh替代 | $$\frac{x}{1+\|x\|}$$ |
| Softplus：ReLU的光滑近似 | $$\begin{aligned} &\int_{-\infty}^{x}\sigma(t)dt \\ &=\ln\left(1+e^x\right) \end{aligned}$$ |
| Squareplus：Softplus的代数近似 | $$\frac{1}{2}\left(x+\sqrt{x^2+b}\right)$$ |
| ReLU | $$\begin{aligned} &\max(x,0) \\ &=\begin{cases} x, & x\geq 0 \\ 0, & x<0 \end{cases} \end{aligned}$$ |
| ReLU6：适配低精度部署 | $$\begin{aligned} &\min\left(\max(x,0),6\right) \\ &=\begin{cases} 6, & x\geq 6 \\ x, & 0\leq x<6 \\ 0, & x<0 \end{cases} \end{aligned}$$ |
| LeakyReLU：解决dead ReLU | $$\begin{aligned} &\max(x,0.01x) \\ &=\begin{cases} x, & x\geq 0 \\ 0.01x, & x<0 \end{cases} \end{aligned}$$ |
| PReLU：可学习参数$\alpha$ | $$\begin{aligned} &\max(x,\alpha x) \\ &=\begin{cases} x, & x\geq 0 \\ \alpha x, & x<0 \end{cases} \end{aligned}$$ |
| RReLU：均匀分布采样$\alpha$ | $$\begin{aligned} &\max(x,\alpha x) \\ &=\begin{cases} x, & x\geq 0 \\ \alpha x, & x<0 \end{cases} \end{aligned}$$ |
| CReLU：拼接正负两路整流 | $$\left[\text{ReLU}(x), \text{ReLU}(-x)\right]$$ |
| SReLU：双侧可学习阈值 | $$\begin{cases} a^r\left(x-t^r\right)+t^r, & x\geq t^r \\ x, & t^l < x < t^r \\ a^l\left(x-t^l\right)+t^l, & x\leq t^l \end{cases}$$ |
| SUGAR：前向ReLU、反向B-SiLU梯度 | $$\text{B-SiLU}(x)=(x+\alpha)\sigma(x)-\frac{\alpha}{2}$$ |
| ELU：解决bias shift | $$\begin{cases} x, & x\geq 0 \\ \alpha\left(e^x-1\right), & x<0 \end{cases}$$ |
| CELU：连续可微的ELU | $$\begin{cases} x, & x\geq 0 \\ \alpha\left(e^{x/\alpha}-1\right), & x<0 \end{cases}$$ |
| SELU：自标准化的ELU | $$\begin{cases} \lambda x, & x\geq 0 \\ \lambda\alpha\left(e^x-1\right), & x<0 \end{cases}$$ |
| ISRLU：使用逆平方根近似ELU | $$\begin{cases} x, & x\geq 0 \\ \frac{x}{\sqrt{1 + \alpha x^2}}, & x<0 \end{cases}$$ |
| PELU：参数化的ELU | $$\begin{cases} \frac{a}{b}x, & x\geq 0 \\ a\left(e^{x/b}-1\right), & x<0 \end{cases}$$ |
| GELU：随机正则化视角 | $$\begin{aligned} &x\Phi(x)=x\int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt \\ &= \frac{x}{2}\left(1+\text{erf}\left(\frac{x}{\sqrt{2}}\right)\right) \end{aligned}$$ |
| Squared ReLU：搜索出的平方整流 | $$\left(\text{ReLU}(x)\right)^2=\max(x,0)^2$$ |
| StarReLU：标准化的平方整流 | $$\begin{aligned} &s\cdot\left(\text{ReLU}(x)\right)^2+b \\ &s\approx 0.8944,\ b\approx -0.4472 \end{aligned}$$ |
| xIELU：正半轴二次、负半轴指数 | $$\begin{cases} \alpha_p x^2 + 0.5x, & x> 0 \\ \alpha_n\left(e^x-1\right)-\alpha_n x + 0.5x, & x\leq 0 \end{cases}$$ |
| PolyCom：多项式组合（PolyReLU形式） | $$\sum_{i=0}^{r} a_i \left(\text{ReLU}(x)\right)^i$$ |
| Swish：强化学习搜索 | $$\begin{aligned} &x\cdot \sigma(\beta x) \\ &= \frac{x}{1+e^{-\beta x}} \end{aligned}$$ |
| HardSwish：降低Swish计算量 | $$\begin{aligned} &x \cdot \frac{\text{ReLU6}(x+3)}{6} \\ &= \begin{cases} x, & x \geq 3 \\ \frac{x(x+3)}{6}, & -3 \leq x <3 \\ 0, & x < -3 \end{cases} \end{aligned}$$ |
| Mish：进一步搜索Swish | $$\begin{aligned} &x\cdot \tanh\left(\text{Softplus}(x)\right) \\ &=x\cdot \tanh\left(\ln\left(1+e^x\right)\right) \end{aligned}$$ |
| ELiSH：遗传算法搜索 | $$\begin{aligned} &\sigma(x) \cdot \text{ELU}(x) \\ &= \begin{cases} \frac{x}{1+e^{-x}}, & x\geq 0 \\ \frac{e^x-1}{1+e^{-x}}, & x<0 \end{cases} \end{aligned}$$ |
| HardELiSH：降低ELiSH计算量 | $$\begin{aligned} &\text{HardSigmoid}(x) \cdot \text{ELU}(x) \\ &= \begin{cases} x, & x\geq 1 \\ x(x+1)/2, & 0 \leq x<1 \\ \left(e^x-1\right)(x+1)/2, & -1\leq x<0 \\ 0, & x\leq -1 \end{cases} \end{aligned}$$ |
| E-Swish：缩放Swish | $$\beta x\sigma(x), \quad \beta \in [1,2]$$ |
| LiSHT：偶函数型自门控 | $$x\tanh(x)$$ |
| TanhExp：面向轻量级网络 | $$x\tanh\left(e^x\right)$$ |
| GELUSine：LLM搜索出的激活函数 | $$\text{GELU}(x)+0.1\sin(x)$$ |
| SIREN：正弦激活 | $$\sin\left(\omega_0\left(Wx+b\right)\right)$$ |
| Snake：单调项加周期项 | $$x+\frac{1}{a}\sin^2(ax)$$ |
| GCU：余弦门控 | $$x\cos(x)$$ |
| Maxout：分段线性单元 |$$\mathop{\max}_{j\in [1,k]}\left(x^\top W_{i,j}+b_{i,j}\right)$$ |
| APL：通过ReLU构造分段线性 | $$\begin{aligned} &\max(0,x) \\ &+\sum_{s=1}^{S}a^s\max\left(0,-x+b^s\right) \end{aligned}$$ |
| PWLU：直接参数化分段线性 | $$\begin{cases} \left(x-B_L\right)K_L+Y^0, & x<B_L \\ \left(x-B_R\right)K_R+Y^N, & x\geq B_R \\ \left(x-B_i\right)K_i+Y^i, & \text{其他} \end{cases}$$ |
| PAU：Padé近似 | $$\frac{a_0+a_1x+\cdots+a_mx^m}{1+\|b_1\|\|x\|+\cdots+\|b_n\|\|x\|^n}$$ |
| OPAU：正交Padé近似 | $$\frac{c_0+c_1f_1(x)+\cdots+c_kf_k(x)}{1+\|d_1\|\|f_1(x)\|+\cdots+\|d_l\|\|f_l(x)\|}$$ |
| KAN：B样条参数化（作用在边上） | $$w_b\cdot \text{silu}(x)+w_s\sum_i c_i B_i(x)$$ |
| ACON：最大值函数的softmax近似 | $$\left(p_1-p_2\right)x\sigma\left(\beta (p_1-p_2)x\right)+p_2x$$ |
| SMU：最大值函数的绝对值近似 | $$\frac{(1+\alpha)x+(1-\alpha)x\,\text{erf}\left(\mu (1-\alpha)x\right)}{2}$$ |
| SAU：使用Dirac函数近似 | $$\frac{(1-\alpha)s}{\sqrt{2\pi}} e^{-\frac{x^2}{2s^2}}+ \frac{x}{2} + \frac{(1-\alpha) x}{2}\text{erf}\left(\frac{x}{\sqrt{2}s}\right)$$ |
| AGLU：统一Sigmoid/ReLU/SiLU的参数族 | $$x\left(\lambda e^{-\kappa x}+1\right)^{-1/\lambda}$$ |
| Dynamic ReLU：全局上下文 | $$\mathop{\max}_{1\leq k \leq K} \left\{a_c^k(x)\,x_c+b_c^k(x)\right\}$$ |
| Dynamic Shift-Max：循环移位多输入 | $$\mathop{\max}_{1\leq k\leq K} \left\{\sum_{j=0}^{J-1} a_{i,j}^k(x)\,x_{C/G}(i,j)\right\}$$ |
| FReLU：卷积窗口输入 | $$\max\left(x,T(x)\right)$$ |
| GLU：门控线性单元 | $$\sigma(xW+b) \otimes (xV+c)$$ |
| ReGLU：使用ReLU进行门控 | $$\text{ReLU}(xW+b) \otimes (xV+c)$$ |
| GEGLU：使用GELU进行门控 | $$\text{GELU}(xW+b) \otimes (xV+c)$$ |
| SwiGLU：使用Swish进行门控 | $$\text{Swish}(xW+b) \otimes (xV+c)$$ |
| dReLU：两路都用ReLU（稀疏推理） | $$\text{ReLU}(xW_{\text{gate}}) \otimes \text{ReLU}(xW_{\text{up}})$$ |
| xATLU / xGELU / xSiLU：扩展门控范围 | $$x\left[g(x)(1+2\alpha)-\alpha\right]$$ |
| PowLU：幂型门控 | $$\begin{cases} x\cdot x^{\frac{m}{\sqrt{x}+1}}\sigma(x), & x> 0 \\ x^2\sigma(x), & x\leq 0 \end{cases}$$ |

## (2) 选型建议

尽管激活函数的研究已经产生了数百种方案，实践中的选择其实相当收敛。以下建议按场景给出：
- **卷积网络（默认）**：从**ReLU**开始。它计算最快、最稳定，且与**BatchNorm**配合良好（**BatchNorm**已经解决了偏置偏移问题，这削弱了**ELU**族的优势）。若追求更高精度且能接受额外开销，可尝试**Swish/SiLU**或**Mish**；它们在较深的网络上通常有稳定的小幅提升。
- **Transformer（视觉/通用）**：**GELU**是事实标准。若在意激活函数本身的计算开销，**StarReLU**是一个直接的替换（比**GELU**便宜得多，且已标准化到零均值单位方差）；**Squared ReLU**同样可用，但输出分布未标准化。
- **大语言模型前馈层**：默认**SwiGLU**（注意把隐藏维度调整为$2d_{ff}/3$以保持参数量）。若推理侧的**稀疏性**是首要目标，考虑**dReLU**或**ReLU**$^2$；但注意这类改动必须在**预训练阶段**引入，不能对已训练好的**SwiGLU**模型直接替换。**Squared ReLU**、**xIELU**、**PolyNorm**代表了这个方向当前的前沿。
- **移动端与量化部署**：使用**ReLU6**、**HardSwish**、**HardSigmoid**。可以参考**MobileNet V3**的做法：浅层用**ReLU**，深层用**HardSwish**。若目标平台的指数运算很慢但平方根较快，**ISRLU/Squareplus**是值得考虑的代数替代品。
- **循环网络**：门控单元仍使用**Sigmoid**（门）和**Tanh**（候选状态），因为有界性对循环稳定性是必要的。若需要加速，可用**ISRU**替换。
- **不使用归一化层的深层全连接网络**：使用**SELU**并配合**LeCun**初始化（权重方差$1/K$），可获得自标准化效果。
- **隐式神经表示（坐标到信号的映射）**：使用**SIREN**（$\omega_0=30$），并且**必须**采用配套的初始化方案，否则网络无法训练。这类任务尤其要避免**ReLU**：**ReLU**网络的二阶导数几乎处处为$0$，任何用到二阶导数的监督（**Laplacian**、**Poisson**方程等）都会失效。
- **音频合成与周期信号**：使用**Snake**或其可学习变体**snakebeta**（**BigVGAN**的选择）。要特别注意：标准激活函数在周期信号上能很好地**插值**，但**外推**会灾难性失败，这一点在评测时容易被掩盖。
- **长尾/不平衡识别**：**AGLU**是一个近乎免费的升级（每通道两个可学习标量），它让不同类别可以使用不同形状的激活曲线。
- **科学计算与符号回归**：考虑**KAN**，可学习的边上激活函数带来了很强的可解释性。但要接受它的效率代价，不要指望把它用在大规模训练上。
- **遇到dead ReLU（大量神经元输出恒为$0$）**：换用**LeakyReLU**、**PReLU**或**ELU**。若希望保留**ReLU**的稀疏性与推理成本，**SUGAR**提供了另一种选择（前向不变，只替换反向梯度）。也应同时检查学习率是否过大、初始化是否恰当。
- **小数据集**：**RReLU**的随机性提供了额外的正则化。
- **可以承担参数与实现成本、追求极致精度**：考虑可学习激活函数（**PWLU**、**PAU**、**ACON**、**Dynamic ReLU**）。它们通常能带来提升，但引入额外参数、超参数和实现复杂度，性价比需要具体权衡。
- **Last but not least**：激活函数是一个极易产生虚假提升的研究方向：报告中的收益常常来自超参数调优、更长的训练或更好的初始化，而非函数本身。简单的激活函数往往优于复杂的激活函数，且保留恒等路径$x$几乎总是有益的。
