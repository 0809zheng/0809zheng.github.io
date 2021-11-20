---
layout: post
title: '深度学习中的Activation Function'
date: 2020-03-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/5e7b4db6504f4bcb040071f1.png'
tags: 深度学习
---

> Activation Functions in Deep Learning.

本文目录：
1. 激活函数的意义
2. 激活函数应具有的性质
3. 一些常用的激活函数

# 1. 激活函数的意义

## (1) 从生物学的角度理解激活函数
早期激活函数的设计受到生物神经网络中[神经元](http://cs231n.github.io/neural-networks-1/)的启发，即对神经元进行简单的建模。

![](https://pic.imgdb.cn/item/5e7b4db6504f4bcb040071f1.png)

大脑中的**神经元(neuron)**通过**树突(dendrites)**接收其他神经元的输入信号，在胞体中进行信号的处理，通过**轴突(axon)**分发信号。当神经元中的信号累积达到一定阈值时产生电脉冲将信号输出，这个阈值称为**点火率(firing rate)**。
- 其他神经元的输入信号建模为$x_i$
- 树突的信号接收过程建模为$w_ix_i$
- 胞体的信号处理过程建模为$\sum_{i}^{}w_ix_i$
- 点火率建模为$-b$
- 信号累计与阈值的比较建模为$\sum_{i}^{}w_ix_i+b$
- 产生电脉冲建模为$f(\cdot)=\text{step}(\cdot)$
- 轴突的输出信号建模为$f(\sum_{i}^{}w_ix_i+b)$

激活函数是用来模拟“信号累积达到阈值并产生电脉冲”的过程。
值得一提的是，这种对神经元的建模是**coarse**的。真实神经元有很多不同的种类；突触是一个复杂的的非线性动态系统，树突进行的是复杂的非线性运算，轴突的输出时刻也很重要。因此近些年来神经网络中神经元的**生物可解释性(Biological Plausibility)**被逐渐弱化。

## (2) 从非线性的角度理解激活函数

在神经网络中，使用**激活函数(activation function)**能够为网络引入非线性，增强网络的非线性表示能力；当不使用激活函数时（或激活函数为**恒等函数 identity function**），多层神经网络退化为单层网络：

$$ W_2(W_1X+b_1)+b_2\\=W_2W_1X+W_2b_1+b_2\\=(W_2W_1)X+(W_2b_1+b_2)\\=W'x+b' $$


# 2. 激活函数应具有的性质
激活函数能为神经网络引入非线性，因此理论上任何非线性函数都可以作为激活函数。在选择激活函数时应考虑以下性质：

### ⚪ 性质1：连续可导
激活函数需要参与反向传播过程，因此需要计算激活函数的导数，这就要求激活函数需要**连续可导**。

例如**ReLU**族激活函数在$x=0$处不可导。既可以人工指定该点处的梯度；又可以选用形状接近的连续函数进行近似(如**softplus**替代**ReLU**, **CELU**替代**ELU**)。

更多关于不可导函数的光滑化的相关内容可参考[博客](https://0809zheng.github.io/2021/11/16/mollifier.html)。

### ⚪ 性质2：计算量小
激活函数应具有尽可能小的**计算量**，通常线性运算(如**ReLU**族)比指数运算(如**S**型曲线)具有更低的计算量。

通常可以对指数函数进行[Taylor展开](https://0809zheng.github.io/2021/08/20/taylor.html#3-%E6%B3%B0%E5%8B%92%E5%85%AC%E5%BC%8F%E7%9A%84%E5%BA%94%E7%94%A8hard-sigmoid%E4%B8%8Ehard-tanh)(如**HardSigmoid**,**HardTanh**)降低激活函数的计算复杂度。

### ⚪ 性质3：没有饱和区
**饱和**的定义是导数很接近$0$。若激活函数存在饱和区，则会使反向传播的梯度为$0$，从而导致**梯度消失(gradient vanishing)**现象。

早期的激活函数通常使用**S**型函数，如**Sigmoid,Tanh**；这类函数会把输出挤压到一个区域内，导致产生饱和区；这类函数也被称为**squashing function**。

**ReLU**等无上界、有下界的激活函数，在正半轴没有饱和区，减缓了梯度消失现象；在负半轴则会置零(产生**died ReLU**现象, 即由于梯度为$0$阻断了反向传播过程)或趋于饱和。

### ⚪ 性质4：没有偏置偏移
若激活函数的输出不是**zero-centered**的，会使得后一层神经元的输入产生**偏置偏移(bias shift)**，从而减慢梯度下降的收敛速度。

对于某一层神经元的计算，假设具有两个参数$w_1,w_2$，则$y=\sigma(w_1x_1+w_2x_2+b)$，反向传播时两个参数$w_1,w_2$的梯度为：

$$ \nabla_{w_1}L=\frac{\partial L}{\partial {w_1}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial {w_1}} = \frac{\partial L}{\partial y}\cdot \sigma' \cdot x_1 = \nabla_yL \cdot \sigma' \cdot x_1  $$

$$ \nabla_{w_2}L= \nabla_yL \cdot \sigma' \cdot x_2  $$

若上一层的激活函数使得该层神经元的输入值大于$0$，则$\text{sign}(\nabla_{w_1}L)=\text{sign}(\nabla_{w_2}L)$，则梯度只能沿着$w_1,w_2$同时增大或减小的方向进行更新，从而减慢梯度下降的收敛速度。

当激活函数的值域同时包含正和负值的输出，则能够有效缓解偏置偏移现象。

### ⚪ 性质5：具有生物可解释性
生物神经元通常具有**单侧抑制**(即大于阈值才会被激活)、**宽兴奋边界**(即输出范围较宽,如$[0,+∞)$)、**稀疏激活**(即同时被激活的神经元较少)等特性。

**ReLU**及之前的激活函数在设计时受到生物学的启发，而其后的激活函数在设计时逐渐淡化了生物可解释性。

### ⚪ 性质6：提取上下文信息
通常的激活函数是标量函数，如**ReLU**对神经元输入的每一个标量值分别进行计算。如果能够将激活函数拓展为多输入函数，则能够捕捉输入的上下文信息，增强神经元的表达能力。

某个特征位置的上下文信息既可以从所有输入特征中获取(如**maxout**，**Dynamic ReLU**)，也可以由在该特征的一个邻域上获取(如**Dynamic Shift-Max**，**FReLU**)。


### ⚪ 性质7：具有通用近似性
直观上神经网络每一层的每个神经元都应具有不同的激活曲线。可以设计一些由超参数控制的通用近似激活函数，使得每个神经元学习不同的激活曲线。每个神经元的激活超参数参与反向传播的梯度更新。

设计通用近似的激活函数主要有两种思路。第一种是使用一些通用的函数逼近方法，如分段线性近似(**APL, PWLU**)、**Padé**近似(**PAU, OPAU**)。第二种是寻找现有激活函数的光滑逼近，如手工设计近似(**ACON, SMU**)、使用**Dirac**函数寻找光滑近似(**SAU**)。


# 3. 一些常用的激活函数
- Reference：[Pytorch中的激活函数层](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

下面介绍的激活函数根据设计思路也可分类如下：
- **S**型激活函数：形如**S**型曲线的激活函数。包括**Step**，**Sigmoid**，**HardSigmoid**，**Tanh**，**HardTanh**
- **ReLU**族激活函数：形如**ReLU**的激活函数。包括**ReLU**，**Softplus**，**ReLU6**，**LeakyReLU**，**PReLU**，**RReLU**，**ELU**，**GELU**，**CELU**，**SELU**
- 自动搜索激活函数：通过自动搜索解空间得到的激活函数。包括**Swish**，**HardSwish**，**Elish**，**HardElish**，**Mish**
- 基于梯度的激活函数：通过梯度下降为每个神经元学习独立函数。包括**APL**，**PAU**，**ACON**，**PWLU**，**OPAU**，**SAU**，**SMU**
- 基于上下文的激活函数：多输入单输出函数，输入上下文信息。包括**maxout**，**Dynamic ReLU**，**Dynamic Shift-Max**，**FReLU**

<style>
table th:first-of-type {
    width: 20%;
}
table th:nth-of-type(2) {
    width: 40%;
}
table th:nth-of-type(3) {
    width: 40%;
}
</style>


| 激活函数 | 表达式 |  函数图像 |
| ---- | :----: |   :----: |
| Step | $$\begin{cases} 1, & x≥0 \\ 0, &x<0 \end{cases}$$ | ![](https://pic.imgdb.cn/item/61962ecd2ab3f51d913852ce.png)   |
| Sigmoid | $$\frac{1}{1+e^{-x}}$$ |![](https://pic.imgdb.cn/item/61962e8f2ab3f51d913837b8.png)   |
| [<font color=Blue>Hardsigmoid</font>](https://0809zheng.github.io/2021/08/20/taylor.html#3-%E6%B3%B0%E5%8B%92%E5%85%AC%E5%BC%8F%E7%9A%84%E5%BA%94%E7%94%A8hard-sigmoid%E4%B8%8Ehard-tanh):降低Sigmoid计算量 | $$\begin{cases} 1, & x≥1 \\ (x+1)/2, & -1<x<1 \\ 0, &x≤-1 \end{cases}$$ | ![](https://pic.imgdb.cn/item/61962e462ab3f51d91380e56.png)   |
| Tanh | $$2\text{Sigmoid}(2x)-1\\=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$ | ![](https://pic.imgdb.cn/item/61962ecd2ab3f51d913852e3.png)  |
| [<font color=Blue>Hardtanh</font>](https://0809zheng.github.io/2021/08/20/taylor.html#3-%E6%B3%B0%E5%8B%92%E5%85%AC%E5%BC%8F%E7%9A%84%E5%BA%94%E7%94%A8hard-sigmoid%E4%B8%8Ehard-tanh):降低Tanh计算量 | $$\begin{cases} 1, & x>1 \\ x, & -1≤x≤1 \\ -1, &x<-1 \end{cases}$$ | ![](https://pic.imgdb.cn/item/61962e462ab3f51d91380e5e.png)   |
| [Softplus](https://www.researchgate.net/publication/4933639_Incorporating_Second-Order_Functional_Knowledge_for_Better_Option_Pricing):连续形式的ReLU | $$\int_{}^{}\text{Sigmoid}(x)dx \\=\ln(1+e^x)$$ | ![](https://pic.imgdb.cn/item/61962ecd2ab3f51d913852c7.png)  |
| [ReLU](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) | $$\max(x,0) \\=\begin{cases} x, & x≥0 \\ 0, &x<0 \end{cases}$$ | ![](https://pic.imgdb.cn/item/61962e8f2ab3f51d913837b0.png)  |
| [<font color=Blue>ReLU6</font>](https://0809zheng.github.io/2021/09/13/mobilenetv1.html):部署移动端 | $$\min(\max(x,0),6) \\=\begin{cases} 6, & x\geq 6 \\ x, & 0\leq x<6 \\ 0, &x<0 \end{cases}$$ | ![](https://pic.imgdb.cn/item/619630e92ab3f51d91394de8.png)  |
| [<font color=Blue>Maxout</font>](https://0809zheng.github.io/2021/10/23/maxout.html)：分段线性单元 |$$\mathop{\max}_{j\in [1,k]}x^TW_{i,j}+b_{ij}$$ | ![](https://pic.imgdb.cn/item/619785362ab3f51d91ee819c.jpg)| 
| [<font color=Blue>LeakyReLU</font>](https://0809zheng.github.io/2021/08/29/lrelu.html):解决Dead ReLU | $$\max(x,0.01x) \\=\begin{cases} x, & x≥0 \\ 0.01x, &x<0 \end{cases}$$ | ![](https://pic.imgdb.cn/item/61962e8f2ab3f51d913837a5.png)  |
| [<font color=Blue>APL</font>](https://0809zheng.github.io/2021/10/26/apl.html):通过ReLU构造分段线性 | $$\max(0,x)\\+\sum_{s=1}^{S}a^s\max (0,-x+b^s)$$ | ![](https://pic.imgdb.cn/item/619618512ab3f51d912c24a3.jpg)  |
| [<font color=Blue>PReLU</font>](https://0809zheng.github.io/2021/08/30/prelu.html):可学习参数$\alpha$ | $$\max(x,\alpha x) \\=\begin{cases} x, & x≥0 \\ \alpha x, &x<0 \end{cases}$$ |  ![](https://pic.imgdb.cn/item/61962f902ab3f51d9138b61c.png)|
| [<font color=Blue>RReLU</font>](https://0809zheng.github.io/2021/08/31/rrelu.html):均匀分布采样$\alpha$ | $$\max(x,\alpha x) \\=\begin{cases} x, & x≥0 \\ \alpha x, &x<0 \end{cases}$$ | ![](https://pic.imgdb.cn/item/619631e82ab3f51d9139c736.jpg) |
| [<font color=Blue>ELU</font>](https://0809zheng.github.io/2021/08/25/elu.html):解决bias shift | $$\begin{cases}x,  & x≥0 \\α(e^x-1), & x<0\end{cases}$$ | ![](https://pic.imgdb.cn/item/61962fcd2ab3f51d9138cec7.png)  |
| [<font color=Blue>GELU</font>](https://0809zheng.github.io/2021/08/24/gelu.html):引入正则化 | $$x\Phi(x)=x\int_{-∞}^{x} \frac{e^{-\frac{t^2}{2}}}{\sqrt{2\pi}}dt \\  = x\cdot \frac{1}{2}(1+\text{erf}(\frac{x}{\sqrt{2}}))$$ |![](https://pic.imgdb.cn/item/61962e462ab3f51d91380e52.png)  |
| [<font color=Blue>CELU</font>](https://0809zheng.github.io/2021/08/22/celu.html):连续可微的ELU | $$\begin{cases}x,  & x≥0 \\α(e^{\frac{x}{\alpha}}-1), & x<0\end{cases}$$ | ![](https://pic.imgdb.cn/item/619631b52ab3f51d9139acb4.png)  |
| [<font color=Blue>SELU</font>](https://0809zheng.github.io/2021/09/02/selu.html):自标准化的ELU | $$\begin{cases}\lambda x,  & x≥0 \\\lambda α(e^x-1), & x<0\end{cases}$$ | ![](https://pic.imgdb.cn/item/6196301d2ab3f51d9138f58d.png)  |
| [<font color=Blue>Swish</font>](https://0809zheng.github.io/2021/09/04/swish.html):自动搜索 | $$x\cdot \text{Sigmoid}(\beta x) \\ = \frac{x}{1+e^{-\beta x}}$$ | ![](https://pic.imgdb.cn/item/61962ecd2ab3f51d913852d6.png) |
| [<font color=Blue>HardSwish</font>](https://0809zheng.github.io/2021/09/15/mobilenetv3.html):降低Swish计算量 | $$x \cdot \frac{\text{ReLU6}(x+3)}{6} \\ = \begin{cases} x , & x \geq 3 \\ \frac{x(x+3)}{6} , & -3 \leq x <3 \\ 0, & x < -3 \end{cases}$$ |![](https://pic.imgdb.cn/item/6196309c2ab3f51d91392d93.png) |
| [<font color=Blue>ELiSH</font>](https://0809zheng.github.io/2021/09/03/elish.html):遗传算法 | $$\text{Sigmoid}(x) \cdot \text{ELU}(x) \\= \begin{cases}\frac{x}{1+e^{-x}},  & x≥0 \\\frac{e^x-1}{1+e^{-x}}, & x<0\end{cases}$$ | ![](https://pic.imgdb.cn/item/61962e462ab3f51d91380e4d.png)  |
| [<font color=Blue>HardELiSH</font>](https://0809zheng.github.io/2021/09/03/elish.html):降低ELiSH计算量 | $$\text{HardSigmoid}(x) \cdot \text{ELU}(x) \\= \begin{cases} x, & x≥1 \\ x(x+1)/2, & 0 \leq x<1 \\ (e^x-1)(x+1)/2, & -1\leq x<0 \\ 0, &x≤-1 \end{cases}$$ | ![](https://pic.imgdb.cn/item/619632bf2ab3f51d913a2484.png)  |
| [<font color=Blue>PAU</font>](https://0809zheng.github.io/2021/10/24/pade.html):Padé近似 | $$\frac{a_0+a_1x+a_2x^2+...+a_mx^m}{1+\|b_1\|\|x\|+\|b_2\|\|x\|^2+...+\|b_n\|\|x\|^n}$$ |![](https://pic.imgdb.cn/item/619618fc2ab3f51d912c898d.jpg)  |
| [<font color=Blue>Mish</font>](https://0809zheng.github.io/2021/08/21/mish.html):进一步搜索Swish | $$x\cdot \text{tanh}(\text{softplus}(x)) \\ =x\cdot \text{tanh}(\ln(1+e^x))$$ | ![](https://pic.imgdb.cn/item/61962e8f2ab3f51d913837a8.png)  |
| [<font color=Blue>Dynamic ReLU</font>](https://0809zheng.github.io/2021/10/27/dyrelu.html)：动态ReLU | $$\mathop{\max}_{1\leq k \leq K} \{a_c^k(x)x_c+b_c^k(x)\}$$ | ![](https://pic.imgdb.cn/item/619707a22ab3f51d919c2561.jpg) |
| [<font color=Blue>Dynamic Shift-Max</font>](https://0809zheng.github.io/2021/11/09/micronet.html):循环移位多输入 | $$\mathcal{\max}_{1\leq k\leq K} \{\sum_{j=0}^{J-1} a_{i,j}^k(x)x_{\frac{C}{G}}(i,j)\}$$ | ![](https://pic.imgdb.cn/item/619393ea2ab3f51d919f26bd.jpg)  |
| [<font color=Blue>FReLU</font>](https://0809zheng.github.io/2020/09/05/frelu.html):卷积窗口输入 | $$\mathcal{\max}_{1\leq k\leq K} \{\sum_{j=0}^{J-1} a_{i,j}^k(x)x_{\frac{C}{G}}(i,j)\}$$ | ![](https://pic.imgdb.cn/item/619632ec2ab3f51d913a3a3a.jpg)  |
| [<font color=Blue>ACON</font>](https://0809zheng.github.io/2021/11/18/acon.html):最大值函数的$\alpha$-**softmax**近似 | $$(p_1-p_2)x\sigma(\beta (p_1-p_2)x)+p_2x$$ | ![](https://pic.imgdb.cn/item/619616512ab3f51d912aeaa5.jpg)  |
| [<font color=Blue>PWLU</font>](https://0809zheng.github.io/2020/10/22/plu.html):分段线性近似 | $$\begin{cases}  (x-B_L)*K_L+Y_P^0, & x<B_L \\ (x-B_R)*K_R+Y_P^N, & x\geq B_R \\ (x-B_{idx})*K_{idx}+Y_P^{idx}, & \text{others} \end{cases}$$ | ![](https://pic.imgdb.cn/item/6196180a2ab3f51d912bffcc.jpg)  |
| [<font color=Blue>OPAU</font>](https://0809zheng.github.io/2021/10/25/opade.html):正交Padé近似 | $$\frac{c_0+c_1f_1(x)+c_2f_2(x)+...+c_kf_k(x)}{1+\|d_1\|\|f_1(x)\|+\|d_2\|\|f_2(x)\|+...+\|d_l\|\|f_l(x)\|}$$ | 见**PAU**  |
| [<font color=Blue>SAU</font>](https://0809zheng.github.io/2021/11/05/sau.html): 使用Dirac函数近似 | $$\frac{(1-\alpha)\sigma}{\sqrt{2\pi}}  e^{-\frac{x^2}{2\sigma^2}}+  \frac{ x}{2} +  \frac{(1-\alpha) x}{2}\text{erf}(\frac{x}{\sqrt{2}\sigma})$$ | ![](https://pic.imgdb.cn/item/61938c232ab3f51d919b76d7.jpg)  |
| [<font color=Blue>SMU</font>](https://0809zheng.github.io/2021/11/17/smu.html): 最大值函数的光滑近似 | $$\frac{(1+\alpha)x+(1-\alpha)x \text{erf}(\mu (1-\alpha)x)}{2}$$ | ![](https://pic.imgdb.cn/item/6195c47d2ab3f51d91f255d6.jpg)  |



**Reference**:
- [Activation Functions: Comparison of trends in Practice and Research for Deep Learning](https://arxiv.org/abs/1811.03378)
