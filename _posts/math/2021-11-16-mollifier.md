---
layout: post
title: '函数的光滑化'
date: 2021-11-16
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6194c14e2ab3f51d9157ea72.jpg'
tags: 数学
---

> Smooth function and smoothing technique.

- 机器学习中的优化方法是基于梯度的，因此光滑的模型更利于优化(其梯度是连续的)。然而机器学习模型中经常存在非光滑函数(如激活函数)，比如常用的**ReLU**激活函数$\max(0,x)$就是非光滑的。
- 在最优化问题中，求函数极值需要求导，有时目标函数是不可导的，比如函数中存在最大值函数$\max(x,y)$，可以将这些不可导函数用可导函数近似。
- 许多任务的评价指标是离散的，如分类任务的评价指标是准确率；而常见的损失函数是连续的，如分类任务使用交叉熵损失。损失函数的降低和评价指标的提升并不是完全的单调关系，但不能直接使用评价指标作为损失函数，因为评价指标是不可导的。

综上所述，需要对非光滑函数进行光滑近似的方法。

本文首先对函数的光滑化进行定义，并介绍几种对函数进行光滑化的方法。


# 1. 函数光滑化的定义

**光滑函数**(**smooth function**)是指在其定义域内无穷阶数连续可导的函数。

函数的光滑化是指对于一个非光滑函数$f$，寻找一个光滑函数$f_{\mu}$，使得$f_{\mu}$是$f$的光滑逼近。一个非光滑函数是否**可光滑化**(**smoothable**)定义如下：

给定一个凸函数$f$，我们称其为$(\alpha,\beta)$-**smoothable**的，如果存在一个凸函数$f_{\mu}$，使得满足：
1. $f_{\mu}$是$\frac{\alpha}{\mu}$光滑的
2. $f_{\mu}(x)\leq f(x)\leq f_{\mu}+\beta \mu$

并且称$f_{\mu}$为$f$在参数$(\alpha,\beta)$下的$\frac{1}{\mu}$光滑逼近。

条件1要求$f_{\mu}$是光滑的，并指定光滑系数$\frac{\alpha}{\mu}$(越小越光滑)。条件2要求$f_{\mu}$从下方逼近$f$，并指定逼近的最大差异$\beta \mu$。参数$\mu$越大，则函数$f_{\mu}$越光滑，但与$f$的差异也就越大。

# 2. 函数光滑化的方法

本节介绍几种函数光滑化的方法，包括：
- 人工选取光滑近似
- 使用**Dirac**函数近似

## (1) 人工选取光滑近似

| 原函数 | 光滑近似函数  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |   :---: |
| $\max(x,y)$  | $\mathop{\lim}_{k \to ∞}\frac{1}{k}\ln(e^{kx}+e^{ky})$    |
| $\max(x_1,x_2,...,x_n)$  | $\text{logsumexp}(x_1,x_2,...,x_n)$    |
| $\text{onehot}(\arg\max(x))$  | $\text{softmax}(x_1,x_2,...,x_n)$    |
| $\arg\max(x)$  | $\sum_{i=1}^{n}i\times \text{softmax}(x)_i$    |
| $\text{accuracy}$  | $\frac{1}{\|\mathcal{B}\|}\sum_{x \in \mathcal{B}}^{} <1_y(x),p(x)>$    |
| $\text{F1-score}$  | $\frac{2\sum_{x \in \mathcal{B}}^{} p(x)y(x)}{\sum_{x \in \mathcal{B}}^{}p(x)+y(x)}$    |
| $\max(x_1,x_2)$  | $\frac{x_1+x_2+(x_1-x_2) \text{erf}(\mu (x_1-x_2))}{2}$    |
| $\max(x_1,x_2)$  | $(x_1-x_2)\sigma(\beta(x_1-x_2))+x_2$    |


下面介绍这些近似的推导过程：

### ⚪ $\max(x,y)=\mathop{\lim}_{k \to ∞}\frac{1}{k}\ln(e^{kx}+e^{ky})$

当$x\geq 0,y\geq 0$时，有最大值函数的近似公式：

$$ \max(x,y) = \frac{1}{2}(|x+y|+|x-y|) $$

因此问题转化为寻找绝对值函数$f(x)=\|x\|$的光滑近似。注意到$f(x)$在$x≠0$时可以求导，其导数为：

$$ f'(x)= \begin{cases} 1,&x>0 \\ -1,&x<0 \end{cases} $$

上式可用单位阶跃函数$$\theta(x)=\begin{cases} 1,&x>0 \\ 0,&x<0 \end{cases}$$表示，即：

$$ f'(x)=2\theta(x)-1 $$

单位阶跃函数$\theta(x)$具有近似函数$\theta(x)=\mathop{\lim}_{k \to ∞}\frac{1}{1+e^{-kx}}$，因此：

$$ f(x) = \int_{}^{} [2\theta(x)-1] dx≈ \int_{}^{} [2\frac{1}{1+e^{-kx}}-1] dx \\ = \frac{2}{k}\ln(1+e^{kx})-x= \frac{2}{k}\ln(1+e^{kx})-\frac{2}{k}\ln(e^\frac{kx}{2}) \\ = \frac{2}{k}\ln(\frac{1+e^{kx}}{e^\frac{kx}{2}}) = \frac{2}{k}\ln(e^\frac{kx}{2}+e^\frac{-kx}{2}) \\ = \frac{1}{k}\ln(e^{kx}+e^{-kx}+2) $$

当$k$足够大时，常数$2$可以省略。因此绝对值函数的一个光滑近似为$\|x\|=\mathop{\lim}_{k \to ∞} \frac{1}{k}\ln(e^{kx}+e^{-kx})$。则最大值函数的一个光滑近似为：

$$ \max(x,y) = \frac{1}{2}(|x+y|+|x-y|) \\ = \frac{1}{2}(\mathop{\lim}_{k \to ∞} \frac{1}{k}\ln(e^{k(x+y))}+e^{-k(x+y)})+\mathop{\lim}_{k \to ∞} \frac{1}{k}\ln(e^{k(x-y)}+e^{-k(x-y)})) \\ = \mathop{\lim}_{k \to ∞}\frac{1}{2k}\ln(e^{2kx}+e^{2ky}+e^{-2kx}+e^{-2ky}) \\ = \mathop{\lim}_{k \to ∞}\frac{1}{k}\ln(e^{kx}+e^{ky}+e^{-kx}+e^{-ky}) $$

注意到上式基于$x\geq 0,y\geq 0$推导，因此可丢弃$e^{-kx},e^{-ky}$。最终得到最大值函数的光滑近似为：

$$ \max(x,y)=\mathop{\lim}_{k \to ∞}\frac{1}{k}\ln(e^{kx}+e^{ky}) $$

注意到$x,y$取负数时上式仍成立。该近似也可推广到多个变量的最大值函数：

$$ \max(x,y,z,...)=\mathop{\lim}_{k \to ∞}\frac{1}{k}\ln(e^{kx}+e^{ky}+e^{kz}+...) $$

上式实际上是在进行如下操作：寻找一个在实数域上单调递增的函数，要求该函数的增长速度超过线性函数，对该函数求和后取逆函数。因此可以构造类似函数，如取$y=x^{k+1}$，则有近似：

$$ \max(x,y)=\mathop{\lim}_{k \to ∞}\sqrt[k+1]{x^{k+1}+y^{k+1}} $$

### ⚪ $\max(x_1,x_2,...,x_n)≈\text{logsumexp}(x_1,x_2,...,x_n)$

根据上面的结论有：

$$ \max(x_1,x_2,...,x_n) = \mathop{\lim}_{k \to ∞}\frac{1}{k}\ln(\sum_{i=1}^{n}e^{kx_i}) $$

通常设置$k=1$，则有：

$$ \max(x_1,x_2,...,x_n) ≈ \ln(\sum_{i=1}^{n}e^{x_i}) \\ = \text{logsumexp}(x_1,x_2,...,x_n) $$

其中**logsumexp**是一种常用的算子。

### ⚪ $\text{onehot}(\arg\max(x))≈\text{softmax}(x_1,x_2,...,x_n)$ 

$\text{onehot}(\arg\max(x))$表示先求序列$x=[x_1,x_2,...,x_n]$中最大值所在的位置，并用一个**onehot**编码表示。考虑向量：

$$ x'=[x_1,x_2,...,x_n]-\max(x_1,x_2,...,x_n) $$

其最大值对应的位置值为$0$，其余位置值为负。既而考虑

$$ e^{x'}=[e^{x_1-\max(x_1,x_2,...,x_n)},e^{x_2-\max(x_1,x_2,...,x_n)},...,e^{x_n-\max(x_1,x_2,...,x_n)}] $$
作为$\text{onehot}(\arg\max(x))$的近似。上式最大值为$e^0=1$，其余位置值接近$0$。根据之前的结论$\max(x_1,x_2,...,x_n) ≈ \ln(\sum_{i=1}^{n}e^{x_i})$，有：

$$ \text{onehot}(\arg\max(x))≈e^{x'} \\ = [e^{x_1-\ln(\sum_{i=1}^{n}e^{x_i})},e^{x_2-\ln(\sum_{i=1}^{n}e^{x_i})},...,e^{x_n-\ln(\sum_{i=1}^{n}e^{x_i})}] \\ = [\frac{e^{x_1}}{\sum_{i=1}^{n}e^{x_i}},\frac{e^{x_2}}{\sum_{i=1}^{n}e^{x_i}},...,\frac{e^{x_n}}{\sum_{i=1}^{n}e^{x_i}}] \\ = \text{softmax}(x_1,x_2,...,x_n) $$

### ⚪  $\arg\max(x)≈\sum_{i=1}^{n}i\times \text{softmax}(x)_i$    

$\arg\max(x)$表示求序列$x=[x_1,x_2,...,x_n]$中最大值所在的位置，注意到$\arg\max(x)$实际上等于：

$$ \text{sum}(\text{序向量}[1,2,...,n] \otimes \text{onehot}(\arg\max(x))) $$

根据上述结论$\text{onehot}(\arg\max(x))≈\text{softmax}(x_1,x_2,...,x_n)$，$\arg\max(x)$可以表示为：

$$ \arg\max(x)≈\sum_{i=1}^{n}i\times \text{softmax}(x)_i $$

上式也称为**SoftArgmax**，编程实现见[博客](https://0809zheng.github.io/2021/03/22/softargmax.html)。

### ⚪  $\text{accuracy}=\frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} <1_y(x),p(x)>$
本节讨论分类任务中正确率的光滑近似。给定一个批量$\mathcal{B}$的样本，用$$1_{y}(x)$$表示样本的真实类别对应的**onehot**编码，$$1_{\hat{y}}(x)$$表示样本的预测类别对应的**onehot**编码。统计两个编码对应的内积之和(预测相同内积为$1$否则为$0$)，即可得到正确率的表达式(预测正确的数量占总数量的比值)：

$$ \text{accuracy}=\frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} <1_y(x),1_{\hat{y}}(x)> $$

网络的预测结果是经过**softmax**的概率分布$p(x)$，则正确率的光滑近似为：

$$ \text{accuracy}=\frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{} <1_y(x),p(x)> $$

其中**onehot**编码可由**softmax**光滑近似。

### ⚪  $\text{F1-score} = \frac{2\sum_{x \in \mathcal{B}}^{} p(x)y(x)}{\sum_{x \in \mathcal{B}}^{}p(x)+y(x)}$

**F1-score**是分类问题常用的评估指标，计算为查准率和查全率的调和平均。对于二分类问题，若记$p(x)$是预测正类的概率，$y(x)$是样本的标签，则对应的混淆矩阵如下：

$$ \begin{array}{l|cc} \text{标签\预测} & \text{正例} & \text{反例} \\ \hline  \text{正例} & TP=p(x)y(x) & FN=(1-p(x))y(x) \\  \text{反例} & FP=p(x)(1-y(x)) & TN=(1-p(x))(1-y(x)) \\ \end{array} $$

则**F1-score**计算为

$$ \text{F1-score} = \frac{2TP}{2TP+FP+FN} = \frac{2\sum_{x \in \mathcal{B}}^{} p(x)y(x)}{\sum_{x \in \mathcal{B}}^{}p(x)+y(x)} $$

上述推导的**F1-score**的光滑近似是可导的，可以将其相反数作为损失函数。但是在采样过程中，上式是**F1-score**的有偏估计。通常应先用交叉熵训练一段时间，再用上式进行微调。

### ⚪  $\max(x_1,x_2) = \frac{x_1+x_2+(x_1-x_2) \text{erf}(\mu (x_1-x_2))}{2}$

最大值函数也可以表示为：

$$ \max(x_1,x_2) = \frac{x_1+x_2+|x_1-x_2|}{2} $$

其中绝对值函数$\|x\|$常用的光滑近似可以使用$x \text{erf}(\mu x)$和$\sqrt{x^2+\mu^2}$。前者从下面逼近$\|x\|$($\mu$越大越逼近)，后者从上面逼近$\|x\|$($\mu$越小越逼近)。

使用上述近似替换最大值函数的表达式，可以得到最大值函数的两种近似：

$$ \max(x_1,x_2) = \frac{x_1+x_2+(x_1-x_2) \text{erf}(\mu (x_1-x_2))}{2} $$

$$ \max(x_1,x_2) = \frac{x_1+x_2+\sqrt{(x_1-x_2)^2+\mu^2}}{2} $$

关于该近似的讨论可参考[SMU激活函数](https://0809zheng.github.io/2021/11/17/smu.html)。

### ⚪  $\max(x_1,x_2) = (x_1-x_2)\sigma(\beta(x_1-x_2))+x_2$

最大值函数$\max(x_1,x_2,...,x_n)$的一个可微近似为：

$$ \max(x_1,x_2,...,x_n) = \frac{\sum_{i=1}^{n}x_ie^{\beta x_i}}{\sum_{i=1}^{n}e^{\beta x_i}} $$

$\beta$是开关因子，当$\beta \to ∞$上式趋近于最大值函数；当$\beta =0$上式为简单的算术平均。

当$n=2$时，记$\sigma(x) = 1/(1+e^{-x})$，则最大值函数为：

$$ \max(x_1,x_2) = \frac{x_1e^{\beta x_1}+x_2e^{\beta x_2}}{e^{\beta x_1}+e^{\beta x_2}}  \\ = x_1\frac{1}{1+e^{-\beta(x_1-x_2)}}+x_2\frac{1}{1+e^{-\beta(x_2-x_1)}}  \\ =x_1 \sigma(\beta(x_1-x_2))+x_2[1-\sigma(\beta(x_1-x_2))] \\ = (x_1-x_2)\sigma(\beta(x_1-x_2))+x_2 $$

关于该近似的讨论可参考[ACON激活函数](https://0809zheng.github.io/2021/11/18/acon.html)。

## (2) 使用Dirac函数近似
**Dirac**函数又称**Dirac**-$\delta$函数、单位冲激函数，是一种广义函数(泛函)，表达式如下：

$$ \delta(x) = \begin{cases} +∞, & x =0 \\ 0, & x≠0 \end{cases}, \quad \int_{-∞}^{+∞}\delta(x)dx = 1 $$

根据**Dirac**函数的性质：

$$ f(x) = \int_{-∞}^{+∞}f(y)\delta(x-y)dy  $$

如果能找到**Dirac**函数的光滑近似$\phi(x)≈\delta(x)$，当$f(x)$是一个具有可数个间断点的连续函数时，可以构造$f(x)$的光滑近似：

$$ f(x)≈ \int_{-∞}^{+∞}f(y)\phi(x-y)dy = (f* \phi)(x) $$

注意到上式表示为两个函数的卷积，其中$f$为近似的原函数，$\phi$为**Dirac**函数的光滑近似。

**Dirac**函数没有显式表达式，因此常采用一些连续函数作为**Dirac**函数的光滑近似。

一种寻找**Dirac**函数的光滑近似的方法是首先构造类似于正态分布的“钟形曲线”，之后设法让钟形曲线的宽度趋近于$0$，并保持积分为$1$。此时常用的近似包括：

$$ \delta(x) = \mathop{\lim}_{\sigma \to 0} \frac{e^{-x^2/2\sigma^2}}{\sqrt{2\pi}\sigma} \quad ① $$

$$ \delta(x) = \frac{1}{\pi}\mathop{\lim}_{a \to 0} \frac{a}{x^2+a^2} \quad ② $$

另一种光滑近似思路是注意到**Dirac**函数的积分是单位阶跃函数$\theta(x)$：

$$ \int_{-∞}^{x}\delta(x)dx = \theta(x) = \begin{cases} 1, & x >0 \\ 0, & x<0 \end{cases} $$

若能找到$\theta(x)$的光滑近似，其导数即为**Dirac**函数的光滑近似。$\theta(x)$的光滑近似通常是一些“S”型曲线，如**sigmoid**函数$\sigma(x)$，则**Dirac**函数的一个光滑近似为：

$$ \delta(x) =  \mathop{\lim}_{t \to ∞} \frac{d}{dx}\sigma(tx) \\= \mathop{\lim}_{t \to ∞}t\sigma(x)(1-\sigma(x))\\ = \mathop{\lim}_{t \to ∞} \frac{te^{tx}}{(1+e^{tx})^2} \quad ③ $$

### ⚪构造取整函数的光滑近似
本文以向下取整函数为例，记为：

$$ \lfloor x \rfloor = n, \quad x\in [n,n+1) $$

若记$\phi(x)$为**Dirac**函数的光滑近似，则有：

$$ \lfloor x \rfloor ≈ \int_{-∞}^{+∞}\lfloor y \rfloor \phi(x-y)dy = \sum_{n=-∞}^{+∞}\int_{n}^{n+1}n\phi(x-y)dy $$

若$\Phi(y)$为$\phi(y)$的原函数，则$\phi(x-y)$的原函数为$-\Phi(x-y)$，则有：

$$ \lfloor x \rfloor ≈ \sum_{n=-∞}^{+∞}-n \Phi(x-y)|_{n}^{n+1} \\ = \sum_{n=-∞}^{+∞} n[\Phi(x-n)-\Phi(x-n+1)] \\ = \mathop{\lim}_{M,N\to ∞} \sum_{n=-M}^{N} [(n-1)\Phi(x-n)-n\Phi(x-n+1)+\Phi(x-n)] \\ = \mathop{\lim}_{M,N\to ∞}  -(M+1)\Phi(x+M)-N\Phi(x-N+1)+\sum_{n=-M}^{N}\Phi(x-n)  $$

注意到$$\Phi(x+M)_{M\to ∞}≈\Phi(∞)=1,\Phi(x-N+1)_{N\to ∞} ≈\Phi(-∞)=0$$，因此上式进一步化简为：

$$ \lfloor x \rfloor ≈  \mathop{\lim}_{M,N\to ∞}  -(M+1)+\sum_{n=-M}^{N}\Phi(x-n) \\ = \mathop{\lim}_{M,N\to ∞} \sum_{n=-M}^{0}[\Phi(x-n)-1]+\sum_{n=0}^{N}\Phi(x-n)  $$

单位阶跃函数$\theta(x)$是**Dirac**函数的一个原函数，其可由**Sigmoid**函数光滑近似，即$\Phi(y)=\sigma(ty)$。

下图展示了一个$t=10,M=5,N=10$时对取整函数的近似情况：

![](https://pic.imgdb.cn/item/6186339f2ab3f51d91c9d9e8.jpg)

更多使用**Dirac**函数构造光滑近似的例子可参考[SAU激活函数](https://0809zheng.github.io/2021/11/05/sau.html)。