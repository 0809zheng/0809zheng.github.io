---
layout: post
title: '渐近理论(Asymptotic Theory) II：Slutsky定理与Delta方法'
date: 2026-10-24
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic2-cover.png'
tags: 数学
pinned: false
---

> Asymptotic Theory II: Slutsky's Theorem and the Delta Method.

在本系列的第 I 篇中，我们建立了渐近理论的两大基石：**弱大数定律（WLLN）**给出了样本均值依概率收敛于总体均值的相合性（$\bar{X}_n \xrightarrow{P} \mu$），**中心极限定理（CLT）**给出了标准化样本均值的渐近正态性（$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$）。

在实际的统计推断与机器学习建模中，研究者面临着两个更为普遍的问题：
1. **多个统计量的代数运算（参数替换问题）**：中心极限定理的标准化分母中包含未知的总体标准差 $\sigma$。在实际构造检验统计量或置信区间时，必须使用样本标准差 $S_n$ 替换未知的常数 $\sigma$。将依分布收敛的量与依概率收敛的量进行加、减、乘、除四则运算后，原有的极限分布性质是否保持不变？
2. **统计量的非线性函数变换（误差传播问题）**：在诸多统计模型中，感兴趣的目标参数并非参数 $\theta$ 本身，而是其非线性光滑函数 $g(\theta)$（例如逻辑回归中的对数赔率、神经网络中的 **Sigmoid/Softmax** 激活映射）。当估计量 $\hat{\theta}_n$ 经过非线性映射变为 $g(\hat{\theta}_n)$ 后，其渐近分布形态与渐近方差将如何演化？

解决上述两个核心问题的数学工具，正是本文将要系统阐述的 **Slutsky 定理（斯卢茨基定理）**与 **Delta 方法（$\Delta$ 方法）**：
* **Slutsky 定理**：确立了“依分布收敛序列”与“依概率收敛于常数的序列”之间的四则运算法则，为统计推断中的未知参数代入（**Plug-in Principle**）提供了严格的合法性证明。
* **Delta 方法**：利用微积分中的泰勒展开将非线性函数在真值邻域内局部线性化（或二次化），精确刻画了渐近正态统计量在非线性变换下的方差传播规律。

本文目录：
1. [连续映射定理与随机渐近阶符号](#cmt-stochastic-order)
   - [1.1 连续映射定理](#continuous-mapping)
   - [1.2 随机渐近阶符号](#stochastic-order)
2. [Slutsky 定理](#slutsky-theorem)
   - [2.1 定理背景](#slutsky-background)
   - [2.2 定理的初等证明](#slutsky-proof)
   - [2.3 未知参数代入原理](#plug-in-principle)
   - [例题：IWAE 自归一化梯度估计的渐近分布](#iwae-gradient)
3. [Delta 方法](#delta-method)
   - [3.1 一元一阶 Delta 方法](#univariate-delta)
   - [3.2 多元一阶 Delta 方法](#multivariate-delta)
   - [3.3 二阶 Delta 方法](#second-order-delta)
   - [例题：Laplace 预测近似中的概率误差传播](#sigmoid-softmax)

<a id="cmt-stochastic-order"></a>

# 1. 连续映射定理与随机渐近阶符号

<a id="continuous-mapping"></a>

## 1.1 连续映射定理（CMT）

在确定性微积分中，若数列 $a_n \to a$ 且函数 $g(x)$ 在点 $a$ 处连续，则有 $g(a_n) \to g(a)$。连续映射定理将这一性质完整推广至随机变量的收敛序列。

> **定理 1.1（连续映射定理，Continuous Mapping Theorem）**
> 设 $g: \mathbb{R}^d \to \mathbb{R}^k$ 为 **Borel** 可测映射，其不连续点集合记为 $D_g$。若 $P(\mathbf{X}\in D_g)=0$，即 $g$ 相对于 $\mathbf{X}$ 的分布几乎处处连续，则：
> 1. 若 $\mathbf{X}_n \xrightarrow{P} \mathbf{X}$，则 $g(\mathbf{X}_n) \xrightarrow{P} g(\mathbf{X})$；
> 2. 若 $\mathbf{X}_n \xrightarrow{d} \mathbf{X}$，则 $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$。

**证明（常数概率极限与处处连续映射两种情形）**：
设函数 $g$ 在常数点 $\mathbf{c}$ 处连续。由多元函数在一点连续的 $\varepsilon$-$\delta$ 定义：对任意给定的 $\varepsilon > 0$，存在 $\delta > 0$，使得对任意满足 $\Vert \mathbf{x} - \mathbf{c}\Vert \le \delta$ 的点 $\mathbf{x}$，均有 $\Vert g(\mathbf{x}) - g(\mathbf{c})\Vert \le \varepsilon$。
取其逆否命题：若随机事件 $\{\Vert g(\mathbf{X}_n) - g(\mathbf{c})\Vert > \varepsilon\}$ 发生，则必然有事件 $\{\Vert \mathbf{X}_n - \mathbf{c}\Vert > \delta\}$ 发生。由事件包含关系的概率单调性，得：

$$ 0 \le P\left(\Vert g(\mathbf{X}_n) - g(\mathbf{c})\Vert > \varepsilon\right) \le P\left(\Vert \mathbf{X}_n - \mathbf{c}\Vert > \delta\right) $$

令 $n \to \infty$，由 $\mathbf{X}_n \xrightarrow{P} \mathbf{c}$ 知右端概率趋于 $0$，故 $g(\mathbf{X}_n) \xrightarrow{P} g(\mathbf{c})$。
对于依分布收敛情形，当 $g$ 处处连续时，任取有界连续函数 $h$，则复合函数 $h \circ g$ 仍为有界连续函数。由第 I 篇定理 2.1（**Portmanteau 定理**）的条件 2，直接有 $\mathbb{E}[h(g(\mathbf{X}_n))] = \mathbb{E}[(h\circ g)(\mathbf{X}_n)] \to \mathbb{E}[(h\circ g)(\mathbf{X})] = \mathbb{E}[h(g(\mathbf{X}))]$，故 $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$。$\blacksquare$

<a id="stochastic-order"></a>

## 1.2 随机渐近阶符号：$o_p(\cdot)$ 与 $O_p(\cdot)$

在经典数学分析中，**Landau** 符号 $o(\cdot)$ 与 $O(\cdot)$ 被广泛用于表示无穷小量与有界量的渐近阶。在数理统计中，为了简化复杂随机展开式的书写，引入了对应的**随机渐近阶符号（Stochastic Order Symbols）**。

> **定义 1.1（依概率无穷小量 $o_p$）**
> 设 $\{X_n\}$ 为随机变量序列，$\{a_n\}$ 为正实数序列。
> 1. 若 $X_n \xrightarrow{P} 0$，则记作 $X_n = o_p(1)$；
> 2. 若 $X_n/a_n \xrightarrow{P} 0$，则记作 $X_n = o_p(a_n)$。

> **定义 1.2（依概率有界量 $O_p$）**
> 设 $\{X_n\}$ 为随机变量序列，$\{a_n\}$ 为正实数序列。
> 1. 若对任意 $\eta>0$，存在有限常数 $M_\eta>0$ 和正整数 $N_\eta$，使得当 $n>N_\eta$ 时恒有
> $$ P\left(\vert X_n\vert > M_\eta\right) < \eta $$
> 则称序列 $\{X_n\}$ **依概率有界（Bounded in Probability）**，记作 $X_n=O_p(1)$；
> 2. 若 $X_n/a_n=O_p(1)$，则记作 $X_n=O_p(a_n)$。

> **定理 1.2（依分布收敛必依概率有界）**
> 若随机变量序列满足 $X_n \xrightarrow{d} X$，则必有 $X_n = O_p(1)$。

**证明**：任取 $\eta > 0$。由于极限随机变量 $X$ 的分布函数满足 $\lim_{x\to -\infty} F_X(x) = 0$ 及 $\lim_{x\to +\infty} F_X(x) = 1$，故存在充分大的正数 $M > 0$，使得 $-M$ 与 $M$ 均为 $F_X$ 的连续点，且满足：

$$ P(\vert X\vert > M) \le F_X(-M) + \left[1 - F_X(M)\right] < \frac{\eta}{2} $$

由 $X_n \xrightarrow{d} X$，在连续点 $\pm M$ 处有 $\lim_{n\to\infty} \left(F_{X_n}(-M) + 1 - F_{X_n}(M)\right) = F_X(-M) + 1 - F_X(M) < \frac{\eta}{2}$。由极限的保号性，存在正整数 $N$，当 $n > N$ 时恒有：

$$ P(\vert X_n\vert > M) \le F_{X_n}(-M) + 1 - F_{X_n}(M) < \eta $$

此即表明 $X_n = O_p(1)$。$\blacksquare$

若 $X_n=O_p(1)$ 且确定性序列 $a_n\to0$，则 $a_nX_n=o_p(1)$。证明如下：给定 $\varepsilon,\eta>0$，先取 $M$ 使大样本时 $P(\|X_n\|>M)<\eta$，再取 $n$ 充分大使 $\|a_n\|M\le\varepsilon$，便有 $P(\|a_nX_n\|>\varepsilon)\le P(\|X_n\|>M)<\eta$。因此中心极限定理给出的 $\sqrt n$ 尺度波动也蕴含未缩放估计量的相合性。

> **应用示例（中心极限定理的渐近阶表示）**：
> 由中心极限定理知 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$。根据定理 1.2，该收敛序列必依概率有界，即：
> $$ \sqrt{n}(\bar{X}_n - \mu) = O_p(1) \iff \bar{X}_n - \mu = O_p\left(n^{-1/2}\right) $$
> 这表明样本均值逼近总体均值的收敛速率为 $n^{-1/2}$ 量级。由于 $n^{-1/2} \to 0$，故亦有 $\bar{X}_n - \mu = o_p(1)$（即弱大数定律）。

<a id="slutsky-theorem"></a>

# 2. Slutsky 定理

<a id="slutsky-background"></a>

## 2.1 定理背景

在实分析中，若两个确定性数列满足 $x_n \to x$ 且 $y_n \to y$，则必然有 $x_n + y_n \to x + y$ 以及 $x_n y_n \to xy$。但在概率论中，**若 $X_n \xrightarrow{d} X$ 且 $Y_n \xrightarrow{d} Y$，通常无法推出 $X_n + Y_n \xrightarrow{d} X + Y$**。
其根本原因在于：单凭边缘分布的收敛性，无法确定 $X_n$ 与 $Y_n$ 之间的相关结构（联合分布）。例如，设 $Z\sim N(0,1)$，取 $X_n=Z$、$Y_n=-Z$，并令候选边缘极限 $X=Y=Z$。由正态分布的对称性，$X_n\xrightarrow{d}X$ 且 $Y_n\xrightarrow{d}Y$，但 $X_n+Y_n\equiv0$，而 $X+Y=2Z\sim N(0,4)$，故两者的边缘收敛并不能推出和收敛于 $X+Y$。这说明仅凭两个边缘分布，无法确定和的极限分布。

**Slutsky 定理**指出：若两个序列中有一个依概率收敛于**确定性常数 $c$**（即 $Y_n \xrightarrow{P} c$），则上述相关性障碍将自动消除。由于 $Y_n$ 的概率质量集中到常数 $c$ 的任意小邻域，而 $X_n$ 的概率质量保持紧性，两者的联合极限为 $(X,c)$，因而运算中的 $Y_n$ 可以在极限意义下替换为 $c$。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-slutsky-001.png)

如上图所示，在二维平面 $(X_n, Y_n)$ 上，由于 $X_n \xrightarrow{d} X$，对任意给定的小尾概率，都可以选取充分大的 $M$，使大样本时的主要概率质量位于 $[-M,M]$ 内；同时由于 $Y_n \xrightarrow{P} c$，纵向的概率质量被无限压缩进宽度为 $2\varepsilon$ 的水平带状区域 $[c-\varepsilon, c+\varepsilon]$ 内。在极限状态下，二维随机向量 $(X_n, Y_n)$ 退化并依分布收敛于 $(X, c)$。

> **定理 2.1（Slutsky 定理）**
> 设 $\{X_n\}, \{Y_n\}$ 为随机变量序列（允许任意相依），$X$ 为随机变量，$c$ 为常数。若 $X_n \xrightarrow{d} X$ 且 $Y_n \xrightarrow{P} c$，则有以下三条结论成立：
> 1. $X_n + Y_n \xrightarrow{d} X + c$；
> 2. $X_n Y_n \xrightarrow{d} cX$；
> 3. $\dfrac{X_n}{Y_n} \xrightarrow{d} \dfrac{X}{c} \quad (c \neq 0)$。

若某些 $Y_n$ 可能等于零，可在该事件上任意定义商；因为 $P(Y_n=0)\le P(\vert Y_n-c\vert>\vert c\vert/2)\to0$，这不改变极限分布。

<a id="slutsky-proof"></a>

## 2.2 定理的初等证明

**(1) 证明加法法则 $X_n + Y_n \xrightarrow{d} X + c$**：

设 $x$ 为极限随机变量 $X + c$ 的分布函数 $F_{X+c}(x) = P(X + c \le x) = F_X(x - c)$ 的任意连续点，即 $x - c$ 为 $F_X$ 的连续点。任取 $\varepsilon > 0$（不妨取使 $x - c \pm \varepsilon$ 均为 $F_X$ 连续点的 $\varepsilon$）。
依据事件 $\{\vert Y_n - c\vert \le \varepsilon\}$ 是否发生，对事件 $\{X_n + Y_n \le x\}$ 进行拆分：
* **上界放缩**：若 $X_n + Y_n \le x$ 且 $\vert Y_n - c\vert \le \varepsilon$ 同时成立，由 $Y_n \ge c - \varepsilon$ 必有 $X_n \le x - Y_n \le x - c + \varepsilon$。故有：

  $$ P(X_n + Y_n \le x) \le F_{X_n}(x - c + \varepsilon) + P(\vert Y_n - c\vert > \varepsilon) $$

* **下界放缩**：同理，若 $X_n \le x - c - \varepsilon$ 且 $\vert Y_n - c\vert \le \varepsilon$ 同时成立，由 $Y_n \le c + \varepsilon$ 必有 $X_n + Y_n \le x$。故有 $F_{X_n}(x - c - \varepsilon) \le P(X_n + Y_n \le x) + P(\vert Y_n - c\vert > \varepsilon)$。

综合上下界，得到双侧夹逼不等式链：

$$ F_{X_n}(x - c - \varepsilon) - P(\vert Y_n - c\vert > \varepsilon) \le P(X_n + Y_n \le x) \le F_{X_n}(x - c + \varepsilon) + P(\vert Y_n - c\vert > \varepsilon) $$

先固定 $\varepsilon > 0$ 令 $n \to \infty$：因 $Y_n \xrightarrow{P} c$，尾概率项 $P(\vert Y_n - c\vert > \varepsilon) \to 0$；因 $X_n \xrightarrow{d} X$，左右两端分别收敛于 $F_X(x - c - \varepsilon)$ 与 $F_X(x - c + \varepsilon)$。再令 $\varepsilon \to 0^+$，由 $F_X$ 在点 $x - c$ 处的连续性与夹逼准则，立得 $\lim_{n\to\infty} P(X_n + Y_n \le x) = F_X(x - c) = F_{X+c}(x)$。

**(2) 证明乘法法则 $X_n Y_n \xrightarrow{d} cX$（利用 $O_p(1) \cdot o_p(1) = o_p(1)$）**：

将乘积作代数恒等拆分：

$$ X_n Y_n = cX_n + X_n(Y_n - c) $$

* **主部 $cX_n$**：由于线性函数 $g(x) = cx$ 处处连续，由本篇定理 1.1（连续映射定理）知 $cX_n \xrightarrow{d} cX$。
* **余项 $X_n(Y_n - c)$**：由 $X_n \xrightarrow{d} X$ 及本篇定理 1.2（依分布收敛必依概率有界，即 $X_n = O_p(1)$），对任意给定的 $\eta > 0$，存在常数 $M > 0$，使得当 $n$ 充分大时恒有 $P(\vert X_n\vert > M) < \frac{\eta}{2}$。
  又对任意 $\varepsilon > 0$，若事件 $\{\vert X_n(Y_n - c)\vert > \varepsilon\}$ 发生，则必然有 $\{\vert X_n\vert > M\}$ 或 $\{\vert Y_n - c\vert > \frac{\varepsilon}{M}\}$ 发生。结合 $Y_n \xrightarrow{P} c$，当 $n$ 充分大时有：

  $$ P\left(\vert X_n(Y_n - c)\vert > \varepsilon\right) \le P\left(\vert X_n\vert > M\right) + P\left(\vert Y_n - c\vert > \frac{\varepsilon}{M}\right) < \frac{\eta}{2} + \frac{\eta}{2} = \eta $$

  此即证明了 $X_n(Y_n - c) \xrightarrow{P} 0$。

将 $cX_n \xrightarrow{d} cX$ 与 $X_n(Y_n - c) \xrightarrow{P} 0$ 代入**(1) 的加法法则**，立即得到 $X_n Y_n \xrightarrow{d} cX + 0 = cX$。

**(3) 证明除法法则 $X_n / Y_n \xrightarrow{d} X / c \quad (c \neq 0)$**：

由于 $c \neq 0$，倒数函数 $g(y) = 1/y$ 在点 $y = c$ 处连续。由 $Y_n \xrightarrow{P} c$ 及连续映射定理得 $\frac{1}{Y_n} \xrightarrow{P} \frac{1}{c}$。再对乘积 $X_n \cdot \frac{1}{Y_n}$ 应用**(2) 的乘法法则**，即得 $\frac{X_n}{Y_n} \xrightarrow{d} \frac{X}{c}$。$\blacksquare$

<a id="plug-in-principle"></a>

## 2.3 未知参数代入原理（Plug-in Principle）

设 $X_i$ 独立同分布，$\mathbb{E}[X_i]=\mu$、$0<\text{Var}(X_i)=\sigma^2<\infty$，且 $n\ge2$。在数理统计中，中心极限定理给出 $$\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} N(0, 1)$$。但在构造总体均值 $\mu$ 的置信区间或进行假设检验时，总体标准差 $\sigma$ 往往是未知参数。统计学家通常使用样本标准差 $$S_n = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X}_n)^2}$$ 替换未知的 $\sigma$，构造出学生 $t$ 统计量：

$$ t_n = \frac{\sqrt{n}\left(\bar{X}_n - \mu\right)}{S_n} $$

当总体并非正态分布时，$t_n$ 在有限样本下不再服从精确的 $t(n-1)$ 分布。但借助渐近理论，我们可以证明不要求总体正态（在上述独立同分布与有限正方差条件下），当 $n \to \infty$ 时，$t_n$ 必然依分布收敛于标准正态分布：
1. **证明样本方差的相合性（$S_n^2 \xrightarrow{P} \sigma^2$）**：将样本方差展开为算术平均的组合：

   $$ S_n^2 = \frac{n}{n-1}\left(\frac{1}{n}\sum_{i=1}^n X_i^2 - \bar{X}_n^2\right) $$

   由弱大数定律，有 $\frac{1}{n}\sum_{i=1}^n X_i^2 \xrightarrow{P} \mathbb{E}[X_1^2] = \sigma^2 + \mu^2$，以及 $\bar{X}_n \xrightarrow{P} \mu$。结合连续映射定理与 **Slutsky** 定理，当 $n \to \infty$ 时：

   $$ S_n^2 \xrightarrow{P} 1 \cdot \left[(\sigma^2 + \mu^2) - \mu^2\right] = \sigma^2 $$

2. **应用连续映射定理与 Slutsky 乘法法则**：由开方函数在 $\sigma^2 > 0$ 处的连续性，得 $S_n \xrightarrow{P} \sigma$，进而 $\frac{\sigma}{S_n} \xrightarrow{P} 1$。将 $t_n$ 改写为乘积形式并应用定理 2.1（**Slutsky 定理**）：

$$ t_n = \underbrace{\frac{\sqrt{n}\left(\bar{X}_n - \mu\right)}{\sigma}}_{\xrightarrow{d} N(0,1)} \cdot \underbrace{\frac{\sigma}{S_n}}_{\xrightarrow{P} 1} \xrightarrow{d} N(0, 1) \cdot 1 = N(0, 1) $$

> **注（一般化参数代入法则 General Plug-in Rule）**：
> 设参数 $\theta$ 的估计量满足渐近正态性 $$\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \tau^2)$$，其中渐近方差 $\tau^2 > 0$ 未知。只要能够构造出 $\tau^2$ 的任意弱相合估计量 $$\hat{\tau}_n^2 \xrightarrow{P} \tau^2$$，由连续映射定理与 **Slutsky** 除法法则，必有：
> 
> $$ \frac{\sqrt{n}\left(\hat{\theta}_n - \theta\right)}{\hat{\tau}_n} \xrightarrow{d} N(0, 1) $$
> 
> 这里取非负方差估计的平方根 $\hat{\tau}_n=\sqrt{\hat{\tau}_n^2}$。据此可构造参数 $\theta$ 的渐近 $1-\alpha$ 置信区间：$\hat{\theta}_n \pm z_{\alpha/2}\frac{\hat{\tau}_n}{\sqrt{n}}$，其中 $z_{\alpha/2}$ 为标准正态分布的上 $\alpha/2$ 分位数。

<a id="iwae-gradient"></a>

## 📖 例题：IWAE 自归一化梯度估计的渐近分布

> **论文背景**：[**Importance Weighted Autoencoders**](https://arxiv.org/abs/1509.00519) (*arXiv* 2015)
>
> **Burda**、**Grosse** 与 **Salakhutdinov** 提出的 **IWAE** 从编码器分布中采样 $K$ 个潜变量，用重要性权重的平均值估计观测 $x$ 的边缘似然 $p_\theta(x)$，并以该平均值的对数期望构造似然下界。对单次采样的平均权重取对数并求导，得到生成网络的一次梯度估计。该梯度是各潜变量样本对应梯度的加权平均，权重除以同组权重之和，称为**自归一化（Self-normalization）**。下面分析增加采样次数后，这一随机梯度如何逼近真实梯度。
>
> **题目描述**：固定观测 $x$ 和编码器分布 $q_\phi(z\mid x)$，只考虑生成网络的一个标量参数 $\theta$，在当前参数值处计算梯度。独立采样 $z_1,\dots,z_K\sim q_\phi(z\mid x)$。对第 $k$ 个样本，定义重要性权重 $w_k$ 和样本梯度 $s_k$：
> 
> $$ w_k=\frac{p_\theta(x,z_k)}{q_\phi(z_k\mid x)},\quad s_k=\partial_\theta\ln p_\theta(x,z_k) $$
> 
> 假设$w_k>0$，$0<p_\theta(x)<\infty$、$\mathbb E_q[|w_1s_1|]<\infty$，且微分与积分可以交换；$\mathbb E_q$ 表示对编码器分布取期望。记真实梯度为 $g=\partial_\theta\ln p_\theta(x)$。
> 1. 推导 $\hat g_K=\partial_\theta\ln(\frac1K\sum_kw_k)$ 的表达式，并证明 $\hat g_K\xrightarrow{P}g$。
> 2. 若 $0<\mathbb E_q[w_1^2(s_1-g)^2]<\infty$，求 $\sqrt K(\hat g_K-g)$ 的极限分布。

**解答**：

**(1) 梯度表达式与相合性**。由于编码器不依赖 $\theta$，对固定的潜变量样本有 $\partial_\theta w_k=w_ks_k$。由对数的求导法则：

$$ \hat g_K=\frac{\frac1K\sum_{k=1}^K\partial_\theta w_k}{\frac1K\sum_{k=1}^Kw_k}=\frac{\frac1K\sum_{k=1}^Kw_ks_k}{\frac1K\sum_{k=1}^Kw_k} $$

将分子、分母中的 $1/K$ 消去，也可写成 $\hat g_K=\sum_k\widetilde w_ks_k$，其中 $\widetilde w_k=w_k/\sum_jw_j$。因此 $\hat g_K$ 是样本梯度 $s_k$ 的自归一化加权平均。

要确定这个比值的极限，分别计算分母和分子中单项的期望。权重中的编码器密度与采样密度相消，得到

$$ \mathbb E_q[w_1]=\int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,\mathrm dz=\int p_\theta(x,z)\,\mathrm dz=p_\theta(x) $$

同样，利用 $p_\theta(x,z)\,\partial_\theta\ln p_\theta(x,z)=\partial_\theta p_\theta(x,z)$：

$$ \mathbb E_q[w_1s_1]=\int\partial_\theta p_\theta(x,z)\,\mathrm dz=\partial_\theta p_\theta(x) $$

因此，分母的样本平均估计边缘似然，分子的样本平均估计边缘似然的导数。不同 $k$ 对应独立采样，因此 $w_k$ 与 $w_ks_k$ 各自都是独立同分布序列。由 **Khinchin 弱大数定律**：

$$ \frac1K\sum_kw_k\xrightarrow{P}p_\theta(x),\quad \frac1K\sum_kw_ks_k\xrightarrow{P}\partial_\theta p_\theta(x) $$

分母收敛到正的常数。由 **Slutsky 除法法则**，再利用依分布收敛到常数与依概率收敛的等价性，得到

$$ \hat g_K\xrightarrow{P}\frac{\partial_\theta p_\theta(x)}{p_\theta(x)}=\partial_\theta\ln p_\theta(x)=g $$

**(2) 梯度误差的渐近分布**。为应用中心极限定理，先将误差写成均值为零的随机变量之和：

$$ \hat g_K-g=\frac{\sum_kw_ks_k-g\sum_kw_k}{\sum_kw_k}=\frac{\frac1K\sum_kw_k(s_k-g)}{\frac1K\sum_kw_k} $$

令 $U_k=w_k(s_k-g)$。第 (1) 问已经证明

$$ \mathbb E_q[U_k]=\mathbb E_q[w_1s_1]-g\mathbb E_q[w_1]=\partial_\theta p_\theta(x)-g\,p_\theta(x)=0 $$

由于 $U_k$ 独立同分布，且二阶矩有限，由 **Lindeberg-Lévy 中心极限定理**：

$$ \frac1{\sqrt K}\sum_kU_k\xrightarrow{d}N\left(0,\mathbb E_q[w_1^2(s_1-g)^2]\right) $$

在误差式两边乘以 $\sqrt K$，得到

$$ \sqrt K(\hat g_K-g)=\frac{\frac1{\sqrt K}\sum_kU_k}{\frac1K\sum_kw_k} $$

分子具有上述正态极限，分母依概率收敛到 $p_\theta(x)>0$。Slutsky 定理不要求它们独立，因此

$$ \sqrt K(\hat g_K-g)\xrightarrow{d}N\left(0,\frac{\mathbb E_q[w_1^2(s_1-g)^2]}{p_\theta(x)^2}\right) $$


<a id="delta-method"></a>

# 3. Delta 方法

<a id="univariate-delta"></a>

## 3.1 一元一阶 Delta 方法

在建模分析中，研究者常需要对基础估计量 $\hat{\theta}_n$ 施加非线性光滑变换 $g(\hat{\theta}_n)$（例如求取对数 $\ln\hat{\theta}_n$、倒数 $1/\hat{\theta}_n$ 或概率赔率 $\frac{\hat{\theta}_n}{1-\hat{\theta}_n}$）。
若已知 $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \tau^2)$，我们需要确定变换后的统计量 $\sqrt{n}\big(g(\hat{\theta}_n) - g(\theta)\big)$ 的极限分布。

**几何直觉（局部线性化与切线缩放）**：
由于 $\hat{\theta}_n - \theta = O_p(n^{-1/2})$，当样本容量 $n$ 充分大时，$\hat{\theta}_n$ 的绝大部分概率质量都高度集中在真值 $\theta$ 的极小邻域内。根据微分思想，无论非线性函数 $g(x)$ 在全局范围内如何弯曲，在 $\theta$ 的无穷小邻域内均可由其**一阶切线** $g(\theta) + g'(\theta)(x - \theta)$ 精确逼近。
正态随机变量经过斜率为 $g'(\theta)$ 的线性变换后，其分布形态仍保持为正态分布，而标准差则按切线斜率的绝对值被缩放了 $\vert g'(\theta)\vert $ 倍（即方差被缩放了 $[g'(\theta)]^2$ 倍）。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-delta-001.png)

> **定理 3.1（一元一阶 Delta 方法）**
> 设随机变量序列 $\{\hat{\theta}_n\}$ 满足渐近正态性 $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \tau^2)$。若函数 $g: \mathbb{R} \to \mathbb{R}$ 在点 $\theta$ 处可导，则有：
> 
> $$ \sqrt{n}\left(g(\hat{\theta}_n) - g(\theta)\right) \xrightarrow{d} N\left(0,\ \tau^2\left[g'(\theta)\right]^2\right) $$

当 $g'(\theta)=0$ 时，结论仍成立，但极限退化于零；若要得到非退化正态极限，则还需 $\tau^2>0$ 且 $g'(\theta)\neq0$。

**证明（利用一阶导数的差商定义与 Slutsky 定理）**：

**(1) 验证估计量的相合性（$\hat{\theta}_n \xrightarrow{P} \theta$）**：
由已知条件 $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \tau^2)$ 及本篇定理 1.2（依分布收敛必依概率有界），知 $\sqrt{n}(\hat{\theta}_n - \theta) = O_p(1)$。两边同除以 $\sqrt{n}$，立得 $\hat{\theta}_n - \theta = O_p(n^{-1/2}) = o_p(1)$，即 $\hat{\theta}_n \xrightarrow{P} \theta$。

**(2) 构造连续的差商余项函数**：
由函数 $g(x)$ 在点 $\theta$ 处可导的定义，差商极限满足 $\lim_{x\to\theta} \frac{g(x) - g(\theta)}{x - \theta} = g'(\theta)$。
据此定义一个在全实轴 $\mathbb{R}$ 上有定义、且在点 $x = \theta$ 处**连续**的差商误差函数 $h(x)$：

$$ h(x) := \begin{cases} \dfrac{g(x) - g(\theta)}{x - \theta} - g'(\theta), & x \neq \theta \\ 0, & x = \theta \end{cases} $$

显然有 $\lim_{x\to\theta} h(x) = 0 = h(\theta)$（即 $h(x)$ 在点 $\theta$ 处连续），且对一切 $x \in \mathbb{R}$ 恒有如下精确代数恒等式：

$$ g(x) - g(\theta) = \left[g'(\theta) + h(x)\right](x - \theta) $$

**(3) 代入随机变量并应用 Slutsky 定理**：
将 $x = \hat{\theta}_n$ 代入上式，并在等式两边同乘以 $\sqrt{n}$：

$$ \sqrt{n}\left(g(\hat{\theta}_n) - g(\theta)\right) = \underbrace{\left[g'(\theta) + h(\hat{\theta}_n)\right]}_{\text{随机系数项}} \cdot \underbrace{\sqrt{n}\left(\hat{\theta}_n - \theta\right)}_{\text{渐近正态项}} $$

* **考察随机系数项**：由 $\hat{\theta}_n \xrightarrow{P} \theta$ 及 $h(x)$ 在点 $\theta$ 处的连续性，应用连续映射定理得 $h(\hat{\theta}_n) \xrightarrow{P} h(\theta) = 0$，从而 $g'(\theta) + h(\hat{\theta}_n) \xrightarrow{P} g'(\theta)$。
* **考察整体乘积**：直接应用定理 2.1（**Slutsky 定理的乘法法则**），得：
  $$ \sqrt{n}\left(g(\hat{\theta}_n) - g(\theta)\right) \xrightarrow{d} g'(\theta) \cdot N(0, \tau^2) = N\left(0,\ \tau^2\left[g'(\theta)\right]^2\right) \quad \blacksquare $$


<a id="multivariate-delta"></a>

## 3.2 多元一阶 Delta 方法

当待估参数为 $m$ 维向量 $\boldsymbol{\theta} \in \mathbb{R}^m$，且非线性变换为向量值映射 $g: \mathbb{R}^m \to \mathbb{R}^s$ 时，一阶导数自然推广为多元微积分中的 **Jacobi 矩阵（雅可比矩阵）**，一维的方差缩放则推广为协方差矩阵的**二次型合同变换**。

> **定理 3.2（多元一阶 Delta 方法）**
> 设 $m$ 维随机向量序列 $\{\hat{\boldsymbol{\theta}}_n\}$ 满足渐近多元正态性 $\sqrt{n}\left(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}\right) \xrightarrow{d} N_m\left(\mathbf{0},\ \boldsymbol{\Sigma}\right)$，其中 $\boldsymbol{\Sigma} \in \mathbb{R}^{m\times m}$ 为对称半正定协方差矩阵。设向量值映射 $g(\boldsymbol{\theta}) = \big(g_1(\boldsymbol{\theta}), \dots, g_s(\boldsymbol{\theta})\big)^\top: \mathbb{R}^m \to \mathbb{R}^s$ 在点 $\boldsymbol{\theta}$ 处可微，记其在点 $\boldsymbol{\theta}$ 处的 **Jacobi** 矩阵为：
> 
> $$ \mathbf{G} = \left.\frac{\partial g(\mathbf{x})}{\partial \mathbf{x}^\top}\right\vert _{\mathbf{x}=\boldsymbol{\theta}} = \begin{pmatrix} \dfrac{\partial g_1}{\partial \theta_1} & \cdots & \dfrac{\partial g_1}{\partial \theta_m} \\ \vdots & \ddots & \vdots \\ \dfrac{\partial g_s}{\partial \theta_1} & \cdots & \dfrac{\partial g_s}{\partial \theta_m} \end{pmatrix}_{s\times m} $$
> 
> 则变换后的随机向量序列满足：
> 
> $$ \sqrt{n}\left(g(\hat{\boldsymbol{\theta}}_n) - g(\boldsymbol{\theta})\right) \xrightarrow{d} N_s\left(\mathbf{0},\ \mathbf{G}\boldsymbol{\Sigma}\mathbf{G}^\top\right) $$

**证明**：

**(1) 多元函数的一阶线性展开与余项控制**：
由多元向量值函数在点 $\boldsymbol{\theta}$ 处可微的定义，当 $\mathbf{x} \to \boldsymbol{\theta}$ 时，有 $g(\mathbf{x}) - g(\boldsymbol{\theta}) = \mathbf{G}\cdot(\mathbf{x} - \boldsymbol{\theta}) + \mathbf{r}(\mathbf{x} - \boldsymbol{\theta})$，其中余项向量满足 $\lim_{\mathbf{x}\to\boldsymbol{\theta}} \frac{\Vert \mathbf{r}(\mathbf{x}-\boldsymbol{\theta})\Vert }{\Vert \mathbf{x}-\boldsymbol{\theta}\Vert } = 0$。代入 $\mathbf{x} = \hat{\boldsymbol{\theta}}_n$ 并同乘以 $\sqrt{n}$，得：

$$ \sqrt{n}\left(g(\hat{\boldsymbol{\theta}}_n) - g(\boldsymbol{\theta})\right) = \mathbf{G}\cdot\sqrt{n}\left(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}\right) + \sqrt{n}\,\mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}) $$

在 $\hat{\boldsymbol{\theta}}_n=\boldsymbol{\theta}$ 时，将以下余项比值定义为零。由于 $\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta} = O_p(n^{-1/2}) = o_p(1)$，由连续映射定理知 $\frac{\Vert \mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta})\Vert }{\Vert \hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}\Vert } = o_p(1)$。从而余项范数满足：

$$ \left\Vert \sqrt{n}\,\mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta})\right\Vert = \underbrace{\frac{\Vert \mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta})\Vert }{\Vert \hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}\Vert }}_{= o_p(1)} \cdot \underbrace{\left\Vert \sqrt{n}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta})\right\Vert }_{= O_p(1)} = o_p(1) \implies \sqrt{n}\,\mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}) \xrightarrow{P} \mathbf{0} $$

**(2) 应用 Cramér-Wold 定理与一元 Slutsky 定理**：
对任意固定的常数向量 $\mathbf{a} \in \mathbb{R}^s$，在展开式两边左乘 $\mathbf{a}^\top$ 转化为一维标量：

$$ \mathbf{a}^\top \sqrt{n}\left(g(\hat{\boldsymbol{\theta}}_n) - g(\boldsymbol{\theta})\right) = (\mathbf{a}^\top \mathbf{G})\sqrt{n}\left(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}\right) + \mathbf{a}^\top \sqrt{n}\,\mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}) $$

设极限随机向量为 $\mathbf{Z} \sim N_m(\mathbf{0}, \boldsymbol{\Sigma})$。由连续映射定理，第一项满足 $(\mathbf{a}^\top \mathbf{G})\sqrt{n}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}) \xrightarrow{d} (\mathbf{a}^\top \mathbf{G})\mathbf{Z}$；第二项满足 $\mathbf{a}^\top \sqrt{n}\,\mathbf{r}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}) \xrightarrow{P} 0$。
调用定理 2.1（**一元 Slutsky 定理的加法法则**），得：

$$ \mathbf{a}^\top \sqrt{n}\left(g(\hat{\boldsymbol{\theta}}_n) - g(\boldsymbol{\theta})\right) \xrightarrow{d} (\mathbf{a}^\top \mathbf{G})\mathbf{Z} \sim N\left(0,\ \mathbf{a}^\top (\mathbf{G}\boldsymbol{\Sigma}\mathbf{G}^\top)\mathbf{a}\right) $$

由第 I 篇的 **Cramér-Wold** 定理，即得 $\sqrt{n}\big(g(\hat{\boldsymbol{\theta}}_n) - g(\boldsymbol{\theta})\big) \xrightarrow{d} N_s(\mathbf{0}, \mathbf{G}\boldsymbol{\Sigma}\mathbf{G}^\top)$。$\blacksquare$

<a id="second-order-delta"></a>

## 3.3 二阶 Delta 方法：一阶导数退化为零的情形

定理 3.1 在 $g'(\theta)=0$ 时给出退化极限。
若真实参数 $\theta$ 恰好位于函数 $g(x)$ 的驻点（如极值点），使得 **$g'(\theta) = 0$**，则一阶泰勒展开中的线性主部完全消失。此时应用一阶公式，得到 $\sqrt{n}(g(\hat{\theta}_n) - g(\theta)) \xrightarrow{d} N(0, 0) = 0$ 的退化极限。若二阶导数不为零且原渐近方差为正，为了捕捉非退化的随机波动，就需要将泰勒展开推进至二阶项，用局部抛物线逼近原函数。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-delta-006.png)

如上图所示，当 $g'(\theta) = 0$ 且 $g''(\theta) > 0$ 时，函数在点 $\theta$ 处取得极小值。输入变量 $\hat{\theta}_n$ 在 $\theta$ 左右两侧的对称正态波动（$-\Delta$ 与 $+\Delta$），经过二阶项 $(\hat{\theta}_n - \theta)^2$ 的平方映射后，全部被向上折叠到 $y \ge g(\theta)$ 的单侧区域，从而使 $n$ 尺度下的输出误差极限由对称正态分布转变为**单侧有界、右偏的缩放卡方分布**。

> **定理 3.3（二阶 Delta 方法）**
> 设随机变量序列 $\{\hat{\theta}_n\}$ 满足 $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \tau^2)$。若函数 $g: \mathbb{R} \to \mathbb{R}$ 在点 $\theta$ 处二阶可导，且满足 $g'(\theta) = 0$ 及 $g''(\theta) \neq 0$，则有：
> 
> $$ n\left(g(\hat{\theta}_n) - g(\theta)\right) \xrightarrow{d} \frac{g''(\theta)}{2}\cdot\tau^2 \chi_1^2 $$
> 
> 其中 $\chi_1^2$ 表示自由度为 $1$ 的标准卡方分布。

**证明**：

**(1) 构造二阶泰勒展开**：
由函数 $g(x)$ 在点 $\theta$ 处二阶可导及 $g'(\theta) = 0$，根据带 **Peano** 余项的二阶泰勒公式，当 $x \to \theta$ 时有 $g(x) - g(\theta) = \frac{g''(\theta)}{2}(x - \theta)^2 + r(x - \theta)$，其中余项满足 $\lim_{x\to\theta} \frac{r(x - \theta)}{(x - \theta)^2} = 0$。
与定理 3.1 完全相同，定义在点 $x = \theta$ 处连续且满足 $h(\theta) = 0$ 的辅助函数：

$$ h(x) := \begin{cases} \dfrac{r(x - \theta)}{(x - \theta)^2}, & x \neq \theta \\ 0, & x = \theta \end{cases} $$

将 $x = \hat{\theta}_n$ 代入，并在等式两边同乘以 **$n$**（即 $(\sqrt{n})^2$）：

$$ n\left(g(\hat{\theta}_n) - g(\theta)\right) = \underbrace{\left[\frac{g''(\theta)}{2} + h(\hat{\theta}_n)\right]}_{\text{随机系数项}} \cdot \underbrace{\left[\sqrt{n}\left(\hat{\theta}_n - \theta\right)\right]^2}_{\text{渐近卡方项}} $$

**(2) 分别求极限并应用 Slutsky 定理**：
* **随机系数项**：由 $\hat{\theta}_n \xrightarrow{P} \theta$ 及连续映射定理知 $h(\hat{\theta}_n) \xrightarrow{P} 0$，故 $\frac{g''(\theta)}{2} + h(\hat{\theta}_n) \xrightarrow{P} \frac{g''(\theta)}{2}$。
* **渐近卡方项**：设 $Y = \tau Z \sim N(0, \tau^2)$，其中 $Z \sim N(0, 1)$。由于二次函数 $f(y) = y^2$ 处处连续，由连续映射定理得 $\big[\sqrt{n}(\hat{\theta}_n - \theta)\big]^2 \xrightarrow{d} Y^2 = \tau^2 Z^2 \sim \tau^2 \chi_1^2$。
* **应用 Slutsky 乘法法则**：两项相乘立得 $n\big(g(\hat{\theta}_n) - g(\theta)\big) \xrightarrow{d} \frac{g''(\theta)}{2}\cdot\tau^2 \chi_1^2$。$\blacksquare$

<a id="sigmoid-softmax"></a>

## 📖 例题：Laplace 预测近似中的概率误差传播

> **论文背景**：[**Laplace Redux — Effortless Bayesian Deep Learning**](https://arxiv.org/abs/2106.14806) (*NeurIPS* 2021)
>
> 训练好的分类网络通常使用一组固定权重进行预测：先输出各类别的 **logit**（归一化之前的实数得分），再通过 **Sigmoid** 或 **Softmax** 得到分类概率。如果考虑训练数据不足带来的参数不确定性，同一张图像在不同权重下的得分与预测概率也会有所不同。**Laplace** 近似在训练得到的权重附近构造高斯分布，用它近似参数后验，即给定训练数据后对参数取值的不确定性分布。
>
> **Daxberger** 等人讨论了如何将这种参数不确定性传递到网络预测中，包括只对最后一层分类头进行 **Laplace** 近似的做法。此时冻结特征提取网络，分类头的权重服从高斯分布；由于最后一层对权重是线性的，固定输入的 **logit** 也服从高斯分布。论文附录还用 **Delta** 方法近似 **Softmax** 的预测均值；下面考察同一预测结构下，得分波动经过概率输出层后如何改变。
>
> **题目描述**：考虑一个已经训练好的图像分类网络。固定一张图像 $x$ 及特征提取网络，只对最后一层分类头的权重采用高斯后验近似。用后验中心的权重计算基准预测；用从该近似后验中抽取的权重计算随机预测。
>
> 为分析小幅权重扰动，用 $n$ 控制后验的集中程度，设权重协方差按 $1/n$ 缩小。最后一层是线性映射，因此 **logit** 的波动也按 $n^{-1/2}$ 缩小。
>
> 1. **二分类**：网络输出一个得分，经过 **Sigmoid** 得到预测概率。记基准得分为 $z\in\mathbb R$，随机权重下的得分为 $\hat z_n$，假设 $\sqrt n(\hat z_n-z)\xrightarrow{d}N(0,\tau^2)$。令 $s(z)=1/(1+e^{-z})$，基准概率为 $p=s(z)$，随机预测概率为 $\hat p_n=s(\hat z_n)$。求 $\sqrt n(\hat p_n-p)$ 的极限分布。
> 2. **多分类**：设 $K\ge2$，网络输出 $K$ 个得分，并通过 **Softmax** 得到总和为 $1$ 的概率向量。记基准得分为 $\mathbf z\in\mathbb R^K$，随机权重下的得分为 $\hat{\mathbf z}_n$，假设 $\sqrt n(\hat{\mathbf z}_n-\mathbf z)\xrightarrow{d}N_K(\mathbf0,\Sigma_z)$。其中 $\Sigma_z$ 是半正定协方差矩阵。令 $\mathbf p=\operatorname{softmax}(\mathbf z)$、$\hat{\mathbf p}_n=\operatorname{softmax}(\hat{\mathbf z}_n)$。求概率误差的极限协方差；并证明它是奇异矩阵，即概率误差不能在整个 $K$ 维空间中自由变化。

**解答**：

**(1) 二分类：切线斜率决定误差缩放。**

**Sigmoid** 的导数为

$$ s'(z)=\frac{e^{-z}}{(1+e^{-z})^2}=p(1-p) $$

因此，当 **logit** 误差很小时，一阶近似给出

$$ \hat p_n-p\approx p(1-p)(\hat z_n-z) $$

概率误差等于 **logit** 误差乘以局部斜率。由定理 3.1（一元一阶 **Delta** 方法）：

$$ \sqrt n(\hat p_n-p)\xrightarrow{d}N\left(0,\tau^2[p(1-p)]^2\right) $$

**(2) 多分类：Jacobi 矩阵决定协方差变换。**

第 $i$ 类的概率为 $p_i=e^{z_i}/\sum_{k=1}^K e^{z_k}$。分别对各个 **logit** 求导：

$$ \frac{\partial p_i}{\partial z_j}=\begin{cases}p_i(1-p_i),&i=j\\-p_ip_j,&i\ne j\end{cases} $$

增加 $z_i$ 会提高第 $i$ 类的概率；增加其他类别的 **logit** 则会降低它。因此，一个 **logit** 的变化会同时影响多个类别，必须用整个 **Jacobi** 矩阵描述误差传播：

$$ G=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^T,\quad \hat{\mathbf p}_n-\mathbf p\approx G(\hat{\mathbf z}_n-\mathbf z) $$

这对应于二分类中的 $\Delta p\approx s'(z)\Delta z$，只是将标量斜率换成了矩阵。由定理 3.2（多元一阶 Delta 方法）：

$$ \sqrt n(\hat{\mathbf p}_n-\mathbf p)\xrightarrow{d}N_K(\mathbf0,G\Sigma_zG^T) $$

**为什么极限协方差是奇异的？** 因为预测概率的总和始终为 $1$，概率误差的各分量之和始终为零：

$$ \sum_{i=1}^K(\hat p_{n,i}-p_i)=0 $$

因此，概率误差只能在 $K-1$ 维子空间 $\{\mathbf u:\mathbf1^T\mathbf u=0\}$ 内波动，不可能在整个 $\mathbb R^K$ 中自由变化。代数上，由 $\mathbf1^T\mathbf p=1$ 及 $G^T=G$，有

$$ G\mathbf1=\mathbf p-\mathbf p(\mathbf p^T\mathbf1)=\mathbf0,\quad (G\Sigma_zG^T)\mathbf1=\mathbf0 $$

所以极限协方差至少有一个零特征值，秩至多为 $K-1$。
