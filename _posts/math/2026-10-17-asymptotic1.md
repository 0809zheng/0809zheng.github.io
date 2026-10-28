---
layout: post
title: '渐近理论I：大数定律与中心极限定理'
date: 2026-10-17
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-cover.png'
tags: 数学
pinned: false
---

> Asymptotic Theory I: Law of Large Numbers and Central Limit Theorem.

在经典数理统计的有限样本理论（**Finite-Sample Theory**）中，统计推断依赖于总体分布的精确先验假设。例如，当总体服从正态分布 $X_i \sim N(\mu, \sigma^2)$ 时，无论样本容量 $n$ 取何值，样本均值 $\bar{X}_n$ 的精确分布均为正态分布，样本方差的精确分布可由卡方分布刻画，进而可导出精确服从 $t(n-1)$ 分布的检验统计量。

然而，在实际的科学计算与工程建模中，精确有限样本理论面临两个根本性困难：
1. **总体分布的未知性与复杂性**：真实世界的数据分布往往呈现偏态、多峰或重尾特征，无法用单一的参数化分布族精确描述。
2. **解析分布的不可计算性（Intractability）**：即使总体分布形式已知，当统计量涉及非线性变换或隐式方程求解（如极大似然估计）时，有限样本容量 $n$ 下的精确概率密度函数通常不存在初等解析表达式。

**渐近理论（Asymptotic Theory，又称大样本理论 Large Sample Theory）**旨在解决上述问题。其基本研究范式是：考察当样本容量 $n \to \infty$ 时统计量序列的极限行为，并利用结构简单的极限分布去近似有限样本下的复杂真实分布。

作为该系列的第 **I** 篇，本文将系统阐述渐近理论的理论基石：**大数定律（LLN）**与**中心极限定理（CLT）**。

本文目录：
1. [依概率收敛与大数定律](#probability-convergence-lln)
   - [1.1 依概率收敛的定义](#conv-in-probability)
   - [1.2 概率论中的尾概率不等式](#tail-inequalities)
   - [1.3 弱大数定律](#weak-lln)
   - [1.4 强大数定律](#strong-lln)
2. [依分布收敛与中心极限定理](#distribution-convergence-clt)
   - [2.1 依分布收敛的定义](#conv-in-distribution)
   - [2.2 中心极限定理](#clt)
   - [2.3 两种收敛模式的关系证明](#convergence-relations)
3. [连续映射定理与随机渐近阶符号](#cmt-stochastic-order)
   - [3.1 连续映射定理](#continuous-mapping)
   - [3.2 随机渐近阶符号](#stochastic-order)
4. [深度学习中的应用](#dl-applications)
   - [例题 4.1 固定模型的经验风险收敛](#erm)
   - [例题 4.2 Mini-batch 梯度的高斯噪声近似](#minibatch)
   - [例题 4.3 无限宽神经网络的高斯过程极限](#infinite-width)


<a id="probability-convergence-lln"></a>

# 1. 依概率收敛与大数定律

<a id="conv-in-probability"></a>

## 1.1 依概率收敛的定义

在实分析中，确定性数列 $\{a_n\}$ 收敛于常数 $a$ 是指距离 $\vert a_n - a\vert $ 随 $n\to\infty$ 趋于零。对于随机变量序列 $\{X_n\}$，由于其取值具有随机性，需要通过事件发生的概率来度量其与极限目标 $X$ 的接近程度。

> **定义 1.1（依概率收敛）**
> 设 $\{X_n\}$ 与 $X$ 为定义在同一概率空间上的随机变量。若对任意正数 $\varepsilon > 0$，均有
> $$ \boxed{\lim_{n\to\infty} P\left(\vert X_n - X\vert > \varepsilon\right) = 0 \quad \left(\text{等价地，}\lim_{n\to\infty} P\left(\vert X_n - X\vert \le \varepsilon\right) = 1\right)} $$
> 则称序列 $\{X_n\}$ **依概率收敛（Convergence in Probability）**于 $X$，记作 $X_n \xrightarrow{P} X$。

在参数估计理论中，设 $\theta$ 为未知总体参数，$\hat{\theta}_n = \hat{\theta}(X_1, \dots, X_n)$ 为基于样本构造的估计量。若 $\hat{\theta}_n \xrightarrow{P} \theta$，则称 $\hat{\theta}_n$ 为参数 $\theta$ 的**弱相合估计量（Weakly Consistent Estimator）**。

<a id="tail-inequalities"></a>

## 1.2 概率论中的尾概率不等式

为了严格证明依概率收敛，需要将形如 $P(\vert X_n - X\vert > \varepsilon)$ 的尾概率与随机变量的矩（数学期望、方差）建立联系。以下两个基础不等式是证明大数定律的核心工具。

> **引理 1.1（Markov 不等式）**
> 设非负随机变量 $Y \ge 0$ 满足 $\mathbb{E}[Y]<\infty$，则对任意 $\varepsilon > 0$，有
> $$ \boxed{P(Y \ge \varepsilon) \le \frac{\mathbb{E}[Y]}{\varepsilon}} $$

**证明**：根据事件 $\{Y\ge\varepsilon\}$ 是否发生，分解数学期望：
$$ \mathbb{E}[Y]=\mathbb{E}\left[Y\mathbf{1}_{\{Y<\varepsilon\}}\right]+\mathbb{E}\left[Y\mathbf{1}_{\{Y\ge\varepsilon\}}\right] $$
第一项非负；第二项中 $Y\ge\varepsilon$，因此
$$ \mathbb{E}[Y]\ge\mathbb{E}\left[Y\mathbf{1}_{\{Y\ge\varepsilon\}}\right]\ge\varepsilon\mathbb{E}\left[\mathbf{1}_{\{Y\ge\varepsilon\}}\right]=\varepsilon P(Y\ge\varepsilon) $$
两边同除以 $\varepsilon>0$ 即得结论。该证明不要求 $Y$ 存在概率密度，因此同时适用于离散、连续与其他分布。$\blacksquare$

> **引理 1.2（Chebyshev 不等式）**
> 设随机变量 $X$ 的均值为 $\mu = \mathbb{E}[X]$，方差为 $\sigma^2 = \text{Var}(X) < \infty$。则对任意 $\varepsilon > 0$，有
> $$ \boxed{P\left(\vert X - \mu\vert \ge \varepsilon\right) \le \frac{\text{Var}(X)}{\varepsilon^2}} $$

**证明**：注意到事件 $\{\vert X - \mu\vert \ge \varepsilon\}$ 与事件 $\{(X - \mu)^2 \ge \varepsilon^2\}$ 完全等价。令非负随机变量 $Y = (X - \mu)^2$，对 $Y$ 应用引理 1.1（**Markov 不等式**），并代入 $\mathbb{E}[Y] = \mathbb{E}[(X-\mu)^2] = \text{Var}(X)$，可得：

$$ P\left(\vert X - \mu\vert \ge \varepsilon\right) = P\left((X - \mu)^2 \ge \varepsilon^2\right) \le \frac{\mathbb{E}\left[(X - \mu)^2\right]}{\varepsilon^2} = \frac{\text{Var}(X)}{\varepsilon^2} \quad \blacksquare $$

<a id="weak-lln"></a>

## 1.3 弱大数定律（Weak Law of Large Numbers）

**大数定律**揭示了算术平均值的稳定性：当试验次数无限增加时，样本均值依概率收敛于总体期望。下面依次讨论 Bernoulli、Chebyshev 与 Markov、Khinchin 三组经典结果，它们分别对应事件频率、二阶矩条件以及仅一阶矩存在的情形。

### 1.3.1 Bernoulli 弱大数定律（频率收敛于概率）

> **定理 1.1（Bernoulli 弱大数定律）**
> 在 $n$ 重独立重复试验中，设事件 $A$ 在每次试验中发生的概率为 $p \in (0, 1)$。定义第 $i$ 次试验的指示随机变量 $X_i = \mathbb{I}_A(\omega_i) \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$（事件发生取 $1$，不发生取 $0$），记 $S_n = \sum_{i=1}^n X_i$ 为事件 $A$ 发生的频数。则事件发生的频率依概率收敛于概率 $p$：
> $$ \boxed{\bar{X}_n = \frac{S_n}{n} \xrightarrow{P} p = \mu \quad (n \to \infty)} $$

**证明**：
由于 $X_i \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$，其总体均值为 $\mu = \mathbb{E}[X_i] = 1 \cdot p + 0 \cdot (1 - p) = p$，方差为 $\text{Var}(X_i) = p(1 - p)$。
从而样本均值（频率）$\bar{X}_n = \frac{S_n}{n} = \frac{1}{n}\sum_{i=1}^n X_i$ 的期望与方差分别为：

$$ \mathbb{E}[\bar{X}_n] = \mathbb{E}\left[\frac{S_n}{n}\right] = p, \quad \text{Var}(\bar{X}_n) = \text{Var}\left(\frac{S_n}{n}\right) = \frac{p(1 - p)}{n} $$

对任意给定的 $\varepsilon > 0$，将引理 **1.2（Chebyshev 不等式）**直接应用于随机变量 $\bar{X}_n = \frac{S_n}{n}$（并注意均值不等式 $p(1-p) \le \frac{1}{4}$），可得：

$$ 0 \le P\left(\left\vert \frac{S_n}{n} - p\right\vert \ge \varepsilon\right) \le \frac{\text{Var}(S_n / n)}{\varepsilon^2} = \frac{p(1 - p)}{n\varepsilon^2} \le \frac{1}{4n\varepsilon^2} $$

由夹逼准则（$n\to\infty$时$1/n \to 0$），立得 $\bar{X}_n = \frac{S_n}{n} \xrightarrow{P} p$。$\blacksquare$

作为大数定律家族的历史起点，**Bernoulli** 大数定律其实就是一般大数定律 $\bar{X}_n \xrightarrow{P} \mu$ 在 **0-1 分布（伯努利分布）**下的特例：当每次试验的随机变量 $X_i \in \{0, 1\}$ 仅用于指示某事件 $A$ 是否发生时，**样本均值 $\bar{X}_n$ 就退化为事件发生的频率 $\frac{S_n}{n}$，总体均值 $\mu$ 就退化为事件发生的概率 $p$**。

### 1.3.2 Chebyshev 与 Markov 弱大数定律（二阶矩存在情形）

> **定理 1.2（Chebyshev 与 Markov 弱大数定律）**
> 设随机变量序列 $X_1, X_2, \dots$ 具有有限二阶矩，记 $\mu_i = \mathbb{E}[X_i]$。
> 1. **Markov 条件（部分和方差条件）**：若部分和的方差满足
> $$ \lim_{n\to\infty} \frac{1}{n^2}\text{Var}\left(\sum_{i=1}^n X_i\right) = 0 $$
> 则有
> $$ \boxed{\bar{X}_n - \frac{1}{n}\sum_{i=1}^n \mu_i \xrightarrow{P} 0} $$
> 2. **Chebyshev 条件（两两不相关且方差一致有界）**：特别地，若 $X_1, \dots, X_n$ 两两不相关（$\text{Cov}(X_i, X_j) = 0,\ \forall i \neq j$），具有共同均值 $\mathbb{E}[X_i] = \mu$，且方差存在一致上界 $\text{Var}(X_i) \le C < \infty$，则
> $$ \boxed{\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu \quad (n \to \infty)} $$

**证明**：记 $\bar{\mu}_n = \frac{1}{n}\sum_{i=1}^n \mu_i = \mathbb{E}[\bar{X}_n]$。对任意给定的 $\varepsilon > 0$，将引理 **1.2（Chebyshev 不等式）** 应用于随机变量 $\bar{X}_n$，可得：

$$ 0 \le P\left(\left\vert \bar{X}_n - \bar{\mu}_n\right\vert \ge \varepsilon\right) \le \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{1}{n^2 \varepsilon^2}\text{Var}\left(\sum_{i=1}^n X_i\right) $$

在 **Markov** 条件下，令 $n \to \infty$，上式右端直接趋于 $0$，即得 $\bar{X}_n - \bar{\mu}_n \xrightarrow{P} 0$。

在 **Chebyshev** 条件下，由两两不相关性知方差可加，故有：

$$ \frac{1}{n^2}\text{Var}\left(\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) \le \frac{nC}{n^2} = \frac{C}{n} \xrightarrow{n\to\infty} 0 $$

从而 **Markov** 条件自然满足，且 $\bar{\mu}_n = \mu$，故 $\bar{X}_n \xrightarrow{P} \mu$。$\blacksquare$

### 1.3.3 Khinchin 弱大数定律（仅一阶矩存在情形）

在实际应用中，某些重尾分布（如自由度为 2 的 $t$ 分布，或尾指数 $\alpha\in(1,2]$ 的 **Pareto** 分布）的方差不存在（即 $\text{Var}(X_i) = \infty$），此时 **Chebyshev** 不等式失效。**Khinchin（辛钦）弱大数定律**证明了：对于独立同分布序列，无需假设方差存在，只要一阶矩（数学期望）有限，弱大数定律依然成立。

为了给出完整证明，引入概率论中的**特征函数（Characteristic Function）**工具。

> **预备知识：特征函数及其泰勒展开**
> 
> 设随机变量 $X$ 的特征函数定义为 $\varphi_X(t) = \mathbb{E}[e^{\mathrm{i}tX}] = \mathbb{E}[\cos(tX)] + \mathrm{i}\mathbb{E}[\sin(tX)]$，其中 $\mathrm{i} = \sqrt{-1}$，$t \in \mathbb{R}$。特征函数具有以下基本性质：
> 1. **线性变换性质**：$\varphi_{aX+b}(t) = e^{\mathrm{i}tb}\varphi_X(at)$。
> 2. **独立和性质**：若 $X_1, \dots, X_n$ 相互独立，则 $\varphi_{\sum_{i=1}^n X_i}(t) = \prod_{i=1}^n \varphi_{X_i}(t)$。
> 3. **可微性与泰勒展开**：若 $\mathbb{E}[\vert X\vert ^k] < \infty$，则 $\varphi_X(t)$ 在 $t=0$ 处 $k$ 阶连续可导，且 $\varphi_X^{(k)}(0) = \mathrm{i}^k \mathbb{E}[X^k]$。由带 **Peano** 余项的泰勒公式，当 $t \to 0$ 时：
>    * 若 $\mathbb{E}[\vert X\vert ] < \infty$ 且 $\mathbb{E}[X] = \mu$，则 $\varphi_X(t) = 1 + \mathrm{i}\mu t + o(t)$；
>    * 若 $\mathbb{E}[X^2] < \infty$ 且 $\mathbb{E}[X] = 0,\ \text{Var}(X) = \sigma^2$，则 $\varphi_X(t) = 1 - \frac{1}{2}\sigma^2 t^2 + o(t^2)$。

> **定理 1.3（Khinchin 弱大数定律）**
> 设 $X_1, X_2, \dots$ 独立同分布，且 $\mathbb{E}[\vert X_1\vert ] < \infty$，记 $\mathbb{E}[X_1] = \mu$，则有
> $$ \boxed{\bar{X}_n \xrightarrow{P} \mu \quad (n \to \infty)} $$

**证明**：
设 $X_i$ 的共同特征函数为 $\varphi(t) = \mathbb{E}[e^{\mathrm{i}tX_1}]$。由一阶矩存在性及带 **Peano** 余项的一阶泰勒展开，当 $t \to 0$ 时有：

$$ \varphi(t) = 1 + \mathrm{i}\mu t + o(t) $$

考察样本均值 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ 的特征函数 $\varphi_{\bar{X}_n}(t)$。由独立同分布性质与特征函数的缩放性质，对任意固定的 $t \in \mathbb{R}$，当 $n \to \infty$ 时 $\frac{t}{n} \to 0$，故有：

$$
\begin{aligned}
\varphi_{\bar{X}_n}(t) &= \mathbb{E}\left[\exp\left(\mathrm{i}t \cdot \frac{1}{n}\sum_{i=1}^n X_i\right)\right] \\&= \left[\varphi\left(\frac{t}{n}\right)\right]^n \\&= \left[1 + \frac{\mathrm{i}\mu t}{n} + o\left(\frac{1}{n}\right)\right]^n
\end{aligned}
$$

由基本极限公式 $\lim_{n\to\infty}\left(1 + \frac{z}{n} + o\left(\frac{1}{n}\right)\right)^n = e^z$，立即得到：

$$ \lim_{n\to\infty} \varphi_{\bar{X}_n}(t) = e^{\mathrm{i}\mu t}, \quad \forall t \in \mathbb{R} $$

注意到 $e^{\mathrm{i}\mu t}$ 恰好是退化于常数 $\mu$ 的随机变量的特征函数。由下文的 **Lévy 连续性定理**（见定理 2.2）可知，样本均值序列依分布收敛于常数 $\mu$，即 $\bar{X}_n \xrightarrow{d} \mu$。再由“依分布收敛于常数等价于依概率收敛于该常数”（将在第 2.3 节证明），立得 $\bar{X}_n \xrightarrow{P} \mu$。$\blacksquare$

<a id="strong-lln"></a>

## 1.4 强大数定律（Strong Law of Large Numbers）

弱大数定律（$\bar{X}_n \xrightarrow{P} \mu$）仅保证在单个固定的大型时刻 $n$，样本均值偏离 $\mu$ 的概率很小；但它并不排除随着时间 $n$ 不断推移，样本轨道在未来的某些时刻再次发生大幅度偏离的可能性。为了刻画**整条样本轨道随 $n \to \infty$ 的收敛性**，需要引入更强的收敛概念。

> **定义 1.2（几乎处处收敛 / 以概率 1 收敛）**
> 设 $\{X_n\}$ 与 $X$ 为定义在同一概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机变量。若满足
> $$ \boxed{P\left(\left\{\omega \in \Omega:\ \lim_{n\to\infty} X_n(\omega) = X(\omega)\right\}\right) = 1} $$
> 则称序列 $\{X_n\}$ **几乎处处收敛（Almost Sure Convergence）**于 $X$，记作 $X_n \xrightarrow{a.s.} X$。

**Borel-Cantelli 引理**指出只要偏差事件级数求和收敛，就必能推出几乎处处收敛。

> **引理 1.3（第一 Borel-Cantelli 引理的收敛判据）**
> 对任意给定的 $\varepsilon > 0$，记偏差事件为 $A_n(\varepsilon) = \{\vert X_n - X\vert > \varepsilon\}$。若对一切 $\varepsilon > 0$，概率级数均收敛：
> $$ \sum_{n=1}^\infty P\left(\vert X_n - X\vert > \varepsilon\right) < \infty $$
> 则必有 $X_n \xrightarrow{a.s.} X$，无须假定这些事件相互独立。

**证明**：根据实分析极限定义，$X_n(\omega) \nrightarrow X(\omega)$ 发生，当且仅当存在某个正有理数 $\varepsilon > 0$，使得对于无穷多个下标 $n$ 均有 $\vert X_n(\omega) - X(\omega)\vert > \varepsilon$（记该上极限事件为 $A(\varepsilon) = \bigcap_{m=1}^\infty \bigcup_{n=m}^\infty A_n(\varepsilon)$）。
由概率的有限可加性与可列次可加性，对任意正整数 $m \ge 1$，有：

$$ 0 \le P\big(A(\varepsilon)\big) = P\left(\bigcap_{k=1}^\infty \bigcup_{n=k}^\infty A_n(\varepsilon)\right) \le P\left(\bigcup_{n=m}^\infty A_n(\varepsilon)\right) \le \sum_{n=m}^\infty P\big(A_n(\varepsilon)\big) $$

由于收敛级数 $\sum_{n=1}^\infty P(A_n(\varepsilon)) < \infty$ 的余项当 $m \to \infty$ 时必趋于零：$\lim_{m\to\infty} \sum_{n=m}^\infty P(A_n(\varepsilon)) = 0$，故 $P(A(\varepsilon)) = 0$。对一切正有理数 $\varepsilon \in \mathbb{Q}^+$ 取可数并集，总不收敛事件的概率仍为 $0$，即 $P(\lim_{n\to\infty} X_n = X) = 1$。$\blacksquare$

> **定理 1.4（Kolmogorov 强大数定律）**
> 设 $X_1, X_2, \dots$ 为独立同分布的随机变量序列，$\mu$ 为有限常数。
> 1. **一阶矩条件**：样本均值满足 $\bar{X}_n \xrightarrow{a.s.} \mu$ 的充要条件是 $\mathbb{E}[\vert X_1\vert ] < \infty$ 且 $\mathbb{E}[X_1] = \mu$：
> $$ \boxed{\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{a.s.} \mu \quad \Longleftrightarrow \quad \mathbb{E}\big[\vert X_1\vert \big] < \infty \ \text{且}\ \mathbb{E}[X_1] = \mu} $$
> 2. **四阶矩条件下的特例**：若 $\mathbb{E}[X_1]=\mu$ 且四阶中心矩有限 $\mathbb{E}\big[(X_1 - \mu)^4\big] = \gamma_4 < \infty$，可用四阶矩展开与 Borel-Cantelli 引理直接证出 $\bar{X}_n \xrightarrow{a.s.} \mu$。

下面给出结论 2 的完整证明；结论 1 的一般证明还需要截断与部分和最大不等式等工具，不由下面的四阶矩论证直接推出。

**证明（利用四阶矩展开与引理 1.3 Borel-Cantelli 引理）**：
不失一般性，令中心化变量 $Y_i = X_i - \mu$，则 $Y_1, \dots, Y_n$ 独立同分布，满足 $\mathbb{E}[Y_i] = 0$，方差 $\mathbb{E}[Y_i^2] = \sigma^2 < \infty$，四阶矩 $\mathbb{E}[Y_i^4] = \gamma_4 < \infty$。记部分和为 $S_n = \sum_{i=1}^n Y_i$。
1. **展开四次方 $\mathbb{E}[S_n^4]$**：
   将 $\left(\sum_{i=1}^n Y_i\right)^4 = \sum_{i,j,k,l} Y_i Y_j Y_k Y_l$ 按下标重合情况分类求期望。
   注意到只要某一个下标单独出现一次（如 $Y_1 Y_2^3$ 或 $Y_1 Y_2 Y_3^2$ 或 $Y_1 Y_2 Y_3 Y_4$），由独立性及 **$\mathbb{E}[Y_1] = 0$**，其数学期望全部恒等于 **$0$**！
   因此，展开式中只有两类项的期望不为零：
   * 第一类：四个下标全相同 $\sum_{i=1}^n \mathbb{E}[Y_i^4] = n\gamma_4$（共 $n$ 项）；
   * 第二类：两对下标分别相同 $\binom{4}{2}\sum_{1 \le i < j \le n} \mathbb{E}[Y_i^2 Y_j^2] = 6 \cdot \frac{n(n-1)}{2}\sigma^4 = 3n(n-1)\sigma^4$。
   由 Cauchy-Schwarz 不等式知 $\sigma^4 = (\mathbb{E}[Y_1^2])^2 \le \mathbb{E}[Y_1^4] = \gamma_4$，故有：
   $$ \mathbb{E}\left[S_n^4\right] = n\gamma_4 + 3n(n-1)\sigma^4 \le n\gamma_4 + 3n(n-1)\gamma_4 \le 3n^2 \gamma_4 $$
2. **应用四阶 Markov 不等式并求和**：
   对任意 $\varepsilon > 0$，将引理 1.1（Markov 不等式）应用于非负随机变量 $(\bar{X}_n - \mu)^4 = \frac{S_n^4}{n^4}$：
   $$ P\left(\vert \bar{X}_n - \mu\vert > \varepsilon\right) = P\left(\left(\bar{X}_n - \mu\right)^4 > \varepsilon^4\right) \le \frac{\mathbb{E}[S_n^4]}{n^4 \varepsilon^4} \le \frac{3n^2 \gamma_4}{n^4 \varepsilon^4} = \frac{3\gamma_4}{\varepsilon^4} \cdot \frac{1}{n^2} $$
   将上式对 $n = 1, 2, \dots, \infty$ 求和。由于 $p$-级数 $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6} < \infty$ 收敛（注意：若只用二阶矩 Chebyshev 不等式，得到的是发散的调和级数 $\sum \frac{1}{n} = \infty$，这就是为什么要升到四阶矩的原因！），立得：
   $$ \sum_{n=1}^\infty P\left(\vert \bar{X}_n - \mu\vert > \varepsilon\right) \le \frac{3\gamma_4}{\varepsilon^4}\sum_{n=1}^\infty \frac{1}{n^2} < \infty $$
   由引理 1.3（**Borel-Cantelli 引理**），得到 $\bar{X}_n \xrightarrow{a.s.} \mu$。$\blacksquare$

<a id="distribution-convergence-clt"></a>


# 2. 依分布收敛与中心极限定理

<a id="conv-in-distribution"></a>

## 2.1 依分布收敛的定义

大数定律表明估计误差满足 $\bar{X}_n - \mu \xrightarrow{P} 0$，但这仅给出了点估计的收敛性。为了进一步构造置信区间或进行假设检验，必须刻画估计量在逼近真值过程中的分布形态。

> **定义 2.1（依分布收敛）**
> 设 $\{X_n\}$ 为随机变量序列，其累积分布函数（**CDF**）为 $F_{X_n}(x) = P(X_n \le x)$；设 $X$ 的分布函数为 $F_X(x) = P(X \le x)$。若在 $F_X$ 的所有连续点 $x \in C(F_X)$ 上，均有
> $$ \boxed{\lim_{n\to\infty} F_{X_n}(x) = F_X(x)} $$
> 则称序列 $\{X_n\}$ **依分布收敛（Convergence in Distribution，又称弱收敛 Weak Convergence）**于 $X$，记作 $X_n \xrightarrow{d} X$。

> **注（为何仅要求在分布函数的连续点上收敛）**
> 考察确定性退化序列 $X_n \equiv \frac{1}{n}$，其极限显然应当为常数 $X \equiv 0$。写出两者的分布函数：
> 
> $$ F_{X_n}(x) = \begin{cases} 0, & x < \frac{1}{n} \\ 1, & x \ge \frac{1}{n} \end{cases}, \qquad F_X(x) = \begin{cases} 0, & x < 0 \\ 1, & x \ge 0 \end{cases} $$
> 
> ![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-002.png)
> 如上图所示：
> * 对任意 $x < 0$，当 $n$ 充分大时 $\frac{1}{n} > 0 > x$，故 $\lim_{n\to\infty} F_{X_n}(x) = 0 = F_X(x)$；
> * 对任意 $x > 0$，当 $n > \frac{1}{x}$ 时 $\frac{1}{n} < x$，故 $\lim_{n\to\infty} F_{X_n}(x) = 1 = F_X(x)$；
> * 唯独在 $F_X(x)$ 的**跳跃间断点 $x = 0$** 处，由于 $\frac{1}{n} > 0$ 恒成立，导致对任意有限的 $n$ 均有 $F_{X_n}(0) = P(\frac{1}{n} \le 0) = 0$，从而 $\lim_{n\to\infty} F_{X_n}(0) = 0 \neq F_X(0) = 1$。
> 
> 由此可见，阶梯函数在水平平移逼近极限位置时，在跳跃边界点处可能出现极限值与函数值不相等的现象。由于单调有界函数的不连续点至多只有可数个，剔除不连续点既消除了这种边界伪反例，又唯一确定了极限分布函数。

在高等概率论中，判断依分布收敛的等价工具有以下两个基本定理：

> **定理 2.1（Portmanteau 定理）**
> 对于随机变量序列 $\{X_n\}$ 与 $X$，下列命题完全等价：
> 1. $X_n \xrightarrow{d} X$；
> 2. 对任意有界连续函数 $h: \mathbb{R} \to \mathbb{R}$，有 $\lim_{n\to\infty} \mathbb{E}[h(X_n)] = \mathbb{E}[h(X)]$；
> 3. 对任意闭集 $C \subset \mathbb{R}$，有 $\limsup_{n\to\infty} P(X_n \in C) \le P(X \in C)$；
> 4. 对任意开集 $G \subset \mathbb{R}$，有 $\liminf_{n\to\infty} P(X_n \in G) \ge P(X \in G)$。

> **定理 2.2（Lévy 连续性定理）**
> 设 $X_n$ 的特征函数为 $\varphi_{X_n}(t)$，$X$ 的特征函数为 $\varphi_X(t)$。则 $X_n \xrightarrow{d} X$ 的充要条件是：对任意 $t \in \mathbb{R}$，均有 $\lim_{n\to\infty} \varphi_{X_n}(t) = \varphi_X(t)$。

<a id="clt"></a>

## 2.2 中心极限定理（Central Limit Theorem）

大数定律指出样本均值的误差满足 $\bar{X}_n - \mu \xrightarrow{P} 0$。若直接考察 $\bar{X}_n - \mu$ 的极限分布，只能得到退化于 $0$ 的单点分布。为了观察到非退化的误差分布形态，需要寻找恰当的放大因子 $a_n \to \infty$，使得 $a_n(\bar{X}_n - \mu)$ 的方差稳定在非零常数。

注意到 $\text{Var}\big(a_n(\bar{X}_n - \mu)\big) = a_n^2 \cdot \frac{\sigma^2}{n}$。要使该方差不随 $n\to\infty$ 退化为 $0$ 或发散至 $\infty$，放大因子必须与 $\sqrt{n}$ 同阶，即 **$a_n \propto \sqrt{n}$**；通常取 $a_n = \sqrt{n}$，使极限方差恰为 $\sigma^2$。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-003.png)

### 2.2.1 独立同分布的中心极限定理（Lindeberg-Lévy）

> **定理 2.3（Lindeberg-Lévy 中心极限定理）**
> 设 $X_1, X_2, \dots$ 独立同分布，具有均值 $\mathbb{E}[X_i] = \mu$ 与有限的正方差 $\text{Var}(X_i) = \sigma^2 \in (0, \infty)$。则当 $n \to \infty$ 时，
> $$ \boxed{Z_n = \frac{\sqrt{n}\left(\bar{X}_n - \mu\right)}{\sigma} \xrightarrow{d} Z \sim N(0, 1)} $$
> 等价地，$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$。

**证明（利用特征函数与二阶泰勒展开）**：

**(1) 标准化随机变量**：令 $Y_i = \frac{X_i - \mu}{\sigma}$。由独立同分布假设知，$Y_1, \dots, Y_n$ 亦独立同分布，且满足：

$$ \mathbb{E}[Y_i] = 0, \qquad \text{Var}(Y_i) = \mathbb{E}[Y_i^2] = 1 $$

此时，目标统计量 $Z_n$ 可表示为标准化变量的算术和：

$$ Z_n = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} = \frac{1}{\sqrt{n}}\sum_{i=1}^n \left(\frac{X_i - \mu}{\sigma}\right) = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i $$

**(2) 对特征函数作二阶泰勒展开**：设 $Y_i$ 的共同特征函数为 $\varphi_Y(t) = \mathbb{E}[e^{\mathrm{i}tY_1}]$。由于二阶矩 $\mathbb{E}[Y_1^2] = 1 < \infty$，特征函数 $\varphi_Y(t)$ 在 $t=0$ 处二阶连续可导，且有：

$$ \varphi_Y(0) = 1, \quad \varphi_Y'(0) = \mathrm{i}\mathbb{E}[Y_1] = 0, \quad \varphi_Y''(0) = \mathrm{i}^2 \mathbb{E}[Y_1^2] = -1 $$

由带 **Peano** 余项的二阶泰勒公式，当 $s \to 0$ 时：

$$ \varphi_Y(s) = \varphi_Y(0) + \varphi_Y'(0)s + \frac{\varphi_Y''(0)}{2}s^2 + o(s^2) = 1 - \frac{1}{2}s^2 + o(s^2) $$

**(3) 计算 $Z_n$ 的特征函数并求极限**：利用独立和的特征函数性质，将 $s = \frac{t}{\sqrt{n}}$ 代入上式。对任意固定的 $t \in \mathbb{R}$，当 $n \to \infty$ 时 $\frac{t}{\sqrt{n}} \to 0$，从而有：

$$
\begin{aligned}
\varphi_{Z_n}(t) &= \mathbb{E}\left[\exp\left(\mathrm{i}t \cdot \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i\right)\right] \\&= \left[\varphi_Y\left(\frac{t}{\sqrt{n}}\right)\right]^n \\&= \left[1 - \frac{t^2}{2n} + o\left(\frac{t^2}{n}\right)\right]^n
\end{aligned}
$$

对任意固定的 $t$，余项满足 $o\left(\frac{t^2}{n}\right) = o\left(\frac{1}{n}\right)$。由重要极限公式，当 $n \to \infty$ 时：
   $$ \lim_{n\to\infty} \varphi_{Z_n}(t) = \lim_{n\to\infty} \left[1 + \frac{-t^2/2}{n} + o\left(\frac{1}{n}\right)\right]^n = e^{-t^2/2}, \quad \forall t \in \mathbb{R} $$

**(4) 应用 Lévy 连续性定理**：注意到 $e^{-t^2/2}$ 恰好是标准正态分布 $Z \sim N(0,1)$ 的特征函数。由定理 2.2（**Lévy 连续性定理**），得出 $Z_n \xrightarrow{d} Z \sim N(0,1)$。$\blacksquare$

### 2.2.2 二项分布的中心极限定理（De Moivre-Laplace）

作为定理 **2.3（Lindeberg-Lévy CLT）**在离散 **Bernoulli** 试验下的特例，**De Moivre-Laplace 定理** 是历史上第一个中心极限定理。

> **推论 2.1（De Moivre-Laplace 中心极限定理）**
> 设 $S_n \sim \text{Binomial}(n, p)$ 表示 $n$ 重独立 **Bernoulli** 试验中成功的次数，其中 $p\in(0,1)$ 固定。则当 $n \to \infty$ 时，
> $$ \boxed{\frac{S_n - np}{\sqrt{np(1 - p)}} \xrightarrow{d} N(0, 1) \quad (n \to \infty)} $$

**证明**：将 $S_n$ 表示为独立同分布变量之和 $S_n = \sum_{i=1}^n X_i$，其中 $X_i \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$，均值为 $\mu = p$，方差为 $\sigma^2 = p(1 - p) > 0$。直接套用定理 2.3，即得 $\frac{\sqrt{n}(\bar{X}_n - p)}{\sqrt{p(1-p)}} = \frac{S_n - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0, 1)$。$\blacksquare$

### 2.2.3 独立不同分布情形：Lyapunov 与 Lindeberg-Feller 中心极限定理

在定理 2.3 中，我们要求随机变量序列 $X_1, X_2, \dots, X_n$ 必须**同分布（Identically Distributed）**。然而，在回归分析（如 $Y_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i$ 中加权和 $\sum_{i=1}^n x_{ij}\varepsilon_i$，每个样本的特征权重 $x_{ij}$ 各不相同）以及分层抽样中，随机变量序列相互独立但**不同分布（Independent but Not Identically Distributed, i.n.i.d.）**。

设独立随机变量序列 $X_1, X_2, \dots, X_n$ 的均值分别为 $\mu_i = \mathbb{E}[X_i]$，方差分别为 $\sigma_i^2 = \text{Var}(X_i) > 0$。记部分和的总方差为：

$$ B_n^2 := \text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \sigma_i^2 $$

在什么条件下，标准化部分和 $Z_n = \frac{1}{B_n}\sum_{i=1}^n (X_i - \mu_i)$ 依然收敛于标准正态分布 $N(0, 1)$？
直观上，如果标准化总和主要由某个非正态变量决定，其余项不能提供有效的平均效应，就不能期待由 CLT 导出正态极限。因此，在下面的独立序列定理中，核心条件是每个单项的相对方差趋于零，并控制超过总标准差一定比例的尾部贡献。

> **定理 2.3A（Lyapunov 中心极限定理）**
> 设独立随机变量序列 $\{X_i\}$ 存在 $2 + \delta$ 阶绝对中心矩，其中 $\delta > 0$。若满足 **Lyapunov 条件**：
> $$ \boxed{\lim_{n\to\infty} \frac{1}{B_n^{2+\delta}}\sum_{i=1}^n \mathbb{E}\left[\left\vert X_i - \mu_i\right\vert ^{2+\delta}\right] = 0, \qquad B_n = \sqrt{\sum_{i=1}^n \sigma_i^2}} $$
> 则当 $n \to \infty$ 时，
> $$ \boxed{Z_n = \frac{\sum_{i=1}^n (X_i - \mu_i)}{\sqrt{\sum_{i=1}^n \sigma_i^2}} \xrightarrow{d} N(0, 1)} $$

**证明（以最常用的三阶矩 $\delta = 1$ 为例，利用特征函数高阶余项界）**：
令中心化变量 $Y_i = X_i - \mu_i$，则 $\mathbb{E}[Y_i] = 0,\ \mathbb{E}[Y_i^2] = \sigma_i^2$，记三阶绝对矩为 $\rho_i^3 = \mathbb{E}[\vert Y_i\vert ^3]$。
1. **利用复指数泰勒余项不等式展开单变量特征函数**：
   由微积分复指数泰勒展开不等式：对任意实数 $u \in \mathbb{R}$，恒有 $\left\vert e^{\mathrm{i}u} - \left(1 + \mathrm{i}u - \frac{1}{2}u^2\right)\right\vert \le \frac{1}{6}\vert u\vert ^3$。
   令 $u = \frac{t}{B_n}Y_i$，两边取数学期望，并利用 $\mathbb{E}[\vert \cdot\vert ] \ge \vert \mathbb{E}[\cdot]\vert $，可得第 $i$ 个变量的特征函数 $\varphi_{Y_i}\big(\frac{t}{B_n}\big)$ 的精确二阶展开：
   $$ \varphi_{Y_i}\left(\frac{t}{B_n}\right) = \mathbb{E}\left[\exp\left(\mathrm{i}\frac{t}{B_n}Y_i\right)\right] = 1 - \frac{t^2 \sigma_i^2}{2 B_n^2} + R_{n, i}(t) $$
   其中余项严格受控于三阶矩：$\big\vert R_{n, i}(t)\big\vert \le \frac{\vert t\vert ^3}{6 B_n^3}\mathbb{E}\big[\vert Y_i\vert ^3\big]$。
2. **利用 Lyapunov 条件证明所有余项之和趋于零**：
   对 $i = 1, \dots, n$ 将余项绝对值求和，并代入 $\delta = 1$ 的 Lyapunov 条件：
   $$ \sum_{i=1}^n \big\vert R_{n, i}(t)\big\vert \le \frac{\vert t\vert ^3}{6} \cdot \left(\frac{1}{B_n^3}\sum_{i=1}^n \mathbb{E}\big[\vert Y_i\vert ^3\big]\right) \xrightarrow{n\to\infty} 0 $$
   同时，由于 $\sigma_i^3 = (\mathbb{E}[Y_i^2])^{3/2} \le \mathbb{E}[\vert Y_i\vert ^3]$，知最大单步方差占比亦趋于零：$\max_{1\le i\le n} \frac{\sigma_i^2}{B_n^2} \le \left(\frac{1}{B_n^3}\sum_{i=1}^n \mathbb{E}[\vert Y_i\vert ^3]\right)^{2/3} \to 0$。
3. **取对数求和导出高斯特征函数**：
   令 $z_{n,i}=-t^2\sigma_i^2/(2B_n^2)+R_{n,i}(t)$。由第二步，$\max_i\vert z_{n,i}\vert\to0$，且 $\sum_i\vert z_{n,i}\vert\le t^2/2+\sum_i\vert R_{n,i}(t)\vert=O(1)$，因此
   $$ \sum_{i=1}^n\vert z_{n,i}\vert^2\le\left(\max_{1\le i\le n}\vert z_{n,i}\vert\right)\sum_{i=1}^n\vert z_{n,i}\vert\to0 $$
   对充分大的 $n$，各 $1+z_{n,i}$ 均位于 $1$ 的邻域内，可使用该邻域中的复对数展开 $\ln(1+z)=z+O(\vert z\vert^2)$。由独立性，
   $$ \varphi_{Z_n}(t)=\exp\left(\sum_{i=1}^n\ln(1+z_{n,i})\right),\qquad \sum_{i=1}^n\ln(1+z_{n,i})=-\frac{t^2}{2}+\sum_{i=1}^nR_{n,i}(t)+O\left(\sum_{i=1}^n\vert z_{n,i}\vert^2\right)\to-\frac{t^2}{2} $$
   从而 $\lim_{n\to\infty} \varphi_{Z_n}(t) = e^{-t^2/2}$，由 **Lévy 连续性定理**得 $Z_n \xrightarrow{d} N(0, 1)$。$\blacksquare$

仅假定二阶矩有限时，**Lindeberg-Feller 中心极限定理**通过截断尾方差刻画渐近正态性与单项方差可忽略之间的关系。

> **定理 2.3B（Lindeberg-Feller 中心极限定理）**
> 设 $X_1,\dots,X_n$ 相互独立，方差有限，$\mu_i=\mathbb{E}[X_i]$、$\sigma_i^2=\text{Var}(X_i)$，$B_n^2=\sum_{i=1}^n\sigma_i^2>0$。则下列两个命题等价：
> 1. **Lindeberg 条件**：对任意 $\varepsilon>0$，
> $$ \boxed{\lim_{n\to\infty} \frac{1}{B_n^2}\sum_{i=1}^n \mathbb{E}\Big[(X_i - \mu_i)^2 \cdot \mathbb{I}\big(\vert X_i - \mu_i\vert > \varepsilon B_n\big)\Big] = 0} $$
> 2. **渐近正态且单项方差可忽略**：$Z_n = \frac{1}{B_n}\sum_{i=1}^n (X_i - \mu_i) \xrightarrow{d} N(0, 1)$，且 $\max_{1\le i\le n}\sigma_i^2/B_n^2\to0$。

该定理在此作为独立不同分布 CLT 的一般结果列出，不展开完整证明；上述 Lyapunov 三阶矩论证给出了一个常用且可直接验证的充分条件。

<a id="convergence-relations"></a>

## 2.3 两种收敛模式的关系证明

在渐近理论中，依概率收敛与依分布收敛存在紧密的逻辑关联。本节采用分布函数定义与微积分夹逼准则，给出两者关系的完整证明。

> **定理 2.4（依概率收敛蕴含依分布收敛）**
> 设 $\{X_n\}$ 与 $X$ 为同一概率空间上的随机变量。若 $X_n \xrightarrow{P} X$，则必有 $X_n \xrightarrow{d} X$。

**证明**：
设 $x$ 为极限分布函数 $F_X$ 的任意连续点，需证 $\lim_{n\to\infty} F_{X_n}(x) = F_X(x)$。
任取 $\varepsilon > 0$。考察事件 $\{X_n \le x\}$，依据事件 $\{\vert X_n - X\vert \le \varepsilon\}$ 是否发生，将其分解为两个互斥事件之并：

$$ \{X_n \le x\} = \left(\{X_n \le x\} \cap \{\vert X_n - X\vert \le \varepsilon\}\right) \cup \left(\{X_n \le x\} \cap \{\vert X_n - X\vert > \varepsilon\}\right) $$

* 对于第一个交事件：若 $X_n \le x$ 且 $\vert X_n - X\vert \le \varepsilon$ 同时成立，由三角不等式必有 $X = X_n + (X - X_n) \le x + \varepsilon$。故该交事件包含于事件 $\{X \le x + \varepsilon\}$。
* 对于第二个交事件：直接利用集合包含关系，它显然包含于其中一个事件 $\{\vert X_n - X\vert > \varepsilon\}$。

由概率的有限可加性与单调性，得到 $F_{X_n}(x)$ 的**上界不等式**：

$$
\begin{aligned}
F_{X_n}(x) = P(X_n \le x) &\le P(X \le x + \varepsilon) + P(\vert X_n - X\vert > \varepsilon) \\&= F_X(x + \varepsilon) + P(\vert X_n - X\vert > \varepsilon)
\end{aligned}
$$

完全对称地，考察事件 $\{X \le x - \varepsilon\}$:

$$ \{X \le x - \varepsilon\} = \left(\{X \le x - \varepsilon\} \cap \{\vert X_n - X\vert \le \varepsilon\}\right) \cup \left(\{X \le x - \varepsilon\} \cap \{\vert X_n - X\vert > \varepsilon\}\right) $$

* 对于第一个交事件：若 $X \le x - \varepsilon$ 且 $\vert X_n - X\vert \le \varepsilon$ 同时成立，由三角不等式必有 $X_n = X + (X_n - X) \le x$。故该交事件包含于事件 $\{X_n \le x\}$。
* 对于第二个交事件：直接利用集合包含关系，它显然包含于其中一个事件 $\{\vert X_n - X\vert \gt \varepsilon\}$。

同理可得：

$$
\begin{aligned}F_X(x - \varepsilon) = P(X \le x - \varepsilon) &\le P(X_n \le x) + P(\vert X_n - X\vert > \varepsilon) \\&= F_{X_n}(x) + P(\vert X_n - X\vert > \varepsilon)
\end{aligned}
$$

将上式移项即得 $F_{X_n}(x)$ 的**下界不等式**。综合上下界，记尾概率误差项为 $\delta_n(\varepsilon) = P(\vert X_n - X\vert > \varepsilon)$，有：

$$ F_X(x - \varepsilon) - \delta_n(\varepsilon) \le F_{X_n}(x) \le F_X(x + \varepsilon) + \delta_n(\varepsilon) $$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-004.png)

现在执行二次极限过程：
1. **固定 $\varepsilon > 0$，令 $n \to \infty$**：由已知条件 $X_n \xrightarrow{P} X$，有 $\lim_{n\to\infty} \delta_n(\varepsilon) = 0$。对不等式链取上极限与下极限，得：
   $$ F_X(x - \varepsilon) \le \liminf_{n\to\infty} F_{X_n}(x) \le \limsup_{n\to\infty} F_{X_n}(x) \le F_X(x + \varepsilon) $$
2. **令 $\varepsilon \to 0^+$**：由于 $x$ 是 $F_X$ 的连续点，故有 $\lim_{\varepsilon\to 0^+} F_X(x - \varepsilon) = F_X(x)$ 以及 $\lim_{\varepsilon\to 0^+} F_X(x + \varepsilon) = F_X(x)$。由夹逼准则，上下极限相等且均为 $F_X(x)$，即 $\lim_{n\to\infty} F_{X_n}(x) = F_X(x)$。$\blacksquare$

> **注（逆命题不成立的反例）**：依分布收敛通常不能推出依概率收敛。例如，设随机变量 $Z \sim N(0,1)$，定义序列 $X_n \equiv -Z$ 且令极限目标 $X = Z$。由于对称性，$X_n$ 与 $X$ 同服从标准正态分布，故 $F_{X_n}(x) = F_X(x)$ 恒成立，显然有 $X_n \xrightarrow{d} X$。然而：
> $$ P(\vert X_n - X\vert > \varepsilon) = P(\vert -Z - Z\vert > \varepsilon) = P\left(\vert Z\vert > \frac{\varepsilon}{2}\right) > 0 $$
> 该概率是一个与 $n$ 无关的正数，不随 $n\to\infty$ 趋于 $0$，故 $X_n$ 不依概率收敛于 $X$。

然而，当极限目标退化为确定性常数时，上述逆命题成立：

> **定理 2.5（退化极限下的等价性）**
> 设 $c$ 为常数，则 $X_n \xrightarrow{d} c \iff X_n \xrightarrow{P} c$。

**证明**：其中一个方向（$\Leftarrow$）已由定理 2.4 给出，仅需证明另一方向（$\Rightarrow$）。
设 $X_n \xrightarrow{d} c$，常数 $c$ 对应的退化分布函数为：

$$ F_c(x) = \begin{cases} 0, & x < c \\ 1, & x \ge c \end{cases} $$

显然，除点 $x = c$ 外，实轴上其余所有点均为 $F_c(x)$ 的连续点。对任意给定的 $\varepsilon > 0$，取 $x_1 = c - \varepsilon < c$ 与 $x_2 = c + \varepsilon > c$，则 $x_1, x_2$ 均为 $F_c$ 的连续点，故有：

$$ \lim_{n\to\infty} F_{X_n}(c - \varepsilon) = F_c(c - \varepsilon) = 0\\ \lim_{n\to\infty} F_{X_n}(c + \varepsilon) = F_c(c + \varepsilon) = 1 $$

利用事件包含关系 $\{\vert X_n - c\vert > \varepsilon\} = \{X_n < c - \varepsilon\} \cup \{X_n > c + \varepsilon\} \subset \{X_n \le c - \varepsilon\} \cup \{X_n > c + \varepsilon\}$，可得：

$$ \begin{aligned} 0 \le P\left(\vert X_n - c\vert > \varepsilon\right) &\le P\left(X_n \le c - \varepsilon\right) + P\left(X_n > c + \varepsilon\right) \\ &= F_{X_n}(c - \varepsilon) + \left[1 - F_{X_n}(c + \varepsilon)\right] \end{aligned} $$

令 $n \to \infty$，上式右端收敛于 $0 + (1 - 1) = 0$。由夹逼准则知 $\lim_{n\to\infty} P(\vert X_n - c\vert > \varepsilon) = 0$，即 $X_n \xrightarrow{P} c$。$\blacksquare$

<a id="cmt-stochastic-order"></a>

# 3. 连续映射定理与随机渐近阶符号

<a id="continuous-mapping"></a>

## 3.1 连续映射定理（CMT）

在确定性微积分中，若数列 $a_n \to a$ 且函数 $g(x)$ 在点 $a$ 处连续，则有 $g(a_n) \to g(a)$。连续映射定理将这一性质完整推广至随机变量的收敛序列。

> **定理 3.1（连续映射定理，Continuous Mapping Theorem）**
> 设 $g: \mathbb{R}^d \to \mathbb{R}^k$ 为 Borel 可测映射，其不连续点集合记为 $D_g$。若 $P(\mathbf{X}\in D_g)=0$，即 $g$ 相对于 $\mathbf{X}$ 的分布几乎处处连续，则：
> 1. 若 $\mathbf{X}_n \xrightarrow{P} \mathbf{X}$，则 $g(\mathbf{X}_n) \xrightarrow{P} g(\mathbf{X})$；
> 2. 若 $\mathbf{X}_n \xrightarrow{d} \mathbf{X}$，则 $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$。

**证明（常数概率极限与处处连续映射两种情形）**：
设函数 $g$ 在常数点 $\mathbf{c}$ 处连续。由多元函数在一点连续的 $\varepsilon$-$\delta$ 定义：对任意给定的 $\varepsilon > 0$，存在 $\delta > 0$，使得对任意满足 $\Vert \mathbf{x} - \mathbf{c}\Vert \le \delta$ 的点 $\mathbf{x}$，均有 $\Vert g(\mathbf{x}) - g(\mathbf{c})\Vert \le \varepsilon$。
取其逆否命题：若随机事件 $\{\Vert g(\mathbf{X}_n) - g(\mathbf{c})\Vert > \varepsilon\}$ 发生，则必然有事件 $\{\Vert \mathbf{X}_n - \mathbf{c}\Vert > \delta\}$ 发生。由事件包含关系的概率单调性，得：

$$ 0 \le P\left(\Vert g(\mathbf{X}_n) - g(\mathbf{c})\Vert > \varepsilon\right) \le P\left(\Vert \mathbf{X}_n - \mathbf{c}\Vert > \delta\right) $$

令 $n \to \infty$，由 $\mathbf{X}_n \xrightarrow{P} \mathbf{c}$ 知右端概率趋于 $0$，故 $g(\mathbf{X}_n) \xrightarrow{P} g(\mathbf{c})$。
对于依分布收敛情形，当 $g$ 处处连续时，任取有界连续函数 $h$，则复合函数 $h \circ g$ 仍为有界连续函数。由定理 2.1（**Portmanteau 定理**）的条件 2，直接有 $\mathbb{E}[h(g(\mathbf{X}_n))] = \mathbb{E}[(h\circ g)(\mathbf{X}_n)] \to \mathbb{E}[(h\circ g)(\mathbf{X})] = \mathbb{E}[h(g(\mathbf{X}))]$，故 $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$。$\blacksquare$

<a id="stochastic-order"></a>

## 3.2 随机渐近阶符号：$o_p(\cdot)$ 与 $O_p(\cdot)$

在经典数学分析中，**Landau** 符号 $o(\cdot)$ 与 $O(\cdot)$ 被广泛用于表示无穷小量与有界量的渐近阶。在数理统计中，为了简化复杂随机展开式的书写，引入了对应的**随机渐近阶符号（Stochastic Order Symbols）**。

> **定义 3.1（依概率无穷小量 $o_p$）**
> 设 $\{X_n\}$ 为随机变量序列，$\{a_n\}$ 为正实数序列。
> 1. 若 $X_n \xrightarrow{P} 0$，则记作 $X_n = o_p(1)$；
> 2. 若 $X_n/a_n \xrightarrow{P} 0$，则记作 $X_n = o_p(a_n)$。

> **定义 3.2（依概率有界量 $O_p$）**
> 设 $\{X_n\}$ 为随机变量序列，$\{a_n\}$ 为正实数序列。
> 1. 若对任意 $\eta>0$，存在有限常数 $M_\eta>0$ 和正整数 $N_\eta$，使得当 $n>N_\eta$ 时恒有
> $$ P\left(\vert X_n\vert > M_\eta\right) < \eta $$
> 则称序列 $\{X_n\}$ **依概率有界（Bounded in Probability）**，记作 $X_n=O_p(1)$；
> 2. 若 $X_n/a_n=O_p(1)$，则记作 $X_n=O_p(a_n)$。

> **定理 3.2（依分布收敛必依概率有界）**
> 若随机变量序列满足 $X_n \xrightarrow{d} X$，则必有 $X_n = O_p(1)$。

**证明**：任取 $\eta > 0$。由于极限随机变量 $X$ 的分布函数满足 $\lim_{x\to -\infty} F_X(x) = 0$ 及 $\lim_{x\to +\infty} F_X(x) = 1$，故存在充分大的正数 $M > 0$，使得 $-M$ 与 $M$ 均为 $F_X$ 的连续点，且满足：

$$ P(\vert X\vert > M) \le F_X(-M) + \left[1 - F_X(M)\right] < \frac{\eta}{2} $$

由 $X_n \xrightarrow{d} X$，在连续点 $\pm M$ 处有 $\lim_{n\to\infty} \left(F_{X_n}(-M) + 1 - F_{X_n}(M)\right) = F_X(-M) + 1 - F_X(M) < \frac{\eta}{2}$。由极限的保号性，存在正整数 $N$，当 $n > N$ 时恒有：

$$ P(\vert X_n\vert > M) \le F_{X_n}(-M) + 1 - F_{X_n}(M) < \eta $$

此即表明 $X_n = O_p(1)$。$\blacksquare$

> **应用示例（中心极限定理的渐近阶表示）**：
> 由中心极限定理知 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$。根据定理 3.2，该收敛序列必依概率有界，即：
> $$ \sqrt{n}(\bar{X}_n - \mu) = O_p(1) \iff \bar{X}_n - \mu = O_p\left(n^{-1/2}\right) $$
> 这表明样本均值逼近总体均值的收敛速率为 $n^{-1/2}$ 量级。由于 $n^{-1/2} \to 0$，故亦有 $\bar{X}_n - \mu = o_p(1)$（即弱大数定律）。

<a id="dl-applications"></a>

# 4. 深度学习中的应用

本章保留经验风险、Mini-batch 梯度与无限宽网络三个应用主题，分别对应大数定律、中心极限定理与随机函数的有限维分布。每道例题先给出论文背景和题目描述，再利用前文定理完成推导。

<a id="erm"></a>

## 例题 4.1（固定模型的经验风险收敛）

> **论文背景**：[**Optimization Methods for Large-Scale Machine Learning**](https://arxiv.org/abs/1606.04838) (*SIAM Review* 2018)。
>
> Bottou、Curtis 与 Nocedal 在大规模机器学习优化框架中区分了期望风险与样本平均风险。实际算法只能基于有限样本优化经验目标，而大数定律说明：对一个固定模型，样本平均损失可以逼近期望损失。这里讨论的是固定参数处的收敛，不把它直接等同于训练后模型的泛化保证。
>
> **题目描述**：设样本 $(X_i,Y_i)$ 独立同分布于 $P_{\text{data}}$，网络参数 $\boldsymbol{\theta}$ 固定且不依赖这些样本。单样本损失为 $L(\boldsymbol{\theta};X,Y)$，并满足 $\mathbb{E}[\vert L(\boldsymbol{\theta};X,Y)\vert]<\infty$。定义
> $$ R(\boldsymbol{\theta})=\mathbb{E}[L(\boldsymbol{\theta};X,Y)],\qquad \hat R_n(\boldsymbol{\theta})=\frac1n\sum_{i=1}^nL(\boldsymbol{\theta};X_i,Y_i) $$
> 1. 证明 $\hat R_n(\boldsymbol{\theta})\xrightarrow{P}R(\boldsymbol{\theta})$。
> 2. 若损失方差 $\tau_L^2(\boldsymbol{\theta})<\infty$，给出经验风险偏离期望风险的尾概率上界。
> 3. 说明上述逐点结论能否直接用于依赖训练样本的参数 $\hat{\boldsymbol{\theta}}_n$。

**解答**：
**(1)** 令 $Z_i=L(\boldsymbol{\theta};X_i,Y_i)$。由于参数固定，$Z_i$ 是独立同分布的随机变量，且 $\mathbb{E}[\vert Z_i\vert]<\infty$、$\mathbb{E}[Z_i]=R(\boldsymbol{\theta})$。由定理 1.3（**Khinchin 弱大数定律**）：
$$ \boxed{\hat R_n(\boldsymbol{\theta})=\frac1n\sum_{i=1}^nZ_i\xrightarrow{P}R(\boldsymbol{\theta})} $$
若调用定理 1.4，还可得到更强的几乎处处收敛 $\hat R_n(\boldsymbol{\theta})\xrightarrow{a.s.}R(\boldsymbol{\theta})$。

**(2)** 由独立性和方差可加性，$\text{Var}(\hat R_n)=\tau_L^2/n$。应用引理 1.2：
$$ \boxed{P\left(\vert\hat R_n(\boldsymbol{\theta})-R(\boldsymbol{\theta})\vert>\varepsilon\right)\le\frac{\tau_L^2(\boldsymbol{\theta})}{n\varepsilon^2}} $$
因此有限方差时，固定模型的经验风险具有明确的概率误差控制。

**(3)** 不能直接把固定参数 $\boldsymbol{\theta}$ 换成训练得到的随机参数 $\hat{\boldsymbol{\theta}}_n$，因为该参数依赖同一批样本，逐点大数定律没有控制所有候选模型。如果另外满足一致收敛条件
$$ \sup_{\boldsymbol{\theta}\in\Theta}\vert\hat R_n(\boldsymbol{\theta})-R(\boldsymbol{\theta})\vert\xrightarrow{P}0 $$
则训练得到的参数同样受该上确界控制。这说明固定模型的损失估计与经验风险最小化后的泛化分析需要区分，后者还需要对函数类作一致控制。

<a id="minibatch"></a>

## 例题 4.2（Mini-batch 梯度的高斯噪声近似）

> **论文背景**：[**Stochastic Gradient Descent as Approximate Bayesian Inference**](https://arxiv.org/abs/1704.04289) (*JMLR* 2017)。
>
> Mandt、Hoffman 与 Blei 用局部高斯噪声模型分析常步长随机梯度下降：批梯度等于期望梯度加上协方差与批大小成反比的噪声。论文把高斯性作为受中心极限定理启发的建模假设；下面在固定参数、独立采样和有限二阶矩条件下推导这一近似的来源，不把它推广为所有深度网络训练阶段的结论。
>
> **题目描述**：固定 $p$ 维参数 $\boldsymbol{\theta}$，设当前批次 $(X_i,Y_i)$ 独立同分布。记单样本梯度 $\mathbf{G}_i=\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta};X_i,Y_i)$，假设微分与期望可交换，且 $\mathbb{E}[\Vert\mathbf{G}_i\Vert_2^2]<\infty$。令
> $$ \mathbf{g}=\mathbb{E}[\mathbf{G}_i]=\nabla R(\boldsymbol{\theta}),\qquad \boldsymbol{\Sigma}=\text{Cov}(\mathbf{G}_i),\qquad \hat{\mathbf{g}}_B=\frac1B\sum_{i=1}^B\mathbf{G}_i $$
> 1. 求 $\hat{\mathbf{g}}_B$ 的均值与协方差。
> 2. 用一元 CLT 对任意固定方向的梯度投影求极限，进而推导批梯度的多元高斯近似。
> 3. 若仅要求单步参数更新的噪声协方差保持不变，批大小扩大 $k$ 倍时应如何调整学习率？它是否等同于线性缩放法则？

**解答**：
**(1)** 由期望的线性性和样本间独立性：
$$ \mathbb{E}[\hat{\mathbf{g}}_B]=\mathbf{g},\qquad \text{Cov}(\hat{\mathbf{g}}_B)=\frac{1}{B^2}\sum_{i=1}^B\text{Cov}(\mathbf{G}_i)=\frac{\boldsymbol{\Sigma}}{B} $$
协方差的 $1/B$ 缩放是精确恒等式，无须假定梯度已经服从正态分布。

**(2)** 固定向量 $a\in\mathbb{R}^p$，则 $a^T(\mathbf{G}_i-\mathbf{g})$ 独立同分布，均值为零、方差为 $a^T\boldsymbol{\Sigma}a$。若该方差为正，由定理 2.3：
$$ a^T\sqrt{B}(\hat{\mathbf{g}}_B-\mathbf{g})\xrightarrow{d}N(0,a^T\boldsymbol{\Sigma}a) $$
若方差为零，则该中心化投影几乎必然为零，同样具有相应的退化极限。由 **Cramér-Wold 定理**（随机向量依分布收敛，当且仅当其每个固定线性投影依分布收敛），得到
$$ \boxed{\sqrt{B}(\hat{\mathbf{g}}_B-\mathbf{g})\xrightarrow{d}N_p(\mathbf{0},\boldsymbol{\Sigma}),\qquad \hat{\mathbf{g}}_B\stackrel{\text{approx}}{\sim}N_p\left(\mathbf{g},\frac{\boldsymbol{\Sigma}}{B}\right)} $$
在固定维数下，这还给出 $\hat{\mathbf{g}}_B-\mathbf{g}=O_p(B^{-1/2})$。论文进一步在局部将协方差近似为常矩阵并取高斯噪声，才得到用于后续随机过程分析的 SGD 模型。

**(3)** 更新 $\Delta\boldsymbol{\theta}=-\eta\hat{\mathbf{g}}_B$ 中的随机噪声为 $-\eta(\hat{\mathbf{g}}_B-\mathbf{g})$，其协方差为 $\eta^2\boldsymbol{\Sigma}/B$。在 $\boldsymbol{\Sigma}$ 不变的局部模型中，若批大小改为 $kB$，则保持单步噪声协方差不变要求
$$ \frac{\eta'^2}{kB}=\frac{\eta^2}{B},\qquad \boxed{\eta'=\sqrt{k}\,\eta} $$
这是平方根缩放，不等同于 $\eta'=k\eta$ 的**线性缩放法则（Linear Scaling Rule）**。若固定每个 epoch 的样本数 $N$，并进一步把各步噪声视为独立且协方差近似不变，则一个 epoch 的累计更新噪声协方差约为 $N\eta^2\boldsymbol{\Sigma}/B^2$；在这一额外近似下，线性缩放使累计噪声保持不变。实际训练中的参数变化和批间依赖不由单次 CLT 保证。

<a id="infinite-width"></a>

## 例题 4.3（无限宽神经网络的高斯过程极限）

> **论文背景**：[**Deep Neural Networks as Gaussian Processes**](https://arxiv.org/abs/1711.00165) (*ICLR* 2018)。
>
> Lee 等人利用神经元贡献的独立求和与多元 CLT，说明随机初始化的无限宽网络对任意有限输入集合具有联合高斯输出，并由激活函数的乘积期望得到协方差核。论文进一步逐层递推深层网络的核；下面保留单隐藏层模型，推导这一机制的基本形式。它描述的是随机初始化先验，不是训练后网络输出的普遍分布结论。
>
> **题目描述**：考虑单隐藏层网络
> $$ f_M(\mathbf{x})=b+\frac1{\sqrt M}\sum_{j=1}^Mv_j\phi(\mathbf{w}_j^T\mathbf{x}+c_j) $$
> 其中 $\mathbf{x}\in\mathbb{R}^d$、隐藏层宽度为 $M$。不同神经元的参数独立同分布，$\mathbf{w}_j\sim N(\mathbf{0},I_d)$、$c_j\sim N(0,1)$、$v_j\sim N(0,\sigma_v^2)$；这些参数与 $b\sim N(0,\sigma_b^2)$ 相互独立。假设对所考察的每个输入，$\mathbb{E}[\phi^2(\mathbf{w}_j^T\mathbf{x}+c_j)]<\infty$。
> 1. 求固定输入 $\mathbf{x}$ 下 $f_M(\mathbf{x})$ 的极限分布。
> 2. 对任意有限输入集合 $\mathbf{x}^{(1)},\dots,\mathbf{x}^{(s)}$，证明输出向量具有联合高斯极限，并求其协方差核。
> 3. 说明 $1/\sqrt M$ 缩放与 CLT 的关系，以及改用 $1/M$ 时求和项的极限。

**解答**：
**(1)** 固定输入 $\mathbf{x}$，记第 $j$ 个神经元的贡献为
$$ Y_j(\mathbf{x})=v_j\phi(\mathbf{w}_j^T\mathbf{x}+c_j) $$
不同神经元的参数独立同分布，因此 $Y_j(\mathbf{x})$ 独立同分布。由 $v_j$ 与隐藏层参数独立、$\mathbb{E}[v_j]=0$：
$$ \begin{aligned}
\mathbb{E}[Y_j(\mathbf{x})]&=\mathbb{E}[v_j]\,\mathbb{E}[\phi(\mathbf{w}_j^T\mathbf{x}+c_j)]=0,\\
\tau^2(\mathbf{x})&:=\text{Var}(Y_j(\mathbf{x}))=\sigma_v^2\mathbb{E}[\phi^2(\mathbf{w}_j^T\mathbf{x}+c_j)]<\infty.
\end{aligned} $$
由定理 2.3（方差为零时求和项本身几乎必然为零），
$$ \frac1{\sqrt M}\sum_{j=1}^MY_j(\mathbf{x})\xrightarrow{d}N(0,\tau^2(\mathbf{x})) $$
由于偏置 $b$ 与求和项独立，二者之和的特征函数是各自特征函数之积，其极限为 $\exp[-t^2(\sigma_b^2+\tau^2(\mathbf{x}))/2]$。应用 Lévy 连续性定理，得到
$$ \boxed{f_M(\mathbf{x})\xrightarrow{d}N\left(0,\sigma_b^2+\sigma_v^2\mathbb{E}[\phi^2(\mathbf{w}_1^T\mathbf{x}+c_1)]\right)} $$

**(2)** 仅证明每个输入的边缘分布为正态，还不足以得到高斯过程。对固定的有限输入集合，构造神经元贡献向量
$$ \mathbf{Y}_j=\left(Y_j(\mathbf{x}^{(1)}),\dots,Y_j(\mathbf{x}^{(s)})\right)^T $$
这些向量在神经元下标 $j$ 上独立同分布，均值为零、二阶矩有限。同一个神经元对不同输入的贡献通常相关，其协方差为
$$ C_{ab}=\mathbb{E}[Y_j(\mathbf{x}^{(a)})Y_j(\mathbf{x}^{(b)})]=\sigma_v^2\mathbb{E}\left[\phi(\mathbf{w}_1^T\mathbf{x}^{(a)}+c_1)\phi(\mathbf{w}_1^T\mathbf{x}^{(b)}+c_1)\right] $$
对任意固定 $a\in\mathbb{R}^s$，$a^T\mathbf{Y}_j$ 满足一元 CLT。与例题 4.2 相同，由 Cramér-Wold 定理可得 $M^{-1/2}\sum_j\mathbf{Y}_j\xrightarrow{d}N_s(\mathbf{0},C)$。偏置 $b$ 对所有输入共享，故其协方差贡献为 $\sigma_b^2\mathbf{1}\mathbf{1}^T$，而不是 $\sigma_b^2I_s$。由独立性：
$$ \boxed{\left(f_M(\mathbf{x}^{(1)}),\dots,f_M(\mathbf{x}^{(s)})\right)^T\xrightarrow{d}N_s(\mathbf{0},K),\qquad K_{ab}=k(\mathbf{x}^{(a)},\mathbf{x}^{(b)})} $$
其中
$$ \boxed{k(\mathbf{x},\mathbf{x}')=\sigma_b^2+\sigma_v^2\mathbb{E}\left[\phi(\mathbf{w}_1^T\mathbf{x}+c_1)\phi(\mathbf{w}_1^T\mathbf{x}'+c_1)\right]} $$
任意有限输入集合都有上述联合高斯极限，因此得到以 $k$ 为协方差核的高斯过程的有限维分布；这里没有额外声称整个随机函数在某一函数空间中弱收敛。

**(3)** 单输入求和项的方差为
$$ \text{Var}\left(\frac1{\sqrt M}\sum_{j=1}^MY_j(\mathbf{x})\right)=\frac1M\sum_{j=1}^M\tau^2(\mathbf{x})=\tau^2(\mathbf{x}) $$
因此 $1/\sqrt M$ 恰好保持求和项的波动在常数量级，并对应 CLT 的尺度 $\sqrt M\,\bar Y_M$；它等价于将输出权重 $v_j/\sqrt M$ 的方差设为 $\sigma_v^2/M$。若改为 $1/M$，则由大数定律
$$ \frac1M\sum_{j=1}^MY_j(\mathbf{x})\xrightarrow{P}0 $$
随机求和项退化为零，只剩独立偏置 $b$。同一组神经元贡献的算术平均满足大数定律，而其 $\sqrt M$ 放大后的波动由中心极限定理刻画。

