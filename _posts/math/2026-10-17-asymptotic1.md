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

作为该系列的第 **I** 篇，本文将系统阐述前两个层次的理论基石：**随机变量序列的收敛模式**、**大数定律（LLN）**与**中心极限定理（CLT）**。

本文目录：
1. [依概率收敛与大数定律](#probability-convergence-lln)
   - [1.1 依概率收敛的定义](#conv-in-probability)
   - [1.2 概率论中的尾概率不等式](#tail-inequalities)
   - [1.3 弱大数定律](#weak-lln)
2. [依分布收敛与中心极限定理](#distribution-convergence-clt)
   - [2.1 依分布收敛的定义](#conv-in-distribution)
   - [2.2 中心极限定理](#clt)
   - [2.3 两种收敛模式的关系证明](#convergence-relations)
3. [连续映射定理与随机渐近阶符号](#cmt-stochastic-order)
   - [3.1 连续映射定理](#continuous-mapping)
   - [3.2 随机渐近阶符号](#stochastic-order)
4. [深度学习中的理论应用](#dl-applications)
   - [4.1 应用一：经验风险最小化](#erm)
   - [4.2 应用二：Mini-batch 梯度](#minibatch)
   - [4.3 应用三：无限宽神经网络](#infinite-width)


<a id="probability-convergence-lln"></a>

# 1. 依概率收敛与大数定律

<a id="conv-in-probability"></a>

## 1.1 依概率收敛的定义

在实分析中，确定性数列 $\{a_n\}$ 收敛于常数 $a$ 是指距离 $\vert a_n - a\vert $ 随 $n\to\infty$ 趋于零。对于随机变量序列 $\{X_n\}$，由于其取值具有随机性，需要通过事件发生的概率来度量其与极限目标 $X$ 的接近程度。

**定义 1.1（依概率收敛）**：设 $\{X_n\}$ 为随机变量序列，$X$ 为随机变量（或常数）。若对任意给定的正数 $\varepsilon > 0$，均有

$$ \boxed{\lim_{n\to\infty} P\left(\vert X_n - X\vert > \varepsilon\right) = 0 \quad \left(\text{等价地，}\lim_{n\to\infty} P\left(\vert X_n - X\vert \le \varepsilon\right) = 1\right)} $$

则称序列 $\{X_n\}$ **依概率收敛（Convergence in probability）**于 $X$，记作 $X_n \xrightarrow{P} X$。

在参数估计理论中，设 $\theta$ 为未知总体参数，$\hat{\theta}_n = \hat{\theta}(X_1, \dots, X_n)$ 为基于样本构造的估计量。若 $\hat{\theta}_n \xrightarrow{P} \theta$，则称 $\hat{\theta}_n$ 为参数 $\theta$ 的**弱相合估计量（Weakly Consistent Estimator）**。

<a id="tail-inequalities"></a>

## 1.2 概率论中的尾概率不等式

为了严格证明依概率收敛，需要将形如 $P(\vert X_n - X\vert > \varepsilon)$ 的尾概率与随机变量的矩（数学期望、方差）建立联系。以下两个基础不等式是证明大数定律的核心工具。

**引理 1.1（Markov 不等式）**：设非负随机变量 $Y \ge 0$ 的数学期望 $\mathbb{E}[Y]$ 存在，则对任意 $\varepsilon > 0$，有

$$ \boxed{P(Y \ge \varepsilon) \le \frac{\mathbb{E}[Y]}{\varepsilon}} $$

**证明**：设 $Y$ 的概率密度函数为 $f(y)$（离散情形将积分替换为求和即可）。将数学期望的积分区间拆分为 $[0, \varepsilon)$ 与 $[\varepsilon, \infty)$ 两部分：

$$ \mathbb{E}[Y] = \int_0^\infty y f(y)\,\mathrm{d}y = \int_0^\varepsilon y f(y)\,\mathrm{d}y + \int_\varepsilon^\infty y f(y)\,\mathrm{d}y $$

由于被积函数非负，舍去第一项积分不改变不等号方向；在第二项积分中，积分区间满足 $y \ge \varepsilon$，故有：

$$ \mathbb{E}[Y] \ge \int_\varepsilon^\infty y f(y)\,\mathrm{d}y \ge \int_\varepsilon^\infty \varepsilon f(y)\,\mathrm{d}y = \varepsilon \int_\varepsilon^\infty f(y)\,\mathrm{d}y = \varepsilon P(Y \ge \varepsilon) $$

两边同除以 $\varepsilon > 0$，即得所证不等式。$\blacksquare$

**引理 1.2（Chebyshev 不等式）**：设随机变量 $X$ 的均值为 $\mu = \mathbb{E}[X]$，方差为 $\sigma^2 = \text{Var}(X) < \infty$。则对任意 $\varepsilon > 0$，有

$$ \boxed{P\left(\vert X - \mu\vert \ge \varepsilon\right) \le \frac{\text{Var}(X)}{\varepsilon^2}} $$

**证明**：注意到事件 $\{\vert X - \mu\vert \ge \varepsilon\}$ 与事件 $\{(X - \mu)^2 \ge \varepsilon^2\}$ 完全等价。令非负随机变量 $Y = (X - \mu)^2$，对 $Y$ 应用引理 **1.1（Markov** 不等式），并代入 $\mathbb{E}[Y] = \mathbb{E}[(X-\mu)^2] = \text{Var}(X)$，可得：

$$ P\left(\vert X - \mu\vert \ge \varepsilon\right) = P\left((X - \mu)^2 \ge \varepsilon^2\right) \le \frac{\mathbb{E}\left[(X - \mu)^2\right]}{\varepsilon^2} = \frac{\text{Var}(X)}{\varepsilon^2} \quad \blacksquare $$

<a id="weak-lln"></a>

## 1.3 弱大数定律（Weak Law of Large Numbers）

**大数定律**揭示了算术平均值的稳定性：当试验次数无限增加时，样本均值依概率收敛于总体均值。根据矩条件的不同，弱大数定律有两个经典版本。

### 1.3.1 Chebyshev 弱大数定律（二阶矩存在情形）

**定理 1.1（二阶矩条件下的弱大数定律）**：设 $X_1, X_2, \dots, X_n$ 为两两不相关的随机变量序列，具有相同的数学期望 $\mathbb{E}[X_i] = \mu$，且方差存在一致上界 $\text{Var}(X_i) \le C < \infty$（特别地，独立同分布且方差有限的序列满足此条件）。记样本均值为 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$，则有

$$ \boxed{\bar{X}_n \xrightarrow{P} \mu \quad (n \to \infty)} $$

**证明**：

**1.计算样本均值的期望与方差**：由数学期望的线性性质，有

$$ \mathbb{E}[\bar{X}_n] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = \frac{1}{n}\cdot n\mu = \mu $$

由随机变量两两不相关性（协方差 $\text{Cov}(X_i, X_j) = 0,\ \forall i\neq j$），方差满足可加性：

$$ \text{Var}(\bar{X}_n) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) \le \frac{1}{n^2}\cdot nC = \frac{C}{n} $$

**2.应用 Chebyshev 不等式**：对任意给定的 $\varepsilon > 0$，将引理 **1.2** 应用于随机变量 $\bar{X}_n$，得

$$ 0 \le P\left(\vert \bar{X}_n - \mu\vert > \varepsilon\right) \le \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} \le \frac{C}{n\varepsilon^2} $$

**3.取极限**：令 $n \to \infty$，由于 $\frac{C}{n\varepsilon^2} \to 0$，由夹逼准则即得 $\lim_{n\to\infty} P(\vert \bar{X}_n - \mu\vert > \varepsilon) = 0$。故 $\bar{X}_n \xrightarrow{P} \mu$。$\blacksquare$

### 1.3.2 Khinchin 弱大数定律（仅一阶矩存在情形）

在实际应用中，某些重尾分布（如自由度为 2 的 $t$ 分布，或尾指数不超过 $2$ 的 **Pareto** 分布）的方差不存在（即 $\text{Var}(X_i) = \infty$），此时 **Chebyshev** 不等式失效。**Khinchin（辛钦）弱大数定律**证明了：对于独立同分布序列，无需假设方差存在，只要一阶矩（数学期望）有限，弱大数定律依然成立。

为了给出完整证明，引入概率论中的**特征函数（Characteristic Function）**工具。

> **预备知识：特征函数及其泰勒展开**
> 
> 设随机变量 $X$ 的特征函数定义为 $\varphi_X(t) = \mathbb{E}[e^{\mathrm{i}tX}] = \mathbb{E}[\cos(tX)] + \mathrm{i}\mathbb{E}[\sin(tX)]$，其中 $\mathrm{i} = \sqrt{-1}$，$t \in \mathbb{R}$。特征函数具有以下基本性质：
> 1. **线性变换性质**：$\varphi_{aX+b}(t) = e^{\mathrm{i}tb}\varphi_X(at)$。
> 2. **独立和性质**：若 $X_1, \dots, X_n$ 相互独立，则 $\varphi_{\sum_{i=1}^n X_i}(t) = \prod_{i=1}^n \varphi_{X_i}(t)$。
> 3. **可微性与泰勒展开**：若 $\mathbb{E}[\vert X\vert ^k] < \infty$，则 $\varphi_X(t)$ 在 $t=0$ 处 $k$ 阶连续可导，且 $\varphi_X^{(k)}(0) = \mathrm{i}^k \mathbb{E}[X^k]$。由带 **Peano** 余项的泰勒公式，当 $t \to 0$ 时：
>    * 若 $\mathbb{E}[\vert X\vert ] < \infty$ 且 $\mathbb{E}[X] = \mu$，则 $\varphi_X(t) = 1 + \mathrm{i}\mu t + o(t)$；
>    * 若 $\mathbb{E}[X^2] < \infty$ 且 $\mathbb{E}[X] = 0,\ \text{Var}(X) = \sigma^2$，则 $\varphi_X(t) = 1 - \frac{1}{2}\sigma^2 t^2 + o(t^2)$。

**定理 1.2（Khinchin 弱大数定律）**：设 $X_1, X_2, \dots, X_n$ 独立同分布，且数学期望有限 $\mathbb{E}[\vert X_1\vert ] < \infty$，记 $\mathbb{E}[X_1] = \mu$，则有

$$ \boxed{\bar{X}_n \xrightarrow{P} \mu \quad (n \to \infty)} $$

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

> **注（几乎处处收敛与强大数定律）**：除依概率收敛外，概率论中还定义了更强的**几乎处处收敛（Almost sure convergence，记作 $X_n \xrightarrow{a.s.} X$）**，即 $P\left(\lim_{n\to\infty} X_n = X\right) = 1$。它要求样本空间中几乎每一条实现轨道都收敛于目标值。**Kolmogorov 强大数定律（SLLN）**指出，在定理 1.2 完全相同的条件下，事实上成立更强的结论 $\bar{X}_n \xrightarrow{a.s.} \mu$。由于几乎处处收敛蕴含依概率收敛，这也从另一角度印证了 **Khinchin** 弱大数定律。

<a id="distribution-convergence-clt"></a>

# 2. 依分布收敛与中心极限定理

<a id="conv-in-distribution"></a>

## 2.1 依分布收敛的定义

大数定律表明估计误差满足 $\bar{X}_n - \mu \xrightarrow{P} 0$，但这仅给出了点估计的收敛性。为了进一步构造置信区间或进行假设检验，必须刻画估计量在逼近真值过程中的分布形态。

**定义 2.1（依分布收敛）**：设 $\{X_n\}$ 为随机变量序列，其对应的累积分布函数（**CDF**）记为 $F_{X_n}(x) = P(X_n \le x)$；设 $X$ 为极限随机变量，其分布函数记为 $F_X(x) = P(X \le x)$。若在 $F_X(x)$ 的**所有连续点** $x \in C(F_X)$ 上，均有

$$ \boxed{\lim_{n\to\infty} F_{X_n}(x) = F_X(x)} $$

则称序列 $\{X_n\}$ **依分布收敛（Convergence in distribution，又称弱收敛 Weak convergence）**于 $X$，记作 $X_n \xrightarrow{d} X$。

> **💡 为何定义中仅要求在“连续点”上收敛？**
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
> 由此可见，阶梯函数在水平平移逼近极限位置时，在跳跃边界点处必然出现极限值与函数值不相等的现象。由于单调有界函数的不连续点至多只有可数个，剔除不连续点既消除了这种边界伪反例，又唯一确定了极限分布函数。

在高等概率论中，判断依分布收敛的等价工具有以下两个基本定理：

**定理 2.1（Portmanteau 定理）**：对于随机变量序列 $\{X_n\}$ 与 $X$，下列命题完全等价：
1. $X_n \xrightarrow{d} X$；
2. 对任意有界连续函数 $h: \mathbb{R} \to \mathbb{R}$，有 $\lim_{n\to\infty} \mathbb{E}[h(X_n)] = \mathbb{E}[h(X)]$；
3. 对任意闭集 $C \subset \mathbb{R}$，有 $\limsup_{n\to\infty} P(X_n \in C) \le P(X \in C)$；
4. 对任意开集 $G \subset \mathbb{R}$，有 $\liminf_{n\to\infty} P(X_n \in G) \ge P(X \in G)$。

**定理 2.2（Lévy 连续性定理）**：设 $X_n$ 的特征函数为 $\varphi_{X_n}(t)$，$X$ 的特征函数为 $\varphi_X(t)$。则 $X_n \xrightarrow{d} X$ 的充要条件是：对任意 $t \in \mathbb{R}$，均有 $\lim_{n\to\infty} \varphi_{X_n}(t) = \varphi_X(t)$。

<a id="clt"></a>

## 2.2 中心极限定理（Central Limit Theorem）

大数定律指出样本均值的误差满足 $\bar{X}_n - \mu \xrightarrow{P} 0$。若直接考察 $\bar{X}_n - \mu$ 的极限分布，只能得到退化于 $0$ 的单点分布。为了观察到非退化的误差分布形态，需要寻找恰当的放大因子 $a_n \to \infty$，使得 $a_n(\bar{X}_n - \mu)$ 的方差稳定在非零常数。

注意到 $\text{Var}\big(a_n(\bar{X}_n - \mu)\big) = a_n^2 \cdot \frac{\sigma^2}{n}$。要使该方差不随 $n\to\infty$ 退化为 $0$ 或发散至 $\infty$，放大因子必须与 $\sqrt{n}$ 同阶，即 **$a_n \propto \sqrt{n}$**；通常取 $a_n = \sqrt{n}$，使极限方差恰为 $\sigma^2$。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-003.png)

**定理 2.3（Lindeberg-Lévy 中心极限定理）**：设 $X_1, X_2, \dots, X_n$ 为独立同分布的随机变量序列，具有有限的数学期望 $\mathbb{E}[X_i] = \mu$ 与有限的正方差 $\text{Var}(X_i) = \sigma^2 \in (0, \infty)$。则当 $n \to \infty$ 时，标准化样本均值依分布收敛于标准正态分布：

$$ \boxed{Z_n = \frac{\sqrt{n}\left(\bar{X}_n - \mu\right)}{\sigma} \xrightarrow{d} Z \sim N(0, 1)} $$

等价地，$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$。

**证明（利用特征函数与二阶泰勒展开）**：

**1.将随机变量标准化**：令 $Y_i = \frac{X_i - \mu}{\sigma}$。由独立同分布假设知，$Y_1, \dots, Y_n$ 亦独立同分布，且满足：

$$ \mathbb{E}[Y_i] = 0, \qquad \text{Var}(Y_i) = \mathbb{E}[Y_i^2] = 1 $$

此时，目标统计量 $Z_n$ 可表示为标准化变量的算术和：

$$ Z_n = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} = \frac{1}{\sqrt{n}}\sum_{i=1}^n \left(\frac{X_i - \mu}{\sigma}\right) = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i $$

**2.对单个变量的特征函数作二阶泰勒展开**：设 $Y_i$ 的共同特征函数为 $\varphi_Y(t) = \mathbb{E}[e^{\mathrm{i}tY_1}]$。由于二阶矩 $\mathbb{E}[Y_1^2] = 1 < \infty$，特征函数 $\varphi_Y(t)$ 在 $t=0$ 处二阶连续可导，且有：

$$ \varphi_Y(0) = 1, \quad \varphi_Y'(0) = \mathrm{i}\mathbb{E}[Y_1] = 0, \quad \varphi_Y''(0) = \mathrm{i}^2 \mathbb{E}[Y_1^2] = -1 $$

由带 **Peano** 余项的二阶泰勒公式，当 $s \to 0$ 时：

$$ \varphi_Y(s) = \varphi_Y(0) + \varphi_Y'(0)s + \frac{\varphi_Y''(0)}{2}s^2 + o(s^2) = 1 - \frac{1}{2}s^2 + o(s^2) $$

**3.计算 $Z_n$ 的特征函数并求极限**：利用独立和的特征函数性质，将 $s = \frac{t}{\sqrt{n}}$ 代入上式。对任意固定的 $t \in \mathbb{R}$，当 $n \to \infty$ 时 $\frac{t}{\sqrt{n}} \to 0$，从而有：

$$
\begin{aligned}
\varphi_{Z_n}(t) &= \mathbb{E}\left[\exp\left(\mathrm{i}t \cdot \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i\right)\right] \\&= \left[\varphi_Y\left(\frac{t}{\sqrt{n}}\right)\right]^n \\&= \left[1 - \frac{t^2}{2n} + o\left(\frac{t^2}{n}\right)\right]^n
\end{aligned}
$$

对任意固定的 $t$，余项满足 $o\left(\frac{t^2}{n}\right) = o\left(\frac{1}{n}\right)$。由重要极限公式，当 $n \to \infty$ 时：
   $$ \lim_{n\to\infty} \varphi_{Z_n}(t) = \lim_{n\to\infty} \left[1 + \frac{-t^2/2}{n} + o\left(\frac{1}{n}\right)\right]^n = e^{-t^2/2}, \quad \forall t \in \mathbb{R} $$

**4.应用 Lévy 连续性定理**：注意到 $e^{-t^2/2}$ 恰好是标准正态分布 $Z \sim N(0,1)$ 的特征函数。由定理 **2.2（Lévy** 连续性定理），立即得出 $Z_n \xrightarrow{d} Z \sim N(0,1)$。$\blacksquare$

<a id="convergence-relations"></a>

## 2.3 两种收敛模式的关系证明

在渐近理论中，依概率收敛与依分布收敛存在紧密的逻辑关联。本节采用分布函数定义与微积分夹逼准则，给出两者关系的完整证明。

**定理 2.4（依概率收敛蕴含依分布收敛）**：设 $\{X_n\}$ 为随机变量序列，$X$ 为随机变量。若 $X_n \xrightarrow{P} X$，则必有 $X_n \xrightarrow{d} X$。

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

**定理 2.5（退化极限下的等价性）**：设 $c$ 为常数，则 $X_n \xrightarrow{d} c \iff X_n \xrightarrow{P} c$

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

**定理 3.1（连续映射定理，Continuous Mapping Theorem）**：设映射 $g: \mathbb{R}^d \to \mathbb{R}^k$ 的不连续点集合记为 $D_g$。若极限随机向量 $\mathbf{X}$ 落在不连续点集上的概率为零，即 $P(\mathbf{X} \in D_g) = 0$（等价于 $g$ 在 $\mathbf{X}$ 的支撑集上几乎处处连续），则：
1. 若 $\mathbf{X}_n \xrightarrow{P} \mathbf{X}$，则 $g(\mathbf{X}_n) \xrightarrow{P} g(\mathbf{X})$；
2. 若 $\mathbf{X}_n \xrightarrow{d} \mathbf{X}$，则 $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$。

**证明（针对统计中最常用的依概率收敛于常数情形 $\mathbf{X}_n \xrightarrow{P} \mathbf{c}$）**：
设函数 $g$ 在常数点 $\mathbf{c}$ 处连续。由多元函数在一点连续的 $\varepsilon$-$\delta$ 定义：对任意给定的 $\varepsilon > 0$，存在 $\delta > 0$，使得对任意满足 $\Vert \mathbf{x} - \mathbf{c}\Vert \le \delta$ 的点 $\mathbf{x}$，均有 $\Vert g(\mathbf{x}) - g(\mathbf{c})\Vert \le \varepsilon$。
取其逆否命题：若随机事件 $\{\Vert g(\mathbf{X}_n) - g(\mathbf{c})\Vert > \varepsilon\}$ 发生，则必然有事件 $\{\Vert \mathbf{X}_n - \mathbf{c}\Vert > \delta\}$ 发生。由事件包含关系的概率单调性，得：

$$ 0 \le P\left(\Vert g(\mathbf{X}_n) - g(\mathbf{c})\Vert > \varepsilon\right) \le P\left(\Vert \mathbf{X}_n - \mathbf{c}\Vert > \delta\right) $$

令 $n \to \infty$，由 $\mathbf{X}_n \xrightarrow{P} \mathbf{c}$ 知右端概率趋于 $0$，故 $g(\mathbf{X}_n) \xrightarrow{P} g(\mathbf{c})$。
对于依分布收敛情形，当 $g$ 处处连续时，任取有界连续函数 $h$，则复合函数 $h \circ g$ 仍为有界连续函数。由定理 **2.1（Portmanteau** 定理）的条件 2，直接有 $\mathbb{E}[h(g(\mathbf{X}_n))] = \mathbb{E}[(h\circ g)(\mathbf{X}_n)] \to \mathbb{E}[(h\circ g)(\mathbf{X})] = \mathbb{E}[h(g(\mathbf{X}))]$，故 $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$。$\blacksquare$

<a id="stochastic-order"></a>

## 3.2 随机渐近阶符号：$o_p(\cdot)$ 与 $O_p(\cdot)$

在经典数学分析中，**Landau** 符号 $o(\cdot)$ 与 $O(\cdot)$ 被广泛用于表示无穷小量与有界量的渐近阶。在数理统计中，为了简化复杂随机展开式的书写，引入了对应的**随机渐近阶符号（Stochastic Order Symbols）**。

**定义 3.1（依概率无穷小量 $o_p$）**：设 $\{X_n\}$ 为随机变量序列，$\{a_n\}$ 为正实数序列。
* 若 $X_n \xrightarrow{P} 0$，则记作 $X_n = o_p(1)$；
* 若 $\frac{X_n}{a_n} \xrightarrow{P} 0$，则称 $X_n$ 是比 $a_n$ 更高阶的依概率无穷小量，记作 $X_n = o_p(a_n)$。

**定义 3.2（依概率有界量 $O_p$）**：设 $\{X_n\}$ 为随机变量序列，$\{a_n\}$ 为正实数序列。
* 若对任意给定的 $\eta > 0$，存在有限正常数 $M_\eta > 0$ 及正整数 $N_\eta$，使得当 $n > N_\eta$ 时恒有
  $$ P\left(\vert X_n\vert > M_\eta\right) < \eta $$
  则称序列 $\{X_n\}$ **依概率有界（Bounded in probability，或称一致紧 Tight）**，记作 $X_n = O_p(1)$；
* 若 $\frac{X_n}{a_n} = O_p(1)$，则记作 $X_n = O_p(a_n)$。

**定理 3.2（依分布收敛必依概率有界）**：若随机变量序列满足 $X_n \xrightarrow{d} X$，则必有 $X_n = O_p(1)$。

**证明**：任取 $\eta > 0$。由于极限随机变量 $X$ 的分布函数满足 $\lim_{x\to -\infty} F_X(x) = 0$ 及 $\lim_{x\to +\infty} F_X(x) = 1$，故存在充分大的正数 $M > 0$，使得 $-M$ 与 $M$ 均为 $F_X$ 的连续点，且满足：

$$ P(\vert X\vert > M) = F_X(-M) + \left[1 - F_X(M)\right] < \frac{\eta}{2} $$

由 $X_n \xrightarrow{d} X$，在连续点 $\pm M$ 处有 $\lim_{n\to\infty} \left(F_{X_n}(-M) + 1 - F_{X_n}(M)\right) = F_X(-M) + 1 - F_X(M) < \frac{\eta}{2}$。由极限的保号性，存在正整数 $N$，当 $n > N$ 时恒有：

$$ P(\vert X_n\vert > M) \le F_{X_n}(-M) + 1 - F_{X_n}(M) < \eta $$

此即表明 $X_n = O_p(1)$。$\blacksquare$

> **应用示例（中心极限定理的渐近阶表示）**：
> 由中心极限定理知 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$。根据定理 3.2，该收敛序列必依概率有界，即：
> $$ \sqrt{n}(\bar{X}_n - \mu) = O_p(1) \iff \bar{X}_n - \mu = O_p\left(n^{-1/2}\right) $$
> 这表明样本均值逼近总体均值的收敛速率为 $n^{-1/2}$ 量级。由于 $n^{-1/2} \to 0$，故亦有 $\bar{X}_n - \mu = o_p(1)$（即弱大数定律）。

<a id="dl-applications"></a>

# 4. 深度学习中的理论应用

大数定律与中心极限定理不仅是经典统计学的核心，同样构成了现代深度学习泛化理论与架构分析的数学基础。

<a id="erm"></a>

## 4.1 应用一：经验风险最小化（ERM）

在监督学习与模型预训练中，设输入输出数据对 $(X, Y)$ 服从未知的真实数据分布 $P_{\text{data}}$，神经网络参数为 $\boldsymbol{\theta}$，单样本损失函数为 $L(\boldsymbol{\theta}; X, Y)$。训练的终极目标是最小化**期望风险（总体损失）**：

$$ R(\boldsymbol{\theta}) = \mathbb{E}_{(X,Y)\sim P_{\text{data}}}\big[L(\boldsymbol{\theta}; X, Y)\big] $$

由于真实分布 $P_{\text{data}}$ 未知，算法只能收集独立同分布的训练样本集 $\{(X_i, Y_i)\}_{i=1}^n$，转而最小化**经验风险（样本平均损失）**：

$$ \hat{R}_n(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n L(\boldsymbol{\theta}; X_i, Y_i) $$

对于任意固定的参数 $\boldsymbol{\theta}$，只要单样本损失的期望有限，由定理 **1.2（Khinchin** 弱大数定律）直接得出：

$$ \hat{R}_n(\boldsymbol{\theta}) \xrightarrow{P} R(\boldsymbol{\theta}) \quad (n \to \infty) $$

这从理论上保证了当训练数据集规模 $n$ 充分大时，经验损失函数是真实期望损失函数的相合估计。

<a id="minibatch"></a>

## 4.2 应用二：Mini-batch 梯度

在每次参数迭代中，算法抽取容量为 $B$（**Batch Size**）的 **Mini-batch** 计算随机梯度:

$$ \hat{\mathbf{g}}_B(\boldsymbol{\theta}) = \frac{1}{B}\sum_{i=1}^B \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}; X_i, Y_i)$$

设真实梯度为 $\mathbf{g}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} R(\boldsymbol{\theta})$，单样本梯度的协方差矩阵为 $\boldsymbol{\Sigma}(\boldsymbol{\theta})$。由多元中心极限定理，当 **Batch Size** $B$ 较大时，随机梯度近似服从高斯分布：

$$ \hat{\mathbf{g}}_B(\boldsymbol{\theta}) \stackrel{\text{approx}}{\sim} N\left(\mathbf{g}(\boldsymbol{\theta}),\ \frac{1}{B}\boldsymbol{\Sigma}(\boldsymbol{\theta})\right) $$

该式表明随机梯度的方差与 **Batch Size** $B$ 成反比：当把 **Batch Size** 扩大 $k$ 倍时，梯度噪声的标准差缩小为原来的 $1/\sqrt{k}$。若只要求单步参数更新的噪声标准差不变，应令学习率（**Learning Rate**）与 $\sqrt{B}$ 成正比；而实践中常用的**线性缩放法则（Linear Scaling Rule）**则令学习率与 $B$ 成正比（保持 $\eta/B$ 不变），使得一个 epoch 内的累计更新噪声大致不变。

<a id="infinite-width"></a>

## 4.3 应用三：无限宽神经网络

在现代深度学习理论中，一个核心结论是：当神经网络的隐藏层宽度趋于无穷大时，随机初始化的神经网络在任意输入上的输出依分布收敛于高斯过程**（Gaussian Process）**。这一深刻结论本质上正是**中心极限定理**的直接推论。

考虑一个含单隐藏层的全连接神经网络，输入向量为 $\mathbf{x} \in \mathbb{R}^{d}$，隐藏层包含 $M$ 个神经元，激活函数为 $\phi(\cdot)$。网络标量输出 $f_M(\mathbf{x})$ 的前向传播公式为：

$$ f_M(\mathbf{x}) = b + \frac{1}{\sqrt{M}}\sum_{j=1}^M v_j \phi\left(\mathbf{w}_j^\top \mathbf{x} + c_j\right) $$

为便于推导，取如下各神经元参数相互独立且同分布（**i.i.d.**）的初始化（这里固定隐藏层权重方差为 $1$；LeCun/He 初始化只改变该方差常数，不改变下面的中心极限定理论证）：
* 第一层权重 $\mathbf{w}_j \stackrel{\text{i.i.d.}}{\sim} N(\mathbf{0}, \mathbf{I}_d)$，偏置 $c_j \stackrel{\text{i.i.d.}}{\sim} N(0, 1)$；
* 第二层输出权重 $v_j \stackrel{\text{i.i.d.}}{\sim} N(0, \sigma_v^2)$，偏置 $b \sim N(0, \sigma_b^2)$，且 $v_j$ 与 $(\mathbf{w}_j, c_j)$ 相互独立。

**【利用中心极限定理推导极限分布】**：

固定任意输入样本 $\mathbf{x}$，记第 $j$ 个隐藏神经元的贡献项为随机变量：

$$ Y_j(\mathbf{x}) = v_j \phi\left(\mathbf{w}_j^\top \mathbf{x} + c_j\right), \quad j = 1, 2, \dots, M $$


**1.验证独立同分布性**：由于不同神经元的参数 $(v_j, \mathbf{w}_j, c_j)$ 相互独立且同分布，故对固定的输入 $\mathbf{x}$，随机变量序列 $Y_1(\mathbf{x}), Y_2(\mathbf{x}), \dots, Y_M(\mathbf{x})$ 相互独立且同分布。

**2.计算均值与方差**：由 $v_j$ 与 $(\mathbf{w}_j, c_j)$ 的独立性及 $\mathbb{E}[v_j] = 0$，有：

$$
\begin{aligned}
\mathbb{E}[Y_j(\mathbf{x})] &= \mathbb{E}[v_j] \cdot \mathbb{E}\left[\phi(\mathbf{w}_j^\top \mathbf{x} + c_j)\right] = 0 \\
\tau^2(\mathbf{x}) :&= \text{Var}\big(Y_j(\mathbf{x})\big) = \mathbb{E}\left[Y_j^2(\mathbf{x})\right] \\&= \mathbb{E}[v_j^2] \cdot \mathbb{E}\left[\phi^2(\mathbf{w}_j^\top \mathbf{x} + c_j)\right] \\&= \sigma_v^2 \mathbb{E}\left[\phi^2(\mathbf{w}_1^\top \mathbf{x} + c_1)\right] < \infty
\end{aligned}
$$

**3.应用中心极限定理**：注意到网络输出公式中的缩放因子 $\frac{1}{\sqrt{M}}$ 恰好对应中心极限定理的标准化尺度：

$$ \frac{1}{\sqrt{M}}\sum_{j=1}^M Y_j(\mathbf{x}) = \sqrt{M}\left(\frac{1}{M}\sum_{j=1}^M Y_j(\mathbf{x}) - 0\right) $$

由定理 **2.3（Lindeberg-Lévy** 中心极限定理），当网络隐藏层宽度 $M \to \infty$ 时，求和项依分布收敛于均值为 $0$、方差为 $\tau^2(\mathbf{x})$ 的正态分布。结合独立的偏置项 $b \sim N(0, \sigma_b^2)$，网络最终输出满足：

$$ f_M(\mathbf{x}) \xrightarrow{d} N\Big(0,\ \sigma_b^2 + \sigma_v^2 \mathbb{E}\big[\phi^2(\mathbf{w}_1^\top \mathbf{x} + c_1)\big]\Big) \quad (M \to \infty) $$

这也解释了输出层缩放因子 $\frac{1}{\sqrt{M}}$ 的来源：它正是中心极限定理的标准化尺度，等价于将输出权重方差设为 $\sigma_v^2/M$。需要区分的是，隐藏层的 **Xavier / Kaiming** 初始化按 fan-in/fan-out 缩放权重方差（例如 Kaiming 初始化取 $2/\text{fan-in}$），其目的是保持逐层前向与反向传播的方差稳定；它与输出层为保证 CLT 而引入的 $1/\sqrt{M}$ 因子属于两个不同层面的缩放。

