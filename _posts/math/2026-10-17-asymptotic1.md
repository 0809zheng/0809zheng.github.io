---
layout: post
title: '渐近理论(Asymptotic Theory) I：大数定律与中心极限定理'
date: 2026-10-17
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-cover.png'
tags: 数学
pinned: false
---

> Asymptotic Theory I: Law of Large Numbers and Central Limit Theorem.

在经典数理统计的有限样本理论（**Finite-Sample Theory**）中，统计推断依赖于总体分布的精确先验假设。例如，当总体服从正态分布 $X_i \sim N(\mu, \sigma^2)$ 时，无论样本容量 $n$ 取何值，样本均值 $\bar{X}_n$ 的精确分布均为正态分布，样本方差的精确分布可由$\chi^2$分布刻画，进而可导出精确服从 $t(n-1)$ 分布的检验统计量。

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
   - [例题：MC dropout 预测均值的蒙特卡洛误差](#exercise-mc-dropout)


<a id="probability-convergence-lln"></a>

# 1. 依概率收敛与大数定律

<a id="conv-in-probability"></a>

## 1.1 依概率收敛的定义

在实分析中，确定性数列 $\{a_n\}$ 收敛于常数 $a$ 是指距离 $\vert a_n - a\vert $ 随 $n\to\infty$ 趋于零。对于随机变量序列 $\{X_n\}$，由于其取值具有随机性，需要通过事件发生的概率来度量其与极限目标 $X$ 的接近程度。

> **定义 1.1（依概率收敛）**
> 设 $\{X_n\}$ 与 $X$ 为定义在同一概率空间上的随机变量。若对任意正数 $\varepsilon > 0$，均有
> $$ \lim_{n\to\infty} P\left(\vert X_n - X\vert > \varepsilon\right) = 0 \quad \left(\text{等价地，}\lim_{n\to\infty} P\left(\vert X_n - X\vert \le \varepsilon\right) = 1\right) $$
> 则称序列 $\{X_n\}$ **依概率收敛（Convergence in Probability）**于 $X$，记作 $X_n \xrightarrow{P} X$。

在参数估计理论中，设 $\theta$ 为未知总体参数，$\hat{\theta}_n = \hat{\theta}(X_1, \dots, X_n)$ 为基于样本构造的估计量。若 $\hat{\theta}_n \xrightarrow{P} \theta$，则称 $\hat{\theta}_n$ 为参数 $\theta$ 的**弱相合估计量（Weakly Consistent Estimator）**。

<a id="tail-inequalities"></a>

## 1.2 概率论中的尾概率不等式

为了严格证明依概率收敛，需要将形如 $P(\vert X_n - X\vert > \varepsilon)$ 的尾概率与随机变量的矩（数学期望、方差）建立联系。以下两个基础不等式是证明各类大数定律的核心工具。

> **引理 1.1（Markov 不等式）**
> 设非负随机变量 $Y \ge 0$ 满足 $\mathbb{E}[Y]<\infty$，则对任意 $\varepsilon > 0$，有
> $$ P(Y \ge \varepsilon) \le \frac{\mathbb{E}[Y]}{\varepsilon} $$

**证明**：根据事件 $\{Y\ge\varepsilon\}$ 是否发生，分解数学期望：

$$ \mathbb{E}[Y]=\mathbb{E}\left[Y\mathbf{1}_{\{Y<\varepsilon\}}\right]+\mathbb{E}\left[Y\mathbf{1}_{\{Y\ge\varepsilon\}}\right] $$

第一项非负；第二项中 $Y\ge\varepsilon$，因此

$$ \mathbb{E}[Y]\ge\mathbb{E}\left[Y\mathbf{1}_{\{Y\ge\varepsilon\}}\right]\ge\varepsilon\mathbb{E}\left[\mathbf{1}_{\{Y\ge\varepsilon\}}\right]=\varepsilon P(Y\ge\varepsilon) $$

两边同除以 $\varepsilon>0$ 即得结论。该证明不要求 $Y$ 存在概率密度，因此同时适用于离散、连续与其他分布。$\blacksquare$

> **引理 1.2（Chebyshev 不等式）**
> 设随机变量 $X$ 的均值为 $\mu = \mathbb{E}[X]$，方差为 $\sigma^2 = \text{Var}(X) < \infty$。则对任意 $\varepsilon > 0$，有
> $$ P\left(\vert X - \mu\vert \ge \varepsilon\right) \le \frac{\text{Var}(X)}{\varepsilon^2} $$

**证明**：注意到事件 $\{\vert X - \mu\vert \ge \varepsilon\}$ 与事件 $\{(X - \mu)^2 \ge \varepsilon^2\}$ 完全等价。令非负随机变量 $Y = (X - \mu)^2$，对 $Y$ 应用引理 1.1（**Markov 不等式**），并代入 $\mathbb{E}[Y] = \mathbb{E}[(X-\mu)^2] = \text{Var}(X)$，可得：

$$ P\left(\vert X - \mu\vert \ge \varepsilon\right) = P\left((X - \mu)^2 \ge \varepsilon^2\right) \le \frac{\mathbb{E}\left[(X - \mu)^2\right]}{\varepsilon^2} = \frac{\text{Var}(X)}{\varepsilon^2} \quad \blacksquare $$

<a id="weak-lln"></a>

## 1.3 弱大数定律（Weak Law of Large Numbers）

**大数定律**揭示了算术平均值的稳定性：*当试验次数无限增加时，样本均值依概率收敛于数学期望（总体均值）*。本节利用 1.2 节的尾概率不等式，先证明独立同分布序列在仅一阶矩有限时的 **Khinchin 弱大数定律**，再直接推出作为其分布特例的 **Bernoulli 弱大数定律**，最后讨论放宽独立同分布限制、但要求二阶矩有限的 **Chebyshev 与 Markov 弱大数定律**。

### 1.3.1 Khinchin 弱大数定律（独立同分布且一阶矩有限）

**Khinchin（辛钦）弱大数定律**证明了：对于独立同分布序列，只要一阶矩（数学期望）有限，弱大数定律成立。

> **定理 1.1（Khinchin 弱大数定律）**
> 设 $X_1, X_2, \dots$ 独立同分布，且 $\mathbb{E}[\vert X_1\vert ] < \infty$，记 $\mathbb{E}[X_1] = \mu$，则有
> $$ \bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu \quad (n \to \infty) $$

**证明**：
通过设定截断阈值 $M > 0$，将每个 $X_i$ 拆解为**有界主部 $U_i$** 与**极端尾部 $V_i$** 之和：

$$ X_i = U_i + V_i, \quad \ U_i = X_i \mathbf{1}_{\{\vert X_i\vert \le M\}},\ \ V_i = X_i \mathbf{1}_{\{\vert X_i\vert > M\}} $$

相应地，样本均值与总体均值也分解为 $\bar{X}_n = \bar{U}_n + \bar{V}_n$ 以及 $\mu = \mu_U + \mu_V$（其中 $\mu_U = \mathbb{E}[U_1]$，$\mu_V = \mathbb{E}[V_1]$）。由三角不等式有：

$$ \vert \bar{X}_n - \mu\vert = \vert (\bar{U}_n - \mu_U) + \bar{V}_n - \mu_V\vert \le \vert \bar{U}_n - \mu_U\vert + \vert \bar{V}_n\vert + \vert \mu_V\vert $$

1. **利用一阶矩有限性控制尾部期望**：
   由于 $\mathbb{E}[\vert X_1\vert ] < \infty$，当截断阈值 $M \to \infty$ 时，尾部积分（或级数余项）必趋于零：$\lim_{M\to\infty} \mathbb{E}[\vert V_1\vert ] = \lim_{M\to\infty} \mathbb{E}\big[\vert X_1\vert \mathbf{1}_{\{\vert X_1\vert > M\}}\big] = 0$。
   因此，对任意给定的 $\varepsilon > 0$ 与 $\delta > 0$，可先取定一个充分大的常数 $M > 0$，使得尾部绝对期望满足：

   $$ \vert \mu_V\vert \le \mathbb{E}\big[\vert V_1\vert \big] < \frac{\varepsilon}{3}\min(1, \delta) \le \frac{\varepsilon}{3} $$

   此时，若事件 $\{\vert \bar{X}_n - \mu\vert > \varepsilon\}$ 发生，由 $\vert \bar{U}_n - \mu_U\vert + \vert \bar{V}_n\vert > \frac{2}{3}\varepsilon$ 知，必然有事件 $\{\vert \bar{U}_n - \mu_U\vert > \frac{\varepsilon}{3}\}$ 或 $\{\vert \bar{V}_n\vert > \frac{\varepsilon}{3}\}$ 至少发生其一。由概率次可加性得：

   $$ P\left(\vert \bar{X}_n - \mu\vert > \varepsilon\right) \le P\left(\vert \bar{U}_n - \mu_U\vert > \frac{\varepsilon}{3}\right) + P\left(\vert \bar{V}_n\vert > \frac{\varepsilon}{3}\right) $$

2. **对有界主部 $\bar{U}_n$ 应用 Chebyshev 不等式**：
   由于 $\vert U_i\vert \le M$ 有界，其方差必然有限：$\text{Var}(U_1) \le \mathbb{E}[U_1^2] \le M^2 < \infty$。又因 $U_1, \dots, U_n$ 独立同分布，样本均值方差为 $\text{Var}(\bar{U}_n) = \frac{1}{n}\text{Var}(U_1) \le \frac{M^2}{n}$。由引理 1.2（**Chebyshev 不等式**）得：

   $$ P\left(\vert \bar{U}_n - \mu_U\vert > \frac{\varepsilon}{3}\right) \le \frac{\text{Var}(\bar{U}_n)}{(\varepsilon / 3)^2} \le \frac{9M^2}{n\varepsilon^2} $$

3. **对极端尾部 $\bar{V}_n$ 应用 Markov 不等式**：
   对非负随机变量 $\vert \bar{V}_n\vert \le \frac{1}{n}\sum_{i=1}^n \vert V_i\vert $ 应用引理 1.1（**Markov 不等式**），结合 $\mathbb{E}[\vert V_1\vert ] < \frac{\varepsilon}{3}\delta$，得：

   $$ P\left(\vert \bar{V}_n\vert > \frac{\varepsilon}{3}\right) \le \frac{\mathbb{E}\big[\vert \bar{V}_n\vert \big]}{\varepsilon / 3} \le \frac{3}{\varepsilon}\,\mathbb{E}\big[\vert V_1\vert \big] < \delta $$

4. **取极限完成证明**：综合上述两项，对任意 $n \ge 1$ 有 $P(\vert \bar{X}_n - \mu\vert > \varepsilon) \le \frac{9M^2}{n\varepsilon^2} + \delta$。固定 $M$ 令 $n \to \infty$，第一项趋于 $0$，故 $\limsup_{n\to\infty} P(\vert \bar{X}_n - \mu\vert > \varepsilon) \le \delta$。再由 $\delta > 0$ 的任意性，即得 $\lim_{n\to\infty} P(\vert \bar{X}_n - \mu\vert > \varepsilon) = 0$。$\blacksquare$

### 1.3.2 Bernoulli 弱大数定律（频率收敛于概率）

雅各布·伯努利（**Jacob Bernoulli**）于 1713 年发表的 **Bernoulli 大数定律**是概率论历史上的第一个大数定律。尽管它在历史上最先被发现，但在数学逻辑上，它恰好是定理 1.1（**Khinchin 弱大数定律**）在 **0-1 分布（伯努利分布）**下的直接特例：当每次试验的随机变量 $X_i \in \{0, 1\}$ 仅用于指示某事件 $A$ 是否发生时，样本均值 $\bar{X}_n$ 即为事件发生的频率 $\frac{S_n}{n}$，总体均值 $\mu$ 即为事件发生的概率 $p$。

> **推论 1.1（Bernoulli 弱大数定律）**
> 在 $n$ 重独立重复试验中，设事件 $A$ 在每次试验中发生的概率为 $p \in (0, 1)$。定义第 $i$ 次试验的指示随机变量 $X_i = \mathbf{1}_A(\omega_i) \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$（事件发生取 $1$，不发生取 $0$），记 $S_n = \sum_{i=1}^n X_i$ 为事件 $A$ 发生的频数。则事件发生的频率依概率收敛于概率 $p$：
> $$ \bar{X}_n = \frac{S_n}{n} \xrightarrow{P} p = \mu \quad (n \to \infty) $$

**证明**：
由于指示变量序列 $X_1, X_2, \dots$ 独立同分布于 $\text{Bernoulli}(p)$，且其一阶绝对矩有限：

$$ \mathbb{E}\big[\vert X_1\vert \big] = \mathbb{E}[X_1] = 1 \cdot p + 0 \cdot (1 - p) = p < \infty $$

直接应用定理 1.1（**Khinchin 弱大数定律**），立得样本均值 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i = \frac{S_n}{n} \xrightarrow{P} \mathbb{E}[X_1] = p$。$\blacksquare$

### 1.3.3 Chebyshev 与 Markov 弱大数定律（二阶矩条件下的非同分布推广）

定理 1.1（**Khinchin**）仅要求一阶矩有限，但要求序列必须“独立且同分布”。如果随机变量序列**不同分布**，甚至变量之间**存在相关性**，我们只需将矩条件加强至**二阶矩（方差）有限**，直接对样本均值应用引理 1.2（**Chebyshev 不等式**），即可得到如下两个推广版本。

> **定理 1.2（Markov 与 Chebyshev 弱大数定律）**
> 设随机变量序列 $X_1, X_2, \dots$ 具有有限二阶矩，记 $\mu_i = \mathbb{E}[X_i]$。
> 1. **Markov 条件（部分和方差条件）**：若部分和的方差满足
> $$ \lim_{n\to\infty} \frac{1}{n^2}\text{Var}\left(\sum_{i=1}^n X_i\right) = 0 $$
> 则有
> $$ \bar{X}_n - \frac{1}{n}\sum_{i=1}^n \mu_i \xrightarrow{P} 0 $$
> 2. **Chebyshev 条件（两两不相关且方差一致有界）**：特别地，若 $X_1, \dots, X_n$ 两两不相关（$\text{Cov}(X_i, X_j) = 0,\ \forall i \neq j$），具有共同均值 $\mathbb{E}[X_i] = \mu$，且方差存在一致上界 $\text{Var}(X_i) \le C < \infty$，则
> $$ \bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu \quad (n \to \infty) $$

**证明**：记 $$\bar{\mu}_n = \frac{1}{n}\sum_{i=1}^n \mu_i = \mathbb{E}[\bar{X}_n]$$。对任意给定的 $\varepsilon > 0$，将引理 **1.2（Chebyshev 不等式）** 应用于随机变量 $\bar{X}_n$，可得：

$$ 0 \le P\left(\left\vert \bar{X}_n - \bar{\mu}_n\right\vert \ge \varepsilon\right) \le \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{1}{n^2 \varepsilon^2}\text{Var}\left(\sum_{i=1}^n X_i\right) $$

在 **Markov** 条件下，令 $n \to \infty$，上式右端直接趋于 $0$，即得 $\bar{X}_n - \bar{\mu}_n \xrightarrow{P} 0$。

在 **Chebyshev** 条件下，由两两不相关性知方差可加，故有：

$$ \frac{1}{n^2}\text{Var}\left(\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) \le \frac{nC}{n^2} = \frac{C}{n} \xrightarrow{n\to\infty} 0 $$

从而 **Markov** 条件自然满足，且 $\bar{\mu}_n = \mu$，故 $\bar{X}_n \xrightarrow{P} \mu$。（特别地，对于推论 1.1 的 **Bernoulli** 分布，因 $\text{Var}(X_i) = p(1-p) \le \frac{1}{4}$，上式直接给出有限样本尾概率界 $P(\vert \frac{S_n}{n} - p\vert \ge \varepsilon) \le \frac{1}{4n\varepsilon^2}$）。$\blacksquare$

<a id="strong-lln"></a>

## 1.4 强大数定律（Strong Law of Large Numbers）

弱大数定律（$\bar{X}_n \xrightarrow{P} \mu$）仅保证在单个固定的大型时刻 $n$，样本均值偏离 $\mu$ 的概率很小；但它并不排除随着时间 $n$ 不断推移，样本轨道在未来的某些时刻再次发生大幅度偏离的可能性。为了刻画**整条样本轨道随 $n \to \infty$ 的收敛性**，需要引入更强的收敛概念。

> **定义 1.2（几乎处处收敛 / 以概率 1 收敛）**
> 设 $\{X_n\}$ 与 $X$ 为定义在同一概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机变量。若满足
> $$ P\left(\left\{\omega \in \Omega:\ \lim_{n\to\infty} X_n(\omega) = X(\omega)\right\}\right) = 1 $$
> 则称序列 $\{X_n\}$ **几乎处处收敛（Almost Sure Convergence）**于 $X$，记作 $X_n \xrightarrow{a.s.} X$。

强大数定律指出样本均值不仅在单时刻接近总体均值，而且整条样本均值轨迹几乎必然收敛并稳定于总体均值。

> **定理 1.3（Kolmogorov 强大数定律）**
> 设 $X_1, X_2, \dots$ 为独立同分布的随机变量序列，$\mu$ 为有限常数。样本均值满足 $$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{a.s.} \mu$$ 的充要条件是 $\mathbb{E}[\vert X_1\vert ] < \infty$ 且 $\mathbb{E}[X_1] = \mu$。


<a id="distribution-convergence-clt"></a>


# 2. 依分布收敛与中心极限定理

<a id="conv-in-distribution"></a>

## 2.1 依分布收敛的定义

大数定律表明估计误差满足 $\bar{X}_n - \mu \xrightarrow{P} 0$，但这仅给出了点估计的收敛性。为了进一步构造置信区间或进行假设检验，必须刻画估计量在逼近真值过程中的分布形态。

> **定义 2.1（依分布收敛）**
> 设 $\{X_n\}$ 为随机变量序列，其累积分布函数（**CDF**）为 $F_{X_n}(x) = P(X_n \le x)$；设 $X$ 的分布函数为 $F_X(x) = P(X \le x)$。若在 $F_X$ 的所有连续点 $x \in C(F_X)$ 上，均有
> $$ \lim_{n\to\infty} F_{X_n}(x) = F_X(x) $$
> 则称序列 $\{X_n\}$ **依分布收敛（Convergence in Distribution，又称弱收敛 Weak Convergence）**于 $X$，记作 $X_n \xrightarrow{d} X$。

> **注（为何仅要求在分布函数的连续点上收敛）**
> 考察确定性退化序列 $X_n \equiv \frac{1}{n}$，其极限显然应当为常数 $X \equiv 0$。写出两者的分布函数：
> 
> $$ F_{X_n}(x) = \begin{cases} 0, & x < \frac{1}{n} \\ 1, & x \ge \frac{1}{n} \end{cases}, \quad F_X(x) = \begin{cases} 0, & x < 0 \\ 1, & x \ge 0 \end{cases} $$
> 
> ![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-002.png)
> 如上图所示：
> * 对任意 $x < 0$，当 $n$ 充分大时 $\frac{1}{n} > 0 > x$，故 $\lim_{n\to\infty} F_{X_n}(x) = 0 = F_X(x)$；
> * 对任意 $x > 0$，当 $n > \frac{1}{x}$ 时 $\frac{1}{n} < x$，故 $\lim_{n\to\infty} F_{X_n}(x) = 1 = F_X(x)$；
> * 唯独在 $F_X(x)$ 的**跳跃间断点 $x = 0$** 处，由于 $\frac{1}{n} > 0$ 恒成立，导致对任意有限的 $n$ 均有 $F_{X_n}(0) = P(\frac{1}{n} \le 0) = 0$，从而 $\lim_{n\to\infty} F_{X_n}(0) = 0 \neq F_X(0) = 1$。
> 
> 由此可见，阶梯函数在水平平移逼近极限位置时，在跳跃边界点处可能出现极限值与函数值不相等的现象。由于单调有界函数的不连续点至多只有可数个，剔除不连续点既消除了这种边界伪反例，又唯一确定了极限分布函数。

在现代概率论中，判断依分布收敛最核心的等价桥梁是 **Portmanteau 定理**（它将分布函数收敛转化为光滑有界测试函数的数学期望收敛）：

> **定理 2.1（Portmanteau 定理）**
> 对于随机变量序列 $\{X_n\}$ 与 $X$，下列命题完全等价：
> 1. $X_n \xrightarrow{d} X$；
> 2. 对任意有界连续函数（或具有有界三阶连续导数的光滑函数）$h: \mathbb{R} \to \mathbb{R}$，均有 $\lim_{n\to\infty} \mathbb{E}[h(X_n)] = \mathbb{E}[h(X)]$；
> 3. 对任意闭集 $C \subset \mathbb{R}$，有 $\limsup_{n\to\infty} P(X_n \in C) \le P(X \in C)$；
> 4. 对任意开集 $G \subset \mathbb{R}$，有 $\liminf_{n\to\infty} P(X_n \in G) \ge P(X \in G)$。

<a id="clt"></a>

## 2.2 中心极限定理（Central Limit Theorem）

大数定律指出样本均值的误差满足 $\bar{X}_n - \mu \xrightarrow{P} 0$。若直接考察 $\bar{X}_n - \mu$ 的极限分布，只能得到退化于 $0$ 的单点分布。为了观察到非退化的误差分布形态，需要寻找恰当的放大因子 $a_n \to \infty$，使得 $a_n(\bar{X}_n - \mu)$ 的方差稳定在非零常数。

注意到 $\text{Var}\big(a_n(\bar{X}_n - \mu)\big) = a_n^2 \cdot \frac{\sigma^2}{n}$。要使该方差不随 $n\to\infty$ 退化为 $0$ 或发散至 $\infty$，放大因子必须与 $\sqrt{n}$ 同阶，即 **$a_n \propto \sqrt{n}$**；通常取 $a_n = \sqrt{n}$，使极限方差恰为 $\sigma^2$。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-asymptotic1-003.png)

### 2.2.1 独立同分布的中心极限定理（Lindeberg-Lévy）

> **定理 2.2（Lindeberg-Lévy 中心极限定理）**
> 设 $X_1, X_2, \dots$ 独立同分布，具有均值 $\mathbb{E}[X_i] = \mu$ 与有限的正方差 $\text{Var}(X_i) = \sigma^2 \in (0, \infty)$。则当 $n \to \infty$ 时，
> $$ Z_n = \frac{\sqrt{n}\left(\bar{X}_n - \mu\right)}{\sigma} \xrightarrow{d} Z \sim N(0, 1) $$
> 等价地，$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$。

**证明（利用 Lindeberg 逐个替换法与泰勒展开）**：

**(1) 标准化与构造正态替身序列**：
令 $Y_i = \frac{X_i - \mu}{\sigma}$，则 $Y_1, \dots, Y_n$ 独立同分布，满足 $\mathbb{E}[Y_i] = 0,\ \mathbb{E}[Y_i^2] = 1$，且目标统计量为 $Z_n = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i$。
另取一组与 $\{Y_i\}$ 完全独立的标准正态随机变量序列 $G_1, G_2, \dots, G_n \stackrel{\text{i.i.d.}}{\sim} N(0, 1)$。由正态分布的可加性，其标准化和 $W_n := \frac{1}{\sqrt{n}}\sum_{i=1}^n G_i \sim N(0, 1)$ 精确服从标准正态分布。
根据定理 2.1（**Portmanteau 定理**），要证 $Z_n \xrightarrow{d} N(0, 1)$，只需证明对任意具有有界二阶与三阶连续导数的光滑测试函数 $h: \mathbb{R} \to \mathbb{R}$（记 $M_2 = \sup_x \vert h''(x)\vert < \infty,\ M_3 = \sup_x \vert h'''(x)\vert < \infty$），均有：

$$ \lim_{n\to\infty} \Big\vert \mathbb{E}\big[h(Z_n)\big] - \mathbb{E}\big[h(W_n)\big]\Big\vert = 0 $$

**(2) 构造混合部分和并作裂项拆解**：
对 $i = 0, 1, \dots, n$，定义包含“前 $i$ 个原变量 $Y$、后 $n-i$ 个正态替身 $G$”的混合部分和 $S_i$，以及剔除第 $i$ 个位置后的公共背景和 $U_i$：

$$ S_i := \frac{Y_1 + \cdots + Y_i + G_{i+1} + \cdots + G_n}{\sqrt{n}}, \qquad U_i := \frac{Y_1 + \cdots + Y_{i-1} + G_{i+1} + \cdots + G_n}{\sqrt{n}} $$

显然有边界关系 $S_n = Z_n$，$S_0 = W_n$，且相邻两步仅在第 $i$ 个分量上将 $G_i$ 替换为了 $Y_i$：

$$ S_i = U_i + \frac{Y_i}{\sqrt{n}}, \qquad S_{i-1} = U_i + \frac{G_i}{\sqrt{n}} $$

由裂项相消恒等式 $h(Z_n) - h(W_n) = \sum_{i=1}^n \big[h(S_i) - h(S_{i-1})\big]$ 与三角不等式，总期望差受控于 $n$ 次单步替换误差之和：

$$ \Big\vert \mathbb{E}\big[h(Z_n)\big] - \mathbb{E}\big[h(W_n)\big]\Big\vert \le \sum_{i=1}^n \left\vert \mathbb{E}\left[h\left(U_i + \frac{Y_i}{\sqrt{n}}\right) - h\left(U_i + \frac{G_i}{\sqrt{n}}\right)\right]\right\vert $$

**(3) 在公共背景 $U_i$ 处作泰勒展开并消去前两阶矩**：
对任意增量 $\Delta$，将 $h(U_i + \Delta)$ 在点 $U_i$ 处作二阶泰勒展开：$h(U_i + \Delta) = h(U_i) + h'(U_i)\Delta + \frac{1}{2}h''(U_i)\Delta^2 + R(U_i, \Delta)$。
由带 **Lagrange** 余项的二阶与三阶泰勒公式，余项 $R(U_i, \Delta) = \frac{1}{2}\big[h''(\xi) - h''(U_i)\big]\Delta^2$ 同时满足两个上界：$\vert R\vert \le M_2 \Delta^2$ 且 $\vert R\vert \le \frac{1}{6}M_3 \vert \Delta\vert ^3$，故恒有组合上界 $\vert R(U_i, \Delta)\vert \le \min\big(M_2 \Delta^2,\ \frac{1}{6}M_3 \vert \Delta\vert ^3\big)$。
分别令 $\Delta = \frac{Y_i}{\sqrt{n}}$ 与 $\Delta = \frac{G_i}{\sqrt{n}}$ 并相减取期望。注意**公共背景 $U_i$ 与 $Y_i, G_i$ 完全独立**，且 $Y_i$ 与正态变量 $G_i$ 的**前两阶矩完全相同**（$\mathbb{E}[Y_i] = \mathbb{E}[G_i] = 0$，$\mathbb{E}[Y_i^2] = \mathbb{E}[G_i^2] = 1$），故零阶、一阶、二阶项在取期望后**全部精确抵消为零**：

$$ \begin{aligned} &\mathbb{E}\left[h\left(U_i + \frac{Y_i}{\sqrt{n}}\right) - h\left(U_i + \frac{G_i}{\sqrt{n}}\right)\right] \\ &= \frac{1}{\sqrt{n}}\mathbb{E}[h'(U_i)]\underbrace{\mathbb{E}[Y_i - G_i]}_{= 0} + \frac{1}{2n}\mathbb{E}[h''(U_i)]\underbrace{\mathbb{E}[Y_i^2 - G_i^2]}_{= 1 - 1 = 0} + \mathbb{E}\left[R\left(U_i, \frac{Y_i}{\sqrt{n}}\right) - R\left(U_i, \frac{G_i}{\sqrt{n}}\right)\right] \end{aligned} $$

**(4) 累加余项并取极限**：
若三阶绝对矩有限 $\gamma_3 = \mathbb{E}[\vert Y_1\vert ^3] < \infty$，直接取三阶余界 $\vert R\vert \le \frac{M_3}{6 n^{3/2}}(\vert Y_i\vert ^3 + \vert G_i\vert ^3)$，对 $n$ 项求和立得：

$$ \Big\vert \mathbb{E}\big[h(Z_n)\big] - \mathbb{E}\big[h(W_n)\big]\Big\vert \le \sum_{i=1}^n \frac{M_3\big(\gamma_3 + \mathbb{E}[\vert G_1\vert ^3]\big)}{6 n^{3/2}} = \frac{M_3\big(\gamma_3 + \mathbb{E}[\vert G_1\vert ^3]\big)}{6\sqrt{n}} \xrightarrow{n\to\infty} 0 $$

（若仅二阶矩 $\mathbb{E}[Y_1^2] = 1$ 有限，只需对任意 $\varepsilon > 0$ 按 $\{\vert Y_1\vert \le \varepsilon\sqrt{n}\}$ 选取 $\frac{M_3}{6}\vert \Delta\vert ^3$ 界，按 $\{\vert Y_1\vert > \varepsilon\sqrt{n}\}$ 选取 $M_2 \Delta^2$ 界，单步余项受控于 $\frac{1}{n}\big(\frac{\varepsilon M_3}{6}\mathbb{E}[Y_1^2] + M_2 \mathbb{E}[Y_1^2 \mathbf{1}_{\{\vert Y_1\vert > \varepsilon\sqrt{n}\}}]\big)$，$n$ 项求和后令 $n \to \infty$ 再令 $\varepsilon \to 0$ 同样趋于 $0$）。故 $Z_n \xrightarrow{d} N(0, 1)$。$\blacksquare$

### 2.2.2 二项分布的中心极限定理（De Moivre-Laplace）

与 **Bernoulli** 弱大数定律是 **Khinchin** 弱大数定律的特例完全平行，历史上第一个中心极限定理——**De Moivre-Laplace 定理**，恰好是定理 **2.2（Lindeberg-Lévy CLT）**在 **0-1 分布（伯努利试验）**下的直接特例。

> **推论 2.1（De Moivre-Laplace 中心极限定理）**
> 设 $S_n \sim \text{Binomial}(n, p)$ 表示 $n$ 重独立 **Bernoulli** 试验中成功的次数，其中 $p\in(0,1)$ 固定。则当 $n \to \infty$ 时，
> $$ \frac{S_n - np}{\sqrt{np(1 - p)}} \xrightarrow{d} N(0, 1) \quad (n \to \infty) $$

**证明**：将 $S_n$ 表示为独立同分布变量之和 $S_n = \sum_{i=1}^n X_i$，其中 $X_i \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$，均值为 $\mu = p$，方差为 $\sigma^2 = p(1 - p) > 0$。直接套用定理 2.2，即得 $\frac{\sqrt{n}(\bar{X}_n - p)}{\sqrt{p(1-p)}} = \frac{S_n - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0, 1)$。$\blacksquare$

### 2.2.3 独立不同分布与多维中心极限定理

在定理 2.2 的 Lindeberg 逐个替换法证明中，我们实际上并未强制要求 $Y_1, \dots, Y_n$ 同分布——只要把第 $i$ 个非正态变量 $X_i - \mu_i$ 替换为具有**相同方差 $\sigma_i^2$** 的独立正态替身 $G_i \sim N(0, \sigma_i^2)$，前两阶泰勒展开项依然精确抵消！唯一的要求是 $n$ 个三阶（或截断二阶）泰勒余项之和趋于零。这直接导出了**独立不同分布**下的两个著名中心极限定理：

> **定理 2.3（Lindeberg-Feller 与 Lyapunov 中心极限定理 —— 独立不同分布情形）**
> 设 $X_1, X_2, \dots$ 相互独立，均值分别为 $\mu_i = \mathbb{E}[X_i]$，方差分别为 $\sigma_i^2 = \text{Var}(X_i) > 0$，记部分和的总标准差为 $B_n = \sqrt{\sum_{i=1}^n \sigma_i^2}$。
> 1. **Lindeberg 条件（二阶矩下的充要条件）**：若对任意 $\varepsilon > 0$，超出 $\varepsilon B_n$ 阈值的截断尾方差占比趋于零：
>    $$ \lim_{n\to\infty} \frac{1}{B_n^2}\sum_{i=1}^n \mathbb{E}\Big[(X_i - \mu_i)^2 \mathbf{1}_{\{\vert X_i - \mu_i\vert > \varepsilon B_n\}}\Big] = 0 $$
>    （该条件成立且 $\max_{1\le i\le n} \sigma_i^2 / B_n^2 \to 0$ 是下述正态收敛的充要条件）；
> 2. **Lyapunov 条件（基于 $2+\delta$ 阶矩的充分条件）**：特别地，若存在常数 $\delta > 0$，使得
>    $$ \lim_{n\to\infty} \frac{1}{B_n^{2+\delta}}\sum_{i=1}^n \mathbb{E}\left[\vert X_i - \mu_i\vert ^{2+\delta}\right] = 0 $$
> 
> 则在 **Lindeberg** 条件（或 **Lyapunov** 条件）下，当 $n \to \infty$ 时均有：
> $$ Z_n = \frac{\sum_{i=1}^n (X_i - \mu_i)}{B_n} \xrightarrow{d} N(0, 1) $$

**证明**：先证 **Lyapunov** 条件蕴含 **Lindeberg** 条件：在事件 $\{\vert X_i - \mu_i\vert > \varepsilon B_n\}$ 上恒有 $\left(\frac{\vert X_i - \mu_i\vert }{\varepsilon B_n}\right)^\delta \ge 1$，故
$$ \frac{1}{B_n^2}\sum_{i=1}^n \mathbb{E}\Big[(X_i - \mu_i)^2 \mathbf{1}_{\{\vert X_i - \mu_i\vert > \varepsilon B_n\}}\Big] \le \frac{1}{\varepsilon^\delta B_n^{2+\delta}}\sum_{i=1}^n \mathbb{E}\left[\vert X_i - \mu_i\vert ^{2+\delta}\right] \xrightarrow{n\to\infty} 0 $$
在 **Lindeberg** 条件下，完全照搬定理 2.2 的逐个替换法，令 $\Delta_i = \frac{X_i - \mu_i}{B_n}$，$n$ 个单步替换的泰勒余项之和受控于 $\frac{\varepsilon M_3}{6}\sum_{i=1}^n \frac{\sigma_i^2}{B_n^2} + \frac{M_2}{B_n^2}\sum_{i=1}^n \mathbb{E}\big[(X_i - \mu_i)^2 \mathbf{1}_{\{\vert X_i - \mu_i\vert > \varepsilon B_n\}}\big]$。先令 $n \to \infty$，第二项由 **Lindeberg** 条件趋于 $0$；再由 $\varepsilon > 0$ 的任意性知总余项趋于 $0$，故 $Z_n \xrightarrow{d} N(0, 1)$。$\blacksquare$

当研究对象为 $d$ 维随机向量时，借助高维依分布收敛的线性投影判据（**Cramér-Wold 定理**：随机向量序列 $\mathbf{Y}_n \xrightarrow{d} \mathbf{Y}$ 当且仅当对一切固定常数向量 $\mathbf{a} \in \mathbb{R}^d$ 均有一维内积 $\mathbf{a}^\top \mathbf{Y}_n \xrightarrow{d} \mathbf{a}^\top \mathbf{Y}$），可将一维中心极限定理直接提升至多维空间：

> **定理 2.4（多维 Lindeberg-Lévy 中心极限定理）**
> 设 $\mathbf{X}_1, \mathbf{X}_2, \dots \in \mathbb{R}^d$ 为独立同分布的 $d$ 维随机向量序列，均值向量为 $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}_1]$，协方差矩阵为 $\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}_1) = \mathbb{E}[(\mathbf{X}_1 - \boldsymbol{\mu})(\mathbf{X}_1 - \boldsymbol{\mu})^\top]$（元素有限）。记样本均值向量为 $\bar{\mathbf{X}}_n = \frac{1}{n}\sum_{i=1}^n \mathbf{X}_i$，则当 $n \to \infty$ 时，
> $$ \sqrt{n}\left(\bar{\mathbf{X}}_n - \boldsymbol{\mu}\right) \xrightarrow{d} N_d\left(\mathbf{0},\ \boldsymbol{\Sigma}\right) $$

**证明**：对任意固定的常数向量 $\mathbf{a} \in \mathbb{R}^d$，构造一维标量序列 $U_i = \mathbf{a}^\top (\mathbf{X}_i - \boldsymbol{\mu})$。由于 $U_1, \dots, U_n$ 独立同分布，均值为 $\mathbb{E}[U_i] = 0$，方差为 $\text{Var}(U_i) = \mathbf{a}^\top \boldsymbol{\Sigma}\mathbf{a} < \infty$，由一维定理 2.2 知 $\mathbf{a}^\top \big[\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu})\big] = \frac{1}{\sqrt{n}}\sum_{i=1}^n U_i \xrightarrow{d} N(0, \mathbf{a}^\top \boldsymbol{\Sigma}\mathbf{a})$。而多维正态分布 $\mathbf{Z} \sim N_d(\mathbf{0}, \boldsymbol{\Sigma})$ 的一维投影恰好服从 $N(0, \mathbf{a}^\top \boldsymbol{\Sigma}\mathbf{a})$，由 **Cramér-Wold** 定理立得 $\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} N_d(\mathbf{0}, \boldsymbol{\Sigma})$。$\blacksquare$


<a id="convergence-relations"></a>

## 2.3 两种收敛模式的关系证明

在渐近理论中，依概率收敛与依分布收敛存在紧密的逻辑关联。

> **定理 2.5（依概率收敛蕴含依分布收敛）**
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

> **定理 2.6（退化极限下的等价性）**
> 设 $c$ 为常数，则 $X_n \xrightarrow{d} c \iff X_n \xrightarrow{P} c$。

**证明**：其中一个方向（$\Leftarrow$）已由定理 2.5 给出，仅需证明另一方向（$\Rightarrow$）。
设 $X_n \xrightarrow{d} c$，常数 $c$ 对应的退化分布函数为：

$$ F_c(x) = \begin{cases} 0, & x < c \\ 1, & x \ge c \end{cases} $$

显然，除点 $x = c$ 外，实轴上其余所有点均为 $F_c(x)$ 的连续点。对任意给定的 $\varepsilon > 0$，取 $x_1 = c - \varepsilon < c$ 与 $x_2 = c + \varepsilon > c$，则 $x_1, x_2$ 均为 $F_c$ 的连续点，故有：

$$ \lim_{n\to\infty} F_{X_n}(c - \varepsilon) = F_c(c - \varepsilon) = 0\\ \lim_{n\to\infty} F_{X_n}(c + \varepsilon) = F_c(c + \varepsilon) = 1 $$

利用事件包含关系 $\{\vert X_n - c\vert > \varepsilon\} = \{X_n < c - \varepsilon\} \cup \{X_n > c + \varepsilon\} \subset \{X_n \le c - \varepsilon\} \cup \{X_n > c + \varepsilon\}$，可得：

$$ \begin{aligned} 0 \le P\left(\vert X_n - c\vert > \varepsilon\right) &\le P\left(X_n \le c - \varepsilon\right) + P\left(X_n > c + \varepsilon\right) \\ &= F_{X_n}(c - \varepsilon) + \left[1 - F_{X_n}(c + \varepsilon)\right] \end{aligned} $$

令 $n \to \infty$，上式右端收敛于 $0 + (1 - 1) = 0$。由夹逼准则知 $\lim_{n\to\infty} P(\vert X_n - c\vert > \varepsilon) = 0$，即 $X_n \xrightarrow{P} c$。$\blacksquare$


<a id="exercise-mc-dropout"></a>

## 📖 例题：MC dropout 预测均值的蒙特卡洛误差

> **论文背景**：[**Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning**](https://arxiv.org/abs/1506.02142) (*ICML* 2016)
>
> **Gal** 与 **Ghahramani** 提出在测试阶段保留 **dropout** 随机掩码，对同一输入重复 $T$ 次前向传播，用这 $T$ 个输出的样本均值近似预测均值，用样本二阶矩减去均值平方（再加上观测噪声项 $\tau^{-1}$）近似预测方差，论文把这一做法称为 **MC dropout**。给定输入与训练完成的权重后，这 $T$ 个输出由独立重采样的掩码生成，因而独立同分布：样本均值的收敛性由大数定律给出，其误差量级与置信区间由中心极限定理给出。
>
> **题目描述**：固定输入 $x$ 与训练完成的权重，设 $T$ 次前向传播的标量输出 $Y_1, \dots, Y_T$ 独立同分布，记
> $$ \mu(x) = \mathbb{E}[Y_1], \qquad s^2(x) = \text{Var}(Y_1) \in (0, \infty) $$
> **MC dropout** 以 $$\hat{\mu}_T = \frac{1}{T}\sum_{t=1}^T Y_t$$ 作为预测均值的估计。
> 1. 求 $\mu(x)$ 的渐近 $1 - \alpha$ 置信区间。
> 2. 设 $s(x) = 0.2$，要使该置信区间（取 $\alpha = 0.05$）的半宽不超过 $0.01$，前向传播次数 $T$ 至少取多少？

**解答**：

**(1)** 由 $Y_t$ 独立同分布且 $0 < s^2(x) < \infty$，定理 2.2（**Lindeberg-Lévy** 中心极限定理）给出

$$ \sqrt{T}\left(\hat{\mu}_T - \mu(x)\right) \xrightarrow{d} N\left(0,\ s^2(x)\right) $$

记 $z_{\alpha/2}$ 为标准正态分布的上 $\alpha/2$ 分位数，则 $$P\big(\vert \hat{\mu}_T - \mu(x)\vert \le z_{\alpha/2}s(x)/\sqrt{T}\big) \to 1 - \alpha$$，故 $\mu(x)$ 的渐近 $1-\alpha$ 置信区间为

$$ \hat{\mu}_T \pm z_{\alpha/2}\frac{s(x)}{\sqrt{T}} $$

**(2)** 置信区间的半宽为 $z_{\alpha/2}s(x)/\sqrt{T}$，要求它不超过 $\delta$ 即

$$ z_{\alpha/2}\frac{s(x)}{\sqrt{T}} \le \delta \iff T \ge \left(\frac{z_{\alpha/2}\,s(x)}{\delta}\right)^2 $$

代入 $z_{0.025} \approx 1.96$、$s(x) = 0.2$、$\delta = 0.01$：

$$ T \ge \left(\frac{1.96 \times 0.2}{0.01}\right)^2 \approx 1536.6 \quad \Longrightarrow \quad T \ge 1537 $$

由于半宽按 $T^{-1/2}$ 下降，精度每提高一个数量级，所需前向传播次数就要增加约 $100$ 倍，这是蒙特卡洛估计的固有代价。
