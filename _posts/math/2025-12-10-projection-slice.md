---
layout: post
title: '投影切片定理(Projection-Slice Theorem)'
date: 2025-12-10
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-projection-slice-cover.png'
tags: 数学
---

> Projection-Slice Theorem.

在医学成像、电子显微和射电天文中，仪器往往无法直接测量物体内部的分布，只能测得这一分布沿一组直线或平面的积分。例如在 **X** 射线计算机断层扫描（**CT**）中，探测器记录的是射线穿过人体后的衰减，它由组织的衰减系数沿射线路径的线积分决定。从这些积分恢复原来的分布，是一个典型的逆问题（**inverse problem**）。

**投影切片定理（Projection-Slice Theorem）**，又称**傅里叶切片定理（Fourier Slice Theorem）**或**中心切片定理（Central Slice Theorem）**，给出了解决这一问题的关键联系：函数沿某一方向投影后的一维傅里叶变换，等于该函数的二维傅里叶变换沿同一方向、穿过原点的一条切片。空间域中的投影由此与频率域中的切片一一对应，用足够多方向的投影就能填满整个频率域，进而重建原函数。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-projection-slice-000.png)

# 1. 投影切片定理

## 1.1 投影与 Radon 变换

设 $f(x, y)$ 是 $\mathbb{R}^2$ 上的可积函数。记单位向量 $\mathbf{n}_\theta = (\cos\theta, \sin\theta)$ 及与之垂直的单位向量 $\mathbf{n}_\theta^\perp = (-\sin\theta, \cos\theta)$。方程 $x\cos\theta + y\sin\theta = t$ 表示法向量为 $\mathbf{n}_\theta$、到原点的有向距离为 $t$ 的直线；固定 $\theta$ 而让 $t$ 取遍 $\mathbb{R}$，就得到一族方向为 $\mathbf{n}_\theta^\perp$ 的平行直线。

函数 $f$ 沿这族直线的积分是一个关于 $t$ 的一维函数，称为 $f$ 在角度 $\theta$ 下的**投影（projection）**：

$$
\begin{aligned}
P_\theta(t) &= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y)\, \delta(x\cos\theta + y\sin\theta - t) \,dx\,dy \\&= \int_{-\infty}^{\infty} f\left(t\,\mathbf{n}_\theta + s\,\mathbf{n}_\theta^\perp\right) ds
\end{aligned}
$$

其中 $\delta(\cdot)$ 是狄拉克 $\delta$ 函数。第二个等号来自坐标旋转 $(x, y) = t\,\mathbf{n}_\theta + s\,\mathbf{n}_\theta^\perp$，该变换的雅可比行列式为 1，而 $\delta(x\cos\theta+y\sin\theta-t)$ 只在 $s$ 方向的直线上取值。因此 $P_\theta(t)$ 是沿 $\mathbf{n}_\theta^\perp$ 方向的线积分，积分结果排列在 $\mathbf{n}_\theta$ 方向的坐标轴 $t$ 上。全体投影 $(\theta, t)\mapsto P_\theta(t)$ 称为 $f$ 的 **Radon 变换（Radon transform）**；以 $\theta$ 和 $t$ 为坐标把它画成图像，称为**正弦图（sinogram）**。

## 1.2 傅里叶变换

函数 $f(x,y)$ 的二维傅里叶变换为 $F(u,v)$：

$$
F(u, v) = \mathcal{F}_2\{f(x, y)\} = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y)\, e^{-i2\pi(ux + vy)} \,dx\,dy
$$

其中 $(u, v)$ 是频率域坐标。投影函数 $P_\theta(t)$ 的一维傅里叶变换为 $S_\theta(\omega)$：

$$
S_\theta(\omega) = \mathcal{F}_1\{P_\theta(t)\} = \int_{-\infty}^{\infty} P_\theta(t)\, e^{-i2\pi\omega t} \,dt
$$

其中 $\omega$ 是沿 $\mathbf{n}_\theta$ 方向的频率。$f$ 可积时，由 **Fubini** 定理 $P_\theta$ 对几乎所有 $t$ 有定义且 $\int\vert P_\theta(t)\vert dt\le\iint\vert f\vert dx\,dy$，因此两个傅里叶变换都存在。

## 1.3 定理陈述与证明

**投影切片定理**：投影 $P_\theta(t)$ 的一维傅里叶变换，等于 $f$ 的二维傅里叶变换沿方向 $\mathbf{n}_\theta$ 穿过原点的切片。在频率域中这条切片的参数方程为 $u = \omega\cos\theta$、$v = \omega\sin\theta$，即

$$
\boxed{S_\theta(\omega) = F(\omega\cos\theta, \omega\sin\theta) = F(\omega\,\mathbf{n}_\theta)}
$$



**证明**：将 $P_\theta$ 的第二种表达式代入 $S_\theta(\omega)$ 的定义：

$$
S_\theta(\omega) = \int_{-\infty}^{\infty} \left[\int_{-\infty}^{\infty} f\left(t\,\mathbf{n}_\theta + s\,\mathbf{n}_\theta^\perp\right) ds\right] e^{-i2\pi\omega t}\, dt
$$

被积函数的绝对值为 $\left\vert f\left(t\,\mathbf{n}_\theta + s\,\mathbf{n}_\theta^\perp\right)\right\vert$，它在 $(t, s)$ 平面上可积，因此可以把二重积分作为整体处理。作坐标变换 $\mathbf{x} = (x, y) = t\,\mathbf{n}_\theta + s\,\mathbf{n}_\theta^\perp$（雅可比行列式为 1），并注意 $t = \mathbf{x}\cdot\mathbf{n}_\theta$：

$$
S_\theta(\omega) = \iint f(\mathbf{x})\, e^{-i2\pi\omega\,(\mathbf{x}\cdot\mathbf{n}_\theta)}\, d\mathbf{x} = \iint f(x, y)\, e^{-i2\pi\left(x\,\omega\cos\theta + y\,\omega\sin\theta\right)}\, dx\,dy
$$

上式正是 $F(u, v)$ 在 $u=\omega\cos\theta$、$v=\omega\sin\theta$ 处的值。$\blacksquare$

若使用含 $\delta$ 函数的定义，交换积分次序后利用筛选性质（**sifting property**）$\int_{-\infty}^{\infty}\delta(a - t)\,e^{-i2\pi\omega t}\,dt = e^{-i2\pi\omega a}$，其中 $a = x\cos\theta + y\sin\theta$，也得到同样的结果。

## 1.4 例：高斯函数与径向对称函数

取 $f(x,y)=e^{-\pi(x^2+y^2)}$。由于 $\Vert t\,\mathbf{n}_\theta + s\,\mathbf{n}_\theta^\perp\Vert^2 = t^2+s^2$，且 $\int_{-\infty}^{\infty}e^{-\pi s^2}ds = 1$，投影为 $P_\theta(t)=e^{-\pi t^2}$，其一维傅里叶变换为 $S_\theta(\omega)=e^{-\pi\omega^2}$；而二维傅里叶变换为 $F(u,v)=e^{-\pi(u^2+v^2)}$，在切片上恰好等于 $e^{-\pi\omega^2}$，与定理一致。

更一般地，若 $f(x,y)=g(r)$ 是径向对称函数（$r=\sqrt{x^2+y^2}$），投影与 $\theta$ 无关，令 $r^2=t^2+s^2$ 得

$$
P(t) = \int_{-\infty}^{\infty} g\left(\sqrt{t^2+s^2}\right) ds = 2\int_{\vert t\vert}^{\infty}\frac{g(r)\,r}{\sqrt{r^2-t^2}}\,dr
$$

这是 **Abel** 变换；而径向对称函数的二维傅里叶变换也是径向对称的，在极坐标下化为零阶 **Hankel** 变换 $G(\rho)=2\pi\int_0^\infty g(r)J_0(2\pi\rho r)\,r\,dr$，其中 $J_0$ 是零阶 **Bessel** 函数。投影切片定理说明：$g$ 的 **Abel** 变换的一维傅里叶变换就是 $g$ 的 **Hankel** 变换。

## 1.5 性质

*   **对称性**：直线 $\mathbf{x}\cdot\mathbf{n}_{\theta+\pi}=t$ 就是直线 $\mathbf{x}\cdot\mathbf{n}_{\theta}=-t$，因此 $P_{\theta+\pi}(t)=P_\theta(-t)$。平行束几何下只需采集 $\theta\in[0,\pi)$ 的投影。
*   **正弦图的名称**：对点源 $f=\delta(\mathbf{x}-\mathbf{x}_0)$，$\mathbf{x}_0=(r_0\cos\phi_0, r_0\sin\phi_0)$，投影为 $P_\theta(t)=\delta\left(t-\mathbf{x}_0\cdot\mathbf{n}_\theta\right)=\delta\left(t-r_0\cos(\theta-\phi_0)\right)$。点源在 $(\theta, t)$ 平面上的轨迹是一条正弦曲线，一般物体的正弦图是这些正弦曲线的叠加。
*   **卷积性质**：二维卷积的傅里叶变换是乘积 $FG$，其切片是两个切片的乘积 $S_\theta^{f}S_\theta^{g}$，再由一维卷积定理得

$$
P_\theta^{f*g} = P_\theta^{f} * P_\theta^{g}
$$

即二维卷积的投影等于两个投影的一维卷积。
*   **质量守恒**：令 $\omega=0$，$\int P_\theta(t)\,dt = S_\theta(0) = F(0,0) = \iint f\,dx\,dy$，每个角度的投影总和都等于 $f$ 的总积分。这是真实投影数据必须满足的一致性条件之一。

# 2. 高维推广

设 $f$ 是 $\mathbb{R}^n$ 上的可积函数，$V$ 是 $\mathbb{R}^n$ 的 $k$ 维子空间。取 $n\times k$ 矩阵 $U$ 与 $n\times(n-k)$ 矩阵 $W$，其列分别构成 $V$ 与正交补 $V^\perp$ 的标准正交基。定义 $f$ 在 $V$ 上的投影（沿 $V^\perp$ 积分）

$$
(\mathcal{P}_V f)(\boldsymbol{\tau}) = \int_{\mathbb{R}^{n-k}} f(U\boldsymbol{\tau} + W\boldsymbol{\sigma})\, d\boldsymbol{\sigma},\qquad \boldsymbol{\tau}\in\mathbb{R}^k
$$

则投影的 $k$ 维傅里叶变换等于 $f$ 的 $n$ 维傅里叶变换在 $V$ 上的切片：

$$
\boxed{\mathcal{F}_k\{\mathcal{P}_V f\}(\boldsymbol{\eta}) = \mathcal{F}_n\{f\}(U\boldsymbol{\eta}),\qquad \boldsymbol{\eta}\in\mathbb{R}^k}
$$

**证明**：$\mathcal{F}_k\{\mathcal{P}_V f\}(\boldsymbol{\eta}) = \int_{\mathbb{R}^k}\int_{\mathbb{R}^{n-k}} f(U\boldsymbol{\tau}+W\boldsymbol{\sigma})\,e^{-i2\pi\boldsymbol{\eta}\cdot\boldsymbol{\tau}}\,d\boldsymbol{\sigma}\,d\boldsymbol{\tau}$。作正交变换 $\mathbf{x}=U\boldsymbol{\tau}+W\boldsymbol{\sigma}$（雅可比行列式为 1），由 $U^\top U=I$、$U^\top W=0$ 得 $\boldsymbol{\tau}=U^\top\mathbf{x}$，于是 $\boldsymbol{\eta}\cdot\boldsymbol{\tau}=(U\boldsymbol{\eta})\cdot\mathbf{x}$，积分化为 $\int_{\mathbb{R}^n}f(\mathbf{x})\,e^{-i2\pi(U\boldsymbol{\eta})\cdot\mathbf{x}}\,d\mathbf{x}=\mathcal{F}_n\{f\}(U\boldsymbol{\eta})$。$\blacksquare$

$n=2$、$k=1$ 即第 1 节的定理。三维中的两种常见情形：

*   **$k=2$（沿直线积分）**：沿方向 $\mathbf{d}$ 对三维函数做线积分得到二维投影图像，其二维傅里叶变换是三维傅里叶变换中过原点、垂直于 $\mathbf{d}$ 的平面。这是冷冻电镜三维重建（4.3 节）和三维平行束 **X** 射线成像的基础。
*   **$k=1$（沿平面积分）**：在每个与 $\mathbf{n}$ 垂直的平面上积分得到一维投影，其傅里叶变换是三维傅里叶变换中过原点、方向为 $\mathbf{n}$ 的一条直线。这种平面积分称为三维 **Radon** 变换。

同样的结构也出现在光场成像中：[Fourier slice photography](https://dl.acm.org/doi/10.1145/1073204.1073256) 将一张照片视为四维光场的二维积分投影，从而把不同焦距的重聚焦照片对应为四维光场傅里叶变换中的二维切片。

# 3. 从投影重建

## 3.1 直接傅里叶重建

投影切片定理直接给出一种重建方法：对每个角度的投影做一维傅里叶变换，得到 $F$ 在一条过原点的径向线上的值；收集 $\theta\in[0,\pi)$ 的全部径向线，就得到 $F$ 在极坐标网格上的样本；把这些样本插值到笛卡尔网格上，再做二维快速傅里叶逆变换即得 $f$。

极坐标网格在低频处密、高频处疏，相邻径向线之间的距离与 $\vert\omega\vert$ 成正比。高频区域的插值误差较大，会在图像中产生伪影，因此直接傅里叶重建需要精心设计插值核。在非均匀快速傅里叶变换（**gridding**）的做法中，每个样本需要乘以与采样密度成反比的权重，这个权重正比于 $\vert\omega\vert$，与下文斜坡滤波器的来源相同。

## 3.2 未滤波反投影的模糊

**反投影（back-projection）**算子把每个投影值“涂抹”回它所对应的整条直线上：

$$
b(x, y) = \int_0^\pi P_\theta(x\cos\theta + y\sin\theta)\, d\theta
$$

即点 $(x,y)$ 处的值等于所有经过该点的直线上的投影值之和。反投影并不是 **Radon** 变换的逆。将 $P_\theta(t)=\int S_\theta(\omega)e^{i2\pi\omega t}d\omega$ 代入并利用投影切片定理：

$$
b(\mathbf{x}) = \int_0^\pi\int_{-\infty}^{\infty} F(\omega\,\mathbf{n}_\theta)\, e^{i2\pi\omega\,(\mathbf{n}_\theta\cdot\mathbf{x})}\, d\omega\, d\theta
$$

作变量替换 $\mathbf{k}=\omega\,\mathbf{n}_\theta$。当 $\omega\in\mathbb{R}$、$\theta\in[0,\pi)$ 时 $\mathbf{k}$ 恰好覆盖整个平面（除原点外一一对应），雅可比行列式为

$$
\left\vert\frac{\partial(u, v)}{\partial(\omega, \theta)}\right\vert = \left\vert\det\begin{pmatrix}\cos\theta & -\omega\sin\theta\\ \sin\theta & \omega\cos\theta\end{pmatrix}\right\vert = \vert\omega\vert
$$

故 $d\omega\,d\theta = d\mathbf{k}/\Vert\mathbf{k}\Vert$，从而 $b(\mathbf{x})=\iint \frac{F(\mathbf{k})}{\Vert\mathbf{k}\Vert}e^{i2\pi\mathbf{k}\cdot\mathbf{x}}\,d\mathbf{k}$，即 $b$ 的傅里叶变换为 $F(\mathbf{k})/\Vert\mathbf{k}\Vert$。在二维中 $1/\Vert\mathbf{x}\Vert$ 的傅里叶变换就是 $1/\Vert\mathbf{k}\Vert$：用极坐标计算，$\int_0^\infty\int_0^{2\pi}\frac{1}{r}e^{-i2\pi\rho r\cos\varphi}\,r\,d\varphi\,dr = 2\pi\int_0^\infty J_0(2\pi\rho r)\,dr = \frac{1}{\rho}$。由卷积定理得

$$
\boxed{b = f * \frac{1}{\Vert\mathbf{x}\Vert}}
$$

未滤波反投影得到的是原图与 $1/r$ 卷积后的模糊结果。以 $f=e^{-\pi r^2}$ 为例，直接计算反投影积分 $b(r)=\int_0^\pi e^{-\pi r^2\cos^2\theta}d\theta$，与在频率域计算 $F/\Vert\mathbf{k}\Vert$ 的逆变换 $2\pi\int_0^\infty e^{-\pi\rho^2}J_0(2\pi\rho r)\,d\rho$，在 $r=0, 0.5, 1, 2$ 处都得到 $3.1416, 2.2039, 1.1225, 0.5110$；当 $r$ 较大时 $b(r)\approx\frac{1}{r}\iint f = \frac{1}{r}$，衰减缓慢，表现为大范围的模糊（见 3.5 节的图）。

## 3.3 滤波反投影

把二维傅里叶逆变换写成极坐标形式，并利用上面的雅可比行列式 $\vert\omega\vert$：

$$
f(\mathbf{x}) = \iint F(\mathbf{k})\,e^{i2\pi\mathbf{k}\cdot\mathbf{x}}\,d\mathbf{k} = \int_0^\pi\int_{-\infty}^{\infty} F(\omega\,\mathbf{n}_\theta)\,\vert\omega\vert\,e^{i2\pi\omega\,(\mathbf{n}_\theta\cdot\mathbf{x})}\,d\omega\,d\theta
$$

由投影切片定理把 $F(\omega\,\mathbf{n}_\theta)$ 换成 $S_\theta(\omega)$，内层积分就是 $\vert\omega\vert S_\theta(\omega)$ 的一维傅里叶逆变换在 $t=\mathbf{n}_\theta\cdot\mathbf{x}$ 处的值。于是得到**滤波反投影（Filtered Back-Projection, FBP）**公式：

$$
\boxed{f(x, y) = \int_0^\pi Q_\theta(x\cos\theta + y\sin\theta)\,d\theta,\qquad Q_\theta = \mathcal{F}_1^{-1}\left\{\vert\omega\vert\,S_\theta(\omega)\right\}}
$$

即先对每个投影做**斜坡滤波（ramp filtering）**，再对滤波后的投影 $Q_\theta$ 做反投影。与 3.2 节对比可见，斜坡滤波器 $\vert\omega\vert$ 恰好抵消了反投影带来的 $1/\Vert\mathbf{k}\Vert$ 因子；它来自极坐标积分的雅可比行列式，而不是经验性的补偿。滤波与反投影都是逐个投影进行的，采集到一个角度就能处理一个角度，这是 **FBP** 在 **CT** 中被广泛使用的原因之一。

实际数据的探测器间隔为 $\Delta t$，投影的频率上限为 $W=\frac{1}{2\Delta t}$，因此使用带限斜坡滤波器 $\vert\omega\vert\,\mathbf{1}[\vert\omega\vert\le W]$。其冲激响应为

$$
h(t) = \int_{-W}^{W}\vert\omega\vert\,e^{i2\pi\omega t}\,d\omega = \frac{W\sin(2\pi Wt)}{\pi t} - \frac{\sin^2(\pi Wt)}{\pi^2t^2}
$$

在采样点 $t=k\Delta t$ 处，$\sin(2\pi Wt)=\sin(\pi k)=0$，$\sin^2(\pi Wt)=\sin^2(\pi k/2)$，于是

$$
h(k\Delta t) = \begin{cases}\frac{1}{4\Delta t^2}, & k=0\\ 0, & k\text{ 为非零偶数}\\ -\frac{1}{\pi^2k^2\Delta t^2}, & k\text{ 为奇数}\end{cases}
$$

这就是 **Ram-Lak** 滤波器，$k=0$ 处的值由 $h(0)=\int_{-W}^{W}\vert\omega\vert d\omega=W^2$ 得到。用这一空间核与投影做线性卷积（补零以避免循环卷积的混叠），比直接在离散傅里叶变换的网格上取 $\vert\omega\vert$ 更准确：后者在零频处取值为 0，且相当于周期卷积，会使重建图像整体偏低。在 3.5 节的实验中，直接取 $\vert\omega\vert$ 且不补零时，圆盘区域内的平均偏差为 $-0.033$，补零到 1024 点后为 $-0.004$，使用带限空间核时为 $+0.0001$。

斜坡滤波器随频率线性增大，会放大投影中的高频噪声。实践中常把它乘以一个在高频处衰减的窗函数，例如 **Hann** 窗 $\frac{1}{2}+\frac{1}{2}\cos\frac{\pi\omega}{W}$（零频处为 1，频率上限处为 0）或 **Shepp-Logan** 窗 $\text{sinc}\frac{\omega}{2W}$（$\text{sinc}\,x=\frac{\sin\pi x}{\pi x}$），以牺牲部分分辨率为代价换取更低的噪声，见 3.5 节的实验。

## 3.4 离散采样条件

设物体包含在直径为 $D$ 的圆内，每个投影在 $D$ 的宽度内有 $N=D/\Delta t$ 个探测器样本，共采集 $M$ 个角度（$\Delta\theta=\pi/M$）。按频域采样定理，支撑在直径 $D$ 内的函数，其傅里叶变换的采样间隔不应超过 $1/D$。每条径向线上的样本间隔可以满足这一要求，而相邻径向线之间的距离在最高频率 $W=\frac{1}{2\Delta t}$ 处最大，为 $W\Delta\theta$。要求 $W\Delta\theta\le\frac{1}{D}$，即

$$
\boxed{M \ge \frac{\pi}{2}N}
$$

角度数不足时，高频区域的径向线之间出现空隙，重建图像中会出现从高对比度结构向外发散的条纹伪影。这是一个保守的估计：它要求最高频率处也满足采样条件，而重建误差主要由能量集中的低频部分决定。

## 3.5 数值实验

下面用一个由 10 个椭圆组成、仿照 **Shepp-Logan** 体模构造的图像验证上述结论。椭圆的投影有解析表达式：对半轴为 $a$、$b$，旋转角为 $\phi$，中心为 $(x_0, y_0)$，密度为 $A$ 的椭圆，

$$
P_\theta(t) = \begin{cases}\frac{2Aab}{s^2}\sqrt{s^2-\tau^2}, & \tau^2<s^2\\ 0, & \text{其他}\end{cases},\qquad s^2=a^2\cos^2(\theta-\phi)+b^2\sin^2(\theta-\phi),\quad \tau = t-(x_0\cos\theta+y_0\sin\theta)
$$

其中 $\frac{2ab}{s^2}\sqrt{s^2-\tau^2}$ 是直线与椭圆相交的弦长。用解析投影生成正弦图，可以避免正向投影的离散化误差。首先定义体模与解析投影：

```python
import numpy as np
import matplotlib.pyplot as plt

# 椭圆体模：(密度增量 A, 半轴 a, 半轴 b, 中心 x0, 中心 y0, 旋转角 φ/度)
ELLIPSES = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0),
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18),
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0),
]

def grid(n):
    x = np.linspace(-1, 1, n)
    return np.meshgrid(x, -x)          # 第 0 行对应 y = +1

def phantom(n):
    X, Y = grid(n)
    img = np.zeros((n, n))
    for A, a, b, x0, y0, phi in ELLIPSES:
        p = np.deg2rad(phi)
        xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
        yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
        img += A * ((xr / a) ** 2 + (yr / b) ** 2 <= 1)
    return img

def radon_ellipses(thetas, t):
    # 椭圆的投影有解析式：P(t) = 2Aab·sqrt(s² - t²)/s²，s² = a²cos²(θ-φ) + b²sin²(θ-φ)
    sino = np.zeros((len(thetas), len(t)))
    for A, a, b, x0, y0, phi in ELLIPSES:
        p = np.deg2rad(phi)
        for i, th in enumerate(thetas):
            s2 = (a * np.cos(th - p)) ** 2 + (b * np.sin(th - p)) ** 2
            tt = t - (x0 * np.cos(th) + y0 * np.sin(th))
            inside = tt ** 2 < s2
            sino[i, inside] += 2 * A * a * b * np.sqrt(s2 - tt[inside] ** 2) / s2
    return sino
```

然后实现斜坡滤波与反投影：

```python
def ramp_kernel(n, dt):
    # 带限斜坡滤波器的空间核：h(0) = 1/(4Δt²)，奇数 k 处为 -1/(π²k²Δt²)，偶数 k 处为 0
    k = np.arange(-(n - 1), n)
    h = np.zeros(len(k))
    h[k == 0] = 1 / (4 * dt ** 2)
    odd = k % 2 == 1
    h[odd] = -1 / (np.pi ** 2 * k[odd] ** 2 * dt ** 2)
    return h

def filter_projections(sino, dt, window=None):
    n = sino.shape[1]
    L = int(2 ** np.ceil(np.log2(3 * n)))     # 补零到足够长度，避免循环卷积的混叠
    H = np.fft.fft(ramp_kernel(n, dt), L)
    if window == 'hann':                       # Hann 窗：零频处为 1，Nyquist 频率处为 0
        H = H * (0.5 + 0.5 * np.cos(2 * np.pi * np.fft.fftfreq(L)))
    Q = np.fft.ifft(np.fft.fft(sino, L, axis=1) * H, axis=1).real
    return Q[:, n - 1:2 * n - 1] * dt

def backproject(Q, thetas, t, n):
    # f(x, y) ≈ (π/M) Σ_m Q_m(x cosθ_m + y sinθ_m)，用线性插值读取 Q_m
    X, Y = grid(n)
    img = np.zeros((n, n))
    for q, th in zip(Q, thetas):
        img += np.interp(X * np.cos(th) + Y * np.sin(th), t, q, left=0, right=0)
    return img * np.pi / len(thetas)

def fbp(sino, thetas, t, n, window=None):
    return backproject(filter_projections(sino, t[1] - t[0], window), thetas, t, n)
```

运行实验。图像为 $256\times256$ 像素，覆盖 $[-1,1]^2$；探测器间隔 $\Delta t = 2/256$，取物体直径 $D=2$，则 $N=256$，采样条件给出 $M\ge\frac{\pi}{2}\times256\approx 402$：

```python
n = 256                                         # 图像为 n×n 像素，覆盖 [-1, 1]²
dt = 2 / n                                      # 探测器间隔
t = (np.arange(-192, 192) + 0.5) * dt           # 探测器覆盖 [-1.5, 1.5]，包含图像对角线
truth = phantom(n)
X, Y = grid(n)
disc = X ** 2 + Y ** 2 <= 1

def rmse(img):
    return np.sqrt(np.mean((img - truth)[disc] ** 2))

# 1. 离散情形下 θ = 0 的投影切片定理精确成立
assert np.allclose(np.fft.fft(truth.sum(axis=0)), np.fft.fft2(truth)[0, :])

# 2. 解析投影与图像逐列求和（θ = 0）的一致性
dx = 2 / (n - 1)
p0 = radon_ellipses(np.array([0.0]), np.linspace(-1, 1, n))[0]
print('max |解析投影 - 离散投影| =', np.abs(p0 - truth.sum(axis=0) * dx).max())

# 3. 投影数 M 对重建误差的影响（采样条件 M >= πN/2，N = 2/Δt）
print('采样条件 πN/2 =', np.pi / 2 * (2 / dt))
for M in (30, 100, 200, 402, 800):
    thetas = np.arange(M) * np.pi / M
    print(f'M = {M:4d}: RMSE = {rmse(fbp(radon_ellipses(thetas, t), thetas, t, n)):.4f}')

# 4. 斜坡滤波器的窗函数：无噪声与含噪声（噪声标准差为投影最大值的 5%）
M = 402
thetas = np.arange(M) * np.pi / M
sino = radon_ellipses(thetas, t)
noisy = sino + np.random.default_rng(0).normal(0, 0.05 * sino.max(), sino.shape)
flat = X ** 2 + (Y - 0.35) ** 2 < 0.004          # 第 5 个椭圆内部的均匀区域
recon = {}
for name, s in (('clean', sino), ('noisy', noisy)):
    for window in ('ram-lak', 'hann'):
        img = fbp(s, thetas, t, n, None if window == 'ram-lak' else window)
        recon[name, window] = img
        print(f'{name} {window:8s} RMSE = {rmse(img):.4f}  均匀区域标准差 = {img[flat].std():.4f}')
```

得到的结果如下。离散情形下 $\theta=0$ 的定理精确成立（断言通过：对图像逐列求和后做一维离散傅里叶变换，恰好等于二维离散傅里叶变换的第 0 行）；解析投影与图像逐列求和的最大差为 $0.012$，来自椭圆边缘在像素网格上的离散化。投影数 $M$ 与重建误差（单位圆内的均方根误差）的关系为：

| $M$ | 30 | 100 | 200 | 402 | 800 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RMSE** | 0.1424 | 0.0577 | 0.0500 | 0.0497 | 0.0496 |

$M$ 增大到约 200 之后误差基本不再下降，剩余误差来自体模阶跃边缘的离散化和吉布斯振荡，与 3.4 节保守估计的量级一致；$M=30$ 时误差明显增大，图中可以看到放射状的条纹伪影。窗函数的影响（$M=402$，噪声标准差为投影最大值的 $5\%$，均匀区域取第 5 个椭圆内部的一个小圆）：

| 投影 | 滤波器 | **RMSE** | 均匀区域标准差 |
| :--- | :--- | :--- | :--- |
| 无噪声 | **Ram-Lak** | 0.0497 | 0.0031 |
| 无噪声 | **Hann** | 0.0648 | 0.0006 |
| 有噪声 | **Ram-Lak** | 0.1203 | 0.1256 |
| 有噪声 | **Hann** | 0.0771 | 0.0480 |

无噪声时 **Ram-Lak** 滤波器保留了更多高频，边缘更锐利，误差更小；有噪声时 **Hann** 窗把均匀区域的噪声标准差降低到约 $38\%$，总误差也更小。在这组实验中，两者的总误差在噪声水平约为投影最大值的 $2\%$ 时持平。最后绘制体模、正弦图与各种重建结果：

```python
thetas30 = np.arange(30) * np.pi / 30
panels = [
    ('phantom', truth),
    ('sinogram', sino),
    ('unfiltered back-projection', backproject(sino, thetas, t, n)),
    ('FBP, M = 30', fbp(radon_ellipses(thetas30, t), thetas30, t, n)),
    ('FBP, Ram-Lak', recon['clean', 'ram-lak']),
    ('FBP, Hann', recon['clean', 'hann']),
    ('FBP, noisy, Ram-Lak', recon['noisy', 'ram-lak']),
    ('FBP, noisy, Hann', recon['noisy', 'hann']),
]
fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
for ax, (title, img) in zip(axes.ravel(), panels):
    if title == 'sinogram':
        ax.imshow(img, cmap='gray', aspect='auto', extent=[t[0], t[-1], 180, 0])
        ax.set_xlabel('t'); ax.set_ylabel('θ (degree)')
    elif title.startswith('unfiltered'):
        ax.imshow(img, cmap='gray', extent=[-1, 1, -1, 1])
    else:
        ax.imshow(img, cmap='gray', vmin=0, vmax=0.5, extent=[-1, 1, -1, 1])
    ax.set_title(title, fontsize=13)
plt.tight_layout()
plt.show()
```

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-projection-slice-001.png)

# 4. 应用

## 4.1 计算机断层扫描（CT）

投影切片定理最著名的应用是 **X** 射线**计算机断层扫描（Computed Tomography, CT）**。

1.  **数据采集**：设组织在 $(x,y)$ 处的线性衰减系数为 $\mu(x,y)$。单能 **X** 射线穿过物体时强度按 **Beer-Lambert** 定律衰减，$I = I_0\exp\left(-\int_L\mu\,ds\right)$，其中 $L$ 是射线路径。因此 $-\ln(I/I_0)$ 正是 $\mu$ 沿射线的线积分，即 $P_\theta(t)$。射线源与探测器绕物体旋转，记录全部角度的投影就得到正弦图。实际 **X** 射线是多能的，低能光子更容易被吸收，这一非线性会造成射束硬化（**beam hardening**）伪影，需要额外校正。
2.  **重建**：平行束几何下，按 3.3 节对每个投影做斜坡滤波（通常加窗）后反投影，即得到 $\mu(x,y)$ 的断层图像；也可以按 3.1 节做直接傅里叶重建。
3.  **扇束与锥束**：现代 **CT** 多采用扇形射线束，可以把扇束数据重排为平行束数据，或使用带几何加权的扇束 **FBP**；三维锥束 **CT** 常用近似的 **FDK**（**Feldkamp-Davis-Kress**）算法。
4.  **迭代重建**：把图像离散为像素向量 $\mathbf{x}$，第 $i$ 条射线的线积分近似为 $\mathbf{a}_i^\top\mathbf{x} = p_i$，重建转化为求解大型线性方程组。代数重建技术（**ART**）按 **Kaczmarz** 方法依次把当前解投影到每个超平面上：

$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \lambda\,\frac{p_i - \mathbf{a}_i^\top\mathbf{x}^{(k)}}{\Vert\mathbf{a}_i\Vert^2}\,\mathbf{a}_i
$$

其中 $\lambda$ 为松弛因子。迭代方法不依赖投影切片定理，可以方便地引入噪声统计模型和正则项，在投影角度稀疏或低剂量扫描时比 **FBP** 更有优势，代价是计算量更大。

## 4.2 磁共振成像（MRI）

在磁共振成像中，测得的信号本身就是傅里叶域的样本。忽略弛豫等效应，接收信号为

$$
s(t) = \int\rho(\mathbf{x})\,e^{-i2\pi\mathbf{k}(t)\cdot\mathbf{x}}\,d\mathbf{x},\qquad \mathbf{k}(t) = \frac{\gamma}{2\pi}\int_0^t\mathbf{G}(\tau)\,d\tau
$$

其中 $\rho$ 是自旋密度，$\mathbf{G}$ 是梯度磁场，$\gamma$ 是旋磁比。若梯度方向固定，$\mathbf{k}(t)$ 沿一条过原点的直线移动，一次读出就采到 **k** 空间的一条径向线；由投影切片定理，这条径向线的一维傅里叶逆变换就是 $\rho$ 在该方向上的投影。旋转梯度方向采集多条径向线，可以用 **FBP** 或 3.1 节的网格化方法重建。最早的磁共振成像（**Lauterbur**，1973）采用的就是这种投影重建方式；径向采样对运动不敏感，至今仍用于部分快速成像序列。

## 4.3 冷冻电镜（Cryo-EM）

在冷冻电镜中，电子束穿过冷冻的生物大分子样品。在弱相位近似下并暂不考虑衬度传递函数（**CTF**）的调制，每张颗粒图像近似为分子三维库仑势沿电子束方向的投影。由第 2 节的三维推广（$k=2$），这张图像的二维傅里叶变换就是分子三维傅里叶变换中过原点、垂直于电子束方向的一个平面切片，在电镜领域称为**中心截面定理（central section theorem）**。

由于每个颗粒在冰层中的取向未知，重建的关键是确定每张图像对应的切片方向。两个过原点、法向量分别为 $\mathbf{d}_1$ 和 $\mathbf{d}_2$ 的平面相交于一条方向为 $\mathbf{d}_1\times\mathbf{d}_2$ 的直线，因此任意两张投影图像的二维傅里叶变换中各有一条过原点的直线取值完全相同，即**公共线（common line）**。利用若干张图像之间的公共线可以确定它们的相对取向，再把各个切片放入三维傅里叶空间并做逆变换（或做加权反投影），得到分子的三维结构。

## 4.4 射电天文

射电天文中的**甚长基线干涉测量（Very Long Baseline Interferometry, VLBI）**组合全球多个射电望远镜的数据，合成等效口径接近地球直径的虚拟望远镜，从而获得极高的角分辨率。

设天空在视场中的亮度分布为 $I(l, m)$，$(l,m)$ 是天空平面上的方向余弦坐标。由两台望远镜（构成一条基线）同时观测射电源，测得的干涉条纹的**可见度（visibility）**记为 $V(u,v)$，$(u,v)$ 是基线在垂直于观测方向的平面上的投影，以波长为单位。根据**范西特-泽尼克定理（Van Cittert-Zernike theorem）**，在源为空间非相干、视场足够小（或阵列共面）以至于可以忽略 $w$ 项，并把天线方向图并入亮度分布的近似下，

$$
V(u, v) = \iint I(l, m)\, e^{-i2\pi(ul + vm)} \,dl\,dm
$$

即可见度是亮度分布二维傅里叶变换的采样，所在的平面称为 **uv** 平面。由于 $I$ 是实函数，$V(-u,-v)=\overline{V(u,v)}$，一条基线在一个时刻同时给出 $(u,v)$ 与 $(-u,-v)$ 两个样本。

**与投影切片定理的联系**：若一组相互平行、长度不同的基线（例如一个线性阵列）在 **uv** 平面上沿一条过原点、方向为 $\mathbf{n}$ 的直线采样，由投影切片定理，这些样本的一维傅里叶逆变换就是亮度分布在方向 $\mathbf{n}$ 上的投影，即沿垂直方向对天空做**条带积分（strip integration）**得到的一维亮度轮廓。[Strip Integration in Radio Astronomy](https://doi.org/10.1071/PH560198)（**Bracewell**，1956）正是在这一背景下研究了由不同方位角的条带积分恢复二维亮度分布的问题，这与 **CT** 中由投影重建图像是同一个数学问题。

为了获得更多样本，天文学家利用地球自转：随着地球转动，基线相对于射电源的投影不断变化，每条基线在 **uv** 平面上划出一段椭圆弧（东西向基线的椭圆以原点为中心）。多条不同方向和长度的基线（例如事件视界望远镜 **EHT**）的轨迹交错分布，这一过程称为**地球自转孔径合成（Earth-rotation aperture synthesis）**。经过数小时观测得到的 **uv** 覆盖仍然稀疏且不规则，图像重建通常包括：

1.  **网格化（gridding）**：通过卷积插值把不规则分布的可见度样本映射到规则的笛卡尔网格上。
2.  **快速傅里叶逆变换**：设采样函数为 $S(u,v)$（采样处为 1，其余为 0），对网格化后的数据做逆变换得到**脏图（dirty image）**$I_D = \mathcal{F}^{-1}\{S\cdot V\} = I * B$，其中 $B=\mathcal{F}^{-1}\{S\}$ 称为**脏束（dirty beam）**，即该 **uv** 覆盖下的点扩散函数。
3.  **反卷积（deconvolution）**：使用 **CLEAN** 或最大熵方法（**Maximum Entropy Method, MEM**）等算法，从脏图中迭代地去除脏束旁瓣造成的伪影，恢复高保真度的射电图像。
