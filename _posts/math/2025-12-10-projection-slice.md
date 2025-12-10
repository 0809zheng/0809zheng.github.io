---
layout: post
title: '投影切片定理(Projection-Slice Theorem)'
date: 2025-12-10
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/693959cf6166b8110136ecfc.png'
tags: 数学
---

> Projection-Slice Theorem.

## 1. 引言

在处理多维数据时，我们常常面临从低维观测中重构高维信号的挑战。例如，在医学成像中，**X**射线设备只能获取穿透物体的线积分，即三维密度函数沿射线路径的投影。如何从这些二维投影数据中恢复出完整的三维内部结构？傅里叶切片定理为解决此类逆问题（**inverse problem**）提供了坚实的理论基础。

**投影切片定理（Projection-Slice Theorem）**，又称**傅里叶切片定理（Fourier Slice Theorem）**，是信号处理、医学成像和计算机视觉等领域的一项基础性数学原理。该定理精确地描述了一个多维函数沿某一方向的投影（积分）与其傅里叶变换在一个特定切片之间的对偶关系。

该定理的核心思想是：**一个函数在一维投影上的傅里叶变换，等价于该函数在二维傅里叶变换域中穿过原点的一个切片。** 这种将空间域的投影操作与频率域的切片操作联系起来的特性，使得我们能够通过在频率域中填充数据来重建原始信号。

## 2. 投影切片定理

为了清晰起见，我们以二维函数为例进行阐述。该定理可以自然地推广到任意维度。

令 $f(x, y)$ 为一个定义在 $\mathbb{R}^2$ 上的二维函数。

### 2.1 投影算子 (Projection Operator)

首先，我们定义沿某个角度 $\theta$ 的投影操作。一个过原点、与 x 轴夹角为 $\theta$ 的直线可由方程 $x\cos\theta + y\sin\theta = t$ 定义，其中 $t$ 是直线到原点的有向距离。

函数 $f(x, y)$ 沿与该直线族正交方向的投影，被定义为一个关于变量 $t$ 的一维函数 $P_\theta(t)$，其形式为线积分：
$$
P_\theta(t) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \delta(x\cos\theta + y\sin\theta - t) \,dx\,dy
$$
其中 $\delta(\cdot)$ 是狄拉克δ函数。这个积分计算了所有落在直线 $x\cos\theta + y\sin\theta = t$ 上的 $f(x, y)$ 的值的总和。

### 2.2 傅里叶变换

函数 $f(x,y)$ 的二维傅里叶变换为 $F(u,v)$：

$$
F(u, v) = \mathcal{F}_2\{f(x, y)\} = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) e^{-i2\pi(ux + vy)} \,dx\,dy
$$

其中 $(u, v)$ 是频率域坐标。

投影函数 $P_\theta(t)$ 的一维傅里叶变换为 $S_\theta(\omega)$：

$$
S_\theta(\omega) = \mathcal{F}_1\{P_\theta(t)\} = \int_{-\infty}^{\infty} P_\theta(t) e^{-i2\pi\omega t} \,dt
$$

其中 $\omega$ 是径向频率。

### 2.3 定理陈述

投影切片定理指出，投影函数 $P_\theta(t)$ 的一维傅里叶变换 $S_\theta(\omega)$，等于原始函数 $f(x, y)$ 的二维傅里叶变换 $F(u, v)$ 沿同样角度 $\theta$ 穿过原点的切片。

在频率域 $(u, v)$ 中，这个切片是一条直线，其参数方程可以表示为 $u = \omega\cos\theta$ 和 $v = \omega\sin\theta$。因此，定理的数学表达式为：
$$
S_\theta(\omega) = F(\omega\cos\theta, \omega\sin\theta)
$$

![](https://pic1.imgdb.cn/item/693959cf6166b8110136ecfc.png)

## 3. 定理证明

我们可以通过直接推导来证明该定理。

从投影 $P_\theta(t)$ 的一维傅里叶变换 $S_\theta(\omega)$ 的定义开始：

$$
S_\theta(\omega) = \int_{-\infty}^{\infty} P_\theta(t) e^{-i2\pi\omega t} \,dt
$$

将 $P_\theta(t)$ 的积分表达式代入：

$$
S_\theta(\omega) = \int_{-\infty}^{\infty} \left[ \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \delta(x\cos\theta + y\sin\theta - t) \,dx\,dy \right] e^{-i2\pi\omega t} \,dt
$$

交换积分顺序（根据富比尼定理）：

$$
S_\theta(\omega) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \left[ \int_{-\infty}^{\infty} \delta(x\cos\theta + y\sin\theta - t) e^{-i2\pi\omega t} \,dt \right] \,dx\,dy
$$

利用狄拉克$δ$函数的筛选性质（**sifting property**），$\int g(t)\delta(t-a)dt = g(a)$，对关于 $t$ 的积分进行求值。这里 $a = x\cos\theta + y\sin\theta$：

$$
\int_{-\infty}^{\infty} \delta(x\cos\theta + y\sin\theta - t) e^{-i2\pi\omega t} \,dt = e^{-i2\pi\omega(x\cos\theta + y\sin\theta)}
$$

将上述结果代回原式：

$$
S_\theta(\omega) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) e^{-i2\pi(x(\omega\cos\theta) + y(\omega\sin\theta))} \,dx\,dy
$$

观察上式，它与 $f(x, y)$ 的二维傅里叶变换 $F(u, v)$ 的定义式完全一致，只需令频率变量 $u = \omega\cos\theta$ 和 $v = \omega\sin\theta$。

因此，我们得到最终结论：

$$
S_\theta(\omega) = F(\omega\cos\theta, \omega\sin\theta)
$$


## 4. 实际应用

### ⚪ 计算机断层扫描 (Computed Tomography, CT)

投影切片定理最著名和最成功的应用是在透射式成像的**计算机断层扫描（CT）**中，特别是基于**滤波反投影（Filtered Back-Projection, FBP）**算法的图像重建。

**CT** 成像的基本流程如下：
1.  **数据采集 (Projection):** **X**射线源和探测器围绕待扫描物体（如人体）旋转。在每个角度 $\theta$，探测器记录穿过物体的**X**射线强度衰减，这等效于获取了物体内部密度函数 $f(x, y)$ 在该角度下的线积分投影 $P_\theta(t)$。这个数据集 $\{P_\theta(t)\}$ 被称为**正弦图（Sinogram）**。
2.  **傅里叶变换 (Slice Calculation):** 根据傅里叶切片定理，对每个角度 $\theta$ 采集到的投影数据 $P_\theta(t)$ 进行一维傅里叶变换，得到 $S_\theta(\omega)$。这等价于获得了物体二维傅里叶变换 $F(u, v)$ 在角度 $\theta$ 上的一个径向切片。
3.  **频率域填充 (Filling k-space):** 通过围绕物体旋转 $180^\circ$ 或 $360^\circ$，可以获得大量不同角度的投影。将这些投影的一维傅里叶变换结果放置在频率域 $(u, v)$ 的相应径向线上，就可以填充（或近似填充）整个二维傅里叶谱。然而，这些径向样本在频率域的分布是不均匀的：在靠近原点的低频区域样本密集，而在远离原点的高频区域样本稀疏。
4.  **滤波 (Filtering):** 直接对非均匀采样的傅里叶数据进行逆变换会导致严重的图像模糊和伪影。为了补偿这种采样密度不均，需要在频率域对每个切片数据 $S_\theta(\omega)$ 应用一个**斜坡滤波器（Ramp Filter）**，其频率响应为 $\|\omega\|$。这个滤波操作强调了高频信息，以抵消因径向采样造成的低频信息过采样。滤波后的切片为 $S'_\theta(\omega) = S_\theta(\omega) \cdot \|\omega\|$。
5.  **反投影 (Back-Projection):** 对滤波后的频域数据执行二维逆傅里叶变换，即可重建出原始图像 $f(x, y)$。在实践中，这一步通常通过一种称为反投影的空间域等效操作来完成，即将每个滤波后的投影涂抹回图像空间，从而得到最终的清晰图像。


### ⚪ 射电天文学

射电天文学中的**甚长基线干涉测量 (Very Long Baseline Interferometry, VLBI)** 技术通过组合来自全球各地多个射电望远镜的数据，合成一个等效口径（**aperture**）接近地球直径的虚拟望远镜，从而实现极高的角分辨率，足以分辨遥远星系核心的精细结构。

与**CT**扫描中物体静止、探测器旋转不同，射电天文学观测的是天空中的一个二维亮度分布函数 $I(l, m)$，其中 $(l, m)$ 是天空平面上的方向余弦坐标。**VLBI**的核心原理是，由两个相距甚远的望远镜（构成一条基线）同时观测一个射电源，所测量到的干涉条纹的**可见度 (visibility)** $V(u, v)$，直接对应于天空亮度分布 $I(l, m)$ 的二维傅里叶变换的一个采样点。

根据**范西特-泽尼克定理 (Van Cittert–Zernike theorem)**，这种关系可以表示为：

$$
V(u, v) = \iint I(l, m) e^{-i2\pi(ul + vm)} \,dl\,dm = \mathcal{F}\{I(l, m)\}
$$

这里的 $(u, v)$ 是在垂直于观测方向的平面上，以波长为单位的基线投影坐标。这个二维频率域被称为 **uv-plane**。

单个基线在某一瞬间只能测量到**uv-plane**上的一个点 $(u, v)$ 和其共轭对称点 $(-u, -v)$。为了获得更多的傅里叶域样本以重建图像，天文学家利用了地球的**自转**。随着地球的转动，望远镜相对于天空射电源的几何构型会发生变化，导致基线在**uv-plane**上的投影轨迹随时间演化。对于一条固定的东西向基线，其轨迹在**uv-plane**上是一个椭圆；对于多条不同方向和长度的基线构成的阵列（如事件视界望远镜**EHT**），它们的轨迹会在**uv-plane**上画出多条交错的椭圆弧。

这个过程被称为**地球自转孔径合成 (Earth-rotation aperture synthesis)**。经过数小时的观测，这些轨迹就能在**uv-plane**上采集到相当数量的样本点，虽然这些样本的分布仍然是不规则和稀疏的。在**uv-plane**上获得稀疏的可见度数据 $\{V(u_k, v_k)\}$ 后，重建天空图像 $I(l, m)$ 就成了一个从不完整傅里叶样本中恢复原始信号的逆问题。
1.  **网格化 (Gridding):** 首先，将这些不规则分布的**uv**样本通过卷积插值等方法，映射到一个规则的笛卡尔网格上。
2.  **快速傅里叶逆变换 (Inverse FFT):** 对网格化后的傅里-叶数据执行快速傅里叶逆变换，得到一幅包含由稀疏采样引起的显著伪影的脏图 (**dirty image**)。
3.  **反卷积 (Deconvolution):** 最后，使用如**CLEAN**或**MEM (Maximum Entropy Method)**等反卷积算法，迭代地移除由不完整**uv**覆盖（即点扩散函数，**PSF**）造成的伪影，从而恢复出一幅高保真度的射电图像。