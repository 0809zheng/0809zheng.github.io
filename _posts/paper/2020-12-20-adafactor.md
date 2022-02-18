---
layout: post
title: 'Adafactor: Adaptive Learning Rates with Sublinear Memory Cost'
date: 2020-12-20
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/620ef63c2ab3f51d913d2a05.jpg'
tags: 论文阅读
---

> Adafactor：减少Adam的显存占用.

- paper：[Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

# 1. Adam的显存问题

**Adam**优化器的更新过程如下：

$$ \begin{align} m_t &= β_1m_{t-1} + (1-β_1)g_t \\ v_t &= β_2v_{t-1} + (1-β_2)g_t^2 \\\hat{m}_t &= \frac{m_t}{1-β_1^t} \\ \hat{v}_t &= \frac{v_t}{1-β_2^t} \\ θ_t&=θ_{t-1}-\gamma \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+ε} \end{align} $$

在**Adam**优化器中，计算梯度$g_t$是最占用显存和计算量的操作。除此之外，**Adam**维护两组缓存变量$m_t$和$v_t$，用于滑动计算梯度的低阶矩估计。这两个缓存变量和参数本身一样大，因此对于参数比较多的模型，缓存变量也消耗比较多的显存。

# 2. 改进Adam

## (1) 移除动量

在计算机视觉模型中，动量方法通常能获得更好的精度，自适应学习率方法的效果相对较差。但对于自然语言处理中的模型，自适应学习率通常更重要。

由于本文主要针对具有较多参数的大规模预训练模型，因此可以移除**Adam**中维护的动量参数，从而减少一半的缓存参数：

$$ \begin{align}  v_t &= β_2v_{t-1} + (1-β_2)g_t^2 \\ \hat{v}_t &= \frac{v_t}{1-β_2^t} \\ θ_t&=θ_{t-1}-\gamma \frac{g_t}{\sqrt{\hat{v}_t}+ε} \end{align} $$

上式相当于**RmSProp**算法的变种，增加了偏差修正步骤。

## (2) 低秩分解
维护变量$v_t$也需要比较多的显存。若将$v_t$看作一个$m \times n$的矩阵$C$，则可以对该矩阵进行低秩分解，寻找$m \times k$的矩阵$A$和$k \times n$的矩阵$B$，且满足$AB=C$，则能够有效降低参数量。

最理想的情况下，令$k=1$，则相当于寻找两组向量$$\{a_i\}_{i=1}^{m}$$和$$\{b_j\}_{j=1}^{n}$$，使得：

$$ a_ib_j ≈ c_{i,j} $$

求解上述方程可以使用**广义KL散度**（也称**I散度**）。广义**KL**散度衡量两组变量的相似程度，而不需要对两组变量归一化为概率分布，仅需要变量取值非负即可。

广义**KL**散度的出发点为不等式$x \log x \geq x-1$ ($x>0$)，令$x=\frac{p}{q}$，且两端乘以$q$，则有：

$$ p \log \frac{p}{q} -p +q \geq 0 $$

在该问题中，令$p=c_{i,j}$，$q=a_ib_j$，则有：

$$ l = \sum_{i,j}^{}( c_{i,j} \log \frac{c_{i,j}}{a_ib_j} -c_{i,j} + a_ib_j )$$

对上式求偏导数：

$$ \frac{\partial l}{\partial a_i} = \sum_{j}^{}( -\frac{c_{i,j}}{a_i}+b_j) \\ \frac{\partial l}{\partial b_j} = \sum_{i}^{} (-\frac{c_{i,j}}{b_j}+a_i) $$

令偏导数为$0$，则有：

$$ a_i \sum_{j}^{}b_j = \sum_{j}^{}c_{i,j} \\ b_j  \sum_{j}^{}a_i = \sum_{i}^{}c_{i,j} $$

注意到若$(a_i,b_j)$是一组解，则$(\lambda a_i,b_j / \lambda)$也是一组解。因此不妨令$\sum_{j}^{}b_j =1$，则得到一组解：

$$ a_i = \sum_{j}^{}c_{i,j} \\ b_j   = \frac{\sum_{i}^{}c_{i,j}}{\sum_{i,j}^{}c_{i,j}} $$

上式表明对矩阵$C$的低秩分解可以分别按行求和与按列求和，相乘后再除以全体的和。这种分解方式类似于求联合分布的两个边缘分布。

对于本文的优化算法，则相当于对$v_t$做低秩分解，用两组缓存变量$v_{i;t}^{(r)}$和$v_{j;t}^{(c)}$代替：

$$ \begin{align}  v_{i;t}^{(r)}  &= β_2v_{i;t-1}^{(r)} + (1-β_2)\sum_{j}^{} (g_{i,j;t}^2+\epsilon) \\ v_{j;t}^{(c)}  &= β_2v_{j;t-1}^{(c)} + (1-β_2)\sum_{i}^{} (g_{i,j;t}^2+\epsilon) \\ v_{i,j;t} &= \frac{v_{i;t}^{(r)} v_{j;t}^{(c)} }{\sum_{j}^{} v_{j;t}^{(c)} } \\ \hat{v}_t &= \frac{v_t}{1-β_2^t} \\ θ_t&=θ_{t-1}-\gamma \frac{g_t}{\sqrt{\hat{v}_t}} \end{align} $$

## (3) 滑动权重

滑动平均权重$\beta_2$通常设置为常数，在这种情况下$\hat{v}_t$的更新过程为：

$$ \hat{v}_t = \frac{v_t}{1-β_2^t} = \frac{\beta_2 v_{t-1}+(1-\beta_2)g_t^2}{1-β_2^t} \\ = \frac{\beta_2 \hat{v}_{t-1}(1-β_2^{t-1})+(1-\beta_2)g_t^2}{1-β_2^t} \\ = \beta_2\frac{1-β_2^{t-1}}{1-β_2^t}\hat{v}_{t-1} + \frac{1-\beta_2}{1-β_2^t}g_t^2 \\ = \beta_2\frac{1-β_2^{t-1}}{1-β_2^t}\hat{v}_{t-1} +(1-\beta_2\frac{1-β_2^{t-1}}{1-β_2^t})g_t^2 $$

若记$\hat{\beta}_{2,t}=\beta_2\frac{1-β_2^{t-1}}{1-β_2^t}$，则$\hat{v}_t$的更新过程为：

$$ \hat{v}_t = \hat{\beta}_{2,t}\hat{v}_{t-1} +(1-\hat{\beta}_{2,t})g_t^2 $$

当$t=1$时，$\hat{\beta}_{2,t}=0$，此时$\hat{v}_t=g_t^2$，使用实时梯度校正学习率；当$t \to ∞$时，$\hat{\beta}_{2,t}=\beta_2$，训练后期梯度变小，仍然校正学习率可能会导致梯度方向的改变，从而导致训练不稳定。因此希望训练后期算法退化为**SGD**，即$\hat{\beta}_{2,t}\to 1$。不妨设置为：

$$ \hat{\beta}_{2,t} = 1-\frac{1}{t^c} $$

当$c=1$时，$\hat{v}_t$的更新过程为：

$$ \hat{v}_t = (\frac{t-1}{t})\hat{v}_{t-1} +\frac{1}{t}g_t^2 \\ = (\frac{t-1}{t})((\frac{t-2}{t-1})\hat{v}_{t-2} +\frac{1}{t-1}g_{t-1}^2) +\frac{1}{t}g_t^2 \\ = \frac{1}{t} \sum_{i=1}^{t}g_i^2 $$

上式表示为所有梯度平方的平均。通常希望越久远的梯度重要性越低，因此取$c<1$，即历史权重$1-\frac{1}{t^c}<1-\frac{1}{t}$。实验中取$c=0.8$。

此时更新过程为：

$$ \begin{align} \hat{\beta}_{2,t} &= 1-\frac{1}{t^c} \\ v_{i;t}^{(r)}  &= \hat{\beta}_{2,t}v_{i;t-1}^{(r)} + (1-\hat{\beta}_{2,t})\sum_{j}^{} (g_{i,j;t}^2+\epsilon) \\ v_{j;t}^{(c)}  &= \hat{\beta}_{2,t}v_{j;t-1}^{(c)} + (1-\hat{\beta}_{2,t})\sum_{i}^{} (g_{i,j;t}^2+\epsilon) \\ \hat{v}_{i,j;t} &= \frac{v_{i;t}^{(r)} v_{j;t}^{(c)} }{\sum_{j}^{} v_{j;t}^{(c)} } \\  θ_t&=θ_{t-1}-\gamma \frac{g_t}{\sqrt{\hat{v}_t}} \end{align} $$

## (4) 层自适应

当模型参数量较大、使用较大批量进行训练时，可以采用如**LARS**一样的层自适应方法。即将参数的更新量进行标准化，然后乘以参数的模长。此时参数更新的方向与梯度大小无关，只由参数本身的大小以及全局学习率共同决定，此时模型所有层的参数更新程度相对一致。

层自适应的更新过程可以表示为：

$$ u_t= \frac{g_t}{\sqrt{\hat{v}_t}} \\ \hat{u}_t = u_t \times\frac{\max(\epsilon_2,\sqrt{\frac{1}{n}\sum_{i=1}^{n}\theta_i^2})}{\max(1,\sqrt{\sum_{i=1}^{n}u_i^2} /d)} \\ θ_t=θ_{t-1}-\gamma \hat{u}_t $$

其中$\max(1,\sqrt{\sum_{i=1}^{n}u_i^2} /d)$表示当更新量$u_t$的模长超过$d$时才进行归一化。

## (5) Adafactor

综上所述，对**Adam**改进后作者提出了**Adafactor**：


$$ \begin{align} \hat{\beta}_{2,t} &= 1-\frac{1}{t^c} \\ v_{i;t}^{(r)}  &= \hat{\beta}_{2,t}v_{i;t-1}^{(r)} + (1-\hat{\beta}_{2,t})\sum_{j}^{} (g_{i,j;t}^2+\epsilon_1) \\ v_{j;t}^{(c)}  &= \hat{\beta}_{2,t}v_{j;t-1}^{(c)} + (1-\hat{\beta}_{2,t})\sum_{i}^{} (g_{i,j;t}^2+\epsilon_1) \\ \hat{v}_{i,j;t} &= \frac{v_{i;t}^{(r)} v_{j;t}^{(c)} }{\sum_{j}^{} v_{j;t}^{(c)} } \\ u_t &= \frac{g_t}{\sqrt{\hat{v}_t}} \\ \hat{u}_t &= u_t \frac{\max(\epsilon_2,|\theta_{t-1}|)}{\max(1,|u_t| /d)} \\  θ_t&=θ_{t-1}-\gamma \frac{g_t}{\sqrt{\hat{v}_t}} \end{align} $$

