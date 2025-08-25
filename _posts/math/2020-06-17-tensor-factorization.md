---
layout: post
title: '张量分解(Tensor Decomposition)'
date: 2020-06-17
author: 郑之杰
cover: 'https://github.com/0809zheng/imagebed_math_0/raw/main/5ee9a89da240b370e3bcba63.jpg'
tags: 数学
---

> Tensor Decomposition(Factorization).

**张量分解（tensor decomposition/factorization）**的思想是找到一个**低秩**结构近似原始数据张量。常用的方法包括：
1. **Tucker**分解
2. **CP**分解
3. 应用：缺失数据修复

# 1. Tucker分解
**Tucker分解（Tucker decomposition）**是一种高阶的主成分分析形式。它将一个张量分解成一个较小的核张量与一系列矩阵（二维张量）的乘积。

![](https://github.com/0809zheng/imagebed_math_0/raw/main/5ee8b6f82cb53f50fedacb48.jpg)

通常，给定一个三阶张量$x \in \Bbb{R}^{M×N×T}$，使用低秩（$R_1,R_2,R_3$）的**Tucker**分解可以表示为：

$$ \mathcal{X}\approx\mathcal{G}\times_1 A\times_2 B\times_3 C $$

其中$g \in \Bbb{R}^{R_1×R_2×R_3}$是**核张量（core tensor）**，$A \in \Bbb{R}^{M×R_1}$、$B \in \Bbb{R}^{N×R_2}$和$C \in \Bbb{R}^{T×R_3}$是**因子矩阵（factor matrix）**。

对于张量$X$的任意第($m,n,t$)个元素，**Tucker**分解也可以写作：

$$ x_{mnt}\approx\sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3}g_{r_1r_2r_3}a_{mr_1}b_{nr_2}c_{tr_3} $$

使用**Python**的**Numpy**库可以便捷实现上述运算过程：

```python
import numpy as np
def tucker_combine(core_tensor, mat1, mat2, mat3):
    return np.einsum('ijk, mi, nj, tk -> mnt', core_tensor, mat1, mat2, mat3)
```

**Tucker**分解可以使用**交替最小二乘法（Alternating Least Square， ALS）**求解。

在**Tucker**分解中，采用平方误差作为损失函数，最小化该损失函数求解最优的核张量和因子矩阵：

$$ \min_{\mathcal{G},A,B,C}\sum_{(m,n,t)\in\Omega}\left(x_{mnt}-\sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3}g_{r_1r_2r_3}a_{mr_1}b_{nr_2}c_{tr_3}\right)^2 $$

该优化问题的困难在于需要同时优化核张量和三个因子矩阵，一种解决方法是使用交替最小二乘法。

交替最小二乘法把优化问题划分成几个独立的子问题，交替地优化每个子问题的损失函数。对于**Tucker**分解，每次优化**g、A、B、C**其中的一个参数，保持其他三个参数不变。求解单一参数的子问题是一个**凸优化**问题。

下面以求解核张量**g**为例，说明交替最小二乘法的优化过程。

$$ \min_{\mathcal{G}}\sum_{(m,n,t)\in\Omega}\left(x_{mnt}-\sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3}g_{r_1r_2r_3}a_{mr_1}b_{nr_2}c_{tr_3}\right)^2 $$

$$ \Rightarrow\min_{\mathcal{G}}\sum_{(m,n,t)\in\Omega}\left(x_{mnt}-\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\text{vec}\left(\mathcal{G}\right)\right)^2 $$

$$ \Rightarrow\min_{\mathcal{G}}\sum_{(m,n,t)\in\Omega}\left(x_{mnt}-\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\text{vec}\left(\mathcal{G}\right)\right)^\top\left(x_{mnt}-\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\text{vec}\left(\mathcal{G}\right)\right) $$

上述问题的解可以表达为：

$$ \text{vec}\left (\mathcal{G}\right)\Leftarrow \\ \left(\sum_{(m,n,t)\in\Omega}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\right)^{-1}\left(\sum_{(m,n,t)\in\Omega}x_{mnt}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\right) $$

同理，求解因子矩阵$A \in \Bbb{R}^{M×R_1}$的优化问题可以写作：

$$ \min_{A}\sum_{(m,n,t)\in\Omega}\left(x_{mnt}-\sum_{r_1=1}^{R_1}\sum_{r_2=1}^{R_2}\sum_{r_3=1}^{R_3}g_{r_1r_2r_3}a_{mr_1}b_{nr_2}c_{tr_3}\right)^2 $$

上述问题可以表示为：

$$ \min_{\boldsymbol{a}_m}\sum_{n,t:(m,n,t)\in\Omega}\left(x_{mnt}-\boldsymbol{a}_m^\top\mathcal{G}_{(1)}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\right)\left(x_{mnt}-\boldsymbol{a}_m^\top\mathcal{G}_{(1)}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\right)^\top $$

其最小二乘解为：

$$ \boldsymbol{a}_{m}\Leftarrow\left(\sum_{n,t:(m,n,t)\in\Omega}\mathcal{G}_{(1)}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)^\top\mathcal{G}_{(1)}^\top\right)^{-1}\left(\sum_{n,t:(m,n,t)\in\Omega}y_{mnt}\mathcal{G}_{(1)}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\right)\\,\forall m\in\left\{1,2,...,M\right\} $$

同理，因子矩阵$B \in \Bbb{R}^{N×R_2}$和$C \in \Bbb{R}^{T×R_3}$的最小二乘解为：

$$ \boldsymbol{b}_{n}\Leftarrow\left(\sum_{m,t:(m,n,t)\in\Omega}\mathcal{G}_{(2)}\left(\boldsymbol{c}_{t}\odot\boldsymbol{a}_{m}\right)\left(\boldsymbol{c}_{t}\odot\boldsymbol{a}_{m}\right)^\top\mathcal{G}_{(2)}^\top\right)^{-1}\left(\sum_{m,t:(m,n,t)\in\Omega}y_{mnt}\mathcal{G}_{(2)}\left(\boldsymbol{c}_{t}\odot\boldsymbol{a}_{m}\right)\right)\\,\forall n\in\left\{1,2,...,N\right\} $$

$$ \boldsymbol{c}_{t}\Leftarrow\left(\sum_{m,n:(m,n,t)\in\Omega}\mathcal{G}_{(3)}\left(\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\left(\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\mathcal{G}_{(3)}^\top\right)^{-1}\left(\sum_{m,n:(m,n,t)\in\Omega}y_{mnt}\mathcal{G}_{(3)}\left(\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\right)\\,\forall t\in\left\{1,2,...,T\right\} $$

**Tucker**分解的核张量一般具有很大的值，容易过拟合；通常在求解核张量**g**时引入**l1**或**l2正则化**增强分解对于数据的鲁棒性：

$$ \min_{\mathcal{G}}\sum_{(m,n,t)\in\Omega}\left(x_{mnt}-\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\text{vec}\left(\mathcal{G}\right)\right)^\top\left(x_{mnt}-\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\text{vec}\left(\mathcal{G}\right)\right)\\+\lambda_g\text{vec}\left(\mathcal{G}\right)^\top\text{vec}\left(\mathcal{G}\right) $$

上述问题的解可以表达为：

$$ \text{vec}\left (\mathcal{G}\right)\Leftarrow \\ \left(\sum_{(m,n,t)\in\Omega}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top+\lambda_gI\right)^{-1}\left(\sum_{(m,n,t)\in\Omega}x_{mnt}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\right) $$


# 2. CP分解
**CP分解（CANDECOMP/PARAFAC decomposition）**也是一种常见的张量分解方法。它将一个张量分解成一系列因子向量外积（**outer product**）的和。

![](https://github.com/0809zheng/imagebed_math_0/raw/main/5ee8b8132cb53f50fedbefc3.jpg)

通常，给定一个三阶张量$x \in \Bbb{R}^{M×N×T}$，CP分解可以表示为：

$$ \mathcal{x}\approx\sum_{r=1}^{R}a_{r}\circ b_{r}\circ c_{r} $$

其中**因子向量（factor vector）**$a_r \in \Bbb{R}^{M}$、$b_r \in \Bbb{R}^{N}$、$c_r \in \Bbb{R}^{T}$是**因子矩阵（factor matrix）**$A \in \Bbb{R}^{M×R}$、$B \in \Bbb{R}^{N×R}$和$C \in \Bbb{R}^{T×R}$的第$r$列。实际上这些向量的外积是秩为$1$的张量，**CP**分解就是把原张量分解成$R$个秩为$1$的张量之和。

对于张量$x$的任意第($m,n,t$)个元素，CP分解也可以写作：

$$ x_{mnt}\approx\sum_{r=1}^{R}a_{mr}b_{nr}c_{tr} $$

实际上，**CP**分解是**Tucker**分解的一种特殊形式，可以表示为：

$$ x_{mnt}\approx\sum_{r=1}^{R}a_{mr}b_{nr}c_{tr}=\sum_{r_1=1}^{R}\sum_{r_2=1}^{R}\sum_{r_3=1}^{R}g_{r_1r_2r_3}a_{mr_1}b_{nr_2}c_{tr_3} $$

其中核张量**g**是对角张量，即:

$$ g_{r_1r_2r_3} = \begin{cases} 1, & r_1=r_2=r_3 \\ 0, & \text{others} \end{cases} $$

使用**Python**的**Numpy**库可以便捷实现上述运算过程：

```python
import numpy as np
def cp_combine(mat1, mat2, mat3):
    return np.einsum('mr, nr, tr -> mnt', mat1, mat2, mat3)
```

在**CP**分解中，采用平方误差作为损失函数，最小化该损失函数求解最优的因子矩阵：

$$ \min _{A, B, C} \sum_{(m, n, t) \in \Omega}\left(x_{m n t}-\sum_{r=1}^{R}a_{mr}b_{nr}c_{tr}\right)^{2} $$

使用**交替最小二乘法**把该优化问题划分成几个独立的子问题，交替地优化每个子问题的损失函数。对于**CP**分解，每次优化**A、B、C**其中的一个参数，保持其他三个参数不变。求解单一参数的子问题是一个**凸优化**问题。

具体地，求解因子矩阵$A \in \Bbb{R}^{M×R}$的第$m$行$a_m \in \Bbb{R}^{R}$可以表示为：

$$ \min_{\boldsymbol{a}_m}\sum_{n,t:(m,n,t)\in\Omega}\left(x_{mnt}-\boldsymbol{a}_m^\top\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\right)\left(x_{mnt}-\boldsymbol{a}_m^\top\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\right)^\top $$

其最小二乘解为：

$$ \boldsymbol{a}_{m}\Leftarrow\left(\sum_{n,t:(m,n,t)\in\Omega}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)^\top\right)^{-1}\left(\sum_{n,t:(m,n,t)\in\Omega}y_{mnt}\left(\boldsymbol{c}_{t}\odot\boldsymbol{b}_{n}\right)\right)\\,\forall m\in\left\{1,2,...,M\right\} $$

同理，因子矩阵$B \in \Bbb{R}^{N×R}$的第$n$行$b_n \in \Bbb{R}^{R}$和因子矩阵$C \in \Bbb{R}^{T×R}$的第$t$行$c_t \in \Bbb{R}^{R}$的最小二乘解为：

$$ \boldsymbol{b}_{n}\Leftarrow\left(\sum_{m,t:(m,n,t)\in\Omega}\left(\boldsymbol{c}_{t}\odot\boldsymbol{a}_{m}\right)\left(\boldsymbol{c}_{t}\odot\boldsymbol{a}_{m}\right)^\top\right)^{-1}\left(\sum_{m,t:(m,n,t)\in\Omega}y_{mnt}\left(\boldsymbol{c}_{t}\odot\boldsymbol{a}_{m}\right)\right)\\,\forall n\in\left\{1,2,...,N\right\} $$

$$ \boldsymbol{c}_{t}\Leftarrow\left(\sum_{m,n:(m,n,t)\in\Omega}\left(\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\left(\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)^\top\right)^{-1}\left(\sum_{m,n:(m,n,t)\in\Omega}y_{mnt}\left(\boldsymbol{b}_{n}\odot\boldsymbol{a}_{m}\right)\right)\\,\forall t\in\left\{1,2,...,T\right\} $$

# 3. 应用：缺失数据修复
张量分解可以用于修复缺失数据。

- 实例：[时空交通数据缺失值修复](https://zhuanlan.zhihu.com/p/50429765)

如下图所示，对于一个含有缺失数据的数据张量$Y$，对其进行张量分解，分解成一系列低秩张量的乘积。将这些低秩张量的乘积作为原数据张量的重构，修复其缺失值：

![](https://github.com/0809zheng/imagebed_math_0/raw/main/5ee9a020a240b370e3aff888.jpg)
