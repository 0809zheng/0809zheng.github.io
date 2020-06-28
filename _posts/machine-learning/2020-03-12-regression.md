---
layout: post
title: '回归'
date: 2020-03-12
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ed758f4c2a9a83be54c0e00.jpg'
tags: 机器学习
---

> Regression.

本文目录：
1. 线性回归
2. 正规方程法
3. 误差分析
4. 非线性回归
5. 过拟合
6. 验证集
7. Tube Regression
8. Ridge Regression
9. Kernel Ridge Regression
10. Lasso Regression

# 1. 线性回归
**线性回归（Linear Regression）**是最基本的回归方法，其假设空间的假设函数是线性函数。

若记每一个样本点$x = (1,x_1,x_2,...,x_d)^T \in \Bbb{R}^{d+1}$，模型权重参数$w = (w_0,w_1,w_2,...,w_d)^T \in \Bbb{R}^{d+1}$，则线性回归模型：

$$ \hat{y} = \sum_{j=0}^{d} {w_jx_j} = w^Tx $$

若将样本点的每一个维度看作不同的特征，则线性回归是对这些特征进行线性组合，对每一个特征乘以一个权重。

若样本集$$X=\{x^{(1)},...,x^{(N)}\}$$，标签集$$y=\{y^{(1)},...,y^{(N)}\}$$，线性回归使用**均方误差（mean squared error，MSE）**衡量模型的好坏：

$$ L(w) = \frac{1}{N} \sum_{i=1}^{N} {(w^Tx^{(i)}-y^{(i)})^2} $$

### 最小二乘法
该方法也称为**最小二乘法（least square method）**，是指给出一个超平面，到所有样本点距离的平方最小：

![](https://pic.downk.cc/item/5ed0bbb7c2a9a83be5882cf8.jpg)

### 均方误差损失可否用于分类？
由下图可以看出，对于分类任务，$0/1$损失始终要比均方误差损失小，故误差上界小：

![](https://pic.downk.cc/item/5ed0baaec2a9a83be5867638.jpg)

虽然上界变得宽松了，但是由于优化问题变得简单（从**NP-hard**变为凸优化），也能得到不错的结果。

### 最小二乘法等价于噪声服从高斯分布的极大似然估计
引入高斯噪声$ε$~$N(0,σ^2)$，对线性回归建模：

$$ y = w^Tx + ε $$

其中(x,y)是样本数据，w是未知的常数，每个样本点受到了高斯噪声的干扰。则条件变量$y \mid x;w$服从于$N(w^Tx,σ^2)$。列出条件概率：

$$ P(y \mid x;w) = \frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y-w^Tx)^2}{2σ^2}) $$

采用极大似然估计的方法估计参数$w$，即：

$$ \hat{w} = \mathop{\arg \max}_{w} log(\prod_{i=1}^{N} {\frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2})}) \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {log(\frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2}))} \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2})} \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {-(y^{(i)}-w^Tx^{(i)})^2} \\ = \mathop{\arg \min}_{w} \sum_{i=1}^{N} {(y^{(i)}-w^Tx^{(i)})^2} $$

问题等价于最小二乘法。

# 2. 正规方程法
将线性回归表示为矩阵形式，

记样本矩阵$$X=[x^{(1)};...;x^{(N)}]^T \in \Bbb{R}^{N×(d+1)}$$，标签向量$$y=[y^{(1)};...;y^{(N)}]^T \in \Bbb{R}^{N}$$，

待求解权重参数$$w \in \Bbb{R}^{d+1}$$，预测结果$$\hat{y} \in \Bbb{R}^{N}$$，则：

$$ \hat{y} = Xw $$

损失函数：

$$ L(w) = \frac{1}{N} \mid\mid Xw-y \mid\mid^2 = \frac{1}{N} (Xw-y)^T(Xw-y) $$

该目标函数是凸函数，可以直接求梯度令其为零，得到全局最小解：

$$ ▽_wL(w) = ▽_w \frac{1}{N} (Xw-y)^T(Xw-y) = 2X^TXw - 2Xy = 0 $$

整理得到：

$$ X^TXw = Xy $$

上式叫做**正规方程（normal equation）**。

通常样本数远大于特征维度$N>>d$，$X^TX$是可逆的，可以得到权重参数的**解析解（closed-form）**：

$$ w = (X^TX)^{-1}Xy = X^+y $$

其中$$X^+=(X^TX)^{-1}$$称为$X$的**伪逆（pseudo inverse）**，当$X^TX$不可逆时，也有其他方法可求。

### 从线性代数的角度理解线性回归
对于线性回归问题：

$$ \hat{y} = Xw $$

上述矩阵方程有解的条件是向量$\hat{y}$在矩阵$X$的列空间中，即向量$\hat{y}$是矩阵$X$的列向量的线性组合。

# 3. 误差分析
预测标签：

$$ \hat{y} = Xw = X(X^TX)^{-1}Xy = XX^+y = Hy $$

其中$H$为**投影矩阵**，几何意义是将向量$y$投影到样本矩阵$X$的列空间：

![](https://pic.downk.cc/item/5ed0be5fc2a9a83be58c1e5f.jpg)

投影误差：

$$ L(w) = \frac{1}{N} \mid\mid Xw-y \mid\mid^2 = \frac{1}{N} \mid\mid XX^+y-y \mid\mid^2 = \frac{1}{N} \mid\mid (H-I)y \mid\mid^2 $$

由此不难得出，$H-I$为也是投影矩阵，将向量$y$投影到与样本矩阵$X$的列空间正交的子空间（左零空间）中。

线性回归的样本内误差$E_{in}$和总样本误差$E_{out}$表示如下：

$$ E_{in} = (noise)(1-\frac{d+1}{N}) $$

$$ E_{out} = (noise)(1+\frac{d+1}{N}) $$

绘制**学习曲线（learning curve）**，误差最终趋近于一个噪声值$σ^2$:

![](https://pic.downk.cc/item/5ed0c0d3c2a9a83be58f9ff8.jpg)

**学习曲线**是指对于一个给定的模型，训练稳定时的$E_{in}$、$E_{out}$随训练样本总数$N$的变化。


# 4. 非线性回归
对样本进行非线性变换，将低维空间变换到高维空间中:

$$ z = Φ(x) $$

如$Q$阶的多项式变换：

$$ Φ_Q(x) = (1,x_1,x_2,...,x_d,x_1^2,x_1x_2,...,x_d^2,...,x_1^Q,x_1^{Q-1}x_2,...,x_d^Q) $$

样本的原始特征维度为$d+1$，变换后的特征维度为$C_{Q+d}^{Q}=O(Q^d)$。

上述多项式变换存在问题，即若输入的范围限定在$±1$之间，则高阶幂的值会比低阶幂的值小得多，需要给高阶幂更大的权重。为了避免这种数据差异很大的情况，可以使用**勒让德多项式（Legendre Polynomials）**：

![](https://pic.downk.cc/item/5ed37d6dc2a9a83be53bb90c.jpg)


# 5. 过拟合
下面是使用$2$阶非线性回归和$10$阶非线性回归对应的**学习曲线（learning curve）**：

![](https://pic.downk.cc/item/5ed2478fc2a9a83be5a5cff6.jpg)

由图可以观察到，当样本数足够的时候，两个模型都可以收敛到期望的误差，高阶模型的误差更小；但当样本数不足的时候，高阶模型更容易出现$$E_{in}<E_{out}$$的情况，即出现过拟合。

过拟合的原因：
1. 样本数$N$不足；
2. **stochastic noise** $σ^2$：样本中的随机误差；
3. **deterministic noise**：目标函数过于复杂带来的系统误差；
4. 模型复杂度过高。

过拟合的解决方法：
1. **data cleaning**：对数据集中标注错误的样本进行修正；
2. **data pruning**：删除数据集中标注错误的样本；
3. **data hinting**：增加更多的样本（数据增广）；
4. **regularization**：正则化，限制模型的复杂度。

# 6. 验证集
**验证集 validation**用于选择学习率、正则化系数等超参数。

不能用训练集选择模型，因为经过足够的训练，模型在训练集上的表现一定会过拟合；也不能用测试集选择模型，因为防止信息泄露测试集不可使用。

首先把训练集进一步划分成训练集和验证集，对于不同的模型，在训练集上训练后在验证集上测试，得到验证误差；选择验证误差最小的模型假设作为最终的假设函数，用所有训练集进行最终的训练，并在测试集上测试得到结果。

![](https://pic.downk.cc/item/5ed32727c2a9a83be5caccad.jpg)

验证集大小$K$的选择也很重要:
- 若$K$太小，则训练集足够多，但是验证集样本太少，不能代表测试集的数据分布；
- 若$K$太大，则验证集足够多，但是训练集样本太少，不能代表测试集的数据分布。

### 留一交叉验证
**留一交叉验证（Leave-One-Out Cross Validation）**是指验证集大小$K=1$，每次仅选择1个样本作为验证集。

实际中遍历样本集，每次选择其中一个作为验证集，其余作为训练集；将所有验证误差的平均作为最终的验证误差。

### V折交叉验证
**V折交叉验证（V-Fold Cross Validation）**是指将样本集分为$V$份，每次取其中一份作为验证集，其余为训练集，

![](https://pic.downk.cc/item/5ed378fcc2a9a83be5352e0b.jpg)

通常V折交叉验证比留一交叉验证结果更可靠，但计算量更大。

# 7. Tube Regression
**Tube Regression**是指在计算样本点的回归误差时，在回归线上下分别划定一个区域（中立区），如果数据点分布在这个区域内，则不计算误差，只有分布在中立区域之外才计算误差。

![](https://pic.downk.cc/item/5ed5f015c2a9a83be552b8e9.jpg)

假设中立区的宽度为$2ε$，则回归误差（上图标红的线段）写作：

$$ err(x) = \begin{cases} 0, & \mid w^Tx-y \mid ≤ ε \\ \mid w^Tx-y \mid - ε, & \mid w^Tx-y \mid > ε \end{cases} = max(0,\mid w^Tx-y \mid - ε) $$

上述误差也被称作**$ε$-insensitive error**。

比较**tube**误差和平方误差，两者是类似的，但**tube**误差对**outlier**不敏感：

![](https://pic.downk.cc/item/5ed5f22dc2a9a83be555ba8f.jpg)

# 8. Ridge Regression
**岭回归（Ridge Regression）**是引入了**L2正则化**的线性回归，求解问题如下：

$$ L(w) = \sum_{i=1}^{N} {(w^Tx^{(i)}-y^{(i)})^2} + λw^2 = (Xw-y)^T(Xw-y)+ λw^Tw $$

令梯度为零可以得到：

$$ ▽_wL(w) = 2X^TXw - 2Xy + 2λw = 0 $$

该问题的解析解为：

$$ w = (X^TX+λI)^{-1}Xy $$

注意到正则化系数$λ$通常大于零，故矩阵$X^TX+λI$一定可逆。

### 岭回归等价于噪声和先验服从高斯分布的最大后验估计
引入高斯噪声$ε$~$N(0,σ^2)$，对线性回归建模：

$$ y = w^Tx + ε $$

贝叶斯角度认为参数$w$不再是常数，而是随机变量，假设其先验概率为$N(0,σ_0^2)$,

由贝叶斯定理，参数$w$的后验概率：

$$ P(w \mid y) = \frac{P(y \mid w)P(w)}{P(y)} $$

由最大后验估计：

$$ \hat{w} = \mathop{\arg \max}_{w}logP(w \mid y) = \mathop{\arg \max}_{w}logP(y \mid w)P(w) \\ = \mathop{\arg \max}_{w} log(\prod_{i=1}^{N} {\frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2})\frac{1}{\sqrt{2\pi}σ_0}exp(-\frac{w^Tw}{2σ_0^2})}) \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {log(\frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2})\frac{1}{\sqrt{2\pi}σ_0}exp(-\frac{w^Tw}{2σ_0^2}))} \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2}-\frac{w^Tw}{2σ_0^2}} \\ = \mathop{\arg \min}_{w} \sum_{i=1}^{N} {(y^{(i)}-w^Tx^{(i)})^2+\frac{σ^2}{σ_0^2}w^Tw} $$

该问题等价于引入L2正则化的最小二乘法（岭回归）。

# 9. Kernel Ridge Regression
将支持向量机中的[核方法](https://0809zheng.github.io/2020/03/14/SVM.html)引入岭回归。

### Representer Theorem
如果线性模型使用了$L2$正则化，即优化目标函数为:

$$ min_w \quad \frac{λ}{N}w^Tw + \frac{1}{N}\sum_{n=1}^{N} {err(y^{(n)},w^Tx^{(n)})} $$

则参数$w$的最优解可以表示为：

$$ w^* = \sum_{n=1}^{N} {β_nx^{(n)}} $$

证明如下：

假设最优解由两部分组成：$$w^* = w_{\|\|}+w_{⊥}$$,

其中$w_{\|\|}$平行于样本数据所张成的空间$span(x^{(1)},...,x^{(N)})$；$w_{⊥}$垂直于样本数据所张成的空间；

则目标函数的第二项：

$$ err(y^{(n)},{w^*}^Tx^{(n)}) = err(y^{(n)},(w_{\|\|}+w_{⊥})^Tx^{(n)}) = err(y^{(n)},w_{\|\|}^Tx^{(n)}) $$

且目标函数的第一项：

$$ {w^*}^Tw^* = (w_{\|\|}+w_{⊥})^T(w_{\|\|}+w_{⊥}) = w_{\|\|}^Tw_{\|\|}+2w_{\|\|}^Tw_{⊥}+w_{⊥}^Tw_{⊥} \\ = w_{\|\|}^Tw_{\|\|}+w_{⊥}^Tw_{⊥} ≥ w_{\|\|}^Tw_{\|\|} $$

显然$w_{\|\|}$是一个满足条件的更优解。

故最优参数$w$可被样本数据线性表示。

根据这个定理，支持向量机、感知机、逻辑回归的所求权重参数都可以表达为上述形式。

岭回归的目标函数：

$$ L(w) = \frac{1}{N}\sum_{i=1}^{N} {(w^Tx^{(i)}-y^{(i)})^2} + \frac{λ}{N}w^Tw $$

上述优化问题的最优解$w$可以表示为：

$$ w^* = \sum_{n=1}^{N} {β_nx^{(n)}} $$

代入目标函数：

$$ L(β) = \frac{1}{N} \sum_{i=1}^{N} {(\sum_{n=1}^{N} {β_n{x^{(n)}}^Tx^{(i)}}-y^{(i)})^2} + \frac{λ}{N}\sum_{i=1}^{N} {\sum_{j=1}^{N} {β_iβ_j{x^{(i)}}^Tx^{(j)}}} $$


### Kernel Trick
核方法是指**引入核函数来代替样本的特征转换和内积运算**，记$$K(x,x')={φ(x)}^Tφ(x')$$，则核逻辑回归的损失函数为：

$$ L(β) = \frac{1}{N} \sum_{i=1}^{N} {(\sum_{n=1}^{N} {β_nK(x^{(n)},x^{(i)})}-y^{(i)})^2} + \frac{λ}{N}\sum_{i=1}^{N} {\sum_{j=1}^{N} {β_iβ_jK(x^{(i)},x^{(j)})}} $$

上式写作矩阵形式：

$$ L(β) = \frac{1}{N} (y-Kβ)^T(y-Kβ) + \frac{λ}{N}β^TKβ $$

对$β$求梯度，令其为零，(注意到$K$是对称矩阵)可得：

$$ ▽_βL(β) = \frac{1}{N}(2K^TKβ-2K^Ty) + \frac{λ}{N}2Kβ = \frac{2K^T}{N}(Kβ-y+λβ) = 0 $$

可解得引入核函数的岭回归的解析解：

$$ β = (K+λI)^{-1}y $$

对矩阵$K+λI$的讨论：
- 由于$λ>0$，$K$是半正定矩阵，其逆矩阵一定存在。
- 该矩阵是一个稠密（dense）矩阵，大部分元素不是0，求逆过程计算量较大。

求得$β$后，可以得到回归函数：

$$ y = \sum_{n=1}^{N} {β_nφ(x^{(n)})^Tφ(x)} = \sum_{n=1}^{N} {β_nK(x^{(n)},x)} $$

# 10. Lasso Regression
**Lasso回归**是引入了**L1正则化**的线性回归，求解问题如下：

$$ L(w) = \sum_{i=1}^{N} {(w^Tx^{(i)}-y^{(i)})^2} + λ\sum_{d}^{} {\mid w_d \mid} $$

Lasso回归等价于高斯噪声、参数w服从拉普拉斯分布的最大后验估计。