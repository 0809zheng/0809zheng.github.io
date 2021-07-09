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
1. 线性回归 **Linear Regression**
1. 广义线性模型 **Generalized Linear Model**
4. 非线性回归 **Non-Linear Regression**
5. 过拟合 **Overfitting**


# 1. 线性回归
**线性回归(Linear Regression)**是最基本的回归方法，其假设空间的假设函数是线性函数。

对于每一个样本数据$x = (x_1,x_2,...,x_d)^T \in \Bbb{R}^{d}$，其中每一个元素$x_i$称为样本的**特征(feature)**或**属性(attribute)**。为样本的每一个特征引入一个重要性**权重(weight)**参数$w_i$，并额外引入一个**偏置(bias)**参数$b$，则线性回归具有很好的**可解释性(comprehensibility)**：输出$\hat{y}$为样本输入特征的加权线性组合。计算如下：

$$ \hat{y} = \sum_{j=1}^{d} {w_jx_j} + b$$

若把偏置参数$b$看作$w_0 \cdot 1$，
即为每一个样本点额外增加一个维度$x = (1,x_1,x_2,...,x_d)^T \in \Bbb{R}^{d+1}$，模型权重参数也额外增加一个维度$w = (w_0,w_1,w_2,...,w_d)^T \in \Bbb{R}^{d+1}$，则线性回归模型可以被简化为：

$$ \hat{y} = \sum_{j=0}^{d} {w_jx_j} = w^Tx $$

若所有样本组成样本集$$X=\{x^{(1)},...,x^{(N)}\}$$，对应的标签集$$y=\{y^{(1)},...,y^{(N)}\}$$。该问题也被称作**多元(multivariate,多变量)**线性回归，使用**均方误差(mean squared error，MSE)**衡量模型的好坏：

$$ L(w) = \frac{1}{N} \sum_{i=1}^{N} {(\sum_{j=0}^{d} {w_jx_j^{(i)}}-y^{(i)})^2} = \frac{1}{N} \sum_{i=1}^{N} {(w^Tx^{(i)}-y^{(i)})^2} $$

求解上述均方误差最小化问题有两种可行的思路，即**最小二乘法**和**正规方程法**。

### 讨论1.1：均方误差损失可否用于分类问题？
由下图可以看出，对于分类任务，$0/1$损失始终要比均方误差损失小，故误差上界小；虽然均方误差损失使得上界变得宽松了，但是由于优化问题变得简单(从**NP-hard**变为凸优化)，也能得到不错的结果。

![](https://pic.downk.cc/item/5ed0baaec2a9a83be5867638.jpg)

## (1). 最小二乘法 Least Square Method
由于均方误差是凸函数(见本节讨论1.2)，因此$w$的最优解在均方误差关于$w$的导数为$0$时取得。为简化问题，假设样本只有一个特征(即单变量线性回归)，对应的均方误差损失如下：

$$ L(w,b) = \frac{1}{N} \sum_{i=1}^{N} {(wx^{(i)}+b-y^{(i)})^2} $$

另上式关于$w$和$b$的导数为$0$，计算如下：

$$ \frac{\partial L(w,b)}{\partial w} = \frac{\partial}{\partial w}[\frac{1}{N} \sum_{i=1}^{N} {(wx^{(i)}+b-y^{(i)})^2}] = \frac{1}{N} \sum_{i=1}^{N} {2(wx^{(i)}+b-y^{(i)})x^{(i)}} = 0 $$

$$ \frac{\partial L(w,b)}{\partial b} = \frac{\partial}{\partial b}[\frac{1}{N} \sum_{i=1}^{N} {(wx^{(i)}+b-y^{(i)})^2}] = \frac{1}{N} \sum_{i=1}^{N} {2(wx^{(i)}+b-y^{(i)})} = 0 $$

联立上述两式可得该问题的**闭式(closed-form)解**：

$$ w = \frac{\sum_{i=1}^{N} y^{(i)}(x^{(i)}-\frac{1}{N} \sum_{i=1}^{N} x^{(i)})}{\sum_{i=1}^{N}(x^{(i)})^2-\frac{1}{N}(\sum_{i=1}^{N}(x^{(i)}))^2} = \frac{\sum_{i=1}^{N} y^{(i)}(x^{(i)}-\overline{x})}{\sum_{i=1}^{N}(x^{(i)})^2-\frac{1}{N}(\sum_{i=1}^{N}(x^{(i)}))^2} $$

$$ b = \frac{1}{N} \sum_{i=1}^{N} (y^{(i)}-wx^{(i)}) $$

上述利用均方误差最小化求解线性回归的方法被称为**最小二乘法(least square method)**。该方法的几何意义是在样本空间中寻找一个超平面，使得所有样本点到该超平面的距离平方最小：

![](https://pic.downk.cc/item/5ed0bbb7c2a9a83be5882cf8.jpg)


### 讨论1.2：均方误差损失是凸函数
定义在区间$\[a,b\]$上的**凸(convex)函数** $f$，是指对于区间中任意两点$x_1<x_2$均有：

$$ f(\frac{x_1+x_2}{2}) ≤ \frac{f(x_1)+f(x_2)}{2} $$

$$f(x)=x^2$$就是一个典型的凸函数，该凸函数的全局最小值位于$\nabla f=0$处。对于实数集上的函数$f$，可由二阶导数$\nabla^2 f$判断其凸性。若二阶导数$\nabla^2 f$在区间上**非负**则为凸函数。验证均方误差的凸性：

$$\nabla^2 L(w) = \nabla^2 \frac{1}{N} \sum_{i=1}^{N} {(w^Tx^{(i)}-y^{(i)})^2} \\ = \nabla \frac{1}{N} \sum_{i=1}^{N} 2(w^Tx^{(i)}-y^{(i)})x^{(i)} = \frac{1}{N} \sum_{i=1}^{N} 2(x^{(i)})^2 ≥0 $$


### 讨论1.3：最小二乘法等价于噪声服从高斯分布的极大似然估计
引入高斯噪声$ε$~$N(0,σ^2)$，对线性回归建模：

$$ y = w^Tx + ε $$

其中$(x,y)$是样本数据，$w$是未知的常数，每个样本点受到了高斯噪声的干扰。则条件变量$y \| x;w$服从于$N(w^Tx,σ^2)$。列出条件概率：

$$ P(y | x;w) = \frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y-w^Tx)^2}{2σ^2}) $$

采用极大似然估计的方法估计参数$w$，即：

$$ \hat{w} = \mathop{\arg \max}_{w} log(\prod_{i=1}^{N} {\frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2})}) \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {log(\frac{1}{\sqrt{2\pi}σ}exp(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2}))} \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {(-\frac{(y^{(i)}-w^Tx^{(i)})^2}{2σ^2})} \\ = \mathop{\arg \max}_{w} \sum_{i=1}^{N} {-(y^{(i)}-w^Tx^{(i)})^2} \\ = \mathop{\arg \min}_{w} \sum_{i=1}^{N} {(y^{(i)}-w^Tx^{(i)})^2} $$

问题等价于最小二乘法。

## (2). 正规方程法
将线性回归表示为矩阵形式：记样本矩阵$$X=[x^{(1)};...;x^{(N)}]^T \in \Bbb{R}^{N×(d+1)}$$，标签向量$$y=[y^{(1)};...;y^{(N)}]^T \in \Bbb{R}^{N}$$，

待求解权重参数$$w \in \Bbb{R}^{d+1}$$，预测结果$$\hat{y} \in \Bbb{R}^{N}$$，则：

$$ \hat{y} = Xw $$

损失函数：

$$ L(w) = \frac{1}{N} || Xw-y ||^2 = \frac{1}{N} (Xw-y)^T(Xw-y) $$

该目标函数是凸函数，可以直接求梯度令其为零，得到全局最小解：

$$ \nabla_wL(w) = \nabla_w \frac{1}{N} (Xw-y)^T(Xw-y) = 2X^TXw - 2Xy = 0 $$

整理得到：

$$ X^TXw = Xy $$

上式叫做**正规方程（normal equation）**。

通常样本数远大于特征维度$N$>>$d$，$X^TX$是可逆的，可以得到权重参数的**解析解（closed-form）**：

$$ w = (X^TX)^{-1}Xy = X^+y $$

其中$$X^+=(X^TX)^{-1}$$称为$X$的**伪逆（pseudo inverse）**，当$X^TX$不可逆时(如样本的特征维度大于样本总数)，也有其他方法可求。

### 讨论1.4：从线性代数的角度理解线性回归
对于线性回归问题：

$$ \hat{y} = Xw $$

上述矩阵方程有解的条件是向量$\hat{y}$在矩阵$X$的列空间中，即向量$\hat{y}$是矩阵$X$的列向量的线性组合。

### 讨论1.5： 正规方程法的误差分析
预测标签：

$$ \hat{y} = Xw = X(X^TX)^{-1}Xy = XX^+y = Hy $$

其中$H$为**投影矩阵**，几何意义是将向量$y$投影到样本矩阵$X$的列空间：

![](https://pic.downk.cc/item/5ed0be5fc2a9a83be58c1e5f.jpg)

投影误差：

$$ L(w) = \frac{1}{N} || Xw-y ||^2 = \frac{1}{N} || XX^+y-y ||^2 = \frac{1}{N} || (H-I)y ||^2 $$

由此不难得出，$H-I$为也是投影矩阵，将向量$y$投影到与样本矩阵$X$的列空间正交的子空间（左零空间）中。

线性回归的样本内误差$E_{in}$和总样本误差$E_{out}$表示如下：

$$ E_{in} = (noise)(1-\frac{d+1}{N}) $$

$$ E_{out} = (noise)(1+\frac{d+1}{N}) $$

绘制**学习曲线（learning curve）**，误差最终趋近于一个噪声值$σ^2$:

![](https://pic.downk.cc/item/5ed0c0d3c2a9a83be58f9ff8.jpg)

**学习曲线**是指对于一个给定的模型，训练稳定时的$E_{in}$、$E_{out}$随训练样本总数$N$的变化。

# 2. 广义线性模型
一般地，线性回归模型可以简写为：

$$ y=w^Tx+b $$

引入单调可微函数$g(\cdot)$，可以建立**广义线性模型(generalized linear model)**，其中$g(\cdot)$称为**联系函数(link function)**:

$$ y=g^{-1}(w^Tx+b) $$

线性回归模型建立的是输入空间到输出空间的线性函数映射；广义线性模型可以通过联系函数$g(\cdot)$建立从输入空间到输出空间的非线性函数映射。比如**对数线性回归(log-linear regression)**：

$$ ln(y) = w^Tx+b $$

# 3. 非线性回归
**非线性回归(non-linear regression)**是指先对样本的特征进行非线性变换，再通过线性回归方法进行建模。其中对样本进行非线性变换，通常是将样本从低维空间变换到高维空间中:

$$ z = Φ(x) $$

如$Q$阶的多项式变换：

$$ Φ_Q(x) = (1,x_1,x_2,...,x_d,x_1^2,x_1x_2,...,x_d^2,...,x_1^Q,x_1^{Q-1}x_2,...,x_d^Q) $$

样本的原始特征维度为$d+1$，变换后的特征维度为$C_{Q+d}^{Q}=O(Q^d)$。

上述多项式变换存在问题，即若输入的范围限定在$±1$之间，则高阶幂的值会比低阶幂的值小得多，需要给高阶幂更大的权重。为了避免这种数据差异很大的情况，可以使用**勒让德多项式（Legendre Polynomials）**：

![](https://pic.downk.cc/item/5ed37d6dc2a9a83be53bb90c.jpg)


# 4. 过拟合
下面是使用$2$阶非线性回归和$10$阶非线性回归对应的**学习曲线(learning curve)**：

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
