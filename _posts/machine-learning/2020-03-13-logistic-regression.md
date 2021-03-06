---
layout: post
title: '逻辑回归'
date: 2020-03-13
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ed75927c2a9a83be54c53ef.jpg'
tags: 机器学习
---

> Logistic Regression.

本文目录：
1. 逻辑回归模型
2. Logistic函数
3. 交叉熵损失
4. 多类别分类
5. 核逻辑回归

# 1. 逻辑回归模型
**逻辑回归（logistic regression）**是一种二分类模型，

若记每一个样本点$x = (1,x_1,x_2,...,x_d)^T \in \Bbb{R}^{d+1}$，模型权重参数$w = (w_0,w_1,w_2,...,w_d)^T \in \Bbb{R}^{d+1}$，则逻辑回归模型：

$$ \hat{y} = σ(\sum_{j=0}^{d} {w_jx_j}) = σ(w^Tx) $$

其中$σ$函数为**Logistic函数**。


# 2. Logistic函数
**Logistic函数**又叫**Sigmoid函数**，其表达式、一阶导数和函数曲线如下：

$$ σ(x)= \frac{1}{1+e^{-x}} $$

$$ σ'(x)= σ(x)(1-σ(x)) $$

![](https://pytorch.org/docs/stable/_images/Sigmoid.png)


# 3. 交叉熵损失
逻辑回归的损失函数可以由**极大似然估计**推导。

### 标签为0和1
把分类标签设置为$$y \in \{0,+1\}$$。

**Logistic函数**把输入压缩到$0$到$1$之间，可以看作对$y$预测的概率：

$$ P(y=1 \mid x) = σ(w^Tx) $$

$$ P(y=0 \mid x) = 1-σ(w^Tx) $$

上式是**Bernoulli分布**，可以表达为：

$$ P(y \mid x) = {σ(w^Tx)}^{y}{(1-σ(w^Tx))}^{1-y} $$

若样本集$$X=\{x^{(1)},...,x^{(N)}\}$$，标签集$$y=\{y^{(1)},...,y^{(N)}\}$$，列出对数似然方程：

$$ ln( \prod_{i=1}^{N} { {σ(w^T x^{(i)})}^{y^{(i)}} {(1-σ(w^T x^{(i)}))}^{1-y^{(i)}} } ) = \sum_{i=1}^{N} { ln( {σ(w^Tx^{(i)})}^{y^{(i)}} {(1-σ(w^Tx^{(i)}))}^{1-y^{(i)}} ) } \\ = \sum_{i=1}^{N} { y^{(i)} ln(σ(w^Tx^{(i)})) + (1-y^{(i)}) ln(1-σ(w^Tx^{(i)})) } $$

似然概率极大化，等价于以下损失函数极小化：

$$ L(w) = \sum_{i=1}^{N} {-y^{(i)}ln(σ(w^Tx^{(i)})) - (1-y^{(i)})ln(1-σ(w^Tx^{(i)}))} $$

上式为**交叉熵损失（cross-entropy loss）**。

若希望损失函数为零，则需：

$$ \begin{cases} w^Tx → +∞, & y=1 \\ w^Tx → -∞, & y=0 \end{cases} $$

此时要求数据集是线性可分的。

### 标签为±1
也可以把分类标签设置为$$y \in \{-1,+1\}$$，从而与感知机、支持向量机等模型的书写方式相同。

由**Logistic函数**的性质：$1-σ(x)=σ(-x)$；输出概率为：

$$ P(y=1 \mid x) = σ(w^Tx) $$

$$ P(y=-1 \mid x) = 1-σ(w^Tx) = σ(-w^Tx) $$

或者统一写为：

$$ P(y \mid x) = σ(yw^Tx) $$

若样本集$$X={x^{(1)},...,x^{(N)}}$$，标签集$$y={y^{(1)},...,y^{(N)}}$$，列出对数似然方程：

$$ ln(\prod_{i=1}^{N} {σ(y^{(i)}w^Tx^{(i)})} = \sum_{i=1}^{N} {ln(σ(y^{(i)}w^Tx^{(i)}))} $$

似然概率极大化，等价于以下损失函数极小化：

$$ L(w) = \sum_{i=1}^{N} {-ln(σ(y^{(i)}w^Tx^{(i)}))} = \sum_{i=1}^{N} {ln(1+exp(-y^{(i)}w^Tx^{(i)}))} $$

上式也为**交叉熵损失（cross-entropy loss）**。

### 损失函数的比较
当标签为$±1$时，比较感知机、线性回归和逻辑回归的损失函数：
- 感知机：**0/1损失**：$$E_{0/1}=[sign(wx)=y]=[sign(ywx)=1]$$
- 线性回归：**均方误差**：$$E_{sqr}=(wx-y)^2=(ywx-1)^2$$
- 逻辑回归：**交叉熵**：$$E_{ce}=ln(1+exp(-ywx))$$
- 以$2$为底的**交叉熵**：$$E_{scaled-ce}=log_2(1+exp(-ywx))=\frac{1}{ln2}E_{ce}$$

经过换底后的交叉熵损失和平方损失都是$0/1$损失的上界：

![](https://pic.downk.cc/item/5ed10ce1c2a9a83be5064c19.jpg)

比较可得，$0/1$损失整体更小，但优化困难（$NP-hard$）；均方误差和交叉熵更大，都是凸优化问题（解析解、梯度下降）。


# 4. 多类别分类
**多类别分类（multiclass classification）**的基本思想是拆分成若干二分类问题，有两种思路。

### One-versus-All
**One-versus-All（OVA）**是指每一次分类中使用所有样本，选择其中某一类为正样本，其余所有样本为负样本。

![](https://pic.downk.cc/item/5ed11069c2a9a83be50c76d3.jpg)

- 优点：实现简单；
- 缺点：对于二分类器来说正负样本不平衡。

### One-versus-One
**One-versus-One（OVO）**是指每一次分类中使用其中两类样本，选择一类为正样本，另一类为负样本。

![](https://pic.downk.cc/item/5ed10fbdc2a9a83be50b55da.jpg)

- 优点：每一次二分类的样本基本上是平衡的；
- 缺点：需要训练组合数的二分类器，增加训练复杂度。


# 5. 核逻辑回归
将支持向量机中的[核方法](https://0809zheng.github.io/2020/03/14/SVM.html)引入逻辑回归。

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

### Kernel Trick
使用$L2$正则化的逻辑回归的损失函数（标签为0和1）如下：

$$ L(w) = \sum_{i=1}^{N} {-y^{(i)}ln(σ(w^Tx^{(i)})) - (1-y^{(i)})ln(1-σ(w^Tx^{(i)}))} + \frac{λ}{N}w^Tw $$

由**Representer Theorem**，权重最优解可以表示为：

$$ w^* = \sum_{n=1}^{N} {β_nx^{(n)}} $$

代入损失函数为：

$$ L(β) = \sum_{i=1}^{N} {-y^{(i)}ln(σ(\sum_{n=1}^{N} {β_n{x^{(n)}}^Tx^{(i)}})) - (1-y^{(i)})ln(1-σ(\sum_{n=1}^{N} {β_n{x^{(n)}}^Tx^{(i)}}))} \\ + \frac{λ}{N}\sum_{i=1}^{N} {\sum_{j=1}^{N} {β_iβ_j{x^{(i)}}^Tx^{(j)}}} $$

核方法是指**引入核函数来代替样本的特征转换和内积运算**，记$$K(x,x')={φ(x)}^Tφ(x')$$，则核逻辑回归的损失函数为：

$$ L(β) = \sum_{i=1}^{N} {-y^{(i)}ln(σ(\sum_{n=1}^{N} {β_nK(x^{(n)},x^{(i)})})) - (1-y^{(i)})ln(1-σ(\sum_{n=1}^{N} {β_nK(x^{(n)},x^{(i)})}))} \\ + \frac{λ}{N}\sum_{i=1}^{N} {\sum_{j=1}^{N} {β_iβ_jK(x^{(i)},x^{(j)})}} $$

这是一个无约束的优化问题，可以用梯度下降法求解。

