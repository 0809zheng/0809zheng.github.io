---
layout: post
title: '随机森林'
date: 2020-03-20
author: 郑之杰
cover: 'https://pic.downk.cc/item/5efb023c14195aa5948bf48e.jpg'
tags: 机器学习
---

> Random Forest.

**随机森林（Random Forest）**是一种把[**bagging**](https://0809zheng.github.io/2020/03/17/bagging.html)和[**决策树**](https://0809zheng.github.io/2020/03/19/decision-tree.html)结合起来的方法。
- **bagging**是一种集成方法，通过对样本集进行bootstrap分成若干个子样本集，训练多个模型进行集成，可以减小模型本身的方差；
- **fully-grown CART**是一种模型方差本身较大的决策树，对于单一模型容易过拟合。

该算法的优点：
1. 每一个子树训练是独立的，该算法可以并行实现；
2. 继承了**CART**的优点；
3. 使用**bagging**集成多个树，避免了过拟合问题。


# 1. random-combination CART
在训练每一个**CART子树**时，为增加模型的能力，每一次选择分支（branching）时，可以随机的选择样本的一部分特征，比如选择其中的$d'$个特征，使用特征$(x_{i_1},x_{i_2},...,x_{i_{d'}})$构建分支。

进一步，可以在选择分支时进行**特征映射Feature Projection**，即：

$$ φ(x) = Px $$

其中$P$是从高维映射到低维的映射矩阵，相当于对原特征进行随机的线性组合，映射到新的特征空间。这样做使得在选择分支时不是使用决策桩针对某一特征，而是使用感知机同时处理多个特征。

![](https://pic.downk.cc/item/5edc9025c2a9a83be50e9fa4.jpg)

上述方法为子树增加了更多的随机性。

# 2. Out-Of-Bag Estimate
使用**bagging**随机生成子数据集也会为算法引入随机性。

假设使用bootstrap在$N$个样本中进行$T$轮有放回的抽样过程，生成$T$个子样本集：

![](https://pic.downk.cc/item/5edc7a71c2a9a83be5dca99a.jpg)

在某一轮中，总样本集中未被选中的的样本称为**out-of-bag（OOB） example**.

假设在$N$个样本中有放回的抽样，每轮抽取$N$个样本组成子样本集，则其中某一样本$(x_n,y_n)$未被抽中（成为**OOB**样本）的概率为：

$$ (1-\frac{1}{N})^N = \frac{1}{(\frac{N}{N-1})^N} = \frac{1}{(1+\frac{1}{N-1})^N} ≈ \frac{1}{e} $$

使用**OOB**样本可以进行**自验证（self-validation）**：

对于每个样本(以上图$(x_N,y_N)$为例)，将所有未使用该样本训练的模型集成起来作为一个子模型：

$$ G_N^-(x) = average(g_2,g_3,...,g_T) $$

可以计算该样本在这个模型上的误差；进一步计算所有样本在其对应模型上的平均表现：

$$ E_{OOB} = \frac{1}{N} \sum_{n=1}^{N} {error(y_n,g_n^-(x_n))} $$

随机森林算法不需要单独的验证集，而是用$E_{OOB}$代替验证集误差。

# 3. Feature Selection
随机森林可以用来进行**特征选择（Feature Selection）**，实现方法是通过**permutation test（随机排序测试）**。

首先对于给定样本集训练随机森林$G$，并进一步计算自验证误差$E_{OOB}$；

对于样本的第i个特征，将其在样本集中的值随机打乱后，计算打乱后的自验证误差$E_{OOB}^p$；

则第i个特征的重要性定义为将该特征随机排序打乱后引入的自验证误差值：

$$ importance(i) = E_{OOB}(G)-E_{OOB}^p(G) $$