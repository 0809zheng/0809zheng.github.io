---
layout: post
title: 'Recommender System：推荐系统'
date: 2020-05-08
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ee37639c2a9a83be5987fba.jpg'
tags: 机器学习
---

> Recommender System.

假设有$M$部电影，$N$个用户对其进行评分。该数据集可表示为：

$$ {(x_n=(n),y_n=(r_{n1},r_{n2},...,r_{nM})),n=1,2,...,N} $$

其中$r_{nm}$表示第$n$个用户对第$m$部电影的评分，若空缺则记为$?$。

将评分表示为矩阵形式$R$：

![](https://pic.downk.cc/item/5ee045e9c2a9a83be5c07d16.jpg)

该数据集中的$x_n$表示用户的$ID$，是一种**类别特征（categorical feature）**，将其转换为**数值特征（numerical feature）**，将每个用户表示为**one-hot编码**（也叫**binary vector encoding**）$x_n=(0,...0,,1,0,...,0)$；


# 1. Linear Network
采用单隐藏层、线性神经元的**线性网络（linear network）**进行建模：

![](https://pic.downk.cc/item/5ee046b7c2a9a83be5c27d08.jpg)

其中输入节点$x_n$取值：第$n$个用户时取值为1，否则为0。输出节点$y_m$表示输入$x_n$时第$n$个用户对第$m$部电影的评价分数。

该网络模型记为：

$$ y = W^TVx $$

其中网络权重矩阵的含义：
- $V \in \Bbb{R}^{d×N}$，第$i$列表示网络学习到的第$i$个用户的$d$维特征；
- $W \in \Bbb{R}^{d×M}$，第$j$列表示网络学习到的第$j$部电影的$d$维特征。

# 2. Low Rank Matrix Factorization
根据线性网络，第$n$个用户对第$m$部电影的评分$r_{nm}$可表示为：

$$ r_{nm} = w_m^Tv_n = v_n^Tw_m $$

相当于对评分矩阵$R$分解：$R=V^TW$

![](https://pic.downk.cc/item/5ee0468ac2a9a83be5c204fe.jpg)

由于矩阵$R$的秩不会超过矩阵$V$和$W$的秩，通常$rank(V)=rank(W)=d<m,n$，该方法称为**低秩矩阵分解（Low Rank Matrix Factorization）**。

定义对于给定样本集上的损失函数：

$$ Loss(w,v) = \sum_{n,m}^{} {(r_{nm}-w_m^Tv_n)^2} + λ\sum_{m}^{}  \mid\mid w_m \mid\mid^2 + λ\sum_{n}^{}  \mid\mid v_n \mid\mid^2 $$

# 3. Learning Method

### (1)Alternating Least Square
上述优化问题需要求解两组参数$W$和$V$。每一个优化时固定一组优化另一组；交替进行。

- 当固定用户特征矩阵$V$优化电影特征矩阵$W$时，对于每一部电影$m$的特征$w_m$，优化问题退化为线性回归问题；
- 当固定电影特征矩阵$W$优化用户特征矩阵$V$时，对于每一个用户$n$的特征$v_n$，优化问题退化为线性回归问题。

这样每次更新共需要求解$M+N$个线性回归问题。

### (2)Stochastic Gradient Descent
求解上述优化问题也可以采用随机梯度下降的方法，此时这种方法也叫**协同过滤（collaborative filtering）**。

每一轮训练使用一个样本$(x_n,y_n=(r_{n1},r_{n2},...,r_{nM}))$，则损失函数为：

$$ Loss(w_m,v_n) = (r_{nm}-w_m^Tv_n)^2 + λ\mid\mid w_m \mid\mid^2 + λ\mid\mid v_n \mid\mid^2 $$

分别对$w_m$和$v_n$求梯度得：

$$ \begin{cases} \frac{\partial Loss(w_m,v_n)}{\partial w_m} = -2(r_{nm}-w_m^Tv_n)v_n + 2λw_m \\ \frac{\partial Loss(w_m,v_n)}{\partial v_n} = -2(r_{nm}-w_m^Tv_n)w_m + 2λv_n \end{cases} $$

进行参数更新：

$$ \begin{cases} w_m = w_m -α( -2(r_{nm}-w_m^Tv_n)v_n+2λw_m) \\ v_n = v_n -α(-2(r_{nm}-w_m^Tv_n)w_m+2λv_n) \end{cases} $$

在实际应用中，由于该算法简单高效，大多采用这种算法。

需要注意的是，参数初始化的时候，如果$W$和$V$设置为全0，则梯度始终为0，无法正常更新。

由上述可以看出，**推荐系统**的主要任务是对用户和电影进行**特征提取（feature extraction）**。
