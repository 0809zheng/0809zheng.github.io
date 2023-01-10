---
layout: post
title: 'Label Propagation for Deep Semi-supervised Learning'
date: 2022-09-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63bcd2b0be43e0d30ee1cf48.jpg'
tags: 论文阅读
---

> 深度无监督学习的标签传播.

- paper：[Label Propagation for Deep Semi-supervised Learning](https://arxiv.org/abs/1904.04717)

**标签传播(Label Propagation)**通过特征嵌入构造样本之间的相似图，然后把有标签样本的标签传播到无标签样本，传播权重正比于图中的相似度得分。

![](https://pic.imgdb.cn/item/63bcdb9abe43e0d30ef28825.jpg)

首先根据有标签数据集$(X_L,Y_L)$训练一个网络$f_{\theta}$；然后通过网络的特征提取部分$\phi_{\theta}$提取有标签样本$X_L$和无标签样本$X_U$的特征；通过特征构造相似度矩阵，并进一步构造无标签样本的伪标签$$\hat{Y}_U$$；最后对所有样本进行训练。

记样本的特征$V=(v_1,...,v_n)$，则相似度通过**k**近邻计算：

$$  a_{ij} =\begin{cases}  [v_i^Tv_j]_+^{\gamma}, & i \neq j \text{ and } v_i \in kNN(v_j) \\ 0, & \text{otherwise} \end{cases} $$

通过相似度计算相似度矩阵$W$：

$$ W = A + A^T $$

然后通过$$D=\text{diag}(W1_n)$$对相似度矩阵进行归一化：

$$ W = D^{-1/2}WD^{-1/2} $$

构造标签矩阵$Y \in \Bbb{R}^{n \times c}$，用于保存有标签数据的**one hot**标签：

$$  Y_{ij} =\begin{cases}  1, & i \in L \text{ and } y_i = j \\ 0, & \text{otherwise} \end{cases} $$

则可以构造变量$Z$的目标函数：一方面使得相似的样本具有相近的$z$值，另一方面使得有标签样本的标签$y$和$z$接近：

$$ J(Z) = \frac{\alpha}{2} \sum_{i,j=1}^n w_{ij} ||\frac{z_i}{\sqrt{d_{ii}}}-\frac{z_j}{\sqrt{d_{jj}}}||^2 + (1-\alpha) ||Y-Z||_F^2 $$

上式等价于线性方程组：

$$ (I-\alpha W)Z = Y $$

求解$Z$既可以通过矩阵求逆，也可以通过共轭梯度法：

$$ Z = (I-\alpha W)^{-1}Y $$

得到$Z$后，无标签数据的伪标签构造如下：

$$ \hat{y}_i = \mathop{\arg\max}_{j} z_{ij} $$

根据有标签数据集$(X_L,Y_L)$和构造的伪标签数据集$$(X_U,\hat{Y}_U)$$，可以构造优化目标。然而不同样本的伪标签的准确性不同，并且不同类别样本的伪标签是不平衡的，因此在优化目标中设置权重系数：

$$ L_w(X,Y_L,\hat{Y}_U;\theta) = \sum_{i=1}^l \zeta_{y_i} l_s(f_{\theta}(x_i),y_i) + \sum_{i=l+1}^n w_i\zeta_{\hat{y}_i} l_s(f_{\theta}(x_i),\hat{y}_i) $$

其中权重$$w_i$$用于衡量伪标签的不确定性：

$$ w_i = 1- \frac{H(\hat{z}_i)}{\log (c)} $$

样本$x_i$对应的$z_i$（归一化后）熵越大，权重$$w_i$$越小，对应伪标签的不确定性越大。

对于数据类别平衡性问题，引入权重$$\zeta_j$$，即属于第$j$个类别的有标签数据和无标签数据的数量之和的倒数：

$$ \zeta_j = (|L_j|+|U_j|)^{-1} $$

![](https://pic.imgdb.cn/item/63bce2d6be43e0d30efef58e.jpg)