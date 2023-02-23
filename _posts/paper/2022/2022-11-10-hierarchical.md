---
layout: post
title: 'Deep Metric Learning with Hierarchical Triplet Loss'
date: 2022-11-10
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63ca56d8be43e0d30ea92974.jpg'
tags: 论文阅读
---

> 通过层次化三元组损失实现深度度量学习.

- paper：[Deep Metric Learning with Hierarchical Triplet Loss](https://arxiv.org/abs/1810.06951)

在深度度量学习中，[<font color=blue>Triplet Loss</font>](https://0809zheng.github.io/2022/11/02/triplet.html)为每一个样本$x$选择一个正样本$x^+$和一个负样本$x^-$，同时最小化正样本对之间的距离和最大化负样本对之间的距离。

$$ \max(0, D[f_{\theta}(x),f_{\theta}(x^+)] -D[f_{\theta}(x),f_{\theta}(x^-)] + \epsilon) $$

给定$N$个训练样本，则可构造$O(N^3)$的三元组，在训练期间遍历所有这些训练元组是不可行的，这样会产生大量冗余信息，并且多数训练数据不具有效训练信息，使网络无法得到很好的训练。

本文提出了**Hierarchical Triplet Loss**，能够通过自适应学习编码数据集的所有类别样本构建层级类结构，并生成样本的**分层树(hierarchical tree)**；并基于分层树进行**Anchor neighbor**采样，再由动态变化的**violate margin**机制构建损失函数。

给定训练集$(X,Y)$和使用传统**triplet loss**预训练的神经网络$f_{\theta}$，计算类间距离矩阵，其中第$p$类和第$q$类之间的距离计算为：

$$ d(p,q) = \frac{1}{N_pN_q} \sum_{i\in p,j \in q} ||f_{\theta}(x_i)-f_{\theta}(x_j)||^2 $$

根据计算的类间距离创建分层树，叶节点是原始类别，然后基于计算的距离矩阵递归地合并不同层级的叶节点来创建层次结构。 树的深度设置$L$级，平均类内距离$d_0$作为合并第$0$级节点的阈值:

$$ d_0 = \frac{1}{C} \sum_{c=1}^C (\frac{1}{n_c^2-n_c}\sum_{i\in c,j \in c} ||f_{\theta}(x_i)-f_{\theta}(x_j)||^2) $$

合并第$l$级节点的阈值设置为:

$$ d_l = \frac{l(4-d_0)}{L}+d_0 $$

距离小于$d_l$的两个类合并到第$l$级的节点中。节点从第$0$级合并到第$L$级，最后生成一个分层树，它从原始类别的叶节点开始到顶部节点，可以捕获整个数据集上的类别关系，并在训练中不断更新。

![](https://pic.imgdb.cn/item/63ca599cbe43e0d30eadec68.jpg)

从构造的分层树中进行**A-N采样(Anchor neighbor sampling)**，作为训练过程中的批量数据。从第$0$级随机选择$l'$个节点，每个节点代表一个原始类别。然后为每个类选择$m-1$个在$0$级最接近的类，以通过在视觉相似的类中学习最具区分性的特征。再在每个类中随机选取$t$个样本，因此批量数据中共有$l'mt$个样本，可构造$A_{l'm}^2A_t^2C_t^1$个三元组，$A_{l'm}^2$表示随机抽取两个类（正类和负类）；$A_t^2$表示从正类中选择两个样本；$C_t^1$表示从负类中随机选择负样本。

![](https://pic.imgdb.cn/item/63ca5e5bbe43e0d30eb6676b.jpg)

在此基础上作者把**Triplet Loss**中的**margin**参数$\epsilon$调整为一个动态参数$\alpha_z$，它是根据构造的分层树上类间关系计算得到的:

$$ \alpha_z = \beta + d_{H(y_a,y_n)} - S_{y_a} $$

其中$\beta=0.1$是一个常数，它鼓励每一次迭代类间距更远；$d_{H(y_a,y_n)}$是把类$y_a,y_n$合并后的节点的阈值；$S_{y_a}$是类$y_a$样本间的平均距离：

$$ S_{y_a} = \frac{1}{n_{y_a}^2-n_{y_a}}\sum_{i,j \in {y_a}} ||f_{\theta}(x_i)-f_{\theta}(x_j)||^2 $$