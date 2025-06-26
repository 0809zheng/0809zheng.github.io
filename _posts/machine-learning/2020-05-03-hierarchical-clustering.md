---
layout: post
title: '层次聚类(Hierarchical Clustering)'
date: 2020-05-03
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eb376e3c2a9a83be5dfbdb8.jpg'
tags: 机器学习
---

> Hierarchical Clustering.

**层次聚类 (Hierarchical Clustering)** 是把数据对象组成一棵聚类树，对数据集进行层次的分解，后面一层生成的簇基于前面一层的结果。根据分解的顺序是自底向上（合并）还是自顶向下（分裂），层次聚类可以分为**凝聚 (agglomerate)**和**分裂 (divisive)**。
- 凝聚聚类：又称自底向上（**bottom-up**）的层次聚类，每一个数据点最开始都是一个簇，每次按一定的准则将最相近的两个簇合并生成一个新的簇，如此往复，直至最终所有数据都属于一个簇。
- 分裂聚类： 又称自顶向下（**top-down**）的层次聚类，最开始所有数据均属于一个簇，每次按一定的准则将某个簇划分为多个簇，如此往复，直至每个数据点均是一个簇。

层次聚类算法是一种贪心算法（**greedy algorithm**），因其每一次合并或划分都是基于某种局部最优的选择。层次聚类的局限在于，一旦某次合并或分裂被执行，就不能修改。

层次聚类算法的优点：（1）距离和规则的相似度容易定义，限制少；（2）不需要预先制定聚类数；（3）可以发现类的层次关系；（4）可以聚类成其它形状。

层次聚类算法的缺点：（1）计算复杂度太高；（2）奇异值也能产生很大影响；（3）算法很可能聚类成链状。

## 1. AGNES算法与Divisive算法

**Agglomerative Nesting (AGNES)**是一种**凝聚**的层次聚类算法，步骤如下：
1. 初始化，每个样本当做一个簇；
2. 计算任意两簇距离，找出距离最近的两个簇，合并这两簇；
3. 重复步骤2，直到最远两簇距离超过阈值，或者簇的个数达到指定值，终止算法。

**Divisive Analysis (DIANA)**是一种**分裂**的层次聚类算法，步骤如下：
1. 初始化，所有样本集中归为一个簇；
2. 在同一个簇中，计算任意两个样本之间的距离，找到距离最远的两个样本点$a,b$，将$a,b$作为两个簇的中心；
3. 计算原来簇中剩余样本点距离 $a,b$ 的距离，距离哪个中心近，分配到哪个簇中；
4. 重复步骤2、3，直到最远两簇距离不足阈值，或者簇的个数达到指定值，终止算法。

![](https://pic.downk.cc/item/5eb376e3c2a9a83be5dfbdb8.jpg)

在**AGNES**算法中，需要计算簇间距离。假设两个簇$C_i$和$C_j$分别有$n_i$和$n_j$个数据，其质心为$m_i$和$m_j$，一些常用的**簇间距离**度量方法：
1. 最小距离：两个类中距离最近的两个样本的距离；$$d_{min}(C_i,C_j)=min_{(p \in C_i,p' \in C_j)}\mid p-p' \mid$$
2. 最大距离：两个类中距离最远的两个样本的距离；$$d_{max}(C_i,C_j)=max_{(p \in C_i,p' \in C_j)}\mid p-p' \mid$$![](https://pic1.imgdb.cn/item/685cb2f658cb8da5c871a075.png)
3. 平均距离：两个簇的平均值作为中心点之间的距离；$$d_{avg}(C_i,C_j)=\frac{1}{n_in_j} \sum_{p \in C_i}^{} {\sum_{p' \in C_j}^{} {\mid p-p' \mid}}$$
4. （类）均值距离：两个簇任意两点距离加总后，取平均值；$$d_{mean}(C_i,C_j)= \mid m_i-m_j \mid$$![](https://pic1.imgdb.cn/item/685cb34158cb8da5c871a145.png)
5. 中间距离：介于最短距离和最长距离之间，相当于三角形的中线；$$d_{mid}(C_i,C_j)= \sqrt{\frac{1}{2}d_{lp}^2+\frac{1}{2}d_{lq}^2-\frac{1}{4}d_{pq}^2}$$![](https://pic1.imgdb.cn/item/685cb3bd58cb8da5c871a2d2.png)
6. 重心距离：将每类中包含的样本数考虑进去。$$d_{gravity}(C_i,C_j)= \frac{n_i}{n_i+n_j}d_{i\cdot}^2+\frac{n_j}{n_i+n_j}d_{j\cdot}^2-\frac{n_in_j}{(n_i+n_j)^2}d_{ij}^2$$

最小和最大度量代表了簇间距离度量的两个极端，它们趋向对离群点或噪声数据过分敏感。当算法选择“最小距离”作为簇间距离时，称之为**最近邻聚类算法**。并且当最近两个簇之间的距离超过阈值时，算法终止，则称其为**单连接算法**；当算法选择“最大距离”作为簇间距离时，称之为**最远邻聚类算法**。并且当最近两个簇之间的最大距离超过阈值时，算法终止，则称其为**全连接算法**。

使用均值距离和平均距离是对最小和最大距离之间的一种折中方法，而且可以克服离群点敏感性问题。尽管均值距离计算简单，但是平均距离也有它的优势，因为它既能处理数值数据又能处理分类数据。

## 2. BIRCH算法

**Balanced Iterative Reducing and Clustering Using Hierarchies (BIRCH)** 使用**聚类特征树（CF-tree）**来进行快速聚类。算法的主要流程为：
1. 扫描数据库，建立一棵存放于内存的**CF-Tree**，它可以被看作数据的多层压缩，试图保留数据的内在聚类结构；
2. 采用某个选定的聚类算法（如**K-means**或者凝聚算法），对**CF**树的叶节点进行聚类，把稀疏的簇当作离群点删除，而把更稠密的簇合并为更大的簇。

### （1）构造CF-Tree

聚类特征**CF**用一个三元组$(N,LS,SS)$概括描述各簇的信息。
- $N$是簇中数据点的数量；
- $LS$是各点的线性求和；$$\sum_n x_n$$
- $SS$是各点的平方和。$$\sum_n \mid\mid x_n \mid\mid^2$$

**CF**具有**可加性**:

$$
CF_1 = (N_1,LS_1,SS_1), CF_2 = (N_2,LS_2,SS_2) \\
CF_1+CF_2 = (N_1+N_2,LS_1+LS_2,SS_1+SS_2)
$$

**CF**结构概括了簇的基本信息，并且是高度压缩的，它存储了小于实际数据点的聚类信息。通过**CF**可以计算簇的信息：
- 簇质心：$$x_0 = \frac{LS}{N}$$
- 簇半径：簇中点到质心的平均距离；$$R = \sqrt{\frac{\sum_n (x_n-x_0)^2}{N}} = \sqrt{\frac{NSS-LS^2}{N^2}}$$
- 簇直径：簇中两两数据点的平均距离；$$D = \sqrt{\frac{\sum_{i,j} (x_i-x_j)^2}{N(N-1)}} = \sqrt{\frac{2NSS-2LS^2}{N(N-1)}}$$

通过**CF**还可以计算簇间距离：
- 中心点欧基里得距离（**centroid Euclidian distance**）：$$\sqrt{(\frac{LS_1}{N_1}-\frac{LS_2}{N_2})^2}$$
- 中心点曼哈顿距离（**centroid Manhattan distance**）：$$\mid \frac{LS_1}{N_1}-\frac{LS_2}{N_2}\mid$$
- 簇连通平均距离（**average inter-cluster distance**）：$$\sqrt{\frac{SS_1}{N_1}-2\frac{LS_1^\top}{N_1}\frac{LS_2}{N_2}+\frac{SS_2}{N_2}}$$
- 全连通平均距离（**average intra-cluster distance**）：$$\sqrt{\frac{2(N_1+N_2)(SS_1+SS_2)-2(LS_1+LS_2)^2}{(N_1+N_2)(N_1+N_2-1)}}$$
- 散布恶化距离（**variance increase distance**）：$$\sqrt{\frac{(N_1+N_2)(SS_1+SS_2)-(LS_1+LS_2)^2}{(N_1+N_2)^2}}-\frac{N_1SS_1-LS_1^2}{N_1^2}-\frac{N_2SS_2-LS_2^2}{N_2^2}$$



