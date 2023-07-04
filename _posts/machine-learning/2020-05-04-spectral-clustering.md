---
layout: post
title: '谱聚类(Spectral Clustering)'
date: 2020-05-04
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64a296311ddac507ccf3ba21.jpg'
tags: 机器学习
---

> Spectral Clustering.

**谱聚类（spectral clustering）**是一种基于图的聚类方法，其主要思想是把所有的数据看做空间中的点，这些点之间用带权重的边连接起来；距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高；通过对所有数据点组成的图进行切图，让切图后不同的子图间边权重之和尽可能低，而子图内的边权重之和尽可能高，从而达到聚类的目的。

谱聚类的优点：
1. 谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效；
2. 由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。

谱聚类的缺点：
1. 如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。
2. 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。

实现谱聚类的关键点是图邻接矩阵的生成方式与切图的方式。

## 1. 构建邻接矩阵

对于一个图$G(V,E)$，一般用点的集合$V$和边的集合$E$来描述。其中$V$是数据集里面所有的点$(v_1,v_2,...,v_n)$。对于$V$中的任意两个点，可以有边连接，也可以没有边连接。定义权重$w_{ij}$为点$x_i$和点$x_j$之间的权重。对于无向图$w_{ij}=w_{ji}$。

利用所有点之间的权重值，可以得到图的邻接矩阵$W$。$W$是一个$n\times n$的矩阵，第$i$行的第$j$个值对应权重$w_{ij}$。

在谱聚类中，距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高。构建邻接矩阵$W$的方法有三类。

### ⚪ $\epsilon$-邻近法

设置一个距离阈值$\epsilon$，然后用欧式距离$s_{ij}$度量任意两点$x_i$和点$x_j$的距离$s_{ij}=\|\|x_i-x_j\|\|_2^2$，根据$s_{ij}$和$\epsilon$的大小关系定义邻接矩阵：

$$
w_{ij} = \begin{cases}
0, & s_{ij} > \epsilon \\
\epsilon, & s_{ij} \leq \epsilon
\end{cases}
$$

两点间的权重要不就是$\epsilon$，要不就是$0$，没有其他的信息。距离远近度量很不精确，因此在实际应用中很少使用$\epsilon$-邻近法。

### ⚪ K邻近法

利用[**KNN**算法](https://0809zheng.github.io/2020/03/23/knn.html)遍历所有的样本点，取每个样本最近的$k$个点作为近邻，只有和样本距离最近的$k$个点之间的$w_{ij}>0$。但是这种方法会造成重构之后的邻接矩阵非对称。如果后面的算法需要对称邻接矩阵，可以采取以下措施。

​第一种**K**邻近法是只要一个点在另一个点的**K**近邻中，则保留：

$$
w_{i j}=w_{j i}=\begin{cases}
0, & x_i \notin K N N\left(x_j\right) \operatorname{and} x_j \notin K N N\left(x_i\right) \\
\exp \left(-\frac{\left\|x_i-x_j\right\|_2^2}{2 \sigma^2}\right), & x_i \in K N N\left(x_j\right) \text{ or } x_j \in K N N\left(x_i\right)
\end{cases}
$$

第二种**K**邻近法是必须两个点互为**K**近邻中，才能保留：


$$
w_{i j}=w_{j i}=\begin{cases}
0, & x_i \notin K N N\left(x_j\right) \operatorname{or} x_j \notin K N N\left(x_i\right) \\
\exp \left(-\frac{\left\|x_i-x_j\right\|_2^2}{2 \sigma^2}\right), & x_i \in K N N\left(x_j\right) \text{ and } x_j \in K N N\left(x_i\right)
\end{cases}
$$

### ⚪ 全连接法

全连接法设置所有的点之间的权重值都大于**0**，可以选择不同的核函数来定义边权重，常用的有多项式核函数、高斯核函数和**Sigmoid**核函数。最常用的是高斯径向核**RBF**函数：
​ 
$$
w_{ij} = w_{ji} = \exp \left(-\frac{\left\|x_i-x_j\right\|_2^2}{2 \sigma^2}\right)
$$

## 2. 无向图切图

无向图$G(V,E)$的切图是指把图切成互相没有连接的$K$个子图，每个子图的集合为$A_1,...,A_K$，它们满足$A_i ∩ A_j = \Phi,A_1∪...∪A_K = V$。

对于任意两个子图点的集合$A,B \subset V, A \cap B = \Phi$，定义$A$和$B$之间的切图权重为：

$$
W(A,B) = \sum_{i\in A,j\in B}w_{ij}
$$

则对于$K$个子图点的集合$A_1,...,A_K$，定义切图**cut**：

$$
cut(A_1,...,A_K) = \frac{1}{2}\sum_{k=1}^K W(A_k,\overline{A}_k)
$$

其中$$\overline{A}_k$$为$A_k$的补集，即除$A_k$子集外其他$V$的子集的并集。

直接最小化切图**cut**会造成分割出很多单个离散的样本点作为一类，分割的类别不均匀。在谱聚类中，为了避免最小切图导致的切图效果不佳，需要对每个子图的规模做出限定，进而有两种切图方式，一种是**RatioCut**，另一种是**Ncut**。

### ⚪ RatioCut

在**RatioCut**中，对每个切图，不光考虑最小化$cut(A_1,...,A_K)$，还同时考虑最大化每个子图点的个数，即：

$$
RatioCut(A_1,...,A_K) = \frac{1}{2}\sum_{k=1}^K \frac{W(A_k,\overline{A}_k)}{|A_k|}
$$

为最小化**RatioCut**，引入图的度矩阵与拉普拉斯矩阵。

对于图中的任意一个点$x_i$，它的度定义为和它相连的所有边的权重之和，即$d_i=\sum_{j}w_{ij}$。利用每个点度的定义，可以得到一个度矩阵$D$，它是一个对角矩阵，只有主对角线有值，第$i$行对应第$i$个点的度数。拉普拉斯矩阵$L$定义为度矩阵$D$与邻接矩阵$W$的差$L=D-W$。

引入指示向量$h_{i}=(h_{i1},...,h_{ik})$指示任意点$x_i$是否被划分到点集$A_k$：

$$
h_{ik} = \begin{cases}
0, & x_i \not\in A_k \\ 
\frac{1}{\sqrt{|A_k|}}, & x_i \in A_k
\end{cases}
$$

则**RatioCut**可以被化简为：

$$
\begin{aligned}
RC(A_1,...,A_K) &= \frac{1}{2}\sum_{k=1}^K \frac{W(A_k,\overline{A}_k)}{|A_k|} \\
&\propto \frac{1}{2}\sum_{k=1}^K \left[ \sum_{i \in A_k,j \not\in A_k}w_{ij}\frac{1}{|A_k|}+\sum_{i \not\in A_k,j \in A_k}w_{ij}\frac{1}{|A_k|} \right] \\
&= \frac{1}{2}\sum_{k=1}^K \left[ \sum_{i \in A_k,j \not\in A_k}w_{ij}\left(\frac{1}{\sqrt{|A_k|}}-0\right)^2+\sum_{i \not\in A_k,j \in A_k}w_{ij}\left(0-\frac{1}{\sqrt{|A_k|}}\right)^2 \right] \\
&= \frac{1}{2}\sum_{k=1}^K \sum_{i,j}w_{ij}\left(h_{ik}-h_{jk}\right)^2 \\
&= \frac{1}{2}\sum_{k=1}^K \left(\sum_{i,j}w_{ij}h_{ik}^2-\sum_{i,j}2w_{ij}h_{ik}h_{jk}+\sum_{i,j}w_{ij}h_{jk}^2\right) \\
&= \frac{1}{2}\sum_{k=1}^K \left(2\sum_{i,j}w_{ij}h_{ik}^2-\sum_{i,j}2w_{ij}h_{ik}h_{jk}\right) \\
&= \sum_{k=1}^K \left(\sum_{i,j}w_{ij}h_{ik}^2-\sum_{i,j}w_{ij}h_{ik}h_{jk}\right) \\
&= \sum_{k=1}^K \left(\sum_{i}(\sum_{j}w_{ij})h_{ik}^2-\sum_{i,j}w_{ij}h_{ik}h_{jk}\right) \\
&= \sum_{k=1}^K \left(h_k^TDh_k - h_k^TWh_k\right) = \sum_{k=1}^K h_k^TLh_k \\
& = H^TLH
\end{aligned}
$$

最小化**RatioCut**等价于最小化$H^TLH$。根据[瑞利商](https://0809zheng.github.io/2021/06/22/rayleigh.html)的定义，使得上式最小化的$H \in R^{N\times K}$是由$L$最小的$K$个特征值对应的特征向量构成的。一般需要对$H$矩阵按行做标准化，即：

$$
h_{ij} \leftarrow \frac{h_{ij}}{(\sum_j h_{ij}^2)^{1/2}}
$$

由于在使用维度规约的时候损失了少量信息，导致得到的优化后的指示向量$h$对应的$H$现在不能完全指示各样本的归属，因此一般在得到$N\times K$维度的矩阵$H$（看作$N$个$K$维样本）后还需要进行一次传统的聚类，比如使用**K-Means**聚类。

### ⚪ NCut

在**NCut**中，对每个切图，不光考虑最小化$cut(A_1,...,A_K)$，还同时考虑最大化每个子图内所有点的度之和$vol(A)=\sum_{i \in A}d_i$，即：

$$
NCut(A_1,...,A_K) = \frac{1}{2}\sum_{k=1}^K \frac{W(A_k,\overline{A}_k)}{vol(A_k)}
$$

引入指示向量$h_{i}=(h_{i1},...,h_{ik})$指示任意点$x_i$是否被划分到点集$A_k$：

$$
h_{ik} = \begin{cases}
0, & x_i \not\in A_k \\ 
\frac{1}{\sqrt{vol(A_k)}}, & x_i \in A_k
\end{cases}
$$

与**RatioCut**的推导过程类似，可以得到：

$$
\begin{aligned}
NCut(A_1,...,A_K) &= \sum_{k=1}^K h_k^TLh_k  = H^TLH
\end{aligned}
$$

由于此时$H^TH \neq I$，因此上式并不满足瑞利商的形式。注意到：

$$
h_i^TDh_i = \sum_{j=1}^nh_{ij}^2d_j = \frac{1}{vol(A_i)}\sum_{j \in A_i} d_j = \frac{1}{vol(A_i)}vol(A_i) =1
$$

因此有$H^TDH=I$。不妨令$H=D^{-1/2}F$，则有：

$$
\begin{aligned}
NCut(A_1,...,A_K) & = H^TLH = F^TD^{-1/2}LD^{-1/2}F
\end{aligned}
$$

上式满足瑞利商的形式（$F^TF=I$）。使得上式最小化的$F$是由$D^{-1/2}LD^{-1/2}$最小的$K$个特征值对应的特征向量构成的。注意到$D^{-1/2}LD^{-1/2}$相当于对拉普拉斯矩阵做了一次标准化：

$$
L_{ij} \leftarrow \frac{L_{ij}}{\sqrt{d_id_j}}
$$

## 3. 实现谱聚类

谱聚类主要的注意点为相似矩阵的生成方式、切图的方式以及最后的聚类方法。最常用的相似矩阵的生成方式是基于高斯核距离的全连接方式，最常用的切图方式是**Ncut**，最常用的聚类方法为**K-Means**。

1. 根据相似矩阵的生成方式构建样本的相似矩阵$S$。
2. 根据相似矩阵$S$构建邻接矩阵$W$，构建度矩阵$D$。
3. 计算出拉普拉斯矩阵$L=D-W$。
4. 构建标准化后的拉普拉斯矩阵$D^{-1/2}LD^{-1/2}$。
5. 计算$D^{-1/2}LD^{-1/2}$最小的$k_1$个特征值所各自对应的特征向量$f$。
6. 将各自对应的特征向量$f$组成的矩阵按行标准化，最终组成$n\times k_1$维的特征矩阵$F$。
7. 对$F$中的每一行作为一个$k_1$维样本，共$n$个样本，用指定聚类方法进行聚类，聚类维数为$k_2$。
8. 得到簇划分$C(c_1,...,c_{k_2})$。

### ⚪ from scratch

```python
def calculate_w_ij(a,b,sigma=1):
    w_ab = np.exp(-np.sum((a-b)**2)/(2*sigma**2))
    return w_ab

# 计算邻接矩阵
def Construct_Matrix_W(data,k=5):
    rows = len(data) # 取出数据行数
    W = np.zeros((rows,rows)) # 对矩阵进行初始化：初始化W为rows*rows的方阵
    for i in range(rows): # 遍历行
        for j in range(rows): # 遍历列
            if(i!=j): # 计算不重复点的距离
                W[i][j] = calculate_w_ij(data[i],data[j]) # 调用函数计算距离
        t = np.argsort(W[i,:]) # 对W中进行行排序，并提取对应索引
        for x in range(rows-k): # 只保留W每行前k大的元素
            W[i][t[x]] = 0
    W = (W+W.T)/2 # 处理可能存在的复数的虚部，都变为实数
    return W

# 计算标准化的拉普拉斯矩阵
def Calculate_Matrix_L_sym(W):
    degreeMatrix = np.sum(W, axis=1) # 按照行对W矩阵进行求和
    L = np.diag(degreeMatrix) - W # 计算对应的对角矩阵减去w
    # 拉普拉斯矩阵标准化，就是选择Ncut切图
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5))) # D^(-1/2)
    L_sym = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix) # D^(-1/2)LD^(-1/2)
    return L_sym

# 归一化
def normalization(matrix):
    sum = np.sqrt(np.sum(matrix**2,axis=1,keepdims=True)) # 求数组的正平方根
    nor_matrix = matrix/sum # 求平均
    return nor_matrix

W = Construct_Matrix_W(your_data) # 计算邻接矩阵
L_sym = Calculate_Matrix_L_sym(W) # 依据W计算标准化拉普拉斯矩阵
lam, H = np.linalg.eig(L_sym) # 特征值分解
    
t = np.argsort(lam) # 将lam中的元素进行排序，返回排序后的下标
H = np.c_[H[:,t[0]],H[:,t[1]]] # 0和1类的两个矩阵按行连接，就是把两矩阵左右相加，要求行数相等。
H = normalization(H) # 归一化处理

from sklearn.cluster import KMeans
model = KMeans(n_clusters=20) # 新建20簇的Kmeans模型
model.fit(H) # 训练
labels = model.labels_ # 得到聚类后的每组数据对应的标签类型
res = np.c_[your_data,labels] # 按照行数连接data和labels
```

### ⚪ via sklearn

```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=k)
y_pred = sc.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred)
```