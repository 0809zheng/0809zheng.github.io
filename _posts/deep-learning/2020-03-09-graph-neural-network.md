---
layout: post
title: '图神经网络'
date: 2020-03-09
author: 郑之杰
cover: ''
tags: 深度学习
---

> Graph Neural Networks.

**本文目录：**
1. Spatial-based GNN
2. Spectral-based GNN
3. Benchmarks

# 1. Spatial-based GNN
**术语**Terminology：
- **Aggregate**:用$neighbor$的$feature$更新$node$的$hidden$ $state$
- **Readout**:把所有$node$的$feature$集合起来代表整个$graph$

![](https://pic.downk.cc/item/5ea590d8c2a9a83be519e2c6.jpg)

**方法**:
1. NN4G (Neural Networks for Graph)
2. DCNN (Diffusion-Convolution Neural Network)
3. DGC (Diffusion Graph Convolution)
4. MoNET (Mixture Model Networks)
5. GraphSAGE
6. GAT (Graph Attention Networks)
7. GIN (Graph Isomorphism Network)

### (1)NN4G (Neural Networks for Graph)
- [paper](https://ieeexplore.ieee.org/document/4773279)

先对输入图的每个节点进行**Embedding**，得到$hidden$ $layer$ $0$：

（以节点$v3$为例）：$$h_3^0 = \overline{w}_0x_3$$

![](https://pic.downk.cc/item/5ea59bbdc2a9a83be5281d20.jpg)

状态更新时，每个节点使用邻节点的状态和该节点的输入更新：

![](https://pic.downk.cc/item/5ea59c77c2a9a83be529444d.jpg)

$Readout$时对每一层的隐状态的均值进行加权求和，得到输出：

![](https://pic.downk.cc/item/5ea59cbcc2a9a83be529a8a7.jpg)


### (2)DCNN (Diffusion-Convolution Neural Network )
- [paper](https://arxiv.org/abs/1511.02136)

**DCNN**在第$l$个隐藏层，使用距离节点的距离为$l$的节点状态进行更新。

如下图，在更新$h_3^1$时，使用距离为2的$h_1^0$和$h_3^0$进行更新：

![](https://pic.downk.cc/item/5ea59dc1c2a9a83be52b2e2f.jpg)

$Readout$时把所有层的状态连接起来进行线性变换：

![](https://pic.downk.cc/item/5ea59e9dc2a9a83be52c63d5.jpg)


### (3)DGC (Diffusion Graph Convolution)
- [paper](https://arxiv.org/pdf/1707.01926.pdf)

**DGC**与**DCNN**的不同在于$Readout$时把所有层的状态相加：

![](https://pic.downk.cc/item/5ea59ed8c2a9a83be52cba90.jpg)

### (4)MoNET (Mixture Model Networks)
- [paper](https://arxiv.org/pdf/1611.08402.pdf)

定义两节点$x$、$y$之间的“距离”$u$：

$$ u(x,y) = (\frac{1}{\sqrt{deg(x)}},\frac{1}{\sqrt{deg(y)}})^T $$

其中$deg(x)$表示$x$的维度。

节点更新时采用加权求和的方法：

$$ h_3^1 = w(\hat{u}_{3,0})×h_0^0 + w(\hat{u}_{3,2})×h_2^0 + w(\hat{u}_{3,4})×h_4^0 $$

其中$w$是一个神经网络，$\hat{u}$是$u$的变换。

![](https://pic.downk.cc/item/5ea5a04bc2a9a83be52eba44.jpg)

### (5)GraphSAGE
- [paper](https://arxiv.org/pdf/1706.02216.pdf)

两个操作：**Sample** and **aggregate**

![](https://pic.downk.cc/item/5ea5a0cbc2a9a83be52f6eea.jpg)

### (6)GAT (Graph Attention Networks)
- [paper](https://arxiv.org/pdf/1710.10903.pdf)

用一个函数$f$实现$attention$机制，用$attention$更新参数：

![](https://pic.downk.cc/item/5ea5a135c2a9a83be53009c2.jpg)

### (7)GIN (Graph Isomorphism Network)
- [paper](https://openreview.net/forum?id=ryGs6iA5Km)

节点的状态更新：

$$ h_v^{(k)} = MLP^{(k)}((1+ε^{(k)})·h_v^{(k-1)} + \sum_{u \in \Bbb{N}(v)}^{} {h_u^{(k-1)}}) $$

使用$MLP$代替了单层网络，$ε$是可学习参数，$\Bbb{N}$是邻节点集合。

该论文证明了节点的状态更新应该用**求和sum**而不是**均值mean**或**最大值max**，因为均值或最大值可能会失效：

![](https://pic.downk.cc/item/5ea5a1acc2a9a83be530b5a3.jpg)


# 2. Spectral-based GNN
**Spectral-based**的思想是将图网络和卷积核通过**Fourier变换**到**spectral domain**，相乘后把结果通过**Inverse Fourier变换**到**vertex domain**。

![](https://pic.downk.cc/item/5ea6c0d3c2a9a83be5a1ce61.jpg)

### Fourier Transform
**信号signal**既可以在**时域Time domain**中表示，又可以在**频域Frequency domain**中表示：

![](https://pic.downk.cc/item/5ea6c285c2a9a83be5a44b65.jpg)

![](https://pic.downk.cc/item/5ea6c2e9c2a9a83be5a4c43f.jpg)

### Spectral Graph Theory
**术语**Terminology：
- **Graph**:$$G=(V,E)$$,$N= \mid V \mid$，本文讨论**无向图undirected graph**。
- **Adjacency matrix(weight matrix)**:$$A \in \Bbb{R}^{N×N}$$表示节点间是否有连接，是**对称矩阵**。
- **Degree matrix**:$$D \in \Bbb{R}^{N×N}$$表示节点的邻节点数量，是**对角矩阵**。
- **Signal on graph(vertices)**:$$f:V → \Bbb{R}$$,$f(i)$表示每个节点$i$的信号。
- **Graph Laplacian**:$$L=D-A$$是对称的半正定矩阵。

**谱分解Spectral Decomposition**：$$L=U \Lambda U^T$$
- 对角矩阵$$\Lambda = diag(λ_0,...,λ_{N-1}) \in \Bbb{R}^{N×N}$$，$λ_i$表示**频率frequency**。
- 正交矩阵$$U = [u_0,...,u_{N-1}] \in \Bbb{R}^{N×N}$$，$u_i$表示**基basis**。

![](https://pic.downk.cc/item/5ea6c643c2a9a83be5a8085c.jpg)

上述概念的例题：

![](https://pic.downk.cc/item/5ea6c6fec2a9a83be5a89035.jpg)

$u_i$表示各节点中频率$λ_i$所占的权重；频率越大，相邻两节点之间的信号变化量越大：

![](https://pic.downk.cc/item/5ea6ce22c2a9a83be5ae6f12.jpg)

信号$x$的**Graph Fourier Transform**：$$\hat{x} = U^Tx$$

![](https://pic.downk.cc/item/5ea6cf12c2a9a83be5af2e47.jpg)

信号$\hat{x}$的**Inverse Graph Fourier Transform**：$$x = U\hat{x}$$

![](https://pic.downk.cc/item/5ea6cfafc2a9a83be5afb05d.jpg)

### Spectral-based Method
1. 将信号$x$通过$FT$转换到$spectral$ $domain$：$$\hat{x} = U^Tx$$;
2. 在$spectral$ $domain$设计滤波器$$g_θ(\Lambda)$$;
3. $vertex$ $domain$的卷积相当于$spectral$ $domain$的乘积：$$\hat{y}=g_θ(\Lambda)\hat{x}$$;
4. 将信号$\hat{y}$通过$IFT$转换到$vertex$ $domain$：$$y = U\hat{y}$$

计算：

$$y = U\hat{y} = Ug_θ(\Lambda)\hat{x} = Ug_θ(\Lambda)U^Tx = g_θ(U\Lambda U^T)x = g_θ(L)x$$

![](https://pic.downk.cc/item/5ea6d0c7c2a9a83be5b0df38.jpg)

$$g_θ(L)$$可以是任何函数：
1. $$g_θ(L)=log(I+L)=L-\frac{L^2}{2}+\frac{L^3}{3}...$$
- 问题：学习复杂度$O(N)$
2. $$g_θ(L)=cos(L)=I-\frac{L^2}{2!}+\frac{L^4}{4!}...$$
- 问题：失去局部性

**引理Lemma**：如果一个图具有$N$个节点，则$L^N$不存在$0$元素，即使所有节点共享。

**方法**:
1. ChebNet
2. GCN(Graph Convolution Network)
3. DropEdge

### (1)ChebNet
- [paper](https://arxiv.org/pdf/1606.09375.pdf)

$$g_θ(L)$$使用$k$阶多项式函数函数：$$g_θ(L) = \sum_{k=0}^{K} {θ_kL^k}$$

由引理知上式是**K-localized**的，上式时间复杂度$O(N^2)$。

引入**Chebyshev polynomial**：

![](https://pic.downk.cc/item/5ea6d6a2c2a9a83be5b742da.jpg)

为使$$λ \in [-1,1]$$，做变换：$$\overline{\Lambda}=\frac{2\Lambda}{λ_{max}}-I$$

则$$T_0(\overline{\Lambda})=I, \quad T_1(\overline{\Lambda})=\overline{\Lambda}, \quad T_k(\overline{\Lambda})=2\overline{\Lambda}T_{k-1}(\overline{\Lambda})-T_{k-2}(\overline{\Lambda}) $$

$$ g_θ(\hat{L}) = \sum_{k=0}^{K} {θ'_kT_k(\hat{L})} $$

$$ y = g_θ(\hat{L})x = \sum_{k=0}^{K} {θ'_kT_k(\hat{L})}x $$

若记$$\hat{x}_k = T_k(\hat{L})x$$,则：

$$ y = \sum_{k=0}^{K} {θ'_k\hat{x}_k} = [\hat{x}_0;...;\hat{x}_K][θ'_0;...;θ'_K] $$

上式时间复杂度$O(KE)$。

实际使用中，可以使用多个$$g_θ$$：

![](https://pic.downk.cc/item/5ea6da28c2a9a83be5baef92.jpg)

### (2)GCN(Graph Convolution Network)
- [paper](https://openreview.net/pdf?id=SJU4ayYgl)
![](https://pic.downk.cc/item/5ea6daefc2a9a83be5bb9f38.jpg)

$GCN$的计算公式也写作：

$$ h_v = f(\frac{1}{\mid N(v) \mid}\sum_{u \in N(v)}^{} {Wx_u}+b, \quad \forall v \in V) $$

### (3)DropEdge
- [paper](https://openreview.net/pdf?id=Hkx1qkrKPr)

随即丢弃**Adjacency Matrix**的一些元素，防止**over-smoothing**。

# 3. Benchmarks

### (1)Graph Classification
- **Dataset**:SuperPixel MNIST and CIFAR10
![](https://pic.downk.cc/item/5ea6bd91c2a9a83be59cf7f5.jpg)
- **Result**:
![](https://pic.downk.cc/item/5ea6bdb4c2a9a83be59d26ab.jpg)

### (2)Regression
- **Dataset**:ZINC molecule graphs dataset
![](https://pic.downk.cc/item/5ea6bdf2c2a9a83be59d74e4.jpg)
- **Result**:
![](https://pic.downk.cc/item/5ea6be15c2a9a83be59d9c05.jpg)

### (3)Node classification
- **Dataset**:Stochastic Block Model dataset

graph pattern recognition and semi-supervised graph clustering
![](https://pic.downk.cc/item/5ea6be68c2a9a83be59dfd51.jpg)
- **Result**:
![](https://pic.downk.cc/item/5ea6be85c2a9a83be59e1f33.jpg)

### (4)Edge classification
- **Dataset**:Traveling Salesman Problem
![](https://pic.downk.cc/item/5ea6beaac2a9a83be59e5364.jpg)
- **Result**:
![](https://pic.downk.cc/item/5ea6becbc2a9a83be59ea853.jpg)
![](https://pic.downk.cc/item/5ea6bee2c2a9a83be59ef91e.jpg)
