---
layout: post
title: '基于密度的聚类(Density-Based Clustering)'
date: 2020-05-05
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/684ff58b58cb8da5c850891b.png'
tags: 机器学习
---

> Density-Based Spatial Clustering of Applications with Noise.

基于**密度**的聚类算法一般假定类别可以通过样本分布的紧密程度决定。同一类别的样本之间紧密相连，在该类别任意样本周围不远处一定有同类别的样本存在。

通过将紧密相连的样本划为一类，就得到了一个聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，就得到了最终的所有聚类类别结果。这类方法既适用于凸样本集，也适用于非凸样本集。

# 1. Mean-Shift算法

**Mean-Shift**是基于核密度估计的聚类算法，沿着密度上升的方向寻找属于同一个簇的数据点，不用预先指定簇的数目。

通用的**Mean-Shift**算法的过程如下：
1. 在未被标记的数据点中随机选择一个点作为中心点**center**；
2. 找出离**center**距离在**bandwidth**之内的所有点，记做集合$M$，认为这些点属于簇$c$;
3. 以**center**为中心点，计算从**center**开始到集合$M$中每个元素的向量，将这些向量加权求和，得到**shift**向量$\frac{1}{k} \sum_{x_i \in S_k}^{} {(x-x_i)}$;
4. **center = center + shift**。即**center**沿着**shift**的方向移动，移动距离是$$\mid\mid shift \mid\mid$$;
5. 重复步骤$$2,3,4$$直到**shift**向量的大小很小（接近收敛），记住此时的**center**。注意，这个迭代过程中遇到的点都应该归类到簇$c$;
6. 如果收敛时当前簇$c$的**center**与其它已经存在的簇$c_2$中心的距离小于阈值，那么把$c_2$和$c$合并。否则，把$c$作为新的聚类，增加$1$类;
7. 重复$$1,2,3,4,5$$直到所有的点都被标记访问。

![](https://pic.downk.cc/item/5eb3b9b0c2a9a83be5223456.jpg)

计算**shift**向量时可以引入核函数$K$，使得随着样本与被偏移点的距离不同，其偏移量对均值偏移向量的贡献也不同：

$$ \text{shift} = \frac{1}{k} \sum_{x_i \in S_k}^{} {K(x,x_i)} = \frac{\sum_{i=1}^{k} {x_ig(\mid\mid \frac{x-x_i}{h} \mid\mid^2)}}{\sum_{i=1}^{k} {g(\mid\mid \frac{x-x_i}{h} \mid\mid^2)}} - x $$

其中$h$是窗口的大小**bandwidth**，$$g(x) = -K'(x)$$。

使用**sklearn**库实现**MeanShift**算法：

```python
import numpy as np
from sklearn.cluster import MeanShift

# 生成随机数据
np.random.seed(42)
data = np.random.randn(6, 2) * 3 + 10

# 调用DBSCAN算法
clustering = MeanShift(bandwidth=2).fit(X)

clustering.labels_
# array([1, 1, 1, 0, 0, 0])
clustering.predict([[0, 0], [5, 5]])
# array([1, 0])
```

# 2. DBSCAN算法
- paper：[A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf)

**DBSCAN（Density-Based Spatial Clustering of Applications with Noise）**是一种基于密度的聚类算法。相比于传统的基于距离的聚类算法（如**K-Means**），**DBSCAN**不需要预先指定聚类数量，能够自动发现任意形状的聚类簇，并能够有效地处理噪声数据。

![](https://pic1.imgdb.cn/item/684ff58b58cb8da5c850891b.png)

**DBSCAN**算法的基本思想是将数据点分为三类：核心点、边界点和噪声点。
- **核心点(core-point)**是指在以该点为圆心，以设定的半径$ε$为半径的圆内，包含大于等于设定的最小样本数$M$的点；
- **边界点(border-point)**是指在以该点为圆心，以设定的半径$ε$为半径的圆内包含核心点的其余点；
- **噪声点(noise-point)**是指既不是核心点也不是边界点的点。

如下图所示，**A**为核心点，**B、C**是边界点，而**N**是噪声点。

![](https://pic1.imgdb.cn/item/684fdf4058cb8da5c84fbaa1.png)

**DBSCAN**算法的流程如下：
1. 随机选择一个未被访问的数据点；
2. 以该数据点为中心，以设定的半径$ε$为半径，找出该数据点半径范围内的所有数据点；
3. 如果该数据点半径范围内的数据点数量大于等于设定的最小样本数$M$，则将该数据点标记为核心点，并将半径范围内的所有数据点加入该核心点的簇中；
4. 如果该数据点半径范围内的数据点数量小于设定的最小样本数$M$，则将该数据点标记为噪声点；
5. 对于核心点的簇，以同样的方式递归地寻找其邻域内的数据点，并将其加入簇中；
6. 直到所有的数据点都被访问过。

在**DBSCAN**算法中，最重要的参数是邻域半径$ε$和邻域密度阈值$M$，它们需要根据数据集的特点和实际需求来进行调整。

一般而言，较大的$ε$会产生较少的簇，较小的$ε$会产生较多的簇。

![](https://pic1.imgdb.cn/item/684fe0ca58cb8da5c84fc581.png)

较大的$M$会产生较多的噪声点，较小的$M$会产生较少的噪声点。

![](https://pic1.imgdb.cn/item/684fe12158cb8da5c84fc7ce.png)

**DBSCAN**算法的主要优点是不需要提前设置聚类的个数；缺点是需要提前确定$ε$和$M$值；对初值选取敏感，对噪声不敏感；对密度不均的数据聚合效果不好。

使用**sklearn**库实现**DBSCAN**算法：

```python
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(42)
data = np.random.randn(1000, 2) * 3 + 10

# 调用DBSCAN算法
model = DBSCAN(eps=2.5, min_samples=5)
y_pred = model.fit_predict(data)

# 可视化结果
plt.scatter(data[:, 0], data[:, 1], c=y_pred)
plt.title('DBSCAN')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

# 3. HDBSCAN算法
- paper：[Density-Based Clustering Based on Hierarchical Density Estimates](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)

**HDBSCAN**算法将 **DBSCAN** 转换为层次聚类算法，然后使用基于聚类稳定性的技术来提取平坦聚类。

定义**相互可达距离(mutual reachability distance)**：

$$
d_{\text{mreach-k}}(a,b) =\max \{ \text{core}_k(a),\text{core}_k(b),d(a,b) \}
$$

其中$\text{core}_k(a)$是点$a$的$k$核心距离(**core distance**)，即点$a$到其第$k$个邻域点的距离。$d(a,b)$是点$a$和点$b$之间的原始度量距离。

![](https://pic1.imgdb.cn/item/6850db2e58cb8da5c852b5af.png)

根据相互可达距离构造最小生成树：每次构建一条边，始终添加连接当前树和尚未加入树的顶点的权重最低的边。

![](https://pic1.imgdb.cn/item/6850dbc058cb8da5c852b5e8.png)

将最小生成树转换为连通分量的层次结构：按距离对树的边进行排序（升序），然后迭代，为每条边创建一个新的合并簇。

![](https://pic1.imgdb.cn/item/6850dc2e58cb8da5c852b606.png)

引入**最小聚类大小(minimum cluster size)**。遍历层次结构，每次分裂时，如果分裂创建的新聚类中有一个簇的点数少于最小聚类大小，则把该簇标记为未聚类点，另一个簇保留其父聚类的聚类标识。如果分裂的两个簇的大小都不小于最小聚类大小，则认为簇分裂成立。

为实现平坦聚类，引入 $\lambda=1/$**distance** 衡量聚类的持久性。对于给定的聚类，定义$\lambda_{\text{birth}}$为聚类分裂成独立聚类的$\lambda$；对于聚类中的每个点$p$，定义$\lambda_p$为该点划分成未聚类点时的$\lambda$。每个聚类的稳定性定义为：

$$
\sum_{p \in \text{cluster}} (\lambda_p-\lambda_{\text{birth}})
$$

首先将所有叶节点（每一个数据点）声明为选定簇。然后向上遍历树（逆拓扑排序），如果分裂前簇的稳定性大于其子簇稳定性总和，则将两个节点子簇合并为一个簇；否则将该簇的稳定性设置为子簇稳定性的总和。到达根节点后，返回当前选定簇的集合为平面聚类的结果。

使用**sklearn**库实现**HDBSCAN**算法：

```python
import numpy as np
from sklearn.cluster import HDBSCAN

# 生成随机数据
np.random.seed(42)
data = np.random.randn(6, 2) * 3 + 10

# 调用DBSCAN算法
hdb = HDBSCAN(min_cluster_size=2)
hdb.fit(X)
hdb.labels_
# array([0, 0, 0, 1, 1, 1])
```

# 4. OPTICS算法
- paper：[OPTICS: ordering points to identify the clustering structure](https://dl.acm.org/doi/10.1145/304181.304187)

在**DBSCAN**算法中，使用了统一的邻域半径$ε$值，对于密度不均的数据选取一个合适的$ε$是很困难的。当数据密度不均匀的时候，如果设置了较小的$ε$值，则较稀疏的簇中的节点密度会比较小；如果设置了较大的$ε$值，则密度较大且离的比较近的簇容易被划分为同一个簇。

**OPTICS (Ordering Points To Identify the Clustering Structure)**是**DBSCAN**算法的一种有效扩展，主要解决对输入参数敏感的问题。**OPTICS**并不显式生成数据聚类，只是对数据集合进行排序，得到一个有序的对象列表和决策图，通过决策图可以知道$ε$取特定值时数据的聚类情况。

对于每个样本点$x$，定义**核心距离（core-distance）**为使得$x$成为核心点的最小邻域半径；对于其余点$y$，定义$y$关于$x$的**可达距离（reachability-distance）**为$x$的核心距离与$d(x,y)$中的较大值。

![](https://pic1.imgdb.cn/item/68511f5058cb8da5c8547bd1.png)

**OPTICS**算法的流程如下：
1. 把所有数据点看作核心点（相当于$ε=\infty$）；
2. 随机选择一个核心点，加入有序列表$p$；
3. 选择当前核心点的可达距离最小的点作为新的核心点，加入有序列表$p$，并把可达距离加入可达距离列表$r$；
4. 重复步骤3，直到所有点都加入有序列表$p$。

**OPTICS**算法返回一个有序列表$p$（记录了数据的处理顺序）和一个可达距离列表$r$（长度比有序列表小$1$，因为第一个处理点没有计算可达距离）。绘制决策图，横轴是有序列表，纵轴是可达距离列表。任意指定一个$ε$值作为水平线，位于该线之上的点为离群点，位于该线之下的点被划分为若干个簇。

![](https://pic1.imgdb.cn/item/6851322658cb8da5c8552f5e.png)

使用**sklearn**库实现**OPTICS**算法：

```python
import numpy as np
from sklearn.cluster import OPTICS

# 生成随机数据
np.random.seed(42)
data = np.random.randn(6, 2) * 3 + 10

# 调用DBSCAN算法
clustering = OPTICS(min_samples=2).fit(data)
clustering.labels_
# array([0, 0, 0, 1, 1, 1])
```
