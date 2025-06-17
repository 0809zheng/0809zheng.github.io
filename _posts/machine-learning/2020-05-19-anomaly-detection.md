---
layout: post
title: '异常检测(Anomaly Detection)'
date: 2020-05-19
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eb51281c2a9a83be56f3c55.jpg'
tags: 机器学习
---

> Anomaly Detection.

**异常检测（Anomaly Detection）**是指检测数据集中是否存在异常值或新给定的数据是否为异常值。若把数据集看作一个概率分布，则异常点是出现概率较低的数据点（**outlier**）。

![](https://pic.downk.cc/item/5eb51281c2a9a83be56f3c55.jpg)

常用的异常检测方法可以分为：
- 监督异常检测：数据集同时包括正常数据和异常数据且有标签，把异常检测转化为分类算法。
- 无监督异常检测：又称为离群点检测，此时数据集同时包括正常数据和异常数据且无标签，检测数据集中是否存在异常值。
1. 基于统计的方法：根据数据集服从的概率分布/模型来检测异常，如参数估计, $3\sigma$准则, 四分位距箱线图, **Grubbs’ Test**。
2. 基于距离的方法：通过计算比较数据与近邻数据集合的距离来检测异常，如**kNN**, 马氏距离。
3. 基于密度的方法：根据样本点的局部密度信息来检测异常，如**LOF**, **COF**, **SOS**。
4. 基于深度的方法：通过计算样本点在数据层次结构中的深度来检测异常，如**MVE**, 孤立森林。
5. 基于聚类的方法：异常点被认为是不属于任何聚类的点、距离最近的聚类结果较远的点或稀疏聚类和较小的聚类里的点，如**DBSCAN**, **CBLOF**, **LDCOF**。
- 半监督异常检测：又称为新奇点检测，此时数据集只包括正常数据，检测新给定的数据是否为异常值。
1. 基于距离/密度的方法：根据样本点的距离/密度信息来检测异常，如**LOF**, **One-Class SVM**。
2. 基于重构的方法：根据样本点的重构损失大小来检测异常，如**PCA**, 自编码器。



# 1. 监督异常检测

监督异常检测方法适用于数据集同时包括正常数据和异常数据且有标签的情况，把异常值检测转化为分类算法。这类方法也称作**Open-set Recognition**。

训练一个分类器，对数据进行分类的同时输出一个置信度$c$，表示该数据是正常的概率；选择一个阈值$λ$，若置信度$c$高于阈值$λ$，则认为数据点是正常的；否则是异常的。

![](https://pic.downk.cc/item/5eb501c2c2a9a83be55e0e9e.jpg)

置信度$c$可以使用网络经过**Softmax**之后得到概率分布的最大值或其熵的负值；也可以训练一个网络分别计算概率分布和置信度（[Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/abs/1802.04865)）：

![](https://pic.downk.cc/item/5eb50267c2a9a83be55ec058.jpg)


# 2. 无监督异常检测

无监督异常检测方法适用于数据集同时包括正常数据和异常数据且无标签的情况。这种异常值检测也被称作为**离群点检测（Outlier Detection）**。

离群点检测假设数据中包含**离群点(outlier)**远离其它**内围点(inlier)**，通过拟合出数据中内围点聚集的区域，忽略有偏离的观测值。

## （1）基于统计的方法

基于**统计**的方法假设数据集服从某种分布(如正态分布)或概率模型，通过判断某数据点是否符合该分布/模型来区分异常点，即通过小概率事件的判别实现异常检测。根据概率模型可分为**参数方法**与**非参数方法**。

### a. 参数方法

参数方法是指通过[参数估计](https://0809zheng.github.io/2020/02/02/parameter-estimation.html)建模已知数据分布$p(x)$(如正态分布)。对于一个新的数据点$x^i$，选择一个阈值$λ$；当$p(x^i)≥λ$时认为数据点是正常的，否则$p(x^i)<λ$时认为数据点是异常点。

通常用正态分布对数据分布建模：

$$ f_{\mu, \Sigma}(x) = \frac{1}{(2 \pi)^{\frac{d}{2}}} \frac{1}{\mid \Sigma \mid^\frac{1}{2}} exp(-\frac{1}{2}(x- \mu)^T \Sigma ^{-1} (x-\mu)) $$

用极大似然估计估计参数：

$$
\begin{aligned}
\mu^* &= \frac{1}{N} \sum_{n=1}^{N} {x^n} \\
\Sigma^* &= \frac{1}{N} \sum_{n=1}^{N} {(x^n-\mu^*)(x^n-\mu^*)^T}
\end{aligned}
$$

### b. 非参数方法

非参数方法是指在数据分布未知时，通过利用数据的变异程度( 如均差、标准差、变异系数、四分位数间距、直方图等) 来发现数据中的异常点。

### ⚪ $3\sigma$准则

$3\sigma$准则是指先假设一组检测数据只含有随机误差，对其进行计算处理得到标准偏差，按一定概率确定一个区间。超过这个区间的误差，就不属于随机误差而是粗大误差，含有该误差的数据应予以剔除。

这种判别处理方法仅局限于对**正态或近似正态分布**的样本数据处理，它是以测量次数充分大为前提，当测量次数少的情形用准则剔除粗大误差是不够可靠的。$3\sigma$准则为：
- 数值分布在$(μ-σ,μ+σ)$中的概率为$0.6827$
- 数值分布在$(μ-2σ,μ+2σ)$中的概率为$0.9545$
- 数值分布在$(μ-3σ,μ+3σ)$中的概率为$0.9973$

可以认为，数据的取值几乎全部集中在$(μ-3σ,μ+3σ)$区间内，超出这个范围的可能性仅占不到0.3%。

![](https://pic1.imgdb.cn/item/682e8c8c58cb8da5c803e1eb.png)

```python
import numpy as np

def find_anomalies(random_data):
    anomalies = []
    normal = []

    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3 # 将上、下限设为3倍标准差

    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    print("lower_limit ", lower_limit)
    print("upper_limit ", upper_limit)

    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
        else:
            normal.append(outlier)
    return np.array(anomalies), np.array(normal)

anomalies, normal = find_anomalies(data)
```


### ⚪ 箱线图

**箱线图 (boxplot)**是基于**四分位数间距（Interquartile range, IQR）**寻找异常点的。箱线图是数据通过其四分位数形成的图形化描述。**IQR**是数据集的上四分位数（**Q3：75th percentile**）与下四分位数（**Q1：25th percentile**）之间的距离(**IQR = Q3 -Q1**)。 

由于**IQR**只考虑中间50%的数据，不受异常值的影响，因此被广泛地应用于异常值检测。离群点被定义为低于箱形图下触须(**Q1 − 1.5 IQR**)或高于箱形图上触须(**Q3 + 1.5 IQR**)的观测值。

![](https://pic1.imgdb.cn/item/682e8e0c58cb8da5c803f47e.png)

该方法的优点是与方差和极差相比受极端值的影响不敏感，且处理大规模数据效果很好；缺点是小规模处理略显粗糙，只适合单个特征数据的检测。

```python
import numpy as np

def find_anomalies(random_data):
    anomalies = []
    normal = []

    iqr_25 = np.percentile(random_data, [25])
    iqr_75 = np.percentile(random_data, [75])

    lower_limit  = iqr_25 - 1.5 * (iqr_75 - iqr_25) 
    upper_limit = iqr_25 + 1.5 * (iqr_75 - iqr_25)
    print("lower_limit ", lower_limit)
    print("upper_limit ", upper_limit)
    # 异常
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
        else:
            normal.append(outlier)
    return np.array(anomalies), np.array(normal)

anomalies, normal = find_anomalies(data)
```

### ⚪ Grubbs’ Test

**Grubbs’ Test**是一种[假设检验](https://0809zheng.github.io/2021/07/09/hypothesis.html)的方法，常被用来检验服从正态分布的单变量数据集中的单个异常值。其思路为检验最大值、最小值偏离均值的程度是否为异常。原假设与备择假设如下：
- **H0**: 数据集中没有异常值
- **H1**: 数据集中有一个异常值

**Grubbs’ Test**的算法流程为：
1. 对所有样本从小到大排序；
2. 求所有样本的均值和标准差；
3. 计算最大/最小样本与均值的差距，差距更大的为可疑值；
4. 求可疑值的**z-score**，如果大于**Grubbs**临界值，则判断为异常值。

**Grubbs**临界值可以查表得到，它由两个值决定：检出水平$α$（越严格越小）和样本数量$n$。

**Grubbs’ Test**需假定数据服从正态分布，只能检测单维度数据，且每次只能检测一个异常值。为了将**Grubbs' Test**扩展到多个异常值检测，需要在数据集中逐步删除与均值偏离最大的值（即最大值或最小值），同步更新对应的**Grubbs**临界值，检验原假设是否成立。

## （2）基于距离的方法

基于**距离**的方法通过计算比较数据与近邻数据集合的距离来检测异常；其假设为正常数据点与其近邻数据相似，而异常数据则与近邻数据差异较大。

### ⚪ [kNN](https://0809zheng.github.io/2020/03/23/knn.html)

**kNN**依次计算每个样本点与它最近的**K**个样本的平均距离，再利用计算的距离与阈值进行比较，如果大于阈值，则认为是异常点。

该方法的优点是不需要假设数据的分布；缺点是仅可以找出全局异常点，无法找到局部异常点。

```python
from sklearn.neighbors import NearestNeighbors

# 使用KNN算法检测异常值
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(df)
distances, indices = neigh.kneighbors(df)

# 距离与阈值进行比较
y_pred = distances < 3

# 去除异常值
df = df.iloc[y_pred, :]
```

基于距离的方法设置的阈值是一个固定值，属于全局性方法。在实际中数据集数据分布通常不均匀，有的地方比较稠密，有的地方比较稀疏，这就可能导致阈值难以确定。

### ⚪ Mahalanobis Distance

[马氏距离（**Mahalanobis Distance**）](https://0809zheng.github.io/2021/03/03/distance.html#-%E9%A9%AC%E5%93%88%E6%8B%89%E8%AF%BA%E6%AF%94%E6%96%AF%E8%B7%9D%E7%A6%BB-mahalanobis-distance)是一种用于测量两个随机变量之间的相似度的指标，可以用于异常检测中。该距离度量的是两个变量之间的距离，而不仅仅是它们在空间上的距离。

在异常检测中，可以使用马氏距离来测量某个数据点与其它数据点之间的距离，并将其与预定义的阈值进行比较，以确定该点是否为异常值。

马氏距离的优点包括：
- 考虑了不同特征之间的相关性，可以更准确地表示数据点之间的距离。
- 可以用于处理高维数据，而且在多元正态分布的情况下表现良好。
- 相比于欧氏距离，马氏距离更能够处理不同特征之间的缩放和偏移。

马氏距离的缺点包括：
- 如果输入数据不是正态分布，则马氏距离的效果可能会降低。
- 计算马氏距离需要求解协方差矩阵，如果数据集的规模较大，则计算代价也会变得很高。
- 在一些非高斯分布的数据集上，马氏距离的效果可能会不如其它距离度量方法，比如欧氏距离等。



## （3）基于密度的方法

基于**密度**的方法通过计算数据集中各数据区域的密度，将密度较低区域作为离群区域。该方法的假设是正常数据样本位于密集的邻域中，而异常数据样本附近的样例较为稀疏。

### ⚪ Local Outlier Factor (LOF)

**局部离群因子（LOF）**是一种基于密度的异常检测算法，它可以有效地识别基于密度分布的局部异常点，即那些密度低于其邻域其他点的点。

该算法首先计算每个点的局部密度，然后将每个点的离群因子**LOF**定义为其邻域的点密度与其自身密度之比的平均值。**LOF**越大，则该点所在位置的密度越小于其周围样本所在位置的密度，表示该点越异常。

对于每个数据点，**LOF**算法的流程如下：
- 计算它的$k$个最近邻。
- 计算它的局部密度（即它的$k$个最近邻的平均距离的倒数）。
- 计算它的**LOF**（即它的邻域点密度与其自身密度之比的平均值）。

对于所有数据点，根据其**LOF**值进行排序，**LOF**值越大，则数据点越可能是异常值。

```python
from sklearn.neighbors import LocalOutlierFactor

# 使用LOF算法检测异常值
clf = LocalOutlierFactor(n_neighbors=5, algorithm='auto', leaf_size=5, metric='minkowski',p=2, 
                         metric_params=None, contamination='auto', novelty=False, n_jobs=-1)
y_pred = clf.fit_predict(df)

# 去除异常值
df = df.iloc[y_pred == 1, :] # 内围点被标记为1，而离群点被标记为-1
```

**LOF**算法的优点是它能够有效地检测出密度不均匀的数据集中的异常值，并且不需要假设任何数据分布模型。然而该算法在处理高维数据时会遇到困难，因为在高维空间中距离很难衡量相似性；此外需要手动调整的参数比较多，如$k$值和**LOF**阈值等。

### ⚪ Connectivity-Based Outlier Factor (COF)

**LOF**中计算距离常用欧式距离（默认数据是球状分布），**基于连接的离群因子（COF）**是**LOF**的变种，其局部密度是基于最短路径方法求出的平均链式距离计算得到。相比于**LOF**，**COF**可以处理低密度下的异常值。

计算每个点的平均链式距离，需要计算每个点在其邻域内的最短路径：只要直接计算该点和其邻域所有点所构成的graph的最小生成树(**minimum spanning tree**)，再以该点为起点执行最短路径算法，就可以得到该点在其邻域内的最短路径。该点的平均链式距离由其最短路径上近邻点的距离加权：

$$
dist(p) = \sum_{i=1}^k \frac{2(k+1-i)}{k(k+1)} dist(e_i)
$$

### ⚪ Stochastic Outlier Selection (SOS)

**随机离群选择（SOS）**算法根据样本的相异度矩阵计算一个异常概率值向量。其假设是如果一个点和其它所有点的**关联度（affinity）**都很小，它就是一个异常点。

**SOS**算法流程如下：
- 计算相异度矩阵$D$ （**dissimilarity matrix**）：各样本两两之间的度量距离；
- 计算关联度矩阵$A$ (**affinity matrix**)：反映度量距离方差，密度越大, 方差越小；
- 计算关联概率矩阵$B$ (**binding probability matrix**)：对关联矩阵按行归一化；
- 算出异常概率向量：

$$
p(x_i) = \prod_{j\neq i} (1-b_{ji})
$$

![](https://pic1.imgdb.cn/item/682ec36158cb8da5c8052114.png)

## （4）基于深度的方法

基于**深度**的方法把数据映射到空间分层结构中，并假设异常值分布在外围（深度较低），而正常数据点靠近分层结构的中心（深度较高）。

### ⚪ Minimum Volume Ellipsoid Estimator (MVE)

**最小椭球估计 (MVE)** 假设内围数据服从高斯分布，拟合出一个最小椭圆形球体的边界，不在此边界范围内的数据点将被判断为异常点。

```python
from sklearn.covariance import EllipticEnvelope

# 使用MVE算法检测异常值
cov = EllipticEnvelope(contamination=0.1) # the proportion of outliers in the dataset
y_pred = cov.fit_predict(df)

# 去除异常值
df = df.iloc[y_pred == 1, :] # 内围点被标记为1，而离群点被标记为-1
```

### ⚪ [孤立森林 (Isolation Forest, iForest)](https://0809zheng.github.io/2021/10/21/iforest.html)

孤立森林是一种基于树的异常检测算法，它通过构建并集成随机树来检测异常值。随机树通过随机选择一个特征和一个分割值来对数据进行分割，直到每个叶子节点包含一个数据点或者达到预定的停止条件。对于新的数据点，计算它在随机树中的路径长度。路径长度越短，则数据点越可能是异常值。

![](https://pic1.imgdb.cn/item/682ed26458cb8da5c80576c2.png)

```python
from sklearn.ensemble import IsolationForest

# 使用Isolation Forest算法检测异常值
clf = IsolationForest(random_state=42)
y_pred = clf.fit_predict(df)

# 去除异常值
df = df.iloc[y_pred == 1, :] # 内围点被标记为1，而离群点被标记为-1
```

## （5）基于聚类的方法

基于**聚类**的异常检测方法通常依赖下列假设之一：
1. **不属于任何聚类的点是异常点**：正常样本属于数据中的一个簇，而异常样本不属于任何簇；典型方法为**DBSCAN**。
2. **距离最近的聚类结果较远的点是异常点**：正常样本靠近它们最近的簇质心，而异常样本离它们最近的簇质心很远。首先进行聚类，然后计算样例与其所属聚类中心的距离和其所属聚类的类内平均距离，用两者的比值衡量异常程度。典型方法为**K-Means**；
3. **稀疏聚类和较小的聚类里的点都是异常点**：正常样本属于大而密集的簇，而异常样本要么属于小簇，要么属于稀疏簇。首先进行聚类，然后启发式地将聚类簇分成大簇和小簇；如果某一样例属于大簇，则利用该样例和其所属大簇计算异常得分；如果某一样例属于小簇，则利用该样例和距离其最近的大簇计算异常得分。这类方法考虑到了数据全局分布和局部分布的差异，可以发现异常簇。典型方法为**CBLOF**、**LDCOF**。

### ⚪ [DBSCAN](https://0809zheng.github.io/2020/05/05/mean-shift.html#2-dbscan%E7%AE%97%E6%B3%95)

**DBSCAN**算法把数据点分为三类：核心点、边界点和噪声点。
- **核心点**是指在以该点为圆心，以设定的半径`eps`为半径的圆内，包含大于等于设定的最小样本数`min_samples`的点；
- **边界点**是指在以该点为圆心，以设定的半径`eps`为半径的圆内包含核心点的其余点；
- **噪声点**是指既不是核心点也不是边界点的点。

对于无法形成聚类簇的孤立点，即为异常点（噪声点）。

```python
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=3, min_samples=2).fit(X)

clustering.labels_
array([ 0,  0,  0,  1,  1, -1])
# -1为异常点：不属于任何一个簇
```

### ⚪ CBLOF

**CBLOF**是基于聚类的局部离群因子（**Cluster-based Local Outlier Factor**），基本思路是对数据进行聚类区分出大簇和小簇，由于异常值占少数，往往会和大部分正常数据有较大偏差，因此通过计算数据与大簇之间的距离来衡量数据的异常程度，距离越大则数据越异常。

区分大小簇的方式有两种，将所有簇按数据量从大到小排序之后：
- 前几簇的和占总量的$α$（一般取$α=0.9$）可以认为是大簇，后几簇为小簇：$\sum_{i=1}^b \|C_i \| \geq \alpha \|D \|$
- 当前一簇是后一簇数量的$β$倍时（一般$β=5$），可以认为前几簇为大簇，后几簇为小簇：$\|C_b \|/\|C_{b+1} \|\leq \beta$

当数据点属于大簇时，计算它与当前簇的聚类中心的距离；当数据点属于小簇时，计算它与最近的大簇的聚类中心的距离。得出的异常分数从大到小排序，就可以挑选出异常值了。

### ⚪ LDCOF

**LDCOF (Local Density Cluster-Based Outlier Factor)**建立在**CBLOF**的基础上，用簇内点的平均距离来正规化异常分数的计算。

# 3. 半监督异常检测

半监督异常检测方法适用于数据集只包括正常数据的情况。这种异常值检测也被称作为**新奇点检测（Novelty Detection）**。

新奇点检测对于新观测值(**observation**)进行判断，判断其是否与现有观测值服从同一分布(即内围点)，相反则被认为不服从同一分布(即新观测值为新奇点)。

## （1）基于距离/密度的方法

第**2.2、2.3**节讨论的基于距离/密度的方法也可以进行新奇点检测，即对新的未见过的样本预测其标签或计算其异常性得分。以**局部离群因子（Local Outlier Factor, LOF）**方法为例：

```python
from sklearn.neighbors import LocalOutlierFactor

# 使用LOF算法检测异常值
clf = LocalOutlierFactor(n_neighbors=5, algorithm='auto', leaf_size=5, metric='minkowski',p=2, 
                         metric_params=None, contamination='auto', novelty=False, n_jobs=-1)
clf.fit(X_train)
y_pred = clf.predict(X_test)

# 计算每个数据点的异常得分
scores = rcf.decision_function(X_test)

# 去除异常值
X_test = X_test.iloc[y_pred == 1, :] # 内围点被标记为1，而离群点被标记为-1
```

### ⚪ [单类别支持向量机 (One-Class SVM)](https://0809zheng.github.io/2020/03/14/SVM.html#8-%E5%8D%95%E7%B1%BB%E5%88%AB%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)

单类别支持向量机在特征空间中获得数据周围的超球面边界，期望最小化这个超球体的体积，从而最小化异常点数据的影响。识别一个新的数据点时，如果这个数据点落在超球面内，就属于这个类；否则是异常样本。

![](https://pic1.imgdb.cn/item/68302af258cb8da5c80a23e0.png)

**One-Class SVM**的优化目标是求一个中心为$o$、半径为$r$的最小球面，能够包括所有训练样本（所有训练数据点$x_i$到中心$o$的距离严格小于$R$）：

$$
\begin{aligned}
\mathop{\min}_{o,r} \quad & r^2 + C\sigma_i ξ_i \\
\text{s. t. } \quad & (x_i-o)^\top (x_i-o) \leq r^2 + ξ_i \\
& \forall_i ξ_i \geq 0
\end{aligned}
$$

$C$是调节松弛变量$ξ_i$的惩罚系数，如果$C$比较小，会给离群点较大的弹性，使得它们可以不被包含进超球体。

```python
sklearn.svm.OneClassSVM(
    kernel='rbf',  # 核函数: linear, poly, rbf, sigmoid, precomputed
    degree=3,      # ploy核函数的阶数
    gamma='auto',  # 非线性核函数的因子，默认1/n_features
    coef0=0.0,     # poly, sigmoid核的独立项
    tol=0.001,     # 训练停止容差 
    nu=0.5,        # 支持向量比例的下界
    shrinking=True,# 加速训练：分阶段地减少候选支持向量
    cache_size=200,# 核缓存MB
    verbose=False,
    max_iter=-1,   # 最大迭代次数
    random_state=None
)
```

该函数提供了以下方法：
- `fit(X)`：训练，根据训练样本探测边界
- `predict(X)`：返回预测值，+1是正常样本，-1是异常样本。
- `decision_function(X)`：返回各样本点到超平面的函数距离（**signed distance**），正的为正常样本，负的为异常样本。
- `set_params(**params)`：设置评估器的参数
- `get_params([deep])`：获取评估器的参数
- `fit_predict(X[, y])`：训练+推理

## （2）基于重构的方法

基于**重构**的方法根据正常数据训练一个重构模型，能够重建正常样本，但是却无法将异于正常分布的数据点较好地还原，导致其重构误差较大。当重构误差大于某个阈值时，将其标记为异常值。

### ⚪ PCA

使用[主成分分析 (**PCA**)](https://0809zheng.github.io/2020/04/11/PCA.html)对数据做特征值分解，会得到特征向量（反应了原始数据方差变化程度的不同方向）和特征值（数据在对应方向上的方差大小）。最大特征值对应的特征向量为数据方差最大的方向，最小特征值对应的特征向量为数据方差最小的方向。

原始数据在不同方向上的方差变化反应了其内在特点。如果单个数据样本跟整体数据样本表现出的特点不太一致，比如在某些方向上跟其它数据样本偏离较大，可能就表示该数据样本是一个异常点。

**PCA**做异常检测有两种思路：
1. 将数据映射到低维特征空间，然后在特征空间不同维度上查看每个数据点跟其它数据的偏差；
2. 将数据映射到低维特征空间，然后由低维特征空间重新映射回原空间，尝试用低维特征重构原始数据，看重构误差的大小。

### 思路1

对数据$x$做特征值分解得到特征向量$e$和特征值$\lambda$。样本$x_i$的异常分数计算为该样本在所有方向上的偏离程度：

$$
s(x_i) = \sum_{j=1}^{n} \frac{(x_i^T e_j)^2}{\lambda_j}
$$

若存在样本点偏离各主成分方向较远，则异常分数高。其中特征值$\lambda$用于归一化，使不同方向上的偏离程度具有相同的尺度。

在计算异常分数时，关于特征向量的选择又有两种方式：
- 考虑前$k$个特征向量方向上的偏差：前$k$个特征向量往往直接对应数据里的某几个原始特征，偏差比较大的数据样本往往就是在原始数据中那几个特征上的极值点。
- 考虑后$r$个特征向量方向上的偏差：后$r$个特征向量通常表示某几个原始特征的线性组合，偏差比较大的数据样本表示它在原始数据里对应的那几个特征上出现了与预计不太一致的情况。

### 思路2

把数据从原始空间投影到主成分空间，然后再从主成分空间投影回原始空间。对于大多数的数据而言，如果只使用第一主成分（或前$k$个主成分）来进行投影和重构，重构之后的误差是较小的。但是对于异常点而言，重构之后的误差相对较大。定义重构情况下的异常分数：

$$
s(x_i) = \sum_{k=1}^{n} |x_i-\hat{x}_{ik}| \cdot \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{n}\lambda_j}
$$

### ⚪ [自编码器 Autoencoder](https://0809zheng.github.io/2020/04/09/autoencoder.html)

自编码器是一种神经网络模型，其基本结构包括一个编码器和一个解码器，其中编码器将输入数据压缩为一个低维度的向量，解码器将这个向量恢复为原始的输入数据。自编码器的训练目标是最小化重构误差，即输入数据与解码器输出数据之间的差异。

自编码器与主成分分析**PCA**类似，但是自编码器在使用非线性激活函数时克服了**PCA**线性的限制。在异常检测中，可以使用自编码器来识别那些与其他数据点显著不同的异常值。先使用自编码器训练一个模型来学习正常数据的表示，然后用该模型对所有数据进行重构，并计算每个数据点的重构误差。重构误差较大的数据点可能是异常值。