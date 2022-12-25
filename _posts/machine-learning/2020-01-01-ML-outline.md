---
layout: post
title: '机器学习 概述'
date: 2020-01-01
author: 郑之杰
cover: ''
tags: 机器学习
---

> Outlines about Machine Learning.

- 提示：请点击任意[<font color=Blue>超链接</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html)以发现更多细节！

**机器学习**(**Machine Learning**)作为一门新兴的学科，已经逐渐渗透进我们的生活之中；迄今为止对于机器学习尚没有官方的定义，下面是一些关于机器学习的定义不妨作为参考：
- **Arthur Samuel**($1959$)：不需要对计算机进行明确的设置的情况下使计算机拥有自主学习的能力。(**Field of study that gives computers the ability to learn without being explicitly programmed.**)
- **Tom Mitchell**($1998$)：假设用性能度量P来估计计算机程序在某任务T上的性能，若程序通过经验E在任务T上获得性能改善，则程序对经验E进行了学习。(**A computer program is said to learn from experience E with respect to some task T and some performance measure P, if itsperformance on T, as measured by P, improves with experience E.**)
- **李航**：统计学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行预测与分析的一门学科。
- **周志华**：机器学习所研究的主要内容，是关于在计算机上从数据中产生"模型"的算法，即"学习算法"。
- **李宏毅**：机器学习就是自动找函数。
- **林轩田**：通过经验数据提高表现。(**Improving some performance measure with experience computed from data.**)

学习机器学习相关算法前，需要先了解机器学习的一些基本定理和概念：
- [<font color=Blue>机器学习的一些定理</font>](https://0809zheng.github.io/2020/02/04/ML-theorem.html)：没有免费午餐定理, 归纳偏置, 奥卡姆剃刀, 采样偏差, 数据窥探, 维度灾难
- [<font color=Blue>计算学习理论</font>](https://0809zheng.github.io/2020/02/05/vcdimension.html)：**Hoeffding**不等式, **VC**维, 模型复杂度, 概率近似正确**PAC**, 结构风险最小化
- [<font color=Blue>模型评估方法</font>](https://0809zheng.github.io/2020/02/06/validation.html)：留出法, 交叉验证法, 自助法
- [特征选择方法]()：
- [<font color=Blue>核方法</font>](https://0809zheng.github.io/2021/07/23/kernel.html)：将线性模型拓展为非线性模型

本文将从不同的角度介绍常用的机器学习方法，包括：
- [<font color=Blue>监督学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#-%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)：学习数据与标签之间的对应关系。
- [<font color=Blue>无监督学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#-%E6%97%A0%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)：从无标签的数据中学习特征表示。
- [<font color=Blue>集成学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0)：结合多个子模型提高决策准确率。
- [<font color=Blue>强化学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)：
- [<font color=Blue>迁移学习</font>](https://0809zheng.github.io/2020/05/22/transfer-learning.html)
- [<font color=Blue>终身学习</font>](https://0809zheng.github.io/2020/05/21/lifelong-learning.html)
- [<font color=Blue>元学习</font>](https://0809zheng.github.io/2020/05/20/meta-learning.html)

# ⚪ 监督学习
**监督学习**(**Supervised Learning**)是指给计算机同时提供训练数据(**data**,也叫**样本sample**)及其标签(**label**,也称**ground truth**)，希望学习数据与标签之间的对应关系。

监督学习又可以分为**回归**问题和**分类**问题，实际上这类似于解决**连续性**问题和**离散性**问题。

## (1) 回归
**回归(Regression)**的输出是连续值，范围为整个实数空间或其中一部分空间。
- [<font color=Blue>线性回归(Linear Regression)</font>](https://0809zheng.github.io/2020/03/12/regression.html)：输入特征的加权线性组合，包括求解的**最小二乘法**和**正规方程法**, 以及**广义线性模型**和**非线性回归模型**。
- [<font color=Blue>正则化的线性回归(Ridge/LASSO Regression)</font>](https://0809zheng.github.io/2020/03/30/ridge.html)：包括引入**L2**正则化的**岭回归**/**核岭回归**, 以及引入**L1**正则化的**LASSO回归**。
- [<font color=Blue>前向逐步回归(Stagewise Regression)</font>](https://0809zheng.github.io/2020/04/04/stagewise.html)：贪心地对权重的每个维度逐步试探(增大或减小一个步长)。
- [<font color=Blue>局部加权线性回归(Local Weighted Linear Regression)</font>](https://0809zheng.github.io/2020/03/31/lwlr.html)：根据实际提供的测试样本为每个训练样本赋予不同权重后再进行回归。
- [<font color=Blue>偏最小二乘回归(Partial Least Squares)</font>](https://0809zheng.github.io/2020/04/05/pls.html)：把训练样本的特征线性组合为互不相关且与标签相关的新特征再进行回归。
- [<font color=Blue>Tube回归(Tube Regression)</font>](https://0809zheng.github.io/2020/03/29/tube.html)：在回归线附近划定一个不计算误差的中立区。
- [<font color=Blue>支持向量回归(Support Vector Regression, SVR)</font>](https://0809zheng.github.io/2020/03/15/support-vector-regression.html)：引入支持向量的**Tube**回归，使得中立区能够覆盖所有样本点。


## (2) 分类
**分类(Classification)**的输出是离散值，把输入样本划分为有限个类别。

- [分类任务的常用性能指标](https://0809zheng.github.io/2020/02/07/classperform.html)：准确率指标,**P-R**曲线与**F1-score**,**ROC**曲线与**AUC**,代价曲线

根据输出范围的取值不同，分类可以划分为**硬分类(hard classify)**和**软分类(soft classify)**：
- **硬分类**：寻找数据空间中的一个或多个分类超平面，并把输入数据划分到某个具体的类别，输出范围是$\{0,1\}$。包括：
1. [<font color=Blue>感知机(Perceptron)</font>](https://0809zheng.github.io/2020/03/11/perceptron.html)：根据错误分类的样本更新参数，包括线性可分数据集的**感知机学习算法(PLA)**和线性不可分数据集的**口袋(pocket)算法**。
2. [<font color=Blue>线性判别分析(Linear Discriminant Analysis, LDA)</font>](https://0809zheng.github.io/2020/03/24/lda.html)：把样本投影到类间距离大、类内方差小的超平面上，包括**二分类LDA**, **多分类LDA**, **核LDA**。
3. [<font color=Blue>支持向量机(Support Vector Machine, SVM)</font>](https://0809zheng.github.io/2020/03/14/SVM.html)：寻找最大间隔分离超平面，包括线性**SVM**, 对偶**SVM**, 核**SVM**, 软间隔**SVM**, 序列最小最优化算法**SMO**, 概率**SVM**, 最小二乘**SVM**
4. [<font color=Blue>k近邻算法(k-Nearest Neighbor, kNN)</font>](https://0809zheng.github.io/2020/03/23/knn.html)：在训练集中搜索与测试样本最邻近的$k$个样本进行多数表决。
5. [<font color=Blue>决策树(Disicion Tree)</font>]()：对数据集进行划分的树形结构算法，递归地根据分支条件进行特征选择，包括**ID3**, **C4.5**, **CART**, 决策树的剪枝。
- **软分类**：预测输入数据可能属于每一个类别的概率，输出范围是$[0,1]$，包括：
1. [<font color=Blue>逻辑回归 Logistic</font>](https://0809zheng.github.io/2020/03/13/logistic-regression.html): **Logistic**回归, 交叉熵损失, 核**Logistic**回归
2. [<font color=Blue>朴素贝叶斯(Naive Bayes)</font>](https://0809zheng.github.io/2020/03/28/naivebayes.html)：在贝叶斯公式中引入条件独立性假设，并进行后验概率最大化。
3. [<font color=Blue>最大熵模型</font>](https://0809zheng.github.io/2021/07/20/me.html)
4. [高斯判别分析]()


## (3) 神经网络
**神经网络**(**Neural Network, NN**)是一类特殊的机器学习方法，通常是由多层感知机构成的。根据其输出层激活函数的选择不同，既可以用于回归又可以用于分类。神经网络又衍生出[深度学习](https://0809zheng.github.io/2020/01/02/DL-outline.html)这一领域。一些特殊的神经网络模型如下：
1. [前馈神经网络(多层感知机)](https://0809zheng.github.io/2020/04/17/feedforward-neural-network.html)
3. [径向基函数网络](https://0809zheng.github.io/2020/04/18/rbf-network.html)
4. [深度信念网络](https://0809zheng.github.io/2020/04/16/deep-belief-network.html)
5. [<font color=Blue>自组织映射网络 SOM</font>](https://0809zheng.github.io/2022/01/06/SOM.html)：竞争学习型的无监督神经网络

## (4) [<font color=Blue>推荐系统</font>](https://0809zheng.github.io/2020/05/08/recommender-system.html)



# ⚪ 无监督学习
**无监督学习(Unsupervised Learning)**是指提供给计算机的数据不再带有标签，希望从无标签的数据中学习出有效的**特征**或**表示**。

常见的无监督学习方法包括**聚类**、**降维**、**异常检测**、**生成模型**。

## (1)聚类
**聚类(Clustering)**是将一组样本根据一定的准则划分到不同的**簇（Cluster）**，如按照数据之间的相似性把相近的数据划分为一类。

常见的聚类方法：
1. [K-Means聚类](https://0809zheng.github.io/2020/05/02/kmeans.html)
2. [层次聚类](https://0809zheng.github.io/2020/05/03/hierarchical-clustering.html)：Agglomerative Nesting(AGNES)、Divisive Analysis(DIANA)
3. [Mean-Shift](https://0809zheng.github.io/2020/05/05/mean-shift.html)
4. [谱聚类](https://0809zheng.github.io/2020/05/04/spectral-clustering.html)

## (2)降维
**降维**(**Dimensionality Reduction**)是指降低数据的特征维度。由于数据的某些特征具有相关性，降维可以作为数据预处理方法，减少特征冗余。此外降低数据的特征维度有利于可视化。


根据降维的运算是线性的还是非线性的，可分为线性降维和非线性降维。
- **线性降维**是在高维空间中寻找一个子空间，把高维空间的数据线性映射到子空间中。线性降维可通过线性变换进行，表示为$Z=WX$，不同的线性降维无非是为线性变换矩阵$W$施加了不同的约束。
1. [<font color=Blue>主成分分析(Principal Component Analysis, PCA)</font>](https://0809zheng.github.io/2020/04/11/PCA.html)：对归一化数据的协方差矩阵进行特征值分解，可以从(几何,线性变换,最大投影方差,最小重构代价,奇异值分解)角度理解, 还包括**主坐标分析(PCoA)**, **概率PCA**。
2. [<font color=Blue>多维缩放(Multiple Dimensional Scaling, MDS)</font>](https://0809zheng.github.io/2021/07/28/mds.html)：原始空间中样本之间的相对位置关系(距离)在低维空间得以保持。
3. [<font color=Blue>局部保留投影(Locality Preserving Projection, LPP)</font>](https://0809zheng.github.io/2021/09/30/lpp.html)：原始空间中样本之间的局部相对位置关系(考虑$k$近邻点)在低维空间得以保持。
- **非线性降维**则假设高维空间到低维空间的函数映射是非线性的。一种非线性降维方法是引入**核方法**，即先构造非线性的高维特征空间，再应用线性降维；另一种方法是**流形(manifold)学习**，即将高维空间中的流形张成一个低维空间，并保留数据的相互关系。
1. [<font color=Blue>核主成分分析(Kernelized Principal Component Analysis, KPCA)</font>](https://0809zheng.github.io/2021/07/27/kpca.html)：先通过核方法把数据映射到高维特征空间，再通过**PCA**投影到低维空间。
2. [<font color=Blue>等度量映射(Isometric Mapping, ISOMAP)</font>](https://0809zheng.github.io/2021/07/30/isomap.html)：原始空间中样本之间的相对位置关系(距离)在低维空间得以保持。
3. [<font color=Blue>局部线性嵌入(Locally Linear Embedding, LLE)</font>](https://0809zheng.github.io/2021/07/31/lle.html)：原始空间中样本之间的局部相对位置关系(考虑$k$近邻点)在低维空间得以保持。
4. [<font color=Blue>t分布随机近邻嵌入(t-distributed Stochastic Neighbor Embedding, t-SNE)</font>](https://0809zheng.github.io/2021/09/23/tsne.html)：用正态分布和**t**分布分别建模原始空间和低维空间中样本点的相对位置关系，使用**KL**散度衡量两个分布的距离。
5. [<font color=Blue>一致流形近似与投影(Uniform Manifold Approximation and Projection, UMAP)</font>](https://0809zheng.github.io/2021/09/24/umap.html)：用相似度函数建模原始空间和低维空间中样本点的相对位置关系，使用交叉熵损失衡量两个函数的距离。


- **非线性降维**：
1. [流形学习](https://0809zheng.github.io/2020/04/07/manifold.html)(LLE、LE、[t-SNE](https://0809zheng.github.io/2020/04/10/t-SNE.html))
2. [稀疏编码](https://0809zheng.github.io/2020/04/08/sparse-coding.html)
3. [自编码器](https://0809zheng.github.io/2020/04/09/autoencoder.html)


## (3)异常检测
[**异常检测(anomaly detection)**](https://0809zheng.github.io/2020/05/19/anomaly-detection.html)用来判断数据集中是否存在异常点，或者一个新的数据点是否正常。

## (4)生成模型
**生成模型（generative model）**通过学习数据的内在结构或数据中不同元素之间的依赖性，对高维数据（如图像、音频、视频、文本）的概率分布进行估计；通过这种数据表示可以进一步生成新的数据。

如果数据的概率分布形式是已知的，则可以通过**极大似然估计**等方法求得数据分布的解析解。

### a. 隐变量模型

**隐变量模型(latent variable model)**是一类强大的生成模型，其主要思想是在已知的**观测数据(observed data)**$x_i$后存在未观测到的**隐变量(latent variable)**$z_i$，其图模型如下：

![](http://adamlineberry.ai/images/vae/graphical-model.png)

隐变量模型的概率分布表示如下：

$$ p_{\theta}(x,z) = p_{\theta}(x | z)p_{\theta}(z) $$

隐变量的引入为模型引入了一些先验知识，增强了模型的可解释性。一些常见的隐变量模型及其隐变量的含义如下：

- [高斯混合模型](https://0809zheng.github.io/2020/05/29/gaussian-mixture-model.html)：隐变量表示属于不同子高斯分布的概率(**the cluster assignments**)
- []()：In latent Dirichlet allocation (LDA) the latent variables are the topic assignments
- [<font color=blue>变分自编码器</font>](https://0809zheng.github.io/2022/04/01/vae.html)：隐变量是数据的压缩表示(**the compressed representations of that data**)

求解隐变量模型的方法包括：
- [期望最大算法](https://0809zheng.github.io/2020/03/26/expectation-maximization.html)：$p(z \| x)$可解
- [变分推断](https://0809zheng.github.io/2020/03/25/variational-inference.html)：$p(z \| x)$不可解

### b. [<font color=blue>能量模型</font>](https://0809zheng.github.io/2020/04/12/energy.html)

**能量模型(energy-based model)**是指使用如下概率模型拟合一批真实数据$x_1,x_2,\cdots,x_n$~$p(x)$：

$$ q_{\theta}(x) = \frac{e^{-U_{\theta}(x)}}{Z_{\theta}}, Z_{\theta} = \int e^{-U_{\theta}(x)}dx $$

其中$U_{\theta}(x)$是带参数的**能量函数**；$Z_{\theta}$是**配分函数**(归一化因子)。

直观地，真实数据应该分布在能量函数中势最小的位置。能量模型的学习过程旨在通过调整能量函数$U_{\theta}(x)$，使得真实数据落入能量函数的极值点处。

![](https://pic1.imgdb.cn/item/634e13f716f2c2beb1b9d59f.jpg)

不同的能量模型具有不同的能量函数$U_{\theta}(x)$形式。

| 模型 | 网络结构 | 能量函数$U_{\theta}(x)$ |
| :---: | :---:  | :---:  |
| [<font color=blue>Hopfield神经网络</font>](https://0809zheng.github.io/2020/04/13/hopfield-network.html) | ![](https://pic1.imgdb.cn/item/634e998716f2c2beb10af2d5.jpg) |$$ -\frac{1}{2}\sum_{i,j}^{} {w_{ij}x_ix_j} - \sum_{i}^{} {b_ix_i} $$ |
| [<font color=blue>玻尔兹曼机 BM</font>](https://0809zheng.github.io/2020/04/14/boltzmann-machine.html) | ![](https://pic1.imgdb.cn/item/634e998716f2c2beb10af2d1.jpg) |$$ -(\sum_{i<j}^{} {w_{ij}x_ix_j} + \sum_{i}^{} {b_{i}x_i}) $$ |
| [<font color=blue>受限玻尔兹曼机 RBM</font>](https://0809zheng.github.io/2020/04/15/restricted-boltzmann-machine.html) | ![](https://pic1.imgdb.cn/item/634e998716f2c2beb10af2e0.jpg) |$$ -(\sum_{i}^{} {a_ix_i} + \sum_{j}^{} {b_jz_j} + \sum_{i,j}^{} {w_{ij}x_iz_j}) $$ |

⭐扩展阅读：
- [<font color=blue>Concept Learning with Energy-Based Models</font>](https://0809zheng.github.io/2020/07/05/concept.html)：(arXiv1811)使用能量模型进行概念学习。

# ⚪ 集成学习
**集成学习(Ensemble Learning)**是指构建多个子模型，并通过某种策略将它们结合起来，从而通过群体决策来提高决策准确率。

若构建的子模型是同种类型的模型(如都是决策树)，则集成是**同质(homogeneous)**的，每个子模型被称为**基学习器(base learner)**，相应的学习算法称为**基学习算法(base learning algorithm)**。

若构建的子模型是不同类型的模型，则集成是**异质(heterogenous)**的，每个子模型被称为**组件学习器(component learner)**。

通常希望构建的子模型具有一定的**准确率**(至少不差于**弱学习器**,即泛化性能略优于随机猜测的学习器)，又具有一定的**多样性**(即不同子模型之间具有一定的差异)。[<font color=Blue>多样性(Diversity)</font>](https://0809zheng.github.io/2021/07/21/diversity.html)衡量子模型之间的两两不相似性，可通过在学习过程中引入随机性以增强多样性。

[<font color=Blue>误差-分歧分解(Error-Ambiguity Decomposition)</font>](https://0809zheng.github.io/2020/03/16/blending.html)指出，集成学习中集成模型的泛化误差$E$是由子模型的平均泛化误差$\overline{E}$和子模型的分歧$\overline{A}$共同决定的：

$$ E= \overline{E}-\overline{A} $$

根据**子模型的构建方式**，目前的集成学习方法可以分成两类：
- **并行化**集成方法：子模型之间不存在强依赖关系，可以同时生成，主要关注降低**方差**。如：
1. [<font color=Blue>Blending</font>](https://0809zheng.github.io/2020/03/16/blending.html)：通过投票法(平均法)组合子模型的结果，包括**Uniform Blending**(平均组合), **Linear Blending**(加权组合), **Stacking**(引入一个新模型组合结果)。
2. [<font color=Blue>Bagging</font>](https://0809zheng.github.io/2020/03/17/bagging.html)：通过**bootstrapping**方法从训练集中采样若干子集，在每个子集上训练子模型。
3. [<font color=Blue>随机森林(Random Forest)</font>](https://0809zheng.github.io/2020/03/20/random-forest.html)：以决策树为基学习器构造**Bagging**模型，训练过程中引入随机子空间算法。
- **序列化**集成方法：子模型之间存在强依赖关系，必须串行生成，主要关注降低**偏差**。如：
1. [<font color=Blue>Boosting</font>](https://0809zheng.github.io/2020/03/18/boosting.html)：**自适应提升(AdaBoost)**通过为样本权重重新赋值训练子模型；**梯度提升(Gradient Boosting)**使用损失函数的负梯度拟合子模型。
2. [<font color=Blue>提升树(Boosting Tree)</font>](https://0809zheng.github.io/2020/03/22/boosttree.html)：以决策树为基学习器构造**Boosting**模型。**自适应提升决策树(Adaptive Boosted Decision Tree, ABDT)**适用于指数损失的二分类任务；回归提升树适用于平方误差损失的回归任务；[<font color=Blue>梯度提升决策树(Gradient Boosted Decision Tree, GBDT)</font>](https://0809zheng.github.io/2020/03/21/GBDT.html)适用于一般损失函数的一般决策问题。




# ⚪ 强化学习
**强化学习(Reinforcement Learning, RL)**是指我们给模型一些输入，但是不提供希望的真实输出，根据模型的输出反馈，如果反馈结果接近真实输出，就给其正向激励；如果反馈结果偏离真实输出，就给其反向激励。不断通过**反馈-修正**这种形式，逐步让模型学习的更好。

⭐扩展阅读：
- [多臂老虎机(Multi-Armed Bandit, MAB)](https://0809zheng.github.io/2020/09/02/multiarm-bandit.html)
- [Investigating Human Priors for Playing Video Games](https://0809zheng.github.io/2020/07/09/game-prior.html)：(arXiv1802)探究电子游戏的人类先验知识。
- [Simple Regret Minimization for Contextual Bandits](https://0809zheng.github.io/2020/09/03/srm.html)：(arXiv1810)上下文多臂老虎机问题的简单遗憾最小化。
- [Reinforcement Learning with Augmented Data](https://0809zheng.github.io/2020/07/02/RAD.html)：(arXiv2004)RAD：把数据增强方法应用到强化学习。
- [Divide-and-Conquer Monte Carlo Tree Search For Goal-Directed Planning](https://0809zheng.github.io/2020/07/03/dcmcts.html)：(arXiv2004)分治的蒙特卡洛树搜索解决目标导向的强化学习问题。


# ⚪ 相关课程与书籍

### ⭐ 扩展阅读：
- [<font color=Blue>On the Measure of Intelligence</font>](https://0809zheng.github.io/2020/10/17/arc.html)：(arXiv1911)测试人工智能的抽象推理能力。
- [<font color=Blue>The Hardware Lottery</font>](https://0809zheng.github.io/2020/10/11/hardware.html)：(arXiv2009)机器学习中的硬件彩票理论。


### ⭐ 相关课程：

- [Machine Learning - Coursera （Andrew Ng）](https://www.coursera.org/learn/machine-learning)：监督学习（线性回归、逻辑回归、推荐系统），无监督学习（K-means聚类、PCA降维、异常检测），系统设计
- [李宏毅机器学习](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)![](https://pic1.imgdb.cn/item/634e994916f2c2beb10a65c6.jpg)
- [林轩田机器学习基石](https://www.bilibili.com/video/BV1Cx411i7op?p=1)：机器学习可行性（**Hoeffding**不等式、**VC**维），三个模型（感知机算法、线性回归、逻辑回归），三个技巧（特征转换、正则化、验证集）
- [林轩田机器学习技法](https://www.bilibili.com/video/BV1ix411i7yp?p=1)：核方法（支持向量机、支持向量回归），集成方法（集成，决策树、随机森林、**GBDT**），特征提取方法（神经网络、深度学习、径向基函数网络，推荐系统）
- [【机器学习】白板推导系列](https://space.bilibili.com/97068901/video)：B站up主“shuhuai008”自制课程，包括若干机器学习的理论推导
- [莫烦Python机器学习教程](https://mofanpy.com/)：B站up主“莫烦Python”自制课程，包含若干python库的入门（numpy、pandas、matplotlib、sklearn、Tensorflow、Keras、Pytorch）




### ⭐ 相关书籍

-  [《统计学习方法》(李航)](https://www.cnblogs.com/lishuairg/p/11734842.html)
- [《机器学习》(周志华)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm)：西瓜书
- [《南瓜书》](https://datawhalechina.github.io/pumpkin-book/#/)：补充西瓜书中的部分公式推导细节
- [《Pattern Recognition and Machine Learning》(Bishop)](http://research.microsoft.com/~cmbishop/PRML)
- [《Hands-On Machine Learning with Scikit-Learn & TensorFlow》(Geron)](https://github.com/ageron/handson-ml)
