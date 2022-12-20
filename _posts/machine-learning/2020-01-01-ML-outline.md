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

一种通用的分类方法是，根据是否提供数据的标签，把机器学习算法分成两类：
- [<font color=Blue>监督学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#监督学习)：回归, 分类
- [<font color=Blue>无监督学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#无监督学习)：聚类, 降维, 异常检测, 生成模型

近些年出现各种不同的机器学习方法，包括但不限于：
- [<font color=Blue>集成学习</font>](https://0809zheng.github.io/2020/01/01/ML-outline.html#集成学习)：并行化, 序列化
- [强化学习](https://0809zheng.github.io/2020/01/01/ML-outline.html#强化学习)
- [<font color=Blue>推荐系统</font>](https://0809zheng.github.io/2020/05/08/recommender-system.html)
- [<font color=Blue>迁移学习</font>](https://0809zheng.github.io/2020/05/22/transfer-learning.html)
- [<font color=Blue>终身学习</font>](https://0809zheng.github.io/2020/05/21/lifelong-learning.html)
- [<font color=Blue>元学习</font>](https://0809zheng.github.io/2020/05/20/meta-learning.html)

# ⚪ 监督学习
**监督学习**(**Supervised Learning**)是指给计算机同时提供训练数据(**data**,也叫**样本sample**)及其标签(**label**,也称**ground truth**)，希望学习数据与标签之间的对应关系。

监督学习又可以分为**回归**问题和**分类**问题，实际上这类似于解决**连续性**问题和**离散性**问题。

## (1)回归
**回归(Regression)**的输出是连续值，范围为整个实数空间或其中一部分空间。
- [<font color=Blue>线性回归 LR</font>](https://0809zheng.github.io/2020/03/12/regression.html)：线性回归, 广义线性模型, 非线性回归
- [<font color=Blue>正则化的线性回归</font>](https://0809zheng.github.io/2020/03/30/ridge.html)：岭回归, 核岭回归, **LASSO**回归
- [<font color=Blue>前向逐步回归 SR</font>](https://0809zheng.github.io/2020/04/04/stagewise.html)：贪心地逐步试探每个回归参数
- [<font color=Blue>局部加权线性回归 LWLR</font>](https://0809zheng.github.io/2020/03/31/lwlr.html)：为每个训练样本赋予不同权重
- [<font color=Blue>偏最小二乘回归 PLS</font>](https://0809zheng.github.io/2020/04/05/pls.html)：构造互不相关且与标签相关的特征
- [<font color=Blue>Tube回归</font>](https://0809zheng.github.io/2020/03/29/tube.html)：引入中立区的线性回归
- [<font color=Blue>支持向量回归 SVR</font>](https://0809zheng.github.io/2020/03/15/support-vector-regression.html)：引入支持向量的**Tube**回归


## (2)分类
**分类(Classification)**的输出是离散值，把输入样本划分为有限个类别。

- [分类任务的常用性能指标](https://0809zheng.github.io/2020/02/07/classperform.html)：准确率指标,**P-R**曲线与**F1-score**,**ROC**曲线与**AUC**,代价曲线

根据输出范围的取值不同，分类可以划分为**硬分类(hard classify)**和**软分类(soft classify)**：
- **硬分类**：输出范围是$\{0,1\}$，将结果划分到某个具体的类别，包括：
1. [<font color=Blue>感知机 Perceptron</font>](https://0809zheng.github.io/2020/03/11/perceptron.html)：感知机学习算法(**PLA**), 口袋(**pocket**)算法
2. [<font color=Blue>k近邻算法 kNN</font>](https://0809zheng.github.io/2020/03/23/knn.html)
3. [<font color=Blue>线性判别分析 LDA</font>](https://0809zheng.github.io/2020/03/24/lda.html)：二分类**LDA**, 多分类**LDA**, 核**LDA**
4. [<font color=Blue>支持向量机 SVM</font>](https://0809zheng.github.io/2020/03/14/SVM.html)：线性**SVM**, 对偶**SVM**, 核**SVM**, 软间隔**SVM**, 概率**SVM**, 最小二乘**SVM**
5. [决策树](): 
- **软分类**：输出范围是$[0,1]$，给出每一个类别可能的概率，包括：
1. [<font color=Blue>逻辑回归 Logistic</font>](https://0809zheng.github.io/2020/03/13/logistic-regression.html): **Logistic**回归, 交叉熵损失, 核**Logistic**回归
2. [<font color=Blue>最大熵模型</font>](https://0809zheng.github.io/2021/07/20/me.html)
3. [高斯判别分析]()
4. [朴素贝叶斯]()

## (3)神经网络
**神经网络**(**Neural Network, NN**)是一类特殊的机器学习方法，通常是由多层感知机构成的。根据其输出层激活函数的选择不同，既可以用于回归又可以用于分类。神经网络又衍生出[深度学习](https://0809zheng.github.io/2020/01/02/DL-outline.html)这一领域。一些特殊的神经网络模型如下：
1. [前馈神经网络(多层感知机)](https://0809zheng.github.io/2020/04/17/feedforward-neural-network.html)
3. [径向基函数网络](https://0809zheng.github.io/2020/04/18/rbf-network.html)
4. [深度信念网络](https://0809zheng.github.io/2020/04/16/deep-belief-network.html)
5. [<font color=Blue>自组织映射网络 SOM</font>](https://0809zheng.github.io/2022/01/06/SOM.html)：竞争学习型的无监督神经网络

# ⚪ 无监督学习
**无监督学习(Unsupervised Learning)**是指提供给计算机的数据不再带有标签。此时的计算机不再受老师的监督，希望从无标签的数据中学习出有效的**特征**或**表示**。

常见的无监督学习方法：
- 聚类
- 降维：线性降维、非线性降维
- 异常检测
- 生成模型：隐变量模型、能量模型

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
- **非线性降维**则假设高维空间到低维空间的函数映射是非线性的。一种非线性降维方法是引入**核方法**，即先构造非线性的高维特征空间，再应用线性降维；另一种方法是**流形(manifold)学习**，即将高维空间中的流形张成一个低维空间，并保留数据的相互关系。

常见的降维方法：
1. [<font color=Blue>主成分分析 PCA</font>](https://0809zheng.github.io/2020/04/11/PCA.html):(线性降维) (几何,线性变换,最大投影方差,最小重构代价,奇异值分解)角度, 主坐标分析(**PCoA**), 概率**PCA**
2. [<font color=Blue>核主成分分析 KPCA</font>](https://0809zheng.github.io/2021/07/27/kpca.html)：(非线性降维) 将核方法引入**PCA**
3. [<font color=Blue>多维缩放 MDS</font>](https://0809zheng.github.io/2021/07/28/mds.html)：(线性降维) 维持近邻样本的距离
4. [<font color=Blue>局部保留投影 LPP</font>](https://0809zheng.github.io/2021/09/30/lpp.html)：(线性降维) 考虑样本的相对位置关系
5. [<font color=Blue>等度量映射 ISOMAP</font>](https://0809zheng.github.io/2021/07/30/isomap.html)：(非线性降维) 流形空间中的**MDS**
6. [<font color=Blue>局部线性嵌入 LLE</font>](https://0809zheng.github.io/2021/07/31/lle.html)：(非线性降维) 维持邻域内样本的线性关系
7. [<font color=Blue>t分布随机近邻嵌入 t-SNE</font>](https://0809zheng.github.io/2021/09/23/tsne.html)：(非线性降维) 维持样本间的概率分布
8. [<font color=Blue>一致流形近似与投影 UMAP</font>](https://0809zheng.github.io/2021/09/24/umap.html)：(非线性降维) 维持样本间的概率分布


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



# ⚪ 集成学习
**集成学习(Ensemble Learning)**是指构建多个子模型，并通过某种策略将它们结合起来，从而通过群体决策来提高决策准确率。若构建的子模型是同种类型的模型(如都是决策树)，则集成是**同质(homogeneous)**的，每个子模型被称为**基学习器(base learner)**，相应的学习算法称为**基学习算法(base learning algorithm)**；若构建的子模型是不同类型的模型，则集成是**异质(heterogenous)**的，每个子模型被称为**组件学习器(component learner)**。

通常希望构建的子模型具有一定的**准确率**(至少不差于**弱学习器**,即泛化性能略优于随机猜测的学习器)，又具有一定的**多样性**(即不同子模型之间具有一定的差异)。

- [<font color=Blue>集成学习中子模型的多样性(diversity)分析</font>](https://0809zheng.github.io/2021/07/22/ead.html): 误差-分歧分解, 多样性度量, 多样性增强
- [<font color=Blue>集成学习中不同子模型的结合(Blending)策略</font>](https://0809zheng.github.io/2020/03/16/blending.html): **voting, averaging, stacking**

根据**子模型的构建方式**，目前的集成学习方法可以分成两类：
- **并行化**集成方法：子模型之间不存在强依赖关系，可以同时生成，主要关注降低**方差**。如：
1. [<font color=Blue>Bagging</font>](https://0809zheng.github.io/2020/03/17/bagging.html)：使用**bootstrap**生成子数据集，训练子模型
1. [<font color=Blue>随机森林</font>](https://0809zheng.github.io/2020/03/20/random-forest.html)：决策树+**Bagging**+随机子空间算法
- **序列化**集成方法：子模型之间存在强依赖关系，必须串行生成，主要关注降低**偏差**。如：
1. [<font color=Blue>Boosting</font>](https://0809zheng.github.io/2020/03/18/boosting.html)：通过为样本权重重新赋值训练新的子模型，如**AdaBoost, Gradient Boosting**
1. [**提升树**](https://0809zheng.github.io/2021/07/24/btree.html)：决策树+**Boosting**
1. [**梯度提升决策树(GBDT)**](https://0809zheng.github.io/2020/03/21/GBDT.html)：决策树+**Gradient Boosting**



# ⚪ 强化学习
**强化学习(Reinforcement Learning, RL)**是指我们给模型一些输入，但是不提供希望的真实输出，根据模型的输出反馈，如果反馈结果接近真实输出，就给其正向激励；如果反馈结果偏离真实输出，就给其反向激励。不断通过**反馈-修正**这种形式，逐步让模型学习的更好。

- [多臂老虎机](https://0809zheng.github.io/2020/09/02/multiarm-bandit.html)


# ⚪ 相关课程
- [Machine Learning - Coursera （Andrew Ng）](https://www.coursera.org/learn/machine-learning)：监督学习（线性回归、逻辑回归、推荐系统），无监督学习（K-means聚类、PCA降维、异常检测），系统设计
- [李宏毅机器学习](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)![](https://pic1.imgdb.cn/item/634e994916f2c2beb10a65c6.jpg)
- [林轩田机器学习基石](https://www.bilibili.com/video/BV1Cx411i7op?p=1)：机器学习可行性（**Hoeffding**不等式、**VC**维），三个模型（感知机算法、线性回归、逻辑回归），三个技巧（特征转换、正则化、验证集）
- [林轩田机器学习技法](https://www.bilibili.com/video/BV1ix411i7yp?p=1)：核方法（支持向量机、支持向量回归），集成方法（集成，决策树、随机森林、**GBDT**），特征提取方法（神经网络、深度学习、径向基函数网络，推荐系统）
- [【机器学习】白板推导系列](https://space.bilibili.com/97068901/video)：B站up主“shuhuai008”自制课程，包括若干机器学习的理论推导
- [莫烦Python机器学习教程](https://mofanpy.com/)：B站up主“莫烦Python”自制课程，包含若干python库的入门（numpy、pandas、matplotlib、sklearn、Tensorflow、Keras、Pytorch）




### ⚪ 相关书籍

-  [《统计学习方法》(李航)](https://www.cnblogs.com/lishuairg/p/11734842.html)
- [《机器学习》(周志华)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLbook2016.htm)：西瓜书
- [《南瓜书》](https://datawhalechina.github.io/pumpkin-book/#/)：补充西瓜书中的部分公式推导细节
- [《Pattern Recognition and Machine Learning》(Bishop)](http://research.microsoft.com/~cmbishop/PRML)
- [《Hands-On Machine Learning with Scikit-Learn & TensorFlow》(Geron)](https://github.com/ageron/handson-ml)
