---
layout: post
title: '主动学习(Active Learning)'
date: 2022-08-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/630b213716f2c2beb1501d64.jpg'
tags: 深度学习
---

> Active Learning.


深度学习模型通常依赖大量标注数据来提供良好性能，然而数据的标注可能面临成本较高、预算有限、标注困难等问题。**主动学习 (Active Learning)**通过从未标注数据中选择一小部分样本进行标注和训练来降低标注成本；其出发点是并非所有样本对下游任务都是同等重要的，因此仅标注更“重要”的样本能够降低标注成本并最大化模型性能。

给定一个未标注的数据集$$\mathcal{U}$$和样本的标注上限$B$，主动学习旨在从数据集$$\mathcal{U}$$中选择至多包含$B$个样本的子集，使得在该子集上训练的模型比在其他数据上训练时的表现更好。

本文仅讨论适合于深度学习的主动学习方法。深度主动学习面临的困难包括：
1. 深度神经网络的预测结果往往是过度自信的，评估未标注数据的不确定性可能会不可靠。
2. 深度网络通常会处理大量数据，因此每次需要选择一批样本进而不是单个样本来进行标注。

深度主动学习中最常见的场景是基于**池**(**pool-based**)的主动学习，即从大量未标注的数据样本中迭代地选择最“有价值”的数据，直到性能达到指定要求或标注预算耗尽。

![](https://pic.imgdb.cn/item/630b213716f2c2beb1501d64.jpg)

选择最“有价值”的数据的过程被称为**采样策略(sampling strategy)**或**查询策略(query strategy)**，衡量数据“价值”的函数被称为**获取函数(acquisition function)**。

深度主动学习方法可以根据不同的**采样策略**进行分类：
- **不确定性采样 (uncertainty sampling)**：选择使得模型预测的不确定性最大的样本。不确定性的衡量可以通过机器学习方法(如**entropy**)、**QBC**方法(如**voter entropy**, **consensus entropy**)、贝叶斯神经网络(如**BALD**, **bayes-by-backprop**)、对抗生成(如**GAAL**, **BGADL**)、对抗攻击(如**DFAL**)、损失预测(如**LPL**)、标签预测(如**forgetable event**, **CEAL**)
- **多样性采样 (diversity sampling)**：选择更能代表整个数据集分布的样本。多样性的衡量可以通过聚类(如**core-set**, **Cluster-Margin**)、判别学习(如**VAAL**, **CAL**, **DAL**)
- **混合策略 (hybrid strategy)**：选择既具有不确定性又具有代表性的样本。样本的不确定性和代表性既可以同时估计(如**exploration-exploitation**, **BatchBALD**, **BADGS**, **Active DPP**, **MAL**)，也可以分两阶段估计(如**Suggestive Annotation**, **DBAL**)。


## 1. 基于不确定性的采样策略 Uncertainty-based Sampling Strategy

基于**不确定性(uncertainty)**的采样策略是指在采样时选择具有较高不确定性的数据样本。深度学习中的[<font color=blue>不确定性</font>](https://0809zheng.github.io/2022/08/02/uncertainty.html)包括偶然不确定性(数据生成过程的固有误差)和认知不确定性(由训练数据的缺乏导致)。


### ⚪ 经典机器学习方法

一些基于不确定性的深度主动学习方法继承于机器学习技术中基于池的主动学习方法(以分类任务为例)。


| 方法 | 采样策略 | 获取函数 |
| :---: | :---:  |  :---:  |
| 最大熵(**maximum entropy**) | 选择预测熵最大的样本 | $$H(y\|x)=-\sum_k p(y=k\|x) \log p(y=k\|x)$$ |
| 间隔(**margin**) | 选择预测结果中排名靠前两个类别的预测概率之差最小的样本 | $$-[p(\hat{y}_1\|x)-p(\hat{y}_2\|x)]$$ |
| 最小置信度(**least confidence**) | 选择预测结果中排名最前类别的预测概率最小的样本 | $$-p(\hat{y}\|x)$$ |
| 变差比(**Variation Ratio**) | 与最小置信度类似，衡量置信度的缺乏 | $$1-p(\hat{y}\|x)$$ |
| 平均标准偏差(**mean standard deviation**) | 选择所有预测类别的平均标准偏差最大的样本 | $$\frac{1}{k}\sum_k \sqrt{\text{Var}[p(y=k\|x)]}$$ |


### ⚪ Query-By-Committee (QBC)

在经典机器学习方法中，也可以训练一系列不同的模型(如设置不同的随机数种子或超参数)，通过给定样本在不同模型之间的输出变化来估计不确定性；这种方法称为**委员会查询(query-by-committee, QBC)**。

若共训练$N$个模型，每个模型的预测结果为$p_1,...,p_N$，则基于**QBC**的获取函数包括：
- 选民熵(**voter entropy**)：计算标签$y$的票数结果的熵：$H(\frac{\text{vote}(y)}{N})$
- 一致熵(**consensus entropy**)：计算平均预测结果的熵：$H(\frac{\sum_n p_n}{N})$
- **KL**散度：计算每个模型预测结果与平均预测结果的**KL**散度：$\frac{1}{N}\sum_n KL[p_n\|\|\frac{\sum_n p_n}{N}]$


### ⚪ [<font color=Blue>Bayesian Active Learning by Disagreement (BALD)</font>](https://0809zheng.github.io/2022/08/03/bald.html)

使用贝叶斯神经网络能够对模型的认知不确定度进行较为准确的估计。贝叶斯神经网络的网络参数是从分布中采样得到的；在进行推断时，通过对参数分布进行积分来集成无穷多的神经网络进行预测。

在实践中采用蒙特卡洛**dropout**进行近似。具体地，在测试时通过多次应用**dropout**，并将结果取平均，便可以实现蒙特卡洛积分近似。**BALD**在此基础上选择模型输出和模型参数的互信息最大的样本：

$$ I(y;w | x,D_{train}) ≈ -\sum_c (\frac{1}{T}\sum_t p_c^t)\log (\frac{1}{T}\sum_t p_c^t) +\frac{1}{T}\sum_{t,c} p_c^t\log p_c^t $$


### ⚪ [<font color=Blue>bayes-by-backprop</font>](https://0809zheng.github.io/2022/08/05/bbb.html)


**bayes-by-backprop**方法通过把网络权重建模为一个变分分布$q(w\|\theta)$来直接估计认知不确定性。构造变分分布参数$\theta$的损失函数：

$$ \mathcal{L}(\theta) =  \log q(w|\theta) - \log p(w)p(D|w) $$

将变分分布$q(w\|\theta)$设置为对角高斯分布，通过梯度更新参数$\theta$，进而构造网络权重。认知不确定性可以通过推断时采样不同的模型参数来计算。

### ⚪ [<font color=Blue>Generative Adversarial Active Learning (GAAL)</font>](https://0809zheng.github.io/2022/08/12/gaal.html)

**GAAL**通过生成对抗网络生成新的样本进行标记和训练，在生成的样本中采样距离决策边界较近的样本(决策边界由支持向量机提供)。

![](https://pic.imgdb.cn/item/631fe54316f2c2beb1163243.jpg)

### ⚪ [<font color=Blue>Bayesian Generative Active Deep Learning (BGADL)</font>](https://0809zheng.github.io/2022/08/13/bgadl.html)

**BGADL**通过**BALD**方法选择互信息较大的未标注样本，通过**VAE-ACGAN**模型实现样本生成、样本判别和类别预测。

![](https://pic.imgdb.cn/item/63202eec16f2c2beb15ef233.jpg)


### ⚪ [<font color=Blue>Deep-Fool Active Learning (DFAL)</font>](https://0809zheng.github.io/2022/08/11/dfal.html)

**DFAL**利用对抗攻击衡量样本的不确定性。对抗攻击的出发点是寻找最小的扰动以跨越决策边界，因此选择与其对抗样本距离较小的样本，并对原样本和对抗样本采用相同的标签进行标注。

![](https://pic.imgdb.cn/item/631e9c9b16f2c2beb1f31099.jpg)

### ⚪ [<font color=Blue>Loss Prediction Loss (LPL)</font>](https://0809zheng.github.io/2022/08/04/loss.html)

**损失预测(loss prediction)**的基本思想是预测未标注样本的损失，更高的损失意味着该样本学习更困难、更值得标注。

![](https://pic.imgdb.cn/item/6318881c16f2c2beb123f731.jpg)

### ⚪ [<font color=Blue>Forgetable Event</font>](https://0809zheng.github.io/2022/08/10/event.html)

遗忘事件记录了样本在训练阶段的预测标签的变化情况。如果模型在训练过程中改变预测结果则表明模型对该样本的不确定性较大，因此在采样时选择标签变化次数较多的可遗忘样本。

### ⚪ [<font color=Blue>Cost-Effective Active Learning (CEAL)</font>](https://0809zheng.github.io/2022/08/16/ceal.html)

**CEAL**是一种结合主动学习和自监督学习的框架。一方面通过主动学习选择具有高不确定性的样本进行人工标注，另一方面选择具有较高预测置信度的样本并为它们指定伪标签进行特征学习。

![](https://pic.imgdb.cn/item/63213e1016f2c2beb15d8e2b.jpg)


## 2. 基于多样性的采样策略 Diversity-based Sampling Strategy


基于**多样性(diversity)**的采样策略也称为基于**代表性(representive)**的采样策略，是指在采样时选择更能代表整个数据集分布的数据样本。多样性采样在选择数据样本时通常根据样本之间的相似度进行衡量。

### ⚪ [<font color=Blue>Core-Set</font>](https://0809zheng.github.io/2022/08/08/coreset.html)

**core-set**方法是指选择$b$个样本加入训练集后，未标注样本池中的任意样本与其距离最近的已标注样本的距离最小化。实现时常采用贪心算法：每次选择一个与已标注样本集的距离最大的样本进行标注，重复$b$次。

![](https://pic.imgdb.cn/item/631aa9b416f2c2beb1315394.jpg)

### ⚪ [<font color=Blue>Cluster-Margin</font>](https://0809zheng.github.io/2022/08/18/cm.html)

**Cluster-Margin**首先为未标注样本集应用具有平均链接的层次聚集聚类算法生成聚类$C$。然后选择**Margin**得分最低的$k_m$个样本，根据聚类$C$将其划分到不同的聚类簇中。最后采用循环采样的方式从每个簇中采样样本，直至选择$k_t$个样本。

![](https://pic.imgdb.cn/item/63218c3a16f2c2beb1b1bee2.jpg)


### ⚪ [<font color=Blue>Variational Adversarial Active Learning (VAAL)</font>](https://0809zheng.github.io/2021/12/02/VAAL.html)

**VAAL**使用$\beta$**-VAE**模型将样本嵌入到低维特征空间，并使用一个判别器区分已标注样本和未标注样本的特征。训练完成后，选择判别器预测概率得分最小的一批未标注样本补充到标注池中，因为这些样本与已标注样本具有最小的特征相关性。

![](https://pic.imgdb.cn/item/61a71dac2ab3f51d91a58f81.jpg)

### ⚪ [<font color=Blue>Contrastive Active Learning (CAL)</font>](https://0809zheng.github.io/2022/08/07/cal.html)

**CAL**选择对比得分较高的样本。对比得分是未标注样本$x$与其特征空间中距离最近的$k$个标注样本$$\{x^l\}$$特征之间的平均**KL**散度，即选择与不同标签的样本具有相似特征的新样本。

![](https://pic.imgdb.cn/item/631a9ebb16f2c2beb12580df.jpg)

### ⚪ [<font color=Blue>Discriminative Active Learning (DAL)</font>](https://0809zheng.github.io/2022/08/21/dal.html)

**DAL**把主动学习建模为未标注类别$$\mathcal{U}$$和已标注类别$$\mathcal{L}$$的二分类问题。训练一个二分类器$\Psi$使其能成功区分未标注样本和已标注样本，然后选择未标注类别得分最大的前$K$个样本：

$$ \mathop{\arg \max}_{x \in \mathcal{U}} \hat{P} (y=u|\Psi(x)) $$

## 3. 混合策略 Hybrid Strategy

**混合策略(Hybrid Strategy)**是指在样本的不确定性和多样性之间进行权衡，选择既具有不确定性又具有较强代表性的样本。


### ⚪ [<font color=Blue>Exploration-Exploitation</font>](https://0809zheng.github.io/2022/08/17/ee.html)

**exploration-exploitation**采用加权求和的方法同时考虑样本的不确定性得分和多样性得分。**开发(exploitation)**阶段旨在选择具有最大不确定性和最小冗余度的样本；**探索(exploration)**阶段旨在选择距离已标注样本集最远的样本。

### ⚪ [<font color=Blue>BatchBALD</font>](https://0809zheng.github.io/2022/08/20/batchbald.html)

**BatchBALD**将**BALD**扩展为批量形式，其获取函数为一个批量样本的模型输出与模型参数之间的互信息：

$$ \begin{aligned} I(y_1,\cdots,y_B;w | x_1,\cdots,x_B,D_{train}) =& H(y_1,\cdots,y_B|x_1,\cdots,x_B,D_{train}) \\ &- E_{p(w|D_{train})}[H(y_1,\cdots,y_B|x_1,\cdots,x_B,w,D_{train})] \end{aligned} $$

### ⚪ [<font color=Blue>Batch Active learning by Diverse Gradient Embedding (BADGE)</font>](https://0809zheng.github.io/2022/08/09/badge.html)

**BADGE**把样本映射到网络最后一层参数的梯度空间中，同时捕捉模型的不确定性和数据样本的多样性。其中不确定性通过梯度的量级衡量；多样性通过$k$-**means**++算法在梯度空间中选择样本点。

### ⚪ [<font color=Blue>Active DPP</font>](https://0809zheng.github.io/2022/08/19/adpp.html)

**Active DPP**使用行列式点过程(**DPP**)建模样本的不确定性和多样性。**DPP**是一类排斥点过程，可以用于生成不同批次的样本。作者使用不确定性**DPP**根据学习模型提供的不确定性得分选择数据样本，并使用探索**DPP**寻找新的决策边界。

![](https://pic.imgdb.cn/item/632a715e16f2c2beb18d3f69.jpg)

### ⚪ [<font color=Blue>Minimax Active Learning (MAL)</font>](https://0809zheng.github.io/2022/08/06/mal.html)

**MAL**最小化特征编码网络$F$的熵使得具有相似预测标签的样本具有相似的特征；最大化分类器$C$的熵使得预测结果为更均匀的类别分布；判别器$D$则试图有效地区分标记样本和未标记样本。在采样时通过判别器$D$的得分衡量样本的多样性，通过分类器$C$的熵衡量样本的不确定性。

![](https://pic.imgdb.cn/item/631a8c5816f2c2beb114235b.jpg)

### ⚪ [<font color=Blue>Suggestive Annotation</font>](https://0809zheng.github.io/2022/08/14/sa.html)

**Suggestive Annotation**首先选择一批具有较高不确定性的样本，再从中选择具有较高代表性的样本进行标注。其中不确定性是利用在标注样本集上训练的集成模型来估计的，而代表性是通过核心集方法进行选择的。

![](https://pic.imgdb.cn/item/632049d816f2c2beb17cf058.jpg)

### ⚪ [<font color=Blue>Diverse mini-batch Active Learning (DBAL)</font>](https://0809zheng.github.io/2022/08/15/dbal.html)

**DBAL**首先选择一批具有较高不确定性的样本，再从中选择具有较高代表性的样本进行标注。其中不确定性是通过加权$k$**-means**算法进行选择的，而代表性是通过$k$**-means**算法进行选择的。


Wasserstein对抗性AL（WAAL）[63]通过H-散度的对抗性训练搜索多样性未标记批次，该批次也具有比标记样本更大的多样性。

两阶段（多阶段）优化 WAAL使用两阶段优化，通过在阶段1中训练用于鉴别特征的DNN和在阶段2中进行批量选择来实现鉴别学习[63]。

## ⚪ 参考文献

- [Learning with not Enough Data Part 2: Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/)：Blog by Lilian Weng.
- [Overview of Active Learning for Deep Learning](https://jacobgil.github.io/deeplearning/activelearning)：Blog by Jacob Gildenblat.
- [Awesome Active Learning](https://github.com/SupeRuier/awesome-active-learning#awesome-active-learning)：(github) Hope you can find everything you need about active learning in this repository.
- [A Comparative Survey of Deep Active Learning](https://arxiv.org/abs/2203.13450)：(arXiv2203)一篇深度主动学习的综述.
- [<font color=Blue>Weight Uncertainty in Neural Networks</font>](https://0809zheng.github.io/2022/08/05/bbb.html)：(arXiv1505)神经网络中的权重不确定性。
- [<font color=Blue>Cost-Effective Active Learning for Deep Image Classification</font>](https://0809zheng.github.io/2022/08/16/ceal.html)：(arXiv1701)CEAL：用于深度图像分类的高性价比主动学习。
- [<font color=Blue>Generative Adversarial Active Learning</font>](https://0809zheng.github.io/2022/08/12/gaal.html)：(arXiv1702)GAAL：生成对抗主动学习。
- [<font color=Blue>What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?</font>](https://0809zheng.github.io/2022/08/02/uncertainty.html)：(arXiv1703)使用贝叶斯深度学习建模深度学习中的不确定性。
- [<font color=Blue>Deep Bayesian Active Learning with Image Data</font>](https://0809zheng.github.io/2022/08/03/bald.html)：(arXiv1703)BALD：贝叶斯不一致主动学习。
- [<font color=Blue>Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation</font>](https://0809zheng.github.io/2022/08/14/sa.html)：(arXiv1706)暗示标注：一种用于生物医学图像分割的深度主动学习框架。
- [<font color=Blue>Active Learning for Convolutional Neural Networks: A Core-Set Approach</font>](https://0809zheng.github.io/2022/08/08/coreset.html)：(arXiv1708)基于核心集的主动学习方法。
- [<font color=Blue>Deep Similarity-Based Batch Mode Active Learning with Exploration-Exploitation</font>](https://0809zheng.github.io/2022/08/17/ee.html)：(ICDM2017)通过探索-开发实现基于深度相似性的批处理主动学习。
- [<font color=Blue>Adversarial Active Learning for Deep Networks: a Margin Based Approach</font>](https://0809zheng.github.io/2022/08/11/dfal.html)：(arXiv1802)DFAL：一种基于决策边界的对抗主动学习方法。
- [<font color=Blue>Diverse mini-batch Active Learning</font>](https://0809zheng.github.io/2022/08/15/dbal.html)：(arXiv1901)DBAL：多样性小批量主动学习。
- [<font color=Blue>Bayesian Generative Active Deep Learning</font>](https://0809zheng.github.io/2022/08/13/bgadl.html)：(arXiv1904)BGADL：贝叶斯生成深度主动学习。
- [<font color=Blue>Variational Adversarial Active Learning</font>](https://0809zheng.github.io/2021/12/02/VAAL.html)：(arXiv1904)VAAL: 变分对抗主动学习。
- [<font color=Blue>Learning Loss for Active Learning</font>](https://0809zheng.github.io/2022/08/04/loss.html)：(arXiv1905)主动学习中的损失预测。
- [<font color=Blue>BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning</font>](https://0809zheng.github.io/2022/08/20/batchbald.html)：(arXiv1906)BatchBALD：深度贝叶斯主动学习的高效多样性批量获取。
- [<font color=Blue>Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds</font>](https://0809zheng.github.io/2022/08/09/badge.html)：(arXiv1906)BADGE：基于多样性梯度嵌入的批量主动学习。
- [<font color=Blue>Batch Active Learning Using Determinantal Point Processes</font>](https://0809zheng.github.io/2022/08/19/adpp.html)：(arXiv1906)Active DPP：基于行列式点过程的批量主动学习。
- [<font color=Blue>Discriminative Active Learning</font>](https://0809zheng.github.io/2022/08/21/dal.html)：(arXiv1907)DAL：判别式主动学习。
- [<font color=Blue>Minimax Active Learning</font>](https://0809zheng.github.io/2022/08/06/mal.html)：(arXiv2012)MAL：最小最大主动学习。
- [<font color=Blue>When Deep Learners Change Their Mind: Learning Dynamics for Active Learning</font>](https://0809zheng.github.io/2022/08/10/event.html)：(arXiv2107)基于遗忘事件的主动学习。
- [<font color=Blue>Batch Active Learning at Scale</font>](https://0809zheng.github.io/2022/08/18/cm.html)：(arXiv2107)Cluster-Margin：一种大规模批量主动学习方法。
- [<font color=Blue>Active Learning by Acquiring Contrastive Examples</font>](https://0809zheng.github.io/2022/08/07/cal.html)：(arXiv2109)CAL：对比主动学习。