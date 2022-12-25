---
layout: post
title: '决策树(Decision Tree)'
date: 2020-03-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63a2e4feb1fccdcd36e7094a.jpg'
tags: 机器学习
---

> Decision Tree.

**决策树(decision tree)**是一种对训练数据集$(X,y)$进行划分的树形结构算法。决策树既可以看作一个**if**-**then**规则的集合，其中的规则互斥且完备；又可以看作描述训练数据集的条件概率分布$P(y \| X)$。

一棵完整的决策树是由结点(**node**)和有向边(**directed edge**)组成的。结点包括内部(**internal**)结点和叶(**leaf**)结点：其中内部结点对输入数据的某个特征维度进行条件判断，叶结点作为决策树的某一路输出。有向边用于把输入数据划分到不同的分支(**branch**)。

决策树的基本算法包含了三个选择：
- **分支个数(number of branches)**：根据每个结点的分支个数可以分为二叉树(**bi-branch**)和多叉树(**multi-branch**)。
- **分支条件(branching criteria)**：也称为**不纯度(impurity)**，用于确定对数据的哪一个特征维度进行选择，通常与数据的熵相关；对于分类任务可以选择信息增益、信息增益比、基尼指数；对于回归任务通常设置为均方误差。
- **终止条件(termination criteria)**：通常为用完所有特征或子集中数据标签全部相同，也可根据迭代次数、深度要求或不纯度的阈值要求进行设置。

决策树可以表示成递归形式：

$$ G(x) = \sum_{n=1}^{N} {[b(x)=n]G_n(x)} $$

其中$N$表示每一个结点的分支数，根据条件进入不同的子树；$b(x)$是分支函数(**branching tree**)，用来决定数据前往哪个分支。

决策树算法递归地根据分支条件进行特征选择，然后把训练数据集中的数据划分到不同的分支中，每一个数据最终会落入一个叶结点中。当给定一个测试数据时，判断其所属的叶结点，并输出对应叶结点的输出值：对于分类取该结点中出现最多的数据类别；对于回归取该结点中数据标签的均值。

根据设置的分支个数和分支条件不同，决策树算法体现为不同的形式：

| 决策树 | 分支个数 | 分支条件 |
| :---: | :---: | :---:  |
| **ID3** | 多叉树 | 信息增益 |
| **C4.5** | 多叉树 | 信息增益比 |
| **CART** | 二叉树 | (分类): 基尼指数 <br> (回归): 均方误差 |



# 1. ID3

**ID3**是一种以信息增益作为分支条件的多叉决策树。

## (1) 信息增益

**信息增益(information gain)**定义为给定特征$A$时，数据集$D$的不确定性减少的程度（也即为互信息）：

$$ g(D,A) = H(D) - H(D|A) $$

其中$H(D)$表示数据集$D$的经验熵(**empirical entropy**)，用于衡量数据集$D$的标签的不确定性。在计算时首先统计每个标签$c$的频数，然后计算标签频率分布的熵：

$$ H(D) = -\sum_{c=1}^C \frac{|D(y=c)|}{|D|} \log \frac{|D(y=c)|}{|D|} $$

$H(D\|A)$表示特征$A$对数据集$D$的经验条件(**conditional**)熵，用于衡量已知特征$A$的条件下数据集$D$的标签的不确定性。在计算时首先构造特征$A$的每种取值情况下的经验熵，然后按照特征取值的频率进行加权：

$$ \begin{aligned} H(D|A) &= \sum_{k=1}^{K} \frac{|D_k|}{|D|} H(D_k) \\ &= -\sum_{k=1}^{K} \frac{|D_k|}{|D|} \sum_{c=1}^C \frac{|D_k(y=c)|}{|D_k|} \log \frac{|D_k(y=c)|}{|D_k|} \end{aligned} $$

## (2) ID3的算法流程

给定训练数据集$D$、数据的特征集$A$和阈值$\epsilon$：
1. 如果数据集$D$中所有样本均属于同一类$c_0$，则决策树$T$为单结点树，并将$c_0$作为该结点的类标记，返回$T$；
2. 如果特征集$A=\Phi$，则$T$为单结点树，并将$D$中出现次数最多的类$c_{\max}$作为该结点的类标记，返回$T$；
3. 计算特征集$A$中每一个特征的信息增益，选择信息增益最大的特征$A_{\max}$；
4. 如果$A_{\max}$的信息增益小于阈值$\epsilon$，则$T$为单结点树，并将该结点中出现次数最多的类$c_{\max}'$作为该结点的类标记，返回$T$；
5. 对于$A_{\max}$的每一种可能取值$a_k$，按照$A_{\max}=a_k$把$D$分割成若干非空子集$D_k$，将$D_k$中出现次数最多的类$c_k$作为该类标记构造子结点；
6. 对第$k$个子结点，以$D_k$为训练集，以$A-A_{\max}$为特征集，递归地调用**1**-**5**，得到并返回子树$T_k$。

## (3) 实现ID3

```python
import math
import operator

class DisicionTree():
    def __init__(self):
        self.myTree = {}
       
    # 递归构建决策树
    def fit(self, dataSet, feature_labels):
        # 取出类别标签
        classList = [example[-1] for example in dataSet]            
        # 如果类别完全相同则停止继续划分
        if classList.count(classList[0]) == len(classList):           
            return classList[0]
        # 遍历完所有特征时返回出现次数最多的类标签
        if len(dataSet[0]) == 1 or len(feature_labels) == 0:                
            return self.majorityCnt(classList)
        # 获取最优特征维度    
        bestFeat = self.chooseBestFeatureToSplit(dataSet)                
        # 得到最优特征标签
        bestFeatLabel = feature_labels[bestFeat]
        # 根据最优特征的标签生成树
        self.myTree.update({bestFeatLabel:{}})
        # 删除已经使用特征标签
        del(feature_labels[bestFeat])
        # 得到训练集中所有最优特征维度的所有属性值
        featValues = [example[bestFeat] for example in dataSet]       
        uniqueVals = set(featValues)
        # 遍历特征，创建决策树
        for value in uniqueVals:
            subLabels = feature_labels[:]
            self.myTree[bestFeatLabel][value] = self.fit(
                dataSet = self.splitDataSet(dataSet, bestFeat, value),
                feature_labels = subLabels,
                )
        return self.myTree

    # 返回classList中出现次数最多的元素
    def majorityCnt(self, classList):
        classCount = {}
        keys = set(classList)
        for key in keys:
            classCount[key] = classList.count(key)
        #根据字典的值降序排序
        sortedClassCount = sorted(classCount.items(), 
                                  key = operator.itemgetter(1), 
                                  reverse = True)  
        return sortedClassCount[0][0]   

    # 根据信息增益选择特征
    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1     # 特征数量
        baseEntropy = self.calcShannonEnt(dataSet) # 计算数据集的经验熵
        bestInfoGain = 0.0                    # 信息增益
        bestFeature = -1                      # 最优特征的索引值
        for i in range(numFeatures): 
            #获取所有数据样本的第i个特征
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)        # 第i个特征的所有取值
            newEntropy = 0.0                  # 经验条件熵
            #计算第i个特征的信息增益
            for value in uniqueVals:
                # 按照第i个特征拆分数据集
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            # 寻找信息增益最大的特征
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 计算经验熵(香农熵)
    def calcShannonEnt(self, dataSet):
        # 返回数据集的样本数
        numEntires = len(dataSet)       
        # 收集所有目标标签 （最后一个维度）
        labels= [featVec[-1] for featVec in dataSet]     
        # 去重、获取标签种类
        keys = set(labels)
        shannonEnt = 0.0
        for key in keys:
           # 计算每种标签出现的次数
           prob = float(labels.count(key)) / numEntires 
           shannonEnt -= prob * math.log(prob, 2) 
        return shannonEnt

    # 数据集分割
    def splitDataSet(self, dataSet, axis, value):        
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                # 去除数据的第i个特征
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    # 进行新数据的分类
    def predict(self, inputTree, feature_labels, testVec):
        # 获取决策树结点
        firstStr = next(iter(inputTree))
        # 获取子树
        secondDict = inputTree[firstStr]
        featIndex = feature_labels.index(firstStr)
        # 把数据划分到不同的分支                                   
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                # 如果对应子树
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.predict(
                        inputTree = secondDict[key],
                        testVec = testVec
                        )
                # 如果对应叶结点
                else: classLabel = secondDict[key]
        return classLabel


# 创建数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有房子', '信贷情况']        #特征标签
    return dataSet, labels                             #返回数据集和分类属性

if __name__ == '__main__':
    # 获取数据集
    dataSet, labels = createDataSet()
    feature_labels = labels[:]
    dt = DisicionTree()
    myTree = dt.fit(dataSet, labels)
    print(myTree) # {'有房子': {0: {...}, 1: 'yes'}, '有工作': {0: 'no', 1: 'yes'}}
    
    # 测试
    testVec = [0,1,1,2] 
    result = dt.predict(myTree, feature_labels, testVec)
    print(result) # yes
```

# 2. C4.5

**C4.5**模型与**ID3**模型类似，主要区别在于使用信息增益比作为分支条件。

数据集$D$在特征$A$上的**信息增益比(information gain radio)**定义为信息增益与条件熵的比值：

$$  g_R(D,A) = \frac{g(D,A)}{H(D)} $$


# 3. CART

**CART (classification and regression tree)**是一种**二叉树**形式的决策树算法，既可以用于分类也可以用于回归。**CART**算法简单，容易实现；具有具有良好的可解释性；并且可以处理特征缺失的情况。对于每一个内部节点，**CART**使用**决策桩算法（decision stump）**。

## (1) 决策桩 Decision Stump
决策桩算法可以看做一维的感知机，即通过一定的准则将数据按照某一个特征维度分成两份：若数据的第$i$个特征不超过$θ$，则前往右边的子树；否则前往左边的子树：

$$ b(x) = [x_i≤θ] + 1 $$

一个好的决策桩对数据进行划分后，应使每一组子数据集内的不纯度最小（即设定合适的分支条件）。根据设置的分支条件不同，**CART**可以分别应用于回归和分类问题。

## (2) 回归CART：最小二乘回归树 least squares regression tree

最小二乘回归树把输入空间划分为$M$个互不相交的子区域$R_1,...,R_M$，计算每个子区域$R_m$上的输出值$C_m$，从而构造回归树模型：

$$ f(x) = \sum_{m=1}^M C_m \cdot I(x \in R_m) $$

![](https://pic.imgdb.cn/item/63a7aa4708b683016376356e.jpg)

最小二乘回归树根据平方误差最小化准则选择最优划分特征：

$$ \min \sum_{x_i \in R_m} (y_i - f(x_i))^2 $$

具体地，对输入数据$x$选择一个切分特征$d$及其对应的切分点$p$，将空间递归地划分为两个子空间：

$$ R_1(d,p) = \{ x | x_d \leq p \},\quad R_2(d,p) = \{ x | x_d  > p \} $$

并将每个子空间的输出值设定为子空间内所有数据的平均标签值：

$$ C_1 = \frac{1}{|R_1(d,p)|} \sum_{x \in R_1(d,p)}y,\quad C_2 = \frac{1}{|R_2(d,p)|} \sum_{x \in R_2(d,p)}y $$

最优切分特征$d$及最优切分点$p$的选择通过求解：

$$ \mathop{\min}_{d,p} [\mathop{\min}_{C_1} \sum_{x \in R_1(d,p)} (y-C_1)^2 +  \mathop{\min}_{C_2} \sum_{x \in R_2(d,p)} (y-C_2)^2  ] $$

在实践中首先遍历特征维度$d$；对于每个特征维度，遍历所有可能的切分点$p$；直至找到使得上式最小的$(d,p)$值。

### ⚪ 回归CART的例子

已知如图所示的训练数据，试用平方误差损失准则生成一个二叉回归树。

![](https://pic.imgdb.cn/item/63a7aab208b683016376e329.jpg)

该数据集只有一个切分变量$x$，分别选择$p=1,...,10$为切分点，把数据分成左右两份。对于每个切分点分别计算两个子集的平均输出$C_1,C_2$，并进一步计算平方误差：

![](https://pic.imgdb.cn/item/63a7ac1c08b6830163793652.jpg)

从结果可得$p=5$时平方误差最小，因此选择第一个最优切分点为$p_1=5$，划分成两个子数据集$R_1=\{1,2,3,4,5\}$和$R_2=\{6,7,8,9,10\}$，两个子数据集的输出分别为$\hat{c}_1=5.06,\hat{c}_2=8.18$。

递归地执行上述过程，即可构造二叉回归树：

![](https://pic.imgdb.cn/item/63a7ace008b68301637a84e3.jpg)

![](https://pic.imgdb.cn/item/63a7ad2608b68301637af5be.jpg)

## (3) 分类CART

对于分类问题，**CART**使用**基尼指数（Gini index）**作为分支条件。基尼指数越大，表明数据集的不确定性越大。

若分类问题共有$K$个类别，且某一样本属于第$k$个类别的概率为$p_k$，则该样本的基尼指数为：

$$ \text{Gini}(p) = \sum_{k=1}^Kp_k(1-p_k) = 1-\sum_{k=1}^Kp_k^2 $$

对于二分类问题（$K=2$），若记某一类样本所占总样本的比例为$μ$，另一类所占比例为$1-μ$，则基尼指数为：

$$ \text{Gini}(μ) = 2μ(1-μ) $$

若样本集$D$共有$N$个样本，则该样本集的基尼指数为：

$$ \text{Gini}(D) =  1 - \sum_{k=1}^{K} {(\frac{\sum_{n=1}^{N} {[y_n=k]}}{N})^2} $$

若样本集$D$根据特征$A$可以进行划分$D=(D_1,D_2)$，则在特征$A$下样本集的基尼指数为：

$$ \text{Gini}(D,A) =  \frac{|D_1|}{N}\text{Gini}(D_1)+\frac{|D_2|}{N}\text{Gini}(D_2) $$

在构造分类**CART**时，根据数据集$D$的每一个特征$A$的每一个取值情况把$D$划分成两部分$(D_1,D_2)$，计算此时的基尼指数$\text{Gini}(D,A)$。遍历所有特征及其所有可能的切分点，选择基尼指数最小的作为最优特征及最优切分点，把数据集划分成两部分；递归地执行上述操作，直至满足停止条件。

## (4) 处理缺失值 Handle missing features
**CART**可以处理预测时缺失特征的情况。一种常用的方法就是代理分支**(surrogate branch)**，即寻找与每个特征相似的替代特征。确定是相似特征的做法是在决策树训练的时候，如果存在一个特征与当前特征切分数据的方式和结果是类似的，则表明二者是相似的，就把该替代的特征也存储下来。当预测时遇到原特征缺失的情况，就用替代特征进行分支判断和选择。

## (5) 实现CART

下面以回归**CART**为例，给出实现过程：

### ⚪ 回归CART from scratch

```python
import numpy as np

class RegressionTree():
    def __init__(self):
        self.myTree = {}
        
    # 构建tree
    def fit(self, dataSet):
        feat, val = self.chooseBestSplit(dataSet)
        if feat == None: return val  # 满足停止条件时返回叶结点值
        # 切分后赋值
        self.myTree['spInd'] = feat
        self.myTree['spVal'] = val
        # 切分后的左右子树
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        self.myTree['left'] = self.fit(lSet)
        self.myTree['right'] = self.fit(rSet)
        return self.myTree

    # 二元切分
    def chooseBestSplit(self, dataSet, ops=(0, 1)):
        # 切分特征的参数阈值，用户初始设置好
        tolS = ops[0]  # 允许的误差下降值
        tolN = ops[1]  # 切分的最小样本数
        # 若所有特征值都相同，停止切分
        if len(set(dataSet[:, -1].T.tolist())) == 1:  # 标签值查重
            return None, np.mean(dataSet[:, -1])  # 使用标签均值生成叶结点
        m, n = np.shape(dataSet)
        # 计算数据集的平方误差（均方误差*总样本数）
        S = np.var(dataSet[:, -1]) * m
        bestS = inf
        bestIndex = 0
        bestValue = 0
        # 遍历数据的每个属性特征
        for featIndex in range(n - 1):
            # 遍历每个特征里不同的特征值
            for splitVal in set((dataSet[:, featIndex].T.tolist())):
                # 对每个特征进行二元切分
                subset1, subset2 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                if (subset1.shape[0] < tolN) or (subset2.shape[0] < tolN): continue
                S1 = np.var(subset1[:, -1]) * subset1.shape[0]
                S2 = np.var(subset2[:, -1]) * subset2.shape[0]
                newS = S1+S2
                # 更新为误差最小的特征
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        # 如果切分后误差效果下降不大，则取消切分，直接创建叶结点
        if (S - bestS) < tolS:
            return None, np.mean(dataSet[:, -1])
        subset1, subset2 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        # 判断切分后子集大小，小于最小允许样本数停止切分
        if (subset1.shape[0] < tolN) or (subset2.shape[0] < tolN):
            return None, np.mean(dataSet[:, -1])
        return bestIndex, bestValue  # 返回特征编号和用于切分的特征值

    # 切分数据集为两个子集
    def binSplitDataSet(self, dataSet, feature, value):
        subset1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
        subset2 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
        return subset1, subset2

    
if __name__ == "__main__":
    x = np.array(list(range(1, 11))).reshape(-1, 1)
    y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00]).reshape(-1, 1)
    myDat = np.concatenate([x,y], 1)
    myTree = RegressionTree()
    print(myTree.fit(myDat))
    """
    {'spInd': 0, 'spVal': 2.0, 'left': {...}, 'right': {...}}
    """
```

### ⚪ 回归CART from sklearn

使用**sklearn**库可以便捷地实现二叉回归树：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Data set
x = np.array(list(range(1, 11))).reshape(-1, 1) # [n, d]
y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00]).ravel() # [n,]

# Fit regression model
model = DecisionTreeRegressor(max_depth=3)
model.fit(x, y)

# Predict
X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis] # [m, d]
y_test = model.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_test, color="cornflowerblue",
         label="max_depth=3", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
plt.show()
```

# 4. 决策树的剪枝

直接生成的决策树被称为**fully-grown tree**，最终会在观测样本集上实现零误差，并且较深的节点分配到的数据量逐渐减少，容易过拟合；因此需要对其进行**剪枝(pruning)**。剪枝是指从已生成的树上裁掉子树或叶结点，并将其对应的根结点或父结点作为新的叶结点。

决策树的剪枝有**预剪枝**和**后剪枝**两种形式：
- **预剪枝(pre-pruning)**：在每次对结点进行实际划分之前，先采用验证集的数据来验证该划分是否能提高准确率。如果能就继续递归地生成结点；如果不能就把结点标记为叶结点并退出划分。
- **后剪枝(post-pruning)**：首先通过训练集生成一颗完整的决策树，然后自底向上地对内部结点进行考察，若将该结点设置为叶结点能够提高泛化性，则进行剪枝。

下面介绍决策树的后剪枝过程。定义决策树$T$的损失函数：

$$ \begin{aligned} C_{\alpha}(T) &= \sum_{t=1}^{|T|} N_t H_t(T) + \alpha \cdot |T| \\ &= - \sum_{t=1}^{|T|} \sum_{c=1}^{C} N_{tc} \log \frac{N_{tc}}{N_t} + \alpha \cdot |T| \end{aligned} $$

其中$\|T\|$是叶结点的个数，用于衡量决策树的模型复杂度；$N_t$是第$t$个叶结点对应的样本数量；$H_t(T)$是第$t$个叶结点的经验熵，

给定决策树$T$和参数$\alpha$，则决策树的单步剪枝过程为：
1. 计算每个叶结点的经验熵；
2. 递归地从叶结点向上回缩。记回缩后的树为$T'$，计算损失函数$C_{\alpha}(T), C_{\alpha}(T')$。如果$C_{\alpha}(T')\leq C_{\alpha}(T)$，则对该叶结点进行剪枝，并将其父结点作为新的叶结点。
3. 递归地调用**2**，直至满足结束条件。

上述过程是逐结点地进行剪枝，也可以直接对子树进行剪枝。记某内部结点$t$，以$t$为单结点树的损失函数为$C_{\alpha}(t)=C(t)+\alpha$，以$t$为根结点的子树$T_t$的损失函数为$C_{\alpha}(T_t)=C(T_t)+\alpha \cdot \|T_t\|$。若两个损失函数一致$C_{\alpha}(t)=C_{\alpha}(T_t)$，则该子树没有实质贡献，此时有：

$$ \alpha = \frac{C(t)-C(T_t)}{|T_t|-1} $$

因此也可以自上而下地遍历决策树的内部结点$t$，并计算该结点的$\frac{C(t)-C(T_t)}{\|T_t\|-1}$值；若该值等于$\alpha$，则对以$t$为根结点的子树$T_t$进行剪枝。

对于固定的$\alpha$，存在最优子树$T_{\alpha}$。特别地，当$\alpha \to + \infty$时最优子树为单结点树；当$\alpha=0$时最优子树为完整的决策树。若$\alpha$从小到大变化，$0<\alpha_0<\alpha_1< \cdots <\alpha_n < + \infty$，，则产生一个最优子树序列$$\{T_0,T_1,\cdots,T_n\}$$；进一步采用交叉验证法可以从中选取最优子树。