---
layout: post
title: '提升树(Boosting Tree)'
date: 2020-03-22
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63a7f0dc08b6830163d71677.jpg'
tags: 机器学习
---

> Boosting Tree.

**提升树(Boosting Tree)**是一种结合[决策树](https://0809zheng.github.io/2020/03/19/decision-tree.html)和[boosting集成学习](https://0809zheng.github.io/2020/03/18/boosting.html)的算法，被认为是统计学习中性能最好的方法之一。

提升树把**CART**分类树或回归树做为基函数，提升方法实际采用加法模型（即基函数的线性组合）与前向分步算法。提升树模型可以表示为决策树的加法模型：

$$ f_M(x) = \sum_{m=1}^M T(x;\Theta_m) $$

其中$T(x;\Theta)$为决策树，$\Theta$是决策树的参数，$M$是树的个数。

# 1. 前向分步加法模型

**加法模型(additive model)**可以表示为一系列基函数的线性组合：

$$ f(x) = \sum_{m=1}^M \beta_m b (x;\theta_m) $$

其中$b (x;\theta)$为基函数，$\theta$是基函数的参数，$\beta_m$是基函数的系数。

给定训练数据$(X,y)$和损失函数$L(y,f(x))$，可以通过经验风险最小化学习加法模型：

$$ \mathop{\min}_{\beta_m ,\theta_m} \sum_{i=1}^N L(y_i,\sum_{m=1}^M \beta_m b (x_i;\theta_m)) $$

直接求解上式比较复杂，因此采用**前向分布(forward stagewise)**算法。对于加法模型，每一步只学习一个基函数及其系数，从前向后地逐步逼近优化的目标，就可以简化优化的复杂度。因此每一步只需优化以下损失函数：

$$ (\beta_m ,\theta_m) = \mathop{\arg \min}_{\beta ,\theta} \sum_{i=1}^N L(y_i,f_{m-1}(x_i)+\beta b (x_i;\theta)) $$

则可以逐步构造加法模型：

$$ f_{m}(x) = f_{m-1}(x) + \beta_m b (x;\theta_m) $$

# 2. 提升树算法

提升树算法采用前向分布算法，给定初始提升树$f_0(x)=0$，则第$m$步的提升树模型是：

$$ f_{m}(x) = f_{m-1}(x) + T(x;\Theta_m) $$

当前决策树$T$的参数通过经验风险最小化构造：

$$ \Theta_m = \mathop{\arg \min}_{\Theta} \sum_{i=1}^N L(y_i,f_{m-1}(x_i)+T (x_i;\Theta)) $$

根据所处理的任务和设置的损失函数不同，提升树具有不同的形式。对于二分类任务，损失函数设置为指数损失函数，此时提升树算法是[AdaBoost算法](https://0809zheng.github.io/2020/03/18/boosting.html#2-adaboost)的特殊情况，称之为**自适应提升决策树ABDT**；对于回归任务，损失函数设置为平方误差损失函数，此时称为**回归提升树**；对于一般损失函数的一般决策问题，详见[<font color=blue>梯度提升决策树GBDT</font>](https://0809zheng.github.io/2020/03/21/GBDT.html)。

## (1) 自适应提升决策树 Adaptive Boosted Decision Tree
通常的**AdaBoost**算法是对样本赋予不同的权重，从而训练得到不同的模型。而决策树没有显式地定义损失函数，直接对样本赋予权重的方法实现起来是困难的，因此采用类似**bootstrap**的方法，预先按照样本的权重比例对样本集进行抽样，每次得到一个新的样本集，其中每个样本出现的概率和它的权重是差不多的。

在抽样得到的样本集上训练得到一个子模型$g_t$后，需要确定该子模型在最终模型中所占的权重。当$g_t$分类错误率$ε_t=0$，则表示该模型在该数据集上完全分类正确，对应权重$α_t=+∞$；而决策树不进行剪枝的话，很容易过拟合，实现$ε_t=0$，从而使权重无限大。因此在训练子模型的时候需要对决策树进行剪枝或限制最大深度。

因此自适应提升决策树的主要实现过程为：
- 使用**AdaBoost**算法集成决策树；
- 训练子树时按权重抽样构造数据集；
- 对训练的子树进行剪枝。

## (2) 回归提升树

回归提升树的基函数采用回归**CART**模型，把输入空间划分为$M$个互不相交的子区域$R_1,...,R_M$，计算每个子区域$R_m$上的输出值$C_m$，从而构造回归树模型：

$$ T (x;\Theta) = \sum_{m=1}^M C_m \cdot I(x \in R_m) $$

回归提升树采用前向分布算法：

$$ \begin{aligned} f_0(x)&=0 \\  f_{m}(x) &= f_{m-1}(x) + T(x;\Theta_m), m = 1,...,M \\ f_M(x) &= \sum_{m=1}^M T(x;\Theta_m) \end{aligned} $$

在前向分布算法的第$m$步，给定当前模型$f_{m-1}(x)$，通过最小化平方误差损失$L(y,f(x))=(y-f(x))^2$计算第$m$棵树的参数$\Theta_m$：

$$ \begin{aligned} \Theta_m &= \mathop{\arg \min}_{\Theta} \sum_{i=1}^N L(y_i,f_{m-1}(x_i)+T (x_i;\Theta)) \\ &= \mathop{\arg \min}_{\Theta} \sum_{i=1}^N (y_i-f_{m-1}(x_i)-T (x_i;\Theta)) ^2\end{aligned} $$

记$r_m=y-f_{m-1}(x)$是当前模型拟合数据的**残差(Residual)**，因此第$m$棵树的学习目标是拟合当前模型的残差。

### ⚪ 回归提升树的例子

已知如表所示的训练数据，$x$的取值范围为区间$[0.5, 10.5]$，$y$的取值范围为区间$[5.0, 10.0]$，学习这个回归问题的提升树模型，考虑只用决策树桩(由一个根节点直接连接两个叶结点的简单决策树)作为基函数。

![](https://pic.imgdb.cn/item/63a7c3ef08b6830163981d1b.jpg)

该数据集只有一个切分变量$x$，分别选择$s=1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5$为切分点，把数据分成左右两份。对于每个切分点分别计算两个子集的平均输出$C_1,C_2$，并进一步计算平方误差：

![](https://pic.imgdb.cn/item/63a7c42108b683016398643b.jpg)

从结果可得$s=6.5$时平方误差最小，因此选择最优切分点为$s=6.5$，划分成两个子数据集$R_1=\{1,2,3,4,5,6\}$和$R_2=\{7,8,9,10\}$，两个子数据集的输出分别为$\hat{c}_1=6.24,\hat{c}_2=8.91$，对应回归树$T_1$：

$$ T_1(x) = \begin{cases} 6.24, & x <6.5 \\ 8.91, & x \geq 6.5 \end{cases} $$

此时加法模型为$f_1(x)=T_1(x)$。使用该加法模型拟合训练数据集，计算得到平方误差损失为$1.93$，如果已经小于定义的终止误差，则停止循环。否则继续构造加法模型。

用$f_1(x)$拟合训练数据的残差$r_2=y-f_{1}(x)$计算为：

![](https://pic.imgdb.cn/item/63a7c43e08b6830163988af0.jpg)

根据残差学习回归树$T_2$，学习过程同上，不同切分点的平方误差如下：

![](https://pic.imgdb.cn/item/63a7c45308b683016398a678.jpg)

从结果可得$s=3.5$时平方误差最小，因此选择最优切分点为$s=3.5$，划分成两个子数据集$R_1=\{1,2,3\}$和$R_2=\{4,5,6,7,8,9,10\}$，两个子数据集的输出分别为$\hat{c}_1=-0.52,\hat{c}_2=0.22$，对应回归树$T_2$：

$$ T_2(x) = \begin{cases} -0.52, & x <3.5 \\ 0.22, & x \geq 3.5 \end{cases} $$

此时加法模型为$f_2(x)$:

$$ f_2(x) = f_1(x)+T_2(x) = \begin{cases} 5.72, & x <3.5 \\ 6.46, & 3.5 \leq x <6.5 \\ 9.13, & x \geq 6.5 \end{cases} $$

递归地构造加法模型，从而实现回归提升树：

![](https://pic.imgdb.cn/item/63a7c47808b683016398dc7b.jpg)

### ⚪ 实现回归提升树

```python
import numpy as np

class DisicionStump():
    def __init__(self, stump, mse, left_value, right_value, residual):
        '''决策树桩模型
        :param stump: 为feature最佳切割点
        :param mse: 为每棵树的平方误差
        :param left_value: 为决策树左值
        :param right_value: 为决策树右值
        :param residual: 为每棵决策树生成后余下的残差
        '''
        self.stump = stump
        self.mse = mse
        self.left_value = left_value
        self.right_value = right_value
        self.residual = residual

class BoostingTree():
    def __init__(self, Tree_num=100):
        self.Tree_num = Tree_num
        self.myTree = []

    def fit(self, feature, label):
        stump_list = self.Get_stump_list(feature)
        residual = label.copy()
        # 生成树并更新残差
        for num in range(self.Tree_num):
            Tree, residual = self.Get_decision_tree(stump_list, feature, residual)
            self.myTree.append(Tree)
        return self.myTree

    def Get_stump_list(self, feature):
        '''根据feature准备好切分点。例如:
        feature为[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        切分点为[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        '''
        # 特征值从小到大排序好,错位相加
        tmp1 = np.append(np.array([0]), feature)
        tmp2 = np.append(feature, np.array([0]))
        stump_list = ((tmp1 + tmp2) / float(2))[1:-1]
        return stump_list
    
    def Get_decision_tree(self, stump_list, feature, label):
        best_mse = np.inf
        best_stump = 0  # min(stump_list)
        residual = np.array([])
        left_value = 0
        right_value = 0
        # 对特征的每个切分点应用决策桩算法
        for i in range(np.shape(stump_list)[0]):
            left_node = []
            right_node = []
            # 根据切分点划分数据
            for j in range(np.shape(feature)[0]):
                if feature[j] < stump_list[i]:
                    left_node.append(label[j])
                else:
                    right_node.append(label[j])
            # 计算两个子集的平方误差
            left_mse = np.sum((np.average(left_node) - np.array(left_node)) ** 2)
            right_mse = np.sum((np.average(right_node) - np.array(right_node)) ** 2)
            # 记录最优决策桩
            if best_mse > (left_mse + right_mse):
                best_mse = left_mse + right_mse
                left_value = np.average(left_node)
                right_value = np.average(right_node)
                best_stump = stump_list[i]
                left_residual = np.array(left_node) - left_value
                right_residual = np.array(right_node) - right_value
                residual = np.append(left_residual, right_residual)
        Tree = DisicionStump(best_stump, best_mse, left_value, right_value, residual)
        return Tree, residual

    def predict(self, feature):
        predict_list = np.zeros_like(feature, dtype=np.float64)
        # 将每棵树对各个特征预测出来的结果进行相加，相加的最后结果就是最后的预测值
        for Tree in self.myTree:
            for i in range(np.shape(feature)[0]):
                if feature[i] < Tree.stump:
                    predict_list[i] = predict_list[i] + Tree.left_value
                else:
                    predict_list[i] = predict_list[i] + Tree.right_value
        return predict_list
    

if __name__ == "__main__":
    feature = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    label = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])
    mytree = BoostingTree(6)
    mytree.fit(feature, label)
    print(mytree.predict(feature))
    """
    [5.63       5.63       5.81831019 6.55164352 6.81969907 6.81969907
     8.95016204 8.95016204 8.95016204 8.95016204]
    """
```