---
layout: post
title: 'Naive Bayes：朴素贝叶斯'
date: 2020-03-28
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63a1c411b1fccdcd362183e2.jpg'
tags: 机器学习
---

> Naive Bayes.

**朴素贝叶斯(Naive Bayes)**是一种软分类方法，它要求输入数据$X$的特征取值是离散的，给出可能属于每一个类别$Y$的概率值。

对于二分类问题，每个样本的类别随机变量$y$服从伯努利(**Bernoulli**)分布，所有样本的类别变量$Y$服从二项(**Binormial**)分布；对于多分类问题，每个样本的类别随机变量$y$服从类别(**Categorial**或**Multinoulli**)分布，所有样本的类别变量$Y$服从多项(**Multinormial**)分布。对于输入数据$X$的每个特征$x_d$，也服从类别分布。

给定数据集$(X,Y)$，朴素贝叶斯把分类模型建模为后验概率$P(Y\|X)$：

$$ P(Y|X) = \frac{P(X,Y)}{P(X)}= \frac{P(X|Y)P(Y)}{P(X)} $$

其中先验概率$P(Y)$可以通过类别$Y$求得；对条件概率$P(X\|Y)$引入**条件独立性假设**，假设数据$X$的特征之间是独立的：

$$ \begin{aligned} P(X=x|Y=c) &= P(X_1=x_1,...,X_d=x_d|Y=c) \\ & = \prod_d P(X_d=x_d|Y=c) \end{aligned} $$

根据训练数据集学习先验概率和条件概率。训练完成后，对于测试数据$X$，通过**后验概率最大化**判断其所属的类别：

$$ \begin{aligned} c^* &= \mathcal{\arg \max}_{c} P(Y=c|X) = \mathcal{\arg \max}_{c} \frac{P(X|Y=c)P(Y=c)}{P(X)} \\ & \propto  \mathcal{\arg \max}_{c} P(X|Y=c)P(Y=c) \\ & =  \mathcal{\arg \max}_{c} \prod_d P(X_d=x_d|Y=c)P(Y=c) \end{aligned} $$

在实践中，概率的连乘会导致较小的数值；为了提高方法的稳定性，计算上式的对数似然：

$$ \sum_d \log P(X_d=x_d|Y=c) + \log P(Y=c) $$

### ⚪ 讨论：后验概率最大化等价于0-1损失的期望风险最小化

定义分类的**0-1**损失：

$$ L(Y, f(X)) = \begin{cases} 0, & Y=f(X) \\ 1, & Y\neq f(X) \end{cases} $$

则模型$f(x)$的期望风险为：

$$ \begin{aligned} \Bbb{E}_{P(X,Y)}[L(Y, f(X))] &= \sum_X \sum_Y L(Y, f(X))P(X,Y) \\ & = \sum_X \sum_Y L(Y, f(X))P(Y|X)P(X)\\ & = \sum_X \sum_c L(Y=c, f(X))P(Y=c|X)P(X) \\ & = \Bbb{E}_{P(X)} [\sum_c L(Y=c, f(X))P(Y=c|X)] \end{aligned} $$

模型$f(x)$的最优值通过期望风险最小化寻找：

$$ \begin{aligned} f(x)& = \mathcal{\arg \min}_{f} \Bbb{E}_{P(X)} [\sum_c L(Y=c, f(X))P(Y=c|X)] \\ & = \mathcal{\arg \min}_{y}  [\sum_c L(Y=c, y)P(Y=c|X=x)] \\ & = \mathcal{\arg \min}_{y}  [\sum_c P(y \neq c|X=x)] \\ & = \mathcal{\arg \min}_{y}  [1- P(y = c|X=x)] \\ & = \mathcal{\arg \max}_{y}  [P(y = c|X=x)] \end{aligned} $$

因此**0-1**损失的期望风险最小化等价于后验概率最大化。

### ⚪ 朴素贝叶斯方法的贝叶斯估计

条件概率$P(X_d=x_d \| Y=c)$的估计值可能为$0$，会影响到后验概率的计算。条件概率的贝叶斯估计为：

$$ P(X_d=x_d | Y=c) = \frac{\sum_i I(X^i_d = x_d,Y^i=c)+\lambda}{\sum_iI(Y^i=c)+S_d \lambda} $$

上式相当于在随机变量各个取值的频数上增加一个正数$\lambda$。当$\lambda=0$时等价于极大似然估计；当$\lambda=1$时称为**拉普拉斯平滑(Laplace smoothing)**。

### ⚪ 实现朴素贝叶斯

```python
class NaiveBayes():
    def __init__(self):
        self.model = {}

    def fit(self, datas, labels):
        # 获取分类的类别, 输入np.array格式, 尺寸为[N, D], [N]
        self.keys = set(labels.tolist())        
        for key in self.keys:
            # 计算P(Y)
            PY = labels.tolist().count(key)/len(labels)
            # 收集标签为Y的数据
            index = np.where(labels==key)[0].tolist()
            data_Y = datas[index]
            # 计算P(X|Y)
            PX_Y = {}
            for d in range(datas.shape[1]):
                data = data_Y[:,d].tolist()
                values = set(data)
                N = len(data)
                PX_Y[d] = {}
                for value in values:
                    PX_Y[d][value] = float(data.count(value)/N)
            # 模型保存
            self.model[key] = {}
            self.model[key]["PY"] = PY
            self.model[key]["PX_Y"] = PX_Y

    def predict(self, data):
        # 预测一个数据的类别，尺寸为[D]
        results = {}
        eps = 0.00001
        for key in self.keys:
            # 获取P(Y)
            PY = self.model.get(key, eps).get("PY")
            # 分别获取P(X|Y)
            model_X = self.model.get(key, eps).get("PX_Y")
            PX_Y = []
            for d in range(len(data)):
                pb = model_X.get(d, eps).get(data[d],eps)
                PX_Y.append(pb)
            result = np.log(PY) + np.sum(np.log(PX_Y))
            results[key] = result
        return results
```