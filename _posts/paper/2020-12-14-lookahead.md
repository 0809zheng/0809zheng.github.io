---
layout: post
title: 'Lookahead Optimizer: k steps forward, 1 step back'
date: 2020-12-14
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/620a1a5c2ab3f51d91a95333.jpg'
tags: 论文阅读
---

> Lookahead：快权重更新k次，慢权重更新1次.

- paper：[Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

对梯度下降算法进行的改进可分为两种，自适应学习率(如**AdaGrad**)和动量(如**momentum**)。作者提出了一种新的优化算法**lookahead**，与上述优化算法是正交的，可以提高任意优化算法的性能。**lookahead**迭代地更新两组权重，能够提高学习稳定性，降低优化器的方差，而额外的计算成本可以忽略不计。

## 1. Lookahead

**lookahead**算法维护两组权重，即**慢权重(slow weight)** $\phi$ 和**快权重(fast weight)** $\theta$。

**快权重** $\theta$是通过任意一种标准优化算法$A$进行更新的。在第$t$轮更新中，快权重首先被初始化为上一轮慢权重$\theta_{t,0}=\phi_{t-1}$，之后进行$k$次更新。若记批量数据为$d$，损失函数为$L$，则第$t$轮中快权重的第$i$次更新为：

$$ \theta_{t,i} = \theta_{t,i-1} +A(L,\theta_{t,i-1},d) $$

**慢权重** $\phi$是通过在参数空间$\theta-\phi$中进行线性插值得到的。快权重每经过$k$次更新后，慢权重更新$1$次：

$$ \phi_{t} = \phi_{t-1} +\alpha(\theta_{t,k}-\phi_{t-1}) $$

慢权重也可以看作每$k$次更新的快权重的指数滑动平均：

$$ \phi_{t} = \phi_{t-1}+\alpha(\theta_{t,k}-\phi_{t-1}) = (1-\alpha) \phi_{t-1} + \alpha \theta_{t,k} \\ = (1-\alpha) [(1-\alpha) \phi_{t-2} + \alpha \theta_{t-1,k}] + \alpha \theta_{t,k} \\ = (1-\alpha)^t \phi_0 + \sum_{i=1}^{t}\alpha (1-\alpha)^{t-i}\theta_{i,k} $$

直观上，**lookahead**算法使用任何标准优化器更新“快权重”$k$次，然后沿最终快权重的方向更新一次“慢权重”；相比于直接使用优化器更新$k$次，运算操作数变为$O(\frac{k+1}{k})$倍。

当快权重沿低曲率方向进行更新时，慢权重通过参数插值平滑振荡，实现快速收敛。下图显示了在**CIFAR-100**上优化**ResNet-32**模型时快权重和慢权重的更新轨迹。当快权重在极小值附近探索时，慢权重更新将推向一个测试精度更高的区域。

![](https://pic.imgdb.cn/item/620b0d972ab3f51d91650004.jpg)

## 2. 实验分析

**lookahead**算法需要设置两个超参数，快权重每轮更新次数$k$和慢权重的步长$\alpha$。通过对超参数不同取值的实验表明，模型对这两个超参数具有一定的鲁棒性，在实践中可以不对它们进行调整，默认$k=5,\alpha=0.5$。

![](https://pic.imgdb.cn/item/620b11cd2ab3f51d9168a8d1.jpg)

作者绘制了在不同阶段的快权重和慢权重更新导致的损失曲线。在每个阶段中，快权重更新可能导致任务性能的大幅下降，结果具有较高的方差。慢权重通过插值降低了方差，稳定模型性能。

![](https://pic.imgdb.cn/item/620b12772ab3f51d91694d3e.jpg)