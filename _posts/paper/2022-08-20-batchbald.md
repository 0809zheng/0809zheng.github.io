---
layout: post
title: 'BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning'
date: 2022-08-20
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/632ac24616f2c2beb1e3da31.jpg'
tags: 论文阅读
---

> BatchBALD：深度贝叶斯主动学习的高效多样性批量获取.

- paper：[BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning](https://arxiv.org/abs/1906.08158)

在主动学习中，只依据样本的不确定性进行采样可能会导致采样偏差，即采样的样本集中具有较大的冗余度。例如一些样本的相似度较高，则它们的不确定性得分可能同时比较高：

![](https://pic.imgdb.cn/item/632ac32b16f2c2beb1e4e688.jpg)

回顾[<font color=Blue>BALD</font>](https://0809zheng.github.io/2022/08/03/bald.html)方法，单个样本的获取函数是模型参数与模型输出之间的互信息：

$$ I(y;w | x,D_{train}) = H(y|x,D_{train}) - E_{p(w|D_{train})}[H(y|x,w,D_{train})] $$

**BatchBALD**方法则是一次性选择一批样本，因此获取函数调整为：

$$ \begin{aligned} I(y_1,\cdots,y_B;w | x_1,\cdots,x_B,D_{train}) =& H(y_1,\cdots,y_B|x_1,\cdots,x_B,D_{train}) \\ &- E_{p(w|D_{train})}[H(y_1,\cdots,y_B|x_1,\cdots,x_B,w,D_{train})] \end{aligned} $$

上式中第二项寻找在给定参数信息的情况下输出结果熵较小的样本，采用蒙特卡洛积分进行近似，运行多次网络，计算每一次运行结果的熵，并取平均：

$$ E_{p(w|D_{train})}[H(y_1,\cdots,y_B|x_1,\cdots,x_B,w,D_{train})] \\ ≈ \frac{1}{k} \sum_{i=1}^B \sum_{j=1}^k H(y_i|w_j) $$

而第一项寻找平均输出结果的熵较大的样本，由于这一项不显式包含参数项，因此无法直接应用蒙特卡洛近似。作者根据等式$p(y) = E_{p(w)}[p(y|w)]$进行如下近似：

$$ H(y_1,\cdots,y_B|x_1,\cdots,x_B,D_{train})  = E_{p(y_1,\cdots,y_B)}[-\log p(y_1,\cdots,y_B)] \\ = E_{p(w)}E_{p(y_1,\cdots,y_B|w)}[-\log E_{p(w)} p(y_1,\cdots,y_B | w)] \\ ≈ -\sum_{\hat{y}_{1:B}} (\frac{1}{k}\sum_{j=1}^kp(\hat{y}_{1:B}|w_j)) \log (\frac{1}{k}\sum_{j=1}^kp(\hat{y}_{1:B}|w_j)) $$
