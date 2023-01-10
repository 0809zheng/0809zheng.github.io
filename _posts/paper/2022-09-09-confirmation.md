---
layout: post
title: 'Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning'
date: 2022-09-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63bd1613be43e0d30e64da55.jpg'
tags: 论文阅读
---

> 深度无监督学习的伪标签和确认偏差.

- paper：[Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/abs/1904.04717)

**自训练(self-training)**是半监督学习中的常用方法，即首先通过有标签样本初始化训练一个教师网络，然后通过该网络预测无标签样本的伪标签，并选择其中置信度最高的一批样本按伪标签进行标注，通过扩增的数据集训练一个学生网络。通过迭代上述过程直至所有未标注样本都被指定一个伪标签。

自训练过程会产生**确认偏差(Confirmation bias)**问题，即不完美的教师网络将会提供错误的伪标签，而对这些错误标签过拟合不利于得到更好的学生网络。

为了缓解确认偏差问题，可以对样本及其软标签应用[**mixup**](https://0809zheng.github.io/2020/06/26/mixup.html)。给定两个样本$(x_i,x_j)$及其对应的真实标签或伪标签$(y_i,y_j)$，则构造新的样本：

$$ \overline{x} = \lambda x_i + (1-\lambda) x_j \\ \overline{y} = \lambda y_i + (1-\lambda) y_j $$

对软标签的插值等价于对交叉熵损失函数的插值：

$$ L = \lambda [y_i \log f_{\theta}(\overline{x})] + (1-\lambda) [y_j \log f_{\theta}(\overline{x})] $$

如果有标签样本数量过少，则上述方法是不充分的。此时可以通过对有标签样本进行过采样来保证每轮训练中有标签样本的最少数量。过采样比增大有标签样本的损失权重效果更好，因为前者鼓励更频繁的参数更新，而不是较少的更大幅度但不稳定的更新。