---
layout: post
title: 'Simple yet Effective Way for Improving the Performance of GAN'
date: 2022-04-25
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63958ce3b1fccdcd36551e76.jpg'
tags: 论文阅读
---

> 通过级联抑制方法增强GAN的判别器.

- paper：[Simple yet Effective Way for Improving the Performance of GAN](https://arxiv.org/abs/1911.10979)

生成对抗网络的判别器相当于一个二分类器，通过卷积层、全连接层等结构后提取特征向量$v$，最后一层全连接层通过权重向量$w$对特征进行打分，通过内积和**Sigmoid**激活函数给出输入数据属于真实数据分布的置信度：

$$ D(x) = \sigma(<v, w>) $$

本文作者指出，这种通过内积进行判别的判别器的性能是比较弱的，因为$<v, w>$只取决于特征向量$v$在权重向量$w$上的投影：

![](https://pic.imgdb.cn/item/63959665b1fccdcd36652bcd.jpg)

此时固定的$<v, w>$可能对应许多不同的特征向量$v$，因此判别器学习到的特征向量具有不确定性。内积$<v, w>$没有考虑特征向量的垂直分量：

$$ v - ||v|| \cos(v,w) \frac{w}{||w||} = v-\frac{<v, w>}{||w||^2}w $$

本文作者提出了级联抑制(**cascading rejection**)方法，来增强判别器的性能。具体做法是使用内积对特征向量$v$进行分类后，对特征向量$v$的垂直分量再做一次分类；并且再次分类也会导致一个新的垂直分量，从而实现迭代地分类：

![](https://pic.imgdb.cn/item/639597a9b1fccdcd3666bc29.jpg)

级联抑制的方法如下：

$$ \begin{aligned} v_1 &=v \\ D_1(x) &= \sigma(<v_1, w_1>) \\ v_2 &=v_1-\frac{<v_1, w_1>}{||w_1||^2}w_1 \\ D_2(x) &= \sigma(<v_2, w_2>) \\ & \cdots  \\ v_n &=v_{n-1}-\frac{<v_{n-1}, w_{n-1}>}{||w_{n-1}||^2}w_{n-1} \\ D_n(x) &= \sigma(<v_n, w_n>) \end{aligned} $$

得到$n$个得分$D_1(x),D_2(x),...,D_n(x)$后，对其加权平均作为最终的判别器损失。该方法能够提升**GAN**的性能：

![](https://pic.imgdb.cn/item/6395990cb1fccdcd3668db98.jpg)

值得一提的是，这种方法能够提高生成对抗网络的性能，但不适合有监督的分类任务。这是因为级联抑制增大了判别器的分类难度，从而间接提高了生成器的性能；而对于分类任务不应该增大判别难度。