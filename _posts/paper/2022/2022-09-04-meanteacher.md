---
layout: post
title: 'Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results'
date: 2022-09-04
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63ba8a92be43e0d30ee69c04.jpg'
tags: 论文阅读
---

> 加权平均一致性目标改进半监督深度学习.

- paper：[Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780)

**Mean Teacher**是一种半监督学习方法，旨在通过追踪模型权重的滑动平均值构造无监督损失函数。

称每一时刻的模型为学生模型，参数为$\theta$；存储模型参数的滑动平均值$\theta'$作为教师模型：

$$ \theta' \leftarrow \beta \theta' + (1-\beta)\theta $$

![](https://pic.imgdb.cn/item/63ba8c52be43e0d30eeac6c8.jpg)

通过学生模型和教师模型预测结果之间的距离构造一致性正则化损失，并对其进行最小化。教师模型能够提供比学生模型更准确的预测结果。

![](https://pic.imgdb.cn/item/63ba8dfcbe43e0d30eeec62a.jpg)

图（a）和（b）表明，输入增强(随机翻转或高斯噪声)或对学生模型引入**dropout**能够获得更好的表现，而教师模型通常不使用**dropout**。图（d）表明，模型表现对指数滑动平均的衰减率超参数$\beta$比较敏感，一个较好的策略是在训练初期使用较小的$\beta=0.99$，当学生模型的学习速度下降时使用更大的$\beta=0.999$。图（f）表明，一致性正则化损失选用均方误差效果最好。


![](https://pic.imgdb.cn/item/63ba8f42be43e0d30ef1282d.jpg)
