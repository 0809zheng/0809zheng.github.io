---
layout: post
title: 'Long-tail learning via logit adjustment'
date: 2021-01-05
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/625ffffc239250f7c54885b1.jpg'
tags: 论文阅读
---

> Logit Adjustment Loss: 将类别出现频率引入logits.

- paper：[Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314)

# 1. 问题建模
对于单标签多分类问题，假设共有$K$个类别，训练数据$(x,y)~\mathcal{D}$，使用神经网络拟合条件概率分布$p(y\|x)$，则目标为最小化交叉熵损失：

$$ \mathop{\arg \min} \Bbb{E}_{(x,y)~\mathcal{D}} [-\log p(y|x)] $$

分类任务的最后一层通常都是**softmax**函数，而**softmax**函数之前的值称为**logits**，记为$f(x)$，则有：

$$ -\log p(y|x) = -\log \frac{e^{f_y(x)}}{\sum_{i=1}^{K}e^{f_i(x)}} = \log[1+\sum_{i≠y}^{}e^{f_i(x)-f_y(x)}] $$

# 2. 拟合互信息

除了拟合条件概率$p(y\|x)$，也可以拟合下面的统计量，且后者是对称的，能够捕捉样本与标签更本质的信息：
$$ \frac{p(y|x)}{p(y)} = \frac{p(x,y)}{p(x)(y)} $$

上式计算了样本$x$与标签$y$共同出现的概率与它们随机出现概率的倍数。如果该值远大于$1$，则表明样本与标签倾向于共同出现而不是随机组合；如果该值远小于$1$，则表明样本与标签倾向于不同时出现。

对上式取对数，即为**点互信息(pointwise mutual information, PMI)**:
$$ \log \frac{p(y|x)}{p(y)} $$

# 3. Logit Adjustment Loss

若使用神经网络建模互信息，则有：
$$ f(x) = \log \frac{p(y|x)}{p(y)} $$

上式也写为：
$$ \log p(y|x) = f(x) + \log p(y) $$

应用**softmax**函数后有：

$$ p(y|x) = \frac{e^{f_y(x)+ \log p(y)}}{\sum_{i=1}^{K}e^{f_i(x)+ \log p(i)}} $$

构造损失函数：

$$ -\log p(y|x) = -\log  \frac{e^{f_y(x)+ \log p(y)}}{\sum_{i=1}^{K}e^{f_i(x)+ \log p(i)}} = \log[1+\sum_{i≠y}^{}(\frac{p(i)}{p(y)})e^{f_i(x)-f_y(x)}] $$

上述损失函数称为**Logit Adjustment Loss**，相当于把各个类别出现的频率$p(y)$作为先验知识引入了训练过程中。

也可以引入调节因子$\tau$：

$$ -\log p(y|x) = -\log  \frac{e^{f_y(x)+\tau \log p(y)}}{\sum_{i=1}^{K}e^{f_i(x)+\tau \log p(i)}} = \log[1+\sum_{i≠y}^{}(\frac{p(i)}{p(y)})^{\tau}e^{f_i(x)-f_y(x)}] $$

# 4. post-hoc
该方法也可以应用于训练完成的模型。在预测过程中，不是直接输出条件概率$p(y\|x)$最大的类别$y$，而是输出使得互信息最大的类别，即：
$$ y^{*} = \mathop{\arg \max}_{y} f_y(x) +\tau \log p(y) $$

# 5. 实验分析

![](https://pic.imgdb.cn/item/62600950239250f7c55b3690.jpg)

结果表明，通过引入互信息处理类别不平衡问题，使网络获得比较好的预测结果。

