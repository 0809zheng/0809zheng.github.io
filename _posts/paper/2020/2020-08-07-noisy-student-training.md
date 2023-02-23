---
layout: post
title: 'Self-training with Noisy Student improves ImageNet classification'
date: 2020-08-07
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f2e66f614195aa5945b2dc5.jpg'
tags: 论文阅读
---

> Noisy Student Training：一种半监督图像分类方法.

- paper：Self-training with Noisy Student improves ImageNet classification
- arXiv：[link](https://arxiv.org/abs/1911.04252)

# 模型

![](https://pic.downk.cc/item/5f2e651314195aa5945a6fe3.jpg)

作者提出了一种半监督图像分类方法，主要包括$4$个步骤：
- 使用标记数据训练**教师网络**；
- 使用训练好的**教师网络**对大量无标签数据分类，制造伪标签；
- 训练一个模型容量相等或更大的**学生网络**学习原标记数据和伪标签数据，同时引入噪声干扰，包括：
1. 数据增强：引入**data noise**；
2. **Dropout**：引入**model noise**；
3. 随机深度 **Stochastic depth** ：引入**model noise**。
- 将训练好的**学生网络**作为新的**教师网络**重复上述过程。

# 算法
记标记数据为$$\{(x_1,y_1),...,(x_n,y_n)\}$$，无标签数据为$$\{\tilde{x}_1,...,\tilde{x}_m\}$$，通常$$n<m$$；

1: 使用标记数据训练**教师网络**$θ^t$，在训练中引入**noise**；

$$ \frac{1}{n} \sum_{i=1}^{n} {l(y_i,f^{noised}(x_i,θ^t))} $$

2: 使用无**noise**的**教师网络**$θ^t$预测无标签数据的标签：

$$ \tilde{y}_i = f(\tilde{x}_i,θ^t) $$

3: 训练一个模型容量相等或更大的**学生网络**$θ^s$，在标记数据和无标签数据上进行：

$$ \frac{1}{n} \sum_{i=1}^{n} {l(y_i,f^{noised}(x_i,θ^s))} + \frac{1}{m} \sum_{i=1}^{m} {l(\tilde{y}_i,f^{noised}(\tilde{x}_i,θ^s))} $$

4: 将训练好的**学生网络**作为新的**教师网络**重复上述过程。

# 实验

![](https://pic.downk.cc/item/5f2e66d314195aa5945b1d8d.jpg)

作者共训练了三轮，使用**ImageNet**作为标记数据集，每轮的设置如下：
1. **教师网络**：**EfficientNet-B7**；**学生网络**：**EfficientNet-L2**；**batch size ratio**：（无标签数据：有标签数据）$$14:1$$；
2. **教师网络**：**EfficientNet-L2**；**学生网络**：**EfficientNet-L2**；**batch size ratio**：（无标签数据：有标签数据）$$14:1$$；
3. **教师网络**：**EfficientNet-L2**；**学生网络**：**EfficientNet-L2**；**batch size ratio**：（无标签数据：有标签数据）$$28:1$$.

![](https://pic.downk.cc/item/5f2e6a8b14195aa5945ce70d.jpg)

作者进行了大量消融实验，得到以下结论：
1. 使用更大的**教师网络**可以得到更好的结果；
2. 大量的无标签数据是必须的；
3. **soft**伪标签比**hard**伪标签表现好；
4. 使用更大的**学生网络**能够增强模型能力；
5. 对于小模型，数据平衡是有用的（对无标签数据进行过滤）；
6. 训练**学生网络**时，在无标签数据上预训练，在标记数据上微调；
7. 使用更大的无标签数据与有标签数据的**batch size ratio**得到更高的准确率；
8. 从头开始训练**学生网络**比使用**教师网络**初始化**学生网络**效果更好。
