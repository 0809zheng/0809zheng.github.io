---
layout: post
title: 'Rethinking the Inception Architecture for Computer Vision'
date: 2021-03-11
author: 郑之杰
cover: 'https://img.imgdb.cn/item/604abe5e5aedab222c7cb40c.jpg'
tags: 论文阅读
---

> Label Smooth：数据集的标签平滑技巧.

- paper：Rethinking the Inception Architecture for Computer Vision
- arXiv：[link](https://arxiv.org/abs/1512.00567)

在图像分类等视觉任务中，神经网络的输出层会输出长度为等于数据集类别数$K$的特征$z$，称之为**logits**。经过**softmax**函数后转化为概率分布$\hat{y}$，记数据集的标签真值为$y$。

网络预测的类别分布计算为：

$$ \hat{y}_i = \frac{exp(z_i)}{\sum_{j=1}^{K} exp(z_j)} $$

模型的目标函数表示为负交叉熵损失：

$$ \mathcal{l}(\hat{y},y) = - \sum_{i=1}^{K} y_ilog\hat{y}_i \\ = - \sum_{i=1}^{K} y_i[z_i - log(\sum_{j=1}^{K} exp(z_j))] $$

下面求解使得目标函数最小的理论解。先求损失函数相对于输出**logits**的偏导数：

$$ \frac{\partial l}{\partial z_i} = -\frac{\partial}{\partial z_i} \sum_{i=1}^{K} y_i[z_i - log(\sum_{j=1}^{K} exp(z_j))] \\ = -\frac{\partial}{\partial z_i} y_i[z_i - log(\sum_{j=1}^{K} exp(z_j))] \\ = -\frac{\partial}{\partial z_i} y_iz_i + \frac{\partial}{\partial z_i} y_ilog(\sum_{j=1}^{K} exp(z_j)) \\ = -y_i + y_i\frac{exp(z_i)}{\sum_{j=1}^{K} exp(z_j)} = -y_i + y_i\hat{y}_i = y_i(\hat{y}_i-1) $$

# one-hot编码
当数据集的标注采用**one-hot**编码时，即对于正确的分类标记$y_{\text{true}}=1$，错误的分类标记$y_{\text{false}}=0$。目标函数最优时，令上述导数为$0$，网络的预测概率应满足：$$\hat{y}_{\text{true}} = 1$$，$$\hat{y}_{\text{false}} = \text{Cons}$$。这就要求网络学习得到的**logits**应满足：$z_{\text{true}}=+∞$，$z_{\text{false}}=\text{Cons}$。

在实践中最优的情况一般无法达到，但通常网络训练会使$z_{\text{true}}$远大于$z_{\text{false}}$，会导致：
1. 导致过拟合，即网络训练将所有的概率赋给真值，导致泛化能力下降；
2. 模型追求真值对应的**logits**远大于其他值的**logits**，但更新梯度是有界的，数值通常不会太大，需要很多次更新才能满足要求。

# Label Smooth
**Label Smooth**的思想是对数据集的标注不再采用**one-hot**编码，而是采用一种容错率更高的编码形式：

$$ y_i = \begin{cases} 1- \epsilon \quad \text{if } i=\text{true} \\ \frac{\epsilon}{K-1} \quad \text{otherwise} \end{cases} $$

此时网络输出**logits**学习的目标是：

$$ \frac{exp(z_{\text{true}})}{\sum_{j=1}^{K} exp(z_j)} = 1- \epsilon $$

$$ \frac{exp(z_{\text{false}})}{\sum_{j=1}^{K} exp(z_j)} = \frac{\epsilon}{K-1} $$

上述两式相除，可得：

$$ \frac{exp(z_{\text{true}})}{exp(z_{\text{false}})} = \frac{(1- \epsilon)(K-1)}{\epsilon} $$

上式两端取对数，可得：

$$ z_{\text{true}} - z_{\text{false}} = log(\frac{(1- \epsilon)(K-1)}{\epsilon}) $$

记$z_{\text{false}}$为$\alpha$，则网络输出**logits**的目标值$z_i^*$记为：

$$ z_i^* = \begin{cases} log(\frac{(1- \epsilon)(K-1)}{\epsilon}) + \alpha \quad \text{if } i=\text{true} \\ \alpha \quad \text{otherwise} \end{cases} $$

应用标签平滑后，网络输出**logits**的目标值是有限值，且正确类和错误类之间的**logits**存在一个**gap**：$log(\frac{(1- \epsilon)(K-1)}{\epsilon})$，其值取决于分类数量$K$和超参数$\epsilon$。在实践中常取$\epsilon=0.1$。
