---
layout: post
title: 'Equalization Loss for Long-Tailed Object Recognition'
date: 2021-01-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60ee8ac45132923bf81348e7.jpg'
tags: 论文阅读
---

> Equalization Loss：长尾目标检测中的均衡损失.

- paper：[Equalization Loss for Long-Tailed Object Recognition](https://arxiv.org/abs/2003.05176)
- code：[github](https://github.com/tztztztztz/eql.detectron2)

### Sigmoid交叉熵损失函数形式的Equalization Loss

在标准的**Sigmoid**交叉熵损失函数中，模型的对于真实类别的预测概率会接近$1$，其他类别的预测概率会接近$0$，但预测不同类别的结果之间并没有相关性：

$$ \mathcal{L}_{\text{CE}}(\hat{y},y) = - \sum_{j=1}^{C} log(\hat{p}_{j}) \\ \hat{p}_{j} = \begin{cases} \hat{y}_{j}, \quad y_j=1 \\ 1-\hat{y}_{j}, \quad \text{others} \end{cases} $$

而预测类别是由**Sigmoid**函数计算的：$\hat{y}_{j}=\sigma (z_j)$。由此计算损失函数对于模型**logit**输出$z_j$的梯度值：

$$ \frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_j} = \frac{\partial \mathcal{L}_{\text{CE}}}{\partial \hat{y}_{j}} \frac{\partial \hat{y}_{j}}{\partial z_j} = \begin{cases} \hat{y}_{j}-1, \quad y_j=1 \\ \hat{y}_{j}, \quad \text{others} \end{cases} $$

由上式可知，当预测类别是真实类别(对应$y_j=1$)时，梯度值会促进模型预测正确的类别；而对于其他类别(对应$y_j=0$)，梯度值会抑制模型对这些类别的预测。

对某一个类别预测时，该类别被看作是前景，其余类别都被看作背景。对**tail class**对应的类别，只有当预测该类别时(数据占比很小)模型的训练才会促进该类别的预测；大多数情况下(即处理其他类别的数据时)模型预测该类别的能力会被抑制。

**Equalization Loss**通过对交叉熵损失函数增加一个权重，减小对**tail class**类别的抑制情况。损失函数计算如下：

$$ \mathcal{L}_{\text{EQ}}(\hat{y},y) = - \sum_{j=1}^{C} w_j \cdot log(\hat{p}_{j}) $$

其中权重设置如下：

$$ w_j = 1-E(r)T_{\lambda}(f_j)(1-y_j) $$

$r$是目标检测提出的一个**proposal region**，当$r$是前景时$E(r)=1$，当$r$是背景时$E(r)=0$。$f_j$是类别$j$出现在数据集中的频率$\frac{n_j}{n}$。$T_{\lambda}(\cdot)$是一个门限函数，$\lambda$是人为选定的阈值，用于区分该类别是**head**还是**tail**类：

$$ T_{\lambda}(x) = \begin{cases} 1, \quad x< \lambda \\ 0, \quad \text{others} \end{cases} $$

对该权重的理解如下。
- 如果当前**proposal region**是背景，则$w_j=1$，此时损失函数退化为**Sigmoid**交叉熵损失；
- 如果当前**proposal region**是前景，且类别为**head class**，则$T_{\lambda}(f_j)=0$，$w_j=1$，此时损失函数也退化为**Sigmoid**交叉熵损失；
- 如果当前**proposal region**是前景，且类别为**tail class**，则$T_{\lambda}(f_j)=1$，$w_j=y_j$，即只有该**tail**类别(对应$y_j=1$的情况)会产生对应的损失并求得梯度，对于其他类别(对应$y_j=0$的情况)则不考虑对该**tail**类别产生的负梯度。

### Softmax交叉熵损失函数形式的Equalization Loss
在标准的**Softmax**交叉熵损失函数中，模型的类别预测概率是通过**Softmax**进行归一化计算得到的，不同类别的预测结果之间具有相关性：

$$ \mathcal{L}_{\text{CE}}(\hat{y},y) = - \sum_{j=1}^{C} y_jlog(\hat{p}_{j}) \\ \hat{p}_{j} = \frac{e^{z_j}}{\sum_{k=1}^{C}e^{z_k}} $$

此时的**Equalization Loss**计算如下：

$$ \mathcal{L}_{\text{EQ}}(\hat{y},y) = - \sum_{j=1}^{C} y_jlog(\tilde{p}_{j}) \\ \tilde{p}_{j} = \frac{e^{z_j}}{\sum_{k=1}^{C} \tilde{w}_k e^{z_k}} $$

其中权重设置如下：

$$ \tilde{w}_k= 1-\beta T_{\lambda}(f_k)(1-y_k) $$

其中$\beta$是额外引入的超参数。由于使用**Softmax**函数，不再考虑背景这一类，因此没有使用$E(r)$函数。

