---
layout: post
title: '损失函数的健全性检查(Sanity Check)'
date: 2026-02-15
author: 郑之杰
cover: ''
tags: 数学
---

> Sanity Check for loss functions.

对于一个深度学习模型，在训练开始时模型的权重通常是随机初始化的。一个好的初始化策略（如 **Kaiming** 或 **Xavier**）会使得模型的输出在送入最后的激活函数（如 **Softmax** 或 **Sigmoid**）之前，其 **logits**（原始输出值）的期望为0，并且方差较小。

这意味着在没有任何学习的情况下，模型对于所有类别的预测倾向是**均等**的，类似于“随机猜测”。通过计算在这种随机猜测情景下各种常见损失函数的理论期望值，能够对模型的理论初始损失进行**健全性检查(Sanity Check)**。这种检查方法非常强大，因为它允许我们在开始大规模训练之前，仅凭第一步的损失值就能判断模型和数据加载是否设置正确。

#### TL;DR:

| 任务类型 | 损失函数 | 最后一层激活 | 理论初始损失 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **二分类** | **Binary Cross-Entropy (BCE)** | **Sigmoid** | $\log(2) \approx 0.693$ | 假设初始预测概率为 0.5 |
| **多分类** | **Cross-Entropy** | **Softmax** | $\log(C)$ | $C$ = 类别数 |
| **回归** | **Mean Squared Error (MSE)** | **Linear** | $Var(y) + (E[y])^2$ | $y$ 是目标值。 |
| **对比学习** | **InfoNCE** | **L2 Normalization** | $\log(N+1)$ | $N$ = 负样本数量 |

## 1. 二分类问题 (Binary Classification)

二分类问题的损失函数是二进制交叉熵损失 (**Binary Cross-Entropy Loss, BCELoss**)，网络最后一层激活函数通常是**Sigmoid**。

模型最后一层输出一个单独的 **logit** $z$。随机初始化使得 $E[z] \approx 0$。**Sigmoid** 函数将 **logit** 转换为一个 $[0, 1]$ 范围内的概率值 $p$，代表“正类”的概率。

$$
p = \text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}
$$

当 $z \approx 0$ 时，$e^{-z} \approx e^0 = 1$。因此，初始预测概率为：

$$
p \approx \frac{1}{1 + 1} = 0.5
$$

模型对于是“正类”还是“负类”的判断是完全不确定的。

对于一个真实标签为 $y$（$y=1$ 代表正类，$y=0$ 代表负类）的样本，**BCELoss** 的定义是：

$$
\mathcal{L} = -[y \log(p) + (1-y) \log(1-p)]
$$

分别计算两种情况：
*   如果真实标签 $y=1$:  $\mathcal{L} = -\log(p) \approx -\log(0.5) = \log(2)$
*   如果真实标签 $y=0$:  $\mathcal{L} = -\log(1-p) \approx -\log(1-0.5) = -\log(0.5) = \log(2)$

#### 结论：对于一个二分类问题，如果模型的初始预测概率为 0.5，初始的二进制交叉熵损失应该约等于 $\log(2) ≈ 0.693$。

## 2. 多分类问题 (Multi-Class Classification)

多分类问题的损失函数是交叉熵损失 (**Cross-Entropy Loss**)，网络最后一层激活函数通常是**Softmax**。

模型最后一层输出原始 **logits** $z = [z_1, z_2, \dots, z_C]$，其中 $C$ 是类别总数。由于随机初始化，我们可以假设 $E[z_i] \approx 0$ 对于所有类别 $i$。

**Softmax** 函数将 **logits** 转换为概率分布：

$$
p_i = \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

当所有 $z_i$ 都接近 0 时，$e^{z_i} \approx e^0 = 1$。因此，每个类别的预测概率都趋向于：

$$
p_i \approx \frac{1}{\sum_{j=1}^{C} 1} = \frac{1}{C}
$$

这意味着模型对每个类别的预测概率都是均等的 $1/C$，即“随机猜测”。

对于一个真实标签为类别 $y$ 的样本（在 **one-hot** 编码中， $y_i=1$ 当 $i=y$，$y_i=0$ 当 $i \neq y$），交叉熵损失的定义是：

$$
\mathcal{L} = -\sum_{i=1}^{C} y_i \log(p_i)
$$

由于只有真实类别 $y$ 对应的 $y_y=1$，其余都为0，损失简化为：

$$
\mathcal{L} = -1 \cdot \log(p_y)
$$

将 $p_y \approx 1/C$ 代入，我们得到初始损失的期望值：

$$
E[\mathcal{L}_{\text{init}}] \approx -\log\left(\frac{1}{C}\right) = \log(C)
$$

#### 对于一个 C 分类问题，初始的交叉熵损失应该约等于 $\log(C)$。

比如对于图像分类任务，**CIFAR-10 (10类)**:初始损失应在 $\log(10) ≈ 2.3$ 左右，**ImageNet (1000类)**: 初始损失应在 $\log(1000) ≈ 6.9$ 左右。

## 3. 回归问题 (Regression)

回归问题的损失函数是均方误差损失 (**Mean Squared Error, MSELoss**)，最后一层激活通常是线性层 (无激活)。

模型最后一层直接输出一个或多个预测值 $\hat{y}$。随机初始化使得 $E[\hat{y}] \approx 0$。

**MSELoss** 的定义是：

$$
\mathcal{L} = (\hat{y} - y)^2
$$

其中 $y$ 是真实值。初始损失的期望值为：

$$
E[\mathcal{L}_{\text{init}}] = E[(\hat{y} - y)^2]
$$

将 $\hat{y}$ 和 $y$ 视为随机变量，展开这个式子：

$$
E[(\hat{y} - y)^2] = E[\hat{y}^2 - 2\hat{y}y + y^2] = E[\hat{y}^2] - 2E[\hat{y}y] + E[y^2]
$$

由于$E[\hat{y}] \approx 0$，所以 $Var(\hat{y}) = E[\hat{y}^2] - (E[\hat{y}])^2 \approx E[\hat{y}^2]$。因为模型输出 $\hat{y}$ 与真实值 $y$ 在初始时是独立的，所以 $E[\hat{y}y] = E[\hat{y}]E[y] \approx 0 \cdot E[y] = 0$。

因此，初始损失简化为：

$$
E[\mathcal{L}_{\text{init}}] \approx Var(\hat{y}) + E[y^2]
$$

其中 $Var(\hat{y})$ 是模型初始输出的方差，通常很小。$E[y^2]$ 是数据集中真实标签的平方的均值。根据方差定义，$E[y^2] = Var(y) + (E[y])^2$。所以，初始的 **MSE** 损失约等于数据集中目标值 $y$ 的方差加上其均值的平方。如果对 $y$ 进行了标准化（均值为0，方差为1），那么初始损失应该约等于 $1$。

#### 结论：对于一个回归问题（MSELoss），初始损失应该约等于训练数据中目标值 $y$ 的二阶矩 $E[y^2]$。

## 4. 对比学习 (Contrastive Learning)

在自监督学习 (**SimCLR, MoCo**) 中，对比学习的损失函数是**InfoNCE Loss**。网络最后一层通常是 **L2** 归一化的投影头输出。

模型为查询 $q$，正样本 $k_+$，以及 $N$ 个负样本 ${k_i}$ 输出经过 **L2** 归一化的特征向量。

在 $t=0$ 时，所有这些向量在单位超球面上随机分布，此时任何两个随机向量之间的余弦相似度 $\text{sim}()$ 的期望为0。

**InfoNCE** 损失定义为:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\exp(\text{sim}(q, k_+) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(q, k_i) / \tau)}
$$

当所有 $\text{sim}()$ 都接近 0 时，$\text{sim}()/τ$ 也接近 0。因此，$e^{\text{sim}/\tau} \approx e^0 = 1$。此时初始损失计算为:

$$
E[\mathcal{L}_{\text{init}}] \approx -\log \frac{1}{1 + \sum_{i=1}^{N} 1} = -\log\left(\frac{1}{1+N}\right) = \log(N+1)
$$

#### 结论：对于对比学习问题（InfoNCE），初始损失应该约等于 $\log($负样本数量 $+ 1)$。
