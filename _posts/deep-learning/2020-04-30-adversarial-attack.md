---
layout: post
title: '对抗攻击'
date: 2020-04-30
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eaa6adcc2a9a83be510f7c9.jpg'
tags: 深度学习
---

> Adversarial Attack.

**本文目录**：
1. Attack
2. Attack Approaches
3. Defense


# 1. Attack
对于一个已训练好的模型$f_θ$，输入样本$x^0$会得到正确的预测结果$y^{true}$，

**攻击attack**指的是对输入$x^0$做微小的改变$x'$，输入网络后将不能得到正确的结果$y^{true}$，甚至得到指定错误的结果$y^{false}$。

- **Non-targeted Attack**：固定模型，修改输入，使预测结果偏离正确结果：

$$ x^* = argmin_{(x')}L(x') = argmin_{(x')}-Loss(y',y^{true}) $$

- **Targeted Attack**：固定模型，修改输入，使预测结果偏离正确结果，接近预定的错误结果：

$$ x^* = argmin_{(x')}L(x') = argmin_{(x')}-Loss(y',y^{true})+Loss(y',y^{false}) $$

通常期望输入的变化是微小的：

$$ d(x^0,x') ≤ ε $$

![](https://pic.downk.cc/item/5eaa3ec2c2a9a83be5f6801e.jpg)

**Constraint**函数$d(x^0,x')$可以使用：
- $L2-norm$：$$d(x^0,x')=\mid\mid x^0-x' \mid\mid_2$$
- $L-infinity$：$$d(x^0,x')=\mid\mid x^0-x' \mid\mid_∞$$

$L2$范数计算差值图像每个像素的平方和；而$L-∞$范数计算差值图像绝对值最大的像素；后者不允许某个像素值的变化特别大：

![](https://pic.downk.cc/item/5eaa4521c2a9a83be5fabb6e.jpg)

# 2. Attack Approaches
根据是否已知模型及其参数，攻击方式可分为:
- **White Box Attack**:已知模型
- **Black Box Attack**:模型未知

### (1)White Box Attack
当已知模型时，问题转化为一个约束优化问题：

$$ x^* = argmin_{d(x^0,x') ≤ ε}L(x') $$

**1. Gradient Descent**

从原始图像$x^0$开始，进行从$t=1$到$T$个迭代：

$$ x^t = x^{t-1} - η grad(L(x^{t-1})) $$

当$$d(x^0,x^t) > ε$$时，进行修正：
- 对于$L2$范数,寻找满足$$d(x^0,x^t) ≤ ε$$且与原$x^t$最接近的$x^t$。
- 对于$L-∞$范数,把$x^t$超过$ε$的元素限制到$ε$。

![](https://pic.downk.cc/item/5eaa6a37c2a9a83be5107b44.jpg)

**2. Fast Gradient Sign Method (FGSM)**
- [paper](https://arxiv.org/abs/1412.6572)

**FGSM**只进行一次梯度下降：

$$ x' = x^0 - ε Δx $$

$$ Δx = \begin{bmatrix} sign(\frac{\partial L}{\partial x_1}) \\ sign(\frac{\partial L}{\partial x_2}) \\ sign(\frac{\partial L}{\partial x_3}) \\ ... \\ \end{bmatrix} $$

该方法可以认为使用了无穷大的学习率，使用$L-∞$范数约束：

![](https://pic.downk.cc/item/5eaa6c64c2a9a83be5122d25.jpg)

**3. One Pixel Attack**
- [paper](https://arxiv.org/abs/1710.08864)

该方法只修改图像的一个像素值，即：

$$ d(x^0,x') = \mid\mid x^0-x' \mid\mid_0 = 1 $$

其中$L0$范数表示非零元素的个数。

该方法的出发点是对于图像分类任务，尤其是类别很多时，攻击不需要使正确的类别得分大幅度降低，只需要让某个错误类别的得分超过正确类别即可。

实现方法：**Differential Evolution**

**4. Universal Adversarial Attack**
- [paper](https://arxiv.org/abs/1610.08401)

构建一个通用的攻击信号可以对不同图像进行攻击。

### (2)Black Box Attack
- [paper](https://arxiv.org/pdf/1611.02770.pdf)

当模型未知时，如果能获得模型的训练数据，用训练数据自行训练一个**proxy network**；

在**proxy network**上获得的**attack object**，通常可以对原模型进行攻击。

![](https://pic.downk.cc/item/5eaa6d11c2a9a83be512a084.jpg)

如果没有训练数据，则自己利用原网络构造一些输入-输出对作为训练数据。

# 3. Defense
**Defense**可以分为被动式和主动式：
- **Passive defense**：不修改模型，找到异常的图像，类似于**异常检测Anomaly Detection**。
- **Proactive defense**：训练模型，使其对对抗攻击具有鲁棒性。

### (1)Passive defense
**1. Smoothing**

对输入图像进行平滑滤波：

![](https://pic.downk.cc/item/5eaa70bac2a9a83be51577e8.jpg)

**2. Feature Squeeze**
- [paper](https://arxiv.org/abs/1704.01155)

用多个$filter$进行检测：

![](https://pic.downk.cc/item/5eaa711fc2a9a83be515b5f7.jpg)

**3. Randomization at Inference Phase**
- [paper](https://arxiv.org/abs/1711.01991)

把输入图像做一些随机的**resize**和**padding**：

![](https://pic.downk.cc/item/5eaa715ec2a9a83be515dc01.jpg)

### (2)Proactive defense
给定训练数据$$X={(x^1,y^1),...,(x^N,y^N)}$$，

从$t=1$到$T$进行迭代：

每次迭代时对每个样本$x^n$，构造一个对抗攻击的样本$\hat{x}^n$，将其作为新的训练数据，

每轮得到新的训练数据$$X'={(\hat{x}^1,y^1),...,(\hat{x}^N,y^N)}$$，更新模型。

**为什么要进行$T$次迭代？**：每次使用对抗攻击的样本更新模型时可能会产生新的漏洞。
