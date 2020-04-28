---
layout: post
title: '深度学习的可解释性'
date: 2020-04-28
author: 郑之杰
cover: ''
tags: 深度学习
---

> Explainable Deep Learning.

简单的机器学习模型具有一定的**可解释性(explanation or  interpretation)**，如线性回归、决策树；而深度学习模型过于复杂，可解释性差。

所谓可解释性，并不是指对模型是如何工作的完全了解，而是从人类的角度给出可以接受的解释。

将每一个输入样本表示为$$\{x_1,...,x_N\}$$。对于计算机视觉，$x_i$可以表示每一个像素或每一部分像素；对于自然语言处理，$x_i$可以表示每一个单词或每一个字符。

**本文目录**：
1. Local Explanation
2. Global Explanation
3. Model Explanation

# 1. Local Explanation
**Local Explanation**是指改变输入中的每一个$x_i$，对输出会有怎样的影响。

### (1)Saliency Map
计算并显示损失对每一个输入的梯度：

![](https://pic.downk.cc/item/5ea7e807c2a9a83be5c21fc9.jpg)

基于梯度的方法的缺陷是：
1. **Gradient Saturation**：$x_i$对输出影响的梯度会逐渐趋于饱和；
2. **Noisy gradient**：随机噪声会影响结果。

为解决上述问题，又提出了以下方法。

### (2)Integrated gradient
- [paper](https://arxiv.org/abs/1611.02639)

### (3)DeepLIFT
- [paper](https://arxiv.org/abs/1704.02685)

### (4)SmoothGrad
**SmoothGrad**是指预先对输入加上噪声，综合使用多次输入的结果：

$$ \hat{M}_c(x) = \frac{1}{n} \sum_{1}^{n} {M_c(x+N(0,σ^2))} $$

# 2. Global Explanation
**Global Explanation**是指对于期望的输出结果，对应的输入$x_i$是怎样的。

### (1)Activation Maximization
这种方法是调整**输入**使某一层的神经元**激活值**或输出层的**输出值**最大化：

$$ x^* = argmax_{(x)}y_i $$

直接用上式可能会得到噪声结果：

![](https://pic.downk.cc/item/5ea7ec9ac2a9a83be5c80063.jpg)

因此对生成的像素进行限制：

$$ R(x) = -\sum_{i,j}^{} {\mid x_{ij} \mid} $$

$$ x^* = argmax_{x}y_i+R(x) $$

![](https://pic.downk.cc/item/5ea7ed10c2a9a83be5c8539e.jpg)

### (2)Constraint from Generator
为了使生成的图像更加接近真实图像而不是噪声，也可以预先训练一个**生成器**控制图像的生成：

![](https://pic.downk.cc/item/5ea7edc9c2a9a83be5c90a55.jpg)


# 3. Model Explanation
**Model Explanation**是指用一个(可解释的)模型去解释另一个(复杂度)模型。

### (1)Linear Model：Local Interpretable Model-Agnostic Explanations(LIME)
一种简单的模型解释方法是用线性模型解释复杂的模型：

![](https://pic.downk.cc/item/5ea7f018c2a9a83be5cbc9aa.jpg)

线性模型很难完整的解释复杂的模型，但是可以模仿一个局部区域：

![](https://pic.downk.cc/item/5ea7f0aac2a9a83be5cc661b.jpg)

**Local Interpretable Model-Agnostic Explanations(LIME)**的实现过程：

1. 给定一个待解释的数据点；
2. 在该数据点附近采样几个点；
3. 用这些数据点拟合一个线性模型；
4. 解释这个线性模型。

**LIME**应用于图像：
1. 把图像分割成一些**子区域segments**；
![](https://pic.downk.cc/item/5ea7f15ec2a9a83be5cd04f4.jpg)
2. 任意使用一些子区域喂入复杂模型，计算得分或概率；
![](https://pic.downk.cc/item/5ea7f1eac2a9a83be5cd9858.jpg)
3. 把使用若干子区域的图像转换成向量$$[x_1,...,x_M]$$，其中$x_m$表示使用了第$m$个区域，使用线性模型拟合；
![](https://pic.downk.cc/item/5ea7f31ac2a9a83be5ceb1c0.jpg)
4. 最终得到模型$$y=w_1x_1+...+w_Mx_M$$，对模型进行一些解释：

- $$w_m ≈ 0$$代表这个子区域对结果没有影响；
- $$w_m > 0$$代表这个子区域支持结果的出现；
- $$w_m < 0$$代表这个子区域反对结果的出现。

### (2)Decision Tree：Tree regularization
- [paper](https://arxiv.org/pdf/1711.06178.pdf)

用决策树模型解释复杂的模型：

![](https://pic.downk.cc/item/5ea7f432c2a9a83be5cfd6f5.jpg)

希望决策树模型尽可能简单,在训练模型时，就加上对解释决策树的复杂度限制：

$$ θ^* = argmin_{θ}L(θ) + λO(T_θ) $$