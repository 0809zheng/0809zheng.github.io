---
layout: post
title: '递归神经网络'
date: 2020-03-08
author: 郑之杰
cover: ''
tags: 深度学习
---

> Recursive Neural Networks.

**递归神经网络（Recursive Neural Network）**是循环神经网络在**有向无环图**上的扩展。

下图是递归神经网络的一般结构：

![](https://pic.downk.cc/item/5ea14499c2a9a83be5c09f98.jpg)

网络有三个隐藏层$h_1,h_2,h_3$，共用函数体的参数，计算关系如下：

$$ h_1 = σ(W \begin{bmatrix} x_1 \\ x_2 \\ \end{bmatrix}+b) $$

$$ h_2 = σ(W \begin{bmatrix} x_3 \\ x_4 \\ \end{bmatrix}+b) $$

$$ h_3 = σ(W \begin{bmatrix} h_1 \\ h_2 \\ \end{bmatrix}+b) $$

$$ y = f(W'h_3+b') $$

递归神经网络主要用来建模**自然语言句子的语义**。

给定一个句子的语法结构（一般为树状结构），可以使用递归神经网络来按照句法的组合关系来合成一个句子的语义。

句子中每个短语成分又可以分成一些子成分，即每个短语的语义都可以由它的子成分语义组合而来，并进而合成整句的语义。

递归神经网络可以退化为循环神经网络：

![](https://pic.downk.cc/item/5ea1466ac2a9a83be5c35961.jpg)

# 1. Recursive Neural Tensor Network
简单的递归神经网络两个输入之间没有直接的关系：

![](https://pic.downk.cc/item/5ea14741c2a9a83be5c4bd9a.jpg)

**Recursive Neural Tensor Network**建立了两个输入之间的关系。

把输入$x_1$、$x_2$拼接成张量$x$：

![](https://pic.downk.cc/item/5ea147a5c2a9a83be5c566b0.jpg)

每一个虚线框内计算得到一个标量，根据输出维度决定使用几个虚线框。

# 2. Matrix-Vector Recursive Network
**Matrix-Vector Recursive Network**对每个输入编码成自己的部分和影响其他输入的部分：

![](https://pic.downk.cc/item/5ea14821c2a9a83be5c61511.jpg)

![](https://pic.downk.cc/item/5ea14882c2a9a83be5c6a23f.jpg)

# 3. Tree LSTM
**Tree LSTM**把每一个函数体用LSTM替代:

![](https://pic.downk.cc/item/5ea14985c2a9a83be5c843f8.jpg)