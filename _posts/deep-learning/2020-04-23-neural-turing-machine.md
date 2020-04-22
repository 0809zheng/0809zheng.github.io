---
layout: post
title: '胶囊网络'
date: 2020-04-20
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e9d6c40c2a9a83be58a1455.jpg'
tags: 深度学习
---

> Capsule Networks.

- paper：Dynamic Routing Between Capsules
- arXiv：[https://arxiv.org/abs/1710.09829](https://arxiv.org/abs/1710.09829)


传统的神经网络的基本单元是**神经元(neuron)**，每一个神经元接收上一层神经元输出标量的线性组合，经过激活函数输出一个**标量(scalar)**。

**胶囊网络(Capsule Network)**用**胶囊(capsule)**代替了神经元，每一个胶囊接收上一层胶囊的输出$v^1,v^2,...$并输出一个**向量(vector)**$v$:

![](https://pic.downk.cc/item/5e9d6631c2a9a83be583701c.jpg)

向量$v$代表了输入数据内的一种pattern类型，$v$的每一个维度代表了这个pattern的一种**特征属性**，$v$的范数norm代表了这个pattern出现的**概率**(归一化后)。

如对于卷积神经网络，一个胶囊的输出$v$可能代表某一种图案（如眼睛、嘴巴），$v$的每一个维度代表了这个图案的方向、大小等特征属性。

### 1. 胶囊的实现过程

![](https://pic.downk.cc/item/5e9d67b6c2a9a83be584fa9d.jpg)

上图是一个胶囊内部的计算过程。

$v^1,v^2$表示上一个胶囊层的输出，经过仿射变换为$u^1,u^2$，变换参数$W^1$、$W^2$由反向传播学习得到：

$$ u^1 = W^1v^1, \quad u^2 = W^2v^2 $$

$u^1,u^2$经过加权的线性组合得到$s$，**耦合系数coupling coefficient**$c_1$、$c_2$是由**动态路由**算法得到的：

$$ s = c_1u^1 + c_2u^2 $$

对$s$进行**非线性压缩(Squash)操作**得到胶囊的输出$v$，这一操作只改变向量的大小，不改变向量的方向，可以看作一种激活函数：

$$ v = Squash(s) = \frac{\mid\mid s \mid\mid ^2}{1+\mid\mid s \mid\mid ^2} \frac{s}{\mid\mid s \mid\mid} $$

注意到$$\mid\mid s \mid\mid = 0$$时$$\mid\mid v \mid\mid = 0$$，$$\mid\mid s \mid\mid → ∞$$时$$\mid\mid v \mid\mid = 1$$，这一操作对$v$的范数实现了归一化，表示存在的概率。

胶囊与神经元的对比如下：

![](https://pic.downk.cc/item/5e9d6d74c2a9a83be58b5302.jpg)

### 2. 动态路由
胶囊间的**动态路由(dynamic routing)**机制可以确保每一层胶囊输出的特征向量被正确地发送到下一层中对应的胶囊。

对某一个胶囊，需要计算**耦合系数**$c_1,c_2,...$，设算法执行$T$次。

引入上一层的胶囊应该耦合到本胶囊的**对数先验概率**$b_1,b_2,...$，并初始化为0：

$$ b_i^0 = 0, \quad i=1,2,... $$

对于某一次迭代$t$，耦合系数总和为1，计算如下：

$$ c_1^t, c_2^t, ... = softmax(b_1^{t-1},b_2^{t-1},...) $$

并对先验概率进行修正：

$$ s^t = c_1^tu^1 + c_2^tu^2 $$

$$ a^t = Squash(s^t) $$

$$ b_i^t = b_i^{t-1} + a^t·u^i, \quad i=1,2,... $$

由上式可以看出，如果本次循环得到的向量与上一层某个胶囊的输出向量**一致性**较高，则对应的先验概率被修正增大。

经过$T$次循环得到最终的耦合系数：

$$ c_i = b_i^T, \quad i=1,2,... $$

整个路由过程不仅仅是存在于胶囊网络的**训练**过程中，也存在于**验证**和**测试**过程，而且需要迭代多次。

动态路由的更新过程如下图所示，与RNN非常类似：

![](https://pic.downk.cc/item/5e9d733fc2a9a83be5915981.jpg)

### 3. 网络特点
- **不变性invariance**：输入发生微小变化时输出不变；
- **等变性equivariance**：输入发生微小变化时输出随输入的变化规律而改变；

最大池化(max pool)具有不变性，但不具有等变性；即输入变换时输出是不变的，从而丢失了位置等信息：

![](https://pic.downk.cc/item/5e9d726dc2a9a83be5907eae.jpg)

胶囊网络既具有不变性又具有等变性；即输入变化时输出能够捕捉到这种微小的变化($v^1$)，但选择性忽略，最终结果不变($$\mid\mid v^1 \mid\mid$$)：

![](https://pic.downk.cc/item/5e9d72b5c2a9a83be590d9cb.jpg)

### 4. CapsNet
胶囊网络也可以引入卷积神经网络，即将卷积核换成胶囊。

![](https://pic.downk.cc/item/5e9d7429c2a9a83be5924fc3.jpg)

对于**胶囊网络(CapsNet)**，用于图像分类时每一个胶囊输出是对一类的预测结果，如上图所示。

为了实现同时对多个对象的识别，每类目标对象对应的胶囊应分别使用**边际损失(margin loss)**函数得到类损失$L_c$，则总边际损失是所有类损失之和。

![](https://pic.downk.cc/item/5e9d80f5c2a9a83be59fc61e.jpg)

- $T_c$是表示$c$类目标对象是否存在，当$c$类目标对象存在时为1，不存在时为0；
- $m^+$是上界，惩罚假阳性，即预测$c$类存在但真实不存在，一般取0.9；
- $m^-$是下界，惩罚假阴性，即预测$c$类不存在但真实存在，一般取0.1；
- $λ$为不存在的对象类别的损失的下调权重，避免最开始从不存在分类对应的胶囊输出的特征向量中学习，一般取0.5。

在论文还引入了重构网络减少分类误差。

### 5. 实验结果

**(1).MNIST**

![](https://pic.downk.cc/item/5e9d74f3c2a9a83be593072e.jpg)

在MNIST数据集上，测试集加了一些仿射变换，使得训练与测试**不匹配(mismatch)**。CapsNet仍然具有很好的鲁棒性。

**(2).向量的维度**

![](https://pic.downk.cc/item/5e9d752ac2a9a83be5933bf6.jpg)

向量的每一个维度代表不同的特征信息。

**(3).MultiMNIST**

![](https://pic.downk.cc/item/5e9d7572c2a9a83be593843d.jpg)

- 图像的上半部分代表输入图像
- 图像的下半部分代表重构图像
- R代表重构数字
- L代表真实标签

在MultiMNIST数据集上，对于输入标签中存在的数字可以很好的重构，但是对于输入中不存在的数字则几乎没有重构结果。


