---
layout: post
title: 'Neural Architecture Search without Training'
date: 2020-09-10
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f5a0b56160a154a67c3e647.jpg'
tags: 论文阅读
---

> 不需要训练网络的神经结构搜索.

- paper：Neural Architecture Search without Training
- arXiv：[link](https://arxiv.org/abs/2006.04647)

![](https://pic.downk.cc/item/5f5a0e50160a154a67c50e46.jpg)

现有的神经结构搜索方法针对每一种搜索得到的结构都需要使用训练集训练得到准确率，这一过程速度慢而且代价高。作者提出了一种不需要模型训练便可以进行结构搜索的方法。

作者使用**NAS-Bench-201**数据集进行结构搜索。该数据集提出的网络结构框架如下图所示。每一个单元共有六条边，每个边可以选择五种操作（$1×1$卷积、$3×3$卷积、$3×3$平均池化、跳跃连接、置零），网络共有$5^6=15625$种可能的结构。

![](https://pic.downk.cc/item/5f5a0fbf160a154a67c58919.jpg)

作者通过测试数据点的线性映射（**linear map**）之间的相关性来计算结构得分。具体地，将网络建模成一个线性模型，即对于输入数据点$x_i$，输出为$f_i=w_ix_i$。进一步构建输入数据集$$X=\{x_n\}_{n=1}^{N}$$的**Jacobian**矩阵：$J=(\frac{\partial f_1}{\partial x_1},\frac{\partial f_2}{\partial x_2},...,\frac{\partial f_N}{\partial x_N})^T$。之后计算协方差矩阵$C_J=(J-M_J)(J-M_J)^T$。最后计算相关矩阵$$\Sigma_J = (\frac{(C_J)_{i,j}}{\sqrt{(C_J)_{i,i}(C_J)_{j,j}}})$$。

作者对已经在**CIFAR-10**上面训练过的结构进行实验。实验发现，对于验证准确率越高的训练模型，数据线性映射之间的相关性越低（可以理解为对于不同的数据点，训练模型表现得像不同的线性模型）：

![](https://pic.downk.cc/item/5f5a12c4160a154a67c6d869.jpg)

作者提出了一个筛选收缩结构的评价得分。记$σ_{J,N}$是相关矩阵$$\Sigma_J$$的$N$个特征值，则得分表示为（其中$k=10^{-5}$为数值稳定性的常数）：

$$ S = -\sum_{i=1}^{N} {[log(σ_{J,i}+k)+(σ_{J,i}+k)^{-1}]} $$

经验证，准确率越高的模型结构上述定义的得分越高：

![](https://pic.downk.cc/item/5f5a13d7160a154a67c755d8.jpg)

作者提出的**training-free**方法，尽管比参数不共享的方法得到的验证准确率低，但是大大压缩了搜索时间：

![](https://pic.downk.cc/item/5f5a140f160a154a67c76cdc.jpg)

