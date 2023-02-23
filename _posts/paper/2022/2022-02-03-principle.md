---
layout: post
title: 'Towards Principled Methods for Training Generative Adversarial Networks'
date: 2022-02-03
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/633103e316f2c2beb14952e4.jpg'
tags: 论文阅读
---

> 训练生成对抗网络的原则性方法.

- paper：[Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/abs/1701.04862)

**GAN**由判别器$D$和生成器$G$组成，其目标函数为：

$$ \begin{aligned} \mathop{ \min}_{G} \mathop{\max}_{D} L(G,D) & =  \Bbb{E}_{x \text{~} P_{data}(x)}[\log D(x)] + \Bbb{E}_{z \text{~} P_{Z}(z)}[\log(1-D(G(z)))] \\ & =  \Bbb{E}_{x \text{~} P_{data}(x)}[\log D(x)] + \Bbb{E}_{x \text{~} P_{G}(x)}[\log(1-D(x))] \end{aligned} $$

# 1. GAN在训练时遇到的问题

### ⚪ 低维支撑集 Low dimensional support set

真实分布$$P_{data}(x)$$与生成分布$$P_G(x)$$的**支撑集(support set)**位于低维**流形(manifold)**上，从而导致**GAN**训练的不稳定性。

- 支撑集：定义在集合$X$上的实值函数$f$的支撑集是指$X$的一个子集，满足$f$恰好在这个子集上非$0$。
- 流形：在每个点附近局部类似于欧氏空间的拓扑空间。当欧氏空间的维数为$n$时，称为$n$流形($n$-**manifold**)。

许多真实世界的数据集$$P_{data}(x)$$的维度只是表面上看起来很高，事实上它们通常集中在一个低维流形上，这也是流形学习的基本假设。考虑到真实世界的图像一旦主题被固定，图像本身就要遵循很多限制，这些限制使得图像不可能具有高维的自由形式。比如一个狗的图像应该具有两个耳朵和一条尾巴，摩天大楼的图像应该具有很高的建筑。

另一方面，生成分布$$P_G(x)$$通常也位于一个低维流形上。当输入噪声变量$z$的维度给定时(如$100$)，即使生成器被要求生成分辨率再高的图像(如$64 \times 64$)，图像的所有像素(如$4096$个)的颜色分布都是由这个低维随机向量定义的，几乎不可能填满整个高维空间。

由于真实分布$$P_{data}(x)$$与生成分布$$P_G(x)$$都在一个低维的流形上，它们几乎肯定是不相交的。当两者具有不相交的支撑集时，总是能够找到一个完美的判别器，完全正确地区分真实样本和生成样本。下面给出高维空间中存在的两个低维流形的情况，分别为三维空间中的直线与平面，不难观察发现它们几乎不可能有重叠。

![](https://pic.imgdb.cn/item/633107ba16f2c2beb14cbe22.jpg)

### ⚪ 梯度消失 Vanishing gradient

当判别器表现较好时，对于从真实分布$$P_{data}(x)$$采样的真实样本判别器输出$D(x)=1$；对于从生成分布$$P_G(x)$$采样的生成样本判别器输出$D(x)=0$。此时损失函数$L$趋近于$0$，在训练过程中梯度也趋近于$0$。下图体现出训练得更好的判别器（对应更大的训练轮数），梯度消失的更快：

![](https://pic.imgdb.cn/item/6331125716f2c2beb156fc56.jpg)

因此**GAN**的训练过程面临进退两难：
- 如果判别器表现较差，则生成器没有准确的反馈，损失函数不能代表真实情况；
- 如果判别器表现较好，则损失函数的梯度趋近于$0$，训练过程变得非常慢甚至卡住。


# 2. 改进GAN的训练过程

## (1) 增加噪声 adding noise

通过上述讨论可知真实分布$$P_{data}(x)$$与生成分布$$P_G(x)$$都落在高维空间中的低维流形上。为了人为地扩大分布的范围，使得两个概率分布有更大的概率重叠，可以在判别器的输入中增加连续噪声。

## (2) 使用更好的分布相似度度量 better metric of distribution similarity

**GAN**的损失函数衡量真实分布$$P_{data}(x)$$与生成分布$$P_G(x)$$之间的[<font color=blue>JS散度</font>](https://0809zheng.github.io/2020/02/03/kld.html#3-js%E6%95%A3%E5%BA%A6)，**JS**散度在两个分布不相交时没有意义。可以选择具有更平滑的值空间的分布度量指标，比如[<font color=blue>Wasserstein距离</font>](https://0809zheng.github.io/2022/05/16/Wasserstein.html)。