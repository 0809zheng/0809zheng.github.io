---
layout: post
title: '深度学习中的Regularization'
date: 2020-03-03
author: 郑之杰
cover: 'http://p0.ifengimg.com/pmop/2018/0117/FF63C065C57341C7727412791090885E7EB230BD_size23_w900_h375.jpeg'
tags: 深度学习
---

> Regularization in Deep Learning.

1. Background
2. L2 Regularization
3. L1 Regularization
4. Elastic Net Regularization
5. Weight Decay
6. Early Stopping
7. Dropout
8. DropConnect
9. Data Augmentation
10. Label Smoothing



# 1. Background
机器学习（尤其是监督学习）的问题包括优化和泛化问题。

优化是指在已有的数据集上实现最小的训练误差（training error），泛化是指在未训练的数据集（通常假设与训练集同分布）上实现最小的泛化误差（generalize error）。深度神经网络具有很强的拟合能力，在训练数据集上错误率通常较低，但是容易过拟合（overfitting）。

正则化（Regularization）是指通过限制模型的复杂度，从而避免过拟合，提高模型的泛化能力的方法。

# 2. L2 Regularization
L2 Regularization通过约束参数的L2范数（L2-norm）减小过拟合。

L2正则化的优化问题可写作：

$$ θ^*=argmin_θ \frac{1}{N} \sum_{n=1}^{N} {L(y^n,f(x^n))}+λ \mid\mid θ \mid\mid _2^2$$

# 3. L1 Regularization
L1 Regularization通过约束参数的L1范数（L1-norm）减小过拟合。

L1正则化的优化问题可写作：

$$ θ^*=argmin_θ \frac{1}{N} \sum_{n=1}^{N} {L(y^n,f(x^n))}+λ \mid\mid θ \mid\mid _1$$

如下图所示，蓝圈为优化函数的等高线，棕色区域为满足L2/L1正则化约束的可行域。当等高线与可行域相交时，L1正则化会优先相交于坐标轴上。故L1正则化会使参数具有稀疏性（sparse）。
![](https://charlesliuyx.github.io/2017/10/03/%E3%80%90%E7%9B%B4%E8%A7%82%E8%AF%A6%E8%A7%A3%E3%80%91%E4%BB%80%E4%B9%88%E6%98%AF%E6%AD%A3%E5%88%99%E5%8C%96/Dq2.png)

# 4. Elastic Net Regularization
- paper:[Regularization and Variable Selection via the Elastic Net](https://www.jstor.org/stable/3647580)

弹性网络正则化（Elastic Net Regularization）是指同时加入L2和L1正则化：

$$ θ^*=argmin_θ \frac{1}{N} \sum_{n=1}^{N} {L(y^n,f(x^n))}+λ_2 \mid\mid θ \mid\mid _2^2+λ_1 \mid\mid θ \mid\mid_1$$

# 5. Weight Decay
权重衰减（Weight Decay）是指在参数更新时引入一个衰减系数β：

$$ θ^t=(1-β)θ^{t-1}-αg^t $$

在随机梯度下降中，Weight Decay与L2 Regularization等价；但在较为复杂的优化方法（如Adam）中，[两者并不等价](https://arxiv.org/abs/1711.05101v1)。

# 6. Early Stopping
**Early Stop**是指训练时，当观察到验证集上的错误不再下降，就停止迭代。

具体停止迭代的时机，可参考[Early stopping-but when?](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5)。

使用Early Stop需要使用到验证集（Validation Set），这也就意味着在训练过程中会有一部分数据无法进行训练。在完成Early Stop后将验证集中的数据加入到训练集中，进行额外的训练。

**①策略一**：使用验证集确定训练步数$t$，再次训练$t$次：

![](https://pic.downk.cc/item/5ea569a8c2a9a83be5f2d016.jpg)

**缺点**：不能确定按照Early Stop确定的训练最佳步数再次训练时仍能得到一个最佳的训练。

**②策略二**：使用验证集确定损失值$ε$，再次训练使损失值$<ε$：

![](https://pic.downk.cc/item/5ea569bbc2a9a83be5f2f081.jpg)

**缺点**：无法保证继续训练是否能达到之前的目标值。

# 7. Dropout
- paper：[ Dropout: A simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html)

**Dropout**是指在训练深度神经网络时，随机丢弃一部分神经元。即对某一层设置概率p，对该层的每个神经元以概率p判断是否要丢弃。
![](https://pic.downk.cc/item/5e7de4c1504f4bcb04745d05.png)

训练时，激活神经元的平均数量是原来的p倍；而在测试时所有神经元都被激活，故测试时需将该层神经元的输出乘以p。

**Inverted Dropout**是指在训练时对某一层按概率p随机丢弃神经元之后将该层的输出除以p；测试时不需再做处理。

理解Dropout：

（1）Ensemble角度

每一次Dropout相当于从原网络中生成一个子网络，每次迭代相当于训练一个不同的子网络；最终的网络可以看作这些子网络的集成；

（2）Regularization角度

每一次Dropout相当于为原网络引入噪声，测试时通过平均抵消掉噪声，每次训练不过分依赖于某一个神经元，增强网络的泛化能力；

（3）Bayesian角度

Bayesian学习假设参数θ为随机变量，先验分布为q(θ)，Bayesian方法的预测为：

$$ E_{q(θ)}(y)=\int_q^{} {f(x;θ)q(θ)dθ}
              ≈\frac{1}{M}\sum_{m=1}^{M} {f(x;θ_m)}$$
			  
不等号由Monte Carlo方法得到。$θ_m$是第m次Dropout的网络参数，看作对全部参数θ的一次采样。

在循环神经网络中，使用[Variational Dropout](https://arxiv.org/abs/1512.05287)。

# 8. DropConnect
**DropConnect**丢弃神经元之间的连接，与Dropout类似：

![](https://pic.downk.cc/item/5ea56859c2a9a83be5f1ad39.jpg)

# 9. Data Augmentation
**数据增强（Data Augmentation）**通过对样本集的操作增加数据量（相当于对样本集加入随机噪声），提高模型鲁棒性，避免过拟合。

图像数据的增强方法主要有：
1. **空间变换**：
- 旋转 Rotation
- 翻转 Flip
- 缩放 Zoom
- 平移 Shift
2. **像素变换**：
- 对比度扰动
- 饱和度扰动
- 颜色变换
- 加噪声 Noise

# 10. Label Smoothing
标签平滑（Label Smoothing）首先在[Inceptionv3](https://arxiv.org/abs/1512.00567)中被提出。

标签平滑是对样本的标签引入一定的噪声。

一个样本的标签一般用one-hot向量表示：

$$ y=(0,...,0,1,0,...,0)^T $$

这是一种Hard Target，若样本标签是错误的，会导致严重的过拟合。引入噪声对标签进行平滑，假设样本以ε的概率为其他类：

$$ y'=(\frac{ε}{K-1},...,\frac{ε}{K-1},1-ε,\frac{ε}{K-1},...,\frac{ε}{K-1})^T $$

上述标签是一种Soft Target，但没有考虑标签的相关性。一种更好的做法是按照类别相关性赋予其他标签不同的概率。
