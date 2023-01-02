---
layout: post
title: '深度学习中的正则化方法(Regularization)'
date: 2020-03-03
author: 郑之杰
cover: 'http://p0.ifengimg.com/pmop/2018/0117/FF63C065C57341C7727412791090885E7EB230BD_size23_w900_h375.jpeg'
tags: 深度学习
---

> Regularization in Deep Learning.

深度学习所处理的问题包括优化问题和泛化问题。**优化(optimization)**问题是指在已有的数据集上实现最小的训练误差；而**泛化(generalization)**问题是指在未经过训练的数据集(通常假设与训练集同分布)上实现最小的**泛化误差(generalize error)**。通常深度神经网络具有很强的拟合能力，因此训练误差较低，但是容易**过拟合(overfitting)**，导致泛化误差较大。

**正则化(Regularization)**指的是通过限制模型的**复杂度**，从而降低对输入或者参数的敏感性，避免过拟合，提高模型的泛化能力。对模型复杂度的限制包括约束模型参数(或者约束目标函数)、约束网络结构、约束优化过程。

- 约束**模型参数**：在目标函数中增加模型参数的正则化项，包括**L2**正则化, **L1**正则化, 弹性网络正则化, 谱正则化
- 约束**网络结构**：在网络结构中添加噪声，包括随机深度, **Dropout**
- 约束**优化过程**：在优化过程中施加额外步骤，包括**Early Stop**, 

# 1. 约束模型参数

## ⚪ L2正则化 L2 Regularization

**L2**正则化通过约束参数的**L2**范数（**L2-norm**）减小过拟合。带有**L2**正则化的优化问题可写作：

$$ w^*= \mathop{\arg\min}_w \frac{1}{N} \sum_{n=1}^{N} {L(y_n,f(x_n);w)}+λ ||w||_2^2$$

### (1) 讨论：L2正则化等价于约束参数矩阵的Frobenius范数

若将模型参数表示为矩阵$W$，则**L2**正则化等价于约束矩阵$W$的**Frobenius**范数：

$$ \sum_{i,j} w_{ij}^2 = ||W||_F^2 $$

矩阵$W$的**Frobenius**范数 $\|\|W\|\|_F$是矩阵的[谱范数](https://0809zheng.github.io/2020/09/19/snr.html) $\|\|W\|\|_2$的一个上界。约束**Frobenius**范数能够使网络更好地满足[Lipschitz连续性](https://0809zheng.github.io/2022/10/11/lipschitz.html)，从而降低模型对输入扰动的敏感性，增强模型的泛化能力。

下面证明**Frobenius**范数是谱范数的上界。对于矩阵$W$和向量$x$，根据柯西不等式：

$$ ||Wx|| \leq ||W||_F \cdot ||x|| $$

而谱范数的定义：

$$ ||W||_2 = \mathop{\max}_{x \neq 0} \frac{||Wx||}{||x||} $$

因此有：

$$ ||W||_2 \leq  ||W||_F $$

### (2) 讨论：L2正则化等价于参数服从正态分布的最大后验估计

从贝叶斯角度出发，把参数$w$看作随机变量，假设其先验概率$p(w)$服从正态分布$N(0,σ_0^2)$。

由贝叶斯定理可得参数$w$的后验概率$p(w\|x,y)$：

$$ p(w |x, y) = \frac{p(x,y | w)p(w)}{p(y)} \propto p(x,y | w)p(w) $$

参数$w$的最大后验估计为：

$$ \begin{aligned} \hat{w} &= \mathop{\arg \max}_{w}\log p(w |x, y) = \mathop{\arg \max}_{w}\log p(x,y | w)p(w) \\ &= \mathop{\arg \max}_{w} \log p(x,y | w) +\log\frac{1}{\sqrt{2\pi}σ_0} \exp(-\frac{w^Tw}{2σ_0^2})  \\ &\propto \mathop{\arg \max}_{w} \log p(x,y | w)-\frac{w^Tw}{2σ_0^2} \\ &= \mathop{\arg \min}_{w} -\log p(x,y | w)+\frac{1}{2σ_0^2}||w||_2^2 \end{aligned} $$

因此参数服从正态分布的最大后验估计等价于引入**L2**正则化。

### (3) 讨论：L2正则化与权重衰减

在标准的梯度下降算法中，应用**L2**正则化后参数的更新过程为：

$$ \begin{aligned} w^{(t+1)} &\leftarrow w^{(t)} - \alpha \nabla_w[L(w)+λ ||w||_2^2] \\ &\leftarrow (1-2\alpha \lambda)w^{(t)} - \alpha \nabla_w L(w) \end{aligned} $$

上式相当于在参数更新时首先对参数引入一个衰减系数$\alpha \lambda$，因此也称为**权重衰减(Weight Decay)**正则化。

值得一提的是，在**Adam**等自适应梯度更新算法中，使用梯度的二阶矩进行梯度缩放。因此对于具有较大梯度的权重，其**L2**正则化项会被缩小，从而与权重衰减正则化不等价。[AdamW算法](https://0809zheng.github.io/2020/11/28/adamw.html)则将权重衰减从梯度更新过程中解耦，使得所有权重以相同的正则化程度进行衰减：

$$ \begin{aligned} w^{(t+1)} &\leftarrow w^{(t)} - \alpha (\frac{\hat{m}^{(t)}(w)}{\sqrt{\hat{v}^{(t)}(w)}+\epsilon}+λ w^{(t)}) \end{aligned} $$



## ⚪ L1正则化 L1 Regularization
**L1**正则化通过约束参数的**L1**范数（**L1-norm**）减小过拟合。带有**L1**正则化的优化问题可写作：

$$ w^*= \mathop{\arg\min}_w \frac{1}{N} \sum_{n=1}^{N} {L(y_n,f(x_n);w)}+λ ||w||_1$$

如下图所示，蓝圈为优化函数的等高线，棕色区域为满足**L2/L1**正则化约束的可行域。当等高线与可行域相交时，L1正则化会优先相交于坐标轴上。故L1正则化会使参数具有稀疏性（**sparse**）。

![](https://pic.imgdb.cn/item/639b1a2cb1fccdcd36c53a4e.jpg)

### (1) 讨论：L1正则化等价于参数服从拉普拉斯分布的最大后验估计

从贝叶斯角度出发，把参数$w$看作随机变量，假设其先验概率$p(w)$服从拉普拉斯分布：

$$ w \text{~} L(0,σ_0^2) = \frac{1}{2σ_0^2} \exp(-\frac{|w|}{σ_0^2}) $$

由贝叶斯定理可得参数$w$的后验概率$p(w\|x,y)$：

$$ p(w |x, y) = \frac{p(x,y | w)p(w)}{p(y)} \propto p(x,y | w)p(w) $$

参数$w$的最大后验估计为：

$$ \begin{aligned} \hat{w} &= \mathop{\arg \max}_{w}\log p(w |x, y) = \mathop{\arg \max}_{w}\log p(x,y | w)p(w) \\ &= \mathop{\arg \max}_{w} \log p(x,y | w) +\log \frac{1}{2σ_0^2} \exp(-\frac{|w|}{σ_0^2})  \\ &\propto \mathop{\arg \max}_{w} \log p(x,y | w)-\frac{|w|}{σ_0^2} \\ &= \mathop{\arg \min}_{w} \log p(x,y | w)+\frac{1}{σ_0^2}||w||_1 \end{aligned} $$

因此参数服从拉普拉斯分布的最大后验估计等价于引入**L1**正则化。

## ⚪ 弹性网络正则化 Elastic Net Regularization

- paper：[Regularization and Variable Selection via the Elastic Net](https://www.jstor.org/stable/3647580)

**弹性网络正则化 (Elastic Net Regularization)**是指同时约束参数的**L2**范数和**L1**范数：

$$ w^*= \mathop{\arg\min}_w \frac{1}{N} \sum_{n=1}^{N} {L(y_n,f(x_n);w)}+λ_2 ||w||_2^2+λ_1 ||w||_1$$

## ⚪ 谱正则化 Spectral Norm Regularization

- paper：[<font color=blue>Spectral Norm Regularization for Improving the Generalizability of Deep Learning</font>](https://0809zheng.github.io/2020/09/19/snr.html)

**谱正则化 (Spectral Norm Regularization)**是指把**谱范数(spectral norm)**的平方作为正则项，从而增强网络的泛化性：

$$ \mathcal{L}(x,y;W) + \lambda ||W||_2^2 $$

谱正则化使网络更好地满足[Lipschitz连续性](https://0809zheng.github.io/2022/10/11/lipschitz.html)。**Lipschitz**连续性保证了函数对于**输入扰动的稳定性**，即函数的输出变化相对输入变化是缓慢的。

谱范数是一种由向量范数诱导出来的矩阵范数，作用相当于向量的模长：

$$ ||W||_2 = \mathop{\max}_{x \neq 0} \frac{||Wx||}{||x||} $$

谱范数$\|\|W\|\|_2$的平方的取值为$W^TW$的最大特征值。

# 2. 约束网络结构

## ⚪ 随机深度 Stochastic Depth
- paper：[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

**随机深度**是指在训练时以一定概率丢弃网络中的模块（令其等价于恒等变换）；测试时使用完整的网络，并且按照丢弃概率对各个模块的输出进行加权。

![](https://pic.imgdb.cn/item/63a6bfa508b683016343b891.jpg)

## ⚪ Dropout
- paper：[Dropout: A simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html)

**Dropout**是指在训练深度神经网络时，随机丢弃一部分**神经元**。即对某一层设置概率$p$，对该层的每个神经元以概率$p$判断是否要丢弃。此时每个神经元的丢弃概率遵循概率$p$的伯努利(**Bernoulli**)分布。
![](https://pic.downk.cc/item/5e7de4c1504f4bcb04745d05.png)

训练时激活神经元的平均数量是原来的$p$倍；而在测试时所有神经元都被激活，故测试时需将该层神经元的输出乘以$1-p$(被保留的概率)。或者采用**Inverted Dropout**，即在训练时对某一层按概率$p$随机丢弃神经元之后将该层的输出除以$1-p$；测试时不需再做处理。

从不同角度理解**Dropout**：
1. **正则化Regularization**角度：每一次**Dropout**相当于为原网络引入噪声，测试时通过平均抵消掉噪声；每次训练不会过度依赖于个别的神经元的输出，增强网络的泛化能力；
2. **集成Ensemble**角度：每一次**Dropout**相当于从原网络中生成一个子网络，每次迭代相当于训练一个不同的子网络；最终的网络可以看作这些子网络的集成；
3. **贝叶斯Bayesian**角度：贝叶斯学习假设参数$w$为随机变量，先验分布为$q(w)$，贝叶斯方法的预测结果如下。其中不等号由**Monte Carlo**方法得到，$w_m$是第$m$次**Dropout**的网络参数，看作对全部参数$w$的一次采样。

$$ E_{q(w)}(y)=\int_{q(w)}^{} {f(x;w)q(w)dw} ≈\frac{1}{M}\sum_{m=1}^{M} {f(x;w_m)}$$
			  
与**Dropout**相关的工作包括：

- Reference：[Survey of Dropout Methods for Deep Neural Networks](https://arxiv.org/abs/1904.13310)

| **Dropout**方法 | 说明 | 示意图 |
| :---: | :---:  | :---:  |
| **Gaussian Dropout** | 每个神经元的丢弃概率遵循概率$p$的高斯分布$N(1,p(1-p))$ | ![](https://pic.imgdb.cn/item/63b042eb2bbf0e79944c33a1.jpg) |
| [**Standout**](https://proceedings.neurips.cc/paper/2013/file/7b5b23f4aadf9513306bcd59afb6e4c9-Paper.pdf) <br> (**NeurIPS2013**) | 神经元的丢弃概率$p$通过信念网络建模 | ![](https://pic.imgdb.cn/item/63b034cd2bbf0e79940d3def.jpg) |
| [**Spatial Dropout**](https://arxiv.org/abs/1411.4280) <br> (**arXiv1411**) | 对卷积特征图的通道维度应用**Dropout** | ![](https://pic.imgdb.cn/item/63b049a12bbf0e799460fd2a.jpg) |
| [<font color=blue>Weighted Channel Dropout</font>](https://0809zheng.github.io/2020/10/19/wcd.html) <br> (**AAAI2019**) | 根据激活的相对幅度来选择通道 | ![](https://pic.imgdb.cn/item/63b2a4f15d94efb26f1548af.jpg) |
| [**Max-Pooling Dropout**](https://arxiv.org/abs/1512.00242v1) <br> (**arXiv1512**) | 把**Dropout**应用到最大池化层 | ![](https://pic.imgdb.cn/item/63b0369c2bbf0e799414a262.jpg) |
| [**Max-Drop**](http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf) <br> (**ACCV2016**) | 把**Gaussian Dropout**应用到最大池化层 | ![](https://pic.imgdb.cn/item/63b045b12bbf0e7994590826.jpg) |
| [**MaxDropout**](https://arxiv.org/abs/2007.13723) <br> (**arXiv2007**) | 对输入特征进行归一化，然后把大于给定阈值$p$的特征位置设置为$0$ | ![](https://pic.imgdb.cn/item/63b022162bbf0e7994bdaa88.jpg) |



# 3. 约束优化过程

## ⚪ Early Stopping
**Early Stop**是指训练时当观察到验证集上的错误不再下降，就停止迭代。具体停止迭代的时机，可参考[Early stopping-but when?](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5)。

![](https://pic.imgdb.cn/item/63b0150c2bbf0e799482b565.jpg)

1. Early Stopping
2. Dropout
3. Data Augmentation
4. Label Smoothing








# 8. Data Augmentation
数据增强（Data Augmentation）通过对样本集的操作增加数据量（相当于对样本集加入随机噪声），提高模型鲁棒性，避免过拟合。

图像数据的增强方法主要有：
1. 旋转 Rotation
2. 翻转 Flip
3. 缩放 Zoom
4. 平移 Shift
5. 加噪声 Noise

# 9. 标签平滑 Label Smoothing
标签平滑（Label Smoothing）首先在[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)中被提出。

标签平滑是对样本的标签引入一定的噪声。

一个样本的标签一般用one-hot向量表示：

$$ y=(0,...,0,1,0,...,0)^T $$

这是一种Hard Target，若样本标签是错误的，会导致严重的过拟合。引入噪声对标签进行平滑，假设样本以ε的概率为其他类：

$$ y'=(\frac{ε}{K-1},...,\frac{ε}{K-1},1-ε,\frac{ε}{K-1},...,\frac{ε}{K-1})^T $$

上述标签是一种Soft Target，但没有考虑标签的相关性。一种更好的做法是按照类别相关性赋予其他标签不同的概率。

