---
layout: post
title: 'Gradient Centralization: A New Optimization Technique for Deep Neural Networks'
date: 2021-06-27
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60dd62325132923bf859d24e.jpg'
tags: 论文阅读
---

> 梯度中心化：一种新的深度神经网络优化技术.

- paper：Gradient Centralization: A New Optimization Technique for Deep Neural Networks
- arXiv：[link](https://arxiv.org/abs/2004.01461v1)
- code：[github](https://github.com/Yonghongwei/Gradient-Centralization)

# 1. 梯度中心化的定义

![](https://pic.imgdb.cn/item/60dd63e15132923bf868180c.jpg)

在深度学习方法中，通常在前向传播过程中根据网络参数$W$计算损失函数$\mathcal{L}$，并在反向传播中计算损失的梯度$\nabla_W \mathcal{L}$，从而应用梯度下降算法更新参数。本文提出了一种**梯度中心化(gradient centralization,GC)**方法，对梯度先应用中心化后再更新参数。梯度中心化可以表示为：

$$ \Phi_{GC}(\nabla_{w_i} \mathcal{L}) = \nabla_{w_i} \mathcal{L}-\mu_{\nabla_{w_i} \mathcal{L}} $$

其中梯度均值向量计算为$\mu_{\nabla_{w_i} \mathcal{L}}=\frac{1}{M}\sum_{j=1}^{M}\nabla_{w_{i,j}} \mathcal{L}$.对于全连接网络，参数表示为$W \in \Bbb{R}^{C_{in} \times C_{out}}$；对于卷积神经网络，参数表示为$W \in \Bbb{R}^{C_{in} \times C_{out}\times (k_1k_2)} $；沿$C_{out}$维度计算平均梯度。若记$e$为$M$维单位向量，则梯度中心化也可表示为：

$$ \Phi_{GC}(\nabla_{W} \mathcal{L}) = P\nabla_{W} \mathcal{L}, \quad P=I-ee^T $$

值得一提的是，矩阵$P$在形式上就是**PCA**算法中的**中心矩阵(centering matrix)**，是一个对称的幂等矩阵，其左乘样本矩阵后可以实现样本矩阵的归一化。

# 2. 梯度中心化嵌入到优化方法中
梯度中心化可以很容易地嵌入到不同优化方法中，只需要在其计算梯度后加上中心化这一步即可，如应用到**SGD**和**Adam**算法中：

![](https://pic.imgdb.cn/item/60dd68f75132923bf895acf7.jpg)

# 3. 梯度中心化的性质

## (1)提高泛化能力  Improving Generalization Performance

### 权重空间正则化 Weight space regularization
矩阵$P$是一个对称的幂等矩阵$P^2=P=P^T$，因此可以看作一个投影矩阵。容易证明：

$$ e^TP = e^T(I-ee^T) = e^T-e^Tee^T = e^T-e^T = 0 $$

因此矩阵$P$负责将向量投影到法向量为$e$的超平面上。梯度中心化$\Phi_{GC}(\nabla_{W} \mathcal{L})$相当于把梯度$\nabla_{W} \mathcal{L}$投影到该超平面上，这使得梯度更新被限制在同一个超平面上进行：

![](https://pic.imgdb.cn/item/60dd79395132923bf8152d60.jpg)

这使得每次梯度更新的参数也在同一个超平面上，即$e^T(w-w^t)=0$。在优化损失函数时，相当于引入了潜在的约束：

$$ \mathop{\min}_{w}\mathcal{L}(w), \qquad s.t. \quad e^T(w-w^0)=0 $$

这是一个关于参数向量$w$的约束优化问题，它正则化了$w$的解空间，从而减少了过度拟合训练数据的可能性，进一步提高了模型的泛化能力。

### 输出特征空间正则化 Output feature space regularization
将梯度中心化应用到**SGD**算法，可以得到对应的更新公式：

$$ w^{t+1} = w^{t} - \alpha^{t}\Phi_{GC}(\nabla_{W^{t}} \mathcal{L}) = w^{t} - \alpha^{t} P\nabla_{W^{t}} \mathcal{L} $$

上式可以展开为：

$$ w^{t} = w^{0} - \sum_{i=0}^{t-1} P\alpha^{i} \nabla_{W^{i}} \mathcal{L} $$

如果对输入$x$增加一个恒定强度变化的较小的扰动$\gamma 1$，则输出激活的变化为：

$$ (w^{t})^T(x+\gamma 1)-(w^{t})^T x = \gamma (w^{t})^T 1 = \gamma 1^T w^{t} \\ = \gamma 1^T (w^{0} - \sum_{i=0}^{t-1} P\alpha^{i} \nabla_{W^{i}} \mathcal{L}) = \gamma 1^T w^{0} - \gamma 1^T\sum_{i=0}^{t-1} P\alpha^{i} \nabla_{W^{i}} \mathcal{L} \\ = \gamma 1^T w^{0} - \gamma 1^TP\sum_{i=0}^{t-1}\alpha^{i} \nabla_{W^{i}} \mathcal{L}  $$

注意到：

$$ 1^TP = 1^T(I-ee^T) = 1^T(I-\frac{1}{M}11^T) = 1^T-\frac{1}{M}1^T11^T = 1^T-1^T = 0 $$

因此：

$$ (w^{t})^T(x+\gamma 1)-(w^{t})^T x = \gamma 1^T w^{0} $$

即对输入进行微小的扰动，其输出变化只与参数的初始值$w^{0}$有关，而与更新后的参数值$w^{t}$无关。若参数的初始值接近$0$，则输出激活对输入特征的强度变化不敏感，输出特征空间对于输入的变化具有更强的稳定性。

实际中常选用的初始化策略(如**Xavier**初始化,**Kaiming**初始化)会使权重参数$w^{0}$接近$0$，下面展示两种初始化方法得到的参数的绝对值均值的对数分布，可以看出大多数参数向量的均值都非常小，如果使用梯度中心化的**SGD**算法更新模型，其输出特征相对于输入特征的强度变化不敏感。

![](https://pic.imgdb.cn/item/60dd77f95132923bf80c4e01.jpg)

## (2)加速训练过程 Accelerating Training Process

### 平滑优化曲面 Optimization landscape smoothing
- 定理1：梯度中心化后的**L2**范数比原梯度的**L2**范数小。

证明如下：

$$ ||\Phi_{GC}(\nabla_{W} \mathcal{L})||_2^2 = \Phi_{GC}(\nabla_{W} \mathcal{L})^T\Phi_{GC}(\nabla_{W} \mathcal{L}) = (P\nabla_{W} \mathcal{L})^TP\nabla_{W} \mathcal{L} \\ = \nabla_{W} \mathcal{L}^TP^TP\nabla_{W} \mathcal{L} = \nabla_{W} \mathcal{L}^T(I-ee^T)^T(I-ee^T)\nabla_{W} \mathcal{L} \\ = \nabla_{W} \mathcal{L}^T(I-2ee^T+ee^Tee^T)\nabla_{W} \mathcal{L} = \nabla_{W} \mathcal{L}^T(I-ee^T)\nabla_{W} \mathcal{L} \\ = \nabla_{W} \mathcal{L}^T\nabla_{W} \mathcal{L} - \nabla_{W} \mathcal{L}^Tee^T\nabla_{W} \mathcal{L} = ||\nabla_{W} \mathcal{L}||_2^2 - ||e^T\nabla_{W} \mathcal{L}||_2^2 \\ ≤ ||\nabla_{W} \mathcal{L}||_2^2 $$

- 定理2：梯度中心化后的**Hessian**矩阵比原梯度的**Hessian**矩阵小。

证明如下：

$$ ||\nabla \Phi_{GC}(\nabla_{W} \mathcal{L})||_2^2 = ||\nabla P\nabla_{W} \mathcal{L}||_2^2 = || P\nabla_{W}^2 \mathcal{L}||_2^2 \\ = \nabla_{W}^2 \mathcal{L}^TP^TP\nabla_{W}^2 \mathcal{L} = \nabla_{W}^2 \mathcal{L}^T(I-ee^T)\nabla_{W}^2 \mathcal{L} \\ = \nabla_{W}^2 \mathcal{L}^T\nabla_{W}^2 \mathcal{L} - \nabla_{W}^2 \mathcal{L}^Tee^T\nabla_{W}^2 \mathcal{L}  = ||\nabla_{W}^2 \mathcal{L}||_2^2 - ||e^T\nabla_{W}^2 \mathcal{L}||_2^2 \\ ≤ ||\nabla_{W}^2 \mathcal{L}||_2^2 $$

上述两个定理表明被梯度中心化约束的损失函数与原始的损失函数相比，能够达到更好的**Lipschitz**特性，从而使得损失函数曲面更加平滑，实现更为快速和高效的训练。

### 抑制梯度爆炸 Gradient explosion suppression
梯度中心化可以抑制梯度爆炸，使得训练过程更加稳定。这个性质和梯度裁剪类似，防止梯度出现过大的值。使用梯度中心化后梯度的**L2**范数和最大值都会变小：

![](https://pic.imgdb.cn/item/60dd6a625132923bf8a1d458.jpg)
