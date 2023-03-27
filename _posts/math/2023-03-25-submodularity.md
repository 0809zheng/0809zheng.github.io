---
layout: post
title: '集函数的子模性(Submodularity)与Lovász延拓(Lovász Extension)'
date: 2023-03-25
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6421507da682492fcc7ed807.jpg'
tags: 数学
---

> Submodular Functions and Lovász Extension.

**集函数(set function)**是以集合为定义域的函数。

# 1. 子模性 Submodularity

**子模性**是集函数的一个性质，许多组合优化与机器学习问题都具有子模性结构。子模性有两种等价的定义：

记具有$n$个元素的集合为$$[n]=\{1,2,...,n\}$$，集合$[n]$的所有子集对应的集合为$2^{[n]}$。如果一个集函数$f:2^{[n]} \to R$是子模的，则对于集合$[n]$中的所有真子集对 $T ⊂ S; S,T ⊂ [n]$ 和集合中的所有元素 $i \in [n]$，存在**收益递减(diminishing returns)**性质：

$$ f(T ∪ \{i\}) - f(T) \geq f(S ∪ \{i\}) - f(S) $$

等价地，如果一个集函数$f:2^{[n]} \to R$是子模的，则对于所有集合$S,T ⊂ [n]$满足：

$$ f(S) + f(T) \geq f(S∪T) + f(S∩T) $$

一些常见的子模函数包括：
- 线性单调函数：寻找最小/最大生成树
- 覆盖函数：查找图/超图覆盖
- 割函数：查找图的最小/最大割
- 熵函数

一般地，集合$S ⊂ [n]$可以通过它的特征向量$$X_S \in H_n = \{0,1\}^n$$表示，其中$X_S(i)=1$表示$i \in S$，否则$X_S(i)=0$。则集函数也可以定义在特征向量上 $f:H_n \to R$。

子模性是**凸性(convexity)**的离散表现形式。给定一个子模函数，至少有三种光滑**延拓(extension)**可以将其构造为$K_n = [0,1]^n$上的连续函数:
- 凸闭包 (**Convex Closure**)
- 凹闭包 (**Concave Closure**)
- 多线性延拓 (**Multilinear Extension**)

对于子模函数，凸闭包等价于**Lovász**延拓。

# 2. Lovász 延拓

**Lovász**延拓是一种非常有用的子模函数的光滑延拓方式（可类比向量函数的光滑化），可以用于子模最小化。一个子模函数的**Lovász**延拓总是凸函数，可以被高效地最小化。因此**Lovász**延拓的最小值可以被用于寻找对应子模函数的最小值。

给定一个子模函数$f:H_n \to R$，对应的**Lovász**延拓$\hat{f}:K_n \to R$定义为：

$$
\hat{f}(x) = \sum_{i=0}^n \lambda_i f(X_{S_i})
$$

其中$∅=S_0⊂S_1⊂S_2...⊂S_n=[n]$，且$\sum_{i=0}^n \lambda_i X_{S_i}=x$，$\sum_{i=0}^n \lambda_i = 1$。

在超立方体$H_n$上，存在 $n!$ 条不同的从全零向量$0_n$到全一向量$1_n$的最短路径。记$P=[0_n=X_0,X_1,...,1_n=X_n]$为一条路径，$C_P$为路径$P$上的点组成的凸包(**convex hull**)。每条路径上共存在$n!$个凸包，这些凸包把$K_n$划分成$n!$个相等的部分。

给定一个点$x \in K_n$，可以对应一个凸包$C_P$和路径$$P=[0_n=X_0,X_1,...,1_n=X_n]$$使得$x \in C_P$。我们可以找到系数$\lambda_i$使得$$x=\sum_{i=0}^n \lambda_i X_{i}$$，并且满足$$\sum_{i=0}^n \lambda_i = 1,\lambda_i \geq 0$$。点$x$处的**Lovász**延拓定义为：

$$
\hat{f}(x) = \sum_{i=0}^n \lambda_i f(X_{i})
$$

记$x=(x_1,x_2,...,x_n)$，$\pi:[n] \to [n]$是$x_1,x_2,...,x_n$的排序排列，即$\pi(i)=j$表示$x_j$是向量$x$中第$i$大的元素：$1 \geq x_{\pi(1)}\geq x_{\pi(2)}\geq \cdots \geq x_{\pi(n)}\geq 0$。若额外指定$x_{\pi(0)}=1,x_{\pi(n+1)}=0$，则有：

$$
\lambda_i = x_{\pi(i)}-x_{\pi(i+1)}
$$

按照上述定义的$\lambda_i$满足$\sum_{i=0}^n \lambda_i = 1,\lambda_i \geq 0$。下面证明$x=\sum_{i=0}^n \lambda_i X_{i}$。令$$X_0=0_n$$，$$e_{\pi(i)}$$表示$\pi(i)$位置为$1$其余位置为$0$的向量，则有：

$$
X_i = X_{i-1} + e_{\pi(i)}
$$

此时求和项$\sum_{i=0}^n \lambda_i X_{i}$等价于：

$$
\begin{aligned}
\sum_{i=0}^n \lambda_i X_i & =\left(1-x_{\pi(1)}\right) X_0+\sum_{i=1}^n\left(x_{\pi(i)}-x_{\pi(i+1)}\right)\left(X_{i-1}+e_{\pi(i)}\right) \\
& =\sum_{i=1}^n\left(x_{\pi(i)}-x_{\pi(i+1)}\right)\left(X_{0}+e_{\pi(1)}+e_{\pi(2)}+\cdots +e_{\pi(i)}\right) \\
& =\sum_{i=1}^n e_{\pi(i)}\left(\sum_{j=i}^n x_{\pi(j)}-x_{\pi(j+1)}\right) \\
& =\sum_{i=1}^n e_{\pi(i)} x_{\pi(i)} \\
& =x
\end{aligned}
$$

由于点$x$处的**Lovász**延拓定义为$X_0,X_1,...,X_n$的凸组合，是一个分段线性函数，是连续但不一定可微的。因此我们可以定义**Lovász**延拓的次梯度(**sub-gradient**)：

$$
g(x) = \sum_{i=1}^n \left( f(X_{i}) - f(X_{i-1}) \right)e_{\pi(i)}
$$

**Lovász**延拓$\hat{f}:K_n \to R$等价地定义为：

$$
\begin{aligned}
\hat{f}(x) &= \sum_{i=0}^n \lambda_i f(X_i) \\ 
& = \sum_{i=0}^n (x_{\pi(i)}-x_{\pi(i+1)}) f(X_i)\\
& = x_{\pi(0)}f(X_0)+ \sum_{i=1}^n x_{\pi(i)} \left( f(X_{i}) - f(X_{i-1}) \right) - x_{\pi(n+1)}f(X_n) \\
& = \sum_{i=1}^n x_{\pi(i)} \left( f(X_{i}) - f(X_{i-1}) \right) =  \sum_{i=1}^n x_{\pi(i)}g_i(x)
\end{aligned}
$$

# 3. Lovász 延拓的应用：构造[Lovász Loss](https://arxiv.org/abs/1705.08790)

**Lovász Loss**是为图像分割任务设计的损失函数，其思想是采用**Lovász**延拓把图像分割中离散的[**IoU Loss**](https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22)变得光滑化。

**IoU Loss**直接优化**IoU index**，后者衡量预测类别为$i$的像素集合$A$和真实类别为$i$的像素集合$B$的交集与并集之比。

$$
L_{I o U}=1- \frac{|A ∩ B |}{|A∪ B |}
$$

论文[IoU is not submodular](https://arxiv.org/abs/1809.00593)指出**IoU index**并不是子模函数；而论文[Yes, IoU loss is submodular - as a function of the mispredictions](https://arxiv.org/abs/1809.01845)指出**IoU Loss**是子模函数，因此可以应用**Lovász**延拓。

首先定义类别$c$的误分类像素集合$M_c$：

$$
\mathbf{M}_c\left(\boldsymbol{y}^*, \tilde{\boldsymbol{y}}\right)=\left\{\boldsymbol{y}^*=c, \tilde{\boldsymbol{y}} \neq c\right\} \cup\left\{\boldsymbol{y}^* \neq c, \tilde{\boldsymbol{y}}=c\right\}
$$

则**IoU Loss**可以写成集合$M_c$的函数：

$$
\Delta_{J_c}: \mathbf{M}_c \in\{0,1\}^N \mapsto 1-\frac{\left|\mathbf{M}_c\right|}{\left|\left\{\boldsymbol{y}^*=c\right\} \cup \mathbf{M}_c\right|}
$$

下面求上述函数的**Lovász**延拓。定义类别$c$的像素误差向量$m(c) \in [0,1]^N$：

$$
m_i(c) = \begin{cases} 1-s_i^c, & \text{if }c=\boldsymbol{y}^*_i \\ s_i^c, & \text{otherwise} \end{cases}
$$

则$$\Delta_{J_c}(\mathbf{M}_c)$$的**Lovász**延拓$$\overline{\Delta_{J_c}}(m(c))$$根据定义可表示为：

$$
\overline{\Delta_{J_c}}: m \in R^N \mapsto \sum_{i=1}^N m_{\pi(i)} g_i(m)
$$

其中$$g_i(m)=\Delta_{J_c}(\{\pi_1,...,\pi_i\})-\Delta_{J_c}(\{\pi_1,...,\pi_{i-1}\})$$，$\pi$是$m$中元素的一个按递减顺序排列：$m_{\pi_1} \geq m_{\pi_2} \geq \cdots \geq m_{\pi_N}$。

```python
import torch
import torch.nn as nn
from einops import rearrange

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    n = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if n > 1:  # cover 1-pixel case
        jaccard[1:n] = jaccard[1:n] - jaccard[0:-1]
    return jaccard

class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            input_c = inputs[:, c]
            loss_c = (target_c - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, lovasz_grad(target_c_sorted)))
        losses = torch.stack(losses)
        return losses.mean()

    def forward(self, inputs, targets):
        # inputs.shape = (batch size, class_num, h, w)
        # targets.shape = (batch size, h, w)
        inputs = rearrange(inputs, 'b c h w -> (b h w) c')
        targets = targets.view(-1)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
    
seg_loss = LovaszLoss()
result = torch.randn((16, 8, 64, 64))
gt = torch.randint(0, 8, (16, 64, 64))
loss = seg_loss(result, gt)
print(loss)
```