---
layout: post
title: 'Circle Loss: A Unified Perspective of Pair Similarity Optimization'
date: 2022-11-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/62660a3e239250f7c568ba4b.jpg'
tags: 论文阅读
---

> Circle Loss: 成对相似性优化的统一视角.

- paper：[Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857)

深度度量学习旨在使用深度神经网络衡量样本对的距离，具体地，通过网络$f_{\theta}$把数据集$(x,y)$嵌入到特征空间，然后计算负样本对的类间相似性$s_n$和正样本对的类内相似性$s_p$，然后最小化$s_n-s_p$。这种优化方式是不够灵活的，因为其对每个单一相似性分数$s_n,s_p$的惩罚强度是相等的。本文提出**Circle loss**，通过最小化$\alpha_ns_n-\alpha_ps_p$对欠优化的相似性得分进行重新加权，使得相似性得分远离最优中心的样本对被更多的关注和惩罚。

![](https://pic.imgdb.cn/item/63cf492e588a5d166c8ebc0c.jpg)

记正负样本对集合为$$\mathcal{P},\mathcal{N}$$，则**Circle loss**允许每个相似性得分根据其优化状态去选择优化权重：

$$ \log(1+\sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}}  \exp(\gamma(\alpha_n^js_n^j-\alpha_p^is_p^i))) \\ =\log(1+ \sum_{j \in \mathcal{N}}  \exp(\gamma\alpha_n^js_n^j)\sum_{i \in \mathcal{P}} \exp(-\gamma\alpha_p^is_p^i))  $$

**Circle Loss**可以动态调整梯度，使得优化方向更加明确。在训练期间，进行反向传播时对$s_n^j,s_p^i$的梯度分别乘以$\alpha_n^j,\alpha_p^i$；记$s_n^j,s_p^i$的最优状态分别是$O_n,O_p$，则有$s_n^j>O_n,s_p^i<O_p$。当一个相似性分数远离最优点时，应该获得更大的权重因子，以便于更好优化使相似性分数趋近于最优值，因此设置权重：

$$ \alpha_n^j = \max(0, s_n^j-O_n) \\ \alpha_p^i = \max(0, O_p-s_p^i) $$

引入类间和类内的阈值$\Delta_n,\Delta_p$，则**Circle Loss**进一步写作：

$$  \log(1+ \sum_{j \in \mathcal{N}}  \exp(\gamma\alpha_n^j(s_n^j-\Delta_n))\sum_{i \in \mathcal{P}} \exp(-\gamma\alpha_p^i(s_p^i-\Delta_p)))  $$

为减小超参数，设置$O_p=1+m,O_n=-m,\Delta_p=1-m,\Delta_n=m$。

```python
import torch
import torch.nn as nn
from numpy.testing import assert_almost_equal

class CircleLoss(nn.Module):
    def __init__(self, gamma=1, m=0.25):
        super(CircleLoss, self).__init__()
        self.gamma = gamma
        self.Op = 1 + m
        self.On = -m
        self.Delta_p = 1-m
        self.Delta_n = m
        
    def forward(self, features, classes):
        
        batch_size = classes.size()[0]
        # 计算特征之间的余弦相似度
        features = 1. * features / (torch.norm(features, 2, dim=-1, keepdim=True).expand_as(features) + 1e-12)
        dists = torch.mm(features, features.transpose(0, 1))  # [batch_size, batch_size]
        
        # 构造全1上三角阵（用于mask掉重复的样本对和自身的样本对）
        s_inds = torch.triu(torch.ones(batch_size, batch_size), 1).type(torch.bool)
        # 取出所有有效样本对的相似度
        s = dists[s_inds]    
        # 匹配正负样本对
        classes_eq = (classes.repeat(batch_size, 1)  == classes.view(-1, 1).repeat(1, batch_size)).data
        pos_inds = classes_eq[s_inds]
        neg_inds = ~classes_eq[s_inds]

        # 计算自适应权重
        alpha_p = F.relu(self.Op-s)
        alpha_n = F.relu(s-self.On)

        # 计算损失函数
        neg_exp = torch.exp(self.gamma*alpha_n*(s-self.Delta_n))
        neg_exp = torch.sum(neg_exp * neg_inds)
        pos_exp = torch.exp(-self.gamma*alpha_p*(s-self.Delta_p))
        pos_exp = torch.sum(pos_exp * pos_inds)
        loss = torch.log(1+neg_exp*pos_exp)
        return loss

features = torch.randn(5, 128)
classes = torch.randint(0, 2, (5,))
loss = CircleLoss()

print(loss(features, classes))
```