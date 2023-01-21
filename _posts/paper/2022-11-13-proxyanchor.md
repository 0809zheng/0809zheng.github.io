---
layout: post
title: 'Proxy Anchor Loss for Deep Metric Learning'
date: 2022-11-13
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63cb981bbe43e0d30e9a868c.jpg'
tags: 论文阅读
---

> 深度度量学习的代理锚点损失.

- paper：[Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/abs/2003.13911)

本文提出了**Proxy-Anchor**损失，为每一个类别赋予了一个**proxy**，将一个批次的数据和所有的**proxy**之间计算距离，并拉近每个类别的数据和该类别对应的**proxy**之间的距离，且拉远与其他类别的**proxy**之间的距离。

![](https://pic.imgdb.cn/item/63cba910be43e0d30eb40a03.jpg)

**Proxy-Anchor**损失和[<font color=blue>Proxy-NCA</font>](https://0809zheng.github.io/2022/11/11/proxynca.html)损失的主要区别在于，**Proxy-NCA**遍历每一个样本，减少该样本和对应类别的**proxy**之间的距离，增大和其他类别的**proxy**之间的距离；而**Proxy-Anchor**损失遍历每一个**proxy**，减少该类别的所有样本与该**proxy**的距离，增大其他类别的样本与该**proxy**的距离。

$$ \frac{1}{|P^+|} \sum_{p \in P^+} \log (1+\sum_{x \in X_p^+}e^{\alpha(D[f_{\theta}(x),p]+\delta)}) \\+ \frac{1}{|P|} \sum_{p \in P} \log (1+\sum_{x \in X_p^-}e^{-\alpha(D[f_{\theta}(x),p]-\delta)})  $$

其中$P$是所有**proxy**集合，$P^+$是数据集中出现的有效**proxy**集合。**Proxy-NCA**没有利用数据-数据之间的相互关系，关联每个数据点的只有**proxy**。**Proxy-Anchor**通过同时考虑所有数据点改善了这一点。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, delta = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.delta = delta
        self.alpha = alpha
        
    def forward(self, X, Y):
        P = self.proxies
        
        # 计算余弦相似度
        def norm(x, axis=-1):
            x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
            return x
        cos = 1-torch.mm(norm(X),norm(P).permute(1,0))

        # 生成one-hot标签
        labels = torch.FloatTensor(Y.shape[0], self.nb_classes).zero_()
        P_one_hot = labels.scatter_(1, Y.data, 1)
        N_one_hot = 1 - P_one_hot
 
        # 统计有效proxy数量
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
    
        # 计算损失函数
        pos_exp = torch.exp(self.alpha * (cos + self.delta))
        neg_exp = torch.exp(-self.alpha * (cos - self.delta))

        P_sim_sum = torch.mul(P_one_hot, pos_exp).sum(dim=0) 
        N_sim_sum = torch.mul(N_one_hot, neg_exp).sum(dim=0) 
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss
    
if __name__ == '__main__':
    nb_classes = 100
    sz_batch = 32
    sz_embedding = 64
    X = torch.randn(sz_batch, sz_embedding)
    Y = torch.randint(low=0, high=nb_classes, size=[sz_batch])
    pnca = Proxy_Anchor(nb_classes, sz_embedding)
    print(pnca(X, Y.view(sz_batch, 1)))
```
