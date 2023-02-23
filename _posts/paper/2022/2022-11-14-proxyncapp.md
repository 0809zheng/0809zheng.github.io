---
layout: post
title: 'ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis'
date: 2022-11-14
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63cba9c4be43e0d30eb5511a.jpg'
tags: 论文阅读
---

> ProxyNCA++: 回顾和改进深度度量学习中的代理邻域成分分析.

- paper：[ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis](https://arxiv.org/abs/2003.13911)

[<font color=blue>Proxy NCA Loss</font>](https://0809zheng.github.io/2022/11/11/proxynca.html)是一种深度度量学习的损失函数，它为每个类别随机初始化一个代理向量$p$，遍历样本时以邻域成分分析(**NCA**)的形式拉近每个样本$x$和该样本类别$y$对应的代理向量$p_y$之间的距离，增大和其他类别的代理向量$p_z$之间的距离。

$$ -\log (\frac{\exp(-D[f_{\theta}(x),p_y])}{\sum_{z \neq y} \exp(-D[f_{\theta}(x),p_z])}) $$

**ProxyNCA++**在此基础上引入一些改进：

![](https://pic.imgdb.cn/item/63cbab3bbe43e0d30eb8a8f4.jpg)

### ⚪ 优化代理分配概率 optimizing proxy assignment probability

**Neighborhood Component Analysis (NCA)**通过指数加权使得样本$x$更接近$y$而不是集合$Z$中的任意元素：

$$ -\log (\frac{\exp(-d(x,y))}{\sum_{z \in Z}\exp(-d(x,z))}) $$

**Proxy-NCA**定义的损失函数为：

$$ -\log (\frac{\exp(-d(x,p_y))}{\sum_{z \neq y} \exp(-d(x,p_z))}) $$

为了与**NCA**对齐，**ProxyNCA++**把损失函数调整为：

$$ -\log (\frac{\exp(-D[f_{\theta}(x),p_y])}{\sum_{z} \exp(-D[f_{\theta}(x),p_z])}) = -\log \text{ softmax}(-D[f_{\theta}(x),p])_y $$

从而可以通过**softmax**函数实现上述损失。

### ⚪ 低温缩放 Low Temperature Scaling

通过在**softmax**函数中引入温度系数$T$可以调整分布的平滑程度，$T$越大则分布越平滑。而度量学习通常希望同类样本的距离尽可能小于与其他类别样本的距离，因此选择较小的温度$T<1$，以获得具有判别性的类别边界：

$$  -\log \text{ softmax}(-D[f_{\theta}(x),p] \cdot \frac{1}{T})_y = -\log (\frac{\exp(-D[f_{\theta}(x),p_y]\cdot \frac{1}{T})}{\sum_{z} \exp(-D[f_{\theta}(x),p_z]\cdot \frac{1}{T})}) $$

![](https://pic.imgdb.cn/item/63cbaf2fbe43e0d30ebeb5c4.jpg)

温度系数$T$的消融结果：

![](https://pic.imgdb.cn/item/63cbafefbe43e0d30ebf936c.jpg)

### ⚪ 全局池化 Global Pooling

把网络中的全局平均池化替换成全局$k$阶最大池化，即选择特征图的$k$个最大元素并计算平均值；结果表明全局最大池化($k=1$)效果最好：

![](https://pic.imgdb.cn/item/63cbb1a8be43e0d30ec2024a.jpg)

### ⚪ 快速移动代理 Fast moving proxies

代理向量$p$被设置为可学习参数，其学习率可以设置得大一些：

![](https://pic.imgdb.cn/item/63cbb34fbe43e0d30ec4283a.jpg)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProxyNCApp(nn.Module):
    def __init__(self, 
        nb_classes,
        sz_embedding,
        smoothing_const = 0.1,
        scaling_x = 1,
        scaling_p = 3,
        temperature = 1/9
    ):
        torch.nn.Module.__init__(self)
        self.proxies = nn.Parameter(torch.randn(nb_classes, sz_embedding))
        self.nb_classes = nb_classes
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p
        self.temp = temperature
 
    def forward(self, X, Y):
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x
        D = torch.cdist(X, P, p = 2) ** 2 # [batch_size, num_proxy]

        # 生成one-hot标签
        labels = torch.FloatTensor(Y.shape[0], self.nb_classes).zero_()
        labels = labels.scatter_(1, Y.data, 1)

        # 应用label smoothing
        labels = labels * (1 - self.smoothing_const)
        labels[labels == 0] = self.smoothing_const / (self.nb_classes - 1)

        loss = torch.sum(-labels * F.log_softmax(-D / self.temp, -1), -1)
        return loss.mean()
    
if __name__ == '__main__':
    nb_classes = 100
    sz_batch = 32
    sz_embedding = 64
    X = torch.randn(sz_batch, sz_embedding)
    Y = torch.randint(low=0, high=nb_classes, size=[sz_batch])
    pnca = ProxyNCApp(nb_classes, sz_embedding)
    print(pnca(X, Y.view(sz_batch, 1)))
```