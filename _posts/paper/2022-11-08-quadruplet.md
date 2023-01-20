---
layout: post
title: 'Beyond triplet loss: a deep quadruplet network for person re-identification'
date: 2022-11-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63c9170bbe43e0d30ec0c85f.jpg'
tags: 论文阅读
---

> 用于行人重识别的四元组损失.

- paper：[Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/abs/1704.01719)

**Quadruplet Loss**是受[<font color=blue>Triplet Loss</font>](https://0809zheng.github.io/2022/11/02/triplet.html)启发的深度度量学习方法。三元组损失为每一个样本$x$选择一个正样本$x^+$和一个负样本$x^-$，同时最小化正样本对之间的距离和最大化负样本对之间的距离：

$$ \max(0, D[f_{\theta}(x),f_{\theta}(x^+)] -D[f_{\theta}(x),f_{\theta}(x^-)] + \epsilon) $$

而**Quadruplet Loss**为每一个样本$x$选择一个正样本$x^+$和两个负样本$x^-_1,x^-_2$，使得正样本对之间的距离同时小于负样本对之间的距离和两个负样本之间的距离：

$$ \max(0, D[f_{\theta}(x),f_{\theta}(x^+)] -D[f_{\theta}(x),f_{\theta}(x^-_1)] + \alpha) \\ + \max(0, D[f_{\theta}(x),f_{\theta}(x^+)] -D[f_{\theta}(x^-_2),f_{\theta}(x^-_1)] + \beta) $$

![](https://pic.downk.cc/item/5ec23be0c2a9a83be54a3bb6.jpg)

![](https://pic.imgdb.cn/item/63c92d61be43e0d30eeaf90b.jpg)

使用**PyTorch**自定义四元组损失：

```python
class QuadrupletMarginLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, p=2):
        super(QuadrupletMarginLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.p = p
        
    def forward(self, anchor, positive, negative1, negative2):
        pos_dist = F.pairwise_distance(anchor, positive, self.p)
        neg_dist1 = F.pairwise_distance(anchor, negative1, self.p)
        neg_dist2 = F.pairwise_distance(negative2, negative1, self.p)
        loss = F.relu(pos_dist - neg_dist1 + self.alpha)
        loss += F.relu(pos_dist - neg_dist2 + self.beta)
        return loss.mean()
```