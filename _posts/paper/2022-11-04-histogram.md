---
layout: post
title: 'Learning Deep Embeddings with Histogram Loss'
date: 2022-11-04
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63c517d9be43e0d30eccf2e4.jpg'
tags: 论文阅读
---

> 通过直方图损失学习深度嵌入.

- paper：[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/abs/1611.00822)

直方图损失(**Histogram Loss**)是一种深度度量学习的损失函数，它首先估计正样本对和负样本对所对应的两个特征距离分布，然后计算正样本对之间的相似度比负样本对之间的相似度还要小的概率。

![](https://pic.imgdb.cn/item/63c51a55be43e0d30ed19d40.jpg)

对于正样本对集合$$\mathcal{P}$$和负样本对集合$$\mathcal{N}$$，分别计算它们对应的特征距离集合：

$$ \mathcal{S}^+ = \{ s_{ij} = <f_{\theta}(x_i),f_{\theta}(x_j)> | (i,j) \in \mathcal{P} \} \\ \mathcal{S}^- = \{ s_{ij} = <f_{\theta}(x_i),f_{\theta}(x_j)> | (i,j) \in \mathcal{N} \} $$

其中距离函数选用余弦相似度，取值范围是$[-1,1]$。根据距离集合构造$R$个**bins**的直方图$H^+$和$H^-$，则直方图的间隔为$\Delta=2/R$。

直方图$H^+$上每个位置$r$处对应的值$h^+$计算为：

$$ h^+ = \frac{1}{|\mathcal{S}^+|} \sum_{(i,j) \in \mathcal{P}} \delta_{i,j,r} $$

其中权重$\delta_{i,j,r}$将一个样本对之间的距离以线性插值的方式分配到相邻的两个节点上，如果这个距离$s_{ij}$离节点$r$越近，则这个权重越大：

$$ \delta_{i,j,r} = \begin{cases} (s_{ij}-t_{r-1})/\Delta, & s_{ij} \in [t_{r-1};t_r] \\ (t_{r+1}-s_{ij})/\Delta , & s_{ij} \in [t_r;t_{r+1}] \\ 0, & \text{otherwise} \end{cases} $$

直方图$H^+$和$H^-$近似作为正样本对和负样本的特征距离分布，下面计算一个随机负样本对的距离比一个随机正样本对的距离小的概率：

$$ p = \int_{-1}^{1}p^-(x) [\int_{-1}^{x}p^+(y)dy] dx = \int_{-1}^{1}p^-(x) \Phi^+(x) dx = \Bbb{E}_{x \text{~}p^-(x)}[\Phi^+(x)] $$

上式可以通过直方图表示为离散形式：

$$ p ≈ \sum_{r=1}^R (h^-_r \sum_{q=1}^rh_q^+) = \sum_{r=1}^R h^-_r \phi_r^+ $$

```python
import torch
from numpy.testing import assert_almost_equal

class HistogramLoss(torch.nn.Module):
    def __init__(self, num_bins, cuda=False):
        super(HistogramLoss, self).__init__()
        self.step = 2 / num_bins
        self.eps = 1 / num_bins
        self.cuda = cuda
        self.t = torch.arange(-1+self.eps, 1, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t.cuda()
        
    def forward(self, features, classes):
        
        # 构造直方图分布
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            # 把样本对距离s分配给直方图的bins [t_{r-1}, t_r]
            indsa = (s_repeat_ - self.t >= 0) & (s_repeat_ - self.t < self.step) & inds
            assert indsa.nonzero().size()[0] == size, ('Another number of bins should be used')
            
            # 把样本对距离s分配给直方图的bins [t_r, t_{r+1}]
            zeros = torch.zeros((1, indsa.size()[1])).byte()
            if self.cuda:
                zeros = zeros.cuda()
            indsb = torch.cat((zeros, indsa))[:-1, :].to(dtype=torch.bool)
            
            # 根据权重delta计算直方图的取值
            s_repeat_[~(indsb|indsa)] = 0
            # indsa corresponds to the first condition of the second equation of the paper
            s_repeat_[indsa] = (s_repeat_ - self.t)[indsa] / self.step
            # indsb corresponds to the second condition of the second equation of the paper
            s_repeat_[indsb] =  (-s_repeat_ + self.t)[indsb] / self.step
            
            return s_repeat_.sum(1) / size
        
        batch_size = classes.size()[0]
        # 计算特征之间的余弦相似度
        features = 1. * features / (torch.norm(features, 2, dim=-1, keepdim=True).expand_as(features) + 1e-12)
        dists = torch.mm(features, features.transpose(0, 1))  # [batch_size, batch_size]
        
        # 构造全1上三角阵（用于mask掉重复的样本对和自身的样本对）
        s_inds = torch.triu(torch.ones(batch_size, batch_size), 1).byte() 
        if self.cuda:
            s_inds= s_inds.cuda()
            
        # 取出所有有效样本对的距离
        s = dists[s_inds].view(1, -1) # num_pairs = batch_size * (batch_size-1) / 2
        s_repeat = s.repeat(self.tsize, 1) # [num_bins, num_pairs]
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()
        # print(s_repeat_floor)
        
        # 匹配正负样本对
        classes_eq = (classes.repeat(batch_size, 1)  == classes.view(-1, 1).repeat(1, batch_size)).data
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()

        # 计算样本对的直方图
        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1, 
                            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1, 
                            err_msg='Not good negative histogram', verbose=True)
        
        # 计算正样本对直方图的累计密度函数
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0]) # [num_bins, num_bins]
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        
        # 计算直方图损失
        loss = torch.sum(histogram_neg * histogram_pos_cdf)
        
        return loss
```