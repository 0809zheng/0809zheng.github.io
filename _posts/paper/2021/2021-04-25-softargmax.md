---
layout: post
title: 'Integral Human Pose Regression'
date: 2021-04-25
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64ca01401ddac507ccfd153f.jpg'
tags: 论文阅读
---

> 积分人体姿态回归.

- paper：[Integral Human Pose Regression](https://arxiv.org/abs/1711.08229)

基于检测的二维人体姿态估计为每个关节生成一个概率热图，并将该关节定位为图中最大可能点。

$$
\boldsymbol{J}_k = \arg \max_{\boldsymbol{p}} \boldsymbol{H}_k(\boldsymbol{p})
$$

然而这种取最大值“**taking-maximum**”操作存在一些缺点：
- 该操作是不可微分的**non-differentiable**，并且阻止了端到端的训练，只能对热图进行监督；
- 网络中下采样环节会造成量化误差**quantization errors**，热图分辨率比输入分辨率低的多。提高分辨率虽然会提升精度，但是会造成计算和存储的高要求。

本文将取最大值操作修改为取期望“**taking-expectation**”操作。关节被估计为热图中所有位置的积分，由它们的概率加权。这种方法称为积分回归 (**integral regression**)。该方法是可微的，允许端到端训练；实现简单，在计算和存储方面带来的开销很小；可以很容易地结合任何基于热图的方法。

$$
\boldsymbol{J}_k = \int_{\boldsymbol{p} \in \Omega} \boldsymbol{p} \cdot \tilde{\boldsymbol{H}}_k(\boldsymbol{p}) = \int_{\boldsymbol{p} \in \Omega} \boldsymbol{p} \cdot \frac{e^{\boldsymbol{H}_k(\boldsymbol{p})}}{\int_{\boldsymbol{q} \in \Omega}e^{\boldsymbol{H}_k(\boldsymbol{q})}}
$$

上式使用**softmax**使热图中的值全部为正，总和为**1**。在实际实现时，通过求和来近似积分：

$$
\boldsymbol{J}_k = \sum_{p_x = 1}^H \sum_{p_y=1}^W \boldsymbol{p} \cdot \frac{e^{\boldsymbol{H}_k(\boldsymbol{p})}}{\sum_{q_x = 1}^H \sum_{q_y=1}^We^{\boldsymbol{H}_k(\boldsymbol{q})}}
$$

![](https://pic.imgdb.cn/item/64ca07051ddac507cc0b2b2b.jpg)

```python
def soft_argmax(heatmaps):
    B, N, H, W = heatmaps.shape

    heatmaps = heatmaps.reshape(B, N, H*W)
    heatmaps = F.softmax(heatmaps, dim=2)
    heatmaps = heatmaps.reshape(B, N, H, W)

    accu_x = heatmaps.sum(dim=2) # [B, N, W]
    accu_y = heatmaps.sum(dim=3) # [B, N, H]

    accu_x = accu_x * torch.arange(1, W+1)[None, None, :]
    accu_y = accu_y * torch.arange(1, H+1)[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y,), dim=-1)
    return coord_out
```