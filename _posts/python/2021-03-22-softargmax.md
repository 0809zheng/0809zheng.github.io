---
layout: post
title: 'Argmax与SoftArgmax'
date: 2021-03-22
author: 郑之杰
cover: ''
tags: Python
---

> Implement SoftArgmax with Pytorch.

在编程时，有时候需要返回一个张量最大值所在的维度序号（如分类任务中返回概率最大的类别编号、定位任务中返回概率最大的空间坐标编号），此时需要用到**argmax**操作。

**Pytorch**中的**argmax**函数定义为`torch.argmax(input, dim=None, keepdim=False)`，其中的`dim`参数指定寻找最大值的维度，`keepdim`参数指定是否保持原张量的维度。

如一个尺寸为$(3,4,5)$的三维张量，若设置`dim=1,keepdim=False`则输出张量的尺寸是$(3,5)$；若设置`dim=1,keepdim=True`则输出张量的尺寸是$(3,1,5)$。

由于**argmax**函数是不可导的，在构建网络时无法反向传播梯度。在实际构建网络时通常使用**SoftArgmax**函数作为替代。对于张量中的每一个位置$i$，做如下近似：

$$ y=argmax(x) ≈ \sum_{i}^{} i \cdot softmax(x)_i $$

如对一个尺寸为$(batchNumber \times Channel \times Height \times Width \times Depth)$的三维空间张量寻找其最大值对应的坐标，返回尺寸为$(batchNumber \times Channel \times 3)$的张量，其**Pytorch**实现如下：

```
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax(nn.Module):
    def __init__(self, height, width, depth, channel):
        super(SoftArgmax, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth
        self.channel = channel
        
        pos_x, pos_y, pos_z = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width),
                np.linspace(-1., 1., self.depth)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width*self.depth)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width*self.depth)).float()
        pos_z = torch.from_numpy(pos_z.reshape(self.height*self.width*self.depth)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)
        
        
    def forward(self, input):
        # input:  (N, C, H, W, D)
        # output: (N, C, 3)
        input = input.view(-1, self.height*self.width*self.depth)
        softmax_attention = F.softmax(input, dim=1)
        
        self.pos_x = self.pos_x.to(input.device)
        self.pos_y = self.pos_y.to(input.device)
        self.pos_z = self.pos_z.to(input.device)
        softmax_attention = softmax_attention.to(input.device)
        
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_z = torch.sum(self.pos_z*softmax_attention, dim=1, keepdim=True)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        coordinates = expected_xyz.view(-1, self.channel, 3)
        return coordinates
```