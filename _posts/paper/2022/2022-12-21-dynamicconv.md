---
layout: post
title: 'Dynamic Convolution: Attention over Convolution Kernels'
date: 2022-12-21
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63b0f93a2bbf0e799406a2d3.jpg'
tags: 论文阅读
---

> DynamicConv：卷积核上的注意力.

- paper：[Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/abs/1912.03458)

**动态卷积(Dynamic Convolution)**根据输入动态集成多个并行的卷积核为一个动态核，这些核通过注意力机制以非线性形式进行融合，该动态核具有数据依赖性。多核集成不仅计算高效，而且具有更强的特征表达能力。

标准的卷积操作记为$y=W^Tx+b$，则动态卷积定义为：

$$ \begin{aligned} y&=\tilde{W}^Tx+\tilde{b} \\ \tilde{W} &= \sum_{k=1}^K \pi_k(x)\tilde{W}_k \\ \tilde{b} &= \sum_{k=1}^K \pi_k(x)\tilde{b}_k \\ \text{s.t. }& 0\leq \pi_k(x) \leq 1,\sum_{k=1}^K \pi_k(x)=1 \end{aligned} $$

![](https://pic.imgdb.cn/item/63b0fab22bbf0e799413c5a0.jpg)

动态卷积包含两个额外的计算：注意力权值计算和动态权值融合。这两点额外计算相比卷积操作的计算量可以忽略：

$$ O(\tilde{W}^Tx+\tilde{b}) >> O(\sum \pi_k(x)\tilde{W}_k) + O(\sum \pi_k(x)\tilde{b}_k) + O(\pi(x)) $$

动态卷积的集成过程非常高效，所引入的参数量比较有限：

![](https://pic.imgdb.cn/item/63b0fb872bbf0e79941b26dc.jpg)

动态权值$O(\pi(x))$通过轻量型的注意力机制实现。

![](https://pic.imgdb.cn/item/63b0fc212bbf0e799420468d.jpg)

动态卷积在进行训练时由于需要同时优化卷积核与注意力部分，因此训练过程具有挑战性。直接把网络中的卷积全部替换为动态卷积收敛较慢且性能仅为$64.8\%$，还不如其静态卷积版本的$65.4\%$。

作者认为是注意力的稀疏使得仅有部分卷积核得到训练，这使得训练低效，这种低效会随着网络的加深而变得更为严重。为验证该问题，作者仅把**DY-MobileNetV2**模型的每个模块的最后**1x1**卷积替换为动态卷积，可以看到训练收敛更快，精度更高($65.9\%$)。

![](https://pic.imgdb.cn/item/63b0fd782bbf0e79942c4663.jpg)

为了缓解动态卷积训练的困难，作者提出采用平滑注意力方式促使更多卷积核同时优化。该平滑过程描述如下，改进的训练机制可以收敛更快，精度更高。

$$ \pi_k=\frac{\exp(z_k/\tau)}{\sum_j \exp(z_j/\tau)} $$


```python
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _attention(nn.Module):
    def __init__(self, in_channels, num_experts, r=4, t=30):
        super(_attention, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, in_channels//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//r, num_experts),
            )
        self.tau = t

    def forward(self, x):
        x = torch.flatten(x)
        x = self.block(x)
        return torch.softmax(x/self.tau, dim=0)
    
    
class DynamicConv2D(_ConvNd):
    r"""
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolution
    kernel_size (int or tuple): Size of the convolving kernel
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    num_experts (int): Number of experts per layer 
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # 全局平均池化
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        # 注意力全连接层
        self._attention_fn = _attention(in_channels, num_experts)
        # 多套卷积层的权重
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs): # [b, c, h, w]
        res = []
        for input in inputs:
            input = input.unsqueeze(0) # [1, c, h, w]
            pooled_inputs = self._avg_pooling(input) # [1, c, 1, 1]
            routing_weights = self._attention_fn(pooled_inputs) # [k,]
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)
```