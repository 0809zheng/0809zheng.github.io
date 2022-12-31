---
layout: post
title: 'Deformable ConvNets v2: More Deformable, Better Results'
date: 2022-12-23
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63afdd022bbf0e79948839f8.jpg'
tags: 论文阅读
---

> 改进的可变形卷积神经网络.

- paper：[Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)

[<font color=blue>Deformable Convolution</font>](https://0809zheng.github.io/2022/12/24/deformable.html)通过对卷积核每个元素查询输入特征的位置加了偏移项$\Delta p_n$，使得网络具有任意形状的感受野。

$$ y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0+p_n+\Delta p_n) $$

本文作者指出，上述可变形卷积对偏置项的范围没有约束，因此可能导致感受野跑出目标的感兴趣区域，使得特征被其他无关内容影响。

![](https://pic.imgdb.cn/item/63afdf742bbf0e799493a30b.jpg)

基于此引入一种调节机制，对每个偏移项$\Delta p_n$额外学习一个偏移权重$\Delta m_n \in [0,1]$，用于评估该特征位置的重要性程度。$\Delta m_n$是由数据学习得到的，因此不能与卷积核权重$w(p_n)$合并。

$$ y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0+p_n+\Delta p_n) \cdot \Delta m_n $$

此时尺寸为$\sqrt{N} \times \sqrt{N}$的可变形卷积核作用于特征的每一个空间位置$p_0$处都会引入$2N$个偏移项(分别控制水平和垂直方向的偏移)和N个偏移权重，则对于$H \times W$的特征图共引入$2N\times H \times W$个偏移项参数和$N\times H \times W$个偏移权重参数，可以通过对输入特征应用标准卷积构造。

**Deformable Convolution v2**可以通过在[<font color=blue>Deformable Convolution</font>](https://0809zheng.github.io/2022/12/24/deformable.html)模块中引入`modulation`实现：

```python
import torch
from torch import nn

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # 用于执行可变形卷积
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # 用于学习卷积核的偏移项
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # 为p_conv的参数更新设置学习率
        self.p_conv.register_backward_hook(self._set_lr)
    
        self.modulation = modulation
        if modulation:
            # 用于学习偏移权重 Δmn
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # 学习坐标偏移量 Δpn
        offset = self.p_conv(x) # (b, 2N, h, w)
        # 学习偏移权重 Δmn
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x)) # (b, N, h, w)

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # 获得偏移后的像素坐标 p=p0+pn+Δpn
        p = self._get_p(offset, dtype) # (b, 2N, h, w)

        # 对像素坐标进行双线性插值，首先获取左上角和右下角坐标
        p = p.contiguous().permute(0, 2, 3, 1) # (b, h, w, 2N)
        q_lt = p.detach().floor() # 左上角坐标
        q_rb = q_lt + 1 # 右下角坐标

        # 裁剪坐标防止超出特征尺寸(h,w)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()

        # 左下角坐标和右上角坐标
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)       

        # 计算双线性插值的系数 (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 获取双线性插值像素的值 (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 计算特征双线性插值的结果x(p0+pn+Δpn) (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 调节偏移权重
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1) # (b, h, w, N)
            m = m.unsqueeze(dim=1) # (b, 1, h, w, N)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1) # (b, c, h, w, N)
            x_offset *= m        

        # 调整特征尺寸 (b, c, h*ks, w*ks)
        x_offset = self._reshape_x_offset(x_offset, ks)

        # 执行可变形卷积 y(p0) = w(pn)·x(p0+pn+Δpn)
        out = self.conv(x_offset)
        return out

    # 获得偏移后的像素坐标 p=p0+pn+Δpn
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        # 计算卷积核索引坐标 pn
        p_n = self._get_p_n(N, dtype) # (1, 2N, 1, 1)
        # 计算中心像素坐标 p0
        p_0 = self._get_p_0(h, w, N, dtype) # (1, 2N, h, w)
        p = p_0 + p_n + offset
        return p

    # 计算卷积核索引坐标 pn=[(-k/2,...,k/2),((-k/2,...,k/2))]
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0) # (2N, 1)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    # 计算中心像素坐标 p0=[(h1,...,hn),(w1,...,wn)]
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # 获取双线性插值像素的值 (b, c, h, w, N)
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # 列出输入特征的hw个像素
        x = x.contiguous().view(b, c, -1) # (b, c, h*w)
        # 根据像素坐标q计算对应到输入像素的索引（每w个元素对应一行）
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # 获取索引对应的像素值
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset
```
