---
layout: post
title: 'Deformable Convolutional Networks'
date: 2022-12-24
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63af955008b683016344ac05.jpg'
tags: 论文阅读
---

> 可变形卷积神经网络.

- paper：[Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)

**可变形卷积(Deformable Convolution)**是指在卷积核所处理的每一个特征元素上额外增加了一个方向向量，等价于卷积核变为任意形状，从而在训练过程中扩展感受野范围。

![](https://pic.imgdb.cn/item/63af95b408b6830163450792.jpg)

标准的卷积核通常是固定尺寸和形状的，对未知目标的变化适应性差；由于不同的位置可能对应有不同尺度或者不同形变的物体，因此通过可变形卷积根据实际情况调整本身的形状，更好的提取输入的特征。

![](https://pic.imgdb.cn/item/63af96fd08b68301634641cf.jpg)

通常标准的卷积操作可以写为特征图上一块局部区域的加权平均：

$$ y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0+p_n) $$

其中$p_0$是输出特征的空间位置，$\mathcal{R}=[-\lfloor\frac{n}{2}\rfloor,...,\lfloor\frac{n}{2}\rfloor]\times [-\lfloor\frac{n}{2}\rfloor,...,\lfloor\frac{n}{2}\rfloor]$是卷积核中元素的索引。

可变形卷积是在加权平均时，对卷积核每个元素查询输入特征的位置加了偏移项$\Delta p_n$：

$$ y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0+p_n+\Delta p_n) $$

![](https://pic.imgdb.cn/item/63af98b908b68301634825da.jpg)

尺寸为$\sqrt{N} \times \sqrt{N}$的可变形卷积核作用于特征的每一个空间位置$p_0$处都会引入$2N$个偏移项(分别控制水平和垂直方向的偏移)，则对于$H \times W$的特征图共引入$2N\times H \times W$个偏移项参数，可以通过对输入特征应用标准卷积构造。

偏移项$\Delta p_n$是通过网络学习得到的，可能为浮点数；而像素坐标值应该为整数。因此对于可变形卷积中的每个像素坐标$p=p_0+p_n+\Delta p_n$，通过左上角坐标**q_lt**、左下角(**q_lb**)、右上角(**q_rt**)、右下角(**q_rb**)坐标进行双线性插值得到：

![](https://pic.imgdb.cn/item/63afa63c08b683016355a35f.jpg)

```python
import torch
from torch import nn

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None):
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

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # 学习坐标偏移量 Δpn
        offset = self.p_conv(x) # (b, 2N, h, w)

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

最后的`_reshape_x_offset`操作是把卷积核操作每次对应的特征区域不重叠地排列出来，以方便通过步长等于卷积核尺寸的卷积构造可变形卷积的输出：

![](https://pic.imgdb.cn/item/63afb1e108b683016361d051.gif)


