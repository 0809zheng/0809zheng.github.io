---
layout: post
title: 'Polarized Self-Attention: Towards High-quality Pixel-wise Regression'
date: 2021-10-04
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6437c3e40d2dde57774ff099.jpg'
tags: 论文阅读
---

> 极化自注意力: 面向高质量像素级回归.

- paper：[Polarized Self-Attention: Towards High-quality Pixel-wise Regression](https://arxiv.org/abs/2107.00782)

本文针对细粒度的像素级任务（比如语义分割）提出了一种更加精细的双重注意力机制：极化自注意力（**Polarized Self-Attention**）。极化自注意力相比于其他注意力机制主要有两个设计上的亮点：
1. 在通道和空间维度保持比较高的分辨率，这一步能够减少降低维度造成的信息损失；
2. 在通道和空间分支都在采用了**Softmax**和**Sigmoid**相结合的非线性函数进行概率估计，能够拟合出细粒度回归结果的输出分布。

![](https://pic.imgdb.cn/item/643fa1e20d2dde5777ea3b81.jpg)

在设计极化自注意力时，作者参考了极化滤波（**polarized filtering**）机制。在摄影时，在横向总是有随机光产生眩光/反射。极化滤波的作用就是只允许正交于横向方向的光通过，以此来提高照片的对比度。 由于在滤波过程中，总强度会损失，所以滤波后的光通常动态范围较小，因此需要额外的提升，用来以恢复原始场景的详细信息。

基于上面的思想，作者提出了极化自注意力机制，先在一个方向上对特征进行压缩，然后对损失的强度范围进行提升；类似于光学透镜过滤光一样，每个自注意力的作用都是用于增强或者抑制特征。具体可分为两个结构：
1. 滤波（**Filtering**）：使得一个维度的特征（比如通道维度）完全坍塌，同时让正交方向的维度（比如空间维度）保持高分辨率。
2. **High Dynamic Range（HDR）**：首先在**attention**模块中最小的**tensor**上用**Softmax**函数来增加注意力的范围，然后再用**Sigmoid**函数进行动态的映射。

![](https://pic.imgdb.cn/item/643fa27e0d2dde5777eb0695.jpg)

### ⚪ Channel-only branch

对于通道维度的分支，先用了**1x1**的卷积将输入的特征$X$转换成了$Q$和$V$，其中$Q$的通道被完全压缩，而$V$的通道维度依旧保持在一个比较高的水平（也就是$C/2$）。因为$Q$的通道维度被压缩，就需要通过**HDR**进行信息的增强，因此作者用**Softmax**对$Q$的信息进行了增强。然后将$Q$和$K$进行矩阵乘法，并在后面接上**1x1**卷积将通道上$C/2$的维度升为$C$。最后用**Sigmoid**函数使得所有的参数都保持在$0-1$之间。

```python
class Channel_only_branch(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        return channel_out
```

### ⚪ Spatial-only branch

对于空间维度的分支，先用了**1x1**的卷积将输入的特征转换为了$Q$和$V$，其中对于$Q$特征，作者还用了**GlobalPooling**对空间维度压缩，转换成了**1x1**的大小；而$V$特征的空间维度则保持在一个比较大的水平（$HxW$）。由于$Q$的空间维度被压缩了，所以作者就用了**Softmax**对$Q$的信息进行增强。然后将$Q$和$K$进行矩阵乘法，然后接上**reshape**和**Sigmoid**使得所有的参数都保持在$0-1$之间。

```python
class Spatial_only_branch(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()
        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        return out
```

### ⚪ Composition

对于两个分支的结果，作者提出了两种融合的方式：并联和串联（先进行通道上的注意力，再进行空间上的注意力）：

![](https://pic.imgdb.cn/item/643fa55c0d2dde5777ee0c52.jpg)