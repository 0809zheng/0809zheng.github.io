---
layout: post
title: '轻量级卷积神经网络'
date: 2021-09-10
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6183331c2ab3f51d915b0c47.jpg'
tags: 深度学习
---

> Lightweight Convolutional Neural Networks.

卷积神经网络被广泛应用在图像分类、目标检测等视觉任务中，并取得了巨大的成功。然而，卷积神经网络通常需要较大的运算量和内存占用，在嵌入式设备等资源受限的环境中受到限制，因此需要进行网络压缩。

**轻量级网络设计**是网络压缩的一种方法，旨在设计计算复杂度更低的网络结构。
从**结构**的角度考虑，卷积层提取的特征存在冗余，可以设计特殊的卷积操作，减少卷积操作的冗余，从而减少计算量。
从**计算**的角度，模型推理过程中存在大量乘法运算，而乘法操作(相比于加法)对于目前的硬件设备不友好，可以对乘法运算进行优化，也可以减少计算量。

本文目录：
1. 设计特殊的卷积
2. 寻找乘法的替代


# 1. 设计特殊的卷积

一个标准的$3\times 3$卷积层表示如下：

![](https://pic.imgdb.cn/item/6183340f2ab3f51d915bf334.jpg)

```python
class VanillaConv(nn.Module):
    """(convolution => [BN] => [ReLU])"""
    def __init__(
            self, in_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, groups=1,
            bn=True, relu=True
            ):
        super().__init__()
        self.vanilla_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
        )
        if bn:
            self.vanilla_conv.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        if relu:
            self.vanilla_conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.vanilla_conv(x)
```

下面介绍一些特殊设计的卷积神经网络：

| 轻量级网络 | 卷积层 | 特殊结构 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---:  |
| [<font color=Blue>SqueezeNet</font>](https://0809zheng.github.io/2021/09/16/squeezenet.html) | 标准卷积 | Fire模块   |
| [<font color=Blue>SqueezeNext</font>](https://0809zheng.github.io/2021/09/17/squeezenext.html) | 标准卷积 | 分离卷积($3\times 1+1\times 3$)   |
| [<font color=Blue>MobileNet</font>](https://0809zheng.github.io/2021/09/13/mobilenetv1.html) | 深度可分离卷积 | 深度(depth-wise)卷积, 逐点(point-wise)卷积   |
| [<font color=Blue>MobileNetV2</font>](https://0809zheng.github.io/2021/09/14/mobilenetv2.html) | 深度可分离卷积 | 线性瓶颈(linear bottleneck), 倒残差(inverted residual)   |
| [<font color=Blue>MobileNetV3</font>](https://0809zheng.github.io/2021/09/15/mobilenetv3.html) | 深度可分离卷积 | 通道注意力机制(SENet), 神经结构搜索(NAS)   |
| [<font color=Blue>ShuffleNet</font>](https://0809zheng.github.io/2021/09/18/shufflenet.html) | 组卷积+深度卷积 | 通道打乱(channel shuffle)   |
| [<font color=Blue>ShuffleNet V2</font>](https://0809zheng.github.io/2021/09/19/shufflenetv2.html) | 标准卷积+深度卷积 | 通道拆分(channel split), 通道打乱(channel shuffle)   |
| [<font color=Blue>IGCNet</font>](https://0809zheng.github.io/2021/09/21/igc.html) | 组卷积 | 交错组卷积(overleaved group conv)   |
| [<font color=Blue>IGCV2</font>](https://0809zheng.github.io/2021/09/22/igcv2.html) | 组卷积 | 交错结构化稀疏卷积(overleaved structured sparse conv)   |
| [<font color=Blue>ChannelNet</font>](https://0809zheng.github.io/2021/09/20/channelnet.html) | 深度卷积+组卷积+通道卷积 | 组通道卷积, 深度可分离通道卷积, 卷积分类层   |
| [<font color=Blue>EfficientNet</font>](https://0809zheng.github.io/2021/09/11/efficientv1.html) | MBConv(即MobileNetV3) | 复合缩放(compound scaling)   |
| [<font color=Blue>EfficientNetV2</font>](https://0809zheng.github.io/2021/09/12/efficientv2.html) | Fused-MBConv | 渐进训练   |
| [<font color=Blue>GhostNet</font>](https://0809zheng.github.io/2021/11/08/ghostnet.html) | Ghost模块 | Ghost BottleNeck   |
| [<font color=Blue>MicroNet</font>](https://0809zheng.github.io/2021/11/09/micronet.html) | 微因子卷积 | 微因子(micro-factorized)深度卷积和逐点卷积  |
| [<font color=Blue>CompConv</font>](https://0809zheng.github.io/2021/08/03/compconv.html) | 分治卷积 | -   |



### ⚪  [<font color=Blue>SqueezeNet</font>](https://0809zheng.github.io/2021/09/16/squeezenet.html)：使用Fire模块代替普通卷积

![](https://pic.imgdb.cn/item/618334e22ab3f51d915cc600.jpg)

```python
class Fire(nn.Module):
    """
    (1x1 convolution => [BN] => ReLU 
    => 1x1+3x3 convolution => [BN] => ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.e1x1 = out_channels//2
        self.e3x3 = out_channels-self.e1x1
        self.s1x1 = out_channels//4
        self.squeeze = VanillaConv(in_channels, self.s1x1, kernel_size=1, padding=0)
        self.expand1x1 = nn.Conv2d(self.s1x1, self.e1x1, kernel_size=1, padding=0)
        self.expand3x3 = nn.Conv2d(self.s1x1, self.e3x3, kernel_size=3, padding=1)
        self.tail = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        e1 = self.expand1x1(s)
        e2 = self.expand3x3(s)
        e = torch.cat([e1,e2],1)
        return self.tail(e)
```

### ⚪ [<font color=Blue>SqueezeNext</font>](https://0809zheng.github.io/2021/09/17/squeezenext.html)：使用分离卷积构造标准卷积块

![](https://pic.imgdb.cn/item/6183357f2ab3f51d915d4997.jpg)

```python
class SqNxt(nn.Module):
    """
    (1x1 convolution => [BN] => ReLU 
    => 1x1 convolution => [BN] => ReLU 
    => 3x1 convolution => [BN] => ReLU 
    => 1x3 convolution => [BN] => ReLU 
    => 1x1 convolution => [BN] => ReLU)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        print(in_channels)
        self.sqnxt = nn.Sequential(
            VanillaConv(in_channels, in_channels//2, kernel_size=1, padding=0),
            VanillaConv(in_channels//2, in_channels//4, kernel_size=1, padding=0),
            VanillaConv(in_channels//4, in_channels//2, kernel_size=(3,1), padding=(1,0)),
            VanillaConv(in_channels//2, in_channels//2, kernel_size=(1,3), padding=(0,1)),
            VanillaConv(in_channels//2, out_channels, kernel_size=1, padding=0)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.sqnxt(x)+self.shortcut(x)
```


### ⚪ [<font color=Blue>MobileNet</font>](https://0809zheng.github.io/2021/09/13/mobilenetv1.html)：使用深度可分离卷积(Depthwise Separable Conv)代替普通卷积

![](https://pic.imgdb.cn/item/6183364c2ab3f51d915e09e9.jpg)

```python
class DSConv(nn.Module):
    """
    (depthwise convolution => [BN] => ReLU6
        => 1x1 convolution => [BN] => ReLU6)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise_separable_conv = nn.Sequential(
            VanillaConv(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            # 此处激活函数为 nn.ReLU6(inplace=True)
            VanillaConv(in_channels, out_channels, kernel_size=1, padding=0),
            # 此处激活函数为 nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.depthwise_separable_conv(x)
```

### ⚪ [<font color=Blue>MobileNetV2</font>](https://0809zheng.github.io/2021/09/14/mobilenetv2.html)：为MobileNet引入线性瓶颈(linear bottleneck)，并设计倒残差(inverted residual)结构

![](https://pic.imgdb.cn/item/618337072ab3f51d915eadba.jpg)

```python
class DSConvv2(nn.Module):
    """
    (1x1 convolution => [BN] => ReLU6
    => depthwise convolution => [BN] => ReLU6 
    => 1x1 convolution => [BN] => Linear)
        """
    def __init__(self, in_channels, out_channels, t=6):
        super().__init__()
        self.inverted_residual = nn.Sequential(
            VanillaConv(in_channels, t*in_channels, kernel_size=1, padding=0),
            # 此处激活函数为 nn.ReLU6(inplace=True)
            VanillaConv(t*in_channels, t*in_channels, kernel_size=3, padding=1, groups=t*in_channels),
            # 此处激活函数为 nn.ReLU6(inplace=True)
            VanillaConv(t*in_channels, out_channels, kernel_size=1, padding=0, relu=False)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.inverted_residual(x)+self.shortcut(x)
```


### ⚪  [<font color=Blue>MobileNetV3</font>](https://0809zheng.github.io/2021/09/15/mobilenetv3.html)：引入通道注意力(Channel Attention)，通过神经结构搜索网络

![](https://pic.imgdb.cn/item/618337ee2ab3f51d915f7e3f.jpg)

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

class DSConvv3(nn.Module):
    """
    (1x1 convolution => [BN] => Hardswish 
    => depthwise convolution => [BN] => Hardswish 
    => SELayer 
    => 1x1 convolution => [BN] => Linear)
    """
    def __init__(self, in_channels, out_channels, t=6):
        super().__init__()
        self.block = nn.Sequential(
            VanillaConv(in_channels, t*in_channels, kernel_size=1, padding=0),
            # 此处激活函数为 nn.Hardswish(inplace=True)
            VanillaConv(t*in_channels, t*in_channels, kernel_size=3, padding=1, groups=t*in_channels),
            # 此处激活函数为 nn.Hardswish(inplace=True)
            SELayer(t*in_channels),
            VanillaConv(t*in_channels, out_channels, kernel_size=1, padding=0, relu=True)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
```

### ⚪ [<font color=Blue>ShuffleNet</font>](https://0809zheng.github.io/2021/09/18/shufflenet.html)：使用组卷积(Group Conv)和通道打乱(Channel Shuffle)代替普通卷积
![](https://pic.imgdb.cn/item/618338722ab3f51d915ff3bd.jpg)
```python
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class ShuffleNet(nn.Module):
    """
    (1x1 group convolution => [BN] => ReLU => ChannelShuffle
    => depthwise convolution => [BN] 
    => 1x1 group convolution => [BN] => Linear)"""
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        mid_channels = int(0.25*in_channels)
        # 如果输入通道太少则无法分组
        g = 1 if in_channels<groups**2 else groups
        self.shuffle_block = nn.Sequential(
            VanillaConv(in_channels, mid_channels, kernel_size=1, padding=0, groups=g),
            ShuffleBlock(groups=g),
            VanillaConv(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, relu=False),
            VanillaConv(mid_channels, out_channels, kernel_size=1, padding=0, groups=groups, relu=False)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return F.relu(self.shuffle_block(x)+self.shortcut(x))
```

### ⚪ [<font color=Blue>ShuffleNet V2</font>](https://0809zheng.github.io/2021/09/19/shufflenetv2.html)：为ShuffleNet引入通道拆分(Channel Split)
![](https://pic.imgdb.cn/item/618338d52ab3f51d916041a3.jpg)
```python
# ShuffleBlock定义见ShuffleNet  
class ShuffleNetv2(nn.Module):
    """
    (ChannelSplit 
    => 1x1 convolution => [BN] => ReLU 
    => depthwise convolution => [BN] 
    => 1x1 convolution => [BN] => ReLU 
    => ChannelShuffle)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 需要处理输入输出特征通道数不相等的情况
        self.cin = in_channels//2
        self.cout = out_channels//2
        self.block = nn.Sequential(
            VanillaConv(self.cin, self.cin, kernel_size=1, padding=0),
            VanillaConv(self.cin, self.cin, kernel_size=3, padding=1, groups=self.cin, relu=False),
            VanillaConv(self.cin, self.cout, kernel_size=1, padding=0)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(self.cin, self.cout, kernel_size=1, padding=0))
        else:
            self.shortcut = nn.Sequential()
        self.shuffle = ShuffleBlock(groups=2)

    def forward(self, x):
        # chunk方法可以对张量分块，后面的块通道数可能少一些
        x1, x2 = x.chunk(2, dim=1)
        y = torch.cat([self.shortcut(x1), self.block(x2)], dim=1)
        return self.shuffle(y)
```

### ⚪  [<font color=Blue>IGCNet</font>](https://0809zheng.github.io/2021/09/21/igc.html)：使用交错组卷积代替普通卷积

![](https://pic.imgdb.cn/item/617a5dac2ab3f51d91e01847.jpg)

```python
# ShuffleBlock定义见ShuffleNet    
class IGCNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.M = 2
        self.L = out_channels//self.M
        self.igconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=self.L),
            nn.BatchNorm2d(out_channels),
            ShuffleBlock(groups=self.L),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, groups=self.M),
            nn.BatchNorm2d(out_channels),
            ShuffleBlock(groups=self.M),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.igconv(x)
```

### ⚪  [<font color=Blue>IGCV2 </font>](https://0809zheng.github.io/2021/09/22/igcv2.html)：使用交错结构化稀疏卷积代替普通卷积

![](https://pic.imgdb.cn/item/617bbd5b2ab3f51d910a8af6.jpg)

```python
# ShuffleBlock定义见ShuffleNet    
class IGCV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.K = 8
        self.L = math.ceil(math.log(out_channels)/math.log(self.K))+1
        self.igcv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels//self.K),
            nn.BatchNorm2d(out_channels),
        )
        for l in range(self.L-1):
            self.igcv2.add_module('shuffle'+str(l+2),
                                  ShuffleBlock(groups=out_channels//self.K))
            self.igcv2.add_module('groupconv'+str(l+2), 
                                  nn.Conv2d(out_channels, out_channels, 
                                            kernel_size=1, padding=0, groups=out_channels//self.K))
            self.igcv2.add_module('batchnorm'+str(l+2), 
                                  nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.relu(self.igcv2(x))
```

### ⚪  [<font color=Blue>ChannelNet</font>](https://0809zheng.github.io/2021/09/20/channelnet.html)：使用通道卷积代替普通卷积

![](https://pic.imgdb.cn/item/6177d9972ab3f51d91f518b8.jpg)

```python
class ChannelConv(nn.Module):
    def __init__(self, group, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv3d(1, group, kernel_size, 
                              stride=(group, 1, 1), 
                              padding=(padding, 0, 0), 
                              bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        return x

GCWConv = ChannelConv(g, (f, 1, 1), (f-g)//2)
DWSCWConv = ChannelConv(1, (f, 1, 1), (f-1)//2)
CCL = ChannelConv(1, (m-n+1, df, df), 0)
```


### ⚪  [<font color=Blue>EfficientNet</font>](https://0809zheng.github.io/2021/09/11/efficientv1.html)：复合缩放网络深度、宽度和分辨率

基本结构与**MobileNetV3**相同，作者称之为**MBConv**：

![](https://pic.imgdb.cn/item/618337ee2ab3f51d915f7e3f.jpg)

复合缩放网络的深度、宽度和分辨率。

![](https://pic.imgdb.cn/item/61d4ed982ab3f51d91ddc9ed.jpg)

### ⚪  [<font color=Blue>EfficientNetV2</font>](https://0809zheng.github.io/2021/09/12/efficientv2.html)：复合缩放结构，渐进训练网络

基本结构采用**MBConv**和一种改进的**Fused-MBConv**。**MBConv**与**MobileNetV3**相同，**Fused-MBConv**是将其中的深度可分离卷积还原为标准卷积。

![](https://pic.imgdb.cn/item/61d558bb2ab3f51d91428d0c.jpg)

```python
# SELayer定义见MobileNetV3
class Fused-MBConv(nn.Module):
    """
    (3x3 convolution => [BN] => ReLU
    => SELayer 
    => 1x1 convolution => [BN] => Linear)
    """
    def __init__(self, in_channels, out_channels, t=4):
        super().__init__()
        self.block = nn.Sequential(
            VanillaConv(in_channels, t*in_channels, kernel_size=3, padding=1, relu=True), 
            SELayer(t*in_channels),
            VanillaConv(t*in_channels, out_channels, kernel_size=1, padding=0, relu=True)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
```

### ⚪  [<font color=Blue>GhostNet</font>](https://0809zheng.github.io/2021/11/08/ghostnet.html)：使用Ghost模块代替普通卷积

![](https://pic.imgdb.cn/item/6188de152ab3f51d912076c8.jpg)

```python
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, s=2, d=3, relu=True):
        super().__init__()
        self.s = s
        self.d = d
        self.mid = out_channels//self.s
        self.primary_conv = VanillaConv(in_channels, self.mid, 
                                        kernel_size=1, padding=0, relu=True)
        self.cheap_operation = VanillaConv(self.mid, out_channels-self.mid, 
                                           kernel_size=self.d, padding=self.d//2, 
                                           groups=self.mid, relu=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out
```

### ⚪  [<font color=Blue>MicroNet</font>](https://0809zheng.github.io/2021/11/09/micronet.html)：使用微因子卷积代替普通卷积

![](https://pic.imgdb.cn/item/61921ecf2ab3f51d9124c0c7.jpg)

### ⚪  [<font color=Blue>CompConv</font>](https://0809zheng.github.io/2021/08/03/compconv.html)：使用分治卷积代替普通卷积

![](https://pic.imgdb.cn/item/610919d05132923bf8261f46.jpg)

# 2. 寻找乘法的替代

### ⚪  [<font color=Blue>AdderNet</font>](https://0809zheng.github.io/2020/09/26/addernet.html)：使用L1距离代替卷积乘法

卷积神经网络的计算可以表示为卷积滤波器$F \in \Bbb{R}^{d \times d \times c_{in} \times c_{out}}$和输入特征$X \in \Bbb{R}^{H \times W \times c_{in}}$的乘积：

$$ Y(m,n,t)=\sum_{i=0}^{d} {\sum_{j=0}^{d} {\sum_{k=0}^{c_{in}} {S(X(m+i,n+j,k),F(i,j,k,t))}}} $$

使用**L1**距离代替卷积计算中的乘法：

$$ Y(m,n,t)=-\sum_{i=0}^{d} {\sum_{j=0}^{d} {\sum_{k=0}^{c_{in}} {| X(m+i,n+j,k)-F(i,j,k,t) | }}} $$

### ⚪  [<font color=Blue>Mitchell’s approximate</font>](https://0809zheng.github.io/2020/09/26/addernet.html)：使用Mitchell近似代替卷积乘法
二进制下的乘法运算可以通过对数和指数转换转变成加法运算：

$$ pq=2^s, \quad s=\log_2p+\log_2q $$

因此计算$p$和$q$的乘积，可以先通过**Mitchell**近似计算快速对数$\log_2p$和$\log_2q$，将其相加后得到$s$；再通过**Mitchell**近似计算快速指数$2^s$。


