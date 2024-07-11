---
layout: post
title: 'Lite-HRNet: A Lightweight High-Resolution Network'
date: 2021-05-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/668e3dffd9c307b7e97aab0c.png'
tags: 论文阅读
---

> Lite-HRNet：轻量级高分辨率网络.

- paper：[Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403)

人体姿态估计一般比较依赖于高分辨率的特征表示以获得较好的性能，但是目前的网络计算量较大，因此需要研究如何在计算资源受到约束的情况下部署一个高效的高分辨率姿态估计模型。

本文在**HRNet**模型的基础上设计了一个高性能的轻量化网络**Lite-HRNet**，首先在**HRNet**中引入**Shuffle Block**，得到了**Naive Lite-HRNet**，并且在性能和复杂度上取得了不错的权衡。进一步发现**Shuffle Block**中的1x1卷积成为了计算瓶颈，于是采用**SENet**模块替换1x1卷积进行特征聚合。

![](https://pic.imgdb.cn/item/668e3f3cd9c307b7e97c4648.png)

**Shuffle Block**会将通道首先分为两个部分，其中的一部分会送入1x1卷积+3x3深度卷积+1x1卷积进行增强，处理完后会和另一部分拼接起来，最终会把通道重新**shuffle**。通过把**HRNet Stem**中的第2个3x3卷积以及所有的**Residual Block**替换为**Shuffle Block**可得到 **Naive Lite-HRNet**。

```python
class Stem(nn.Module):
    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):  
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
		
		# Stem中的第一个卷积不使用shuffle block
		# ConvModule是MMCV中的一个基本卷积模块：conv/norm/activation
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))
		
        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels
		
		# Shuffle Block中左侧不做增强的分支
        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )
		
		# Shuffle Block中右侧增强分支
        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,  # groups=in_channels 深度可分离卷积
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)
        out = torch.cat((self.branch1(x1), x2), dim=1)
        out = channel_shuffle(out, 2)  # shuffle channel
        return out
```

其中1x1卷积的计算复杂度为$\Theta\left(C^{2}\right)$，3x3深度卷积的计算复杂度为$\Theta(9 C)$，其中$C$为通道数。在**Shuffle Block**中，当$C>5$时，两个1x1卷积的计算复杂度就会超过一个3x3深度卷积的计算复杂度。

为了降低计算复杂度，本文提出使用逐元素加权操作代替1x1卷积：

$$
\mathrm{Y}_{s}=\mathrm{W}_{s} \odot \mathrm{X}_{s}
$$
 
其中$W_s$权重从不同分辨率的特征图中计算得到，起到跨通道、跨分辨率的特征交互的作用。对于第$s$个阶段来说，其具有$s$个平行分支，每个分支的分辨率各不相同，相应地其也会有$s$个权重$W_{1}, W_{2}, \ldots, W_{s}$。这$s$个权重由$s$个分辨率特征图计算而来：

$$
\left(\mathrm{W}_{1}, \mathrm{~W}_{2}, \ldots, \mathrm{W}_{s}\right)=\mathcal{H}_{s}\left(\mathrm{X}_{1}, \mathrm{X}_{2}, \ldots, \mathrm{X}_{s}\right)
$$

其中$$\mathcal{H}_{s}$$操作是通道注意力。

```python
class CrossResolutionWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
    	# mini_size即为当前stage中最小分辨率的shape：H_s, W_s
        mini_size = x[-1].size()[-2:]  # H_s, W_s
        # 将所有stage的input均压缩至最小分辨率，由于最小的一个stage的分辨率已经是最小的了
        # 因此不需要进行压缩
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)  # ReLu激活
        out = self.conv2(out)  # sigmoid激活
        out = torch.split(out, self.channels, dim=1)
        out = [
        	# s为原输入
        	# a为权重，并通过最近邻插值还原回原输入尺度
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out
```

在引入跨分辨率信息后，本文还引入了一个单分辨率内部空间域的增强操作：

$$
\mathbf{w}_{s}=\mathcal{F}_{s}\left(\mathrm{X}_{s}\right)
$$

```python
class SpatialWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
```

**Lite-HRNet**的完整结构如下：

![](https://pic.imgdb.cn/item/668e473fd9c307b7e987ce08.png)

