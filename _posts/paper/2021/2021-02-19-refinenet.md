---
layout: post
title: 'RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation'
date: 2021-02-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63f961acf144a010079eb4f7.jpg'
tags: 论文阅读
---

> RefineNet: 高分辨率语义分割的多路径优化网络.

- paper：[RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)

作者认为，在图像分割模型中，下采样和池化等操作将图像缩小到32倍以后丢失了许多细节特征，而反卷积等恢复的方法不能准确恢复很多重要的特征。一些低级的特征对于物体的边界和细节预测十分重要，有必要利用中间层的特征来恢复图像信息，生成高分辨率的预测图。

本文作者设计了**RefineNet**。利用各个级别的特征图，通过多层特征来生成高分辨率图像。用一个多路的细化网络，同时所有的组件都遵循残差结构。

**RefineNet**把编码器产生的多个分辨率特征和上一阶段解码器的输出同时作为输入，进行一系列卷积、融合、池化，使得多尺度特征的融合更加深入。

![](https://pic.downk.cc/item/5ebcea7ac2a9a83be531a81b.jpg)

**RefineNet**模块包括三个组件：
1. 残差卷积单元**RCU：Residual Conv Unit**：用于特征提取
2. 多分辨率融合单元**Muitl-resolution Fusion**：用于特征融合
3. 链式残差池化单元**Chained Residual Pooling**：通过一级一级的残差学习帮助模型学习到较好的残差校正结果

![](https://pic.downk.cc/item/5ebceacbc2a9a83be5320358.jpg)

残差卷积单元的实现如下，分别处理四种分辨率的输入特征：

```python
class ResidualConvUnit(nn.ModuleList):
    def __init__(self, in_channels):
        super(ResidualConvUnit, self).__init__()
        for i in range(4):
            self.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels//(2**(3 - i)) , in_channels//(2**(3 - i)) , 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels//(2**(3 - i)) ),
                    nn.ReLU(),
                    nn.Conv2d(in_channels//(2**(3 - i)) , in_channels//(2**(3 - i)) , 3, padding=1, bias=False),
                )
            )  
        
        
    def forward(self, x):
        outs = []   
        for index, module in enumerate(self):
            x1 = module(x[index])
            x1 = x[index] + x1
            outs.append(x1)
        return outs
```

多分辨率融合单元的实现如下，对四种分辨率的特征进行融合：

```python
class MultiResolutionFusion(nn.ModuleList):
    def __init__(self, in_channels, out_channels, scale_factors = [1,2,4,8]):
        super(MultiResolutionFusion, self).__init__()
        self.scale_factors = scale_factors
        
        for index, scale in enumerate(scale_factors):
            self.append(
                nn.Sequential(
                    nn.Conv2d(in_channels//2** (len(scale_factors)-index-1), out_channels, kernel_size=3, padding=1)
                    )
            )
 
    def forward(self, x):
        outputs = []
        for index, module in enumerate(self):
            xi = module(x[index])
            xi = F.interpolate(xi, scale_factor=self.scale_factors[index], mode='bilinear', align_corners=True)
            outputs.append(xi)
        return outputs[0] + outputs[1] + outputs[2] + outputs[3]
```

链式残差池化单元的实现如下：

```python
class ChainedResidualPool(nn.ModuleList):
    def __init__(self, in_channels, blocks=4):
        super(ChainedResidualPool, self).__init__()
        self.in_channels = in_channels
        self.blocks = blocks
        self.relu = nn.ReLU()
        for i in range(blocks):
            self.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(in_channels, in_channels,kernel_size=3, padding=1, stride=1, bias=False),
                )
            )
        
    def forward(self, x):
        x = self.relu(x)
        path = x
        for index, CRP in enumerate(self):
            path = CRP(path)
            x = x + path  
        return x
```

**RefineNet**构建如下：

```python
class RefineNet(nn.Module):
    def __init__(self, num_classes):
        super(RefineNet, self).__init__()
        self.backbone = ResNet.resnet101() # 输出四种尺寸的特征图
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False),
        )
        self.ResidualConvUnit = ResidualConvUnit(2048)
        self.MultiResolutionFusion = MultiResolutionFusion(2048, 256)
        self.ChainedResidualPool = ChainedResidualPool(256)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.ResidualConvUnit(x)
        x = self.MultiResolutionFusion(x)
        x = self.ChainedResidualPool(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x =  self.final(x)
        return x
```