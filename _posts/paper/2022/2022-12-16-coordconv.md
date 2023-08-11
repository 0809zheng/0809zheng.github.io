---
layout: post
title: 'An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution'
date: 2022-12-16
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64d49ec61ddac507cc923c71.jpg'
tags: 论文阅读
---

> 卷积神经网络的一个有趣的弱点与CoordConv解.

- paper：[An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)

卷积神经网络的应用十分广泛，但是卷积神经网络在进行坐标变换时存在缺陷，即它无法将笛卡尔空间中的坐标表示转换为**one-hot**像素空间中的坐标。本文提出了一种**CoordConv**的解决方案。

# 1. 卷积神经网络的缺陷

## （1）监督式渲染 Supervised Rendering
监督式渲染任务是指，向一个网络中输入$(i, j)$坐标，要求它输出一个$64×64$的图像，并在坐标处画一个正方形。可以借鉴图片生成的方法，用反卷积层画正方形。

为了测试这种方法，创建了一个数据集，其中在$64×64$的画布上随机放置了一些$9×9$的方块，将数据集中方块所有可能的位置列出后，总共有$3136$个样本。为了评估模型生成的表现，将样本分为两组训练/测试数据集：一组是将数据集中$80\%$坐标用于训练，$20\%$用于测试；另一组中将画布从中分为四个象限，坐标位于前三个象限的用于训练，第四象限的坐标用于测试。两组数据的分布如图所示。

![](https://pic.imgdb.cn/item/64d4a1631ddac507cc98ccdf.jpg)

结果表明，**CNN**表现得极差。即使有**1M**的参数、训练了**90**分钟，模型在第一个数据集上也没达到**0.83**的**IOU**分数，在第二个数据集上设置都没超过**0.36**。

![](https://pic.imgdb.cn/item/64d4a1b61ddac507cc999a93.jpg)

## （2）监督式坐标分类

监督式坐标分类任务是指让网络简单地绘制一个像素，其中的数据集包括成对的$(i, j)$坐标，并且有单一对应像素的图像。

![](https://pic.imgdb.cn/item/64d4a1881ddac507cc99276d.jpg)

作者又尝试了拥有不同参数的网络，发现即使有些网络能记住训练集，但没有一个的测试准确率超过$86\%$。并且训练时间都超过了一小时。

![](https://pic.imgdb.cn/item/64d4a2221ddac507cc9ab156.jpg)

# 2. CoordConv

卷积神经网络具有平移等变性，也就是说当每个卷积过滤器应用到输入上时，它不知道每个过滤器在哪。**CoordConv**在输入特征上添加两个分别表示$x$, $y$坐标的通道，从而打破卷积的平移等变性。

![](https://pic.imgdb.cn/item/64d4a2eb1ddac507cc9c795e.jpg)

```python
class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
​
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
​
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
​
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
​
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
​
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
​
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
​
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret
​
​
class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
​
    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
```


**CoordConv**模型在监督式坐标分类和监督式渲染任务上都达到了最佳训练和测试性能。另外，**CoordConv**的参数比之前少**10—100**倍，训练时间比之前快了**150**倍。

![](https://pic.imgdb.cn/item/64d4a33a1ddac507cc9d3a59.jpg)