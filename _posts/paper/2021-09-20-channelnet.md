---
layout: post
title: 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions'
date: 2021-09-20
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6173ccb32ab3f51d91fb3bcb.jpg'
tags: 论文阅读
---

> ChannelNets: 使用通道卷积构建高效卷积神经网络.

- paper：ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions
- arXiv：[link](https://arxiv.org/abs/1809.01330)
- code：[github](https://github.com/GenDisc/ChannelNet)

标准卷积的输出特征的通道和输入特征的通道通常是全连接的，即输出特征的每个通道由输入特征的所有通道共同决定。

深度可分离卷积的通道连接情况如下图(a)所示，包括深度卷积和逐点卷积。深度卷积的通道是稀疏连接的，其输出的每一个通道只与其对应的输入通道有关；逐点卷积的通道是全连接的。

下图(b)表示逐点卷积应用了组卷积的情况。组卷积预先把特征通道进行分组，卷积只在不同的通道组内进行，这使得通道的连接变得稀疏。由于不同组之间的特征无法交流，组卷积后通常使用通道打乱操作。

![](https://pic.imgdb.cn/item/6173cc962ab3f51d91fb20fb.jpg)

作者提出了**通道卷积**(**channel-wise conv**)，将卷积层中的特征图的密集连接替换为稀疏连接。通道卷积是在通道维度上进行滑动，实践中可以通过**Conv3d**来实现(将通道看作一个空间维度)。基于通道卷积构建了轻量级网络**ChannelNet**。**ChannelNet**使用了三种通道卷积模块：组通道卷积、深度可分离通道卷积以及卷积分类层。

## 1. 组通道卷积 GCWConv
**组通道卷积**(**group channel-wise conv**)模块可以解决由于组卷积造成的不同组之间的信息不一致问题，如上图(c)所示。一个组卷积模块通常堆叠两次重复的深度卷积+组卷积，并使用通道卷积进行不同组之间的交互。

以分成$g$组为例，相当于使用$g$个**Conv3d**沿通道维度滑动，则滑动步长设置为$g$，通过调整卷积核尺寸$f$以及填充$p$，每个**Conv3d**生成长度为$n/g$的特征，最后将这些特征连接起来。上述参数应满足：

$$ \frac{n+2p-f}{g}+1=\frac{n}{g} $$

此时的通道卷积实现如下：
```python
ChannelConv = nn.Conv3d(1, g, 
                        kernel_size=(f, 1, 1), 
                        stride=(g, 1, 1), 
                        padding=((f-g)//2, 0, 0))
# x = (N,C,H,W)
x = x.unsqueeze(1)  # x = (N,1,C,H,W)
x = ChannelConv(x)  # x = (N,g,C/g,H,W)
x = x.view(x.size(0), -1, x.size(3), x.size(4))
```

## 2. 深度可分离通道卷积 DWSCWConv
**深度可分离通道卷积**(**depth-wise separable channel-wise conv**)是指使用通道卷积代替了深度可分离卷积中的逐点卷积，如上图(d)所示。此时相当于使用一个**Conv3d**沿通道维度滑动，卷积核尺寸$f$以及填充$p$应满足：

$$ \frac{n+2p-f}{1}+1=n $$

此时的通道卷积实现如下：
```python
ChannelConv = nn.Conv3d(1, 1, 
                        kernel_size=(f, 1, 1), 
                        stride=(1, 1, 1), 
                        padding=((f-1)//2, 0, 0))
# x = (N,C,H,W)
x = x.unsqueeze(1)  # x = (N,1,C,H,W)
x = ChannelConv(x)  # x = (N,1,C,H,W)
x = x.view(x.size(0), -1, x.size(3), x.size(4))
```

## 3. 卷积分类层 CCL
全连接分类层通常具有较大的参数量(如**MobileNet**中占总参数的$24.33\%$)。全连接分类层通常包含一个全局池化层和一个全连接层，如下图所示。

![](https://pic.imgdb.cn/item/6173cce82ab3f51d91fb6cb0.jpg)

作者提出了一个**卷积分类层**(**convolutional classification layer**)，实际上是一个三维卷积层，将输入特征看作尺寸为$d_f\times d_f \times m$的单通道特征图，使用一个尺寸为$d_f \times d_f \times (m-n+1)$的三维卷积核，则可得到尺寸为$1\times 1 \times n$的输出特征。

此时的通道卷积实现如下：
```python
ChannelConv = nn.Conv3d(1, 1, 
                        kernel_size=(m-n+1, df, df), 
                        stride=(1, 1, 1), 
                        padding=(0, 0, 0))
# x = (N,m,df,df)
x = x.unsqueeze(1)  # x = (N,1,m,df,df)
x = ChannelConv(x)  # x = (N,1,n,1,1)
x = x.view(x.size(0), -1)
```

## 4. ChannelNet
**ChannelNet**以**MobileNet**为基本框架，提出了不同程度的修改。
- **ChannelNet-v1**：将**MobileNet**基本模块(称为**DWSConv**模块)中的$1\times 1$卷积替换为组卷积，如下图(a)所示(称为**GM**模块)；由于组卷积限制了不同组之间的交流，因此在最后再引入组通道卷积，如下图(b)所示(称为**GCWM**模块)。

![](https://pic.imgdb.cn/item/6173cd022ab3f51d91fb8062.jpg)

- **ChannelNet-v2**：将输出卷积替换为深度可分离通道卷积模块。

- **ChannelNet-v3**：将输出层替换为卷积分类层。

三种网络结构如下表：

![](https://pic.imgdb.cn/item/617607cc2ab3f51d91a914ef.jpg)

三种网络的实验结果如下：

![](https://pic.imgdb.cn/item/6173cdbf2ab3f51d91fc1b27.jpg)