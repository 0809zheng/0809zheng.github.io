---
layout: post
title: 'Multi-Context Attention for Human Pose Estimation'
date: 2021-04-06
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/649b92891ddac507cc24d883.jpg'
tags: 论文阅读
---

> 人体姿态估计中的多重上下文注意力.

- paper：[Multi-Context Attention for Human Pose Estimation](https://arxiv.org/abs/1702.07432)

本文为人体姿态估计任务设计了一种多重上下文注意力机制。首先采用[堆叠沙漏网络(stacked hourglass networks)](https://0809zheng.github.io/2021/04/03/hourglass.html)生成不同分辨率特征的注意力图，不同分辨率特征对应着不同的语义。然后利用**CRF(Conditional Random Field)**对注意力图中相邻区域的关联性进行建模。并同时结合了整体注意力模型和肢体部分注意力模型，整体注意力模型针对的是整体人体的全局一致性，部分注意力模型针对不同身体部分的详细描述，因此能够处理从局部显著区域到全局语义空间的不同粒度内容。另外还设计了新颖的沙漏残差单元(**Hourglass Residual Units, HRUs**)增加网络的接受野，可以学习得到不同尺度的特征。

![](https://pic.imgdb.cn/item/649b95661ddac507cc2a4b4f.jpg)

## 1. Nested Hourglass Network

采用**8-stack hourglass**网络作为基础网络，并采用沙漏残差单元(**Hourglass Residual Units, HRUs**)代替残差单元：

![](https://pic.imgdb.cn/item/649b963a1ddac507cc2bafc7.jpg)

```python
class BnReluConv(nn.Module):
	def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
		super(BnReluConv, self).__init__()
		self.bn = nn.BatchNorm2d(inChannels)
		self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		return x

class BnReluPoolConv(nn.Module):
		def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
			super(BnReluPoolConv, self).__init__()
			self.bn = nn.BatchNorm2d(inChannels)
			self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding)
			self.relu = nn.ReLU()

		def forward(self, x):
			x = self.bn(x)
			x = self.relu(x)
			x = F.max_pool2d(x, kernel_size=2, stride=2)
			x = self.conv(x)
			return x

class ConvBlock(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(ConvBlock, self).__init__()
		self.brc1 = BnReluConv(inChannels, outChannels//2, 1, 1, 0)
		self.brc2 = BnReluConv(outChannels//2, outChannels//2, 3, 1, 1)
		self.brc3 = BnReluConv(outChannels//2, outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.brc1(x)
		x = self.brc2(x)
		x = self.brc3(x)
		return x

class PoolConvBlock(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(PoolConvBlock, self).__init__()
		self.brpc = BnReluPoolConv(inChannels, outChannels, 3, 1, 1)
		self.brc = BnReluConv(outChannels, outChannels, 3, 1, 1)

	def forward(self, x):
		x = self.brpc(x)
		x = self.brc(x)
		x = F.interpolate(x, scale_factor=2)
		return x

class SkipLayer(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(SkipLayer, self).__init__()
		if (inChannels == outChannels):
			self.conv = None
		else:
			self.conv = nn.Conv2d(inChannels, outChannels, 1)

	def forward(self, x):
		if self.conv is not None:
			x = self.conv(x)
		return x

class HourGlassResidual(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(HourGlassResidual, self).__init__()
		self.cb = ConvBlock(inChannels, outChannels)
		self.pcb = PoolConvBlock(inChannels, outChannels)
		self.skip = SkipLayer(inChannels, outChannels)

	def forward(self, x):
		out = self.cb(x)
		out = out + self.pcb(x)
		out = out + self.skip(x)
		return out
```

## 2. Hierarchical Attention Mechanism

不同的 **stack** 具有不同的语义：底层 **stacks** 对应局部特征，高层 **stacks** 对应全局特征。因此不同 **stacks** 生成的注意力图编码着不同的语义.

底层**stacks (stack1 - stack4)**采用多分辨率注意力(**Multi-Resolution Attention**)来对整体人体进行编码。

![](https://pic.imgdb.cn/item/649b9a731ddac507cc327f2f.jpg)

高层**stacks (stack5 - stack8)**设计分层的 **coarse-to-fine** 注意力机制对局部关节点进行缩放处理。

![](https://pic.imgdb.cn/item/649b9b5b1ddac507cc3400ec.jpg)

