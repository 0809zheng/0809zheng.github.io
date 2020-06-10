---
layout: post
title: '卷积神经网络的结构发展'
date: 2020-06-03
author: 郑之杰
cover: ''
tags: 深度学习
---

> Architectures about Convolutional Neural Networks.

卷积神经网络的结构发展概述，主要包括：
1. 早期探索：Hubel实验、LeNet、AlexNet、ZFNet、VGGNet
2. 深度化：ResNet、DenseNet
3. 模块化：GoogLeNet、Inceptionv3和Inception-ResNet、ResNeXt、Xception
4. 注意力：SENet、scSE、CBAM
5. 高效化：SqueezeNet、MobileNet、ShuffleNet、GhostNet
6. 自动化：NASNet、EfficientNet

下面的论文主要按照发表在**arXiv**上的时间顺序。

# Survey

### A Survey of the Recent Architectures of Deep Convolutional Neural Networks
- arXiv:[https://arxiv.org/abs/1901.06032](https://arxiv.org/abs/1901.06032)


# CNN History

### Receptive fields, binocular interaction and functional architecture in the cat's visual cortex
- intro:Hubel,Wiesel
- pdf:[http://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/additional/systems/JPhysiol-1962-Hubel-106-54.pdf](http://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/additional/systems/JPhysiol-1962-Hubel-106-54.pdf)

卷积神经网络的结构设计灵感来自Hubel和Wiesel的工作，因此在很大程度上遵循了灵长类动物视觉皮层的基本结构。卷积神经网络的学习过程与灵长类动物的视觉皮层腹侧通路（V1-V2-V4-IT/VTC）非常相似。灵长类动物的视觉皮层首先从视网膜位区域接收输入，在该区域通过外侧膝状核执行多尺度高通滤波和对比度归一化。然后通过分类为V1，V2，V3和V4的视觉皮层的不同区域执行检测。实际上，视觉皮层的V1和V2部分类似于卷积层和下采样层，而颞下区类似于更深的层，最终对图像进行推断。

### Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Visual Pattern Recognition
- intro:neocognitron
- pdf:[https://www.cs.princeton.edu/courses/archive/spring08/cos598B/Readings/Fukushima1980.pdf](https://www.cs.princeton.edu/courses/archive/spring08/cos598B/Readings/Fukushima1980.pdf)


# CNN Architectures

### Gradient-based learning applied to document recognition
- intro:LeNet5
- pdf:[http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

![](https://pic.downk.cc/item/5e82aee6504f4bcb04cdd1bc.png)

LeNet由LeCun在1998年提出，是历史上最早提出的卷积网络之一，被用于手写字符的识别。该网络由五层交替的卷积和下采样层组成，后接全连接层，网络参数并不庞大，大约有6万个参数。它具有对数字进行分类的能力，而不会受到较小的失真、旋转以及位置和比例变化的影响。当时GPU未广泛用于加速训练，传统多层全连接神经网络计算负担大，卷积神经网络利用了图像的潜在基础，即相邻像素彼此相关，不仅减少了参数数量和计算量，而且能够自动学习特征。在原始的LeNet模型中，使用了带参数的下采样层，模型最后还使用了高斯径向基连接，这些方法在今天已经很少再使用了，但这个网络奠定了卷积网络“卷积层-下采样层-全连接层”的拓扑结构，产生了深远的影响。

### ImageNet Classification with Deep Convolutional Neural Networks
- intro:AlexNet
- keypoint:local response normalization、ReLU、dropout、group convolution
- pdf:[http://stanford.edu/class/cs231m/references/alexnet.pdf](http://stanford.edu/class/cs231m/references/alexnet.pdf)

![](https://pic.downk.cc/item/5e82af1b504f4bcb04cdfba2.png)

AlexNet被认为是第一个深层卷积神经网络，设计了一个八层的网络结构，大约有6000万个参数。网络使用了Dropout帮助训练，还使用了ReLU作为激活函数提高收敛速度。网络也使用了最大池化和局部响应归一化。这个网络获得了2012年ImageNet图像分类竞赛的冠军，之后的网络大多在该网络的基础上进行演变。值得一提的是，受到当时条件的限制，网络是在两块GPU上进行并行训练的，只在部分层有GPU之间的交互。这种方法后来被研究者进一步的研究，演变成如今的“组卷积”的概念。

### Visualizing and Understanding Convolutional Networks
- intro:ZFNet
- keypoint:deconvolution、unpooling
- arXiv:[https://arxiv.org/abs/1311.2901](https://arxiv.org/abs/1311.2901)

![](https://pic.downk.cc/item/5e82af70504f4bcb04ce4445.png)

早期网络超参数的学习机制主要是基于反复试验，而不知道改进背后的确切原因，缺乏了解限制了网络在复杂图像上的性能。2013年，Zeiler和Fergus提出了ZFNet，设计该网络的初衷是定量可视化网络性能。网络活动可视化的想法是通过解释神经元的激活来监视网络性能。通过实验发现，减小卷积核的尺寸和步幅能够保留更多的特征，从而最大限度地提高学习能力。拓扑结构的重新调整带来了性能提高，表明特征可视化可用于识别设计缺陷并及时调整参数。

### Network In Network
- intro:NiN
- keypoint:1×1 conv、global average pooling
- arXiv:[https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400)

![](https://pic.downk.cc/item/5e816d3b504f4bcb04f28f50.png)

### Very Deep Convolutional Networks for Large-Scale Image Recognition
- intro:VGG16、VGG19
- arXiv:[https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

![](https://static.oschina.net/uploads/space/2018/0314/022939_Pl12_876354.png)

VGGNet提出了一种简单有效的卷积神经网络结构设计原则，该模型用多层3x3卷积层代替了11x11和5x5卷积层，并通过实验证明，同时放置多层3x3滤波器可以达到大尺寸滤波器的效果，并且减少了参数的数量。该网络提出的使用小尺寸的卷积核并且增加网络深度的设计思想一直沿用至今。VGG的主要限制是计算成本高，比如VGG16这个16层的卷积网络，由于增加了层数，即使使用小尺寸的滤波器，也有约1.4亿个参数，仍承受着很高的计算负担。

### Going Deeper with Convolutions
- intro:GoogLeNet
- keypoint:inception、split-transform-merge
- arXiv:[https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)

![](https://pic.downk.cc/item/5e8170f0504f4bcb04f4f7b5.png)
[GoogLeNet结构](https://static.oschina.net/uploads/space/2018/0317/141544_FfKB_876354.jpg)

GoogleNet提出了网络模块在设计时的基本思想，即(split、transform、merge)的方法。该模型引入了如图所示的inception模块，首先把上一层的输出split拆分成4路，分别通过1x1、3x3、5x5的卷积操作和一个3x3的最大池化操作进行transform变换，再把转换后的特征merge连接起来作为该模块的输出。在模块中封装了不同大小的卷积核以捕获不同尺度的空间信息。在采用大尺寸卷积核之前，使用1x1卷积作为瓶颈层，从而减小了模块的参数量。

### Highway Networks
- intro:Highway Networks
- arXiv:[https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)

![](https://pic.downk.cc/item/5e817166504f4bcb04f546f3.png)

Srivastava等人提出了Highway Networks。Highway Networks通过门控机制引入新的跨层连接，利用深度来学习丰富的特征表示。实验表明，即使深度为900层，Highway Networks的收敛速度也比普通网络快得多。

### Rethinking the Inception Architecture for Computer Vision
- intro:Inception-V2、Inception-V3
- arXiv:[https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)

![](https://pic.downk.cc/item/5e818862504f4bcb040451fb.png)

后续的inceptionv2版本主要引入了batchnorm进行优化，在inceptionv3模块中，首先将一个5x5的卷积层转换为两层3×3的卷积层，两者具有相同的感受野但后者的参数量更少一些。进一步，该模块把nxn的卷积转换成nx1的卷积和1xn的卷积的串联或并联的形式。

### Deep Residual Learning for Image Recognition
- intro:ResNet
- keypoint:skip connection
- arXiv:[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

![](https://pic.downk.cc/item/5e8172ce504f4bcb04f63231.png)

由何凯明提出的ResNet通过引入残差学习使得训练深层的卷积网络成为可能。该网络设计了一种如图所示的残差连接，即在一些卷积层前后用一个恒等映射进行连接，这样主路径的卷积网络学习的是信号的残差，由于连接了这样一条恒等映射，使得在反向传播时梯度可以更容易的回传，从而减小了训练网络的困难。作者训练了超过100层的网络，并在分类挑战赛中取得了很好的成绩。

### Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
- intro:Inception-V4, Inception-ResNet
- arXiv:[https://arxiv.org/abs/1602.07261v1](https://arxiv.org/abs/1602.07261v1)

Inception-V4:
![](https://pic.downk.cc/item/5e82afcf504f4bcb04ce8d19.jpg)
![](https://pic.downk.cc/item/5e82afe2504f4bcb04ce99d5.jpg)
![](https://pic.downk.cc/item/5e82aff4504f4bcb04cea61f.jpg)

Inception-ResNet:
![](https://pic.downk.cc/item/5e82b009504f4bcb04ceb33c.jpg)
![](https://pic.downk.cc/item/5e82b01a504f4bcb04cebd2f.png)

Inception-ResNet则是在Inception模块中引入了残差连接。

### SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
- intro:SqueezeNet
- keypoint:fire module
- arXiv:[https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)

![](https://pic.downk.cc/item/5e82b031504f4bcb04cecd60.png)
![](https://pic.downk.cc/item/5e82b044504f4bcb04ced960.png)

SqueezeNet设计了Fire模块，这个模块包括squeeze和expand操作。squeeze 操作使用一系列1x1卷积，用较小的参数量对特征进行压缩，expand 操作用一系列1x1卷积和3x3卷积进行特征扩容。通过堆叠该模块，网络实现了AlexNet的精度，但参数量是后者的1/510。

### Stacked Hourglass Networks for Human Pose Estimation
- intro:Hourglass Network
- arXiv:[https://arxiv.org/abs/1603.06937](https://arxiv.org/abs/1603.06937)

![](https://pic.downk.cc/item/5e82b057504f4bcb04cee503.png)

### Deep Networks with Stochastic Depth
- intro:Stochastic Depth
- arXiv:[https://arxiv.org/abs/1603.09382](https://arxiv.org/abs/1603.09382)

![](https://pic.downk.cc/item/5e8175b9504f4bcb04f82c8d.png)


### Wide Residual Networks
- intro:WideResNet
- arXiv:[https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146)

![](https://pic.downk.cc/item/5e8175f2504f4bcb04f851cc.png)


### Densely Connected Convolutional Networks
- intro:DenseNet
- arXiv:[https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)

![](https://pic.downk.cc/item/5e82b06c504f4bcb04cef7ae.png)

ResNet的问题在于训练了一个深层的网络，但是网络中许多层可能贡献很少或根本没有信息。为了解决此问题，DenseNet使用了跨层连接，以前馈的方式将每一层连接到更深的层中，假设网络有l层，建立了(l(l+1))/2个直接连接，并且DenseNet对先前层的特征使用了级联而不是相加，可以显式区分网络不同深度的信息。

### Xception: Deep learning with depthwise separable convolutions
- intro:Xception
- keypoint:Depthwise Separable Convolution(depthwise + pointwise)
- arXiv:[https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)

![](https://pic.downk.cc/item/5e82b07f504f4bcb04cf06e0.png)

Xception可以被认为是一种极端的Inception结构。它首先使用1x1卷积解耦空间和特征相关性，降低通道数提高网络的计算效率，然后对特征进行k次空间变换，这里的k就是之前定义的cardinality。Xception通过在空间轴上分别对每个特征图进行卷积，使计算变得容易，然后通过1x1卷积进行跨通道关联，调节特征图深度。

### Deep Pyramidal Residual Networks
- intro:Pyramidal ResNet
- arXiv:[https://arxiv.org/abs/1610.02915](https://arxiv.org/abs/1610.02915)

![](https://pic.downk.cc/item/5e818246504f4bcb04008f58.png)


### Aggregated Residual Transformations for Deep Neural Networks
- intro:ResNeXt
- keypoint:cardinality
- arXiv:[https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)

![](https://pic.downk.cc/item/5e81810c504f4bcb04ffcb7c.png)

ResNeXt是对Inception模块的进一步改进。ResNeXt引入了一个概念，叫做cardinality，cardinality是一个附加的维度，用来表示对特征进行拆分和变换时的路径数目。比如图中把输入拆分成了32条路径，这里的cardinality就是32。实验表明，增加cardinality能够显着改善了性能，尽管会带来一些运算量的增加。


### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- intro:MobileNet
- keypoint:Depthwise Separable Convolution(pointwise + depthwise)
- arXiv:[https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)

![](https://pic.downk.cc/item/5e8183cb504f4bcb04016f4b.png)

MobileNet引入了深度可分离卷积，深度可分离卷积将传统的卷积操作分解为通道卷积和空间卷积，通道卷积是指对输入特征的每一个通道分别用一个卷积核进行逐通道地运算，而空间卷积是指使用多个1x1的卷积进行逐像素地运算，这种方法相对于传统卷积极大地减少了参数量。


### Residual Attention Network for Image Classification
- intro:Residual Attention Network
- keypoint:trunk brunch + mask brunch
- arXiv:[https://arxiv.org/abs/1704.06904](https://arxiv.org/abs/1704.06904)

![](https://pic.downk.cc/item/5e82b31f504f4bcb04d14747.png)


### ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
- intro:ShuffleNet
- keypoint:group convolution + channel shuffle
- arXiv:[https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)

![](https://pic.downk.cc/item/5e82b098504f4bcb04cf1784.png)
![](https://pic.downk.cc/item/5e82b0ab504f4bcb04cf26dd.png)

ShuffleNet引入了组卷积的概念，如图a是普通的组卷积，即将图像按特征通道进行分组，卷积操作只在每个组内进行。ShuffleNet在组卷积的同时还引入了shuffle操作。即进行一定的组卷积之后将特征的通道打乱，实现组间交互，这样压缩了模型的参数量，一定程度上保证了信息的流动。

### Learning Transferable Architectures for Scalable Image Recognition
- intro:NASNet
- keypoint:neural architecture search
- arXiv:[https://arxiv.org/abs/1707.07012](https://arxiv.org/abs/1707.07012)

![](https://pic.downk.cc/item/5ee0c910c2a9a83be5bd3eaa.jpg)

直接在大型数据集上学习网络结构的计算成本过高，NASNet通过强化学习的方法先在小数据上学习网络模块，通过堆叠这些网络模块将网络迁移到更复杂的数据集上。具体地，NASNet学习两种模块，一种叫做Normal模块，这个模块的输入和输出尺寸是保持一致的，另一种叫做Reduction模块，这个模块通过下采样把特征映射的尺寸减小为一半。网络在搜索，预先设定了一些基本操作，如不同尺寸的卷积、池化、空洞卷积和深度可分离卷积，通过网络自动组合这些操作寻找最优的拓扑结构。值得一提的是，这样的搜索方法成本是相当高的，NASNet在搜索时使用了500块GPU。

### Squeeze-and-Excitation Networks
- intro:SENet
- arXiv:[https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)

![](https://pic.downk.cc/item/5e82b533504f4bcb04d2fbf5.png)
![](https://pic.downk.cc/item/5e82b54c504f4bcb04d30d40.png)

SENet是最早引入注意力机制的网络之一，它在网络中引入了一条通道选择路径，该路径包括两个操作：squeeze挤压和excitation激发。挤压模块把特征映射的每一个通道挤压成一个标量，生成特征的通道统计信息；激发模块通过一个变换根据通道的统计信息重新调整每一个通道的权重，并根据权重调整原始的特征映射的每一个通道。该模块可以应用于任何卷积网络中，是一种通道的注意力机制。


### MobileNetV2: Inverted Residuals and Linear Bottlenecks
- intro:MobileNetV2
- arXiv:[https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

![](https://pic.downk.cc/item/5e85a86b504f4bcb04e92abd.jpg)


### Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks
- intro:cSE(即SENet)、sSE、scSE
- arXiv:[https://arxiv.org/abs/1803.02579](https://arxiv.org/abs/1803.02579)

![](https://pic.downk.cc/item/5e82b6f7504f4bcb04d478f4.png)

注意力机制不仅可以应用于通道，也可以应用于空间中，Roy等人提出了三个不同的注意力模块：cSE、sSE、scSE。图b中的cSE模块引入了通道注意力，模型结构和之前的SENet是相同的。图c中的sSE模块是把特征映射的每一个像素位置挤压成一个标量，生成特征的空间统计信息，并进一步通过一个变换根据空间的统计信息重新调整每一个像素位置的权重，并根据权重调整原始的特征映射的每一个空间像素。图d中的scSE模块则是结合了通道和空间注意力，采用并联的方式生成特征。图a表示可以将这些注意力机制应用于任何一个卷积网络中。


### A New Channel Boosted Convolutional Neural Network using Transfer Learning
- intro:Channel Boosted CNN
- arXiv:[https://arxiv.org/abs/1804.08528v4](https://arxiv.org/abs/1804.08528v4)

![]()


### CBAM: Convolutional Block Attention Module
- intro:CBAM
- arXiv:[https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)

![](https://pic.downk.cc/item/5e82b8e2504f4bcb04d628c8.png)

卷积块注意力模型（CBAM）和scSE模型类似，也使用了通道和空间注意力，不同的是这里是将通道注意力和空间注意力进行串行使用，能够减少网络的参数和计算成本。

### Competitive Inner-Imaging Squeeze and Excitation for Residual Network
- intro:Competitive SENet
- arXiv:[https://arxiv.org/abs/1807.08920v4](https://arxiv.org/abs/1807.08920v4)

![](https://pic.downk.cc/item/5e82bbfd504f4bcb04d8bf5a.png)
![](https://pic.downk.cc/item/5e82bc12504f4bcb04d8d3ee.png)


### ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
- intro:ShuffleNet V2
- keypoint:channel split
- arXiv:[https://arxiv.org/abs/1807.11164v1](https://arxiv.org/abs/1807.11164v1)

![](https://pic.downk.cc/item/5e85a622504f4bcb04e6f26d.jpg)


### Selective Kernel Networks
- intro:SKNet
- arXiv:[https://arxiv.org/abs/1903.06586](https://arxiv.org/abs/1903.06586)

![](https://pic.downk.cc/item/5e82bc3b504f4bcb04d8ffa8.png)


### ANTNets: Mobile Convolutional Neural Networks for Resource Efficient Image Classification
- intro:ANTNet
- arXiv:[https://arxiv.org/abs/1904.03775](https://arxiv.org/abs/1904.03775)

![]()


### Searching for MobileNetV3
- intro:MobileNetV3
- arXiv:[https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)

![]()


### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- intro:EfficientNet
- keypoint:Compound Model Scaling
- arXiv:[https://arxiv.org/abs/1905.11946?context=stat.ML](https://arxiv.org/abs/1905.11946?context=stat.ML)

![](https://pic.downk.cc/item/5e81b404504f4bcb04225c0e.png)

EfficientNet提出了一种兼顾速度与精度的多维度混合的模型放缩方法。它首先设计了一个基准模型，如图a所示，缩放模型的几个维度：包括网络深度（指的是网络的层数）、网络宽度（指的是网络每一层的通道数）、特征分辨率（指的是网络每一层特征的尺寸大小），通过对这三个维度固定的比例进行搜索，得到一个最佳的网络结构。


### GhostNet: More Features from Cheap Operations
- intro:GhostNet
- arXiv:[https://arxiv.org/abs/1911.11907](https://arxiv.org/abs/1911.11907)

![](https://pic.downk.cc/item/5e82be97504f4bcb04db1b4f.png)

GhostNet通过实验发现，卷积得到的特征映射之间是相似的，可以用线性变换近似。网络使用较少的卷积操作获得一些特征图像，然后对这些特征图像做线性变换映射成新的特征图像，这种操作在计算上是廉价的，通过实验取得了很好的效果。

