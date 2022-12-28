---
layout: post
title: '图像识别(Image Recognition)'
date: 2020-05-06
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eba62edc2a9a83be59d07b6.jpg'
tags: 深度学习
---

> Image Recognition.

**图像识别**是**计算机视觉**的基本任务，旨在对每张图像内出现的物体进行类别区分。图像识别任务面临的主要问题是**语义鸿沟(semantic gap)**，表现在：
- 不同图像的低级视觉特征(**low-level visual features**)相似，但高级语义(**high-level concepts**)差别很大；
- 不同图像的低级视觉特征差距很大，但高级语义相同。

传统的图像处理是先对图像手工提取特征，再根据特征对图像进行分类。而深度学习方法不需要手工提取特征，使用卷积神经网络自动提取特征并进行分类。

**本文目录**：
1. 图像识别模型
2. 图像识别基准

# 1. 图像识别模型

本节主要介绍应用于图像识别任务的**卷积神经网络**，按照其结构发展概述如下：
1. 早期探索：奠定“卷积层-下采样层-全连接层”的拓扑结构。如**LeNet5**, **AlexNet**, **ZFNet**, **NIN**, **SPP-net**, **VGGNet**
2. 深度化：增加堆叠卷积层的数量。如**Highway Network**, **ResNet**, **Stochastic Depth**, **DenseNet**, **Pyramidal ResNet**
3. 模块化：设计用于堆叠的网络模块。如**Inception v1-4**, **WideResNet**, **Xception**, **ResNeXt**, **NASNet**, **ResNeSt**, **ConvNeXt**
4. 轻量化：设计轻量级卷积层，可参考[<font color=Blue>轻量级卷积神经网络</font>](https://0809zheng.github.io/2021/09/10/lightweight.html)。
5. 其他结构：**Noisy Student**, **SCAN**, **NFNet**, **ResNet-RS**

## (1) 早期探索

卷积神经网络结构设计的早期探索过程，奠定了“卷积层-下采样层-全连接层”的拓扑结构，并通过实现和理论分析总结了使用小尺寸的卷积核(如$3 \times 3$)并且增加网络深度的设计思想。

### ⚪ LeNet5
- paper：[Gradient-based learning applied to document recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

**LeNet**由**LeCun**在**1998**年提出，是历史上最早提出的卷积网络之一，被用于手写字符的识别。该网络由五层交替的卷积和下采样层组成，后接全连接层，网络参数并不庞大，大约有**6**万个参数。它具有对数字进行分类的能力，而不会受到较小的失真、旋转以及位置和比例变化的影响。当时**GPU**未广泛用于加速训练，传统多层全连接神经网络计算负担大，卷积神经网络利用了图像的潜在基础，即相邻像素彼此相关，不仅减少了参数数量和计算量，而且能够自动学习特征。在原始的LeNet模型中，使用了带参数的下采样层，模型最后还使用了高斯径向基连接，这些方法在今天已经很少再使用了，但这个网络奠定了卷积网络“卷积层-下采样层-全连接层”的拓扑结构，产生了深远的影响。

![](https://pic.imgdb.cn/item/63a6b78308b683016336776f.jpg)


### ⚪ AlexNet
- paper：[ImageNet Classification with Deep Convolutional Neural Networks](http://stanford.edu/class/cs231m/references/alexnet.pdf)

**AlexNet**被认为是第一个深层卷积神经网络，设计了一个八层的网络结构，大约有**6000**万个参数。网络使用了**Dropout**帮助训练，还使用了**ReLU**作为激活函数提高收敛速度。网络也使用了最大池化和局部响应归一化。这个网络获得了**2012**年**ImageNet**图像分类竞赛的冠军，之后的网络大多在该网络的基础上进行演变。值得一提的是，受到当时条件的限制，网络是在两块**GPU**上进行并行训练的，只在部分层有**GPU**之间的交互。这种方法后来被研究者进一步研究，演变成如今的“组卷积”的概念。

![](https://pic.imgdb.cn/item/63a6b7a808b683016336b4c2.jpg)

### ⚪ ZFNet
- paper：[Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

早期网络超参数的学习机制主要是基于反复试验，而不知道改进背后的确切原因，缺乏了解限制了网络在复杂图像上的性能。**2013**年，**Zeiler**和**Fergus**提出了**ZFNet**，设计该网络的初衷是定量可视化网络性能。网络激活可视化的想法是通过解释神经元的激活来监视网络性能。通过实验发现，减小卷积核的尺寸和步幅能够保留更多的特征，从而最大限度地提高学习能力。拓扑结构的重新调整带来了性能提高，表明特征可视化可用于识别设计缺陷并及时调整参数。

![](https://pic.imgdb.cn/item/63a6b7ce08b683016336ff18.jpg)

### ⚪ NIN
- paper：[Network In Network](https://arxiv.org/abs/1312.4400)

**NIN**提出用非线性函数($1\times 1$卷积)增强卷积层提取的局部特征，同时使用全局平均池化层替代全连接层作为网络尾部的分类部分。

![](https://pic.imgdb.cn/item/63a6b87c08b68301633819d3.jpg)

### ⚪ SPP-net
- paper：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)

通常的卷积神经网络包含卷积层和全连接层，后者要求输入固定数量的特征数，因此在网络输入时会把图像通过裁剪和拉伸调整为固定的大小，然而这样会改变图像的尺寸和纵横比，并扭曲原始信息。**SPP-net**通过在卷积层和全连接层中间引入空间金字塔池化结构，能够把任意不同尺寸和不同纵横比的图像特征转换为固定尺寸大小的输出特征向量。

![](https://pic.imgdb.cn/item/63abf68f08b6830163947507.jpg)

### ⚪ VGGNet
- paper：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

**VGGNet**提出了一种简单有效的卷积神经网络结构设计原则，该模型用多层**3x3**卷积层代替了**11x11**和**5x5**卷积层，并通过实验证明，同时放置多层**3x3**卷积可以达到大尺寸卷积的效果，并且减少了参数的数量。该网络提出的使用小尺寸的卷积核并且增加网络深度的设计思想一直沿用至今。**VGGNet**的主要限制是计算成本高，比如**VGGNet**这个**16**层的卷积网络，由于增加了层数，即使使用小尺寸的滤波器，也有约**1.4**亿个参数，仍承受着很高的计算负担。

![](https://pic.imgdb.cn/item/63a6ba2508b68301633a9cd1.jpg)

## (2) 深度化设计

深度化设计的思路是通过增加堆叠卷积层的数量来增强模型的非线性表示能力。

### ⚪ Highway Network
- paper：[Highway Networks](https://arxiv.org/abs/1505.00387)

**Highway Network**通过门控机制引入新的跨层连接，利用深度来学习丰富的特征表示。实验表明，即使深度为**900**层，**Highway Network**的收敛速度也比普通网络快得多。

$$ y=H(x) \cdot T(x) + x \cdot(1-T(x)) $$

![](https://pic.imgdb.cn/item/63a6bd8208b6830163401304.jpg)

### ⚪ ResNet
- paper：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

**ResNet**通过引入残差学习使得训练深层的卷积网络成为可能。该网络设计了一种如图所示的残差连接，即在一些卷积层前后用一个恒等映射进行连接，这样主路径的卷积网络学习的是信号的残差，由于连接了这样一条恒等映射，使得在反向传播时梯度可以更容易的回传，从而减小了训练网络的困难。作者训练了超过**100**层的网络，并在分类挑战赛中取得了很好的成绩。

![](https://pic.imgdb.cn/item/63a6be8a08b683016341bbc2.jpg)

### ⚪ Stochastic Depth
- paper：[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

随机深度是指在训练时以一定概率丢弃网络中的模块（等价于恒等变换）；测试时使用完整的网络，并且按照丢弃概率对各个模块的特征进行加权。

![](https://pic.imgdb.cn/item/63a6bfa508b683016343b891.jpg)

### ⚪ DenseNet
- paper：[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

**ResNet**的问题在于训练了一个深层的网络，但是网络中许多层可能贡献很少或根本没有信息。为了解决此问题，**DenseNet**使用了跨层连接，以前馈的方式将每一层连接到更深的层中，假设网络有$L$层，**DenseNet**建立了$(L(L+1))/2$个直接连接，并且对先前层的特征使用了级联而不是相加，可以显式区分网络不同深度的信息。

![](https://pic.imgdb.cn/item/63a6c19a08b683016347250c.jpg)

### ⚪ Pyramidal ResNet
- paper：[Deep Pyramidal Residual Networks](https://arxiv.org/abs/1610.02915)

**Pyramidal ResNet**也是一种残差网络结构，在整个网络结构中逐渐增加特征的通道数

![](https://pic.imgdb.cn/item/63a6c55608b68301634d0bbc.jpg)




## (3) 模块化设计

模块化设计的思想是首先设计包含卷积层的网络模块，然后通过堆叠相同的模块构造深度网络。

### ⚪ Inception (GoogLeNet)
- paper：[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

**GoogLeNet**提出了网络模块在设计时的基本思想，即**拆分-变换-融合(split、transform、merge)**的方法。该模型引入了如图所示的**inception**模块，首先把上一层的输出拆分成**4**路，分别通过**1x1**、**3x3**、**5x5**的卷积操作和一个**3x3**的最大池化操作进行变换，再把转换后的特征连接起来作为该模块的输出。在模块中封装了不同大小的卷积核以捕获不同尺度的空间信息。在采用大尺寸卷积核之前，使用**1x1**卷积作为瓶颈层，从而减小了模块的参数量。

![](https://pic.imgdb.cn/item/63a6bca208b68301633ec290.jpg)

### ⚪ Inception-V2, Inception-V3
- paper：[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

后续的**inceptionv2**版本主要引入了**batchnorm**进行优化。在**inceptionv3**模块中，首先将一个**5x5**的卷积层转换为两层**3×3**的卷积层，两者具有相同的感受野但后者的参数量更少一些；进一步，该模块把**nxn**的卷积转换成**nx1**的卷积和**1xn**的卷积的串联或并联的形式。

![](https://pic.downk.cc/item/5e818862504f4bcb040451fb.png)

### ⚪ Inception-V4, Inception-ResNet
- paper：[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261v1)

**Inception-ResNet**是在**Inception**模块中引入了残差连接。

![](https://pic.downk.cc/item/5e82b01a504f4bcb04cebd2f.png)

### ⚪ WideResNet
- paper：[Wide Residual Networks](https://arxiv.org/abs/1605.07146)

**WideResNet**通过增大网络结构的宽度(特征通道数)来改善网络的性能，并在卷积层之间引入了**dropout**进行正则化；实现了使用浅层网络获得跟深层网络相当的准确度。

![](https://pic.imgdb.cn/item/63a6c14008b6830163466b0a.jpg)

### ⚪ Xception
- paper：[Xception: Deep learning with depthwise separable convolutions](https://arxiv.org/abs/1610.02357)

**Xception**可以被认为是一种极端(**eXtreme**)的**Inception**结构。它设计了深度可分离卷积，即首先使用**1x1**卷积降低通道数提高网络的计算效率，然后分别对特征的每个通道进行$3 \times 3$卷积。

![](https://pic.imgdb.cn/item/63a6c2f508b683016349403e.jpg)

### ⚪ ResNeXt
- paper：[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

**ResNeXt**引入了一个概念**cardinality**，是指一个附加的维度，用来表示对特征进行拆分和变换时的路径数目。比如图中把输入特征拆分成了**32**条路径，这里的**cardinality**就是**32**。实验表明增加**cardinality**能够显着改善性能，尽管会带来一些运算量的增加。

![](https://pic.imgdb.cn/item/63a6c5e508b68301634dd2be.jpg)

### ⚪ NASNet
- paper：[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

直接在大型数据集上学习网络结构的计算成本过高，**NASNet**通过神经结构搜索的方法在小数据上学习网络模块。网络在搜索时预先设定了一些基本操作，如不同尺寸的卷积、池化、空洞卷积和深度可分离卷积，通过网络自动组合这些操作寻找最优的拓扑结构。值得一提的是，这样的搜索方法成本是相当高的，**NASNet**在搜索时使用了**500**块**GPU**。

![](https://pic.imgdb.cn/item/63a6d1cf08b68301636105e0.jpg)

### ⚪ ResNeSt

- paper：[<font color=blue>ResNeSt: Split-Attention Networks</font>](https://0809zheng.github.io/2020/09/09/resnest.html)

**ResNeSt**结合了多路径设计和通道注意力机制。多路径(**multi-path**)设计参考了**ResNeXt**，引入超参数**cardinality**控制分组卷积的分支数$k$；通道注意力机制参考了**SKNet**，引入超参数**radix**控制注意力计算的分支数$r$。

![](https://pic.downk.cc/item/5fec3e5a3ffa7d37b366a46a.jpg)

### ⚪ ConvNeXt

- paper：[<font color=blue>A ConvNet for the 2020s</font>](https://0809zheng.github.io/2020/09/09/resnest.html)

**ConvNeXt**通过把标准**ResNet**逐步修改为**Swin Transformer**，在此过程中发现了导致卷积神经网络和视觉**Transformer**存在性能差异的几个关键组件，在此基础上设计的卷积模块结合训练技巧与微观设计，实现了性能最佳的卷积网络。

![](https://pic.imgdb.cn/item/63ac435f08b68301632949ce.jpg)

## (4) 轻量化设计

**轻量级**网络设计旨在设计计算复杂度更低的卷积网络结构，更多细节可参考[<font color=Blue>轻量级(LightWeight)卷积神经网络</font>](https://0809zheng.github.io/2021/09/10/lightweight.html)。
- 从**结构**的角度考虑，卷积层提取的特征存在冗余，可以设计特殊的卷积操作，减少卷积操作的冗余，从而减少计算量。如**SqueezeNet**, **SqueezeNext**, **MobileNet V1,2,3**, **ShuffleNet V1,2**, **IGCNet V1,2**, **ChannelNet**, **EfficientNet V1,2**, **GhostNet**, **MicroNet**, **CompConv**。
- 从**计算**的角度，模型推理过程中存在大量乘法运算，而乘法操作(相比于加法)对于目前的硬件设备不友好，可以对乘法运算进行优化，也可以减少计算量。如**AdderNet**使用**L1**距离代替卷积乘法；使用**Mitchell**近似代替卷积乘法。

## (5) 其他结构

### ⚪ Noisy Student
- paper：[<font color=blue>Self-training with Noisy Student improves ImageNet classification</font>](https://0809zheng.github.io/2020/08/07/noisy-student-training.html)

**Noisy Student**是一种半监督的图像分类方法。首先使用标记数据集训练一个教师网络；其次使用训练好的教师网络对大量无标签数据进行分类，构造伪标签；然后训练一个模型容量相等或更大的学生网络同时学习原标记数据和伪标签数据，在学习过程中引入数据增强、**Dropout**和随机深度等噪声干扰；最后将训练好的学生网络作为新的教师网络，并重复上述过程。

![](https://pic.imgdb.cn/item/63ac21ab08b6830163f13903.jpg)

### ⚪ Semantic Clustering by Adopting Nearest neighbors (SCAN)
- paper：[<font color=blue>SCAN: Learning to Classify Images without Labels</font>](https://0809zheng.github.io/2020/07/15/scan.html)

**SCAN**是一种无监督的图像分类方法，包括特征学习和聚类两个步骤。首先通过特征学习从图像中提取特征，学习过程采用自监督表示学习方法；然后对每一张图像的特征，在特征空间中寻找最近邻的$k$个特征，通过调整网络使得图像特征与这最近邻的$k$个特征内积最大（即相似度最高）。同时通过聚类给图像分配一个伪标签，通过调整网络最大化图像特征属于该类别的概率。

### ⚪ Normalizer-Free ResNet (NFNet)
- paper：[<font color=blue>High-Performance Large-Scale Image Recognition Without Normalization</font>](https://0809zheng.github.io/2021/04/21/nfnet.html)

**NFNet**是一个不使用**BatchNorm**的大批量图像分类网络，通过引入自适应梯度裁剪来保证较大**batch size**下训练的稳定性：

$$ \begin{aligned} G \leftarrow \begin{cases} \lambda\frac{||W||}{||G||}G, & ||G|| \geq \lambda \\ G, & \text{otherwise} \end{cases} \end{aligned} $$

### ⚪ Revisiting ResNet (ResNet-RS)
- paper：[<font color=blue>Revisiting ResNets: Improved Training and Scaling Strategies</font>](https://0809zheng.github.io/2021/03/18/resnetrs.html)

**ResNet-RS**通过改进**ResNet**的网络结构，引入新的训练和正则化方法，并调整缩放策略，使得经典**ResNet**重新成为分类**SOTA**模型。作者提出的推荐缩放策略为在可能出现过拟合问题的大训练轮数下首选扩大深度，否则扩大宽度；并且应缓慢扩大分辨率。







# 2. 图像识别基准

常见的图像识别基准有**MNIST**, **CIFAR**, **Places2**, **Cats vs Dogs**, **ImageNet**, **PASCAL VOC**.

### (1) MNIST（Mixed National Institute of Standards and Technology）
[MNIST](http://yann.lecun.com/exdb/mnist/)是由纽约大学的**Yann LeCun**整理的手写数字识别数据集。其训练集包含$60000$张图像，测试集包含$10000$张图像，每张图像都进行了尺度归一化和数字居中处理，固定尺寸大小为$28×28$。

![](https://pic.imgdb.cn/item/63a6d2f308b683016362ab7f.jpg)

### (2) CIFAR（Canada Institute For Advanced Research）
[CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)是由加拿大先进技术研究院的**AlexKrizhevsky**, **Vinod Nair**和**Geoffrey Hinton**收集而成的小图片数据集，包含**CIFAR-10**和**CIFAR-100**两个数据集。
- **CIFAR-10**：由$60000$张$32*32$的$RGB$彩色图片构成，共$10$个分类。包含$50000$张训练图像，$10000$张测试图像。![](https://pic.imgdb.cn/item/63a6d3c708b683016363d0f6.jpg)
- **CIFAR-100**：由$60000$张图像构成，包含$100$个类别，每个类别$600$张图像，其中$500$张用于训练，$100$张用于测试。其中这$100$个类别又组成了$20$个大的类别，每个图像包含小类别和大类别两个标签。![](https://pic.imgdb.cn/item/63a6d3e608b683016363f89d.jpg)

### (3) Places2

[Places2](http://places2.csail.mit.edu/index.html)是由**MIT**开发的一个场景图像数据集，可用于以场景和环境为应用内容的视觉认知任务。包含一千万张图片，**400**多个不同类型的场景环境，每一类有**5000-30000**张图片。

![](https://pic.imgdb.cn/item/63a6d42508b6830163644969.jpg)

### (4) Cats vs Dogs
[Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)是**kaggle**上的猫狗分类数据集，共**25000**张图片，猫、狗各**12500**张。

### (5) ImageNet
[ImageNet](http://www.image-net.org/)是斯坦福大学的计算机科学家李飞飞建立的目前世界上最大的图像识别数据库之一，目前已经包含**14197122**张图像，**21841**个类别。

基于**ImageNet**举办的**ILSVRC（ImageNet Large-Scale Visual Recognition Challenge）**比赛是图像识别领域最重要的赛事，催生出一系列著名的卷积神经网络。该比赛使用的数据集是**ImageNet**的一个子集，总共有$1000$类，每类大约有$1000$张图像。具体地，有大约$120$万张训练图像，$5$万张验证图像，$15$万张测试图像。

![](https://pic.downk.cc/item/5eba62edc2a9a83be59d07b6.jpg)

### (6) PASCAL VOC
[PASCAL VOC](http://pjreddie.com/projects/pascal-voc-dataset-mirror/)数据集是一个视觉对象的分类识别和检测的基准测试集，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。该数据集包含**VOC2007**（$430M$）、**VOC2012**（$1.9G$）两个下载版本。

