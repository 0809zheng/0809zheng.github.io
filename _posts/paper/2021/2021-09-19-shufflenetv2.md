---
layout: post
title: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'
date: 2021-09-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61712d8d2ab3f51d91a1c751.jpg'
tags: 论文阅读
---

> ShuffleNet V2: 高效卷积神经网络结构设计的实践准则.

- paper：ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
- arXiv：[link](https://arxiv.org/abs/1807.11164)

高效神经网络的结构设计质量主要是通过计算复杂度的间接度量作为评估指标，比如**FLOPs** (**float-point operations**, 浮点操作数量)。然而，其他因素如内存访问成本和硬件特性也会影响网络的速度。因此，作者尝试在不同硬件平台上评估高效网络的一些直接指标，并提出了一种新的网络结构，**ShuffleNet v2**。

![](https://pic.imgdb.cn/item/617140032ab3f51d91b5fcd6.jpg)

上图显示了在不同硬件平台上，具有相近**FLOPs**的不同网络具有不同的速度。因此使用**FLOPs**作为计算复杂度的度量是不准确的。造成**FLOPs**指标和速度指标之间的差异的主要原因如下：
1. **FLOPs**没有考虑到内存访问成本(**memory access cost, MAC**)，组卷积所需的**MAC**较高，可能会成为**GPU**运算的瓶颈；
2. 在**FLOPs**相同的情况下，并行度高的模型比低并行度的模型运算更快；
3. 硬件平台不同，具有相同**FLOPs**的操作可能有不同的运行时间。比如**CUDNN**库针对$3\times 3$卷积进行了优化，因此并不能说$3\times 3$卷积比$1\times 1$卷积慢$9$倍。

作者进一步分析了**ShuffleNet**和**MobileNet V2**两个网络中不同操作的运行时间占比。尽管**FLOPs**是由卷积操作提供的，但其他操作也占用了一部分时间，如读取数据、打乱和元素操作。

![](https://pic.imgdb.cn/item/6172148f2ab3f51d9173d803.jpg)

作者依此提出了四条高效网络设计准则：

### ① 输入输出通道数相等能够最小化内存访问成本

以$1\times 1$卷积为例。假设特征的空间尺寸为$h\times w$，输入通道数为$c_1$，输出通道数为$c_2$，则该卷积的**FLOPs**为$B=hwc_1c_2$。该卷积的内存访问成本为：

$$ MAC = hwc_1+hwc_2+c_1c_2 $$

这三项分别表示对输入特征、输出特征和卷积核权重的内存访问。由均值不等式：

$$ MAC \geq 2\sqrt{hwB}+\frac{B}{hw} $$

因此内存访问成本可以表示为**FLOPs**的下界，且当$c_1=c_2$时取下界。下表展示了不同通道数比例下的运行速度：

![](https://pic.imgdb.cn/item/617218ea2ab3f51d91768cb9.jpg)

### ② 过多的组卷积会增加内存访问成本

组卷积限制了卷积在不同的输入通道组内运算，能够降低**FLOPs**，但可能增加内存访问成本。

假设输入通道分成$g$组，则组卷积的**FLOPs**为$B=hwc_1c_2/g$。组卷积的内存访问成本为：

$$ MAC = hwc_1+hwc_2+\frac{c_1c_2}{g} \\ = hwc_1+\frac{Bg}{c_1}+\frac{B}{hw} $$

当固定输入特征的尺寸$hwc_1$以及计算**FLOPs** $B$时，分组数量增大，则内存访问成本增加。下表表明使用更大的分组会显著降低处理速度：

![](https://pic.imgdb.cn/item/61721b3d2ab3f51d9177e545.jpg)

### ③ 在模块中使用更多层会降低并行度

**GoogLeNet**系列和一些自动设计的网络模块中会使用许多小的操作而不是几个大的操作，比如一个**ResNet**模块中通常会使用$2$-$3$个卷积层，而**NASNet-A**的一个模块使用了$13$个卷积和池化层。虽然这种碎片化的操作能够提高精度，但会引入额外的开销(内核启动和同步等)。下表表明串行或并行的碎片化的操作都降低了处理速度：

![](https://pic.imgdb.cn/item/617220862ab3f51d917ba261.jpg)

### ④ 逐元素的操作是不可忽略的

一些逐元素的操作，如激活函数、张量相加和偏置相加等，在轻量型模型中是不可忽略的，尤其是在**GPU**上。下表表明，在移除**ReLU**和跳跃连接后，网络的处理速度得到提升：

![](https://pic.imgdb.cn/item/617221962ab3f51d917c38da.jpg)

基于上述原则，作者设计了**shufflenet v2**的基本模块，如下图cd所示。作为对比，**shufflenet**的基本模块如下图ab所示。

![](https://pic.imgdb.cn/item/617224102ab3f51d917de14f.jpg)

**shufflenet v2**的基本模块采用了一个主体分支的形式(遵循原则③)。作者没有使用组卷积(遵循原则②)，并且卷积的输入和输出特征保持相同(遵循原则①)。对于跳跃连接，作者没有使用相加操作，而是使用了连接操作(遵循原则④)。

作者引入了**通道拆分**(**channel split**)，即在输入端把特征拆分成$c'$通道和$c-c'$通道，分别作为卷积分支和跳跃连接的输入特征。通常取$c'=c/2$。在输出端使用通道打乱保证不同通道之间的交流。
特征中的一半直接通过跳跃连接输出，这相当于**DenseNet**中的特征复用。

通常相邻的模块之间的特征存在较大的冗余，临近模块的特征复用可以减少这种冗余。第$i$个和第$i+j$个模块之间直连的通道数量为$(\frac{c-c'}{c})^jc$，即特征复用的数量随模块之间的距离呈指数衰减。

实验结果表明，**shufflenet v2**在准确率、浮点运算数量和处理速度上均优于之前的模型：

![](https://pic.imgdb.cn/item/617229b42ab3f51d9181d8b1.jpg)