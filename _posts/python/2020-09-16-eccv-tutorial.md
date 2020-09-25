---
layout: post
title: 'ECCV 2020 Tutorial：PyTorch性能调优指南'
date: 2020-09-16
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f61a1b0160a154a67686e25.jpg'
tags: Python
---

> ECCV 2020 Tutorial on PyTorch Performance Tuning Guide.

- [video&pdf](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/)

# 目录：
1. use async data loading / augmentation
2. enable cuDNN autotuner
3. increase batch size
4. remove unnecessary computation
5. use DistributedDataParallel instead of DataParallel
6. efficiently zero-out gradients
7. apply PyTorch JIT to fuse pointwise operations
8. checkpoint to recompute intermediates

# 1. use async data loading / augmentation
- 使用异步的数据加载和增强。

**Pytorch**的`DataLoader`支持异步的数据加载和增强操作，默认情况下有：

```
{num_workers=0, pin_memory=False}
```
- `num_workers`：是加载数据（**batch**）的线程数目。
1. 当加载**batch**的时间**小于**模型的训练时间时，**GPU**每次训练完都可以直接从**CPU**中取到下一个**batch**的数据，无需额外的等待，因此也不需要多余的**worker**，即使增加**worker**也不会影响训练速度；
2. 当加载**batch**的时间**大于**模型的训练时间时，**GPU**每次训练完都需要等待**CPU**完成数据的载入，若增加**worker**，即使**worker_1**还未就绪，**GPU**也可以取**worker_2**的数据来训练。
- `pin_memory`：设置**锁页内存**，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存（如硬盘）进行交换，而不锁页内存在主机内存不足时数据会存放在虚拟内存中，而显卡中的显存全部是锁页内存。当计算机的内存充足时，可以设置`pin_memory=True`，意味着生成的**Tensor**数据最开始是属于内存中的锁页内存，这样将内存的**Tensor**转义到**GPU**的显存就会更快一些。当系统卡住，或者交换内存使用过多的时候，设置`pin_memory=False`。

下表是训练**MNIST**图像分类实验中不同参数的对照试验（环境**PyTorch 1.6 + NVIDIA Quadro RTX 8000**）：

![](https://pic.downk.cc/item/5f61af36160a154a676b7bef.jpg)

# 2. enable cuDNN autotuner
- 允许**cuDNN**进行调校

在训练卷积神经网络时，**cuDNN**支持多种不同的算法计算卷积，使用调校工具**autotuner**可以运行一个较小的**benchmark**检测这些算法，并从中选择表现最好的算法。

对于卷积神经网络，只需要设置：

```
torch.backends.cudnn.benchmark = True
```

下表是使用`nn.Conv2d(64,3)`处理大小为`(32,64,64,64)`数据的对照试验（环境**PyTorch 1.6 + NVIDIA Quadro RTX 8000**）：

![](https://pic.downk.cc/item/5f61b1c1160a154a676c093e.jpg)

# 3. increase batch size
- 增加批量大小

在**GPU**内存允许的情况下增加**batch size**，通常结合混合精度训练使**batch size**更大。该方法通常结合学习率策略或更换优化方法：
- 学习率衰减，增加学习率**warmup**，调节权重衰减；
- 换用为大**batch**设计的优化方法：LARS、LAMB、NVLAMB、NovoGrad

# 4. remove unnecessary computation
- 移除不必要的计算

如**batch norm**中会有**rescale**和**reshift**操作，因此其之前的卷积层中的**bias**参数可以被合并：

![](https://pic.downk.cc/item/5f61b35b160a154a676c5e14.jpg)

# 5. use DistributedDataParallel instead of DataParallel
- 使用`DistributedDataParallel`代替`DataParallel`

`DataParallel`针对单一进程开启多个线程，用一个**CPU**核驱动多个**GPU**，总体还是在这些**GPU**上运行单一**python**进程。

`DistributedDataParallel`同时开启多个进程，用多个**CPU**核分别驱动多个**GPU**，每个**GPU**上都运行一个进程。

![](https://pic.downk.cc/item/5f61b58a160a154a676cd629.jpg)

# 6. efficiently zero-out gradients
- 有效地进行梯度置零

每次更新时需要进行梯度置零，通常使用以下语句：
```
model.zero_grad() 或 optimizer.zero_grad()
```

上述语句会对每一个参数执行**memset**（为新申请的内存做初始化工作），反向传播更新梯度时使用‘$+=$’操作（读+写）。为提高效率，可以将梯度置零语句替换成：

```
for param in model.parameters():
    param.grad = None
```

上述语句不会对每个参数执行**memset**，并且反向传播更新梯度时使用‘$=$’操作（写）。

# 7. apply PyTorch JIT to fuse pointwise operations
- 使用**PyTorch JIT**融合逐点操作

**PyTorch JIT**能够将逐点操作（**pointwise operations**）融合到单个**CUDA**核上，从而减小执行时间。

如下图，只需要在执行语句前加上`@torch.jit.script`便可以实现（环境**PyTorch 1.6 + NVIDIA Quadro RTX 8000**）：

![](https://pic.downk.cc/item/5f61bb7e160a154a676e5101.jpg)

# 8. checkpoint to recompute intermediates
- 使用**checkpoint**重新计算中间值

在常规的训练过程中，前向传播会存储中间运算的输出值用以反向传播，这一步需要更多的内存，从而限制了训练时**batch size**的大小；在反向传播更新参数时不再需要额外的运算。

`torch.utils.checkpoint`提供了**checkpoint**操作。在前向传播只存储部分中间运算的输出值，减小了对内存的占用，可以使用更大的**batch size**；反向传播时需要额外的运算。

**checkpoint**操作是一种用时间换内存的操作。通常需要选择对合适的操作进行，如较小的重复计算代价（**re-computation cost**）和较大的内存占用（**memory footprint**），包括激活函数、上下采样和较小堆积深度（**accumulation depth**）的矩阵向量运算。

