---
layout: post
title: 'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity'
date: 2021-05-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6095f70ad1a9ae528f3875f6.jpg'
tags: 论文阅读
---

> Switch Transformer：训练万亿级参数的语言模型.

- paper：Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- arXiv：[link](https://arxiv.org/abs/2101.03961)

在**NLP**领域，模型的大小(用模型的参数量衡量)通常与模型的性能成正比。**GPT-3**模型具有$1750$亿参数($175$ **billion**)，使用**稀疏注意力(sparse attention)**运算，这类运算很难发挥**GPU**等硬件的性能。作者提出了**Switch Transformer**，采用**稀疏路由(sparse routing)**设计网络结构，但计算上仍然是密集的，充分使用**GPU**等硬件的性能，可以训练出具有$1.6$万亿参数($1.6$ **trillion**)的语言模型。

**Switch Transformer**是按照**MoE**结构范式设计的，采用特殊的分布式训练设置和一些训练技巧，分别介绍如下。

## Mixture of Expert (MoE)

**混合专家系统(Mixture of Expert, MoE)**是一种神经网络的设计范式，于上世纪$90$年代被提出。**MoE**是指根据数据的不同产生方式分别训练不同子模型，每个模型被称为一个**专家(expert)**，对于每次数据输入使用一个门控模块决定使用哪个专家，模型的实际输出为各个专家输出的加权组合。每个专家通常具有不同的网络结构。

在深度学习中，模型对于输入通常重复使用相同的模型参数。将**MoE**的思想引入，设计一个具有**稀疏激活(sparsely-activated)**特征的模型。尽管模型本身具有较多的参数量，但对于每次输入只激活其中的一部分，并没有额外的计算量。这个过程是通过动态路由实现的。

![](https://pic.imgdb.cn/item/60963ae3d1a9ae528fe78188.jpg)

## Partitioning Strategy

![](https://pic.imgdb.cn/item/60963b55d1a9ae528fec84ff.jpg)

在深度学习中，可能会出现由于模型过大或数据集过大，无法在单张**GPU**或单个**core**上训练的情况。选择合适的分布式训练方法能够有效加快训练过程，减少训练难度。模型参数和数据集的分布式训练在**MoE**范式下有以下几种形式：
- 数据并行(**data parallelism**)：模型参数在所有**core**上共享；数据被划分到每个**core**上；
- 模型并行(**model parallelism**)：模型参数被划分到每个**core**上；数据在所有**core**上共享；
- 数据和模型并行(**data and model parallelism**)：模型参数被划分到每组**core**上，而在不同组之间共享；数据被划分到每组**core**上，而在不同组之间共享；
- 专家和数据并行(**expert and data parallelism**)：模型参数在每个**core**上都不同，对应不同的专家；数据被划分到每个**core**上；
- 专家,模型和数据并行(**expert, model and data parallelism**)：模型参数被划分到每组**core**上，而在不同组之间不同，对应不同的专家；数据被划分到每组**core**上，而在不同组之间共享。

## Switch Transformer

将**MoE**引入**Transformer**的过程如下。

**Transformer**的主体部分是由多头自注意力层**MHA**和前向传播层**FFN**堆叠组合而成。**MHA**实现不同**token**之间的交互，**FFN**是对每个**token**进行非线性变换，其输出作为下一层的输入，可以看作其实现了不同层之间的交互。

由于**FFN**通常具有更多参数，因此将其作为专家。通过设置多个**FFN**，对于每个输入**token**计算一个线性得分，将其作为路由选择合适的**FFN**。对于模型整体，由于引入多个**FFN**增加的模型的参数量，但每次计算中只使用了其中一个**FFN**，并没有增加计算量。

![](https://pic.imgdb.cn/item/60963ffdd1a9ae528f22f16c.jpg)

作者还提出了一些使训练更稳定的技巧：
1. 精度转换。在**float16**精度下训练使得模型不稳定。作者提出，在计算路由时将数据扩张为**float32**精度，计算结束后再恢复成**float16**精度，能够提高模型训练的稳定性。
2. 更小的参数初始化。作者提出，将模型参数初始化为原来的$1/10$，能够提高模型训练的稳定性。
3. 正则化。作者提出，采用**expert dropout**，即在专家层采用更大的**dropout rate**($0.4$,其余层设置为$0.1$)，能够提高模型训练的稳定性。

