---
layout: post
title: 'Power pooling: An adaptive pooling function for weakly labelled sound event detection'
date: 2025-10-17
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/68f5a6223203f7be0080178c.png'
tags: 论文阅读
---

> Power pooling：弱标签声音事件检测的自适应池化函数.

- paper：[Power pooling: An adaptive pooling function for weakly labelled sound event detection](https://arxiv.org/abs/2010.09985)


# 0. TL; DR

作者为多实例学习（**MIL**）设计了一种简单有效的幂池化（**Power Pooling**）函数。在计算加权平均时，不再直接使用帧级别的预测概率$y_f$作为权重，而是使用它的n次幂$(y_f)^n$，指数$n$是一个可学习的参数，使其能够根据声音事件自身的特性（如时长）动态调整其聚合策略。

在**DCASE 2017**（纯弱标签）和**DCASE 2019**（半监督）两个公开的大规模声音事件检测数据集上，**Power Pooling**在所有粗粒度（片段级）和细粒度（事件级、段级）的F1分数上，均全面超越了包括线性**softmax**池化、注意力池化等在内的所有对比方法。

# 1. 背景介绍

声音事件检测（**SED**）的目标是识别一段音频中发生了什么声音事件（分类），以及它们何时发生（定位，即起止时间）。传统方法依赖于带有精确时间戳的强标签数据进行训练，但获取这种标签成本高昂。

近年来，随着大规模弱标签数据集（如**Google AudioSet**，仅提供10秒片段内包含哪些事件的类别标签）的出现，弱监督**SED**可以被建模为**多实例学习（Multiple Instance Learning, MIL）**：**包 (Bag)**是一个音频片段（如10秒），**实例 (Instance)**是音频片段中的每一帧，标签仅在包（片段）级别提供。一个阳性包（包含某事件）至少有一个阳性实例（包含该事件的帧）。

在**MIL**框架下，模型首先对每一帧进行预测，得到一个帧级别概率 **(frame-level probability)** $y_f$。然后，通过一个**池化函数**将所有帧的概率聚合成一个片段级别概率 **(clip-level probability)** $y_c$。这个$y_c$再与真实的片段标签计算损失，进行反向传播。理想情况下，它应该给真正的阳性帧分配正梯度（使其概率增加），给阴性帧分配负梯度（使其概率降低）。

![](https://pic1.imgdb.cn/item/68f5a7813203f7be00801efe.png)

**Max-pooling**只给概率最高的帧分配梯度，其他帧梯度为0，导致训练困难，信息利用不充分；**Average-pooling**:给所有帧分配相同的正梯度，无法区分阳性包中的阴性实例，导致定位模糊。线性**Softmax**池化 **(Linear Softmax Pooling)**使用帧概率$y_f$自身作为权重进行加权平均。
    
$$ y_c = \frac{\sum_i (y_f^i \cdot y_f^i)}{\sum_i y_f^i} $$

其梯度的正负取决于$y_f^i$是否大于一个固定的阈值$d = y_c / 2$。这使得它能够同时产生正负梯度，实现对事件边界的定位。然而固定的阈值无法灵活适应这种多样性。

# 2. Power Pooling

**幂池化（Power Pooling）**在线性**softmax**池化的基础上，为作为权重的帧概率引入一个可学习的指数$n$。

$$ y_c = \frac{\sum_i y_f^i \cdot (y_f^i)^n}{\sum_i (y_f^i)^n} $$

其中$y_f^i$是第$i$帧的预测概率，$(y_f^i)^n$是第$i$帧的权重，$n$是一个非负的、可学习的指数参数。

通过对这个新的池化函数求导，可以得到其梯度：

$$ \frac{\partial y_c}{\partial y_f^i} = \frac{(n+1)(y_f^i)^n - n(y_f^i)^{n-1} y_c}{\sum_j (y_f^j)^n} $$

梯度的正负转换点，即新的阈值$d$，发生在分子为0时：

$$ (n+1)(y_f^i)^n = n(y_f^i)^{n-1} y_c \implies y_f^i = \frac{n}{n+1} y_c $$

因此，新的阈值$d$为：

$$ d = \frac{n}{n+1} y_c $$

对应的阈值比例$θ$为：

$$ \theta = \frac{d}{y_c} = \frac{n}{n+1} $$

此时关键的阈值比例$θ$不再是固定的$1/2$，而是由可学习的指数$n$所决定。$n$ -> 0 (趋近于0)时$θ$ -> 0，阈值变得极低，池化行为趋近于**Average-pooling**，模型会鼓励大量的帧都获得正梯度，适合学习长时事件。$n$ -> +∞ (趋近于无穷大)时$θ$ -> 1，阈值变得极高，池化行为趋近于**Max-pooling**，模型只会给概率最高的极少数帧分配正梯度，适合学习短时瞬态事件。$n = 1$时$θ = 1/2$，**Power Pooling**退化为线性**softmax**池化。

通过反向传播，模型可以自动地为每个类别找到一个最优的$n_c$值。为了防止$n$变得过大导致梯度消失问题，作者还在总损失中加入了一个对$n_c$的正则化项：$\lambda \sum_c n_c^2$。

# 3. 实验分析

作者在**DCASE 2017 Task 4**（纯弱标签）和**DCASE 2019 Task 4**（半监督，包含弱标签、强标签和无标签数据）两个大规模数据集上进行了验证。

在两个数据集上，**Power Pooling**在所有三个级别（事件级、段级、片段级）的F1分数上，都取得了最佳性能，全面超越了包括**Max, Average, Attention, Auto-pool**以及线性**softmax**池化在内的所有对比方法。**Power Pooling**在保持了相近精度的同时，显著提高了召回率。这表明，通过自适应地降低某些长时事件的阈值，模型成功地找回了许多之前被错误抑制的阳性帧，从而减少了漏检。

![](https://pic1.imgdb.cn/item/68f5a8fd3203f7be0080233a.png)

作者可视化了在训练过程中，“吸尘器”事件的帧级别预测曲线。可以观察到，**Power Pooling**学习到的指数$n$小于1，这导致其梯度转换阈值$d_{power}$低于线性**softmax**池化的固定阈值$d_{linear}$。在事件发生的早期阶段（0.5s-3s），许多帧的预测概率虽然不高，但它们高于$d_{power}$，因此获得了正向的梯度，概率得以提升。而在线性**softmax**池化中，这些帧因为低于$d_{linear}$而被错误地抑制了。这直观地展示了**Power Pooling**如何通过一个更合适的低阈值，来更完整地定位长时事件。

![](https://pic1.imgdb.cn/item/68f5a9663203f7be0080235f.png)

作者展示了模型为不同事件类别最终学到的指数$n_c$值。结果清晰地显示出一个强相关性：事件的平均持续时间越长，学习到的$n_c$值就越小；持续时间越短，$n_c$值就越大。模型确实学会了根据事件特性，在**Average-pooling**（适合长时）和**Max-pooling**（适合短时）之间进行自适应的权衡。

![](https://pic1.imgdb.cn/item/68f5a9983203f7be0080238f.png)