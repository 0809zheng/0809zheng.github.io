---
layout: post
title: 'AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition'
date: 2023-02-16
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/657920c6c458853aef438535.jpg'
tags: 论文阅读
---

> AdaptFormer：微调视觉Transformer用于可扩展视觉识别.

- paper：[AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition](https://arxiv.org/abs/2205.13535)

预训练后的**ViT**广泛用于视觉领域，每个模型都需要独立完整的微调才能适应不同的任务，但巨大的计算量和存储负担限制了可迁移性。作者提出了**Transformer**自适应方法，即**AdaptFormer**，只需要训练很少的参数，其他大部分固定参数是跨任务共享的。
- **AdaptFormer**简单高效，引入了轻量级模块，只向**ViT**添加了不到$2\%$的额外参数，能够在不更新其原始预训练参数的情况下增加**ViT**的可迁移性。
- **AdaptFormer**即插即用，可扩展到多种视觉任务，在参数放大时具有优异的鲁棒性。
- **AdaptFormer**大大改善了目标域中**ViTs**的性能表现，显著优于现有的微调方法。

**AdaptFormer**与**VPT**在**SSv2**数据集上的**Parameter-Accuracy trade-off**，同时与全量微调相比，**AdaptFormer**用更少的参数达到更高的性能。

![](https://pic.imgdb.cn/item/657921f4c458853aef4681d2.jpg)

作者借鉴冻结**backbone**和引入可调参数的思想，将参数加到**Transformer**的**MLP**层。**AdaptFormer**用**AdaptMLP**代替了**Transformer**编码器中的**MLP**块。**AdaptMLP**由两个并行的子分支组成：
- 左分支中的**MLP**层与原始网络相同，也叫冻结分支；
- 右分支是引入的**task-specific**轻量级模块，设计为**bottleneck**结构，轻量级编码器-解码器结构旨在通过降低中间维度来限制新引入的参数量。**bottleneck**结构由两个全连接层，一个非线性激活函数和一个缩放因子组成，与原始**ViT**模型的前馈网络 (**FFN**) 并行设置。

![](https://pic.imgdb.cn/item/6579225ac458853aef47800d.jpg)

对于多头自注意力层（**MHSA**），公式计算如下：

$$ 
x_l^\prime=A t t e n t i o n(Q,K,V)=S o f t m a x(\frac{Q K^{T}}{\sqrt{d}})V
$$

对于**AdaptMLP**层，$W_{down}$是下投影层参数，$W_{up}$是上投影层参数，$\hat{d}$是**bottleneck**中间维度，公式计算如下：

$$
\tilde{x}_l = ReLU(LN(x_l^\prime)\cdot W_{down}) \cdot W_{up}
$$

最后通过残差连接融合特征：

$$
x_l = MLP(LN(x_l^\prime)) + s\cdot \tilde{x}_l + x_l^\prime
$$


在微调阶段，原始模型部件（图中的蓝色块）从预训练的**checkpoint**加载权重并保持不变，避免下游任务之间的交互。新添加的参数（橙色块）在特定数据域上随任务特定损失进行更新。在微调后，保持共享参数固定，并额外加载前一阶段微调的额外参数的权重。 **AdaptFormer**仅通过微调少量额外参数就获得了强大的迁移学习能力，避免了任务间的干扰。

![](https://pic.imgdb.cn/item/65792468c458853aef4d1fb4.jpg)

作者基于**ViT**模型进行实验，直接加载原始模型预训练的权重，在微调过程中保持预训练权重**frozen**。对于新添加的模块，向下投影层**Down**用**Kaiming Normal**初始化，其余的部分用零初始化（以零初始化初始新添加的参数，使得新函数近似于原始函数。如果初始化偏离同一函数太远，则模型不稳定，无法训练）。

**AdaptFormer**始终优于**linear probing**和**Visual Prompt tuning（VPT）**方法，在视频领域优势更加显著。相比全量调整，参数量只有不到$2\%$，但准确率高$5\%$。

![](https://pic.imgdb.cn/item/657a692fc458853aef0caa84.jpg)


在参数量的比较上，相比于**VPT**方法，本文的方法在两个数据集上都能达到更高的性能。当参数数量超过任务特定值时，**VPT**的准确性会显著下降，而**AdaptFormer**对不断增加的参数具有鲁棒性。逐渐增加**VPT**的**token**数量，**token≤4**时是稳定的，**≥8**训练会崩溃；而中间维度控制**AdaptFormer**引入的参数的数量，可以看出**Adaptformer**随着参数量的增加精度保持稳定。

![](https://pic.imgdb.cn/item/657a697dc458853aef0dd9ca.jpg)

一系列消融实验：
1. 中间维度$\hat{d}$：$\hat{d}$越小，引入的参数越少。在**SSv2**数据集上，当中间维度增加到**64**时，准确度持续提高，当中间维度大约为**64**时，达到饱和点。当中间维度甚至降低到$1$时，**AdaptFormer**也可以获得不错的性能。
2. 添加层数：**AdaptFormer**的性能与添加的层数呈正相关。当引入相同数量的层时，并行效果比串行效果好。原因是因为并行设计使用一个独立的分支来保持原始特征，并通过元素的缩放和来聚合更新的上下文；同时串行设计相当于添加更多的层，这可能会导致优化困难。
3. 缩放因子$s$：$s$是用于平衡**task-agnostic**特性（由原始冻结分支生成）和**task-specific**特性（由可调**bottleneck**分支生成）。结果表明$s$在$0.1$左右达到最佳性能。
4. 帧数：嵌入**patch token**的数量随着视频帧的数量线性增加。作者使用不同数量的帧进行了实验，即$2，4，8$，观察到增加帧数对所有这三种微调方法都是有益的。**AdaptFormer**始终优于线性方式和**VPT**方法。

![](https://pic.imgdb.cn/item/657a6aa3c458853aef124a0b.jpg)

与前两个方法相比，全量微调策略在特征方面表现良好，但需要消耗大量的计算资源。**AdaptFormer**有助于以更少的可学习参数生成更多的可分离表示。

![](https://pic.imgdb.cn/item/657a6aeac458853aef137594.jpg)