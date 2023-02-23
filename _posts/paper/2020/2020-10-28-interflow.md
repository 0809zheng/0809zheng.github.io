---
layout: post
title: 'Interflow: Aggregating Multi-layer Feature Mappings with Attention Mechanism'
date: 2020-10-28
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63b7d1f9be43e0d30e561f81.jpg'
tags: 论文阅读
---

> Interflow：通过注意力机制汇聚多层特征映射.

- paper：[Interflow: Aggregating Multi-layer Feature Mappings with Attention Mechanism](https://arxiv.org/abs/2106.14073v3)

传统的卷积神经网络模型具有层次结构，并通过最后一层的特征映射来获得预测输出。然而在实践中很难确定最优网络深度，并保证中间层学习到显著的特征。

本文提出了**Interflow**，根据深度把卷积网络划分为几个阶段，并利用每个阶段的特征映射进行预测。把这些预测分支输入到一个注意力模块中学习这些预测分支的权值，并将其聚合得到最终的输出。

![](https://pic.imgdb.cn/item/63b81ab0be43e0d30e146939.jpg)

**Interflow**对浅层和深层学习到的特征进行加权和融合，使各个阶段的特征信息得到合理有效的处理，使中间层能够学习到更多有判别性的特征，增强了模型的表示能力。此外通过引入注意力机制，**Interflow**可以缓解梯度消失问题，降低网络深度选择的难度，减轻可能出现的过拟合问题，避免网络退化。

注意力机制具有两种形式。其中硬注意力机制是指直接把每个分支的权重看作一个超参数。但是当分支流的数量太大时，会引入太多的超参数，因此很难得到最优组合。

![](https://pic.imgdb.cn/item/63b81bbebe43e0d30e17c0d4.jpg)

软注意力机制允许模型独立学习权重。具体地，利用**1×n**卷积让模型学习每个分支的权重。因此它使模型能够识别出特定任务需要注意的阶段特征信息，并合理有效地整合了不同阶段的特征信息。

本文实验中使用**Interflow**的具体**CNN**模型示意图。在该模型中，将**VGGNet-16**的卷积层作为特征提取网络，将**13**个卷积层分为**4**个阶段，并对每个阶段的输出特征映射应用自适应平均池化和全连接层，这样就得到了分类置信系数，即各阶段分支的特征信息。进一步通过注意力机制对特征进行加权，然后得到最终输出。

![](https://pic.imgdb.cn/item/63b81cd5be43e0d30e1bfa3b.jpg)