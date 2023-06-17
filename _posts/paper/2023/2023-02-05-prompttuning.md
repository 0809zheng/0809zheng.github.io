---
layout: post
title: 'The Power of Scale for Parameter-Efficient Prompt Tuning'
date: 2023-02-05
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/648d89321ddac507ccfc8e95.jpg'
tags: 论文阅读
---

> Prompt Tuning：参数高效的提示微调.

- paper：[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)

大模型全量微调对每个任务训练一个模型，开销和部署成本都比较高。同时，离散的**prompts**（指人工设计**prompts**提示语加入到模型）方法，成本比较高，并且效果不太好。

基于此，作者提出了**Prompt Tuning**，通过反向传播更新参数来学习**prompts**，而不是人工设计**prompts**；同时冻结模型原始权重，只训练**prompts**参数，训练完以后，用同一个模型可以做多任务推理。

![](https://pic.imgdb.cn/item/648d89fb1ddac507ccfda28d.jpg)

**Prompt Tuning**方法给每个任务定义了自己的**Prompt**（可学习**token**），然后在输入层拼接到输入数据上。通过实验发现，随着预训练模型参数量的增加，**Prompt Tuning**的方法会逼近全参数微调的结果。

![](https://pic.imgdb.cn/item/648d8aa51ddac507ccfe8020.jpg)

同时**Prompt Tuning** 还提出了 **Prompt Ensembling**，也就是在一个批次里同时训练同一个任务的不同 **prompt**（即采用多种不同方式询问同一个问题），这样相当于训练了不同模型，比模型集成的成本小多了。

![](https://pic.imgdb.cn/item/648d8af31ddac507ccfeecbf.jpg)

除此之外，**Prompt Tuning** 论文中还探讨了 **Prompt token** 的初始化方法和长度对于模型性能的影响。通过消融实验结果发现，与随机初始化和使用样本词汇表初始化相比，**Prompt Tuning**采用类标签初始化模型的效果更好。不过随着模型参数规模的提升，这种**gap**最终会消失。

**Prompt token** 的长度在**20**左右时的表现已经不错（超过**20**之后，提升**Prompt token**长度，对模型的性能提升不明显了），同样的，这个**gap**也会随着模型参数规模的提升而减小（即对于超大规模模型而言，即使 **Prompt token** 长度很短，对性能也不会有太大的影响）。

![](https://pic.imgdb.cn/item/648d8b5a1ddac507ccff85d1.jpg)