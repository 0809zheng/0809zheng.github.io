---
layout: post
title: 'AdapterDrop: On the Efficiency of Adapters in Transformers'
date: 2023-02-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/648e5fec1ddac507cc0db899.jpg'
tags: 论文阅读
---

> AdapterDrop：提高Transformer中的Adapter模块的效率.

- paper：[AdapterDrop: On the Efficiency of Adapters in Transformers](https://arxiv.org/abs/2010.11918)

近年来[<font color=blue>Adapter</font>](https://0809zheng.github.io/2023/02/01/adapter.html)已被证明可以很好地用于机器翻译、跨语言迁移、社区问答和迁移学习的任务组合。尽管它们最近很受欢迎，但**Adapter**的计算效率尚未在参数效率之外得到探索。

作者通过对**Adapter**的计算效率进行分析，发现与全量微调相比，**Adapter**在训练时快$60\%$，但是在推理时慢$4\%-6\%$。基于此，作者提出了**AdapterDrop**方法缓解该问题。

**AdapterDrop**在不影响任务性能的情况下，对**Adapter**动态高效的移除，尽可能的减少模型的参数量，提高模型在反向传播（训练）和正向传播（推理）时的效率。

![](https://pic.imgdb.cn/item/648e68af1ddac507cc191a6a.jpg)

实验表明，从较低的 **Transformer** 层中删除**Adapter**可以显着提高多任务设置中的推理速度。 例如，将前五个**Transformer**层中的**Adapter**丢弃，在对 **8** 个任务进行推理时，速度提高了 $39\%$。并且即使有多个丢弃层，**AdapterDrop** 也能保持良好的结果。

![](https://pic.imgdb.cn/item/648e69111ddac507cc198e71.jpg)

除此之外，作者还研究了对 **AdapterFusion**中的**Adapter**进行剪枝后的效果。

![](https://pic.imgdb.cn/item/648e6bed1ddac507cc1d3abe.jpg)

通过实验表明可以移除 **AdapterFusion** 中的大多数**Adapter**而不影响任务性能。使用剩余的两个**Adapter**，实现了与具有八个**Adapter**的完整 **AdapterFusion** 模型相当的结果，并将推理速度提高了 $68\%$。

![](https://pic.imgdb.cn/item/648e6c441ddac507cc1daad2.jpg)

因此，作者建议在实际部署这些模型之前执行 **AdaperFusion** 剪枝。 这是一种简单而有效的技术，即使在完全保持性能的情况下也能实现效率提升。

总之，**AdapterDrop** 通过从较低的 **Transformer** 层删除可变数量的**Adaper**来提升推理速度。 当对多个任务执行推理时，动态地减少了运行时的计算开销，并在很大程度上保持了任务性能。