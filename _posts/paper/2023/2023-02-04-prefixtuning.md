---
layout: post
title: 'Prefix-Tuning: Optimizing Continuous Prompts for Generation'
date: 2023-02-04
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/648d79211ddac507ccda0f9d.jpg'
tags: 论文阅读
---

> Prefix-Tuning：优化生成的连续提示.

- paper：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

对于预训练语言模型的微调工作主要是人工设计离散的模版或者自动化搜索离散的模版。对于人工设计的模版，模版的变化对模型最终的性能特别敏感，加一个词、少一个词或者变动位置都会造成比较大的变化。而对于自动化搜索模版，成本也比较高，这种离散化的**token**搜索出来的结果可能并不是最优的。

传统的微调范式利用预训练模型去对不同的下游任务进行微调，对每个任务都要保存一份微调后的模型权重，一方面微调整个模型耗时长；另一方面也会占很多存储空间。

基于上述两点，**Prefix-Tuning**提出固定预训练语言模型，为语言模型添加可训练、任务特定的前缀，这样就可以为不同任务保存不同的前缀，微调成本也小；这种**Prefix**实际就是连续可微的虚拟**Token**，相比离散的**Token**更好优化，效果更好。

![](https://pic.imgdb.cn/item/648d79631ddac507ccda7778.jpg)

**Prefix-Tuning**是指在输入**token**之前构造一段任务相关的**virtual tokens**作为**Prefix**，在训练时只更新**Prefix**部分的参数，而语言模型中的其他部分参数固定。

针对不同的模型结构，需要构造不同的**Prefix**：
- 自回归架构模型：在句子前面添加前缀，得到 **z = [PREFIX; x; y]**，合适的上文能够在固定语言模型的情况下去引导生成下文（比如：**GPT3**的上下文学习）。
- 编码器-解码器架构模型：编码器和解码器都增加了前缀，得到 **z = [PREFIX; x; PREFIX'; y]**。编码器端增加前缀是为了引导输入部分的编码，解码器端增加前缀是为了引导后续**token**的生成。

![](https://pic.imgdb.cn/item/648d7bd31ddac507ccdfd35c.jpg)

为了防止直接更新**Prefix**的参数导致训练不稳定和性能下降的情况，在**Prefix**层前面加了**MLP**结构，训练完成后，只保留**Prefix**的参数。

![](https://pic.imgdb.cn/item/648d7c791ddac507cce19460.jpg)

除此之外，通过消融实验证实，只调整**embedding**层的表现力不够，将导致性能显著下降，因此，在每层都加了**prompt**的参数，改动较大。

![](https://pic.imgdb.cn/item/648d7cba1ddac507cce1f792.jpg)

另外实验还对比了**Prefix**的位置对于生成效果的影响，**Prefix-tuning**也是要略优于**Infix-tuning**的。其中，**Prefix-tuning**形式为 **[PREFIX; x; y]**，**Infix-tuning**形式为 **[x; INFIX; y]**。

![](https://pic.imgdb.cn/item/648d7d0f1ddac507cce26886.jpg)