---
layout: post
title: 'Extracting Training Data from Large Language Models'
date: 2021-06-03
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60b889478355f7f718e1d820.jpg'
tags: 论文阅读
---

> 从大型预训练语言模型中提取训练数据.

- paper：Extracting Training Data from Large Language Models
- arXiv：[link](https://arxiv.org/abs/2012.07805)

最近越来越多大型的预训练语言模型被提出，这些模型通常是在私有数据集上训练的。本文提出了一种**训练数据提取攻击(training data extraction attack)**方法，旨在通过对语言模型输入一个查询序列，恢复原始的训练样本。值得一提的是，本文只研究如何从一个黑盒模型中提取训练数据，并不要求提取指定数据。

本文选用**GPT-2**模型作为实验对象。该攻击效果如下图所示，通过对**GPT-2**输入一些前缀词，模型输出了一个人的姓名、邮箱、手机号、传真号和实际地址。

![](https://pic.imgdb.cn/item/60bad1598355f7f718e56694.jpg)

整个攻击和评估流程如下图所示。
1. 文本生成 **generate text**：攻击时向**GPT-2**输入前缀词(可能是空白)生成若干文本，对这些生成的文本进行排序和去重；
2. 成员推断 **membership inference**：评估时选择前$100$个生成文本，通过人工在互联网上搜索判断这些文本是否是输入数据集中的文本，并与实际情况进行比较(询问**OpenAI**)。

![](https://pic.imgdb.cn/item/60b889a18355f7f718e85c81.jpg)

作者提出了一些提高文本生成和成员推断质量的策略。在文本生成过程中，引入一个**decaying temperature**参数降低已生成文本的置信度，使得生成的文本具有更高的多样性；在成员推断中剔除两类生成质量较低、生成置信度较高的文本：(**trivial memorization**)过于常见的文本，如数字$1$到$100$；(**repeated substrings**)重复出现相同的字符串。

下表展示了对模型攻击后得到的输入信息。在生成的$1800$个候选文本中共有$604$个文本出现在训练集中，其中出现了一些较为敏感的信息，如人名、联系方式等。

![](https://pic.imgdb.cn/item/60bad18b8355f7f718e972d5.jpg)

下表展示了不同尺寸的语言模型对输入文本的记忆程度(即通过攻击能够被提取)。对于一篇包含大量重复出现的**URL**的文档，不同尺寸的模型对这些**URL**的记忆程度是不同的。这些**URL**出现频率越大，越容易被模型记住；模型尺寸越大，越容易记住更多文本。

![](https://pic.imgdb.cn/item/60bad1bc8355f7f718ed563c.jpg)