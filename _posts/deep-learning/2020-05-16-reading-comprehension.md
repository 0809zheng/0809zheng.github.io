---
layout: post
title: '阅读理解'
date: 2020-05-16
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ee5dfe1c2a9a83be540edac.jpg'
tags: 深度学习
---

> Reading Comprehension.

**阅读理解（Reading Comprehension）**是指让计算机理解文本，完成理解文本有关的任务，包括：
- 完型填空：原文中挖出一个空来，由机器根据对文章上下文的理解去补全。代表数据集有CNN/Daily Mail。
- 多项选择：每篇文章对应多个问题，每个问题有多个候选答案，机器需要在这些候选答案中找到最合适的那个。代表数据集有RACE。
- 区域预测：也称为抽取式问答（Extractive QA），即给定文章和问题，机器需要在文章中找到答案对应的区域（span），给出开始位置和结束位置。代表数据集有SQuAD（Stanford Question Answering Dataset）。
- 自由形式：不限定问题所处的段落，即一个问题可能是需要理解多个段落甚至多篇文章。代表数据集有DuReader（百度），MS MARCO（微软）。

# 1. Benchmarks

### SQuAD
- SQuAD：Stanford Question Answering Dataset
- SQuAD1.1 paper：SQuAD:100000+ Question for Machine Comprehension of Test
- SQuAD2.0 paper: Know What You Don't Know: Unanswerable Questions for SQuAD
- Website：[link](https://rajpurkar.github.io/SQuAD-explorer/)

### MS MARCO
- MS MARCO：Microsoft MAchine Reading Comprehension
- Paper：MS MARCO: A Human Generated MAchine Reading COmprehension Dataset
- Website: [link](http://www.msmarco.org/)

### DuReader
- DuReader：百度阅读理解数据集
- Website: [link](http://ai.baidu.com/broad/introduction?dataset=dureader)

# 2. 问答型阅读理解框架

![](https://pic.downk.cc/item/5ee5dfe1c2a9a83be540edac.jpg)

- 向量化层：分别将原文和问题中的 tokens 映射为向量表示（Word2Vec，Glove）。
- 编码层：主要使用循环神经网络来对原文和问题进行编码，这样编码后每个 token 的向量表示就蕴含了上下文的语义信息。
- 交互层：主要负责分析问题和原文之间的交互关系，并输出编码了问题语义信息的原文表示，即 query-aware 的原文表示。
- 答案层：基于 query-aware 的原文表示来预测答案范围（答案起始位置和终止位置）。

## Microsoft R-net
- paper: R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS.

![](https://pic.downk.cc/item/5ee5e087c2a9a83be542178a.jpg)

- 向量化层：使用 GloVe 词向量和 Char embedding 两种方法以丰富输入特征。
- 编码层：采用循环神经网络进行编码。
- 交互层：双交互层结构，第一层基于门限的注意力循环神经网络（gated-attention based recurrent network）匹配question和passage，获取问题的相关段落表示（question-aware passage representation）；第二层基于自匹配注意力机制的循环神经网络（self-matching attention network）将passage和它自己匹配，从而实现整个段落的高效编码。
- 答案层：基于指针网络（pointer-network）定位答案所在位置。

## BiDAF
- paper: Bidirectional attention flow for machine comprehension.

![](https://pic.downk.cc/item/5ee5e243c2a9a83be545211c.jpg)

- 向量化层：混合了词级 Embedding 和字符级 Embedding，词级embedding 使用预训练的词向量进行初始化，而字符级embedding 使用 CNN 进一步编码。
- 编码层：两种 Embedding 共同经过 2 层 Highway Network 作为Encode 层输入。
- 交互层：Interaction 层中引入了双向注意力机制，即首先计算一个原文和问题的 Alignment matrix ，然后基于该矩阵计算Query2Context 和 Context2Query 两种注意力，并基于注意力计算 query-aware 的原文表示，接着使用双向 LSTM 进行语义信息的聚合。
- 答案层：使用 Boundary Model 来预测答案开始和结束位置。