---
layout: post
title: '词嵌入'
date: 2020-04-29
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea917bbc2a9a83be5f8a32b.jpg'
tags: 深度学习
---

> Word Embedding.

**词嵌入Word Embedding**是一种**unsupervised learning**的方法，让机器阅读大量**文档documents**从而学习到**单词words**的意思。

**本文目录**：
1. 背景
2. Count based
3. Perdition based


# 1. 背景
在自然语言处理任务中，对于每一个单词（或字符）的表示可以使用**One-hot编码**，即**1-of-N encoding**：

![](https://pic.downk.cc/item/5ea830c6c2a9a83be513750f.jpg)

这种方法每一个单词是独立的，没有考虑到单词之间的相关性。

进一步提出了**word class**，把语义相近的单词分为一类：

![](https://pic.downk.cc/item/5ea83138c2a9a83be514034e.jpg)

此方法对同一类的单词仍然没有很好的区分。为此，提出了**词嵌入Word Embedding**的概念：

![](https://pic.downk.cc/item/5ea83164c2a9a83be5143a5e.jpg)

词嵌入是把单词映射到一个$k$维的向量空间，每一个单词用一个$k$维向量表示。

# 2. Count based
**Count based Embedding**是指在一个文档中若两个单词同时出现的频率很高，那么两个单词的向量非常接近。

![](https://pic.downk.cc/item/5ea838cac2a9a83be51dedc7.jpg)

**Glove Vector**
- project：[glove](http://nlp.stanford.edu/projects/glove/)


# 3. Perdition based
**Perdition based Embedding**是与**downstream**任务一起训练的。

使用一层**共享参数**的神经网络作为词嵌入层，把输入单词仿射变换为词嵌入向量。

1. **Language Modeling**：词嵌入与语言模型一起训练![](https://pic.downk.cc/item/5ea83c7ec2a9a83be522d13c.jpg)
2. **Continuous bag of word (CBOW) model**：根据上下文预测单词![](https://pic.downk.cc/item/5ea83cbcc2a9a83be523234a.jpg)
3. **Skip-gram**：根据单词预测上下文![](https://pic.downk.cc/item/5ea83cd1c2a9a83be5233faf.jpg)
