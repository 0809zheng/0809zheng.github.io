---
layout: post
title: '预训练语言模型ELMO、BERT、GPT'
date: 2020-04-27
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea4013dc2a9a83be5b17721.jpg'
tags: 深度学习
---

> 三种SOTA的语言模型：ELMO、BERT和GPT.

传统的词嵌入把每一个$character$或$word$转化成一个词向量。

相同**type**的词向量在不同的句子中位于不同的**token**，可能具有不同的含义，如：
- It is safest to deposit your money in the **bank**.
- The victim was found lying dead on the river **bank**.

**Contextualized Word Embedding**是指对每一个**token**进行词嵌入，当文本不同时，同一个**type**也具有不同的词嵌入向量。

# 1. ELMO
- **paper**：Deep contextualized word representations
- **arXiv**：[https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)

**Embeddings from Language Model(ELMO)**是一个基于$RNN$的语言模型，从大量句子训练得到。

用一个双向的、深层的$RNN$训练语言模型，每个方向、每一层的隐状态看作对输入的词嵌入编码：

![](https://pic.downk.cc/item/5ea41aa0c2a9a83be5cfaf61.jpg)

对每一层的隐状态向量加权求和，作为最终的词嵌入向量：

![](https://pic.downk.cc/item/5ea41b13c2a9a83be5d01258.jpg)

# 2. BERT
- **paper**：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **arXiv**：[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

**Bidirectional Encoder Representations from Transformers(BERT)**是从大量未标注文本学习得到的语言模型。

$BERT$的结构是$Transformer$的$Encoder$：

![](https://pic.downk.cc/item/5ea41f93c2a9a83be5d4d2ec.jpg)

### (1)Training

**1. Masked LM**

训练时，对输入序列的一部分$word$加上$mask$，预测这些$word$；

具体做法是把$mask$位置的词向量喂入一个简单的线性多元分类网络，同时训练$BERT$和这个简单的网络：

![](https://pic.downk.cc/item/5ea42002c2a9a83be5d553b4.jpg)

**2. Next Sentence Prediction**

训练时，把两个句子用$$[SEP]$$连接，首部加上$$[CLS]$$，判断这两个句子是否有连接关系；

具体做法是把$$[CLS]$$位置的词向量喂入一个简单的线性二元分类网络，同时训练$BERT$和这个简单的网络：

![](https://pic.downk.cc/item/5ea4210fc2a9a83be5d66b65.jpg)

$BERT$在训练时，同时使用上述两种方法。

### (2)Application

**1. Sentiment analysis & Document Classification**
- input: single sentence
- output: class

![](https://pic.downk.cc/item/5ea4219ac2a9a83be5d6f26e.jpg)

**2. Slot filling**
- input: single sentence
- output: class of each word

![](https://pic.downk.cc/item/5ea42270c2a9a83be5d7c96b.jpg)

**3. Natural Language Inference**
- input: two sentences (premise + hypothesis)
- output: class (T/F/ unknown)

![](https://pic.downk.cc/item/5ea422ccc2a9a83be5d82083.jpg)

**4. Extraction-based Question Answering(QA)**
- input: two sentences (question + document)
- output: 答案在文本中的起始和终止位置

训练两个向量，橙色向量寻找起始位置，蓝色向量寻找终止位置：

![](https://pic.downk.cc/item/5ea42370c2a9a83be5d8be22.jpg)

**Enhanced Representation through Knowledge Integration(ERNIE)**是为中文设计的$BERT$模型：

![](https://pic.downk.cc/item/5ea423e5c2a9a83be5d93163.jpg)

# 3. GPT
- **paper**：Improving Language Understanding by Generative Pre-Training
- **arXiv**：[NLPIR](http://www.nlpir.org/wordpress/2019/06/16/improving-language-understanding-by-generative-pre-training/)

**Generative Pre-Training (GPT)**的结构是$Transformer$的$Decoder$：

![](https://pic.downk.cc/item/5ea425b0c2a9a83be5db120d.jpg)

![](https://pic.downk.cc/item/5ea42c3fc2a9a83be5e17324.jpg)
