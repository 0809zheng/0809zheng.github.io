---
layout: post
title: '预训练语言模型(Pretrained Language Model)'
date: 2020-04-27
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ea4013dc2a9a83be5b17721.jpg'
tags: 深度学习
---

> Pretrained Language Models.

**预训练语言模型**(**Pretrained Language Models,PLMs**)是一种从大量无标签的语料库中学习通用的自然语言特征表示的方法。笔者认为，预训练模型之于自然语言处理，就好比**backbone**之于计算机视觉。使用预训练语言模型的步骤如下：
1. 在大量无标签的语料库上进行特定任务的**预训练**；
2. 在下游任务的语料库上进行**微调**。

本文首先介绍语言的特征表示，并介绍预训练语言模型的发展。

# 1. 语言的特征表示
自然语言处理中对于语言的特征表示应能够从文本语料库中学习到内在语言规则和常识知识，如词义、句法结构、词类、语用学信息等。一种好的语言特征表示应具有与具体任务无关的通用含义，又能够针对具体的任务提供有用的信息。目前对语言的特征表示有两种形式，即**上下文无关的嵌入(non-contextual embedding)**和**上下文相关的嵌入(contextual embedding)**。

![](https://pic.imgdb.cn/item/60ebf3395132923bf857acf4.jpg)

### (1) Non-Contextual Embedding

上下文无关的嵌入通常是由**词嵌入(word embedding)**实现的，即把句子中的每一个**word**转化成一个词向量：$x \to e_x$。在这类方法中，不同句子中的相同**word**都会被嵌入为同一个词向量，然而相同**word**在不同的句子中位于不同的**token**位置，可能具有不同的含义，如下面两个句子：
- It is safest to deposit your money in the **bank**.
- The victim was found lying dead on the river **bank**.

在上面两个句子中**bank**分别表示银行和河岸；因此这种词嵌入无法解决**多义问题**。此外，由于词向量的个数是有限的，对于之前不存在的词，则无法得到相应的词嵌入向量(即**OOV问题**,**out of vocabulary**)。

基于上下文无关的嵌入方法可以被认为是早期的预训练语言模型，代表模型有**Word2Vec**,**CBOW**,**Glove**。这类模型结构简单，尽管是从无标注语料库中训练得到的，也能获得高质量的词向量；其学习到的词向量能够捕捉文本中潜在的语法和语义信息，但这类预训练词向量无法随上下文而动态变化，只是简单地学习"共现词频"，无法理解更高层次的文本概念，如多义性、句法特征、语义角色、指代等。

### (2) Contextual Embedding
上下文相关的嵌入是指根据当前文本的上下文，灵活地对每一个**token**位置(注意不是对每一个**word**)进行词嵌入；当文本不同时，同一个**word**也会具有不同的词嵌入向量。这通常是由一个神经网络编码器$f_{enc}(\cdot)$实现的：$[h_1,...,h_T]=f_{enc}([x_1,...,x_T])$。随着**LSTM**,**Transformer**等模型的引入，这种结合上下文信息的预训练语言模型获得了更多的关注。这类预训练语言模型能够根据预训练任务学习包含词的上下文信息的词表示，并用于不同的下游任务中。这类预训练语言模型的优点如下：
1. 可以在大规模预训练语料库中学习到**通用语言表示**；
2. 可以提供一个更好的下游任务**初始化模型**，提高下游任务的表现并加速收敛；
3. 可以看作一种**正则化**，防止模型在小数据集上过拟合。

# 2. 预训练语言模型的发展

### (1) 预训练语言模型的结构

**Transformer**模型是编码-解码端 （**Encoder-Decoder**）的架构。但是当前对于语言模型的分类，将语言模型分为三个类型：**编码端（Encoder-Only）**，**解码端（Decoder-Only）**和**编码-解码端（Encoder-Decoder）**。

**① 编码端（Encoder-Only）架构**

编码端架构（如**BERT, RoBERTa**）可以生成上下文向量表征，但不能直接用于生成文本。这些上下文向量表征通常用于**自然语言理解**任务（形式为分类任务，如文本分类、情感分类）。该架构的优势是对于文本的上下文信息有更好的理解，因此该模型架构才会多用于理解任务。该架构的优点是对于每个$x_i$，上下文向量表征可以双向地依赖于左侧上下文$(x_{1:i−1})$和右侧上下文$(x_{i+1:L})$。但是缺点在于不能自然地生成文本，且需要更多的特定训练目标（如掩码语言建模）。

**② 解码端（Decoder-Only）架构**

解码器架构（如**GPT**系列）是常见的自回归语言模型，通常用于**自然语言生成**任务：给定一个提示$x_{1:i}$，它们可以生成上下文向量表征，并对下一个词元$x_{i+1}$  （以及递归地，整个完成$x_{i+1:L}$） 生成一个概率分布。与编码端架构比，其优点为能够自然地生成文本，有简单的训练目标（最大似然）。缺点也很明显，对于每个$x_i$，上下文向量表征只能单向地依赖于左侧上下文$(x_{1:i−1})$。

**③ 编码-解码端（Encoder-Decoder）架构**

编码-解码端架构（如**BART, T5**）在某种程度上结合了两者的优点：它们可以使用双向上下文向量表征来处理输入$x_{1:L}$，并且可以生成输出$y_{1:L}$。该模型的具有编码端、解码端两个架构的共同的优点，对于每个$x_i$，上下文向量表征可以双向地依赖于左侧上下文$(x_{1:i−1})$和右侧上下文$(x_{i+1:L})$，可以自由的生成文本数据。缺点就是需要更多的特定训练目标。


### (2) 预训练语言模型的任务

预训练语言模型的预训练任务通常有以下几类：
- **概率语言建模 Language Modeling(LM)**

概率语言建模是自然语言处理中最常见的无监督任务，通常指**自回归(autoregressive)**或单向语言建模，即给定前面所有词预测下一个词：

$$ p(x_{1:T}) = \prod_{t=1}^{T} p(x_{t}|x_{0:t-1}) $$

- **掩码语言建模 Masked Language Modeling(MLM)**

掩码语言建模是指从输入序列中遮盖一些**token**(为这些**token**加上**mask**)，然后训练模型通过其余的**token**预测**masked token**。然而这种预训练方法会使预训练和微调之间产生不匹配(**discrepancy**)问题，因为在下游任务中`MASK`等预训练中使用的特殊**token**并不存在。这类方法也称为**自编码(autoencoding)**式语言模型。

- **序列到序列的掩码语言建模 Seq2Seq Masked Language Modeling(Seq2Seq MLM)**

掩码语言建模**MLM**通常用于解决分类问题，即将**masked**序列作为输入，将模型输出送入**softmax**分类器以预测**masked token**。序列到序列的掩码语言建模是指采用编码器-解码器结构，将**masked**序列输入编码器，解码器以自回归的方式顺序生成**masked token**。

- **增强掩码语言建模 Enhanced Masked Language Modeling(E-MLM)**

增强掩码语言建模**E-MLM**是指在掩码语言建模的过程中使用了一些增强方法。不同的模型使用了不同的增强方法，详见下表。

- **排列语言建模 Permuted Language Modeling(PLM)**

排列语言建模是指在输入序列的随机排列上进行语言建模。给定输入序列，从所有可能的序列排列中随机抽样一个排列。将该排列序列中的一些**token**选定为目标，训练模型根据其余**token**和目标的正常位置(**natural position**)来预测这些目标**token**。

### (3) 常见的预训练语言模型

| 预训练模型 | 结构 | 预训练任务 | 参数量(M百万,B十亿) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: |
| [ELMo](https://0809zheng.github.io/2021/01/01/elmo.html) | 双向LSTM | LM | - |
| [GPT](https://0809zheng.github.io/2021/01/03/gpt.html) | Transformer解码器 | LM | $117$M |
| [GPT2](https://0809zheng.github.io/2021/01/11/gpt2.html) | Transformer解码器 | LM | $117$-$1542$M |
| [GPT3](https://0809zheng.github.io/2020/07/13/gpt3.html) | Transformer解码器 | LM | $125$M-$175$B |
| [BERT](https://0809zheng.github.io/2021/01/02/bert.html) | Transformer编码器 | MLM+相邻句子预测(Next Sentence Prediction) | $110$-$340$M |
| [ALBERT](https://0809zheng.github.io/2021/01/14/albert.html) | Transformer编码器 | MLM+句子顺序预测(Sentence-Order Sentence Prediction) | $12$-$235$M |
| [ELECTRA](https://0809zheng.github.io/2021/01/16/electra.html) | Transformer编码器 | MLM+替换词检测(Replaced Token Detection) | $14$-$335$M |
| [REALM](https://0809zheng.github.io/2020/12/27/realm.html) | Transformer编码器 | MLM+知识检索(Knowledge Retrieval) | $330$M |
| [MASS](https://0809zheng.github.io/2021/08/18/mass.html) | Transformer | Seq2Seq MLM | $220$M-$11$B |
| [UniLM](https://0809zheng.github.io/2021/08/17/unilm.html) | Transformer编码器 | Seq2Seq MLM | $340$M |
| [BART](https://0809zheng.github.io/2021/03/14/bart.html) | Transformer | Seq2Seq MLM | $139$M-$406$M |
| [T5](https://0809zheng.github.io/2021/01/08/t5.html) | Transformer | Seq2Seq MLM | $220$M-$11$B |
| [T5.1.1](https://0809zheng.github.io/2021/01/09/t511.html) | Transformer | Seq2Seq MLM | $220$M-$11$B |
| [mT5](https://0809zheng.github.io/2021/01/10/mt5.html) | Transformer | Seq2Seq MLM | $300$M-$13$B |
| [RoBERTa](https://0809zheng.github.io/2021/08/16/roberta.html) | Transformer编码器 | E-MLM(Dynamic Masking) | $355$M |
| [DeBERTa](https://0809zheng.github.io/2021/04/02/deberta.html) | Transformer编码器 | E-MLM(Disentangled Attention+Enhanced Mask Decoder) | $390$M |
| [XLNet](https://0809zheng.github.io/2021/08/19/xlnet.html) | Transformer编码器 | PLM | $110$-$340$M |


# ⚪ 参考文献
- [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271)：(arXiv2003)一篇预训练模型的综述。
- [<font color=Blue>Deep contextualized word representations</font>](https://0809zheng.github.io/2021/01/01/elmo.html)：(arXiv1802)ELMo：使用语言模型进行词嵌入。
- [<font color=Blue>Improving Language Understanding by Generative Pre-Training</font>](https://0809zheng.github.io/2021/01/03/gpt.html)：(NLPIR2018)GPT：使用生成式预训练模型提高对语言的理解。
- [<font color=Blue>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</font>](https://0809zheng.github.io/2021/01/02/bert.html)：(arXiv1810)BERT：从Transformer中获得上下文的编码表示。
- [<font color=Blue>MASS: Masked Sequence to Sequence Pre-training for Language Generation</font>](https://0809zheng.github.io/2021/08/18/mass.html)：(arXiv1905)MASS：序列到序列的掩码语言建模。
- [<font color=Blue>Unified Language Model Pre-training for Natural Language Understanding and Generation</font>](https://0809zheng.github.io/2021/08/17/unilm.html)：(arXiv1905)UniLM：使用BERT实现序列到序列的预训练。
- [<font color=Blue>XLNet: Generalized Autoregressive Pretraining for Language Understanding</font>](https://0809zheng.github.io/2021/08/19/xlnet.html)：(arXiv1906)XLNet：使用排列语言建模训练语言模型。
- [<font color=Blue>RoBERTa: A Robustly Optimized BERT Pretraining Approach</font>](https://0809zheng.github.io/2021/08/16/roberta.html)：(arXiv1907)RoBERTa：鲁棒优化的BERT预训练方法。
- [<font color=Blue>ALBERT: A Lite BERT for Self-supervised Learning of Language Representations</font>](https://0809zheng.github.io/2021/01/14/albert.html)：(arXiv1909)ALBERT：一种轻量型的BERT模型。
- [<font color=Blue>BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension</font>](https://0809zheng.github.io/2021/03/14/bart.html)：(arXiv1910)BART: 用于自然语言生成、翻译和理解的去噪序列到序列预训练模型。
- [<font color=Blue>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</font>](https://0809zheng.github.io/2021/01/08/t5.html)：(arXiv1910)T5：编码器-解码器结构的预训练语言模型。
- [<font color=Blue>Language Models are Unsupervised Multitask Learners</font>](https://0809zheng.github.io/2021/01/11/gpt2.html)：(2019)GPT2：语言模型是无监督的多任务模型。
- [<font color=Blue>REALM: Retrieval-Augmented Language Model Pre-Training</font>](https://0809zheng.github.io/2020/12/27/realm.html)：(arXiv2002)REALM：通过检索增强预训练语言模型。
- [<font color=Blue>GLU Variants Improve Transformer</font>](https://0809zheng.github.io/2021/01/09/t511.html)：(arXiv2002)T5.1.1：使用GLU改进预训练语言模型T5。
- [<font color=Blue>ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators</font>](https://0809zheng.github.io/2021/01/16/electra.html)：(arXiv2003)ELECTRA：判别式的预训练语言模型。
- [<font color=Blue>Language Models are Few-Shot Learners</font>](https://0809zheng.github.io/2020/07/13/gpt3.html)：(arXiv2005)GPT3：语言模型是少样本学习模型。
- [<font color=Blue>DeBERTa: Decoding-enhanced BERT with Disentangled Attention</font>](https://0809zheng.github.io/2021/04/02/deberta.html)：(arXiv2006)DeBERTa：使用分解注意力机制和增强型掩膜解码器改进预训练语言模型。
- [<font color=Blue>mT5: A massively multilingual pre-trained text-to-text transformer</font>](https://0809zheng.github.io/2021/01/10/mt5.html)：(arXiv2010)mT5：多语言版本的预训练语言模型T5。

