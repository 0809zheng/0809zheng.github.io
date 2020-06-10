---
layout: post
title: '图像描述'
date: 2020-05-14
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ed50aebc2a9a83be53834d5.jpg'
tags: 深度学习
---

> Image Caption.

**图像描述（image caption）**是自动为图像生成对图像内容的自然语言描述，包括两个子任务：
- 理解图像，正确获取图像相关信息（计算机视觉）
- 基于对图像的理解生成语言描述（自然语言处理）


本文目录：
1. Benchmarks
2. 评价指标
3. 传统方法
4. 深度学习方法

# 1. Benchmarks
图像描述常用的数据集包括：

### Flickr8K
- 网站：[link](https://forms.illinois.edu/sec/1713398)

共8092张图像，有University of Illinois研究人员从Flickr.com收集，通过Amazon Mechanical Turk提供的众包服务获取对应的图像描述，每张图像包含5个不同描述，对图像中人物、目标、场景和活动进行了准确描述。描述平均长度为11.8个单词。

### Flickr30K
- 网站：[link](http://shannon.cs.illinois.edu/DenotationGraph/)

对Flickr8K的扩展，包含31783张图像，158915条描述，其余同上。

### MS COCO
- 网站：[link](http://cocodataset.org/)

MS COCO可用于目标检测、实例分割和图像描述等任务。2014年发布部分包含82783张训练集图像、40504张验证集图像和40775张测试集图像，但是测试集图像描述注释非公开可用，因此大多会对训练集和验证集进行二次划分，而不使用其测试集。

### Visual Genome
- 网站：[link](http://visualgenome.org/)

大规模数据集，包含超过108K张图像和更多图像属性及目标之间的交互关系信息，对于引入了语义关系、空间关系等的图像描述任务，可采用VGG进行预训练。

# 2. 评价指标
图像描述常用的评价指标包括：

### BLEU-{1,2,3,4}
起初用于机器翻译质量评估，核心思想在于“待检测语句越接近参考语句，则越好”。通过对比待检测语句和参考语句在n-gram层面的相似度进行评估，不考虑语法正确性、同义词和相近表达，仅在较短语句下比较可信。

### METEOR
常用于机器翻译评估，首先对待检测语句与参考语句进行对齐（单词精准匹配、snowball stemmer词干匹配、基于WordNet数据集近义词匹配等），然后基于对齐结果计算相似度得分，解决了BLEU存在的一些缺陷。

### CIDEr
针对图像描述任务提出，将每个语句视为一篇文档，表示为tf-idf向量形式，计算待检测语句和参考语句之间的余弦相似度进行评估。

### SPICE
针对图像描述任务提出，基于图的语义表示对描述中的目标、属性和关系进行编码，比之前的基于n-gram的度量方法能更准确的比较模型之间的优劣。

# 3. 传统方法
传统的图像描述方法是指基于**检索(Retrieval-Based)**和基于**模板(Template-Based)**的方法生成图像的描述。
缺陷：基于检索的方法提取出来的描述可能并不完全符合
图像；基于模板的方法生成的图像描述可能显得过于生硬，
缺乏多样性。

### 基于检索的方法

![](https://pic.downk.cc/item/5ed50560c2a9a83be52fbf05.jpg)

**思路**：给定待检索图像$I$，从一系列给定的文本描述库（即数据集：图像+描述）中检索出与该图像$I$最为匹配的一系列图像所对应的描述，并从中再选取最合适的描述作为待检索图像$I$的描述。

**缺陷**：生成的描述质量好坏很大程度上受制于给定的文本描述库，文本描述库由人为建立，因此可保证语句流畅和语法正确；但是文本描述库需要足够大以保证描述内容及语义的准确性，但实际上这种方法往往不能覆盖足够丰富的场景，生成的描述可能并不能正确适应新图像出现的目标和场景。

参考文献：
- Im2text: Describing images using 1 million captioned photographs
- Framing image description as a ranking task: Data, models and evaluation metrics

### 基于模板的方法

![](https://pic.downk.cc/item/5ed50621c2a9a83be530caee.jpg)

**思路**：给定待检索图像$I$，先从图像中检索一些实体目标、属性或语义目标，然后利用指定的语法规则将检索得到的信息进行组合或者将检索到的相关信息填入到预定义的语句模板的空白中，从而得到待检索图像$I$的描述。

**缺陷**：无法生成可变长度的图像描述，限制了不同图像描述之间的多样性，描述显得呆板不自然；另一方面性能也受制于图像目标检测的结果，因此生成的描述可能会遗漏图像细节。

参考文献：
- Composing simple image descriptions using web-scale n-grams
- Baby talk: Understanding and generating simple image descriptions.

# 4. 深度学习方法
受机器翻译Encoder-Decoder模型结构启发，深度学习方法普遍使用CNN作为Encoder对图像进行编码，使用RNN作为Decoder对编码信息进行解码为语言描述，把图像描述任务视为一个从图像语言到自然语言的“翻译”任务。

### show and tell
- paper：[Show and tell: A neural image caption generator](https://arxiv.org/abs/1411.4555)

**Show and Tell**应该是最早将机器翻译中Encoder-Decoder结构应用于图像描述任务中的一项工作。模型结构如图所示:

![](https://pic.downk.cc/item/5ed507a2c2a9a83be5328e28.jpg)

模型流程：
1. 采用卷积神经网络（具体为GoogLeNet Inception V3）作为Encoder部分，将图像编码为固定长度的向量，作为图像特征映射$x_{-1}$。
2. 将图像特征映射$x_{-1}$送入作为Decoder部分的LSTM，逐步生成图像描述。

注意：图像特征映射$x_{-1}$仅在最开始作为LSTM的输入（LSTM的初始状态可设为全零），可视为对LSTM进行初始化计算第一步的状态，而后LSTM的输入均为描述单词的词嵌入向量及上一步的LSTM状态输出。

论文的解释是，在每一步都输入图像特征映射没有得到效果提升，反而容易导致过拟合，但是在此之后一些论文的工作中，又在每个时间步输入了图像特征映射。

### show, attend and tell
- paper：[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

**Show, attend and Tell**是对**show and tell**的一个扩展。在基本的Encoder-Decoder结构上引入了**视觉注意力机制（attentionmechanism）**，可以在Decoder生成图像描述过程中动态关注图像的显著区域。模型结构如图所示:

![](https://pic.downk.cc/item/5ed5085ec2a9a83be533ccf9.jpg)

模型流程：
1. 采用卷积神经网络（具体为VGGNet）作为Encoder部分，将图像编码为$L$个$K$维的向量，每个向量对应图像的一部分区域（实际上就是CNN的中间层响应激活输出，假设其维度为[14, 14, 512]，则$L$=14×14，$K$=512）。
2. 在每一步基于图像特征向量$a$计算该步上下文向量$z_t=\sum_{i=1}^{L} {α_{ti}a_i}$（Soft注意力机制），送入作为Decoder部分的LSTM，逐步生成图像描述。其中$α_t \in \Bbb{R}^L$为第$t$步注意力概率向量，且满足$\sum_{i=1}^{L} {α_{ti}} = 1$。可通过简单的MLP和Softmax激活函数进行计算。

注意：在**Show, attend and tell**模型中，LSTM的输入包含上一步生成描述单词的词嵌入向量、上一步的LSTM状态输出以及基于注意力机制计算的上下文特征向量。此外，在论文中，还介绍了Hard注意力，但在之后的论文中，Soft注意力使用较多。

**Show, attend and Tell**注意力层结构如下图（左）所示。注意力层利用LSTM上一步的状态输出$h_{t-1}$计算上下文向量$z_t$，并作为第$t$步LSTM的输入。也有部分论文进行了改进，他们认为第$t$步的单词输出就应该与LSTM第$t$步的状态输出更相关，因此计算的上下文向量$z_t$直接用于计算单词概率（中）。还有使用多层LSTM用于单词概率生成（右）。

![](https://pic.downk.cc/item/5ed50a08c2a9a83be5368c06.jpg)