---
layout: post
title: 'Unsupervised Translation of Programming Languages'
date: 2020-09-12
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f5c2b23160a154a6705a2b1.jpg'
tags: 论文阅读
---

> TransCoder：无监督的编程语言转换模型.

- paper：Unsupervised Translation of Programming Languages
- arXiv：[link](https://arxiv.org/abs/2006.03511)

# 1. 问题背景
该论文实现了一个**transcompiler**的模型，其作用是将一种编程语言写成的代码转换为另一种编程语言，并保持其原始功能不变。它与传统**compiler**不同，是**high-level to high-level**的，中间不会经过低级语言（例如汇编语言、机器语言）的过渡。该模型可以用于代码在不同平台之间的迁移，或用于淘汰过时的编程语言，这些工作用人工来做都非常耗时。

**transcompile**基于机器翻译模型，但编程语言具有与自然语言大不相同的特性，决定了它不能简单地套用现有的机器翻译模型。例如，在自然语言中，偶尔的漏字或赘余并不会影响人的阅读，但对于编程语言来说可能是致命的。

早期的**transcompiler**多使用**rule-based**的方法，即根据每种编程语言的语法规则，对输入的语料建立**抽象语法树（AST）**。但不同的编程语言有不同的语法规则与依赖，编写这些规则是非常耗时耗力的工作。

后来人们考虑使用**NMT模型**，但面临的问题是缺少训练数据。有监督的机器翻译需要大量平行语料。而在编程语言这方面，合格的平行语料并不仅仅需要满足功能相同，还需要有相同的编码思路才能让模型学习到正确的对应规则。这样的语料非常稀少，限制了**NMT模型**在这一领域的表现。

作者提出了**TransCoder**，基于单一语言（**monolingual**，没有对应的平行语料）语料库训练的无监督的**transcompiler**模型，可以实现**c++**、**python**和**java**之间的程序转换。

# 2. 模型介绍
该模型是一个**Seq2Seq**模型，采用**Transformer**结构。模型的训练过程分为三部分：
1. **Cross-lingual Masked Language Model pretraining**
2. **Denoising auto-encoding**
3. **Back-translation**

### （1）Cross-lingual Masked Language Model pretraining

预训练的作用是让模型能够将源码中语义相同的**token**嵌入到空间中相近的向量。这样能够提取源码中的语义信息，而忽略掉表层的实现方式。下图展示的是预训练后不同**token**经过编码器得到的向量经过**t-SNE**之后的可视化：

![](https://pic.downk.cc/item/5f5c32b5160a154a670760b5.jpg)

预训练的策略采用**masked language model**的方法。先在用于训练的数据上随机将一些**token**替换为**MASK**，随后让模型将这些**MASK**恢复为原来的**token**。通过训练时混合使用不同语言的数据，模型能够学习到不同的语言中一些相同的表示。

作者认为，这种预训练过程之所以能发挥作用，依赖的是不同语言中一些具有明显含义的相同**token**，文中称为**anchor point**。在自然语言中，这些**anchor point**大多表现为数字、专有名词等。对于一些较难产生**anchor point**的数据，这种训练方法的表现并不好，例如中英互译的语料，它们本身的字符集都不同，模型难以找到**anchor point**。

然而本实验选用的三种编程语言，**c++**、**python**和**java**，它们本身的保留关键字就有大量重合，非常容易找到大量**anchor point**。这种预训练策略会很自然地建立不同编程语言之间的联系。

下图给出了一个预训练过程的例子：

![](https://pic.downk.cc/item/5f5c3173160a154a67072074.jpg)

### （2）Denoising auto-encoding
预训练对模型做了初始化，能很好地**fit**模型的**encoder**部分，使模型有很好的语义抽取能力。但还需要模型的**decoder**很好地把**encoder**得到的语义信息还原成目标语言的形式。

作者采用了**denoising auto-encoding**的训练方法。仍然是**Seq2Seq**的训练过程，但训练数据被加入一些随机的噪声（例如**mask**掉一些**token**、移除一些**token**或打乱其顺序）。模型需要输出正确的句子，即将输入中的噪声去掉。

在数据开头加了一个特殊的**token**，指定其希望输出的语言形式是**c++**、**python**还是**java**，这样模型就能够完成不同语言之间的转换。

作者强调**denoising**的训练过程是有意义的。因为编程语言的特性就是一个小差错（例如少了一个数字或字母）会很大地影响代码的实际功能。通过这一训练，模型生成的代码在语法上的准确性能够进一步提高，对输入噪音也比较**robust**。

下图给出了一个**denoising**过程的例子：

![](https://pic.downk.cc/item/5f5c343a160a154a6707ae7d.jpg)

### （3）Back-translation
为了进一步提升效果，作者加入了**back-translation**的过程，这是一种弱监督的训练方法。将一种语言的代码通过模型转换成另一种语言，再将其转换回原语言对应的代码。下图给出了一个这种训练过程的例子：

![](https://pic.downk.cc/item/5f5c34a7160a154a6707c853.jpg)

# 3. 实验分析
训练数据是从**Github**上扒下来的代码。需要注意的是训练数据都是单一语言的，没有平行语料。理想情况下模型应该能够训练出**program**级别的**transcompiler**，但难度太大，文中只训练了**function**级别，即在程序中抽取一些函数用来训练。

作者指出，将抽取出的函数分为独立函数和类函数两种。独立函数是指不需要实例化就可以使用的函数，例如**c++**中定义在全局的函数，**Java**中的**static**函数等。预训练使用了全部函数，但后续的训练过程只使用了独立函数。

在抽取数据的时候，函数中的一些**comment**也被保留了。作者进行了是否保留的消融实验：

![](https://pic.downk.cc/item/5f5c35b1160a154a6708155d.jpg)

在对**raw data**进行**tokenize**的时候，针对不同的编程语言使用不同的**tokenizer**。因为不同的语言对很多字符的处理不同，例如在**c++**中，多余的空格、缩进等等都是没有意义的，但在**python**中就一定要把缩进也**tokenize**，不然没有办法获得正确的语法结构。下图是对一个**python**代码做**tokenize**的例子：

![](https://pic.downk.cc/item/5f5c363b160a154a67082f54.jpg)

作者使用了三种评价指标：**BLEU**、**reference match**和**computational accuracy**。其中**BLEU**是机器翻译中常用的指标，**reference match**是完全基于匹配的指标，**computational accuracy**是专门为此任务设计，基于单元测试来衡量两个代码段在功能上相似性的指标。

**computational accuracy**指标对于**transcompiler**来说是最根本的评价标准，因为转换的程序首先要功能正确，其次再考虑美观性和可读性。而**BLEU**衡量了输出的句子与**ground truth**在形式上的相似度。**BLEU**指标虽然比纯粹匹配的**reference match**要科学很多，但它无法很好地评判**transcompiler**的功能。因为相同的功能可能有多样化的实现方法，而表面上相似的代码可能由于一个符号的差别而呈现迥然不同的效果。

下图是在三种指标上的表现，**reference match**指标几乎不能使用，**computational accuracy**指标和**BLEU**指标的趋势是类似的：

![](https://pic.downk.cc/item/5f5c36d9160a154a6708513c.jpg)

模型生成代码时采用**beam search**，即在搜索的时候，只扩展那些概率较大的节点，剪枝掉剩下的节点。具体扩展多少节点由参数beam size决定。实验设置了不同的**beam size** $N$。最终的$N$个结果只要有一个结果能通过单元测试，就算整体通过。

实验**baseline**采用两个已有的**transcompile**框架，一个是**Java**转**python**的**j2py**，另一个是**tangible software solution**里面的**c++**转**java**模块。

贪心的结果（每次取**probability**最大的）低于**baseline**。随着**beam size**的增大，结果的**computational accuracy**也渐渐增大。值得注意的是**Beam 10-Top 1**的结果比起**Beam 10**有大幅度下降，但比纯贪心的**Beam 1**要好一点。作者任务，这说明模型在一定范围内能够生成正确的代码，只是没能给正确的结果分配最高的概率，导致**beam search**的时候选不到。

文中也给出了一些转换失败的案例，如最下面的**python**转到**java**的例子，因为**python**中`min`可以对数组操作，而**java**中`min`只能作用于两个数。

![](https://pic.downk.cc/item/5f5c3924160a154a6708e0f0.jpg)