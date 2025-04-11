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

本文首先介绍语言的特征表示，然后介绍预训练语言模型的发展，最后尝试理解预训练语言模型。

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

[<font color=Blue>On the Role of Bidirectionality in Language Model Pre-Training</font>](https://0809zheng.github.io/2022/07/12/plmrole.html)一文指出，如果是以**fine-tuning**方式解决下游任务，编码端架构效果更好；若是以**zero shot/few shot prompting**这种模式解决下游任务，解码端架构效果更好。这是因为解码端架构能够直接生成完整的序列，在少样本范式下更具优势；而编码端架构需要额外的推理步骤来处理**masked token**，在微调范式下能够充分利用上下文信息。

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
| [Gopher](https://0809zheng.github.io/2021/12/30/gopher.html) | Transformer解码器 | LM | $44$M-$280$B |
| [Jurassic-1](https://0809zheng.github.io/2021/12/31/jurassic1.html) | Transformer解码器 | LM | $7$B-$178$B |

# 3. 理解预训练语言模型

### (1) 预训练语言模型学到了哪些知识？

预训练语言模型从文本数据中学习到的知识包括语言类知识和世界知识两大类。
- **语言类知识**是指词法、词性、句法、语义等有助于人类或机器理解自然语言的知识，又包括浅层语言知识和抽象语言知识。
1. **浅层语言知识**是指词法、词性、句法等知识，通常存储在**Transformer**的低层和中层；
2. **抽象语言知识**是指语义类知识，通常存储在**Transformer**的中层和高层。
- **世界知识**是指真实事件或常识等有助于人类或机器理解真实世界的知识，又包括事实型知识和常识性知识。这类知识主要分布在**Transformer**的中层和高层，尤其聚集在中层。
1. **事实型知识 (Factual Knowledge)**是指在这个世界上发生的一些真实事件，如“特朗普是现任美国总统”（这类知识可能会失效！）。
2. **常识性知识 (Common Sense Knowledge)**是指这个世界存在的生活常识和规律，如“太阳从东方升起”。

[<font color=Blue>BERTnesia: Investigating the capture and forgetting of knowledge in BERT</font>](https://0809zheng.github.io/2021/06/26/bertnesia.html)一文指出，预训练语言模型学习到的世界知识不仅存储在最后一层，中间层也贡献了大量知识；并且随着模型层深增加，能够学习到的世界知识数量逐渐以指数级增加。在对模型进行微调时，世界知识可能会被遗忘，遗忘程度取决于微调目标和训练数据。

[<font color=Blue>When Do You Need Billions of Words of Pretraining Data?</font>](https://0809zheng.github.io/2021/03/31/plmdata.html)一文指出，仅需约**1000**万至**1**亿词汇的预训练数据即可学习到可靠的语言类知识，但要掌握典型的世界知识则需要数十亿词汇的数据。大型预训练模型在大规模数据上性能提升的主要驱动力是世界知识。

### (2) 预训练语言模型如何存储知识？

预训练语言模型的知识存储在**Transformer**的模型参数里。从**Transformer**的结构看，模型参数由两部分构成：自注意力层（约占总参数的三分之一）和全连接层（约占总参数的三分之二）。自注意力层主要用于计算**token**或知识间的相关性，并对全局信息进行整合，更可能是在建立知识之间的联系，大概率不会存储具体的知识点；则可以推论出模型的知识主体是存储在全连接层结构中。

[<font color=Blue>Transformer Feed-Forward Layers Are Key-Value Memories</font>](https://0809zheng.github.io/2021/05/05/kvm.html)一文指出，把全连接层看作键-值记忆单元（$FF(x)=f(x⋅K^\top )⋅V$），其中第一层的参数$K$作为输入序列的模式检测器，第二层的参数$V$存储了对应模式下输出词汇表上的概率分布。

![](https://pic1.imgdb.cn/item/67f784d088c538a9b5c87c5b.png)

[<font color=Blue>Knowledge Neurons in Pretrained Transformers</font>](https://0809zheng.github.io/2021/05/06/knowledgeneuron.html)一文进一步提出了“知识神经元”的概念，并采用一种基于集成梯度的知识归因方法来识别表达特定知识的神经元。给定一个输入提示 $x$ 和一个关系事实 $\langle h, r, t \rangle$（由已知词、关系词和目标词向量构成的三元组），模型的输出 $P_x(\hat{w}^{(l)}_i)$ 定义为预训练模型预测正确答案 $y^*$ 的概率：

$$ P_x(\hat{w}^{(l)}_i) = p(y^* | x, w^{(l)}_i = \hat{w}^{(l)}_i) $$

为了计算神经元 $w^{(l)}_i$ 的归因分数 $\text{Attr}(w^{(l)}_i)$，从 $w^{(l)}_i = 0$ 到 $w^{(l)}_i$ 的原始值，逐步计算梯度并进行积分：

$$ \text{Attr}(w^{(l)}_i) = w^{(l)}_i \int_{0}^{1} \frac{\partial P_x(\alpha w^{(l)}_i)}{\partial w^{(l)}_i} \, d\alpha $$

按照上述计算识别出归因分数大于某个阈值 $t$ 的神经元可以作为粗略的知识神经元集合。通过保留同一个事实的不同提示中广泛共享的神经元，可以进一步过滤掉“假阳性”神经元。

![](https://pic1.imgdb.cn/item/67f78e5188c538a9b5c88abd.png)

### (3) 预训练语言模型如何修改知识？

预训练语言模型中存储的事实型知识可能会过时（如美国总统换届），因此修正**LLM**模型里存储的错误或者过时的知识是有必要的。下面介绍三种修改知识的方法。

### ① 更换训练数据

假设想要删除某一类知识，可以定位并删除对应的数据源，然后重新预训练整个模型。由于模型预训练的成本太高。所以这种方法比较适合对于某个特定类别数据的一次性大规模删除场合（如去除偏见和毒性等内容的处理），不适合少量多次的常规知识修正场景。

实现该功能要求对于指定的某条知识，可以定位到是哪些训练数据导致**LLM**学会了这条知识，即实现数据归因（**Data Attribution**）功能。[<font color=Blue>Towards Tracing Factual Knowledge in Language Models Back to the Training Data</font>](https://0809zheng.github.io/2022/07/13/tda.html)一文设计了三种用于事实追踪（识别哪些训练样本教会了语言模型生成特定的事实性断言）的数据归因方法：
- 梯度归因方法**TracIn**：在训练过程中，每当对训练样本 $z$ 进行梯度更新时，记录测试样本 $z_{\text{query}}$ 的损失变化；通过内积来估计影响力：

$$
I_t(z, z_{\text{query}}) = \nabla_{\theta} L(z_{\text{query}}, \theta_t)^\top \nabla_{\theta} L(z, \theta_t)
$$

- 嵌入归因方法：从**Transformer**语言模型中提取中间层的输出，通过余弦相似度计算训练样本 $z$ 和测试样本 $z_{\text{query}}$的关联性：

$$
I(z, z_{\text{query}}) = \frac{\text{LM}_{\text{inter}}(z)^\top \text{LM}_{\text{inter}}(z_{\text{query}})}{\|\text{LM}_{\text{inter}}(z)\| \|\text{LM}_{\text{inter}}(z_{\text{query}})\|}
$$

- 信息检索方法**BM25**：通过计算训练样本 $z$ 和测试样本 $z_{\text{query}}$之间的词项重叠来选择支持样本：

$$
I(z, z_{\text{query}}) = \sum_{t \in z_{\text{query}}} \log \left( \frac{N + 1}{N_t} \right) \times \left( \frac{(k_1 + 1) \cdot f(z, t)}{k_1 \cdot \left( (1 - b) + b \cdot \frac{L(z)}{L_{\text{avg}}} \right) + f(z, t) + 1} \right)
$$

### ② 微调LLM

可以根据要修正的新知识来构建微调数据集，然后微调预训练的**LLM**模型。这个方法会带来灾难性遗忘问题，即模型可能会遗忘掉一些不应该遗忘的知识。

[<font color=Blue>Modifying Memories in Transformer Models</font>](https://0809zheng.github.io/2022/07/14/modifymem.html)一文提出了一种约束优化方法，可以在不降低**Transformer**模型对未修改事实性能的前提下，显式修改模型中特定的事实性知识。给定一个预训练的**Transformer**模型，其参数为$θ_0$，存储了一系列事实$F$。目标是将$F$中的一小部分事实$S$替换为新的事实$M$，得到新的模型参数$θ^{new}$，使其存储$F^′ = (F \backslash S) ∪ M$。优化目标为：

$$
\begin{aligned}
\min_{\theta \in \Theta} \quad  & \frac{1}{m} \sum_{x \in D_M} L(x; \theta) \\
\text{subject to} \quad & \|\theta - \theta_0\|_\infty \leq \delta
\end{aligned}
$$

上述优化问题可以通过投影梯度下降求解：
1. 使用预训练模型初始化参数$θ_0$。
2. 在每个迭代中，计算梯度并更新参数。
3. 将更新后的参数投影到约束集合内，确保参数变化不超过$δ$。

### ③ 修改LLM的模型参数


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
- [<font color=Blue>When Do You Need Billions of Words of Pretraining Data?</font>](https://0809zheng.github.io/2021/03/31/plmdata.html)：(arXiv2011)什么时候需要数十亿单词的预训练数据？
- [<font color=Blue>Transformer Feed-Forward Layers Are Key-Value Memories</font>](https://0809zheng.github.io/2021/05/05/kvm.html)：(arXiv2012)Transformer全连接层是键值记忆单元。
- [<font color=Blue>Modifying Memories in Transformer Models</font>](https://0809zheng.github.io/2022/07/14/modifymem.html)：(arXiv2012)修正Transformer模型中的记忆。
- [<font color=Blue>Knowledge Neurons in Pretrained Transformers</font>](https://0809zheng.github.io/2021/05/06/knowledgeneuron.html)：(arXiv2104)预训练Transformer中的知识神经元。
- [<font color=Blue>BERTnesia: Investigating the capture and forgetting of knowledge in BERT</font>](https://0809zheng.github.io/2021/06/26/bertnesia.html)：(arXiv2106)BERTnesia：探究 BERT 中知识的捕获与遗忘。
- [<font color=Blue>Scaling Language Models: Methods, Analysis & Insights from Training Gopher</font>](https://0809zheng.github.io/2021/12/30/gopher.html)：(arXiv2112)扩展语言模型：训练 Gopher 的方法、分析和见解。
- [<font color=Blue>Jurassic-1: Technical details and evaluation</font>](https://0809zheng.github.io/2021/12/31/jurassic1.html)：(AI21 Labs)Jurassic-1：技术细节与评估。
- [<font color=Blue>On the Role of Bidirectionality in Language Model Pre-Training</font>](https://0809zheng.github.io/2022/07/12/plmrole.html)：(arXiv2205)探讨语言模型预训练中的双向性。
- [<font color=Blue>Towards Tracing Factual Knowledge in Language Models Back to the Training Data</font>](https://0809zheng.github.io/2022/07/13/tda.html)：(arXiv2205)将语言模型中的事实知识追溯到训练数据。

