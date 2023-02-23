---
layout: post
title: 'A Survey on Visual Transformer'
date: 2021-02-10
author: 郑之杰
cover: 'https://pic.downk.cc/item/5fe9385b3ffa7d37b32a5795.jpg'
tags: 论文阅读
---

> 一篇关于视觉Transformer的综述.

- paper：A Survey on Visual Transformer
- arXiv：[link](https://arxiv.org/abs/2012.12556)

**Transformer**是基于**自注意力机制(self-attention mechanism)**的深度神经网络，该模型在$2017$年$6$月被提出，最早应用于机器翻译任务中。**BERT**于$2018$年$10$月被提出，使用无标签的文本进行模型的双向预训练。$2020$年$5$月提出的**GPT-3**拥有$1750$亿参数，并在自然语言处理任务上取得最好的性能。

![](https://pic.downk.cc/item/5fe9916d3ffa7d37b3c174be.jpg)

该类模型最近被扩展到计算机视觉任务上。作者按照应用场合对这些方法进行分类，分为基础图像分类、**high-level**视觉任务(目标检测、分割和车道线检测)、**low-level**视觉任务(超分辨率、图像降噪和风格迁移)和视频处理。在这些任务上，**Transformer**表现出比**CNN**或**RNN**更具有竞争力的结果。这些方法总结如下：

![](https://pic.downk.cc/item/5fe992c03ffa7d37b3c3b86e.jpg)

# ⚪ Transformer的基本原理
**Transformer**是由一个编码器和一个解码器构成的。编码器是由**自注意力层(self-attention layer)**和**前馈神经网络层(feed-forward neural network)**交替堆叠组成的。解码器是由自注意力层、**编解码注意力层(encoder-decoder attention layer)**和前馈神经网络层组成的。假设每一个**token**嵌入到维度是$d_{model}=512$的向量。

![](https://pic.downk.cc/item/5fe996633ffa7d37b3c974f1.jpg)

![](https://pic.downk.cc/item/5fe9967d3ffa7d37b3c99b6b.jpg)

## ① 自注意力层
自注意力层的结构如上图左所示。输入向量首先被转换为三个不同的向量：**query vector** $q$、**key vector** $k$、**value vector** $v$，其维度都是$d_{q}=d_{k}=d_{v}=d_{model}=512$。将不同输入得到的向量表示成矩阵$Q$、$K$、$V$。

采用下面的步骤计算自注意力：
1. 计算不同输入向量之间的得分$S=Q \cdot K^T$；
2. 为了梯度的稳定性，对得分进行标准化$S_n = \frac{S}{\sqrt{d_{k}}}$；
3. 将得分转化为概率$P=softmax(S_n)$；
4. 计算加权$V$矩阵$Z = V \cdot P$。

上述过程也可以表示为：

$$ Attention(Q,K,V) = softmax(\frac{Q \cdot K^T}{\sqrt{d_{k}}}) \cdot P $$

解码器中的编解码注意力层与上述自注意力层结构相同，其$K$和$V$是编码器的输出，$Q$是解码器中上一层的输出。

注意到上述过程与每个**token**的位置无关，缺乏捕捉位置信息的能力。为此引入**位置编码(position encoding)**加到输入向量中。用$pos$表示**token**在整个输入中的位置，$i$表示位置编码当前的维度：

$$ PE(pos,2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) $$

$$ PE(pos,2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) $$

## ② 多头自注意力
**多头自注意力(multi-head attention)**能提高自注意力层的表现，其结构如上图右所示。设有$h=8$个**head**，输入向量被转换为三组不同的向量，其维度减少为$d_{q}=d_{k}=d_{v}=\frac{d_{model}}{h}=64$。进而表示成矩阵$$\{Q_i\}_{i=1}^{h}$$、$$\{K_i\}_{i=1}^{h}$$、$$\{V_i\}_{i=1}^{h}$$，则多头自注意力表示为：

$$ MultiHead(Q',K',V') = Concat(head_1,...,head_h)W^o $$

$$ head_i = Attention(Q_i,K_i,V_i) $$

其中$W^o \in \Bbb{R}^{d_{model} \times d_{model}}$是线性映射矩阵。

## ③ 其他部分

### Residual in the encoder and decoder
编码器和解码器的每个子层还引入了残差连接，并使用**LayerNorm**：

$$ LayerNorm(X+Attention(X)) $$

### Feed-forward neural network
自注意力层后使用了前馈神经网络层。具体地，其由两层线性转换层和一个**ReLU**激活函数组成：

$$ FFNN(X) = W_2 \sigma (W_1X) $$

隐藏层维度设为$d_{h}=2048$。

### Final layer in decoder
解码器的最后一层将堆叠的向量转化成**token**。由一个线性层和**softmax**函数组成。线性层将向量转化为维度$d_{word}$的**logits**，**softmax**函数进一步将其转化为概率。其中$d_{word}$是字典中的**word**数量。

# ⚪ Self-attention for Computer Vision
计算机视觉中的自注意力机制是一种**非局部滤波(non-local filtering)**操作。假设输入图像为$X \in \Bbb{R}^{n \times c}$，其中$n=h \times w$代表特征的像素数，$c$代表通道数。则输出信号$y_i$计算为：

$$ y_i = \frac{1}{C(x_i)} \sum_{j}^{} f(x_i, x_j)g(x_j) $$

其中$x_i \in \Bbb{R}^{1 \times c}$和$y_i \in \Bbb{R}^{1 \times c}$表示输入和输出信号的第$i$个位置。函数$f(\cdot)$计算两个因子的相似程度，$g(\cdot)$计算嵌入的特征表示。输出被权重因子$C(x_i)$标准化。

函数$f(\cdot)$可选**Gaussian**函数：

$$ f(x_i, x_j) = e^{\theta(x_i) \phi(x_j)^T} $$

其中$\theta(\cdot)$和$\phi(\cdot)$也表示嵌入层。若选择线性嵌入：$\theta(X)=XW_{\theta}$,$\phi(X)=XW_{\phi}$,$g(X)=XW_{g}$，其中$W_{\theta} \in \Bbb{R}^{c \times d_k}$,$W_{\phi} \in \Bbb{R}^{c \times d_k}$,$W_{g} \in \Bbb{R}^{c \times d_v}$；设置归一化因子$C(x_i)=\sum_{j}^{} f(x_i, x_j)$。则输出信号$y_i$也可表示为：

$$ y_i = \frac{1}{\sum_{j}^{} e^{x_iw_{\theta,i}w_{\phi,j}^Tx_j^T}} \sum_{j}^{} e^{x_iw_{\theta,i}w_{\phi,j}^Tx_j^T}x_jw_{g,j} $$

或写作矩阵形式：

$$ Y=softmax(XW_{\theta}W_{\phi}^{T}X)g(X) $$

与**Transformer**中的自注意力相比，可以得到关系：$W_q=W_{\theta}$,$W_k=W_{\phi}$,$W_v=W_{g}$。因此计算机视觉中的自注意力也可以表示成：

$$ Y=softmax(QK^T)V=Attention(Q,K,V) $$

通常最后一层输出引入残差连接，最终表示为：

$$ Z=YW_Z+X $$

上述自注意模块通常用作卷积神经网络结构的组成部分，它对较大感受野具有较小的缩放特性。这种结构常用在网络的顶部，以捕获计算机视觉任务的长程交互关系。

# ⚪ Transformer in NLP
**RNN**是**NLP**领域最常用的模型结构之一。这类模型的信息流需要从前一个隐状态顺序传递到下一个隐状态，从而阻碍了模型的加速和并行化。**Transformer**可以大规模并行计算，并表现出超越**RNN**的性能；这也导致了**预训练模型(pre-trained model, PTM)**的兴起。

### BERT
**BERT**是一种多层**Transformer**编码器结构。在预训练时进行两个任务：
1. **Masked language modeling**：随机遮挡输入的一些**token**，训练模型去预测它们；
2. **Next sentence prediction**：用一对句子作为输入，预测这两个句子是否具有连接关系。

经过预训练，在**BERT**后面加一层输出层微调便可以应用于不同的下游任务中。如使用第一个**token**做分类进行情感分析等**sequence-level**的任务；使用所有**token**进行**softmax**进行名称实体识别等**token-level**的任务。

### GPT
**GPT**是一种多层**Transformer**解码器结构。与**BERT**不同，**GPT**是一个单向语言模型，适用于语言理解和生成等任务。

一些常用的**Transformer**模型如下图所示：

![](https://pic.downk.cc/item/5fea81693ffa7d37b3ab2cfa.jpg)


# ⚪ Transformer in CV
计算机视觉中使用的**Transformer**主要使用其编码器模块。它可以看作一种新的特征选择器。相比于仅关注局部特征的**CNN**，其能捕获长距离的特征，更容易利用全局信息；相比于必须顺序计算隐状态的**RNN**，其自注意力层和全连接层可以并行计算，易于加速。

通常认为图像比文本具有更高的维度、更多的噪声和更多的模态，更难建立生成模型。如何将**Transformer**应用于计算机视觉领域仍是值得讨论的问题，作者从图像分类、**high-level**视觉、**low-level**视觉、视频分析和高效**Transformer**等五个方面进行总结。

## ① Image Classification

下面是一些应用于图像分类的**Transformer**模型：
- [iGPT](https://0809zheng.github.io/2020/12/29/igpt.html)
- [ViT](https://0809zheng.github.io/2020/12/30/vit.html)

**iGPT**是使用图像生成作为预训练任务的自监督模型；而**ViT**是使用图像分类作为预训练任务的监督模型。后者在图像分类任务上表现更好。但如何解决**path**内部和不同**path**之间的关联仍是挑战。

上述两模型与卷积神经网络模型**BiT**之间的比较如下表所示：

![](https://pic.downk.cc/item/5febea1e3ffa7d37b3c23963.jpg)


## ② High-level Vision
最近**Transformer**被应用于一些**high-level**的视觉任务，如目标检测、分割、车道线检测。

### Object Detection
下面是一些应用于图像分类的**Transformer**模型：
- [DETR](https://0809zheng.github.io/2020/06/20/detr.html)
- [Deformable DETR](https://0809zheng.github.io/2020/12/31/ddetr.html)
- [End-to-End Object Detection with Adaptive Clustering Transformer](https://arxiv.org/abs/2011.09315)：使用局部敏感哈希(**LSH**)自适应地对查询特征并进行聚类，提高**DETR**的速度。
- [Rethinking transformer-based set prediction for object detection](https://arxiv.org/abs/2011.10881)：只使用**Transformer**的编码器部分进行目标检测。
- [UP-DETR: unsupervised pre-training for object detection with transformers](https://arxiv.org/abs/2011.09094)：使用无监督预训练方法：**random query patch detection**。

### Segmentation

- **Max-DeepLab**：全景分割
- **VisTR**：视频实例分割
- **Cell-DETR**：从显微镜图像中进行细胞分割
- **Point Transformer**：**3D**点云的语义分割

### Lane Detection
- **LSTR**：将车道线检测看作多项式拟合车道的任务，并利用神经网络预测多项式的参数。 

## ③ Low-level Vision
**Low-level**的视觉任务通常将整张图像作为输出(如高分辨率或去噪)，更具挑战性。

- [Image Transformer](https://0809zheng.github.io/2021/02/04/it.html)：图像生成
- TTSR：超分辨率、纹理迁移
- [IPT (Image Processing Transformer)](https://0809zheng.github.io/2020/12/28/deit.html)：超分辨率、去噪、去雨等多任务学习

## ④ Video Processing
视频中包含**空间(spatial)**和**时序(temporal)**维度的信息，可以用**Transformer**处理。

### High-level Video Processing
- **人类动作识别 (Human Action Recognition)**：辨识和定位视频中人类的动作。如 action transformer、temporal transformer。
- **人脸对齐 (Face Alignment)**：定位人脸的面部特征点。如 two-stream transformer。
- **视频检索 (Video Retrieval)**：根据视频的内容检索相似的视频。
- **行为识别 (Activity Recognition)**：辨识一群人的行为。如actor-transformer。
- **视频目标检测 (Video Object Detection)**：如memory enhanced global-local aggregation(MEGA)、spatiotemporal transformer。
- **多任务学习 (Multi-task Learning)**：如video multi-task transformer network。

### Low-level Video Processing
- **视频帧合成 (Frame/Video Synthesis)**：合成两个连续帧之间的插入帧或一个帧序列后的帧。如ConvTransformer、 recurrent transformer。
- **视频修补 (Video Inpainting)**：修补视频帧中缺失的区域。如spatial-temporal transformer network。

### Multimodality
- **视频总结 (Video Captioning/Summarization)**：根据视频的内容生成描述的文本。

## ⑤ Efficient Transformer
尽管**Transformer**获得巨大的成功，但其高内存占用和巨大的资源消耗使其无法部署在资源受限的设备中。常用的压缩**Transformer**模型的方法，包括网络剪枝、知识蒸馏、量化和结构设计。下表列出一些压缩后的**Transformer**模型：

![](https://pic.downk.cc/item/5ff2c5993ffa7d37b302da56.jpg)

