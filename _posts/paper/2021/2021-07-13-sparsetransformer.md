---
layout: post
title: 'Generating Long Sequences with Sparse Transformers'
date: 2021-07-13
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60ed15cd5132923bf8e08880.jpg'
tags: 论文阅读
---

> Sparse Transformer：使用稀疏注意力的Transformer.

- paper：Generating Long Sequences with Sparse Transformers
- arXiv：[link](https://arxiv.org/abs/1904.10509)

在标准的**Transformer**中，自注意力机制的整体运算复杂度为$O(N^2)$，这是因为对于长度为$N$的输入序列，其每一个位置都会和该序列的所有位置进行交互并计算注意力(相关度)，从而得到$N^2$大小的注意力矩阵。该矩阵的第$i$列代表第$i$个输出位置与所有输入位置的(未归一化)相关性。

![](https://pic.imgdb.cn/item/60ed16a45132923bf8e602d3.jpg)

作者提出了**Sparse Transformer**，将**Transformer**中自注意力矩阵稀疏化，假设序列具有**局部紧密相关**和**远程稀疏相关**性，即在每个序列位置上只保留相对距离不超过$k$以及相对距离为$k,2k,3k,...$的自注意力运算：

![](https://pic.imgdb.cn/item/60ed16c85132923bf8e6e8ad.jpg)

上述稀疏注意力可以被进一步拆分成两部分。第一部分是一种局部的自注意力，即每个序列位置只于其一个邻域内的位置进行交互。这种做法类似于卷积神经网络中的普通卷积，若保留相对距离不超过$k$的位置，则每个位置只跟其邻域内的$2k+1$个位置(包括其自身)计算相关性，使得运算复杂度降低为$O((2k+1)N)$。但这种注意力失去了长程关联性，只能捕捉局部相关性。

![](https://pic.imgdb.cn/item/60ed17095132923bf8e886a0.jpg)

另外一部分做法则类似于卷积神经网络中的**空洞(Atrous)**卷积。即每个序列位置只跟与其相对距离为$k,2k,3k,...$的位置进行自注意力运算，使得运算复杂度降低为$O(N^2/k)$。这种注意力机制能够捕捉全局的稀疏关联性。

![](https://pic.imgdb.cn/item/60ed16ea5132923bf8e7c320.jpg)

这种稀疏自注意力也可以被用于图像生成任务中(对于序列生成任务，需要对注意力矩阵的一半进行**mask**，即每个输出位置只能从之前的输入位置处获取信息)。对于普通的自注意力，每个输出位置和之前所有输入位置计算相关性；对于**Sparse Transformer**提出的**strided**注意力，每个输出位置只与之前输入的一部分进行相关性计算；作者还额外提出了一种**fixed**注意力，即预先指定与每个输出位置进行相关性计算的位置。

![](https://pic.imgdb.cn/item/60ed1c745132923bf80f980e.jpg)

实验表明该稀疏注意力机制在多种任务上都取得不错的结果：

![](https://pic.imgdb.cn/item/60ed1cfd5132923bf813b71f.jpg)