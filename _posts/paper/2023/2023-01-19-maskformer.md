---
layout: post
title: 'Per-Pixel Classification is Not All You Need for Semantic Segmentation'
date: 2023-01-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6422b1f2a682492fcc95eb45.jpg'
tags: 论文阅读
---

> MaskFormer：逐像素分类并不是语义分割所必需的.

- paper：[Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278)

图片语义分割(**semantic segmentation**)问题一直以来都被当做一个像素级分类(**per-pixel classification**)问题解决的。本文作者指出把语义分割看成一个**mask classification**问题不仅更自然的把语义级分割(**semantic-level segmentation**)和实例级分割(**instance-level segmentation**)联系在了一起，并且在语义分割上取得了比像素级分类方法更好的结果。作者提出了一种简单的**MaskFormer**方法，可以将现有的任意基于像素分类的算法无缝转换成**mask**分类算法。**mask**分类预测一组二值掩码，并为每个掩码分配一个类。

![](https://pic.imgdb.cn/item/642e78f4a682492fcccefaa6.jpg)

对于一个$H\times W$大小的输入图像，掩膜分类任务可以被分成2个任务：
1. 将图像划分为$N$个区域（$N$不需要等于$K$），用二值掩膜表示。
2. 对每个区域作为一个整体划分到$K$个类别中，注意，允许多个区域划分成相同类别，使得该算法能应用到语义和实例级分割任务中。

为了训练模型，需要计算预测值和真实值之间的匹配度。假设预测的结果为$(p_i,m_i)_{i=1}^N$，其中$p_i$是第$i$个区域的预测类别，$$m_i=\{0,1\}^{H\times W}$$是第$i$个区域的二值掩膜。真实标签为$(c_i^{gt},m_i^{gt})_{i=1}^{N^{gt}}$。通常$N > N^{gt}$，并为真实值填充一组背景。第 $i$ 个预测与具有类别标签 $i$ 的真实区域相匹配，如果预测区域$i$的类别在真实标签中不存在，则与背景匹配。

$$
\mathcal{L}_{\text {mask-cls }}\left(z, z^{\mathrm{gt}}\right)=\sum_{j=1}^N\left[-\log p_{\sigma(j)}\left(c_j^{\mathrm{gt}}\right)+\mathbb{1}_{c_j^{\mathrm{gt}} \neq \varnothing} \mathcal{L}_{\text {mask }}\left(m_{\sigma(j)}, m_j^{\mathrm{gt}}\right)\right]
$$

![](https://pic.imgdb.cn/item/642e7af5a682492fccd18af3.jpg)

### ⚪ 像素级模块

输入图像（$H\times W$）在经过骨干网络之后，通常都会得到低分辨率的特征图，像素级模块中的**pixel decoder**模块会将特征图上采样到$H\times W$大小，注意，任何基于像素分类的分割模型的骨干网络都适合像素级模块设计，包括最近的基于 **Transformer** 的模块。**MaskFormer** 将此类模型无缝转换为掩膜分类模型。

### ⚪ Transformer模块

**Transformer** 模块使用标准的 **Transformer** 解码器 来处理图像特征和 $N$ 个可学习的位置**embedding**（即**query**），其输出是 $N$ 个特征向量，分别对应$N$个可能存在的分割区域的全局信息。

### ⚪ 分割模块

**Transformer** 模块输出的$N$ 个特征向量可以分别用于生成 $N$ 个区域的二值掩码和预测类别。对于预测类别，在**sofmax**之后使用线性分类器，以产生每个分割类别的概率预测。对于二值掩码，采用**MLP**将分割**embedding**转换成**mask embedding**，最后通过对**mask embedding**和像素级模块输出的**pixel embedding**进行相乘并应用**sigmoid**激活得到与原图尺寸相同的二值预测值。

### ⚪ 掩膜分类推理

像素级分类的分割模型的分割推理是将图像按像素对每个像素值划分到$N$个类别中的某一类，划分的方式是先将每个像素计算$N$个类别的预测概率，然后用**argmax**函数求$N$个可能性的最大值。

对于掩码级的分割模型，已经生成了$N$个具有不同类别的掩码。对语义分割来说，共享同一预测类别的不同掩码可以合并；对实例分割来说，这些分割块的标签不合并。