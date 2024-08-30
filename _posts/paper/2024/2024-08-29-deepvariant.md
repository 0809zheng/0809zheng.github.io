---
layout: post
title: 'A universal SNP and small-indel variant caller using deep neural networks'
date: 2024-08-29
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/66d0605fd9c307b7e91b1e7b.png'
tags: 论文阅读
---

> 使用深度神经网络的通用单核苷酸多态性和插入/缺失变异比对器.

- paper：[A universal SNP and small-indel variant caller using deep neural networks](https://www.nature.com/articles/nbt.4235)
- code：[DeepVariant](https://github.com/google/deepvariant)

# 1. 背景介绍

单核苷酸多态性（**Single Nucleotide Polymorphism，SNP**）是指基因组中单个核苷酸腺嘌呤(**A**)、胸腺嘧啶(**T**)、胞嘧啶(**C**)或鸟嘌呤(**G**)在个体之间或个体配对染色体之间的差异，是造成基因组多样性的最常见的**DNA**序列变异。

插入缺失（**insertion-deletion，InDel**）是指在**DNA**序列中添加或删除少量碱基，主要指在基因组某个位置上发生较短长度的线性片段插入(**Insert**)或者缺失(**Deletion**)的现象，也是一种常见的**DNA**序列变异。

![](https://pic.imgdb.cn/item/66d03dffd9c307b7e9efeaa1.png)

**SNP**和**INDEL**变异检测有助于我们更深入地了解基因组，生物性状的表现，物种的起源与进化，认识基因变异和疾病的之间的联系。从测序数据中进行准确的变异检测也是生物学、医学研究和精准医学的基础。

# 2. DeepVariant

**DeepVariant**是由**Google Brain**开发的一个开源软件，用于对基因组序列进行变异检测。它利用深度学习技术来实现高质量、高准确性的变异检测，能够在人类基因组的全长上实现高达99％的准确率。

**DeepVariant**使用卷积神经网络（**CNN**）来对原始**DNA**测序数据进行分析，并识别基因组中的单核苷酸变异（**SNP**）和插入/缺失（**indel**）。**DeepVariant**的基本工作流程有三个步骤：
1. **make_examples**: 将测序数据（经过比对的**bam**文件）中可能存在突变的候选位点编码成**pileup image**。
2. **call_variants**: 调用预训练的**Google Inception V2**模型，输出候选位点对应的三种不同基因型的概率。
3. **postprocess_variants**: 对候选位点进行合并处理，输出最终的突变检测**VCF**文件。

![](https://pic.imgdb.cn/item/66d03f3dd9c307b7e9f0fa2d.png)

**pileup image**是根据**reads**提供的突变个数生成的图像，候选点位始终位于图像的中心。比如候选位点参考基因组碱基为**A**，测序得到的**reads**显示该位点含有**G**和**T**两种，因此可能存在3种等位基因(**{G,T,G&T}**)，因此生成3张图像。并结合真实的突变信息进行标注，一共有3种标签：
- **0/homozygous reference**：纯合子并且和参考基因组相同；
- **1/heterozygous**：杂合子；
- **2/homozygous alternative**：纯合子但与参考基因组不同。

由于全基因组测序数据的平均覆盖深度一般不会超过$100$，因此**pileup image**图像高度设置为$100$；对于测序覆盖度大于$100$的区域，则通过下采样技术将其覆盖度降低至$100$以满足图像生成的要求。此时最多可编码$95$个**read**，顶部的$5$行保留用于参考序列。

现有主流测序平台产生的测序数据读长在$100bp$左右，对于候选的变异位点，向基因组两端延伸$100bp$即可将大部分覆盖待检位点的测序数据包含进来；考虑到插入缺失变异的长度，为其预留$20bp$的编码空间，因此**pileup image**图像宽度设置为$221$。

**pileup image**图像编码为$6$个通道：
1. **read base**：碱基类型，不同强度代表**A,C,G,T**
2. **base quality**：碱基的测序质量，由测序仪设定，数值越大表示质量较高
3. **mapping quality**：**read**的比对质量，由对齐器设置，数值越大表示质量较高
4. **strand**：**read**链的方向；灰色为正向，白色为反向
5. **read supports variant**：**read**是否支持变异；白色表示支持给定的替代等位基因，灰色表示不支持
6. **base differ from ref**：**read**链与**ref**链对应位置碱基是否相同；白色表示与参考值不同，灰色表示与参考值匹配

![](https://pic.imgdb.cn/item/66d05b05d9c307b7e9165d8f.png)

**DeepVariant**采用的**CNN**网络的输入层尺寸是$299×299$，因此需要对原始图像($221×100×6$)的空间尺寸进行缩放。