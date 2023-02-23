---
layout: post
title: 'Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax'
date: 2021-06-14
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60f524c25132923bf824e472.jpg'
tags: 论文阅读
---

> BAGS：按照类别样本的量级对长尾数据集进行分组分类.

- paper：Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax
- arXiv：[link](https://arxiv.org/abs/2006.10408)

作者认为在长尾分布中分类器(通常是最后一个全连接层)参数的**范数(norm)**并不是均匀分布的，而是和类别的出现频率成正相关，即具有样本数更多的头部类对应的分类部分会有更大的范数。本文提出了按类别样本数进行分组分类的方法**Balanced Group Softmax (BAGS)**，仅对同一量级的类别**logits**计算**softmax**，而不是对所有类别一起计算**softmax**。

作者首先使用两阶段检测器**Faster R-CNN**进行分析。把**Faster R-CNN**解耦为特征提取模块和分类模块(最后一层全连接层)，分析模型在平衡数据集**COCO**(下图粉色曲线)和不平衡数据集**LVIS**(下图绿色曲线)上不同输出类别的**样本数**以及对应的**全连接层权重范数**的大小。可以看出，在类别均衡的数据集上训练的模型除类别$0$(背景)外在不同类别中的表现比较稳定(包括检测出的目标数量以及对应的权重范数)；而在长尾数据集上训练后模型具有明显的长尾效应，即样本数越小的类别对应的权重范数越小，不同类别对应的分类器的权值严重失衡，尾部类别被激活的机会很少。

![](https://pic.imgdb.cn/item/60f52a3a5132923bf8524f69.jpg)

本文主要讨论**目标检测**中的长尾问题。通常的长尾问题可以采用重采样或重加权的解决办法。然而重采样的方法会增加检测任务的训练时间，并增加尾部类别过拟合的风险；重加权的方法对超参数的选择敏感，并且难以处理目标检测中的“背景”类别(通常具有最多的样本数)。在两阶段的目标检测任务中，**backbone**网络$f_{\text{back}}$从输入图像$I$中提取特征图$F=f_{\text{back}}(I)$，根据**proposal** $b_k$从特征$F$中提取经过对齐的对应特征$F_k=\text{ROIAlign}(F,b_k)$，使用分类头$f_{\text{head}}$从中提取类别特征$h=f_{\text{head}}(F_k)$，并通过全连接层输出长度为$C+1$($C$个类别+背景)的特征**logits** $z=Wh+b$。经过**softmax**函数进行归一化得到每个类别$j$对应的概率$p_j=\text{softmax}(z_j)=\frac{e^{z_j}}{\sum_{i=0}^{C}e^{z_j}}$，并计算交叉熵损失$$\mathcal{L}_k(p,y)=-\sum_{j=0}^{C}y_j \log(p_j)$$。

![](https://pic.imgdb.cn/item/60f52e7a5132923bf8751d07.jpg)

作者按照类别的样本数对这些类别进行分组，将其划分为互补的$n+1$组(即分成一个背景组,若干个尾部类别组和一个头部类别组)。在论文中分成$5$组，类别间隔设置为$s_1^l=0,s_2^l=10,s_3^l=10^2,s_4^l=10^3,s_5^l=+∞$。由于不同组之间的类别是互斥的，每个组会有至少一个得分较高的类别，导致出现大量**false positive**，因此每个组内再引入一个**others**类表明该样本属于其他组的概率。最终模型输出的**logits** $z \in \Bbb{R}^{(C+1)\times(n+1)}$。对于类别$j$，它在$n+1$个分组中都有对应的概率(在某个组是类别$j$，在其余组是**others**)，在计算损失函数时对每个组分别计算：

$$ \mathcal{L} = -\sum_{n=0}^{N} \sum_{i \in \mathcal{G}_n}^{} y_i^nlog(p_i^n) $$

在测试时，首先生成预测**logits** $z \in \Bbb{R}^{(C+1)\times(n+1)}$。分组通过**softmax**归一化，忽略除背景组外的所有组的**others**，将这些组内的类别得分按照原始类别**ID**进行排序，将这些得分乘以背景组中的**others**得分得到每个类别的实际得分，与背景得分一起作为最终的预测结果。

实验是基于**Faster R-CNN**模型在**LVIS**数据集上进行的，对比其他长尾分布方法：

![](https://pic.imgdb.cn/item/60f53f985132923bf8de9fc0.jpg)

分析**BAGS**对分类器的权值的影响，应用该方法后不同类别对应的分类器权值分布情况得到改善：

![](https://pic.imgdb.cn/item/60f53fcc5132923bf8dfc35f.jpg)