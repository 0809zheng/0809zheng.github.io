---
layout: post
title: 'DivideMix: Learning with Noisy Labels as Semi-supervised Learning'
date: 2022-09-13
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63bf7457be43e0d30e43c5fe.jpg'
tags: 论文阅读
---

> DivideMix：通过噪声标签实现半监督学习.

- paper：[DivideMix: Learning with Noisy Labels as Semi-supervised Learning](https://arxiv.org/abs/2002.07394)

**DivideMix**把半监督学习和噪声标签学习(**Learning with noisy labels, LNL**)结合起来，通过高斯混合模型根据每个样本的损失值动态地把训练数据分成包含干净样本的标注数据集和包含噪声样本的未标注数据集。

具体地，根据每个样本$x_i$的交叉熵损失$l_i=y_i^T \log f_{\theta}(x_i)$，通过$z=2$的高斯混合模型把样本集分成两份，干净样本应该比噪声样本具有更低的损失值。高斯混合模型中具有更小均值的簇被看作干净样本，其聚类中心为$c$。把高斯混合模型的后验概率$w_i=p_{GMM}(c\|l_i)$作为采样结果为干净样本的概率，当其超过阈值$\tau$时把样本视为干净样本，否则视为噪声样本。该数据聚类过程被称为**co-divide**。

为了避免半监督学习的确认偏差，**DivideMix**同时训练了两个独立的网络，每个网络使用根据另一个网络预测结果划分的数据集。

![](https://pic.imgdb.cn/item/63bf7560be43e0d30e454baa.jpg)

**DivideMix**采用[<font color=blue>MixMatch</font>](https://0809zheng.github.io/2022/09/11/mixmatch.html)方法的训练过程，在此基础上引入如下改进：
- 干净样本的标签**co-refinement**：根据另一个网络预测为干净样本的概率$w_i$对真实标签$y_i$和多次数据增强后的预测结果均值$$\hat{y}_i$$进行加权；
- 噪声样本的标签**co-guessing**：平均两个模型对于噪声样本的预测结果。

![](https://pic.imgdb.cn/item/63bf7bb9be43e0d30e4f27c3.jpg)
