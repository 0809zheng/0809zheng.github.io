---
layout: post
title: 'Deep Ensembles: A Loss Landscape Perspective'
date: 2020-07-12
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f0acba314195aa594176453.jpg'
tags: 论文阅读
---

> 探讨深度集成学习的损失曲面.

- paper：mixup: Deep Ensembles: A Loss Landscape Perspective
- arXiv：[link](https://arxiv.org/abs/1912.02757v1)

深度学习模型在进行训练时最终会收敛到局部极小值附近。之前的研究表明，一个损失函数存在许多局部极小值，在这些值上函数的表现是类似的。

![](https://pic.downk.cc/item/5f0ad7db14195aa5941aa6a7.jpg)


# 实验

对于一次初始化的训练，作者记录了$30$次**checkpoint**，评估这$30$次结果的参数余弦相似度以及预测结果的相似度。通过实验发现，在最初的训练过程中参数的变化较大，对结果的预测变化也比较大；训练逐渐趋于稳定后参数几乎不再变化；作者绘制了同一个随机初始化模型的三次模型训练收敛的结果（降维到2维展示），发现模型更新之后会收敛到附近的局部极值：

![](https://pic.downk.cc/item/5f0adb2614195aa5941b79b7.jpg)

作者进一步比较了收敛到局部极值的不同模型的参数余弦相似度以及预测结果的相似度。通过实验发现，不同的模型参数即使都收敛到局部极值，其参数差异仍然比较大，并且对结果的预测差异也很大：

![](https://pic.downk.cc/item/5f0adbc114195aa5941ba0e6.jpg)

作者认为，通过训练一系列模型，使每个模型都收敛到局部极小值，将这些模型集成起来对最终的结果有很大的提升。而每个模型都由不同的随机初始化确定。为了进行对比试验，作者还提出了一些在局部极小值附近构造子模型的方法，并探索集成不同模型和集成这些扰动较小子模型对结果带来的提升。这些方法包括：
- Random subspace sampling
- Monte Carlo dropout subspace
- Diagonal Gaussian subspace
- Low-rank Gaussian subspace

其采样过程如图所示：

![](https://pic.downk.cc/item/5f0adc4a14195aa5941bc777.jpg)

作者通过实验发现，独立优化的模型和基准模型同样达到较高的准确率，并且其对结果的预测差异很大。而上述四种扰动模型对结果的预测和验证集准确率存在较强的相关性，这说明集成独立训练的模型对结果会有帮助，但集成这些扰动模型对结果的影响不大：

![](https://pic.downk.cc/item/5f0adc8414195aa5941bd66f.jpg)

作者进一步通过实验说明，对于同一个初始化结果，沿不同的方向优化之后到达局部极值，其对应的模型差异也很大：

![](https://pic.downk.cc/item/5f0ade4f14195aa5941c5e8e.jpg)

