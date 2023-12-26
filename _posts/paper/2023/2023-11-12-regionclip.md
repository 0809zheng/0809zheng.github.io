---
layout: post
title: 'RegionCLIP: Region-based Language-Image Pretraining'
date: 2023-11-12
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/658a72a2c458853aefc127a6.jpg'
tags: 论文阅读
---

> RegionCLIP：基于区域的语言图像预训练.

- paper：[RegionCLIP: Region-based Language-Image Pretraining](https://arxiv.org/abs/2112.09106)

视觉-语言模型取得了很大的突破，这些模型使用了大量图文对来学习图像和文本的匹配。为了探索这种思路能否在 **region-caption** 的情况下起作用，作者基于预训练好的 **CLIP** 模型构建了一个 **R-CNN** 形式的目标检测器**RegionCLIP**。

在 **LVIS** 数据集上，当使用 **proposal** 作为输入时，**CLIP** 的得分无法指代定位的质量；使用 **gt** 框作为输入，**CLIP** 在 **LVIS** 框上的分类准确率只有 $19\%$，所以直接将预训练好的 **CLIP** 拿来用于对 **region** 的分类不太适合。

![](https://pic.imgdb.cn/item/658a7980c458853aefda376b.jpg)

**CLIP** 模型的训练是使用整个 **image** 作为输入的，使用的是 **image-level** 的文本描述来训练的，所以模型学习到的是整张图的特征，无法将文本概念和图像中的区域联系起来。**RegionCLIP**在预训练过程中将 **image region** 和 **text token** 进行对齐，先从输入图像中抠出候选区域，然后使用 **CLIP** 模型将抠出的区域和 **text embedding** 进行匹配。

![](https://pic.imgdb.cn/item/658a7ae7c458853aefdff48d.jpg)

由于 **CLIP** 缺少 **region** 层面的训练，所以 **RegionCLIP** 构建了一些 **region** 的伪标签来和 **image-text** 一起预训练：从网络数据中收集图像描述语句，然后使用 **NLP parser** 来提取出有效的目标词汇，构建词汇池，然后将词汇池的每个词都填入 **prompt** 模版（**a photo of xxx**），并且对每个词汇对应的 **prompt** 模版使用 **CLIP** 的 **text encoder** 来得到语义特征，所有的 **region concept** 都能够使用 **semantic embedding**  $$\{l_j\}_{j=1,...,C}$$来表示。

为了使得构建的 **region** 伪标签和 **region** 对应，使用 **CLIP** 的 **visual encoder** 来提取每个 **region** 的 **visual feature**  $v_i^t$，计算其和词汇池中的向量的距离，得分最大的向量就作为该 **region** 对应的伪标签$$\{v_i, l_m\}$$。

$$
S(v,l)=\frac{v^{T}\cdot l}{|| v||\cdot||l||}
$$


将 **images-concept** 和 **region-concept** 的数据联合进行预训练，训练的时候会同时使用对比学习 **loss** 和蒸馏 **loss**：

$$
{\cal L}={\cal L}_{c n t r s t}+{\cal L}_{d i s t}+{\cal L}_{c n t r s t-i m g}
$$

**region-text** 对比学习 **loss** 计算学生模型学习的 **region-text pairs** 的相似度：

$$
L_{c n t r s t}={\frac{1}{N}}\sum_{i}-\log(p(v_{i},l_{m})) \\
p(v_{i},l_{m})=\frac{\exp(S(v_{i},l_{m})/\tau)}{\exp(S(v_{i},l_{m})/\tau)+\sum_{k\in{\cal{N}}_{r_{i}}}\exp(S(v_{i},l_{k})/\tau)}
$$

蒸馏 **loss** 计算教师模型和学生模型得到的 **region-text** 的 **matching score**:

$$
L_{d i s t}=\frac{1}{N}\sum_{i}L_{K L}(q_{i}^{t},q_{i})
$$

**image-text** 的对比 **loss** 可以从 **region level** 扩展而来，即一个 **box** 覆盖了整张图，文本描述来源于网络。

预训练之后，训练得到的 **visual encoder** 可以直接用于 **region reasoning** 任务，比如从 **RPN** 获得区域，从训练的 **visual encoder** 得到该区域的视觉表达，然后和文本词汇表达进行匹配，得到相似度最高的文本。实验证明使用 **RPN score** 能够提升 **zero-shot** 推理的效果，所以作者使用 **RPN objectness score + category confidence score** 的均值来作为最终的得分，用于匹配。

预训练中的 **visual encoder** 是从 **teacher model** 提供的 **region-text alignment** 中学习的，不需要人为一些操作，所以也会有噪声；可以进一步微调 **visual encoder**。作者通过初始化目标检测器的 **visual backbone** 来实现，先使用现有的 **RPN** 网络来进行目标区域的定位，然后将区域和文本匹配。

