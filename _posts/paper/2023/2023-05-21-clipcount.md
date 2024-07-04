---
layout: post
title: 'Teaching CLIP to Count to Ten'
date: 2023-05-21
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/66826c9bd9c307b7e9a3e2ca.png'
tags: 论文阅读
---

> 教CLIP模型从一数到十.

- paper：[Teaching CLIP to Count to Ten](https://arxiv.org/abs/2302.12066)

大型视觉语言模型**CLIP**通过学习丰富的联合图像-文本表示，促进了许多下游任务的进展。该模型有一个明显的局限性：不能封装诸如计数之类的组合概念。作者提出了一种新的计数对比损失，通过创建具有错误事实的示例，微调**CLIP**来提高其对计数概念的定量理解。此外作者引入了**CountBench**，这是一个新的图像文本计数基准，用于评估模型对目标计数的理解程度。

**CLIP**无法实现准确计数的主要原因包括：
- 随着目标数量的增加，准确指定目标数量的文本标题在数据中变得极其罕见。例如对于6个以上的目标，标题通常包含一个一般形式的数量（如“一组”或“许多”），而不是准确的计数。
- 将图像中可见物体的数量与标题中的数字相关联不能充分促进**CLIP**的判别训练目标。这是因为其他文本和视觉特征(如名词和对象类别)对于将图像与其真实标题关联起来更有帮助。

本文框架由两个主要阶段组成。首先自动创建一个计数训练集，包括干净和多样化的图像以及描述场景中可见物体数量的相应标题。然后利用该数据集通过指定的基于计数的对比损失来微调**CLIP**，该对比损失与原始的通用图像-文本目标一起训练。

![](https://pic.imgdb.cn/item/668276cdd9c307b7e9b40cdb.png)


## 1. 创建计数训练集

获取图像-文本计数数据集的简单方法是通过只考虑标题包含数字的示例来过滤大规模数据集。然而，这种方法会产生高度噪声的数据集，因为标题中的数字通常指的是与计数无关的其他属性，如年龄、时间、地址等。

为了确保标题中的数字正确地引用图像中特定目标的实例数，在数据管道中采用了几个阶段的自动过滤：
- 过滤掉标题中不包含拼写数字(**two,...,ten**)的所有示例；没有拼写的数字或者大于10的数字，大多与时间(如日期)或地址一起出现，而不是与图像中物体的数字一起出现；
- 验证拼写的数字确实用作目标计数器，并且计数的目标在图像中是可见和可检测的。计数验证是通过应用现成的目标检测器**MobileNetV3**，并计算每个目标的检测次数来自动实现的。
- 平衡数据至关重要。由于描述超过6个物体的图像很少，选择将数字2-6与较大的数字7-10分开平衡。对于数字2-6指代的图像，采样大约37K个样本；对于7-10的图像，使用过滤后的所有样本，有大约7K的7样本到大约1.5K的10样本。

## 2. 微调CLIP

在两个训练集上对**CLIP**模型进行微调：
1. 从网络上收集的非常大的数据集，其中包含一般的野外图像和说明。
2. 过滤后的图像-文本计数数据集，在标题中列出了目标计数。

**CLIP**的微调训练损失包括常规的**CLIP**对比损失和计数损失。其中对比损失在所有数据集上计算，计数损失只在图像-文本计数数据集上计算。

对于每个图像-文本对$(i_k, t_k)$，通过将标题$t_k$中的数字与不同的随机数交换，自动创建一个错误标题$t_k^{CF}$。在训练中，将$(i_k, t_k, t_k^{CF})$送到**CLIP**的文本和图像编码器，以获得它们的嵌入$(ei_k, et_k, et_k^{CF})$。计数损失构造为对比损失的形式，强制图像与原始标题的相似度得分高，与错误标题的相似度得分低:

$$
L_{count} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{e^{ei_k \cdot et_k}} {e^{ei_k \cdot et_k} + e^{ei_k \cdot et_k^{CF}}}
$$

## 3. 图像文本计数基准 CountBench

作者引入了一个新的目标计数基准**CountBench**，它从公开可用的**LAION-400M**图像文本数据集中自动整理(并手动验证)。**CountBench**总共包含540张图像，其中包含2-10个特定目标的实例个60张，其相应的标题反映了这个数字。

![](https://pic.imgdb.cn/item/66827b61d9c307b7e9bac4ff.png)

## 4. 实验分析

在**CountBench**上评估模型的零样本目标计数能力。对于**CountBench**中的每张图像，将其标题中的数字替换为所有数字2-10，并计算图像与九个标题中的每个标题之间的相似度得分。标题中与图像相似度得分最高的数字被认为是预测数字。

下表报告了计数精度(正确数字的选择)和模型预测与正确数字的平均偏差。所提方法实现了明显优越的计数精度，并且当缺失计数损失或数据集过滤时准确度具有巨大差距。

![](https://pic.imgdb.cn/item/668287d3d9c307b7e9ce926b.png)

作者对计数评估的混淆矩阵进行可视化，改进的**CLIP**模型在所有数字上都明显优于基线。同样明显的是一些较高的数字的准确性下降，因为它们在训练数据中的数量明显降低。

![](https://pic.imgdb.cn/item/66828860d9c307b7e9cf8f02.png)

为了更好地理解模型学习了什么，使用可解释性方法来可视化模型的推理。对于每个图像标题对，将其**CLIP**嵌入的余弦相似度作为其文本得分，并由图像中每个**patch**和文本中每个**token**的相关性分数生成相关性图。结果表明标题中拼写数字的相关性得分明显高于基线模型，并且模型专注于图像中的所有相关目标。

![](https://pic.imgdb.cn/item/66828e68d9c307b7e9da0126.png)

为了验证模型不是简单地关注图像中出现的所有目标，使用错误文本提示生成相关性图。当使用正确的数字时，模型只关注相关的目标，这表明模型学会了将标题中的拼写数字与适当数量的目标关联起来，而不是利用快捷方式或不需要的内容。

![](https://pic.imgdb.cn/item/66828eaed9c307b7e9da6d16.png)
