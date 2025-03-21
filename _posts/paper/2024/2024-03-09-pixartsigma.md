---
layout: post
title: 'PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation'
date: 2024-03-09
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/67dd19dd88c538a9b5c2c70e.png'
tags: 论文阅读
---

> PixArt-Σ：4K文本到图像生成的扩散Transformer的由弱到强的训练.

- paper：[PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://arxiv.org/abs/2403.04692)

# 0. TL; DR

本文介绍了一种名为**PixArt-Σ**的文本到图像（**T2I**）扩散模型，它能够在**4K**分辨率下直接生成高质量图像。**PixArt-Σ**基于[<font color=Blue>PixArt-α</font>](https://0809zheng.github.io/2023/09/30/pixartalpha.html)模型，通过“从弱到强”的训练策略，利用高质量数据和高效的**Token**压缩技术，显著提升了生成图像的保真度和对文本提示的对齐能力。

该模型仅使用**0.6B**参数，相比现有的**T2I**扩散模型（如**SDXL**的**2.6B**参数和**SD Cascade**的**5.1B**参数）更小，同时在图像质量和语义对齐方面表现出色。此外，**PixArt-Σ**能够直接生成**4K**分辨率的图像，无需后处理，为电影和游戏等行业的高质量视觉内容创作提供了有力支持。

# 1. 背景介绍

近年来，文本到图像（**T2I**）生成模型取得了显著进展，如**DALL·E 3、Midjourney**和**Stable Diffusion**等模型，它们能够生成逼真的图像，对图像编辑、视频生成和**3D**资产创建等下游应用产生了深远影响。然而，开发一个顶级的**T2I**模型需要大量的计算资源，例如从头开始训练**Stable Diffusion v1.5**需要大约**6000**个**A100 GPU**天，这对资源有限的研究人员构成了重大障碍，阻碍了**AIGC**社区的创新。因此，如何在有限资源下高效地提升T2I模型的性能成为了一个重要的研究方向。

# 2. PixArt-Σ 模型

**PixArt-Σ**的核心思想是通过“从弱到强”的训练策略，从**PixArt-α**的“较弱”基线模型逐步演进为“更强”的模型。具体来说，这一过程包括以下几个方面：

### （1）高质量训练数据

**PixArt-Σ**采用了比**PixArt-α**更高质量的图像数据，这些数据具有更高的分辨率（超过**1K**，包括约**2.3M**的**4K**分辨率图像）和更丰富的艺术风格。同时，为了提供更精确和详细的描述，**PixArt-Σ**使用了更强大的图像描述生成器**Share-Captioner**，替换了**PixArt-α**中使用的**LLaVA**，并将文本编码器的**Token**长度从**120**扩展到**300**，以增强模型对文本和视觉概念之间对齐的能力。

![](https://pic1.imgdb.cn/item/67dd1c5488c538a9b5c2c7ee.png)

### （2）高效的Token压缩

为了应对**4K**超高分辨率图像生成带来的计算挑战，**PixArt-Σ**基于**Diffusion Transformer（DiT）**架构，通过引入**KV Token**压缩技术，显著提高了模型在处理长序列**Token**时的效率。具体来说，**PixArt-Σ**在**Transformer**的深层（**14-27**层）引入了**KV Token**压缩，通过组卷积将**2×2**的**Token**压缩为一个**Token**，从而减少了计算复杂度。

此外，**PixArt-Σ**还采用了“**Conv Avg Init**”初始化策略，通过将卷积核的权重初始化为平均操作符，使得模型在初始状态下能够产生粗略的结果，加速了微调过程。这种设计有效地减少了训练和推理时间，对于**4K**图像生成的训练和推理时间减少了约**34%**。

![](https://pic1.imgdb.cn/item/67dd1cd788c538a9b5c2c840.png)

### （3）训练细节
**PixArt-Σ**的训练过程包括以下几个阶段：
- **VAE**适应：将**PixArt-α**的**VAE**替换为**SDXL**的**VAE**，并继续微调扩散模型。这一过程仅需**5**个**V100 GPU**天。
- 文本-图像对齐：使用高质量数据集进行微调，以提高模型对文本和图像对齐的能力。这一过程需要**50**个**V100 GPU**天。
- 高分辨率微调：从低分辨率模型（如**512px**）微调到高分辨率模型（如**1024px**），并引入**KV Token**压缩。这一过程通过“**PE Interpolation**”技巧初始化高分辨率模型的位置嵌入，显著提高了高分辨率模型的初始状态，加快了微调过程。

![](https://pic1.imgdb.cn/item/67dd1ecb88c538a9b5c2cd92.png)


# 3. 实验分析

**PixArt-Σ**在图像质量和语义对齐方面表现出色。通过与现有的**T2I**模型（如**PixArt-α、SDXL**和**Stable Cascade**）进行比较，**PixArt-Σ**在**FID**和**CLIP-Score**等指标上均取得了更好的结果。此外，**PixArt-Σ**还能够直接生成**4K**分辨率的图像，无需后处理，这为电影和游戏等行业的高质量视觉内容创作提供了有力支持。

![](https://pic1.imgdb.cn/item/67dd201288c538a9b5c2d180.png)

为了评估**PixArt-Σ**的性能，作者进行了人类偏好研究和**AI**偏好研究。在人类偏好研究中，**PixArt-Σ**在图像质量和对文本提示的遵循能力方面优于其他六种**T2I**生成器。在**AI**偏好研究中，使用**GPT-4 Vision**作为评估器，**PixArt-Σ**同样表现出色，证明了其在图像质量和语义对齐方面的优势。

![](https://pic1.imgdb.cn/item/67dd205488c538a9b5c2d24b.png)

作者还进行了消融研究，以评估不同**KV Token**压缩设计对生成性能的影响。实验结果表明，将**KV Token**压缩应用于**Transformer**的深层（**14-27**层）能够取得最佳性能。此外，使用“**Conv 2×2**”方法进行**Token**压缩的效果优于其他方法，如随机丢弃和平均池化。在不同分辨率下，**KV Token**压缩对图像质量（**FID**）有轻微影响，但对语义对齐（**CLIP-Score**）没有影响。尽管随着压缩比的增加，图像质量略有下降，但**KV Token**压缩显著提高了训练和推理的速度，特别是在高分辨率图像生成中。

![](https://pic1.imgdb.cn/item/67dd208b88c538a9b5c2d317.png)