---
layout: post
title: 'LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation'
date: 2024-03-03
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/664afbb2d9c307b7e995212f.png'
tags: 论文阅读
---

> LayoutDiffusion: 布局到图像生成的可控扩散模型.

- paper：[LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation](https://arxiv.org/abs/2303.17189)

由于文本的模糊性及其在精确表达图像空间位置方面的局限，文本引导的扩散模型在生成包含多个目标的复杂图像时，会出现物体缺失和错误地生成物体的位置、形状和类别等问题。而布局引导的图像扩散模型通过将带有边界框和类别注释的目标作为输入条件，可以在保持生成图像高质量的同时获得更强的可控性。

文本引导的扩散模型对所有条件输入采用简单的多模态融合方法(如交叉注意力)或直接拼接输入。与文本和图像的融合相比，图像与布局的融合是一个复杂的多模态融合问题。布局对目标的位置、大小和类别有更多的限制。这对模型的可控性要求较高，往往会导致生成图像的自然性和多样性降低。此外，布局对每个**token**更加敏感，布局**token**的丢失将直接导致目标的丢失。本文旨在专门设计布局与图像之间的融合机制。

## 1. LayoutDiffusion

**LayoutDiffusion**模型由三部分组成：布局融合模块(**Layout Fusion Module, LFM**)、目标感知交叉注意机制(**object-aware Cross-Attention Mechanism, OaCA**)和相应的无分类器(**classify-free**)训练和采样方案。
- **LFM**融合了每个目标的信息，并对多个目标之间的关系进行建模，从而提供了整个布局的潜在表示。
- **OaCA**将图像**patch**特征与布局在统一的坐标空间中进行交叉注意力计算，将两者的位置表示为边界框，使模型更加关注与目标相关的信息。
- 无分类器采样过程的速度进行一些优化，在25次迭代中可以显着优于**SOTA**模型。

![](https://pic.imgdb.cn/item/6650509ed9c307b7e9623aba.png)

### （1）布局嵌入 Layout Embedding

包含$n$个目标的布局记为$l=\{o_1,...,o_n\}$。其中每个目标$o_i=\{b_i,c_i\}$，$b_i=(x_0^i,y_0^i,x_1^i,y_1^i)\in [0,1]^4$是边界框，$c_i\in [0, \mathcal{C}+1]$是类别。

为了支持可变长度序列的输入，通过在$l$前面添加一个$o_l$和在最后添加一些$o_p$来将$l$填充到固定长度的$k$。$o_l=\{b_l=(0,0,1,1),c_l=0\}$表示整个布局，$o_p=\{b_p=(0,0,0,0),c_p=\mathcal{C}+1\}$表示一些没有目标的填充。

最后通过投影矩阵$W_B\in \mathbb{R}^{4\times d_L},W_C\in \mathbb{R}^{1\times d_L}$将布局$l=\{o_1,...,o_k\}$转换为布局嵌入$L\in \mathbb{R}^{k\times d_L}$:

$$
L=bW_B+cW_C
$$

### (2) 布局融合模块 Layout Fusion Module

布局嵌入中的每个目标与其他目标没有关系。这导致对整个场景的理解较差，特别是当多个目标重叠并相互阻挡时。因此，为了鼓励布局的多个目标之间进行更多的交互，作者提出了布局融合模块(**LFM**)，这是一种使用多层自注意力来融合布局嵌入的编码器。

$$
L^\prime = \text{LFM}(L) = \{O_1^\prime,...,O_k^\prime\}
$$

### （3）目标感知交叉注意力机制 Object-aware Cross-Attention Mechanism

图像**patch**仅局限于特征的语义信息，缺乏空间信息。因此通过添加包含位置和大小的区域信息来构造结构化图像**patch**。图像中第$(u,v)$个**patch**的区域信息定义为：

$$
b_{I_{u,v}} = \left( \frac{u}{h}, \frac{v}{w}, \frac{u+1}{h}, \frac{v+1}{w} \right)
$$

然后将图像**patch**的区域信息与图像中的目标框编码到统一的位置嵌入空间中：

$$
P_I = b_IW_BW_P\\
P_L = bW_BW_P
$$

对图像中的布局进行条件化的最简单方法之一是直接将布局的全局信息$O_1^\prime$添加到图像特征中：

$$
I^\prime = I+O_1^\prime W
$$

考虑到物体的位置、大小和类别的融合，将目标感知交叉注意力(**OaCA**)定义为:

$$
\begin{aligned}
Q &= \text{concat}_{\text{channel}}\left[ Q_I, P_L\right] \\
K &= \text{concat}_{\text{channel}}\left[ \text{concat}_{\text{length}}\left[ K_I,K_L\right], \text{concat}_{\text{length}}\left[ P_I,P_L\right]\right]\\
V &= \text{concat}_{\text{length}}\left[ V_I,V_L\right]
\end{aligned}
$$

其中布局$L^\prime$的键矩阵$K_L$和值矩阵$V_L$计算为：

$$
K_L,V_L = \text{Conv}(\frac{1}{2}(\text{Norm}(C_L)+L^\prime))
$$

其中类别嵌入$C_L$关注的是布局的类别信息，$L^\prime$关注的是目标本身以及可能与之有关系的其他目标的综合信息。通过对两者求平均，既能得到目标的一般信息，又能强调目标的类别信息。

图像$I$的查询矩阵$Q_I$、键矩阵$K_I$和值矩阵$V_I$计算为：

$$
Q_I, K_I,V_I = \text{Conv}(\text{Norm}(I))
$$

### （4）无分类器采样过程

为了支持布局作为条件输入，我们采用了无分类器引导条件生成。它是通过在有条件输入和没有条件输入的扩散模型的预测之间进行插值来完成的。

对于布局条件，首先构造一个填充布局$l_φ = \{o_l, o_p,···,o_p\}$。在训练过程中，扩散模型布局的条件将以固定概率被替换为$l_φ$。采样时，使用下式对有布局条件的图像进行采样:

$$
\hat{\epsilon}_\theta (x_t,t|l) = (1-s)\cdot \epsilon_\theta (x_t,t|l_\phi) + s\cdot \epsilon_\theta (x_t,t|l)
$$

作者还对无分类器采样过程的速度进行了一些优化。将**DPM-solver**用于无条件分类器采样，这是一种具有收敛阶保证的快速高阶扩散**ODE**专用解算器，以加快条件采样速度。

## 2. 实验分析

实验所使用的数据集包括：
- **COCO-Stuff**：使用**COCO 2017**分割挑战子集，该子集的训练集/验证集/测试集分别包含**40K / 5k / 5k**图像。使用其中包含3到8个覆盖图像的2\%以上且不属于人群的图像，共有25210张训练图像和3097张测试图像。
- **Visual Genome**：将108077张数据按$8:1:1$划分，分别选择在训练集中出现至少2000次和500次的目标和关系类别，并选择具有3到30个边界框并忽略所有小目标的图像。训练/验证/测试集分别拥有62565 / 5062 / 5096张图像。

所使用的评估指标包括：
- **Fr'echet Inception Distance (FID)**通过在**imagenet**预训练的**Inception-v3**网络上测量真实图像与生成图像之间特征分布的差异来显示生成图像的整体视觉质量。
- **Inception Score (IS)**使用在**ImageNet**网络上预训练的**Inception-v3**计算生成图像输出的统计分数。
- 多样性评分(**Diversity Score, DS**)通过比较两个生成的相同布局的图像在**DNN**特征空间中的**LPIPS**度量来计算它们之间的多样性。
- **Classification Score (CAS)**首先裁剪目标区域，并根以32×32的分辨率调整其大小。使用生成图像的裁剪图训练**ResNet-101**分类器，并在真实图像的裁剪图上进行测试。
- **YOLOScore**使用预训练的**YOLOv4**模型对生成图像上的**80**个目标类别**bbox mAP**进行评估。

结果表明**LayoutDiffusion**生成更精确的高质量图像，其布局对应的目标可识别性和准确性更高。

![](https://pic.imgdb.cn/item/665060c3d9c307b7e973623b.png)

![](https://pic.imgdb.cn/item/6650603fd9c307b7e972f0ff.png)

此外，来自相同布局的图像具有高质量和多样性(不同的照明，纹理，颜色和细节)。并且在初始布局的基础上不断添加一个额外的布局，在每一步中，**LayoutDiffusion**将新目标添加到非常精确的位置，具有一致的图像质量，显示用户友好的交互性。

![](https://pic.imgdb.cn/item/6650609bd9c307b7e9733d41.png)

作者还进行了不同组件的消融实验。与基线相比，多样性得分(**DS**)略有下降。**DS**在物理上与其他指标(如**CAS**和**YOLOScore)**所代表的可控性相反。对生成图像的精确控制导致了对多样性的更多限制。

![](https://pic.imgdb.cn/item/6650613dd9c307b7e973c991.png)

