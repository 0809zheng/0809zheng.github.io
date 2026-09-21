---
layout: post
title: '深度学习(Deep Learning)概述'
date: 2020-01-02
author: 郑之杰
cover: ''
tags: 深度学习
pinned: true
---

> Outlines about Deep Learning.

- 提示：持续更新中...请点击任意[<font color=Blue>高亮位置</font>](https://0809zheng.github.io/2020/01/02/DL-outline.html)以发现更多细节！

**深度学习**(**Deep Learning**)是一种以深度神经网络为工具的机器学习方法。
本文首先介绍深度学习的**基本组件**和**方法技巧**，其次介绍深度神经网络的**类型**，最后介绍深度学习在计算机视觉、自然语言处理等领域的**应用**。

本文目录：
- **深度学习的基本组件和方法技巧**
  - **深度学习的基本组件**：激活函数、优化方法、正则化方法、归一化方法、参数初始化方法
  - **深度学习的方法**：半监督学习、自监督学习、度量学习、多实例学习、多任务学习、主动学习、迁移学习、终身学习、元学习
  - **深度学习的技巧**：图像的数据增强、混合精度训练、长尾分布、对抗训练、大模型的参数高效微调
- **深度神经网络的类型**
  - **卷积神经网络**：卷积神经网络的基本概念、卷积神经网络中的池化层、卷积神经网络中的注意力机制、卷积神经网络中的自注意力机制、轻量级卷积神经网络
  - **循环神经网络**：循环神经网络的基本概念、序列到序列模型、序列到序列模型中的注意力机制
  - **自注意力网络**：自注意力机制、**Transformer**、**Transformer**中的位置编码、降低**Transformer**的计算复杂度、
  - **深度生成模型**：生成对抗网络、变分自编码器、流模型、流匹配模型
  - **其他类型的网络**：递归神经网络、记忆增强神经网络、图神经网络、状态空间模型
- **深度学习的应用**
  - **High-Level视觉**：图像识别、目标检测、开放集合目标检测、图像分割、点云分类、目标计数
  - **Low-Level视觉**：图像超分辨率、全色锐化
  - **GenAI**：图像到图像翻译、布局引导图像生成
  - **Human-Centric感知**：人体姿态估计、人脸检测, 识别与验证、行人检测与属性识别、时空动作检测、射频人体感知
  - **自然语言处理**：预训练语言模型、
  - **多模态**：文本检测与识别、视觉-语言预训练


# 1. 深度学习的基本组件和方法技巧

## (1) 深度学习的基本组件
### ⚪ [<font color=Blue>激活函数 (Activation Function)</font>](https://0809zheng.github.io/2020/03/01/activation.html)
**激活函数**为神经网络引入不可或缺的非线性。设计激活函数时可以考虑的七条准则包括：连续可导、计算量小、没有饱和区、没有偏置偏移、具有生物可解释性、能够提取上下文信息、具有通用近似性。

常见的激活函数按设计思路可以分为七族：
- **S**型激活函数：形如**S**型曲线，单调有界、处处光滑，因此两端必然饱和；如今主要用于门控机制与概率输出。包括**Step**, **Sigmoid**, **HardSigmoid**, **Tanh**, **HardTanh**, **ISRU**, **Softsign**
- **ReLU**族激活函数：正半轴保持（近似）恒等映射从而不饱和，差别在于如何处理负半轴与原点处的不可导。包括光滑与有界变体(**ReLU**, **Softplus**, **Squareplus**, **ReLU6**)、修正负半轴斜率(**LeakyReLU**, **PReLU**, **RReLU**, **CReLU**, **SReLU**, **SUGAR**)、负半轴指数饱和(**ELU**, **CELU**, **SELU**, **ISRLU**, **PELU**)、概率视角(**GELU**)、以及反过来修改正半轴的幂与多项式(**Squared ReLU**, **StarReLU**, **xIELU**, **PolyCom**)
- 自动搜索的激活函数：用强化学习、遗传算法或大语言模型在函数空间中搜索得到。包括**Swish**, **HardSwish**, **Mish**, **ELiSH**, **HardELiSH**, **E-Swish**, **LiSHT**, **TanhExp**, **GELUSine**
- 周期性激活函数：引入正弦、余弦项以表示高频信号，主要用于隐式神经表示。包括**SIREN**, **Snake**, **GCU**
- 通用近似激活函数：把激活函数本身参数化，从数据中学习它的形状。包括分段线性参数化(**Maxout**, **APL**, **PWLU**)、有理函数参数化(**PAU**, **OPAU**)、样条参数化(**KAN**)、非光滑激活函数的光滑化(**ACON**, **SMU**, **SAU**)、统一参数族(**AGLU**)
- 上下文相关的激活函数：多输入单输出，由全局、通道或空间邻域的上下文决定函数形状。包括**Dynamic ReLU**, **Dynamic Shift-Max**, **FReLU**
- 门控激活函数：用一路输出调制另一路，是当前大模型前馈层的标准形式。包括**GLU**, **ReGLU**, **GEGLU**, **SwiGLU**, **dReLU**, **xATLU**, **xGELU**, **xSiLU**, **PowLU**

### ⚪ [<font color=Blue>优化方法 (Optimization)</font>](https://0809zheng.github.io/2020/03/02/optimization.html)

深度学习中的**优化**问题是指在已有的数据集上最小化训练损失$L(\theta)$，通常用基于梯度的数值方法求解。在实际应用梯度方法时，可以根据截止到当前步$t$的历史梯度信息$$\{g_{1},...,g_{t}\}$$计算修正的参数更新量$h_t$（比如累积动量、累积二阶矩校正学习率等）。指定每次计算梯度所使用的数据批量$\mathcal{B}$和学习率$\gamma$，则第$t$次参数更新为：

$$ \begin{aligned} g_t&=\frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}}^{}\nabla_{\theta} l(\theta_{t-1};x) \\ h_t &= f(g_{1},...,g_{t}) \\ \theta_t&=\theta_{t-1}-\gamma h_t \end{aligned} $$

梯度下降可以从**动力学**、**逼近**、**概率**三个角度理解，此外还有隐式梯度正则化、与核方法的联系、深度集成与损失曲面等分析视角。针对基于梯度的方法的不同缺陷，优化器可以分为八族：
- 基础方法与**动量**：更新过程容易在鞍点与病态方向上停滞，引入动量以累积一致的下降方向。包括**SGD**, **Momentum**, **NAG**, **signSGD**, **Signum**
- **自适应**学习率：参数不同维度的梯度尺度差异巨大，为每个分量单独设置步长。包括**RProp**, **AdaGrad**, **RMSProp**, **AdaDelta**
- 动量与自适应的结合（**Adam**族）：当前的主流选择，差别在于如何估计与修正一阶、二阶矩。包括**Adam**, **Adamax**, **AdamW**, **Nadam**, **AMSGrad**, **RAdam**, **AdaBound**, **AdaBelief**, **AdaX**, **Amos**, **Lion**, **Adan**
- 二阶信息与**矩阵型预条件**：一阶信息不足以描述曲率，用近似的**Hessian**或**Fisher**信息矩阵做预条件。包括牛顿法, **L-BFGS**, 共轭梯度, **K-FAC**, **Shampoo**, **SOAP**, **Sophia**, **Muon**, **PSGD**, **AdaBK**, **KL-Shampoo**, **Newton-Muon**, **Pion**, **DeltaMomentum**
- **降低显存**占用：优化器状态与参数同样大，通过分解、共享或融合压缩这部分开销。包括**Adafactor**, **SM3**, **Adam-mini**, **GaLore**, **LoMo**, **AdaLomo**
- **层级**自适应与大批量训练：分布式训练中整体批量过大会导致精度崩塌，按层归一化更新量可以缓解。包括**LARS**, **LAMB**, **NovoGrad**
- 免调参与步长自适应：学习率本身难以选择，在训练中在线估计它。包括指数梯度更新, **D-Adaptation**, **Prodigy**, **Schedule-Free**, **ScheduleFree++**
- 不依赖反向传播的梯度估计：反向传播的显存占用与串行性受限，只用前向计算近似梯度。包括前向梯度, 零阶优化(**MeZO**)

与优化器正交的还有三类训练技巧：**学习率与批量的调度**（**warmup**、余弦与线性衰减、梯度裁剪、线性缩放律、增大批量代替衰减学习率）、**权重平均**（**SWA**, **EMA**, **Lookahead**）与**数据流水线加速**（**Data Echoing**）。

### ⚪ [<font color=Blue>正则化方法 (Regularization)</font>](https://0809zheng.github.io/2020/03/03/regularization.html)

**正则化**指的是通过**引入噪声**或限制模型的**复杂度**，降低模型对输入或参数的敏感性，从而避免过拟合、提高泛化能力。从偏差-方差分解的角度看，正则化是用可控的偏差增加换取方差的显著下降；其实现有两条路径：**限制假设空间**与**在训练中注入噪声**。

常用的正则化方法按作用对象分为三族：
- 约束**目标函数**：在损失函数中增加关于参数或梯度的惩罚项。包括**L2**正则化, **L1**正则化, 弹性网络正则化, **L0**正则化, 谱正则化, 正交正则化, 自正交性正则化, **WEISSI**正则化, 梯度惩罚（对参数与对输入两种形式）
- 约束**网络结构**：在网络结构中随机地丢弃神经元、连接、通道或整层。包括**Dropout**, **Gaussian Dropout**, **DropConnect**, **Spatial Dropout**, **DropBlock**, **Weighted Channel Dropout**, **R-Drop**, 随机深度, **DropPath**, **LayerDrop**, **Shake-Shake**, **ShakeDrop**, **Token Dropping**
- 约束**优化过程**：在优化过程中施加额外的步骤或约束。包括数据增强, 梯度裁剪, **Early Stopping**, 标签平滑, 权重衰减与**AdamW**, 变分信息瓶颈, 虚拟对抗训练, 对抗训练, **Flooding**, **SAM**, 权重平均(**SWA**, **EMA**, **Model Soup**), 噪声标签下的正则化

这些方法之间存在深刻的联系：**Dropout**与**Early Stopping**都可以在特定条件下等价于**L2**正则化，**SGD**本身也带有隐式正则化效应，而"梯度惩罚"是贯穿多种方法的一条统一主线。


### ⚪ [<font color=Blue>归一化方法 (Normalization)</font>](https://0809zheng.github.io/2020/03/04/normalization.html)

输入数据的特征通常具有不同的量纲和取值范围，使得不同特征的**尺度**差异很大。**归一化**泛指把数据特征的不同维度转换到相同尺度的方法。它之所以有效，通常归因于缓解特征尺度差异、减少内部协变量偏移、平滑损失曲面、引入尺度不变性（从而自动调节有效学习率）与隐式正则化效应。

深度学习中的归一化方法按作用对象分为六族：
- **数据层面**的归一化：在数据进入网络之前统一尺度。包括最小-最大值归一化, 标准化, 白化, 分位数归一化
- **激活值**归一化（逐层归一化）：在网络内部对激活值做归一化，区别在于统计量沿哪些维度计算。（依赖批量统计）**BN**, **SyncBN**, **Ghost BN**, **Batch Renormalization**, **AdaBN**, **L1-Norm BN**, **Generalized BN**, **Decorrelated BN**, **IterNorm**；（不依赖批量统计）**LRN**, **LN**, **RMSNorm**, **IN**, **GN**, **FRN**, **PONO**, **RN**；（可学习与自动搜索）切换归一化**SN**, **EvoNorm**, **Attentive Normalization**
- **条件与自适应**归一化：用外部条件生成归一化的缩放与偏移参数。包括**CIN**, **CBN**, **AdaIN**, **SPADE**, **FiLM**, **adaLN**, **adaLN-Zero**, **Modulated Convolution**
- **参数**归一化：归一化权重而不是激活值。包括权重归一化**WN**, 中心化权重归一化, 权重标准化**WS**, 余弦归一化**CN**, 谱归一化**Spectral Norm**
- 归一化在**Transformer**中的位置：位置的选择直接决定深层网络能否稳定训练。包括**Post-LN**, **Pre-LN**, **Sandwich-LN**, **Peri-LN**, **DeepNorm**, **Mix-LN**, **LayerNorm Scaling**, **ScaleNorm**, **FixNorm**, **QK-Norm**, **nGPT**
- **去掉**归一化：用初始化或权重约束替代归一化层。包括**Fixup**, **SkipInit**, **ReZero**, **NF-Net**, **DyT**, **DyISRU**

### ⚪ [<font color=Blue>参数初始化方法 (Parameter Initialization)</font>](https://0809zheng.github.io/2020/03/05/initialization.html)

对神经网络进行训练时需要先对参数进行初始化。糟糕的初始化不仅会使模型效果变差，还有可能使得模型根本训练不动或者不收敛。初始化要解决的核心问题有两个：**打破神经元之间的对称性**，以及让**前向激活值与反向梯度的方差**在逐层传播中保持稳定（否则表现为梯度消失或梯度爆炸）。初始化的具体取值还与激活函数、归一化层和学习率相互耦合。

常见的初始化方法分为六族：
- **朴素**初始化：不考虑网络结构的简单方案。包括零初始化, 常数初始化, 随机正态初始化, 随机均匀初始化, 稀疏初始化, 偏置的初始化惯例
- **方差缩放**初始化：按扇入或扇出缩放初始化方差，使信号方差逐层守恒。包括**LeCun**初始化, **Xavier**初始化, **Kaiming**初始化, **SELU**的自归一化不动点
- **正交、恒等与等距**初始化：让权重矩阵（近似）保持向量的长度与夹角。包括正交初始化, 恒等初始化, **ZerO**初始化, **Delta-Orthogonal**初始化
- **残差网络**的初始化：把残差分支初始化成接近零映射，使网络在初始化时接近恒等映射。包括**Zero-γ**, **Fixup**, **SkipInit**, **ReZero**, **LayerScale**, **T-Fixup**, **Admin**, **DeepNorm**
- **Transformer**与大模型的初始化：兼顾深度、宽度与超参数的可迁移性。包括小标准差初始化, 残差缩放$1/\sqrt{2L}$, 嵌入层初始化与权重绑定, **muP**, 谱条件(**Spectral Condition**), **Depth-muP**, **CompleteP**
- **数据驱动与学习式**初始化：用数据或优化过程本身决定初始值。包括**LSUV**, 数据依赖初始化, **MetaInit**, **GradInit**, 模仿初始化, 预训练权重迁移与模型生长, **LoRA**的初始化


## (2) 深度学习的方法

### ⚪ [<font color=Blue>半监督学习 (Semi-Supervised Learning)</font>](https://0809zheng.github.io/2022/09/01/semi.html)

**半监督学习**是指同时从有标签数据和无标签数据中进行学习。半监督学习的假设包括平滑性假设、聚类假设、低密度分离假设和流形假设。

常用的半监督学习方法包括：
- **一致性正则化**：假设神经网络的随机性或数据增强不会改变输入样本的真实标签，如$\Pi$**-Model**, **Temporal Ensembling**, **Mean Teacher**, **VAT**, **ICT**, **UDA**。
- **伪标签**：根据当前模型的最大预测概率为无标签样本指定假标签，如**Label Propagation**, **Confirmation Bias**, **Noisy Student**, **Meta Pseudo Label**。
- **一致性正则化+伪标签**：既构造无标签样本的伪标签，又同时建立监督损失和无监督损失，如**MixMatch**, **ReMixMatch**, **FixMatch**, **DivideMix**。


### ⚪ [<font color=Blue>自监督学习 (Self-Supervised Learning)</font>](https://0809zheng.github.io/2022/10/01/self.html)

**自监督学习**是一种无监督表示学习方法，旨在根据无标签数据集中的一部分信息预测剩余的信息，并以有监督的方式来训练该数据集。

适用于图像数据集的自监督任务包括：
- **前置任务(pretext task)**：通过从数据集中自动构造伪标签而设计的对目标任务有帮助的辅助任务，如**Exemplar-CNN**, **Context Prediction**, **Jigsaw Puzzle**, **Image Colorization**, **Learning to Count**, **Image Rotation**, **Jigsaw Clustering**, **Evolving Loss**, **PIC**, **MP3**。
- **对比学习(contrastive learning)**：学习一个特征嵌入空间使得正样本对彼此靠近、负样本对相互远离。(对比损失函数) **NCE**, **CPC**, **CPC v2**, **Alignment and Uniformity**, **Debiased Contrastive Loss**, **Hard Negative Samples**, **FlatNCE**; (并行数据增强) **InvaSpread**, **SimCLR**, **SimCLRv2**, **BYOL**, **SimSiam**, **DINO**, **SwAV**, **PixContrast**, **Barlow Twins**; (存储体) **InstDisc**, **MoCo**, **MoCo v2**, **MoCo v3**; (多模态) **CMC**, **CLIP**; (应用) **CURL**, **CUT**, **Background Augmentation**, **FD**。
- **掩码图像建模(masked image modeling)**：随机遮挡图像中的部分**patch**，并以自编码器的形式重构这部分**patch**，如**BEiT**, **MAE**, **SimMIM**, **iBOT**, **ConvMAE**, **QB-Heat**, **LocalMIM**, **DeepMIM**。

### ⚪ [<font color=Blue>度量学习 (Metric Learning)</font>](https://0809zheng.github.io/2022/11/01/metric.html)

**深度度量学习**通过共享权重的**Siamese**网络把原始样本映射到低维特征空间，并设计合理的度量损失使得同类样本在特征空间上的距离比较近，不同类样本之间的距离比较远。

度量学习的目标在于最小化相似样本(正样本对)之间的距离，最大化不相似样本(负样本对)之间的距离。深度度量损失包括：
- 基于**对(pair)**的度量损失：考虑一个批次样本中样本对之间的关系，最小化正样本对$(x,x^+)$之间的距离，最大化负样本对$(x,x^-)$之间的距离。如**Contrastive Loss**, **Binomial Deviance Loss**, **Triplet Loss**, **Improved Triplet Loss**, **Batch Hard Triplet Loss**, **Hierarchical Triplet Loss**, **Angular Loss**, **Quadruplet Loss**, **N-pair Loss**, **Lift Structured Loss**, **Histogram Loss**, **Ranked List Loss**, **Soft Nearest Neighbor Loss**, **Multi-Similarity Loss**, **Circle Loss**。
- 基于**代理(proxy)**的度量损失：为每个类别赋予一个代理样本，拉近每个类别的样本和该类别对应的代理样本之间的距离，拉远与其他类别对应的代理样本之间的距离。如**Magnet Loss**, **Clustering Loss**, **Proxy-NCA**, **ProxyNCA++**, **Proxy-Anchor**。

### ⚪ [<font color=Blue>多实例学习 (Multi-Instance Learning)</font>](https://0809zheng.github.io/2025/10/01/mil.html)

在标准的监督学习框架下，每一个实例(**instance**)都会被赋予一个标签。**多实例学习**是一种弱监督学习框架，它将数据组织成一种层级结构：由多个实例构成一个**包（Bag）**，标签是在“包”的层面上给出的。

多实例学习的标准假设是：一个包被标记为正，当且仅当它至少包含一个正实例；一个包被标记为负，当且仅当它所有实例均为负。

现有的**Deep MIL**方法根据其分类粒度(**Classification Granularity**)可以分为：
- 实例级(**instance-level**) **MIL**：先对每个实例分别做预测，再对预测结果进行聚合；如**mi-Net**, **Adaptive Pooling**, **Certainty Pooling**, **Power Pooling**, **DSMIL**, **MIVAE**, **CausalMIL**, **Additive MIL**, **MILLET**, **MIREL**, **FocusMIL**。
- 包级(**bag-level**) **MIL**：先将所有实例的特征向量聚合，再对包特征做预测；如**MI-Net**, **Attention MIL**, **DP-MINN**, **GNN-MIL**, **Loss-Attention**, **TransMIL**, **SA-AbMILP**, **DTFD-MIL**, **IBMIL**, **DAS-MIL**, **MHIM-MIL**, **Extreme MIL**。

### ⚪ [<font color=Blue>多任务学习 (Multi-Task Learning)</font>](https://0809zheng.github.io/2021/08/28/MTL.html)

**多任务学习**是指同时学习多个属于不同领域的任务，并通过特定任务的领域信息提高泛化能力。多任务学习的方法设计可以分别从**网络结构**与**损失函数**两个角度出发。

一个高效的多任务网络，应同时兼顾特征共享部分和任务特定部分。根据模型在处理不同任务时网络参数的共享程度，多任务学习方法的网络结构可分为：
- **硬参数共享 (Hard Parameter Sharing)**：模型的主体部分共享参数，输出结构任务独立。如**Multilinear Relationship Network**, **Fully-adaptive Feature Sharing**。
- **软参数共享 (Soft Parameter Sharing)**：不同任务采用独立模型，模型参数彼此约束。如**Cross-Stitch Network**, **Sluice Network**, **Multi-Task Attention Network**。

多任务学习将多个相关的任务共同训练，其总损失函数是每个任务的损失函数的加权求和式：$$\mathcal{L}_{total} = \sum_{k}^{} w_k\mathcal{L}_k$$。多任务学习的目的是寻找模型参数的**帕累托最优解**，因此需要设置合适的任务权重。一些权重自动设置方法包括**Uncertainty**, **Gradient Normalization**, **Dynamic Weight Average**, **Multi-Objective Optimization**, **Dynamic Task Prioritization**, **Loss-Balanced Task Weighting**

### ⚪ [<font color=Blue>主动学习 (Active Learning)</font>](https://0809zheng.github.io/2022/08/01/activelearning.html)

**主动学习**是指从未标注数据中只选择一小部分样本进行标注和训练来降低标注成本。深度主动学习最常见的场景是基于**池**(**pool-based**)的主动学习，即从大量未标注的数据样本中迭代地选择最“有价值”的数据，直到性能达到指定要求或标注预算耗尽；选择最“有价值”的数据的过程被称为**采样策略**。

深度主动学习方法可以根据不同的**采样策略**进行分类：
- **不确定性采样 (uncertainty sampling)**：选择使得模型预测的不确定性最大的样本。不确定性的衡量可以通过机器学习方法(如**entropy**)、**QBC**方法(如**voter entropy**, **consensus entropy**)、贝叶斯神经网络(如**BALD**, **bayes-by-backprop**)、对抗生成(如**GAAL**, **BGADL**)、对抗攻击(如**DFAL**)、损失预测(如**LPL**)、标签预测(如**forgetable event**, **CEAL**)
- **多样性采样 (diversity sampling)**：选择更能代表整个数据集分布的样本。多样性的衡量可以通过聚类(如**core-set**, **Cluster-Margin**)、判别学习(如**VAAL**, **CAL**, **DAL**)
- **混合策略 (hybrid strategy)**：选择既具有不确定性又具有代表性的样本。样本的不确定性和代表性既可以同时估计(如**exploration-exploitation**, **BatchBALD**, **BADGS**, **Active DPP**, **VAAL**, **MAL**)，也可以分两阶段估计(如**Suggestive Annotation**, **DBAL**)。

### ⚪ [<font color=Blue>迁移学习 (Transfer Learning)</font>](https://0809zheng.github.io/2020/05/22/transfer-learning.html)

**迁移学习**是指将解决某个问题时获取的知识应用在另一个不同但相关的问题中。根据源域数据和目标域数据的标签存在情况，迁移学习可以细分为：
- 源域数据有标签，目标域数据有标签：微调(**Fine Tuning**)
- 源域数据有标签，目标域数据无标签：领域自适应(**Domain Adaptation**)、零样本学习(**Zero-Shot Learning**)
- 源域数据无标签，目标域数据有标签：**self-taught learning**
- 源域数据无标签，目标域数据无标签：**self-taught clustering**

模型**微调**是指用带有标签的源域数据预训练模型后，再用带有标签的目标域数据微调模型。

**领域自适应**是指通过构造合适的特征提取模型，使得源域数据和目标域数据的特征落入相同或相似的特征空间中，再用这些特征解决下游任务。常用的领域自适应方法包括：
- 基于差异的方法：直接计算和减小源域和目标域数据特征向量的差异，如**Deep Domain Confusion**, **Deep Adaptation Network**, **CORAL**, **CMD**。
- 基于对抗的方法：引入域判别器并进行对抗训练，如**DANN**, **SDT**, **PixelDA**。
- 基于重构的方法：引入解码器重构输入样本，如**Domain Separation Network**。

### ⚪ [<font color=Blue>终身学习（lifelong learning）</font>](https://0809zheng.github.io/2020/05/21/lifelong-learning.html)

**终身学习**也叫**持续学习（Continuous Learning, Never Ending Learning）**或**增量学习（Incremental Learning）**；是指把模型应用到新任务后，对之前的任务和新任务都能有较好的表现；作为对比，迁移学习则不能保证模型在之前的任务上还有较好的表现。

一些常见的终身学习算法包括多任务学习、**Elastic Weight Consolidation**、**Gradient Episodic Memory**、**Progressive Neural Networks**、**Net2Net**、**Curriculum Learning**、**SupSup**。

### ⚪ [<font color=Blue>元学习（meta-learning）</font>](https://0809zheng.github.io/2020/05/20/meta-learning.html)

**元学习（Meta Learning）**又叫**学会学习（Learning to learn）**，是指给定数据集后，训练一个函数$F$，使得该函数$F$能够选择一个合适的函数$f$解决问题。

一些常见的元学习算法包括**MAML**、**Raptile**、**iMAML**。

## (3) 深度学习的技巧

### ⚪ [<font color=Blue>图像的数据增强 (Image Augmentation)</font>](https://0809zheng.github.io/2021/11/22/dataaugment.html)

**图像数据增强**通过随机变换、样本重组或生成模型扩展训练分布，并把任务先验编码成不变性或等变性约束。它不仅是正则化技巧，也是检测、分割、自监督学习和小样本学习中定义训练样本的重要组成部分。

常用方法按生成机制分为六族：
- **几何与光度变换**：随机裁剪, 翻转, 旋转, 仿射, 颜色抖动, 模糊, 噪声与压缩退化
- **区域遮挡与内容保留**：**Cutout**, **Random Erasing**, **Hide-and-Seek**, **GridMask**, **RandConv**, **KeepAugment**
- **混合样本增强**：**Mixup**, **Manifold Mixup**, **CutMix**, **RICAP**, **FMix**, **Puzzle Mix**, **ResizeMix**, **SaliencyMix**, **SnapMix**, **TransMix**
- **自动与鲁棒增强**：**AutoAugment**, **PBA**, **Fast AutoAugment**, **RandAugment**, **TrivialAugment**, **AugMix**
- **任务感知增强**：小目标复制粘贴, **Mosaic**, **Copy-Paste**, **ClassMix**, **ReLabel**, 自监督多裁剪
- **生成式增强**：类别/文本条件生成, 图像编辑, 检测框/掩码/密度图等结构条件扩散生成

### ⚪ [<font color=Blue>混合精度训练 (Mixed Precision Training)</font>](https://0809zheng.github.io/2020/04/30/mpt.html)

混合精度训练是指在训练深度学习模型的过程中，同时使用不同的数值精度（如半精度浮点数**float16**和单精度浮点数**float32**）进行计算，以提高计算速度并降低内存占用。混合精度训练的关键技巧包括**FP32**权重备份、损失缩放和改进算术方式。


### ⚪ [<font color=Blue>长尾分布 (Long-Tailed)</font>](https://0809zheng.github.io/2020/03/02/optimization.html)

实际应用中的数据集大多服从**长尾分布**，即少数类别(**head class**)占据绝大多数样本，多数类别(**tail class**)仅有少量样本。解决长尾分布问题的方法包括：
- 重采样 **Re-sampling**：通过对**head class**进行欠采样或对**tail class**进行过采样，人为地构造类别均衡的数据集。包括**Random under/over-sampling**, **Class-balanced sampling**, **Meta Sampler**等。
- 重加权 **Re-weighting**：在损失函数中对不同类别样本的损失设置不同的权重，通常是对**tail class**对应的损失设置更大的权重。其中在$\log$运算之外调整损失函数的本质是在调节样本权重或者类别权重(如**Inverse Class Frequency Weighting**, **Cost-Sensitive Cross-Entropy Loss**, **Focal Loss**, **Class-Balanced Loss**)。在$\log$运算之内调整损失函数的本质是调整**logits**得分$z$，从而缓解对**tail**类别的负梯度(如**Equalization Loss**, **Equalization Loss v2**, **Logit Adjustment Loss**, **Balanced Softmax Loss**, **Seesaw Loss**)。
- 其他方法：一些方法将长尾分布问题解耦为特征的表示学习和特征的分类。一些方法按照不同类别的样本数量级对类别进行分组(如**BAGS**)。


### ⚪ [<font color=Blue>对抗训练 (Adversarial Training)</font>](https://0809zheng.github.io/2020/07/26/adversirial_attack_in_classification.html)

**对抗训练**是指通过构造对抗样本，对模型进行对抗攻击和防御来增强模型的稳健性。对抗训练的一般形式如下：

$$
\mathcal{\min}_{\theta} \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \mathcal{\max}_{\Delta x \in \Omega}  \mathcal{L}(x+\Delta x,y;\theta) \right]
$$

- 对抗攻击是指想办法造出更多的对抗样本；常用的对抗攻击方法包括：**FGSM**, **I-FGSM**, **MI-FGSM**, **NI-FGSM**, **DIM**, **TIM**, **One Pixel Attack**, **Black-box Attack**。
- 对抗防御是指想办法让模型能正确识别更多的对抗样本；常用的对抗防御方法包括**Smoothing**, **Feature Squeezing**, **Randomization**, **Proactive defense**。

### ⚪ [<font color=Blue>大模型的参数高效微调 (Parameter-Efficient Fine-Tuning)</font>](https://0809zheng.github.io/2023/02/02/peft.html)

将预训练好的大型模型在下游任务上进行微调已成为处理不同任务的通用范式；但是随着模型越来越大，对模型进行全部参数的微调（**full fine-tuning**）变得非常昂贵。**参数高效微调**是指冻结预训练模型的大部分参数，仅微调少量或额外的模型参数。

参数高效微调方法有以下几种形式：
- 增加额外参数(**addition**)：在原始模型中引入额外的可训练参数，如**Adapter**, **AdapterFusion**, **AdapterDrop**, **P-Tuning**, **Prompt Tuning**, **Prefix-Tuning**, **P-Tuning v2**, **Ladder Side-Tuning**
- 选取部分参数(**specification**)：指定原始模型中的部分参数可训练，如**BitFit**, **Child-Tuning**
- 重参数化(**reparameterization**)：将微调过程重参数化为低维子空间的优化，如**Diff Pruning**, **LoRA**, **AdaLoRA**, **QLoRA**, **GLoRA**, **LoRA+**, **LoRA-GA**
- 混合方法：如**MAM Adapter**, **UniPELT**

- [深度学习的可解释性](https://0809zheng.github.io/2020/04/28/explainable-DL.html)



## () 网络压缩
网络压缩旨在平衡网络的准确性和运算效率。
压缩预训练的网络 设计新的网络结构
- [网络压缩](https://0809zheng.github.io/2020/05/01/network-compression.html)：网络剪枝、知识蒸馏、结构设计、模型量化

# 2. 深度神经网络的类型

## (1) 卷积神经网络

### ⚪ [<font color=Blue>卷积神经网络(Convolutional Neural Network)的基本概念</font>](https://0809zheng.github.io/2020/03/06/CNN.html)

**卷积神经网络**是由卷积层、激活函数和池化层堆叠构成的深度神经网络，可以从图像数据中自适应的提取特征。

卷积层是一种局部的互相关操作，使用卷积核在输入图像或特征上按照光栅扫描顺序滑动，并通过局部仿射变换构造输出特征。它引入了三条针对自然图像的**归纳偏置**：稀疏连接、权值共享和平移等变性；这既是卷积高效的原因，也是后续各种改进试图放松的对象。

卷积层的**基本超参数与形态**：$1\times 1$卷积, 扩张卷积(**Dilated Conv**, **HDC**, **IC-Conv**), 转置卷积（及棋盘效应）, 子像素卷积(**PixelShuffle**, **ICNR**), 组卷积, 深度卷积。

卷积层的基准形式为$y(p_0)=\sum_{p_n \in \mathcal{R}} w(p_n)\cdot x(p_0+p_n)+b$，改进可以按“改动了基准公式的哪一部分”分为六族：
- **采样位置自适应**（$p_0+p_n \to p_0+p_n+\Delta p_n$）：主动卷积**ACU**, 可变形卷积**Deformable Conv v1,v2**, **DCNv3**(**InternImage**), **DCNv4**, **LDConv**, 圆形卷积
- **卷积核权重动态化**（$w(p_n) \to w(p_n \mid x)$）：**CondConv**, **DynamicConv**, **DyNet**, **ODConv**, **DRConv**, **Involution**, **LR-Net**, **Conv2Former**
- 改变**聚合方式**（$\sum w \cdot x \to \sum w \cdot g(x)$）：差分卷积(中心差分卷积**CDC**, 交叉中心差分卷积, 像素差分卷积**PDC**), 部分卷积**Partial Conv**, 门控卷积**Gated Conv**, 稀疏卷积(空间稀疏卷积, 子流形稀疏卷积)
- **分解与重参数化**卷积核（$w \to w_1 \otimes w_2 \otimes \cdots$）：空间可分离卷积, 深度可分离卷积, 平展卷积, **PConv**, **DO-Conv**, **ACNet**, **RepVGG**, **DBB**
- 扩大**感受野**（改变$\mathcal{R}$）：**RepLKNet**, **SLaK**, **UniRepLKNet**, **InceptionNeXt**, 八度卷积**OctConv**, 快速傅里叶卷积**FFC**, 小波卷积**WTConv**
- 注入**位置信息**（$x \to [x; \text{coord}]$）：**CoordConv**, 零填充与大核带来的隐式位置编码

卷积的**高效实现**决定了**FLOPs**能否兑换成实际速度：**im2col + GEMM**, **FFT**卷积, **Winograd**卷积；深度卷积与大核卷积属于访存受限算子，需要专用核实现。

### ⚪ [<font color=Blue>卷积神经网络中的池化(Pooling)层</font>](https://0809zheng.github.io/2021/07/02/pool.html)


**池化层**可以对特征图进行降采样，从而减小计算成本、扩大感受野、降低过拟合的风险。任何池化都可以拆解为**聚合**与**抽取**两步，据此可以按三个维度分类：聚合函数是确定性还是随机的、是否包含可学习参数、以及输出的空间尺寸是缩小还是坍缩为一个向量。理解这一点也就理解了池化与平移不变性的真实关系——不变性主要来自数据增强与全局池化，而抽取步骤恰恰是破坏它的元凶。

**通用的降采样池化**分为三族：
- **确定性池化**：最大池化, 平均池化, 混合池化与门控池化, 幂平均池化($L_p$), **LSE**池化, **AvgMax**池化, 抗锯齿池化**BlurPool**, 小波池化
- **随机池化**：随机池化**Stochastic Pooling**, 分数最大池化**FMP**, 随机空间采样池化**S3Pool**, 池化窗口内的**Max-Pooling Dropout**
- **可学习与保细节的池化**：细节保留池化**DPP**, 局部重要性池化**LIP**, 软池化**SoftPool**, 动态优化池化**DynOPool**, 带步长的卷积, 空间到通道的无损降采样**SPD-Conv**

**面向下游任务的池化**把特征坍缩为定长表示：
- **全局池化与二阶统计**：全局平均池化**GAP**, 全局最大池化**GMP**, 广义均值池化**GeM**, 协方差池化(**iSQRT-COV**), 双线性池化
- **空间对齐与多尺度池化**：空间金字塔池化**SPP**, 感兴趣区域池化**RoI Pooling**, **RoI Align**, **PPM**与**ASPP**
- **序列与集合的池化**：注意力池化, 多头注意力池化**PMA**, **NetVLAD**, **SimPool**, **Token Merging**

与池化相对的**反池化与上采样**：最大反池化, 平均反池化, 最近邻/双线性插值, 转置卷积, 子像素卷积。

### ⚪ [<font color=Blue>卷积神经网络中的注意力机制(Attention Mechanism)</font>](https://0809zheng.github.io/2020/11/18/AinCNN.html)

卷积神经网络中的**注意力机制**表现为在特征的某个维度上计算相应**统计量**，并根据所计算的统计量对该维度上的每一个元素赋予不同的权重，用以增强网络的特征表达能力。它把卷积"对所有通道、所有位置一视同仁"的固定权重变成了**输入自适应的重标定**，并且几乎所有模块都遵循同一个通用形式：**聚合上下文 → 生成权重 → 融合回特征**。这类模块的共同特点是即插即用、开销极低，但也因此存在增益不稳定、难以复现的问题。

卷积层的特征维度包括通道维度和空间维度，因此注意力机制可以应用在不同维度上：
- **通道注意力(Channel Attention)**：（基于全局池化）**SENet**, **CMPE-SE**, **GENet**, **TSE**, **SPANet**, **ECA-Net**；（更强的统计量）**SRM**风格池化, **GSoP**全局二阶池化, **FcaNet**频域通道注意力；（归一化与门控视角）**GCT**, **NAM**, **ATAC**
- **空间注意力(Spatial Attention)**：**Residual Attention Network**, **SGE**, **ULSAM**, **CSRA**
- **通道与空间的混合**：（并联）**scSE**, **BAM**, **SA-Net**, **Triplet Attention**；（串联）**CBAM**；（坐标分解与三维权重）**Coordinate Attention**, **ELA**, **EMA**, **SCSA**, **SCNet**
- **分支、核选择与大核注意力**：（多分支与核选择）**SKNet**, **Split-Attention**(**ResNeSt**), **EPSA**；（跨层与跨分支融合）**AFF**, **Interflow**；（大核注意力）**LKA**/**VAN**, **MSCA**, **LSK**, **D-LKA**, **CAA**
- **无参数与能量函数视角**：**SimAM**（由空间抑制的能量函数解析导出，零可学习参数即可产生三维权重）
- **其他作用对象与模块间连接**：（作用在卷积权重与样本上）**WE**, **AW-Conv**, **BA$^2$M**；（模块之间的连接）**DIA**, **DCANet**

### ⚪ [<font color=Blue>卷积神经网络中的自注意力机制(Self-Attention Mechanism)</font>](https://0809zheng.github.io/2020/11/21/SAinCNN.html)

卷积神经网络中的**自注意力机制**表现为**非局部滤波**操作，通过计算任意两个位置之间的关系直接捕捉远程依赖，而不用局限于相邻点，相当于构造了一个**和特征图尺寸一样大**的卷积核，从而可以捕捉更多信息。它的通用形式$y_i=\frac{1}{\mathcal{C}(x)}\sum_j f(x_i,x_j)g(x_j)$源自图像去噪中的**非局部均值**，与**Transformer**的自注意力只是相似度函数与归一化方式的差别。这类模块的核心矛盾在于$O(N^2)$的复杂度：特征图的空间尺寸稍大就无法承受，因此绝大多数后续工作都在解决这个问题。

卷积神经网络中的自注意力机制包括：
- **Non-local及其直接改进**：**Non-local Block**, **RNL**（用区域代替单点）, **AAConv**（多头与相对位置编码）, **CGNL**（把通道纳入成对建模）, **DNL**（解耦成对项与一元项）, **SNL**（谱视角下的对称化）
- **降低复杂度：稀疏化与分解**：**CCNet**（十字形注意力与循环覆盖）, **ISANet**（长程与短程交错）, **Axial Attention**/**Axial-DeepLab**（沿坐标轴分解）, **NLSA**（哈希做内容自适应稀疏）, **LightNL**
- **降低复杂度：低秩与矩阵分解**：**Efficient Attention**（把**softmax**拆到两侧实现线性化）, **GCNet**（退化为**SE**）, **ANNNet**（金字塔池化采样键值）, **A$^2$-Net**（双线性池化收集全局描述子）, **EMANet**（**EM**算法迭代出紧凑基）, **OCRNet**（类别中心作中间表示）, **Hamburger**（直接做矩阵分解）, **FLatten**, **Agent Attention**
- **多维度联合的自注意力**：**DANet**（空间与通道对偶）, **PSA**（极化自注意力）, **DMSANet**, **PSANet**（用卷积直接预测注意力图）
- **把自注意力当作卷积的替代算子**：**SASA**, **HaloNet**, **SAN**（**pairwise**与**patchwise**两种聚合）, **LambdaNetworks**（把上下文编码成线性函数）, **CoTNet**（静态上下文引导动态注意力）

### ⚪ [卷积神经网络的可视化](https://0809zheng.github.io/2020/12/16/custom.html)

### ⚪ [<font color=Blue>轻量级(LightWeight)卷积神经网络</font>](https://0809zheng.github.io/2021/09/10/lightweight.html)

**轻量级**网络设计旨在设计计算复杂度更低的卷积网络结构。一个贯穿全篇的关键认识是：**FLOPs不等于实际延迟**。深度卷积、通道重排、逐元素加法等算子的**算术强度**很低，属于访存受限，因此"**FLOPs**降低$8$倍、实测只快$2$倍"是常态；评价轻量模型必须在目标硬件上实测时延，而不能只看参数量与**FLOPs**。

轻量化的基本手段包括分解卷积、通道重排与稀疏连接、特征复用与廉价算子、降分辨率降通道与缩放、以及替代乘法。据此常见的轻量级网络可以分为六条路线：
- **分解卷积路线**：**SqueezeNet**, **SqueezeNext**, **Xception**, **MobileNet V1,2,3**, **MobileNeXt**, **ESPNet**/**ESPNetv2**, **MixConv**, **DiCENet**
- **分组卷积与通道重排路线**：**ShuffleNet V1,2**, **IGCNet V1,2,3**, **ChannelNet**, **MicroNet**, **CondenseNet**
- **特征冗余与廉价操作路线**：**GhostNet V1,2,3**, **CompConv**, **ShiftNet**, **FasterNet**(**PConv**), **StarNet**
- **复合缩放与架构搜索路线**：**MnasNet**, **EfficientNet V1,2**, **RegNet**, **Slimmable Networks**
- **替换乘法的路线**：**AdderNet**（用**L1**距离代替卷积乘法）, **Mitchell**近似（在对数域把乘法变成加法）
- **面向推理延迟设计的路线**：**PeleeNet**, **HarDNet**, **RepVGG**, **MobileOne**, **RepViT**, **MobileNetV4**, **EfficientFormer**/**EfficientViT**, **MobileMamba**





## (2) 循环神经网络


### ⚪ [<font color=Blue>循环神经网络(Recurrent Neural Network)的基本概念</font>](https://0809zheng.github.io/2020/03/07/RNN.html)

**循环神经网络(RNN)**可以处理输入长度不固定的文本等时间序列数据。**RNN**每一时刻的隐状态$h_t$不仅和当前时刻的输入$x_t$相关，也和上一时刻的隐状态$h_{t-1}$相关。**RNN**具有通用近似性、图灵完备性等特点，但这些理论表达力与实际可学习性之间存在明显的鸿沟。

$$ h_t = f(h_{t-1},x_t), \quad y_t = g(h_t) $$

**RNN**的训练依赖**随时间反向传播(BPTT)**（也可用**实时循环学习RTRL**，以$O(d^3)$的计算换取$O(1)$的时间复杂度和在线更新能力）。它存在**长程依赖问题**：理论上可以建立长时间间隔的状态之间的依赖关系，但由于梯度沿时间维度以雅可比矩阵连乘的形式传播，实际上只能学习到短期的依赖关系。缓解长程依赖的**非门控手段**包括梯度裁剪, 恒等初始化与**IRNN**, 酉循环网络**uRNN**, 跳跃连接与多时间尺度, 截断**BPTT**, 归一化。

更根本的解决措施是引入**门控机制**，用加性的状态更新把连乘的雅可比矩阵变成接近恒等的传播路径。门控循环网络可以分为：
- **经典门控单元**：**LSTM**, **GRU**, **MGU**, **JANET**与**chrono**初始化
- **面向并行化的简化循环单元**（去掉门对$h_{t-1}$的依赖，从而可用并行扫描把串行递推压缩到对数深度）：**QRNN**, **SRU**, **IndRNN**, **LRU**
- **结构化与连续时间的记忆**：**ON-LSTM**（把句法层级编码进神经元顺序）, **LMU**（用勒让德多项式最优压缩历史）, 连续时间**RNN**与**Neural ODE**
- **隐状态即模型**：**TTT**（把隐状态换成一个在测试时通过自监督梯度下降持续更新的小模型）
- 门控循环网络**复兴**：**xLSTM**（指数门控、矩阵记忆与稳定器状态）, **minLSTM**/**minGRU**（去掉隐状态依赖后可完全并行）

也可以通过增加循环层的深度增强**RNN**的特征提取能力，包括**Stacked RNN**, **Bidirectional RNN**, 残差与高速连接；循环网络**专用的正则化**包括**Variational**/**Recurrent Dropout**（跨时间步共享掩码）与**Zoneout**（随机保持上一时刻的状态）。

### ⚪ [**<font color=Blue>序列到序列模型 (Sequence to Sequence)</font>**](https://0809zheng.github.io/2020/04/21/sequence-2-sequence.html)

**序列到序列(Sequence to Sequence，Seq2Seq)模型**用编码器把变长源序列压成上下文向量，再用解码器以自回归方式生成变长目标序列。它把序列建模与条件语言模型合并到同一张计算图上，是**Transformer**出现之前神经机器翻译、摘要、对话等任务的通用框架。

本文主要内容包括：
- **编码器—解码器结构**：**Cho**的**RNN Encoder-Decoder**、**Sutskever**的多层**LSTM**加反向输入、双向与深层扩展。
- **训练与解码**：教师强制、贪婪搜索、束搜索与长度归一化。
- **训练—推理错配**：**Scheduled Sampling**、**MIXER**与**Self-Critical Sequence Training**等序列级方法。
- **输出词表扩展**：条件**Seq2Seq**、**Pointer Network**、**CopyNet**、**Pointer-Generator**。
- **覆盖度与结构化输出**：**Coverage Mechanism**及其损失约束。

### ⚪ [**<font color=Blue>序列到序列模型中的注意力机制 (Attention Mechanism)</font>**](https://0809zheng.github.io/2020/04/22/attention.html)

**注意力机制(Attention Mechanism)**让解码器每一步都在编码器所有隐状态上重新查询，取代固定的上下文向量，从而突破**Seq2Seq**的信息瓶颈。

本文主要内容包括：
- **两种打分家族**：**Bahdanau**加性注意力与**Luong**乘性注意力，以及得分函数的通用形式（**dot**、**general**、**scaled dot-product**、**location-based**、**cosine**）。
- **注意力的作用范围**：全局注意力对整段源序列打分，局部注意力用单调或预测式对齐把窗口收窄。
- **硬性与软性**：软性注意力可微稳定，硬性注意力配合**REINFORCE**或**Straight-Through**用于**Show, Attend and Tell**等任务，**sparsemax**是二者之间的稀疏折中。
- **结构化约束**：**Coverage Mechanism**避免重复关注、**Monotonic Attention/MoChA**支持流式解码、**HAN**用层次化注意力聚合长文档。

## (3) 自注意力网络

### ⚪ [**<font color=Blue>自注意力机制 (Self-Attention Mechanism)</font>**](https://0809zheng.github.io/2020/04/24/self-attention.html)

**自注意力(Self-Attention)机制**把[注意力机制](https://0809zheng.github.io/2020/04/22/attention.html)收缩到同一条序列内部，用于捕捉单个序列$X$的内部关系。把输入序列$X$映射为查询矩阵$Q$, 键矩阵$K$和值矩阵$V$；根据查询矩阵$Q$和键矩阵$K$生成注意力图，并作用于值矩阵$V$获得自注意力的输出$H$。

$$ H = \operatorname{softmax}(QK^\top/\sqrt{d_k})V $$

本文主要内容包括：
- **算子对比**：卷积、循环与自注意力在每层复杂度、序列操作数与最大路径长度上的差异。
- **QKV**实现：查询/键/值投影、缩放点积注意力、以及输出与残差。
- **多头自注意力**：**narrow**与**wide**两种切分方式，以及$$W^O$$的合并作用。
- **位置编码**：置换等变性、学习式位置嵌入、**Sinusoidal**位置编码。
- **受限与掩码**：**restricted**局部窗口、因果/填充/双向掩码，作为通向长序列注意力的第一步。

### ⚪ [<font color=Blue>Transformer</font>](https://0809zheng.github.io/2020/04/25/transformer.html)

**Transformer**是一个基于多头自注意力机制的深度网络模型，网络结构包括编码器和解码器。编码器生成基于注意力的特征表示，该表示具有从全局上下文中定位特定信息的能力；解码器从特征表示中进行检索。

![](https://pic.imgdb.cn/item/618b94ea2ab3f51d91f6d24e.jpg)

### ⚪ [<font color=Blue>Transformer中的位置编码 (Position Encoding)</font>](https://0809zheng.github.io/2022/07/01/posencode.html)

**Transformer**中的自注意力机制具有置换等变性(**permutation equivariant**)，导致打乱输入序列的顺序只会同步打乱输出而不改变对应关系，因此必须显式注入位置信息以打破全对称性。从**Taylor**展开与注意力**logit**中的位置偏置这一统一视角看，各类位置编码可归纳为：
- **绝对位置编码**：只依赖于单一位置，将绝对位置信息加入到输入序列中，相当于引入索引的嵌入。比如**Sinusoidal**, **Learnable**, **FLOATER**, **Complex-order**
- **相对位置编码**：不同位置的交互项，通过微调自注意力运算过程使其能分辨不同**token**之间的相对位置。比如经典**RPE**, **XLNet**式, **T5**式分桶, **DeBERTa**式解耦, **Swin**式二维偏置, **URPE**
- **旋转位置编码(RoPE)**：用旋转矩阵作用于查询与键，使内积只依赖相对位置，是当前大模型的绝对主流；变体包括二维**RoPE**、混合频率等
- **长度外推(length extrapolation)**：让短序列训练的模型迁移到更长上下文，包括单调偏置类(**ALiBi**、**KERPLE**)、**RoPE**的插值与频率缩放(**位置插值PI**、**NTK-aware**、**YaRN**)、免训练的位置重映射
- **无位置编码与内容相关位置编码**：**NoPE**（因果掩码隐式提供位置）、**CoPE**（内容决定位置）、**CPVT**（卷积零填充生成条件位置编码）


### ⚪ [<font color=Blue>降低Transformer的计算复杂度</font>](https://0809zheng.github.io/2021/07/12/efficienttransformer.html)

自注意力运算中**计算注意力矩阵**以及**加权求和计算输出**这两个步骤引入了$O(N^2)$的计算复杂度。降低复杂度有四条改进路线，另有一类专门针对推理阶段的显存瓶颈：
- **稀疏注意力（改变注意力矩阵的支撑集）**：让每个位置只与一部分位置计算相关性。固定模式如**Sparse Transformer**, **Longformer**, **Big Bird**；内容自适应如**Reformer**（**LSH**分桶）, **Routing Transformer**, **Clustered Attention**。
- **低秩与核化（改变注意力矩阵的秩）**：低秩投影把键值压缩到固定长度，如**Linformer**, **Nyströmformer**, **Synthesizer**；核化把**softmax**替换为特征映射内积实现线性化，如**Linear Transformer**, **Performer**, **Efficient Attention**, **External Attention**, **FLASH**。
- **递归与状态（把注意力写成RNN）**：片段递归如**Transformer-XL**；现代线性注意力用统一的门控状态更新模板$S_i=\Lambda_i\odot S_{i-1}+\phi(k_i)v_i^\top$，如**RetNet**, **RWKV**, **GLA**, **DeltaNet**（详见[状态空间模型](https://0809zheng.github.io/2024/07/01/ssm.html)）。
- **IO感知（不改数学定义，只改访存模式）**：如**FlashAttention**系列通过分块与重计算减少显存读写，以及分布式与服务化优化。
- **推理阶段的KV cache**：减少**KV**头（**MQA**, **GQA**, **MLA**）或减少缓存的**token**数量，是长上下文自回归解码的关键瓶颈。


## (4) 深度生成模型

**生成模型**(**generative model**)是指使用带参数$\theta$的概率分布$p_{\theta}(x)$拟合已有数据样本集$$\{x\}$$。由于概率分布$p_{\theta}(x)$的形式通常是未知的，可以将其假设为离散型或连续型分布；若进一步引入**隐变量(latent variable)** $z$，则可以间接地构造概率分布$p_{\theta}(x)$：

$$ p_{\theta}(x) = \int p_{\theta}(x,z) dz = \int p_{\theta}(x|z)p(z) dz  $$

参数$\theta$的求解可以通过极大似然估计。若记真实数据分布为$\tilde{p}(x)$，则优化目标为最大化对数似然$$\Bbb{E}_{x\text{~}\tilde{p}(x)}[\log p_{\theta}(x)]$$。由于该算式包含积分运算，直接求解比较困难；不同的生成模型通过不同的求解技巧避开这个困难。


### ⚪ 自回归模型 (Auto-Regressive)

条件分布的乘积

从最严格的角度来看，图像应该是一个离散的分布，因为它是由有限个像素组成的，而每个像素的取值也是离散的、有限的，因此可以通过离散分布来描述。这个思路的成果就是PixelRNN一类的模型了，我们称之为“自回归流”，其特点就是无法并行，所以计算量特别大。所以，我们更希望用连续分布来描述图像。当然，图像只是一个场景，其他场景下我们也有很多连续型的数据，所以连续型的分布的研究是很有必要的。

的本质，就是希望用一个我们知道的概率模型来拟合所给的数据样本，也就是说，我们得写出一个带参数θ的分布qθ(x)。然而，我们的神经网络只是“万能函数拟合器”，却不是“万能分布拟合器”，也就是它原则上能拟合任意函数，但不能随意拟合一个概率分布，因为概率分布有“非负”和“归一化”的要求。这样一来，我们能直接写出来的只有离散型的分布，或者是连续型的高斯分布。

### ⚪ [<font color=Blue>生成对抗网络 (Generative Adversarial Network)</font>](https://0809zheng.github.io/2022/02/01/gan.html)

**生成对抗网络**通过交替优化的对抗训练绕开了似然的直接求解，使用生成器$G$构造真实分布的近似分布$$P_G(x)$$，并使用判别器衡量生成分布和真实分布之间的差异。

$$ \begin{aligned} \mathop{ \min}_{G} \mathop{\max}_{D}  \Bbb{E}_{x \text{~} P_{data}(x)}[\log D(x)] + \Bbb{E}_{z \text{~} P(z)}[\log(1-D(G(z)))] \end{aligned} $$

生成对抗网络的设计是集目标函数、网络结构、优化过程于一体的，**GAN**的各种变体也是基于对这些方面的改进：
- 改进目标函数：基于分布散度(如**f-GAN**, **BGAN**, **Softmax GAN**, **RGAN**, **LSGAN**, **WGAN-div**, **GAN-QP**, **Designing GAN**)、基于积分概率度量(如**WGAN**, **WGAN-GP**, **DRAGAN**, **SN-GAN**, **GN-GAN**, **GraN-GAN**, **c-transform**, **McGAN**, **MMD GAN**, **Fisher GAN**)
- 改进网络结构：调整神经网络(如**DCGAN**, **SAGAN**, **BigGAN**, **Self-Modulation**, **StyleGAN1,2,3**, **TransGAN**)、引入编码器(如**VAE-GAN**, **BiGAN**, **VQGAN**)、使用能量模型(如**EBGAN**, **LSGAN**, **BEGAN**, **MAGAN**, **MEG**)、由粗到细的生成(如**LAPGAN**, **StackGAN**, **PGGAN**, **SinGAN**)
- 改进优化过程：**TTUR**, **Dirac-GAN**, **VDB**, **Cascading Rejection**, **ADA**, **Hubness Prior**, **R3GAN**
- 其他应用：条件生成(如**CGAN**, **InfoGAN**, **ACGAN**, **Projection Discriminator**)、[<font color=Blue>图像到图像翻译</font>](https://0809zheng.github.io/2020/05/23/image_translation.html)(有配对数据, 如**Pix2Pix**, **BicycleGAN**, **LPTN**; 无配对数据, 如**CoGAN**, **PixelDA**, **CycleGAN**, **DiscoGAN**, **DualGAN**, **UNIT**, **MUNIT**, **TUNIT**, **StarGAN**, **StarGAN v2**, **GANILLA**, **NICE-GAN**, **CUT**, **SimDCL**)、超分辨率(如**SRGAN**, **ESRGAN**)、图像修补(如**Context Encoder**, **CCGAN**, **SPADE**)、机器学习应用(如**Semi-Supervised GAN**, **AnoGAN**, **ClusterGAN**)

### ⚪ [<font color=Blue>变分自编码器 (Variational Autoencoder)</font>](https://0809zheng.github.io/2022/04/01/vae.html)

**变分自编码器**的优化目标不是对数似然，而是对数似然的变分下界：

$$  \log p_{\theta}(x)  \geq \mathbb{E}_{z \text{~} q_{\phi}(z|x)} [\log p_{\theta}(x | z)] - KL[q_{\phi}(z|x)||p(z)]  $$

**VAE**的优化目标共涉及三个不同的概率分布：由概率编码器表示的后验分布$q_{\phi}(z\|x)$、隐变量的先验分布$p(z)$以及由概率解码器表示的生成分布$p_{\theta}(x\|z)$。对**VAE**的各种改进可以落脚于对这些概率分布的改进：

- 后验分布$q(z\|x)$：后验分布为模型引入了正则化；一种改进思路是通过调整后验分布的正则化项增强模型的解耦能力(如**β-VAE**, **Disentangled β-VAE**, **InfoVAE**, **DIP-VAE**, **FactorVAE**, **β-TCVAE**, **HFVAE**)。
- 先验分布$p(z)$：先验分布描绘了隐变量分布的隐空间；一种改进思路是通过引入标签实现半监督学习(如**CVAE**, **CMMA**)；一种改进思路是通过对隐变量离散化实现聚类或分层特征表示(如**Categorical VAE**, **Joint VAE**, **VQ-VAE**, **VQ-VAE-2**, **FSQ**)；一种改进思路是更换隐变量的概率分布形式(如**Hyperspherical VAE**, **TD-VAE**, **f-VAE**, **NVAE**)。
- 生成分布$p(x\|z)$：生成分布代表模型的数据重构能力；一种改进思路是将均方误差损失替换为其他损失(如**EL-VAE**, **DFCVAE**, **LogCosh VAE**)。
- 改进整体损失函数：也有方法通过调整整体损失改进模型，如紧凑变分下界(如**IWAE**, **MIWAE**)或引入**Wasserstein**距离(如**WAE**, **SWAE**)。
- 改进模型结构：如**BN-VAE**通过引入**BatchNorm**缓解**KL**散度消失问题；引入对抗训练(如**AAE**, **VAE-GAN**)。

### ⚪ [<font color=Blue>流模型 (Flow-based Model)</font>](https://0809zheng.github.io/2022/05/01/flow.html)

**流模型**通过一系列可逆变换(双射函数$f$)建立较为简单的先验分布$p(z)$与较为复杂的实际数据分布$p(x)$之间的映射关系：

$$ \begin{aligned} x&=f_K \circ \cdots \circ f_1(z) \\ p(x) &= p(z)\cdot |\prod_{k=1}^{K} \det J_{f_k}(z_{k-1})|^{-1} \end{aligned} $$

由于流模型给出了概率分布$p(x)$的显式表达式，可直接最大化对数似然：

$$ \begin{aligned}  \log p(x)  = \log  p(z) - \sum_{k=1}^{K}\log  | \det J_{f_k}(z_{k-1})| \end{aligned}  $$

从优化目标中可以看出，流模型是由先验分布$p(z)$和双射函数$x=f(z)$唯一确定的。根据双射函数的不同设计思路，流模型分为以下两类：
- **标准化流**(**Normalizing Flow**)：通过数学定理与性质设计**Jacobian**行列式$\det J_{f}(z)$容易计算的双射函数$x=f(z)$。标准化流是最基础的流模型，事实上其他类别的流模型可以看作标准化流的延申。这类模型包括**Normalizing Flow**, **iResNet**等。
- **自回归流**(**Autoregressive Flow**)：把双射函数$x=f(z)$建模为自回归模型，即$x$的第$i$个维度$x_i$的生成只依赖于前面的维度$x_{1:i-1}$(自回归流)或$z_{1:i-1}$(逆自回归流)，此时**Jacobian**矩阵$J_{f}(z)$为三角矩阵，行列式容易计算。这类模型包括**IAF**, **MAF**, **NICE**, **Real NVP**, **Glow**, **Flow++**等。


### ⚪ [扩散模型]()

### ⚪ [<font color=Blue>流匹配模型 (Flow Matching Model)</font>](https://0809zheng.github.io/2025/05/01/flowmatching.html)

**流匹配**通过回归一个时变速度场$v_\theta(x,t)$来构造从先验分布$p_0$到数据分布$p_1$的连续变换，其轨迹由**ODE** $dx_t/dt = v_\theta(x_t,t)$ 给出。直接匹配边际速度场$u_t(x)$是不可行的，但可以证明**条件流匹配**目标与之具有相同的梯度：

$$ \begin{aligned} \mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(z), p_t(x|z)} \| v_\theta(x, t) - u_t(x|z) \|^2, \quad \nabla_\theta \mathcal{L}_{FM} = \nabla_\theta \mathcal{L}_{CFM} \end{aligned} $$

其中条件路径$p_t(x\|z)$与条件速度$u_t(x\|z)$由人为设计，因而有解析形式，使训练完全**无模拟**。若采用仿射高斯路径$x_t=\alpha_tx_1+\sigma_tx_0$，则速度、$x_1$、$x_0$与分数四种预测目标可以相互线性换算，扩散模型的概率流**ODE**正是其中一个特例；流匹配的额外自由度在于路径调度、源分布的任意性以及回归目标与时间加权的选择。流匹配模型的研究可以分为以下几类：
- **连续流匹配**：处理欧氏空间或流形上的连续数据，区别在于条件变量$z$（如何配对起点与终点）与条件路径$p_t(x\|z)$（如何在这对点之间移动）的选择。包括**InterFlow**, **FM**, **I-CFM**, **OT-CFM**, **SB-CFM**, **Rectified Flow**, **2-Rectified Flow++**, **Equivariant OT FM**, **RFM**, **FFM**, **FoldFlow**, **OFM**, **COT-FM**, **Metric FM**, **VFM**, **Meta FM**, **WFM**
- **离散流匹配**：处理文本、序列、分子图等分类数据。连续状态方法把离散状态松弛到概率单纯形上并沿测地线流动（如**Dirichlet FM**, **Fisher-Flow**, **SFM**, **GAF**, **Gumbel-Softmax Flow**, **α-Flow**）；离散状态方法则全程保持状态离散，用连续时间马尔可夫链的速率矩阵描述状态跳跃（如**DFM**, **Discrete Guidance**, **Discrete Flow Matching**, **DFM-KO**）
- **“拉直”流匹配与少步采样**：轨迹越直，用越少的**ODE**求解步数就能保持精度。思路包括降低轨迹曲率、改进耦合、递归**Reflow**与蒸馏、以及直接建模跨步长的平均速度。包括**Minimizing Trajectory Curvature**, **Multisample FM**, **InstaFlow**, **BOSS**, **PeRFlow**, **Flow Map Matching**, **Consistency-FM**, **Shortcut Models**, **MeanFlow**
- **流匹配的条件生成**：分为在训练时把条件作为输入的模型内条件化，以及在推理时引导无条件模型的指导方法（如预测器指导、无分类器指导**CFG**与**∆FM**、免训练指导**TFG-Flow**）
- **多边际流匹配**：除首尾分布外还给定中间时刻的观测快照，需要恢复穿过所有快照且平滑的轨迹。包括**3MSBM**（提升到相空间、优化加速度）, **MMSFM**（重叠窗口上的测度值样条）, **OTP-FM**（中间边际诱导的势能项）
- **规模化实践**：在固定架构下逐维消融扩散与流匹配的差异，并确立大规模图像生成的配方。包括**SiT**（连续时间+线性路径+速度预测+可调扩散系数的**SDE**采样）, **SD3**（**logit-normal**时间步采样、**MM-DiT**联合注意力、分辨率偏移）

### ⚪ 其他生成网络

[Generative Moment Matching Network](https://0809zheng.github.io/2022/03/27/gmmn.html)

  
## (4) 其他类型的神经网络

### ⚪ [<font color=Blue>递归神经网络 (Recursive Neural Network)</font>](https://0809zheng.github.io/2020/03/08/recursive-neural-network.html)

**递归神经网络**在树或有向无环图上递归地共享组合函数：先编码叶节点，再根据子节点表示计算父节点，直至得到根节点表示。循环神经网络是其链式特例，而**TreeRNN**也可以看成树上的单次有向消息传递。它适合句法树、程序抽象语法树、场景层级和**3D**部件树等具有可靠层级结构的数据。

递归网络的方法可以分为四族：
- **基础组合与结构训练**：二叉**TreeRNN**, 节点级监督, 结构反向传播(**BPTS**), 成分树与依存树
- **增强组合函数**：递归自编码器(**RAE**), 矩阵-向量递归网络(**MV-RNN**), 递归神经张量网络(**RNTN**)
- **门控与高效树计算**：**Child-Sum Tree-LSTM**, **N-ary Tree-LSTM**, **SPINN**
- **潜在结构与现代混合模型**：**RL-SPINN**, **Gumbel Tree-LSTM**, 可微**chart parser**, **Tree Transformer**

显式树结构能缩短句法相关成分之间的路径并提供可解释的组合过程，但依赖解析质量、难以批量并行。对一般自然语言任务，预训练Transformer通常是默认选择；对AST、XML和部件树等原生层级输入，递归网络仍具有直接而有效的结构归纳偏置。

### ⚪ [**<font color=Blue>图神经网络 (Graph Neural Network)</font>**](https://0809zheng.github.io/2020/03/09/graph-neural-network.html)

**图神经网络(Graph Neural Network，GNN)**通过共享的局部函数在节点之间传递消息，并用排列不敏感的聚合与读出函数学习节点、边和整张图的表示。谱图卷积与空间邻域聚合可以统一到“传播、聚合、更新、读出”的视角下理解。

本文主要内容包括：
- **统一框架**：**消息传递神经网络(Message Passing Neural Network，MPNN)**、排列等变性与不变性、感受野、归纳学习与直推学习。
- **经典模型**：**Spectral CNN、ChebNet、GCN、SGC、APPNP、GraphSAGE、GAT、GATv2、GIN、PNA、DiffPool**。
- **理论与优化**：**Weisfeiler-Lehman(1-WL)**表达上界、高阶图网络、位置编码，以及过平滑、过压缩和异配性问题。
- **扩展方向**：异构图、时空图、连续时间动态图、几何等变网络与图**Transformer**，包括**R-GCN、HGT、TGN、EGNN、Graphormer、GraphGPS、Exphormer**。
- **训练与评测**：**FastGCN、Cluster-GCN、GraphSAINT、SIGN**等扩展方法，图对比学习与掩码建模，以及**OGB**数据划分和信息泄漏问题。

### ⚪ [**<font color=Blue>胶囊网络 (Capsule Network)</font>**](https://0809zheng.github.io/2020/04/20/Capsule-Network.html)

**胶囊网络(Capsule Network，CapsNet)**用向量或矩阵同时表示实体的存在性与姿态：激活强度回答“是否存在”，实例化参数描述位置、尺度和朝向；子胶囊通过姿态变换向候选父胶囊投票，再由路由机制按预测一致性组合部件与整体。

本文主要内容包括：
- **表示基础**：不变性与等变性、部件—整体关系、**Transforming Auto-Encoder**、向量胶囊、**Squash**非线性与投票张量。
- **经典模型**：**Dynamic Routing、CapsNet、Matrix Capsules、EM Routing**，以及边际损失、重构正则和**Spread Loss**。
- **后续路线**：面向分割的**SegCaps**、具有严格变换保证的**Group Equivariant Capsule Network**、非迭代的**Self-Routing**与**Efficient-CapsNet**、无监督对象分解的**Stacked Capsule Autoencoder**。
- **实证边界**：区分“姿态等变、对象分组、样本效率和鲁棒性”的设计目标与实验事实，并分析投票张量、迭代路由、训练饥饿和深层扩展问题。

### ⚪ [**<font color=Blue>记忆增强神经网络 (Memory Augmented Neural Network)</font>**](https://0809zheng.github.io/2020/04/23/memory-network.html)

**记忆增强神经网络(Memory Augmented Neural Network，MANN)**在控制器之外增加显式、可寻址的外部记忆，把“计算状态”与“信息存储”分离。不同架构的关键差异在于记忆中存什么、如何寻址，以及推理期间是否允许写回。

本文主要内容包括：
- **统一接口**：控制器、记忆矩阵、内容寻址、位置寻址、加权读取与擦除—增加写入。
- **只读记忆**：**Memory Networks、bAbI、MemN2N、KV-MemNN**与**DMN**，以及多跳问答和键值分离。
- **可读写记忆**：**NTM**的内容—位置联合寻址，以及**DNC**的动态分配、使用率和时间链接。
- **任务化结构**：少样本学习中的**MANN/LRUA**，以及可微栈、队列和双端队列。
- **训练与边界**：控制器捷径、地址弥散、槽位干扰、长度外推、记忆消融，以及外部记忆与注意力的关系。

### ⚪ [<font color=Blue>状态空间模型 (State Space Model)</font>](https://0809zheng.github.io/2024/07/01/ssm.html)

**状态空间**包含完整描述系统的最小变量数，这些变量称为**状态向量**。**状态空间模型**是用于描述这些状态向量的模型，并根据额外的输入预测它们的下一个状态。

**状态方程**描述了系统内部状态$h(t)$随时间和输入的演化：

$$
h^\prime(t) = Ah(t) + Bx(t)
$$

**观测方程**描述了系统的输出$y(t)$如何依赖于系统状态和控制输入：

$$
y(t) = Ch(t)
$$

现代研究揭示状态空间模型、线性注意力与门控线性**RNN**在"带状态的线性递归"这一模板下是统一的（均可写成$h_t=A_t h_{t-1}+B_t x_t$的循环形式，并支持并行/递归/分块递归三种等价计算）。深度学习中的相关模型包括：
- 处理序列的**SSM**：如**HiPPO**, **LSSL**, **S4**, **S5**, **DSS**, **S4D**, **H3**, **Hyena**, **Mamba**, **Mamba-2**, **MoE-Mamba**, **RTF**。
- 线性注意力与门控线性**RNN**：**Linear Transformer**, **Fast Weight Programmer**, **LRU**, **RetNet**, **HGRN**, **GLA**, **RWKV**, **xLSTM**, **Griffin**, **DeltaNet**, **Gated DeltaNet**。
- 混合架构（**SSM/线性注意力 + 局部注意力**）：**Jamba**, **Samba**, **Zamba**, **Falcon Mamba**。
- 处理图像的**SSM**：**Vim**, **VMamba**, **MambaOut**, **MambaR**。

# 3. 深度学习的应用

## (1) High-Level视觉

### ⚪ [<font color=Blue>图像识别 (Image Recognition)</font>](https://0809zheng.github.io/2020/05/06/image-classification.html)

**图像识别**是计算机视觉的基本任务，旨在对每张图像内出现的物体进行类别区分。基于深度学习的图像识别方法不需要手工提取特征，而是使用卷积神经网络自动提取特征并进行分类。应用于图像识别任务的卷积神经网络的结构发展包括：
1. 早期探索：奠定“卷积层-下采样层-全连接层”的拓扑结构。如**LeNet5**, **AlexNet**, **ZFNet**, **NIN**, **SPP-net**, **VGGNet**
2. 深度化：增加堆叠卷积层的数量。如**Highway Network**, **ResNet**, **Stochastic Depth**, **DenseNet**, **Pyramidal ResNet**, **DPN**, **Res2Net**, **SpineNet**, **SpinalNet**
3. 模块化：设计用于堆叠的网络模块。如**Inception v1-4**, **WideResNet**, **Xception**, **ResNeXt**, **SENet**, **ResNeSt**
4. 缩放与架构搜索：用**NAS**或复合缩放自动设计网络。如**NASNet**, **MnasNet**, **DARTS**, **EfficientNet**/**V2**, **RegNet**
5. 结构重参数化：训练用多分支、推理时合并为单路。如**ACNet**, **RepVGG**, **DBB**
6. 大核卷积与现代卷积网络：用大卷积核重新逼近自注意力的全局感受野。如**ConvNeXt v1-2**, **RepLKNet**, **SLaK**, **InternImage**, **MogaNet**, **InceptionNeXt**, **UniRepLKNet**, **OverLoCK**
7. 轻量化：设计轻量级卷积层，可参考[<font color=Blue>轻量级卷积神经网络</font>](https://0809zheng.github.io/2021/09/10/lightweight.html)。
8. 训练配方与稳定性：架构之外的训练技巧同样关键。如**Bag of Tricks**, **ResNet strikes back**, **NFNet**, **ResNet-RS**
9. 少标注学习范式：如**Noisy Student**, **Meta Pseudo Labels**, **SCAN**, **BiT**（**ImageNet-21k**预训练）

图像识别的常用训练基准是**ImageNet-1k/21k**；评估分布外泛化与鲁棒性时还会用到**ImageNet-V2**, **-A**, **-R**, **-Sketch**, **-C**等测试集。

除卷积网络外，基于自注意力的[<font color=Blue>视觉Transformer</font>](https://0809zheng.github.io/2023/01/01/vit.html)已成为图像识别的另一条主线。

### ⚪ [<font color=blue>目标检测 (Object Detection)</font>](https://0809zheng.github.io/2020/05/08/object-detection.html)

**目标检测**任务是指在图像中检测出可能存在的目标；包括**定位**和**分类**两个子任务：其中定位是指确定目标在图像中的具体位置，分类是确定目标的具体类别。

传统的目标检测算法首先在图像中生成候选区域，然后对每个候选区域提取特征向量，最后对每个候选区域提取的特征进行分类。常用的候选区域生成方法包括滑动窗口、**Felzenszwalb**算法、选择搜索算法。常用的特征描述子包括图像梯度向量、方向梯度直方图**HOG**、尺度不变特征变换**SIFT**、可变形部位模型**DPM**。

基于深度学习的目标检测模型包括：
- **两阶段**的目标检测模型：首先在图像中生成可能存在目标的候选区域，然后对这些候选区域进行预测。如**R-CNN**, **Fast RCNN**, **Faster RCNN**, **SPP-Net**, **FPN**, **Libra RCNN**, **Cascade RCNN**, **Sparse RCNN**
- **单阶段**的目标检测模型：把图像中的每一个位置看作潜在的候选区域，直接进行预测。如**OverFeat**, **YOLOv1-3**, **SSD**, **RetinaNet**, **Guided Anchoring**, **ASFF**, **EfficientDet**, **YOLT**, **Poly-YOLO**, **YOLOv4**, **YOLOv5**, **RTMDet**
- **Anchor-Free**的目标检测模型：把目标检测任务视作关键点检测等其它形式的任务，直接对目标的位置进行预测。(**anchor-point**方法) **FCOS**, **YOLOX**, **YOLOv6**, **YOLOv7**, **YOLOv8**, **YOLOv9**, **YOLOv10**; (**key-point**方法) **CornerNet**, **CenterNet**, **RepPoints**
- 基于**Transformer**的目标检测模型：**DETR**, **Deformable DETR**

目标检测的常用评估指标包括准确率、召回率、**F-score**、**P-R**曲线、平均准确率**AP**、类别平均准确率**mAP**。

**非极大值抑制**算法是目标检测等任务中常用的后处理方法，能够过滤掉多余的检测边界框。提高**NMS**算法精度的方法包括**Soft-NMS**, **IoU-Guided NMS**, **Weighted NMS**, **Softer-NMS**, **Adaptive NMS**, **DIoU-NMS**。提高**NMS**算法效率的方法包括**CUDA NMS**, **Fast NMS**, **Cluster NMS**, **Matrix NMS**。

目标检测中的损失函数包括边界框的**分类**损失和**回归**损失。
- 分类损失用于区分边界框的类别，即边界框内目标的类别，对于两阶段的检测方法还包含边界框的正负类别；常用的分类损失函数包括**Cross-Entropy loss**, **Focal loss**, **Generalized Focal Loss**, **Varifocal Loss**, **GHM**, **Poly loss**。
- 回归损失衡量预测边界框坐标$x_{pred}$和**GT**边界框坐标$x_{gt}$之间的差异，常用的回归损失函数包括**L1 / L2 loss**, **Smooth L1 loss**, **Dynamic SmoothL1 Loss**, **Balanced L1 loss**, **IoU loss**, **GIoU loss**, **DIoU loss**, **CIoU loss**, **EIoU loss**, **SIoU loss**, **MPDIoU loss**。

**标签分配**策略是指在训练目标检测器时，为特征图不同位置的预测样本分配合适的标签（即区分**anchor**是正样本还是负样本），用于计算损失。标签分配根据非负即正划分为**硬标签分配(hard LA)**和**软标签分配(soft LA)**。
- 硬标签分配策略是指根据阈值把样本划分为正样本或者负样本。依据在训练阶段是否动态调整阈值，硬标签分配策略又可以细分为静态和动态两种：
1. **静态分配**策略主要依据于模型的先验知识（例如距离阈值和**iou**阈值等）来选取不同的正负样本；
2. **动态分配**策略依据在训练阶段采用不同的统计量来动态地设置阈值，并划分正负样本；如**DLA**, **MCA**, **HAMBox**, **ATSS**, **SimOTA**, **DSLA**。
- 软标签分配策略则会根据预测结果与**GT**计算正负权重，在候选正样本(中心点落在**GT**框内)的基础上依据正负样本权重分配正负样本，且在训练的过程中动态调整分配权重。常见的软标签分配策略包括**Noisy Anchor**, **AutoAssign**, **SAPD**, **TOOD**。

### ⚪ [<font color=blue>开放集合目标检测 (Open-Set Object Detection)</font>](https://0809zheng.github.io/2023/11/01/opensetdet.html)

**开集目标检测**是指在可见类的数据上进行训练，然后完成对不可见类数据的定位与识别。一些常见的开集目标检测方法包括：
- 基于无监督学习的开集检测器：通过聚类、弱监督等手段实现开集检测，如**OSODD**, **Detic**, **VLDet**
- 基于多模态学习的开集检测器：
1. 基于**Referring**的开集检测器：借助多模态视觉-语言模型实现检测，如**ViLD**, **RegionCLIP**, **VL-PLM**, **Grad-OVD**
2. 基于**Grounding**的开集检测器：把开集检测任务建模为边界框提取+短语定位任务，如**OVR-CNN**, **MDETR**, **GLIP**, **DetCLIP**, **DetCLIPv2**, **Grounding DINO**

### ⚪ [<font color=Blue>图像分割 (Image Segmentation)</font>](https://0809zheng.github.io/2020/05/07/semantic-segmentation.html)

**图像分割**是对图像中的每个像素进行分类，可以细分为：
- **语义分割**：注重类别之间的区分，而不区分同一类别的不同个体；
- **实例分割**：注重类别以及同一类别的不同个体之间的区分；
- **全景分割**：对于可数的对象实例(如行人、汽车)做实例分割，对于不可数的语义区域(如天空、地面)做语义分割。

图像分割模型通常采用**编码器-解码器**结构。编码器从预处理的图像数据中提取特征，解码器把特征解码为分割掩码。图像分割模型的发展趋势可以大致总结为：
- 全卷积网络：**FCN**, **SegNet**, **RefineNet**, **U-Net**, **V-Net**, **M-Net**, **W-Net**, **Y-Net**, **UNet++**, **Attention U-Net**, **GRUU-Net**, **BiSeNet V1,2**, **DFANet**, **SegNeXt**
- 上下文模块：**DeepLab v1,2,3,3+**, **PSPNet**, **FPN**, **UPerNet**, **EncNet**, **PSANet**, **APCNet**, **DMNet**, **OCRNet**, **PointRend**, **K-Net**
- 基于**Transformer**：**SETR**, **TransUNet**, **SegFormer**, **Segmenter**, **MaskFormer**, **SAM**
- 通用技巧：**Deep Supervision**, **Self-Correction**

图像分割中常用的评估指标包括：**PA**, **CPA**, **MPA**, **IoU**, **MIoU**, **FWIoU**, **Dice Coefficient**。

图像分割的损失函数用于衡量预测分割结果和真实标签之间的差异。根据损失函数的推导方式不同，图像分割任务中常用的损失函数可以划分为：
- 基于分布的损失：**Cross-Entropy Loss**, **Weighted Cross-Entropy Loss**, **TopK Loss**, **Focal Loss**, **Distance Map Penalized CE Loss**
- 基于区域的损失：**Sensitivity-Specifity Loss**, **IoU Loss**, **Lovász Loss**, **Dice Loss**, **Tversky Loss**, **Focal Tversky Loss**, **Asymmetric Similarity Loss**, **Generalized Dice Loss**, **Penalty Loss**
- 基于边界的损失：**Boundary Loss**, **Hausdorff Distance Loss**

### ⚪ [<font color=blue>点云分类 (Point Cloud Classification)</font>](https://0809zheng.github.io/2023/04/01/pointcloud.html)

**点云分类**即点云形状分类，是一种重要的点云理解任务。该任务的方法通常首先学习每个点的嵌入，然后使用聚合方法从整个点云中提取全局形状嵌入，并通过分类器进行分类。根据神经网络输入的数据格式，三维点云分类方法可分为：
- 基于多视图(**Multi-view based**)的方法：将点云投影为多个二维图像，如**MVCNN**, **MHBN**。
- 基于体素(**Voxel-based**)的方法：将点云转换为三维体素表示，如**VoxNet**, **OctNet**。
- 基于点(**Point-based**)的方法：直接处理原始点云，如**PointNet**, **PointNet++**, **PointCNN**, **DGCNN**, **PCT**。

### ⚪ [<font color=blue>目标计数 (Object Counting)</font>](https://0809zheng.github.io/2023/05/01/counting.html)

**目标计数**任务旨在从图像或视频中统计特定目标实例的数量。本文主要讨论基于**回归**的计数方式，即直接学习图像到目标数量或目标密度图的映射关系，通用的目标技术方案包括：
1. **少样本计数 (Few-Shot Counting)**：提供目标样本**exemplar**在查询图像中进行匹配，如**GMN**, **FamNet**, **LaoNet**, **CFOCNet**, **SAFECount**, **BMNet+**, **Counting-DETR**, **CounTR**, **LOCA**, **SPDCN**, **VCN**, **SAM Counting**, **CACViT**, **DAVE**, **SSD**。
2. **无参考计数 (Reference-less Counting)**：自动挖掘和计数所有显著性目标，如**LC**, **RLC**, **MoVie**, **DSAA**, **CaOC**, **RepRPN-Counter**, **RCC**, **GCNet**, **ZSC**, **ABC123**, **OmniCount**。
3. **文本引导计数 (Text-Guided Counting)**：通过预训练视觉语言模型进行目标计数，如**CountCLIP**, **CLIP-Count**, **CounTX**, **VLCounter**, **CLIP Counting**, **ExpressCount**。

## (2) Low-Level视觉

### ⚪ [<font color=blue>图像超分辨率 (Super Resolution)</font>](https://0809zheng.github.io/2020/08/27/SR.html)

图像**超分辨率**旨在将低分辨率图像**LR**放大为对应的高分辨率图像**HR**，从而使图像更清晰。图像超分辨率的传统方法主要是基于插值的方法，如最邻近插值、双线性插值、双三次插值；而基于深度学习的图像超分辨率方法，可以根据**上采样的位置**不同进行分类：

- **预定义上采样(Predefined upsampling)**：首先对图像应用预定义的插值方法进行上采样，再通过卷积网络增加细节，如**SRCNN**, **VDSR**。
- **单次上采样(Single upsampling)**：先通过卷积网络提取丰富的特征，再通过预定义或可学习的单次上采样增加分辨率，如**FSRCNN**, **ESPCN**, **EDSR**, **RCAN**, **SAN**。
- **渐进上采样(Progressive upsampling)**：通过多次上采样逐渐增加分辨率，如**LapSRN**。
- **循环采样(Iterative up and downsampling)**：循环地进行上采样和下采样，增加丰富的特征信息，如**DBPN**, **DRN**。
- 其他结构：如**SRGAN**, **ESRGAN**引入生成对抗网络；**LIIF**学习二维图像的连续表达形式。

图像超分辨率的评估指标主要包括峰值信噪比**PSNR**和结构相似度**SSIM**。

### ⚪ [<font color=blue>全色锐化 (Panchromatic Sharpening)</font>](https://0809zheng.github.io/2024/10/08/pansharpen.html)

**全色锐化**是指将全色图像的高分辨率空间细节信息与多光谱图像的丰富光谱信息进行融合，得到高质量、理想的高空间分辨率多光谱图像。像素级全色图像锐化方法通常分为:
1. 成分替换法(**CS-based**)：使用全色图像对多光谱图像的成分进行替换，如**Brovey**变换, **PCA**变换, **IHS**变换, **GS**变换, **GSA**, **CNMF**, **GFPCA**。
2. 多分辨率分析法(**MRA-based**)：对全色图像和多光谱图像不同尺度的高、低频成份进行融合，如**SFIM**变换, **Wavelet**变换, **MTF-GLP**, **MTF-GLP-HPM**。
3. 模型优化法(**MO-based**)：建立并优化融合图像与全色图像和多光谱图像之间的能量函数，如**SIRF**, **PSFG**$S^2$**LR**, **LGC**, **PGCP-PS**, **BPSM**, **F-BMP**。
4. 深度学习方法(**DL-based**)：使用深度学习模型自动学习图像特征，从而实现图像分辨率的提升，如**PNN**, **PanNet**, **MSDCNN**, **GPPNN**, **SRPPNN**, **INNformer**, **PanFormer**, **SFIIN**, **MIDPS**, **PanFlowNet**, **Pan-Mamba**, **HFIN**。

## (3) GenAI (Generative AI)

生成式人工智能 (**GenAI**) 旨在从现有数据（如文本、图像、视频、音频和代码）中学习，然后生成具有相似特征的数据。

### ⚪ [<font color=blue>图像到图像翻译 (Image-to-Image Translation)</font>](https://0809zheng.github.io/2020/05/23/image_translation.html)

**图像到图像翻译**旨在学习一个映射使得图像可以从源图像域变换到目标图像域，同时保留图像内容。根据是否提供了一对一的学习样本对，将图像到图像翻译任务划分为**有配对数据(paired data)**和**无配对数据(unpaired data)**两种情况。
- 有配对数据(监督图像翻译)是指在训练数据集中具有一对一的数据对；即给定联合分布$p(X,Y)$，学习条件映射$f_{x \to y}=p(Y\|X)$和$f_{y \to x}=p(X\|Y)$。代表方法有**Pix2Pix**, **BicycleGAN**, **LPTN**。
- 无配对数据(无监督图像翻译)是指模型在多个独立的数据集之间训练，能够从多个数据集合中自动地发现集合之间的关联，从而学习出映射函数；即给定边缘分布$p(X)$和$p(Y)$，学习条件映射$f_{x \to y}=p(Y\|X)$和$f_{y \to x}=p(X\|Y)$。代表方法有**CoGAN**, **PixelDA**, **CycleGAN**, **DiscoGAN**, **DualGAN**, **UNIT**, **MUNIT**, **TUNIT**, **StarGAN**, **StarGAN v2**, **GANILLA**, **NICE-GAN**, **CUT**, **SimDCL**。


### ⚪ [<font color=blue>布局引导图像生成 (Layout-to-Image Generation)</font>](https://0809zheng.github.io/2024/03/01/grounded_generator.html)

**布局引导图像生成**是图像感知任务（如目标检测、图像分割）的逆过程，即根据给定的布局生成对应的图像。根据布局控制条件的输入形式，布局引导的图像生成模型包括：
- 文本级**L2I**模型：通过将空间布局转换成文本**token**实现布局控制，如**ReCo**, **LayoutDiffusion**, **GeoDiffusion**, **DetDiffusion**。
- 像素级**L2I**模型：通过提供像素级空间对齐条件实现布局控制，如**GLIGEN**, **LayoutDiffuse**, **ControlNet**, **InstanceDiffusion**。

## (4) Human-Centric感知


### ⚪ [<font color=blue>人体姿态估计 (Human Pose Estimation)</font>](https://0809zheng.github.io/2020/05/31/pose-estimation.html)

**人体姿态估计**是指从图像、视频等输入信号中估计人体的姿态信息。姿态通常以关键点组成的人体骨骼表示。

**2D**单人人体姿态估计通常是从已完成定位的人体图像中计算人体关节点的位置，并进一步生成**2D**人体骨架。这些方法可以进一步分为：
- 基于回归的方法：直接将输入图像映射为人体关节的**坐标**或人体模型的**参数**，如**DeepPose**, **TFPose**, **Poseur**, **PCT**。
- 基于检测的方法：将输入图像映射为**图像块(patch)**或人体关节位置的**热图(heatmap)**，从而将身体部位作为检测目标；如**CPM**, **Hourglass**, **Chained**, **MCA**, **FPM**, **HRNet**, **TokenPose**, **ViTPose**。

与单人姿态估计相比，多人姿态估计需要同时完成**检测**和**估计**任务。根据完成任务的顺序不同，多人姿态估计方法分为：
- 自上而下的方法：先做**检测**再做**估计**。即先通过目标检测的方法在输入图像中检测出不同的人体，再使用单人姿态估计方法对每个人进行姿态估计；如**RMPE**, **CPN**, **MSPN**, **RTMPose**。
- 自下而上的方法：先做**估计**再做**检测**。即先在图像中估计出所有人体关节点，再将属于不同人的关节点进行关联和组合；如**DeepCut**, **DeeperCut**, **Associative Embedding**, **OpenPose**。

**3D**人体姿态估计是从图片或视频中估计出关节点的三维坐标，与**2D**人体姿态估计相比，**3D**人体姿态估计需要估计**深度**信息。**3D**人体姿态估计方法可以分为：
- 直接回归的方法：直接把图像映射为**3D**关节点，如**DconvMP**, **VNect**, **ORPM**, **Volumetric Prediction**, **3DMPPE**。
- **2D→3D**的方法：从**2D**姿态估计结果中估计深度信息，如**2D+Matching**, **SimpleBasline-3D**。
- 基于模型的方法：引入人体模型，如**SMPLify**, **SMPLify-X**。

人体姿态估计中的技巧包括：
- 数据处理：一些常用的数据预处理和后处理方法，如**AID**, **PoseAug**, **UDP**, **SmoothNet**。
- 量化误差消除：在**Heatmap-based**方法中经过下采样的特征图会产生**量化误差**，消除方法包括**DARK**, **PIP-Net**, **SimCC**, **DSNT**, **IPR**, **Debiased IPR**, **Sampling-Argmax**, **CAL**, **SIKR**。
- 轻量化：探索轻量化姿态估计网络，如**Lite-HRNet**, **HR-NAS**, **Lite Pose**, **MoveNet**。
- 训练技巧：一些常用的姿态估计训练技巧，如**OKDHP**, **Bone Loss**, **HDM**, **PeCLR**, **RLE**, **SCAI**, **DWPose**, **POMNet**。

二维人体姿态估计中常用的评估指标包括**PCP**, **PCK**, **OKS**, **AP**, **mAP**。三维人体姿态估计中常用的评估指标包括**MPJPE**。

常用的二维人体姿态估计数据集包括**LSP**, **FLIC**, **MPII**, **MS COCO**, **AIC**。常用的三维人体姿态估计数据集包括**Human3.6M**, **MPI-INF-3DHP**, **CMU Panoptic**, **AMASS**。

### ⚪ [<font color=blue>人脸检测, 识别与验证 (Face Detection, Recognition, and Verification)</font>](https://0809zheng.github.io/2020/05/10/face-recognition.html)

**人脸检测**是指检测任意一幅给定的图像中是否含有人脸，如果是则返回人脸的位置、大小和姿态，是人脸验证与识别的关键步骤。常用的人脸检测方法包括**Eigenface**, **SSH**。

**人脸识别**是指判断给定的人脸图像属于用户数据库中的哪个人（或没有匹配），是一种多分类问题。常用的人脸识别方法包括**DeepFace**。

**人脸验证**是指判断给定的人脸图像和用户**ID**是否匹配，是一种二分类问题。常用的人脸识别方法包括**DeepID**, **DeepID2**。

### ⚪ [<font color=blue>行人检测与属性识别 (Pedestrian Detection and Attribute Recognition)</font>](https://0809zheng.github.io/2020/05/12/pedestrian-attribute-recognition.html)

**行人检测**是指找出图像或视频帧中所有的行人，包括位置和大小；常用的行人检测方法包括**DeepParts**。

**行人属性识别**是指从行人图像中挖掘具有高级语义的属性信息；常用的行人属性识别方法包括**DeepSAR**, **DeepMAR**, **HydraPlus-Net**。

### ⚪ [<font color=blue>时空动作检测 (Spatio-Temporal Action Detection)</font>](https://0809zheng.github.io/2021/07/15/stad.html)

**时空动作检测**旨在识别视频中目标动作出现的区间和对应的类别，并在空间范围内用一个包围框标记出人物的空间位置。按照处理方式不同，时空动作检测可方法以分为：

- **帧级的检测器(frame-level detector)**：每次检测时输入单帧图像，得到单帧图像上的检测结果；之后把检测结果沿时间维度进行连接，得到视频检测结果。如**T-CNN**。
- **管级的检测器(tubelet-level detector)**：每次检测时输入多帧连续视频帧，对每帧上预定义的检测框进行修正，并对不同输入的结果在时序上进行连接。如**ACT-detector**, **MOC-detector**。

### ⚪ [<font color=blue>射频人体感知(RF-based Human Perception)</font>](https://0809zheng.github.io/2024/06/01/rfbased.html)

**射频人体感知**又称为**可见光谱外的人体感知**，是指使用雷达系统进行人体感知应用。雷达系统向检测环境中发射电磁波信号，照射人体目标，并接收反射信号用于执行下游任务。与光学系统相比，雷达系统可以在低能见度等特殊环境中工作，并且可以提供更好的隐私保护性。在特定频段工作的雷达系统还可以穿透墙壁等非金属障碍物，从而实现隐蔽场景下的人体感知。

根据发射信号的工作频段不同，射频人体感知方法可以细分为基于毫米波雷达的方法、基于**WiFi**阵列的方法与基于穿墙雷达的方法。（部分工作简写为标题首字母）
- 基于毫米波雷达的方法：工作频段$30$-$300$**GHZ**，人体目标被视为散射体，可以捕获细粒度的人体细节，如**mm-Pose**, **HMRER-SRNN**, **ITL**, **1-D-DAN**。
- 基于**WiFi**阵列的方法：工作频段$2.4$-$5$**GHZ**，人体目标被视为反射体，可以通过深度学习技术学习人体统计信息，如**RF-Pose**, **RF-Pose3D**, **RF-Avatar**, **TWPIRT-MMEDP**, **Person-in-WiFi**, **RF-Action**, **WiPose**, **RF-ReID**, **TGUL**。
- 基于穿墙雷达的方法：工作频段$0$-$3$**GHZ**，超宽带穿墙雷达系统可用于非接触式穿墙人体感知，如**UDA-MDHMC**, **ADA-MDHAC**, **SCGRNN**, **TWHPR-UWB**, **UWB-Pose**, **TWHMR-TLEL**, **HPR-TWRI**, **UHCE-TWRI**, **TWHPR-CMLSSL**, **RPSNet**, **MIMDSN**, **RadarFormer**。

## (5) 自然语言处理


### ⚪ [<font color=Blue>预训练语言模型 (Pretrained Language Model)</font>](https://0809zheng.github.io/2020/04/27/elmo-bert-gpt.html)

预训练语言模型是一种从大量无标签的语料库中学习通用的自然语言特征表示的方法。使用预训练语言模型的步骤如下：1. 在大量无标签的语料库上进行特定任务的**预训练**；2. 在下游任务的语料库上进行**微调**（或直接通过**提示**驱动）。

语言的特征表示可以分为**上下文无关的嵌入**（如**Word2Vec**，同一个词在任何句子中都得到同一个向量，无法表达多义性）和**上下文相关的嵌入**（如**ELMo**，根据上下文为每个词元位置动态生成表示）两类。

根据采用的模型结构不同，预训练语言模型可以划分为以下几类：
- **编码端（Encoder-Only）架构**：优点是可以提取文本的上下文表征，适用于自然语言理解任务；缺点是不能自然地生成文本，且需要更多的特定训练目标。典型模型包括**ELMo**, **BERT**, **RoBERTa**, **SpanBERT**, **ERNIE**, **ALBERT**, **ELECTRA**, **REALM**, **DeBERTa**, **DeBERTaV3**, **XLNet**, **ModernBERT**。
- **解码端（Decoder-Only）架构**：优点是能够自然地生成文本，有简单的训练目标（最大似然），适用于自然语言生成任务；缺点是文本的上下文表征只能单向地依赖于左侧上下文。典型模型包括**GPT**系列, **Gopher**, **Jurassic-1**, **PaLM**, **OPT**, **BLOOM**, **LLaMA**系列, **Mistral**, **Mixtral**, **Qwen**系列, **DeepSeek**系列。这一路线是当前大模型的绝对主流。
- **编码-解码端（Encoder-Decoder）架构**：优点是可以使用双向上下文表征来处理输入文本，并且可以生成输出文本；缺点是需要更多的特定训练目标。典型模型包括**MASS**, **UniLM**, **BART**, **T5**, **T5.1.1**, **mT5**。此外**GLM**和**UL2**试图用统一的预训练目标把三种架构的能力合并到一个模型中。

预训练语言模型的预训练任务包括概率语言建模、掩码语言建模、序列到序列的掩码语言建模、增强掩码语言建模、排列语言建模、前缀语言建模等。

模型规模、数据规模与算力之间的关系由**规模化定律(scaling law)**刻画：**Kaplan**等人给出了损失关于$N,D,C$的幂律形式，**Chinchilla**修正了最优配比（参数量与训练词元数应当同比例增长），而部分能力只在超过一定规模后才出现，即**涌现能力(emergent ability)**。

现代大模型在预训练之后还有两个对齐阶段：**指令微调**（在大量任务的指令化数据上微调以获得零样本泛化，如**FLAN**）和**偏好对齐**（如基于人类反馈强化学习的**InstructGPT/RLHF**，以及把奖励模型解析地消去、直接在偏好数据上优化策略的**DPO**）。

预训练语言模型从文本数据中学习到的知识包括语言类知识（包括浅层语言知识和抽象语言知识）和世界知识（包括事实性知识和常识性知识）两大类。其中语言类知识主要分布在模型的浅层和中层，世界知识主要分布在模型的中层和深层。大型预训练模型在大规模数据上性能提升的主要驱动力是世界知识。

预训练语言模型的知识存储在Transformer的全连接层结构中。全连接层可以看作键-值记忆单元（$FFN(x)=f(x⋅K^\top )⋅V$），其中第一层的参数$K$作为输入序列的模式检测器，第二层的参数$V$存储了对应模式下输出词汇表上的概率分布。

修正预训练语言模型里存储的错误或者过时的知识有三种手段：① 通过数据归因定位并删除对应的数据源，并重新进行预训练；② 在知识修正的数据集上进行约束微调；③ 定位存储知识的模型参数并进行修正（通常是修改全连接层参数$V$，如**Knowledge Neuron**, **MEND**, **ROME**, **MEMIT**）。

### ⚪ [1](https://0809zheng.github.io/2020/08/27/SR.html)

- [词嵌入](https://0809zheng.github.io/2020/04/29/word-embedding.html)
- [文本摘要](https://0809zheng.github.io/2020/05/13/text-summary.html)
- [连接时序分类](https://0809zheng.github.io/2020/06/11/ctc.html)
- [音乐生成](https://0809zheng.github.io/2020/10/26/musicgen.html)

## (6) 多模态

- [图像描述](https://0809zheng.github.io/2020/05/14/image-caption.html)

### ⚪ [<font color=blue>文本检测与识别 (Text Detection and Recognition)</font>](https://0809zheng.github.io/2020/05/15/text-detection-recognition.html)

**文本检测**是指找出图像中的文字区域；文本识别是指对定位好的文字区域进行识别，将图像中的文字区域进转化为字符信息。常用的文本检测与识别方法包括**EAST**, **CRNN**, **Mask TextSpotter**。

### ⚪ [<font color=blue>视觉-语言预训练 (Vision-Language Pretraining)</font>](https://0809zheng.github.io/2024/01/01/vlp.html)

视觉-语言预训练旨在从大规模的图像-文本对中学习通用的跨模态表示，使得模型能够理解图像和文本之间的语义关联。预训练完成后，模型可以直接在下游的视觉-语言任务上进行微调。

根据视觉数据和语言数据的特征交互和对齐方式不同，视觉-语言预训练方法可以分为：
- 单塔结构模型；将文本和视觉特征连接到一起，然后使用**Transformer**编码器提取特征；如**VisualBERT**, **VL-BERT**, **UNITER**, **ImageBERT**, **Oscar**, **Pixel-BERT**, **VinVL**, **ViLT**, **Frozen**, **VLMo**, **VL-BEiT**, **BEiT-3**。
- 双塔结构模型；将文本和视觉特征分别编码，然后使用交叉注意力来实现不同模态之间的交互；如**ViLBERT**, **LXMERT**, **ALBEF**。
- 编解码器模型：通过完整的**Transformer**模型把视觉和语言任务统一为**Token**生成任务；如**VL-T5**, **SimVLM**, **GIT**, **CoCa**。
- 对比学习模型：通过使匹配的图像和文本在嵌入空间中彼此靠近来提取图像和文本的共享表示；如**ALIGN**, **CLIP**, **SLIP**, **GLIP**, **GLIPv2**, **BLIP**, **BLIP-2**, **MaskCLIP**, **Chinese CLIP**, **FLIP**, **A-CLIP**, **SigLIP**, **SigLIP 2**, **LaCLIP**。


# 5. 参考文献与扩展阅读

### ⚪ Life-Long Deep Learning

深度学习技术发展日新月异，快速并持续地获取新技术的发展十分关键。按照个人时间充裕程度，学习深度学习技术的渠道可以分为以下三种：
1. 时间充裕指数⭐⭐⭐：推荐每天刷[**arxiv**](https://arxiv.org/)获取最新研究论文，或者借助[**Cool Papers**](https://papers.cool/)辅助刷**arxiv**论文。
2. 时间充裕指数⭐⭐：推荐关注顶会的出分和放榜时间，通过[**OpenReview**](https://openreview.net/)等工具获取相关顶会的投递和接收论文，可以使用[**Paper Copilot**](https://papercopilot.com/)工具查看顶会论文的得分排名等统计信息。
3. 时间充裕指数⭐：推荐关注一些深度学习相关的自媒体，利用随便化时间从公众号、知乎、B站等渠道获取最新技术动态，不过能获取的大都是知名大佬的作品或网红论文，不建议作为主要学习方式。


### ⚪ 深度学习的相关课程
- [Deep Learning \| Coursera （Andrew Ng）](https://www.coursera.org/specializations/deep-learning)
- [吴恩达Tensorflow2.0实践系列课程](https://www.bilibili.com/video/BV1zE411T7nb?from=search&seid=890015452850895449)
- [CS231n：计算机视觉（李飞飞）](http://cs231n.stanford.edu/syllabus.html)
- [CS294-158：深度无监督学习](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
- [旷视x北大《深度学习实践》](https://www.bilibili.com/video/BV1E7411t7ay)
- **YouTuber**：[李宏毅](https://www.youtube.com/@HungyiLeeNTU)、[Yannic Kilcher](https://www.youtube.com/@YannicKilcher)

### ⚪ 深度学习的相关书籍
- [Deep Learning（花书）](https://book.douban.com/subject/27087503/)
- [《神经网络与深度学习》（邱锡鹏）](https://nndl.github.io/)
- [《动手学深度学习》（李沐等）](http://zh.d2l.ai/)

### ⚪ 深度学习的相关博客
- 企业博客：[OpenAI](https://openai.com/blog/)、[DeepMind](https://www.deepmind.com/blog)、[DeepLearning.AI](https://www.deepmind.com/blog)
- 个人博客：[Lil’Log](https://lilianweng.github.io/)、[科学空间](https://spaces.ac.cn/)
