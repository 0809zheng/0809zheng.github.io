---
layout: post
title: '深度学习(Deep Learning)概述'
date: 2020-01-02
author: 郑之杰
cover: ''
tags: 深度学习
---

> Outlines about Deep Learning.

- 提示：请点击任意[<font color=Blue>高亮位置</font>](https://0809zheng.github.io/2020/01/02/DL-outline.html)以发现更多细节！

**深度学习**(**Deep Learning**)是一种以深度神经网络为工具的机器学习方法。
本文首先介绍深度神经网络的**类型**，其次介绍深度学习的**基本组件**和**方法技巧**，最后介绍深度学习在计算机视觉和自然语言处理等领域的**应用**。

本文目录：
- **深度神经网络的类型**
1. **卷积神经网络**：卷积神经网络的基本概念、卷积神经网络中的池化层、卷积神经网络中的注意力机制、轻量级卷积神经网络
2. **循环神经网络**：循环神经网络的基本概念、序列到序列模型、序列到序列模型中的注意力机制
3. **自注意力网络**：自注意力机制、**Transformer**、**Transformer**中的位置编码、降低**Transformer**的计算复杂度、预训练语言模型、
4. **深度生成模型**：生成对抗网络、变分自编码器、流模型
5. **其他类型的网络**：递归神经网络、记忆增强神经网络、图神经网络
- **深度学习的基本组件和方法技巧**
1. **深度学习的基本组件**：激活函数、优化方法、正则化方法、归一化方法、参数初始化方法
2. **深度学习的方法**：半监督学习、自监督学习、度量学习、多任务学习、主动学习、迁移学习
3. **深度学习的技巧**：长尾分布、对抗训练、大模型的参数高效微调
- **深度学习的应用**
1. **计算机视觉**：图像识别、目标检测、开放集合目标检测、图像分割、图像超分辨率、图像到图像翻译、时空动作检测、人脸检测, 识别与验证、行人检测与属性识别、点云分类、目标计数
2. **自然语言处理**：
3. **AI for Science**

# 1. 深度神经网络的类型

## (1) 卷积神经网络

### ⚪ [<font color=Blue>卷积神经网络(Convolutional Neural Network)的基本概念</font>](https://0809zheng.github.io/2020/03/06/CNN.html)

**卷积神经网络**是由卷积层、激活函数和池化层堆叠构成的深度神经网络，可以从图像数据中自适应的提取特征。

![](https://pic.downk.cc/item/5ea54956c2a9a83be5d81c10.jpg)

卷积层是一种局部的互相关操作，使用卷积核在输入图像或特征上按照光栅扫描顺序滑动，并通过局部仿射变换构造输出特征；具有局部连接、参数共享和平移等变性等特点。

卷积神经网络中的卷积层包括标准卷积, 转置卷积, 扩张卷积(**Dilated Conv**, **IC-Conv**), 可分离卷积(空间可分离卷积, 深度可分离卷积, 平展卷积), 组卷积, 可变形卷积, 差分卷积(中心差分卷积, 交叉中心差分卷积, 像素差分卷积), 动态卷积(**CondConv**, **DynamicConv**, **DyNet**, **ODConv**, **DRConv**), **Involution**, 圆形卷积, 八度卷积, 稀疏卷积(空间稀疏卷积, 子流形稀疏卷积), **CoordConv**。

### ⚪ [<font color=Blue>卷积神经网络中的池化(Pooling)层</font>](https://0809zheng.github.io/2021/07/02/pool.html)


**池化层**可以对特征图进行降采样，从而减小网络的计算成本，降低过拟合的风险。卷积神经网络中的池化方法包括：
- 通用的池化方法：最大池化, 平均池化, 混合池化, 分数最大池化, 幂平均池化, 随机池化, 随机空间采样池化(**S3Pool**), 细节保留池化(**DPP**), 局部重要性池化(**LIP**), 软池化, 动态优化池化(**DynOPool**)
- 为下游任务设计的池化方法：全局平均池化(**GAP**), 协方差池化, 空间金字塔池化(**SPP**), 感兴趣区域池化(**RoI Pooling**), 双线性池化

### ⚪ [<font color=Blue>卷积神经网络中的注意力机制(Attention Mechanism)</font>](https://0809zheng.github.io/2020/11/18/AinCNN.html)

卷积神经网络中的**注意力机制**表现为在特征的某个维度上计算相应**统计量**，并根据所计算的统计量对该维度上的每一个元素赋予不同的权重，用以增强网络的特征表达能力。

卷积层的特征维度包括通道维度和空间维度，因此注意力机制可以应用在不同维度上：
- **通道注意力(Channel Attention)**：**SENet**, **CMPT-SE**, **GENet**, **GSoP**, **SRM**, **SKNet**, **DIA**, **ECA-Net**, **SPANet**, **FcaNet**, **EPSA**, **TSE**, **NAM**
- **空间注意力(Spatial Attention)**：**Residual Attention Network**, **SGE**, **ULSAM**
- 通道+空间：(**并联**)**scSE**, **BAM**, **SA-Net**, **Triplet Attention**; (**串联**)**CBAM**; (**融合**)**SCNet**, **Coordinate Attention**, **SimAM** 
- 其他注意力：**DCANet**, **WE**, **ATAC**, **AFF**, **AW-Convolution**, **BA^2M**, **Interflow**, **CSRA**


### ⚪ [卷积神经网络的可视化](https://0809zheng.github.io/2020/12/16/custom.html)

### ⚪ [<font color=Blue>轻量级(LightWeight)卷积神经网络</font>](https://0809zheng.github.io/2021/09/10/lightweight.html)

**轻量级**网络设计旨在设计计算复杂度更低的卷积网络结构。
- 从**结构**的角度考虑，卷积层提取的特征存在冗余，可以设计特殊的卷积操作，减少卷积操作的冗余，从而减少计算量。如**SqueezeNet**, **SqueezeNext**, **MobileNet V1,2,3**, **ShuffleNet V1,2**, **IGCNet V1,2**, **ChannelNet**, **EfficientNet V1,2**, **GhostNet**, **MicroNet**, **CompConv**。
- 从**计算**的角度，模型推理过程中存在大量乘法运算，而乘法操作(相比于加法)对于目前的硬件设备不友好，可以对乘法运算进行优化，也可以减少计算量。如**AdderNet**使用**L1**距离代替卷积乘法；使用**Mitchell**近似代替卷积乘法。





## (2) 循环神经网络


### ⚪ [<font color=Blue>循环神经网络(Recurrent Neural Network)的基本概念</font>](https://0809zheng.github.io/2020/03/07/RNN.html)

**循环神经网络(RNN)**可以处理输入长度不固定的文本等时间序列数据。**RNN**每一时刻的隐状态$h_t$不仅和当前时刻的输入$x_t$相关，也和上一时刻的隐状态$h_{t-1}$相关。**RNN**具有通用近似性、图灵完备性等特点。

![](https://pic.downk.cc/item/5e9fdc29c2a9a83be5533395.jpg)

**RNN**存在长程依赖问题：理论上可以建立长时间间隔的状态之间的依赖关系，但是由于梯度消失现象，实际上只能学习到短期的依赖关系。解决措施是引入门控机制，如**LSTM**, **GRU**, **QRNN**, **SRU**, **ON-LSTM**。

也可以通过增加循环层的深度增强**RNN**的特征提取能力，包括**Stacked RNN**, **Bidirectional RNN**。



### ⚪ [<font color=Blue>序列到序列模型 (Sequence to Sequence)</font>](https://0809zheng.github.io/2020/04/21/sequence-2-sequence.html)

**序列到序列(Seq2Seq)模型**是一种序列生成模型，能够根据一个随机长度的输入序列生成另一个随机长度的序列。**Seq2Seq**模型通常采用编码器-解码器结构：

![](https://pic.imgdb.cn/item/63b431e6be43e0d30e71b68d.jpg)

**Seq2Seq**模型在生成序列时可以通过贪婪搜索或束搜索实现。序列生成时存在曝光偏差问题，可以通过计划采样缓解。

典型的**Seq2Seq**模型包括条件**Seq2Seq**模型、指针网络。

### ⚪ [<font color=Blue>序列到序列模型中的注意力机制 (Attention Mechanism)</font>](https://0809zheng.github.io/2020/04/22/attention.html)

在**Seq2Seq**模型中，将输入序列通过编码器转换为一个上下文向量$c$，再喂入解码器。注意力机制是指在解码器的每一步中，通过输入序列的所有隐状态$h_{1:T}$构造注意力分布$(α_1,...,α_t,...,α_T)$，然后构造当前步的上下文向量$c= \sum_{t=1}^{T} {α_th_t}$。


## (3) 自注意力网络

### ⚪ [<font color=Blue>自注意力机制 (Self-Attention Mechanism)</font>](https://0809zheng.github.io/2020/04/24/self-attention.html)

**自注意力机制**用于捕捉单个序列$X$的内部关系。把输入序列$X$映射为查询矩阵$Q$, 键矩阵$K$和值矩阵$V$；根据查询矩阵$Q$和键矩阵$K$生成注意力图，并作用于值矩阵$V$获得自注意力的输出$H$。

![](https://pic.downk.cc/item/5ea28825c2a9a83be5477d93.jpg)

### ⚪ [<font color=Blue>Transformer</font>](https://0809zheng.github.io/2020/04/25/transformer.html)

**Transformer**是一个基于多头自注意力机制的深度网络模型，网络结构包括编码器和解码器。编码器生成基于注意力的特征表示，该表示具有从全局上下文中定位特定信息的能力；解码器从特征表示中进行检索。

![](https://pic.imgdb.cn/item/618b94ea2ab3f51d91f6d24e.jpg)

### ⚪ [<font color=Blue>Transformer中的位置编码 (Position Encoding)</font>](https://0809zheng.github.io/2021/07/12/efficienttransformer.html)

**Transformer**中的自注意力机制具有置换不变性(**permutation invariant**)，导致打乱输入序列的顺序对输出结果不会产生任何影响。通过**位置编码**把位置信息引入输入序列中，以打破模型的全对称性。
- **绝对位置编码**：只依赖于单一位置，将绝对位置信息加入到输入序列中，相当于引入索引的嵌入。比如**Sinusoidal**, **Learnable**, **FLOATER**, **Complex-order**, **RoPE**
- **相对位置编码**：不同位置的交互项，通过微调自注意力运算过程使其能分辨不同**token**之间的相对位置。比如**XLNet**, **T5**, **DeBERTa**, **URPE**


### ⚪ [<font color=Blue>降低Transformer的计算复杂度</font>](https://0809zheng.github.io/2021/07/12/efficienttransformer.html)

自注意力运算中**计算注意力矩阵**以及**加权求和计算输出**这两个步骤引入了$O(N^2)$的计算复杂度。因此可以改进这两个步骤，从而降低计算复杂度。
- 改进注意力矩阵的计算: 这类方法的改进思路是使得注意力矩阵的计算**稀疏化**，即对输入序列中的每一个位置只计算其与一部分位置(而不是全部位置)之间的相关性，表现为注意力矩阵是稀疏的。如**Sparse Transformer**, **Reformer**, **Longformer**, **Big Bird**。
- 改进输出的加权求和: 这类方法的改进思路是使得自注意力的计算**线性化**。如**Efficient Attention**, **Synthesizer**, **Linformer**, **Linear Transformer**, **Performer**, **Nyströmformer**, **External Attention**, **FLASH**。

### ⚪ [<font color=Blue>预训练语言模型 (Pretrained Language Model)</font>](https://0809zheng.github.io/2020/04/27/elmo-bert-gpt.html)

预训练语言模型是一种从大量无标签的语料库中学习通用的自然语言特征表示的方法。使用预训练语言模型的步骤如下：1. 在大量无标签的语料库上进行特定任务的**预训练**；2. 在下游任务的语料库上进行**微调**。

根据预训练的任务不同，预训练语言模型可以划分为以下几类：
- **词嵌入(word embedding)**：上下文无关的嵌入
- **概率语言建模 Language Modeling(LM)**：自回归或单向语言建模，即给定前面所有词预测下一个词。如**ELMo**, **GPT 1,2,3**。
- **掩码语言建模 Masked Language Modeling(MLM)**：从输入序列中遮盖一些**token**，然后训练模型通过其余的**token**预测**masked token**。如**BERT**, **ALBERT**, **ELECTRA**, **REALM**。
- **序列到序列的掩码语言建模 Seq2Seq Masked Language Modeling(Seq2Seq MLM)**：采用编码器-解码器结构，将**masked**序列输入编码器，解码器以自回归的方式顺序生成**masked token**。如**MASS**, **UniLM**, **T5**, **T5.1.1**, **mT5**。
- **增强掩码语言建模 Enhanced Masked Language Modeling(E-MLM)**：在掩码语言建模的过程中使用了一些增强方法。如**RoBERTa**, **DeBERTa**。
- **排列语言建模 Permuted Language Modeling(PLM)**：在输入序列的随机排列上进行语言建模。如**XLNet**。






## (4) 深度生成模型

**生成模型**(**generative model**)是指使用带参数$\theta$的概率分布$p_{\theta}(x)$拟合已有数据样本集$$\{x\}$$。由于概率分布$p_{\theta}(x)$的形式通常是未知的，可以将其假设为离散型或连续型分布；若进一步引入**隐变量(latent variable)** $z$，则可以间接地构造概率分布$p_{\theta}(x)$：

$$ p_{\theta}(x) = \int p_{\theta}(x,z) dz = \int p_{\theta}(x|z)p(z) dz  $$

参数$\theta$的求解可以通过极大似然估计。若记真实数据分布为$\tilde{p}(x)$，则优化目标为最大化对数似然$$\Bbb{E}_{x\text{~}\tilde{p}(x)}[\log p_{\theta}(x)]$$。由于该算式包含积分运算，直接求解比较困难；不同的生成模型通过不同的求解技巧避开这个困难。


### ⚪ 自回归模型 (Auto-Regressive)

从最严格的角度来看，图像应该是一个离散的分布，因为它是由有限个像素组成的，而每个像素的取值也是离散的、有限的，因此可以通过离散分布来描述。这个思路的成果就是PixelRNN一类的模型了，我们称之为“自回归流”，其特点就是无法并行，所以计算量特别大。所以，我们更希望用连续分布来描述图像。当然，图像只是一个场景，其他场景下我们也有很多连续型的数据，所以连续型的分布的研究是很有必要的。

的本质，就是希望用一个我们知道的概率模型来拟合所给的数据样本，也就是说，我们得写出一个带参数θ的分布qθ(x)。然而，我们的神经网络只是“万能函数拟合器”，却不是“万能分布拟合器”，也就是它原则上能拟合任意函数，但不能随意拟合一个概率分布，因为概率分布有“非负”和“归一化”的要求。这样一来，我们能直接写出来的只有离散型的分布，或者是连续型的高斯分布。

### ⚪ [<font color=Blue>生成对抗网络 (Generative Adversarial Network)</font>](https://0809zheng.github.io/2022/02/01/gan.html)

**生成对抗网络**通过交替优化的对抗训练绕开了似然的直接求解，使用生成器$G$构造真实分布的近似分布$$P_G(x)$$，并使用判别器衡量生成分布和真实分布之间的差异。

$$ \begin{aligned} \mathop{ \min}_{G} \mathop{\max}_{D}  \Bbb{E}_{x \text{~} P_{data}(x)}[\log D(x)] + \Bbb{E}_{z \text{~} P(z)}[\log(1-D(G(z)))] \end{aligned} $$

生成对抗网络的设计是集目标函数、网络结构、优化过程于一体的，**GAN**的各种变体也是基于对这些方面的改进：
- 改进目标函数：基于分布散度(如**f-GAN**, **BGAN**, **Softmax GAN**, **RGAN**, **LSGAN**, **WGAN-div**, **GAN-QP**, **Designing GAN**)、基于积分概率度量(如**WGAN**, **WGAN-GP**, **DRAGAN**, **SN-GAN**, **GN-GAN**, **GraN-GAN**, **c-transform**, **McGAN**, **MMD GAN**, **Fisher GAN**)
- 改进网络结构：调整神经网络(如**DCGAN**, **SAGAN**, **BigGAN**, **Self-Modulation**, **StyleGAN1,2,3**, **TransGAN**)、引入编码器(如**VAE-GAN**, **BiGAN**, **VQGAN**)、使用能量模型(如**EBGAN**, **LSGAN**, **BEGAN**, **MAGAN**, **MEG**)、由粗到细的生成(如**LAPGAN**, **StackGAN**, **PGGAN**, **SinGAN**)
- 改进优化过程：**TTUR**, **Dirac-GAN**, **VDB**, **Cascading Rejection**, **ADA**, **Hubness Prior**
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

### ⚪ 其他生成网络

[Generative Moment Matching Network](https://0809zheng.github.io/2022/03/27/gmmn.html)

  
## (4) 其他类型的神经网络

### ⚪ [<font color=Blue>递归神经网络 (Recursive Neural Network)</font>](https://0809zheng.github.io/2020/03/08/recursive-neural-network.html)

**递归神经网络**是循环神经网络在有向无环图上的扩展，主要用来建模自然语言句子的语义。给定一个句子的语法结构（一般为树状结构），可以使用递归神经网络来按照句法的组合关系来合成一个句子的语义。句子中每个短语成分又可以分成一些子成分，即每个短语的语义都可以由它的子成分语义组合而来，并进而合成整句的语义。

![](https://pic.downk.cc/item/5ea14499c2a9a83be5c09f98.jpg)

典型的递归神经网络包括递归神经张量网络、矩阵-向量递归网络、**Tree LSTM**。

### ⚪ [<font color=Blue>记忆增强神经网络 (Memory Augmented Neural Network)</font>](https://0809zheng.github.io/2020/04/23/memory-network.html)

**记忆增强神经网络**在神经网络中引入外部记忆单元来提高网络容量。记忆网络的模块包括：主网络$C$负责信息处理以及与外界的交互；外部记忆单元$M$用来存储信息；读取模块$R$根据主网络生成的查询向量从外部记忆单元读取信息；写入模块$W$根据主网络生成的查询向量和要写入的信息更新外部记忆单元。读取或写入操作通常使用注意力机制实现。

![](https://pic.downk.cc/item/5ea1641fc2a9a83be5e6cfc3.jpg)

典型的记忆增强神经网络包括端到端记忆网络、神经图灵机。

### ⚪ [<font color=Blue>图神经网络 (Graph Neural Network)</font>](https://0809zheng.github.io/2020/03/09/graph-neural-network.html)


**图神经网络**是用于处理图结构的神经网络，其核心思想是学习一个函数映射$f(\cdot)$，图中的节点$v_i$通过该映射可以聚合它自己的特征$x_i$与它的邻居特征$x_{j \in N(v_i)}$来生成节点$v_i$的新表示。

![](https://pic.downk.cc/item/5ea59bbdc2a9a83be5281d20.jpg)

**GNN**可以分为两大类，基于空间（**spatial-based**）和基于谱（**spectral-based**）。
- 基于空间的**GNN**直接根据邻域聚合特征信息，把图粗化为高级子结构，可用于提取图的各级表示和执行下游任务。如**NN4G**, **DCNN**, **DGC**, **MoNET**, **GraphSAGE**, **GAT**, **GIN**。
- 基于谱的**GNN**把图网络通过傅里叶变换转换到谱域，引入滤波器处理图谱后通过逆变换还原到顶点域。如**ChebNet**, **GCN**, **DropEdge**。

### ⚪ [胶囊网络](https://0809zheng.github.io/2020/04/20/Capsule-Network.html)



# 2. 深度学习的基本组件和方法技巧

## (1) 深度学习的基本组件
### ⚪ [<font color=Blue>激活函数 (Activation Function)</font>](https://0809zheng.github.io/2020/03/01/activation.html)
**激活函数**能为神经网络引入非线性，在设计激活函数时可以考虑的性质包括：连续可导、计算量小、没有饱和区、没有偏置偏移、具有生物可解释性、提取上下文信息、通用近似性。

常见的激活函数根据设计思路分类如下：
- **S**型激活函数：形如**S**型曲线的激活函数。包括**Step**，**Sigmoid**，**HardSigmoid**，**Tanh**，**HardTanh**
- **ReLU**族激活函数：形如**ReLU**的激活函数。包括**ReLU**，**Softplus**, **Squareplus**，**ReLU6**，**LeakyReLU**，**PReLU**，**RReLU**，**ELU**，**GELU**，**CELU**，**SELU**
- 自动搜索激活函数：通过自动搜索解空间得到的激活函数。包括**Swish**，**HardSwish**，**Elish**，**HardElish**，**Mish**
- 基于梯度的激活函数：通过梯度下降为每个神经元学习独立函数。包括**APL**，**PAU**，**ACON**，**PWLU**，**OPAU**，**SAU**，**SMU**
- 基于上下文的激活函数：多输入单输出函数，输入上下文信息。包括**maxout**，**Dynamic ReLU**，**Dynamic Shift-Max**，**FReLU**

### ⚪ [<font color=Blue>优化方法 (Optimization)</font>](https://0809zheng.github.io/2020/03/02/optimization.html)

深度学习中的**优化**问题是指在已有的数据集上实现最小的训练误差$l(\theta)$，通常用基于梯度的数值方法求解。在实际应用梯度方法时，可以根据截止到当前步$t$的历史梯度信息$$\{g_{1},...,g_{t}\}$$计算修正的参数更新量$h_t$（比如累积动量、累积二阶矩校正学习率等）。指定每次计算梯度所使用数据批量 $\mathcal{B}$ 和学习率 $\gamma$，则第$t$次参数更新为：

$$ \begin{aligned} g_t&=\frac{1}{\|\mathcal{B}\|}\sum_{x \in \mathcal{B}}^{}\nabla_{\theta} l(θ_{t-1}) \\ h_t &= f(g_{1},...,g_{t}) \\ θ_t&=θ_{t-1}-\gamma h_t \end{aligned} $$

基于梯度的方法存在一些缺陷，不同的改进思路如下：
- 更新过程中容易陷入局部极小值或鞍点；常见解决措施是在梯度更新中引入**动量**(如**momentum**, **NAG**, **Funnelled SGDM**)。
- 参数的不同维度的梯度大小不同，导致参数更新时在梯度大的方向震荡，在梯度小的方向收敛较慢；常见解决措施是为每个特征设置**自适应**学习率(如**RProp**, **AdaGrad**, **RMSprop**, **AdaDelta**)。
- 可以结合基于动量的方法和基于自适应学习率的方法，如**Adam**, **AdamW**, **Adamax**, **Nadam**, **AMSGRad**, **Radam**, **AdaX**, **Amos**, **Lion**。这类方法需要同时存储与模型参数具有相同尺寸的动量和方差，通常会占用较多内存，一些减少内存占用的优化算法包括**Adafactor**, **SM3**。
- 在分布式训练大规模神经网络时，整体批量通常较大，权重更新的次数减少，常见解决措施是通过**层级自适应**实现每一层的梯度归一化(如**LARS**, **LAMB**, **NovoGrad**)。
- 其他优化方法：随机权重平均、零阶优化、使用前向梯度代替反向传播梯度、**Lookahead**、**Data Echoing**。

### ⚪ [<font color=Blue>正则化方法 (Regularization)</font>](https://0809zheng.github.io/2020/03/03/regularization.html)

**正则化**指的是通过**引入噪声**或限制模型的**复杂度**，降低模型对输入或者参数的敏感性，避免过拟合，提高模型的泛化能力。常用的正则化方法包括：
- 约束**目标函数**：在目标函数中增加模型参数的正则化项，包括**L2**正则化, **L1**正则化, **L0**正则化, 弹性网络正则化, 谱正则化, 自正交性正则化, **WEISSI**正则化, 梯度惩罚
- 约束**网络结构**：在网络结构中添加噪声，包括随机深度, **Dropout**及其系列方法,
- 约束**优化过程**：在优化过程中施加额外步骤，包括数据增强, 梯度裁剪, **Early Stop**, 标签平滑, 变分信息瓶颈, 虚拟对抗训练, **Flooding**


### ⚪ [<font color=Blue>归一化方法 (Normalization)</font>](https://0809zheng.github.io/2020/03/04/normalization.html)


输入数据的特征通常具有不同的量纲和取值范围，使得不同特征的**尺度**差异很大。**归一化**泛指把数据特征的不同维度转换到相同尺度的方法。深度学习中常用的归一化方法包括：
1. 基础归一化方法：最小-最大值归一化、标准化、白化、逐层归一化
2. 深度学习中的特征归一化：局部响应归一化**LRN**、批归一化**BN**、层归一化**LN**、实例归一化**IN**、组归一化**GN**、切换归一化**SN**
3. 改进特征归一化：（改进**BN**）**Batch Renormalization**, **AdaBN**, **L1-Norm BN**, **GBN**, **SPADE**；（改进**LN**）**RMS Norm**；（改进**IN**）**FRN**, **AdaIN**
4. 深度学习中的参数归一化：权重归一化**WN**、余弦归一化**CN**、谱归一化**SN**

### ⚪ [<font color=Blue>参数初始化方法 (Parameter Initialization)</font>](https://0809zheng.github.io/2020/03/05/initialization.html)

对神经网络进行训练时，需要对神经网络的参数进行初始化。糟糕的初始化不仅会使模型效果变差，还有可能使得模型根本训练不动或者不收敛。

常见的初始化方法包括零初始化、随机初始化、稀疏初始化、**Xavier**初始化、**Kaiming**初始化、正交初始化、恒等初始化、**ZerO**初始化、模仿初始化。

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

## (3) 深度学习的技巧


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
- 重参数化(**reparameterization**)：将微调过程重参数化为低维子空间的优化，如**Diff Pruning**, **LoRA**, **AdaLoRA**, **QLoRA**, **GLoRA**
- 混合方法：如**MAM Adapter**, **UniPELT**

- [深度学习的可解释性](https://0809zheng.github.io/2020/04/28/explainable-DL.html)



## () 网络压缩
网络压缩旨在平衡网络的准确性和运算效率。
压缩预训练的网络 设计新的网络结构
- [网络压缩](https://0809zheng.github.io/2020/05/01/network-compression.html)：网络剪枝、知识蒸馏、结构设计、模型量化

# 3. 深度学习的应用

## (1) 计算机视觉

### ⚪ [<font color=Blue>图像识别 (Image Recognition)</font>](https://0809zheng.github.io/2020/05/06/image-classification.html)

**图像识别**是计算机视觉的基本任务，旨在对每张图像内出现的物体进行类别区分。基于深度学习的图像识别方法不需要手工提取特征，而是使用卷积神经网络自动提取特征并进行分类。应用于图像识别任务的卷积神经网络的结构发展包括：
1. 早期探索：奠定“卷积层-下采样层-全连接层”的拓扑结构。如**LeNet5**, **AlexNet**, **ZFNet**, **NIN**, **SPP-net**, **VGGNet**
2. 深度化：增加堆叠卷积层的数量。如**Highway Network**, **ResNet**, **Stochastic Depth**, **DenseNet**, **Pyramidal ResNet**
3. 模块化：设计用于堆叠的网络模块。如**Inception v1-4**, **WideResNet**, **Xception**, **ResNeXt**, **NASNet**, **ResNeSt**, **ConvNeXt v1-2**
4. 轻量化：设计轻量级卷积层，可参考[<font color=Blue>轻量级卷积神经网络</font>](https://0809zheng.github.io/2021/09/10/lightweight.html)。
5. 其他结构：**Noisy Student**, **SCAN**, **NFNet**, **ResNet-RS**

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

目标检测中的损失函数包括边界框的**分类**损失和**回归**损失。其中分类损失用于区分边界框的类别，即边界框内目标的类别，对于两阶段的检测方法还包含边界框的正负类别；常用的分类损失函数包括**Cross-Entropy loss**, **Focal loss**, **Generalized Focal Loss**, **Varifocal Loss**, **GHM**, **Poly loss**。

而回归损失衡量预测边界框坐标$x_{pred}$和**GT**边界框坐标$x_{gt}$之间的差异，常用的回归损失函数包括**L1 / L2 loss**, **Smooth L1 loss**, **Dynamic SmoothL1 Loss**, **Balanced L1 loss**, **IoU loss**, **GIoU loss**, **DIoU loss**, **CIoU loss**, **EIoU loss**, **SIoU loss**, **MPDIoU loss**。

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


### ⚪ [人体姿态估计](https://0809zheng.github.io/2020/05/08/pose-estimation.html)

- [三维人体模型](https://0809zheng.github.io/2021/01/07/3dhuman.html)
- [人体姿态估计的评估指标](https://0809zheng.github.io/2020/11/26/eval-pose-estimate.html)

### ⚪ [<font color=blue>图像超分辨率 (Super Resolution)</font>](https://0809zheng.github.io/2020/08/27/SR.html)

图像**超分辨率**旨在将低分辨率图像**LR**放大为对应的高分辨率图像**HR**，从而使图像更清晰。图像超分辨率的传统方法主要是基于插值的方法，如最邻近插值、双线性插值、双三次插值；而基于深度学习的图像超分辨率方法，可以根据**上采样的位置**不同进行分类：

- **预定义上采样(Predefined upsampling)**：首先对图像应用预定义的插值方法进行上采样，再通过卷积网络增加细节，如**SRCNN**, **VDSR**。
- **单次上采样(Single upsampling)**：先通过卷积网络提取丰富的特征，再通过预定义或可学习的单次上采样增加分辨率，如**FSRCNN**, **ESPCN**, **EDSR**, **RCAN**, **SAN**。
- **渐进上采样(Progressive upsampling)**：通过多次上采样逐渐增加分辨率，如**LapSRN**。
- **循环采样(Iterative up and downsampling)**：循环地进行上采样和下采样，增加丰富的特征信息，如**DBPN**, **DRN**。
- 其他结构：如**SRGAN**, **ESRGAN**引入生成对抗网络；**LIIF**学习二维图像的连续表达形式。

图像超分辨率的评估指标主要包括峰值信噪比**PSNR**和结构相似度**SSIM**。

### ⚪ [<font color=blue>图像到图像翻译 (Image-to-Image Translation)</font>](https://0809zheng.github.io/2020/05/23/image_translation.html)

**图像到图像翻译**旨在学习一个映射使得图像可以从源图像域变换到目标图像域，同时保留图像内容。根据是否提供了一对一的学习样本对，将图像到图像翻译任务划分为**有配对数据(paired data)**和**无配对数据(unpaired data)**两种情况。
- 有配对数据(监督图像翻译)是指在训练数据集中具有一对一的数据对；即给定联合分布$p(X,Y)$，学习条件映射$f_{x \to y}=p(Y\|X)$和$f_{y \to x}=p(X\|Y)$。代表方法有**Pix2Pix**, **BicycleGAN**, **LPTN**。
- 无配对数据(无监督图像翻译)是指模型在多个独立的数据集之间训练，能够从多个数据集合中自动地发现集合之间的关联，从而学习出映射函数；即给定边缘分布$p(X)$和$p(Y)$，学习条件映射$f_{x \to y}=p(Y\|X)$和$f_{y \to x}=p(X\|Y)$。代表方法有**CoGAN**, **PixelDA**, **CycleGAN**, **DiscoGAN**, **DualGAN**, **UNIT**, **MUNIT**, **TUNIT**, **StarGAN**, **StarGAN v2**, **GANILLA**, **NICE-GAN**, **CUT**, **SimDCL**。


### ⚪ [<font color=blue>时空动作检测 (Spatio-Temporal Action Detection)</font>](https://0809zheng.github.io/2021/07/15/stad.html)

**时空动作检测**旨在识别视频中目标动作出现的区间和对应的类别，并在空间范围内用一个包围框标记出人物的空间位置。按照处理方式不同，时空动作检测可方法以分为：

- **帧级的检测器(frame-level detector)**：每次检测时输入单帧图像，得到单帧图像上的检测结果；之后把检测结果沿时间维度进行连接，得到视频检测结果。如**T-CNN**。
- **管级的检测器(tubelet-level detector)**：每次检测时输入多帧连续视频帧，对每帧上预定义的检测框进行修正，并对不同输入的结果在时序上进行连接。如**ACT-detector**, **MOC-detector**。


### ⚪ [<font color=blue>人脸检测, 识别与验证 (Face Detection, Recognition, and Verification)</font>](https://0809zheng.github.io/2020/05/10/face-recognition.html)

**人脸检测**是指检测任意一幅给定的图像中是否含有人脸，如果是则返回人脸的位置、大小和姿态，是人脸验证与识别的关键步骤。常用的人脸检测方法包括**Eigenface**, **SSH**。

**人脸识别**是指判断给定的人脸图像属于用户数据库中的哪个人（或没有匹配），是一种多分类问题。常用的人脸识别方法包括**DeepFace**。

**人脸验证**是指判断给定的人脸图像和用户**ID**是否匹配，是一种二分类问题。常用的人脸识别方法包括**DeepID**, **DeepID2**。

### ⚪ [<font color=blue>行人检测与属性识别 (Pedestrian Detection and Attribute Recognition)</font>](https://0809zheng.github.io/2020/05/12/pedestrian-attribute-recognition.html)

**行人检测**是指找出图像或视频帧中所有的行人，包括位置和大小；常用的行人检测方法包括**DeepParts**。

**行人属性识别**是指从行人图像中挖掘具有高级语义的属性信息；常用的行人属性识别方法包括**DeepSAR**, **DeepMAR**, **HydraPlus-Net**。

### ⚪ [<font color=blue>文本检测与识别 (Text Detection and Recognition)</font>](https://0809zheng.github.io/2020/05/15/text-detection-recognition.html)

**文本检测**是指找出图像中的文字区域；文本识别是指对定位好的文字区域进行识别，将图像中的文字区域进转化为字符信息。常用的文本检测与识别方法包括**EAST**, **CRNN**, **Mask TextSpotter**。

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

## (2) 自然语言处理

### ⚪ [1](https://0809zheng.github.io/2020/08/27/SR.html)

- [词嵌入](https://0809zheng.github.io/2020/04/29/word-embedding.html)
- [文本摘要](https://0809zheng.github.io/2020/05/13/text-summary.html)
- [图像描述](https://0809zheng.github.io/2020/05/14/image-caption.html)
- [连接时序分类](https://0809zheng.github.io/2020/06/11/ctc.html)
- [音乐生成](https://0809zheng.github.io/2020/10/26/musicgen.html)


## (3) AI for Science

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://0809zheng.github.io/2021/06/28/fno.html)：(arXiv2010)为偏微分方程设计的傅里叶神经算子。
- [Advancing mathematics by guiding human intuition with AI](https://0809zheng.github.io/2022/01/08/mathai.html)：(Nature 2021.12)用人工智能引导人类直觉推进数学发展。
- [Noether Networks: Meta-Learning Useful Conserved Quantities](https://0809zheng.github.io/2022/06/19/noether.html)：(arXiv2112)Noether网络：通过元学习学习有用的守恒量。
- [Competition-Level Code Generation with AlphaCode](https://0809zheng.github.io/2022/03/13/alphacode.html)：(arXiv2203)AlphaCode: 竞赛级别的代码生成。
- [Discovering faster matrix multiplication algorithms with reinforcement learning](https://0809zheng.github.io/2022/11/21/alphatensor.html)：(Nature 2022.10)AlphaTensor：通过强化学习发现更快的矩阵乘法算法。
- [Faster sorting algorithms discovered using deep reinforcement learning](https://0809zheng.github.io/2023/06/07/alphadev.html)：(Nature 2023.06)AlphaDev：通过深度强化学习发现更快的排序算法。



# 4. 参考文献与扩展阅读

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