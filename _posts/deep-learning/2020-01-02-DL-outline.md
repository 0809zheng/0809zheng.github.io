---
layout: post
title: '深度学习 概述'
date: 2020-01-02
author: 郑之杰
cover: ''
tags: 深度学习
---

> Outlines about Deep Learning.

- 提示：请点击任意[<font color=Blue>超链接</font>](https://0809zheng.github.io/2020/01/02/DL-outline.html)以发现更多细节！

**深度学习**(**Deep Learning**)是一种以深度神经网络为工具的机器学习方法。
本文首先介绍深度神经网络的**类型**，其次介绍深度学习的**基本组件**和**方法技巧**，最后介绍深度学习在计算机视觉和自然语言处理等领域的**应用**。

本文目录：
- 深度神经网络的类型
1. 卷积神经网络：轻量级卷积神经网络
2. 循环神经网络：
3. 自注意力网络：预训练语言模型、降低Transformer的计算复杂度
4. 生成模型：
5. 其他类型的网络：
- 深度学习的基本组件和方法技巧
1. 深度学习的基本组件：激活函数、优化方法
2. 深度学习的方法技巧：长尾分布
- 深度学习的应用
1. 计算机视觉
2. 自然语言处理
3. 

# 1. 深度神经网络的类型

## (1) 卷积神经网络



  

- [卷积神经网络](https://0809zheng.github.io/2020/03/06/CNN.html)
- [卷积神经网络中的池化方法](https://0809zheng.github.io/2021/07/02/pool.html)
- [卷积神经网络的结构发展](https://0809zheng.github.io/2020/06/03/CNN-architecture.html)
- [卷积神经网络中的注意力机制](https://0809zheng.github.io/2020/11/18/AinCNN.html)

### ⚪ [卷积神经网络的可视化](https://0809zheng.github.io/2020/12/16/custom.html)

### ⚪ [<font color=Blue>轻量级卷积神经网络</font>](https://0809zheng.github.io/2020/12/16/custom.html)

轻量级网络设计旨在设计计算复杂度更低的卷积网络结构。
- 从**结构**的角度考虑，卷积层提取的特征存在冗余，可以设计特殊的卷积操作，减少卷积操作的冗余，从而减少计算量。如**SqueezeNet**, **SqueezeNext**, **MobileNet V1,2,3**, **ShuffleNet V1,2**, **IGCNet V1,2**, **ChannelNet**, **EfficientNet V1,2**, **GhostNet**, **MicroNet**, **CompConv**。
- 从**计算**的角度，模型推理过程中存在大量乘法运算，而乘法操作(相比于加法)对于目前的硬件设备不友好，可以对乘法运算进行优化，也可以减少计算量。如**AdderNet**使用**L1**距离代替卷积乘法；使用**Mitchell**近似代替卷积乘法。





## (2) 循环神经网络
- [循环神经网络](https://0809zheng.github.io/2020/03/07/RNN.html)
- [递归神经网络](https://0809zheng.github.io/2020/03/08/recursive-neural-network.html)
- [记忆增强神经网络](https://0809zheng.github.io/2020/04/23/memory-network.html)

## (3) 自注意力网络


### ⚪ [<font color=Blue>预训练语言模型</font>](https://0809zheng.github.io/2020/04/27/elmo-bert-gpt.html)

预训练语言模型是一种从大量无标签的语料库中学习通用的自然语言特征表示的方法。使用预训练语言模型的步骤如下：1. 在大量无标签的语料库上进行特定任务的**预训练**；2. 在下游任务的语料库上进行**微调**。

根据预训练的任务不同，预训练语言模型可以划分为以下几类：
- **词嵌入(word embedding)**：上下文无关的嵌入
- **概率语言建模 Language Modeling(LM)**：自回归或单向语言建模，即给定前面所有词预测下一个词。如**ELMo**, **GPT 1,2,3**。
- **掩码语言建模 Masked Language Modeling(MLM)**：从输入序列中遮盖一些**token**，然后训练模型通过其余的**token**预测**masked token**。如**BERT**, **ALBERT**, **ELECTRA**, **REALM**。
- **序列到序列的掩码语言建模 Seq2Seq Masked Language Modeling(Seq2Seq MLM)**：采用编码器-解码器结构，将**masked**序列输入编码器，解码器以自回归的方式顺序生成**masked token**。如**MASS**, **UniLM**, **T5**, **T5.1.1**, **mT5**。
- **增强掩码语言建模 Enhanced Masked Language Modeling(E-MLM)**：在掩码语言建模的过程中使用了一些增强方法。如**RoBERTa**, **DeBERTa**。
- **排列语言建模 Permuted Language Modeling(PLM)**：在输入序列的随机排列上进行语言建模。如**XLNet**。

### ⚪ [<font color=Blue>降低Transformer的计算复杂度</font>](https://0809zheng.github.io/2021/07/12/efficienttransformer.html)

自注意力运算中**计算自注意力矩阵**以及**加权求和计算输出**这两个步骤引入了$O(N^2)$的计算复杂度。因此可以改进这两个步骤，从而降低计算复杂度。
- 改进注意力矩阵的计算: 这类方法的改进思路是使得注意力矩阵的计算**稀疏化**，即对输入序列中的每一个位置只计算其与一部分位置(而不是全部位置)之间的相关性，表现为注意力矩阵是稀疏的。如**Sparse Transformer**, **Reformer**, **Longformer**, **Big Bird** 。
- 改进输出的加权求和: 这类方法的改进思路是使得自注意力的计算**线性化**。如**Efficient Attention**, **Synthesizer**, **Linformer**, **Linear Transformer**, **Performer**, **Nyströmformer**, **External Attention**, **FLASH**。




- [词嵌入](https://0809zheng.github.io/2020/04/29/word-embedding.html)
- [注意力机制](https://0809zheng.github.io/2020/04/22/attention.html)
- [自注意力模型](https://0809zheng.github.io/2020/04/24/self-attention.html)
- [Transformer](https://0809zheng.github.io/2020/04/25/transformer.html)
- [Transformer中的位置编码](https://0809zheng.github.io/2021///.html)
- 语言模型：[Seq2Seq语言模型,指针网络](https://0809zheng.github.io/2020/04/21/sequence-2-sequence.html)


## (4) 生成模型

**生成模型**(**generative model**)是指使用带参数$\theta$的概率分布$p_{\theta}(x)$拟合已有数据样本集$$\{x\}$$。由于概率分布$p_{\theta}(x)$的形式通常是未知的，可以将其假设为离散型或连续型分布；若进一步引入**隐变量(latent variable)** $z$，则可以间接地构造概率分布$p_{\theta}(x)$：

$$ p_{\theta}(x) = \int p_{\theta}(x,z) dz = \int p_{\theta}(x|z)p(z) dz  $$

参数$\theta$的求解可以通过极大似然估计。若记真实数据分布为$\tilde{p}(x)$，则优化目标为最大化对数似然$$\Bbb{E}_{x\text{~}\tilde{p}(x)}[\log p_{\theta}(x)]$$。由于该算式包含积分运算，直接求解比较困难。不同的生成模型通过不同的求解技巧避开这个困难。


### ⚪ 自回归模型 (Auto-Regressive)

从最严格的角度来看，图像应该是一个离散的分布，因为它是由有限个像素组成的，而每个像素的取值也是离散的、有限的，因此可以通过离散分布来描述。这个思路的成果就是PixelRNN一类的模型了，我们称之为“自回归流”，其特点就是无法并行，所以计算量特别大。所以，我们更希望用连续分布来描述图像。当然，图像只是一个场景，其他场景下我们也有很多连续型的数据，所以连续型的分布的研究是很有必要的。

的本质，就是希望用一个我们知道的概率模型来拟合所给的数据样本，也就是说，我们得写出一个带参数θ的分布qθ(x)。然而，我们的神经网络只是“万能函数拟合器”，却不是“万能分布拟合器”，也就是它原则上能拟合任意函数，但不能随意拟合一个概率分布，因为概率分布有“非负”和“归一化”的要求。这样一来，我们能直接写出来的只有离散型的分布，或者是连续型的高斯分布。

### ⚪ [生成对抗网络](https://0809zheng.github.io/2020/05/18/generative-adversarial-network.html)

GAN则是通过一个交替训练的方法绕开了这个困难，确实保留了模型的精确性，所以它才能有如此好的生成效果。但不管怎么样，GAN也不能说处处让人满意了，所以探索别的解决方法是有意义的。

### ⚪ [变分自编码器 (Variational Autoencoder)](https://0809zheng.github.io/2022/04/01/vae.html)

**变分自编码器 (Variational Autoencoder, VAE)**的优化目标不是对数似然，而是对数似然的变分下界：

$$  \log p_{\theta}(x)  \geq \mathbb{E}_{z \text{~} q_{\phi}(z|x)} [\log p_{\theta}(x | z)] - KL[q_{\phi}(z|x)||p(z)]  $$

**VAE**的优化目标共涉及三个不同的概率分布：由概率编码器表示的后验分布$q_{\phi}(z\|x)$、隐变量的先验分布$p(z)$以及由概率解码器表示的生成分布$p_{\theta}(x\|z)$。对**VAE**的各种改进可以落脚于对这些概率分布的改进：

- 后验分布$q(z\|x)$：后验分布为模型引入了正则化；一种改进思路是通过调整后验分布的正则化项增强模型的解耦能力(如**β-VAE**, **Disentangled β-VAE**, **InfoVAE**, **DIP-VAE**, **FactorVAE**, **β-TCVAE**, **HFVAE**)。
- 先验分布$p(z)$：先验分布描绘了隐变量分布的隐空间；一种改进思路是通过引入标签实现半监督学习(如**CVAE**, **CMMA**)；一种改进思路是通过对隐变量离散化实现聚类或分层特征表示(如**Categorical VAE**, **Joint VAE**, **VQ-VAE**, **VQ-VAE-2**)；一种改进思路是更换隐变量的概率分布形式(如**Hyperspherical VAE**, **TD-VAE**, **f-VAE**, **NVAE**)。
- 生成分布$p(x\|z)$：生成分布代表模型的数据重构能力；一种改进思路是将均方误差损失替换为其他损失(如**EL-VAE**, **DFCVAE**, **LogCosh VAE**)。
- 改进整体损失函数：也有方法通过调整整体损失改进模型，如紧凑变分下界(如**IWAE**, **MIWAE**)或引入**Wasserstein**距离(如**WAE**, **SWAE**)。
- 改进模型结构：如**BN-VAE**通过引入**BatchNorm**缓解**KL**散度消失问题。

### ⚪ [流模型 (Flow-based Model)](https://0809zheng.github.io/2022/05/01/flow.html)

**流模型**(**flow-based model**)通过一系列可逆变换(双射函数$f$)建立较为简单的先验分布$p(z)$与较为复杂的实际数据分布$p(x)$之间的映射关系：

$$ \begin{aligned} x&=f_K \circ \cdots \circ f_1(z) \\ p(x) &= p(z)\cdot |\prod_{k=1}^{K} \det J_{f_k}(z_{k-1})|^{-1} \end{aligned} $$

由于流模型给出了概率分布$p(x)$的显式表达式，可直接最大化对数似然：

$$ \begin{aligned}  \log p(x)  = \log  p(z) - \sum_{k=1}^{K}\log  | \det J_{f_k}(z_{k-1})| \end{aligned}  $$

从优化目标中可以看出，流模型是由先验分布$p(z)$和双射函数$x=f(z)$唯一确定的。根据双射函数的不同设计思路，流模型分为以下两类：
- **标准化流**(**Normalizing Flow**)：通过数学定理与性质设计**Jacobian**行列式$\det J_{f}(z)$容易计算的双射函数$x=f(z)$。标准化流是最基础的流模型，事实上其他类别的流模型可以看作标准化流的延申。这类模型包括**Normalizing Flow**, **iResNet**等。
- **自回归流**(**Autoregressive Flow**)：把双射函数$x=f(z)$建模为自回归模型，即$x$的第$i$个维度$x_i$的生成只依赖于前面的维度$x_{1:i-1}$(自回归流)或$z_{1:i-1}$(逆自回归流)，此时**Jacobian**矩阵$J_{f}(z)$为三角矩阵，行列式容易计算。这类模型包括**IAF**, **MAF**, **NICE**, **Real NVP**, **Glow**, **Flow++**等。


### ⚪ [扩散模型]()




  
## (4) 其他类型的神经网络
- [图神经网络](https://0809zheng.github.io/2020/03/09/graph-neural-network.html)
- [胶囊网络](https://0809zheng.github.io/2020/04/20/Capsule-Network.html)



# 2. 深度学习的基本组件和方法技巧

## (1) 深度学习的基本组件
### ⚪ [<font color=Blue>激活函数 (Activation Function)</font>](https://0809zheng.github.io/2020/03/01/activation.html)
**激活函数**能为神经网络引入非线性，常见的激活函数根据设计思路分类如下：
- **S**型激活函数：形如**S**型曲线的激活函数。包括**Step**，**Sigmoid**，**HardSigmoid**，**Tanh**，**HardTanh**
- **ReLU**族激活函数：形如**ReLU**的激活函数。包括**ReLU**，**Softplus**，**ReLU6**，**LeakyReLU**，**PReLU**，**RReLU**，**ELU**，**GELU**，**CELU**，**SELU**
- 自动搜索激活函数：通过自动搜索解空间得到的激活函数。包括**Swish**，**HardSwish**，**Elish**，**HardElish**，**Mish**
- 基于梯度的激活函数：通过梯度下降为每个神经元学习独立函数。包括**APL**，**PAU**，**ACON**，**PWLU**，**OPAU**，**SAU**，**SMU**
- 基于上下文的激活函数：多输入单输出函数，输入上下文信息。包括**maxout**，**Dynamic ReLU**，**Dynamic Shift-Max**，**FReLU**

### ⚪ [<font color=Blue>优化方法 (Optimization)</font>](https://0809zheng.github.io/2020/03/02/optimization.html)

深度学习中的**优化**问题是指在已有的数据集上实现最小的训练误差$l(\theta)$，通常用基于梯度的数值方法求解。在实际应用梯度方法时，可以根据截止到当前步$t$的历史梯度信息$$\{g_{1},...,g_{t}\}$$计算修正的参数更新量$h_t$（比如累积动量、累积二阶矩校正学习率等）。指定每次计算梯度所使用数据批量 $\mathcal{B}$ 和学习率 $\gamma$，则第$t$次参数更新为：

$$ \begin{aligned} g_t&=\frac{1}{\|\mathcal{B}\|}\sum_{x \in \mathcal{B}}^{}\nabla_{\theta} l(θ_{t-1}) \\ h_t &= f(g_{1},...,g_{t}) \\ θ_t&=θ_{t-1}-\gamma h_t \end{aligned} $$

基于梯度的方法存在一些缺陷，不同的改进思路如下：
- 更新过程中容易陷入局部极小值或鞍点(这些点处的梯度也为$0$)；常见解决措施是在梯度更新中引入**动量**(如**momentum**, **NAG**)。
- 参数的不同维度的梯度大小不同，导致参数更新时在梯度大的方向震荡，在梯度小的方向收敛较慢；常见解决措施是为每个特征设置**自适应**学习率(如**AdaGrad**, **RMSprop**, **AdaDelta**)。这类算法的缺点是改变了梯度更新的方向，一定程度上造成精度损失。
- 在分布式训练大规模神经网络时，整体批量通常较大，训练的模型精度会剧烈降低。这是因为总训练轮数保持不变时，批量增大意味着权重更新的次数减少。常见解决措施是通过**层级自适应**实现每一层的梯度归一化(如**LARS**, **LAMB**, **NovoGrad**)，从而使得更新步长依赖于参数的数值大小而不是梯度的大小。
- 上述优化算法通常会占用较多内存，比如常用的**Adam**算法需要存储与模型参数具有相同尺寸的动量和方差。一些减少内存占用的优化算法包括**Adafactor**, **SM3**。
- 也有一些不直接依赖于一阶梯度的方法，如零阶优化方法或使用前向梯度代替反向传播梯度。

### ⚪ 


- [正则化方法](https://0809zheng.github.io/2020/03/03/regularization.html)
- [标准化方法](https://0809zheng.github.io/2020/03/04/normalization.html)
- [参数初始化方法](https://0809zheng.github.io/2020/03/05/initialization.html)



## (2) 深度学习的方法技巧

### ⚪ [<font color=Blue>长尾分布 (Long-Tailed)</font>](https://0809zheng.github.io/2020/03/02/optimization.html)

实际应用中的数据集大多服从长尾分布，即少数类别(**head class**)占据绝大多数样本，多数类别(**tail class**)仅有少量样本。解决长尾分布问题的方法包括：
- 重采样 **Re-sampling**：通过对**head class**进行欠采样或对**tail class**进行过采样，人为地构造类别均衡的数据集。包括**Random under/over-sampling**, **Class-balanced sampling**, **Meta Sampler**等。
- 重加权 **Re-weighting**：在损失函数中对不同类别样本的损失设置不同的权重，通常是对**tail class**对应的损失设置更大的权重。其中在$\log$运算之外调整损失函数的本质是在调节样本权重或者类别权重(如**Inverse Class Frequency Weighting**, **Cost-Sensitive Cross-Entropy Loss**, **Focal Loss**, **Class-Balanced Loss**)。在$\log$运算之内调整损失函数的本质是调整**logits**得分$z$，从而缓解对**tail**类别的负梯度(如**Equalization Loss**, **Equalization Loss v2**, **Logit Adjustment Loss**, **Balanced Softmax Loss**, **Seesaw Loss**)。
- 其他方法：一些方法将长尾分布问题解耦为特征的表示学习和特征的分类。一些方法按照不同类别的样本数量级对类别进行分组(如**BAGS**)。



### ⚪ 

- [深度学习的可解释性](https://0809zheng.github.io/2020/04/28/explainable-DL.html)
- [对抗攻击](https://0809zheng.github.io/2020/04/30/adversarial-attack.html)：[图像分类中的对抗攻击](https://0809zheng.github.io/2020/07/26/adversirial_attack_in_classification.html)、[目标检测中的对抗攻击](https://0809zheng.github.io/2020/07/25/adversirial_attack_in_object_detection.html)
- [多任务学习](https://0809zheng.github.io/2021/08/28/MTL.html)


## () 网络压缩
网络压缩旨在平衡网络的准确性和运算效率。
压缩预训练的网络 设计新的网络结构
- [网络压缩](https://0809zheng.github.io/2020/05/01/network-compression.html)：网络剪枝、知识蒸馏、结构设计、模型量化

# 3. 深度学习的应用

## (1) 计算机视觉

### ⚪ [图像分类](https://0809zheng.github.io/2020/05/06/image-classification.html)

### ⚪ [目标检测](https://0809zheng.github.io/2020/05/31/object-detection.html)

- [目标检测中的回归损失函数](https://0809zheng.github.io/2021/02/01/iouloss.html)
- [提高非极大值抑制算法的精度](https://0809zheng.github.io/2021/05/12/nms_accuracy.html)
- [提高非极大值抑制算法的效率](https://0809zheng.github.io/2021/05/11/nms_efficiency.html)

### ⚪ [语义分割](https://0809zheng.github.io/2020/05/07/semantic-segmentation.html)
- [图像分割的评估指标](https://0809zheng.github.io/2021/09/09/segmenteval.html)：**PA**, **CPA**, **MPA**, **IoU**, **MIoU**, **FWIoU**, **Dice Coefficient**

### ⚪ [人体姿态估计](https://0809zheng.github.io/2020/05/08/pose-estimation.html)

- [三维人体模型](https://0809zheng.github.io/2021/01/07/3dhuman.html)
- [人体姿态估计的评估指标](https://0809zheng.github.io/2020/11/26/eval-pose-estimate.html)

### ⚪ [图像超分辨率](https://0809zheng.github.io/2020/08/27/SR.html)

- [人脸检测](https://0809zheng.github.io/2020/05/09/face-detection.html)
- [人脸识别](https://0809zheng.github.io/2020/05/10/face-recognition.html)
- [行人检测](https://0809zheng.github.io/2020/05/11/pedestrian-detection.html)
- [行人属性识别](https://0809zheng.github.io/2020/05/12/pedestrian-attribute-recognition.html)
- [文本检测与识别](https://0809zheng.github.io/2020/05/15/text-detection-recognition.html)


## (2) 自然语言处理

- [文本摘要](https://0809zheng.github.io/2020/05/13/text-summary.html)
- [图像描述](https://0809zheng.github.io/2020/05/14/image-caption.html)
- [连接时序分类](https://0809zheng.github.io/2020/06/11/ctc.html)
- [音乐生成](https://0809zheng.github.io/2020/10/26/musicgen.html)


## (3) 自然语言处理





### 深度学习的相关课程
- [Deep Learning | Coursera （Andrew Ng）](https://www.coursera.org/specializations/deep-learning)
- [吴恩达Tensorflow2.0实践系列课程](https://www.bilibili.com/video/BV1zE411T7nb?from=search&seid=890015452850895449)
- [CS231n：计算机视觉（李飞飞）](http://cs231n.stanford.edu/syllabus.html)
- [CS294-158：深度无监督学习](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
- [旷视x北大《深度学习实践》](https://www.bilibili.com/video/BV1E7411t7ay)



### 深度学习的相关书籍
- [《神经网络与深度学习》（邱锡鹏）](https://nndl.github.io/)
- [《动手学深度学习》（李沐等）](http://zh.d2l.ai/)

