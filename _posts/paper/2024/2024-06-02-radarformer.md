---
layout: post
title: 'RadarFormer: End-to-End Human Perception With Through-Wall Radar and Transformers'
date: 2024-06-02
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/674da14ad0e0a243d4dc1825.png'
tags: 论文阅读
---

> RadarFormer：基于Transformer的穿墙雷达端到端人体感知.

- paper：[RadarFormer: End-to-End Human Perception With Through-Wall Radar and Transformers](https://ieeexplore.ieee.org/document/10260326)

# 1. 背景介绍

通常来说，适用于穿墙雷达图像的人体感知方法都是基于卷积神经网络设计的，遵循如下图所示的通用信号处理流程，即首先采集原始的雷达回波信号，通过雷达成像算法生成雷达图像，再通过卷积神经网络进行特征提取和下游任务的预测。

![](https://pic.imgdb.cn/item/674da172d0e0a243d4dc185a.png)

这种通用信号处理流程存在一些缺点。首先，雷达成像算法的选择是经验性的，需要精心设计；这将不可避免地丢失一些有用的信息并引入系统误差，且带来额外的计算负担，进一步阻碍了端到端的信号处理。其次，卷积神经网络与雷达信号的物理特性不完全匹配。卷积神经网络的归纳偏置包括局部性和空间不变性，这并不完全适用于雷达信号。由于雷达图像的空间分辨率远低于光学图像，低成像分辨率使得每幅雷达图像中包含的人体信息不完整，通常只包括人体的一部分。为了从雷达信号中提取出更高质量的目标信息，应该将注意力集中于信号的全局上下文特征。

本文设计了基于自注意力的穿墙雷达端到端人体感知方法**RadarFormer**，通过这种方式能够绕过雷达成像算法，实现端到端的信号处理（如图中红色箭头所示）。

# 2. 基于自注意力机制的穿墙雷达数据处理方法

本节提供了一个建设性的证明，表明在某些假设下，使用自注意力模型从雷达回波信号中提取的特征与使用卷积层从雷达图像中提取的特征具有相同的特征表示能力。通过分析基于卷积层的雷达图像特征提取过程与基于自注意力的雷达回波特征提取过程，证明了两种信号处理方法的等价性。

## (1) 使用卷积层处理雷达图像

假设雷达系统由$M$个发射天线和$N$个接收天线组成，则具有$MN$个等效信道。$m=1,2,...,M$表示第$m$个发射天线，$n=1,2,...,N$表示第$n$个接收天线。成像区域中点$p$的BP成像结果表示为：

$$
    \boldsymbol{I}(p) = \sum_{mn=1}^{MN} \boldsymbol{S}\left(mn,\tau_{mn,p}\right)
$$

其中$$\boldsymbol{S}$$是雷达回波信号，$\tau_{mn,p}$为点$p$从第$m$个发射天线到第$n$个接收天线（$m =1,2,...,M; n =1,2,...,N$）的往返传播时延。

基于卷积层的雷达图像特征提取过程如图所示。大小为$N_K\times N_K\times N_K$的三维卷积核以光栅扫描顺序在输入雷达图像$$\boldsymbol{I}$$上滑动。当滑动到空间像素$p$时，$p$的特征由其邻域贡献，并通过局部仿射变换提取：

$$
    \boldsymbol{F}(p)=\sum_{\delta \in \Delta_K}^{}\boldsymbol{I}\left(p+\delta\right)\cdot W_{\delta}+b_1
$$

其中$\delta$是卷积核中的元素相对于其中心位置的位置偏移量。对于三维卷积核，$$\Delta_K = \left[-\lfloor \frac{N_K}{2} \rfloor,...,\lfloor \frac{N_K}{2} \rfloor\right]\times \left[-\lfloor \frac{N_K}{2} \rfloor,...,\lfloor \frac{N_K}{2} \rfloor\right]\times \left[-\lfloor \frac{N_K}{2} \rfloor,...,\lfloor \frac{N_K}{2} \rfloor\right]$$。$W_{\delta}$和$b_1$是卷积核参数。

![](https://pic.imgdb.cn/item/674da3fed0e0a243d4dc1b85.png)


一般认为，单个卷积核倾向于提取具有某种特定模式类型的特征。因此可以使用多个卷积核来提取包含不同信息的特征，从而产生更多的输出特征通道。从图中也可以看出，各点的特征包括在这个点附近感受野的信息。当堆叠更多卷积层时，感受野会增大。在$p$点提取的特征也可以用雷达回波信号$$\boldsymbol{S}$$表示为：

$$
    \boldsymbol{F}(p)=\sum_{\delta \in \Delta_K}^{}\sum_{mn=1}^{MN}\boldsymbol{S}\left(mn,\tau_{mn,p+\delta}\right)\cdot W_{\delta}+b_1 \tag{1}
$$

## (2) 使用自注意力处理雷达回波

自注意力模型最初被设计用于处理自然语言中的序列数据，近年来逐渐被应用并普及到计算机视觉等其他领域。有证据表明自注意力模型可以模拟任何连续函数，并且是图灵完备的。注意到原始雷达回波数据以等效信道的顺序存储，每个通道都携带复杂的时间序列信息，这自然适合用自注意力模型进行直接处理。由于自注意力模型可以通过高度并行的计算过程捕获特征的全局相关性，因此它可以在保持性能的同时有效地减少推理时间。

基于自注意力模型的雷达回波特征提取过程如图所示。自注意力模型通过查询输入序列中任意位置与其他所有位置的相关性来计算每个位置的输出特征，反映在基于点积的查询-键-值模式中。

![](https://pic.imgdb.cn/item/674da2b8d0e0a243d4dc19ea.png)

首先通过仿射变换将输入序列$$\boldsymbol{S}$$转换为查询矩阵$$\boldsymbol{Q}$$、键矩阵$$\boldsymbol{K}$$和值矩阵$$\boldsymbol{V}$$：

$$
    \boldsymbol{Q}=\boldsymbol{S}W_q,\boldsymbol{K}=\boldsymbol{S}W_k,\boldsymbol{V}=\boldsymbol{S}W_v
$$

其中$W_q, W_k, W_v$表示仿射权值参数。然后使用点积注意力算子（忽略缩放标量）和归一化的**Softmax**函数来计算输出特征$$\boldsymbol{F}\left(\boldsymbol{S}\right)$$：

$$
    \boldsymbol{F}\left(\boldsymbol{S}\right)= \text{softmax}\left(\boldsymbol{Q}\boldsymbol{K}^T\right)\boldsymbol{V}= \text{softmax}\left(\boldsymbol{S}W_qW_k^T\boldsymbol{S}^T\right)\boldsymbol{S}W_v
$$

为了在具有差异性的特征空间中提取更多的互相关信息，在实践中将自注意力模型调整为多头形式。假设总共使用了$N_H$个自注意力头，每个头部都执行标准的自注意力运算。连接每个头部$h=1,2,...,N_H$的所有中间特征$$\boldsymbol{F}^h(S)$$，并通过一个可学习的仿射变换计算多头自注意力的最终输出特征$$\boldsymbol{F}^H$$：

$$
    \boldsymbol{F}^H\left(\boldsymbol{S}\right)= \mathop{\text{concat}}_{h \in N_H}\left[\boldsymbol{F}^h\left(\boldsymbol{S}\right)\right]\cdot W_{N_H}+b_2 \tag{2}
$$

其中$W_{N_H}$和$b_2$为仿射变换的参数。$b_2$通常被设置为零，在公式中显示地保留它，以保持与卷积层特征处理的一致性。

## (3) 两者的等价性

观察公式$(2)$，如果矩阵$W_{N_H}$被划分为$N_H$个子矩阵$W_{N_H}=\left[W_{N_H}^1,...,W_{N_H}^h,...,W_{N_H}^{N_H}\right]$，则可以交换级联操作和矩阵乘法运算的执行顺序，这相当于执行矩阵乘法运算后再求和，下面进行详细说明。假设序列的长度为$T$，多头自注意力机制共采用$N_H$头，每个头的特征尺寸为$n$。则级联操作会构造一个大小为$T\times nN_H$的特征矩阵，如图所示。

![](https://pic.imgdb.cn/item/674da4d1d0e0a243d4dc1c6b.png)

然后使用仿射变换来处理特征矩阵（为简单起见省略偏置参数）。仿射变换矩阵$W_{N_H}$的大小为$nN_H\times mN_H$，则矩阵乘法运算会构造大小为$T\times mN_H$的输出矩阵，如图所示。

![](https://pic.imgdb.cn/item/674da4e5d0e0a243d4dc1c7f.png)

另一方面，沿列维度将仿射变换矩阵$W_{N_H}$划分为$N_H$个子矩阵$W_{N_H}=\left[W_{N_H}^1,...,W_{N_H}^h,...,W_{N_H}^{N_H}\right]$，其中每个子矩阵的大小为$n\times mN_H$。对每个自注意力头的特征（尺寸为$T\times n$）和对应的子矩阵转置执行矩阵乘法运算，以获得$N_H$个尺寸为$T\times mN_H$的中间特征。最后对这些中间特征进行求和运算，可以得到与上述相同的大小为$T\times mN_H$的输出矩阵，如图所示。

![](https://pic.imgdb.cn/item/674da4f9d0e0a243d4dc1c89.png)

上述两条运算链是完全等价的，实际上对应着矩阵乘法的两种不同解释。对于第一条运算链，矩阵乘法被解耦为为第一个矩阵的行向量和第二个矩阵的列向量的一系列内积（对应于不同的输出位置）；对于第二条运算链，矩阵乘法被解耦为第一个矩阵的列向量和第二个矩阵的行向量的一系列外积然后求和。在此基础上，基于多头自注意力模型的雷达回波特征提取结果可以重写为：

$$
    \begin{aligned}
        \boldsymbol{F}^H(\boldsymbol{S})&= \sum_{h=1}^{N_H}\boldsymbol{F}^h(\boldsymbol{S})\cdot W_{N_H}^h +b_2 \\ &= \sum_{h=1}^{N_H} \text{softmax}\left(\boldsymbol{S}W_q^h {(W_k^h)}^T\boldsymbol{S}^T\right) \boldsymbol{S}W_v^h\cdot W_{N_H}^h +b_2
        \end{aligned}
$$

为简化标记，记$W_1^h=W_q^h {(W_k^h)}^T,W_2^h=W_v^h W_{N_H}^h$。则上式等价于：

$$
    \boldsymbol{F}^H(\boldsymbol{S})= \sum_{h=1}^{N_H} \text{softmax}\left(\boldsymbol{S}W_1^h\boldsymbol{S}^T\right)\cdot \boldsymbol{S}\cdot W_2^h +b_2
$$

由于Softmax函数将输入矩阵的每一列转换为归一化的概率分布，将Softmax$\left(\boldsymbol{S}W_1^h\boldsymbol{S}^T\right)$重写为一个掩码序列$\left[\text{mask}_{1}^h,...,\text{mask}_{mn}^h,...,\text{mask}_{MN}^h\right]$。最后将多头自注意力的输出特征表示为：

$$
    \boldsymbol{F}^H(\boldsymbol{S})= \sum_{h=1}^{N_H}\sum_{mn=1}^{MN} \text{mask}_{mn}^h\cdot \boldsymbol{S}\left(mn,t\right)\cdot W_2^h +b_2  \tag{3}
$$

对比公式$(1)$与公式$(3)$，不难发现通过卷积层从雷达图像中提取的特征可以与通过自注意力模型从雷达回波中提取的特征建立相似关系，从而导出如下定理：

- **定理1**：使用卷积核大小为$N_K\times N_K\times N_K$的三维卷积层从后向投影成像算法生成的三维雷达图像中提取的特征等价于通过$N_H$头的自注意力模型直接从雷达回波中提取的特征。

通过适当设置卷积层和多头自注意力模型的参数取值，可以建设性地证明该定理。下面提供了一组充分条件：
1. 卷积核的大小$N_K\times N_K\times N_K$和多头自注意力头的数量$N_H$满足$N_K=\sqrt[3]{N_H}$；
2. 第$h$个自注意力头对应于卷积核的第$\delta$个位置，$\text{mask}_{mn}^h$为单位热向量，其中位置$\tau_{mn,p+\delta}$处取值为1，其他位置取值为0；
3. 可学习参数$W_{\delta}=W_2^h,b_1=b_2$。

需要强调的是，上述条件并不是满足定理的唯一条件集合。虽然这些问题在实际应用中要复杂得多，如卷积层具有可选的填充和步长参数；神经网络能够从大量的数据中自动学习网络参数，适当的最优参数也能与定理的条件相匹配。

总之，该定理提供了一种从雷达回波中提取有用特征的可能性，而无需实际的成像过程。此外，自注意力模型可以并行操作，推理速度更快，内存占用更小，可以实现端到端信号处理。

# 3. RadarFormer模型

**RadarFormer**是一种基于快慢时间自注意力的穿墙雷达人体感知模型，旨在从原始的穿墙雷达回波中提取人体的细粒度姿态信息，并将其表示为人体姿态估计任务的具体预测结果。雷达回波存储为序列数据$$\boldsymbol{S} \in \mathbb{R}^{MN\times T}$$，其中$MN$为收发天线组合数，将其视为序列长度。$T$是雷达信号的快时间。

## (1) 基本流程

**RadarFormer**遵循大多数自注意力模型采用的编码器-解码器架构。值得注意的是，部分人体姿态估计任务同时依赖于空间和时间信息，这意味着单个雷达回波所携带的信息是不充足的。例如，很难仅根据一帧来区分目标在当前时刻是坐下还是站起来。此外，射频信号的物理特性导致单个回波捕捉的人体区域不完整。因此除了将单个雷达回波作为模型的输入外，之前的$N−1$帧连续回波沿时间维度融合后作为附加输入。将单个雷达回波称为当前回波$$\boldsymbol{S}^{\text{curr}}$$，将前$N−1$帧的融合回波称为历史回波$$\boldsymbol{S}^{\text{hist}}$$。当前回波沿着快时间维度存储空间信息，同时历史回波沿着慢时间维度存储目标的时序信息。

![](https://pic.imgdb.cn/item/674ee76dd0e0a243d4dcc62b.png)

历史回波和当前回波通过嵌入维度$d$的嵌入层，然后添加位置编码。位置编码主要有两个作用。第一个作用是向序列中隐式地添加索引信息。对于每个回波序列，第$mn$个位置表示由第$m$个发射天线发射并由第$n$个接收天线接收的信号。由于自注意力模型具有排列不变性，因此补充序列的索引差分信息。第二个作用是作为滤波器算子。在实际场景中，反射回波不仅来自感兴趣的人体目标，也来自墙壁等其他物体。此外目标的微小运动也会影响雷达信号。因此设置可学习的位置编码，相当于对雷达回波进行自适应滤波，以提高杂波干扰下雷达系统的目标检测能力。位置编码的维度与输入序列的嵌入维度相同，并直接添加到嵌入序列中。这实际上相当于首先构造一个带有位置索引的向量（例如one-hot向量）连接到输入序列，然后再通过嵌入层，如图所示。

![](https://pic.imgdb.cn/item/674ee7b3d0e0a243d4dcc6f3.png)

编码器将历史回波和当前回波映射为特征表示序列$$\boldsymbol{Z}$$。给定$$\boldsymbol{Z}$$和长度为$N_{\text{tar}}$的可学习目标查询向量$$\boldsymbol{Q}^{\text{tar}}$$，解码器为每个目标生成一个输出特征。为了进一步将这些特征转换为感兴趣的下游任务的预测，多层感知机（**Multi-Layer Perceptron，MLP**）作用于解码器的输出。由于不同下游任务的预测结果具有不同特征和维度，不适合使用单个**MLP**来同时解码所有任务。因此将输出任务解耦，为每个任务分配一个独立的**MLP**。特定于不同姿态估计任务的**MLP**通常具有简单的结构，并且可以通过灵活地调整**MLP**的数量自然地扩展到不同的人体姿态估计任务上。

## （2）目标函数

这项工作主要讨论三种下游人体姿态估计任务，即关节点定位任务、动作识别任务和身份识别任务。
- **关节点定位任务**旨在定位人体关节点的三维空间坐标。预定义$C_{\text{pose}}$个关节点，则每个目标的关节点定位任务的输出记为$$\boldsymbol{y}^{\text{pose}} \in \mathbb{R}^{3\times C_{\text{pose}}}$$；
- **动作识别任务**旨在沿着时序维度分析人体动作。预定义$C_{\text{act}}$个动作类别，则每个目标的动作识别任务的输出记为$$\boldsymbol{y}^{\text{act}} \in \mathbb{R}^{C_{\text{act}}}$$；
- **身份识别任务**旨在识别不同感兴趣目标的身份。预定义$C_{\text{id}}$个目标身份，则每个目标的身份识别任务的输出记为$$\boldsymbol{y}^{\text{id}} \in \mathbb{R}^{C_{\text{id}}}$$。

用$$\hat{\boldsymbol{Y}}=\left\{\hat{\boldsymbol{Y}}_i\right\}_{i=1}^{N_{\text{tar}}}$$表示预测集，其中第$i$个元素是$$\hat{\boldsymbol{Y}}_i=\left(\hat{p}_i,\boldsymbol{\hat{y}}^{\text{pose}}_i,\boldsymbol{\hat{y}}^{\text{act}}_i,\boldsymbol{\hat{y}}^{\text{id}}_i\right)$$。$\hat{p}_i$给出第$i$个查询向量找到目标的概率，$$\boldsymbol{\hat{y}}^{\text{pose}}_i$$给出对应的三维关节点联合坐标，$$\boldsymbol{\hat{y}}^{\text{act}}_i$$给出对应的动作类别，$$\boldsymbol{\hat{y}}^{\text{id}}_i$$给出对应的身份类别。同时提供了目标的真实标注集$$\boldsymbol{Y}$$，由于$N_{\text{tar}}$显著大于实际存在的目标数，因此将$$\boldsymbol{Y}$$的长度用空集填充至$N_{\text{tar}}$。由于预测序列是无序生成的，因此需要建立$$\boldsymbol{Y}$$与$$\hat{\boldsymbol{Y}}$$之间的最优二分匹配$$\tilde{\mathcal{M}}$$。$$\tilde{\mathcal{M}}$$取对$N_{\text{tar}}$个元素进行排列$$\mathcal{M} \in \boldsymbol{M}_{N_{\text{tar}}}$$的最低成本：

$$
    \tilde{\mathcal{M}} = \mathop{\arg \min}_{\mathcal{M} \in \boldsymbol{M}_{N_{\text{tar}}}} \sum_{i}^{N_{\text{tar}}} \mathcal{L}_{\text{assign}}\left(\boldsymbol{Y}_i,\hat{\boldsymbol{Y}}_{\tilde{\mathcal{M}}(i)}\right)
$$

其中成对的真值$$\boldsymbol{Y}_i$$和预测$$\hat{\boldsymbol{Y}}_{\tilde{\mathcal{M}}(i)}$$是由匈牙利算法给出的最优赋值。分配成本$$\mathcal{L}_{\text{assign}}\left(\boldsymbol{Y}_i,\hat{\boldsymbol{Y}}_{\tilde{\mathcal{M}}(i)}\right)$$考虑三维关节点坐标、动作类别和身份类别的联合预测损失，注意只有当目标存在时（$p_i=1$）才会进行分配：

$$
    \mathcal{L}_{\text{assign}}\left(\boldsymbol{Y}_i,\hat{\boldsymbol{Y}}_{\tilde{\mathcal{M}}(i)}\right) = p_i\mathcal{L}_{\text{pose}}\left(\boldsymbol{y}^{\text{pose}}_i,\boldsymbol{\hat{y}}^{\text{pose}}_{\tilde{\mathcal{M}}(i)}\right) + p_i\mathcal{L}_{\text{act}}\left(\boldsymbol{y}^{\text{act}}_i,\boldsymbol{\hat{y}}^{\text{act}}_{\tilde{\mathcal{M}}(i)}\right) + p_i\mathcal{L}_{\text{id}}\left(\boldsymbol{y}^{\text{id}}_i,\boldsymbol{\hat{y}}^{\text{id}}_{\tilde{\mathcal{M}}(i)}\right)
$$

其中三维关节点坐标的预测损失采用平均绝对损失：

$$
    \mathcal{L}_{\text{pose}}\left(\boldsymbol{y}^{\text{pose}}_i,\boldsymbol{\hat{y}}^{\text{pose}}_{\tilde{\mathcal{M}}(i)}\right) = \left\|\boldsymbol{y}^{\text{pose}}_i-\boldsymbol{\hat{y}}^{\text{pose}}_{\tilde{\mathcal{M}}(i)}\right\|_1
$$

动作类别的预测损失采用交叉熵损失：

$$
    \mathcal{L}_{\text{act}}\left(\boldsymbol{y}^{\text{act}}_i,\boldsymbol{\hat{y}}^{\text{act}}_{\tilde{\mathcal{M}}(i)}\right) = -\boldsymbol{y}^{\text{act}}_i \log \boldsymbol{\hat{y}}^{\text{act}}_{\tilde{\mathcal{M}}(i)}
$$

身份类别的预测损失也采用交叉熵损失：

$$
    \mathcal{L}_{\text{id}}\left(\boldsymbol{y}^{\text{id}}_i,\boldsymbol{\hat{y}}^{\text{id}}_{\tilde{\mathcal{M}}(i)}\right) = -\boldsymbol{y}^{\text{id}}_i \log \boldsymbol{\hat{y}}^{\text{id}}_{\tilde{\mathcal{M}}(i)}
$$

由于$$\tilde{\mathcal{M}}$$给出了预测值和真实值之间的最优一对一赋值，因此可以为后续的训练过程构建损失函数。一般来说，$N_{\text{tar}}$的数量远远大于实际可能存在目标的数量，这不可避免地引入了类别不平衡问题。为了缓解大量非目标查询造成的负梯度效应，将类别平衡损失添加到总损失中。则总损失表示为：

$$
\begin{aligned}
\mathcal{L}_{\text{total}} = \sum_{i}^{N_{\text{tar}}} [&-\frac{1}{E_{N_c}}\log \hat{p}_{\tilde{\mathcal{M}}(i)} + p_i\mathcal{L}_{\text{pose}}\left(\boldsymbol{y}^{\text{pose}}_i,\boldsymbol{\hat{y}}^{\text{pose}}_{\tilde{\mathcal{M}}(i)}\right)\\& + p_i\mathcal{L}_{\text{act}}\left(\boldsymbol{y}^{\text{act}}_i,\boldsymbol{\hat{y}}^{\text{act}}_{\tilde{\mathcal{M}}(i)}\right) + p_i\mathcal{L}_{\text{id}}\left(\boldsymbol{y}^{\text{id}}_i,\boldsymbol{\hat{y}}^{\text{id}}_{\tilde{\mathcal{M}}(i)}\right)]
\end{aligned}
$$

## （3）网络结构

**RadarFormer**是一种编码器-解码器结构的快慢时间自注意力模型。其中编码器堆叠了$N_E$块由慢时间自注意力层、快时间自注意力层和逐位置全连接层组成的模块，并且在每两个层之间引入了残差连接和层归一化。编码器结构如图所示。

![](https://pic.imgdb.cn/item/674ee90fd0e0a243d4dcc9ce.png)

编码器将当前回波$$\boldsymbol{S}^{\text{curr}}$$和历史回波$$\boldsymbol{S}^{\text{hist}}$$首先输入一个慢时间自注意力层。由于当前回波$$\boldsymbol{S}^{\text{curr}}$$和历史回波$$\boldsymbol{S}^{\text{hist}}$$在慢时间维度上携带不同的时间信息，因此慢时间自注意力鼓励当前雷达回波查询历史时刻的目标状态。计算慢时间自注意力的查询矩阵$$\boldsymbol{Q}$$来自当前回波$$\boldsymbol{S}^{\text{curr}}$$，键矩阵$$\boldsymbol{K}$$和值矩阵$$\boldsymbol{V}$$来自历史回波$$\boldsymbol{S}^{\text{hist}}$$：

$$
    \boldsymbol{Q}=\boldsymbol{S}^{\text{curr}}W_q,\boldsymbol{K}=\boldsymbol{S}^{\text{hist}}W_k,\boldsymbol{V}=\boldsymbol{S}^{\text{hist}}W_v
$$

慢时间自注意力的计算可以通过标准的自注意力模型来表达：

$$
    \begin{aligned}\text{SlowTime}&\text{SelfAttn}\left(\boldsymbol{S}^{\text{curr}},\boldsymbol{S}^{\text{hist}}\right)\\:=&\text{SelfAttn}\left(\boldsymbol{S}^{\text{curr}}W_q,\boldsymbol{S}^{\text{hist}}W_k,\boldsymbol{S}^{\text{hist}}W_v\right)\end{aligned}
$$

之后通过建立残差连接来优化梯度传播：

$$
    \boldsymbol{S}^{\text{curr}}\leftarrow \text{SlowTimeSelfAttn}\left(\boldsymbol{S}^{\text{curr}},\boldsymbol{S}^{\text{hist}}\right)\oplus \boldsymbol{S}^{\text{curr}}
$$

通过建立残差连接，梯度信号可以从深层直接传播回浅层，进一步缓解了梯度消失和模型退化问题。然后采用层归一化对特征进行归一化：

$$
    \text{LayerNorm}\left(\boldsymbol{S}^{\text{curr}}\right) := \gamma \frac{\boldsymbol{S}^{\text{curr}}-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

其中$\mu,\sigma$表示所有输入元素的均值和标准差。$\gamma,\beta$可作为模型非线性补偿的可学习的重缩放和重中心化参数。$\epsilon$是一个用来保持数值稳定性的小数。

对于当前回波$$\boldsymbol{S}^{\text{curr}}$$，它在快时间维度上携带不同收发天线组合的空间范围信息，因此定义快时间自注意力层，通过自注意力交互来探索这些信息：

$$
    \begin{aligned}\text{FastTime}&\text{SelfAttn}\left(\boldsymbol{S}^{\text{curr}}\right)\\:=&\text{SelfAttn}\left(\boldsymbol{S}^{\text{curr}}W_q,\boldsymbol{S}^{\text{curr}}W_k,\boldsymbol{S}^{\text{curr}}W_v\right)\end{aligned}
$$

然后再次应用残差连接和层归一化：

$$
    \boldsymbol{S}^{\text{curr}}\leftarrow \text{LayerNorm}\left(\text{FastTimeSelfAttn}\left(\boldsymbol{S}^{\text{curr}}\right)\oplus \boldsymbol{S}^{\text{curr}}\right)
$$

然后将$$\boldsymbol{S}^{\text{curr}}$$输入到逐位置全连接层中。逐位置全连接层通过非线性变换增强了特征的表示能力，其中序列$$\boldsymbol{S}^{\text{curr}}$$的第$i$个向量转换为：

$$
    \text{FF}\left(\boldsymbol{S}^{\text{curr}}_i\right):= w_{f2}\sigma\left(w_{f1}\boldsymbol{S}^{\text{curr}}_i+b_{f1}\right)+b_{f2}
$$

式中$w_{f1},w_{f2}$和$b_{f1},b_{f2}$表示权重和偏置，$\sigma(\cdot)$表示非线性激活函数。最后通过残差连接和层归一化，编码器输出的特征表示$$\boldsymbol{Z}$$计算为：

$$
    \boldsymbol{Z}= \text{LayerNorm}\left(\text{FF}\left(\boldsymbol{S}^{\text{curr}}\right)\oplus \boldsymbol{S}^{\text{curr}}\right)
$$

解码器旨在从特征表示$$\boldsymbol{Z}$$中解码所有可能存在目标的特征。由于在检测场景中可能同时存在多个目标，用一个可学习的查询矩阵$$\boldsymbol{Q}^{\text{tar}} \in \mathbb{R}^{N_{\text{tar}} \times d}$$来感知所有可能存在的目标，其每一行向量对应一个目标。在训练过程中$$\boldsymbol{Q}^{\text{tar}}$$与其他模型参数共同更新。目标的最大查询数量$N_{\text{tar}}$被设置为显著大于检测空间中可能出现的典型目标数量。

解码器堆叠了$N_D$个由自注意力层、交叉注意力层和逐位置全连接层组成的模块，如图所示。对于每个解码器模块，它使用编码器输出的特征表示$$\boldsymbol{Z}$$作为输入，并更新目标查询$$\boldsymbol{Q}^{\text{tar}}$$。具体来说，查询$$\boldsymbol{Q}^{\text{tar}}$$首先被发送到一个标准的自注意力层中。它并行地模拟了每两个目标之间可能存在的关系，这与原始自注意力模型中解码器的自回归计算不同。

![](https://pic.imgdb.cn/item/674ee9ccd0e0a243d4dccb62.png)

自注意力层使用$$\boldsymbol{Q}^{\text{tar}}$$生成查询矩阵、键矩阵和值矩阵：

$$
    \boldsymbol{Q}^{\text{tar}} \leftarrow \text{SelfAttn}\left(\boldsymbol{Q}^{\text{tar}}W_q,\boldsymbol{Q}^{\text{tar}}W_k,\boldsymbol{Q}^{\text{tar}}W_v\right)
$$

然后使用交叉注意力层将特征表示$$\boldsymbol{Z}$$和目标查询$$\boldsymbol{Q}^{\text{tar}}$$相结合，提取目标特征。其中键矩阵和值矩阵由$$\boldsymbol{Z}$$生成，查询矩阵由$$\boldsymbol{Q}^{\text{tar}}$$生成：

$$
    \text{CrossAttn}\left(\boldsymbol{Q}^{\text{tar}},\boldsymbol{Z}\right) := \text{SelfAttn}\left(\boldsymbol{Q}^{\text{tar}}W_q,\boldsymbol{Z}W_k,\boldsymbol{Z}W_v\right)
$$

在每个注意力层之后也应用了残差连接和层标准化。通过这种方式，解码器将为每个目标输出携带丰富感知信息的特征。

# 4. 实验分析

## （1）数据集采集

使用自研穿墙雷达系统构造一个穿墙雷达人体感知数据集。雷达系统以1 m的中心高度工作，检测场景的高度向范围为-2 m至2 m，方位向范围为-2.5 m至2.5 m，距离向范围为0至6 m。雷达系统探测存在人体目标的检测区域，区分目标的主要反射点，并将信息存储在雷达回波中。回波的大小是$(MN, T) = (32, 640)$。考虑到雷达系统探测的局限性，在每个检测场景中最多出现10个目标。

共采集162280对雷达回波和光学图像作为训练集，并采用5折交叉验证将其划分为训练子集与验证子集，分别进行模型训练与性能评估。额外在墙壁遮挡和低能见度场景中仅采集雷达回波作用于泛化性评估，共采集32480个雷达回波。由于单个雷达回波携带的人体信息不完整，因此为每个当前回波保留之前的5帧作为历史回波。

使用同步采集的光学信号为雷达回波生成标签。具体而言，对于三维关节点定位任务，预定义$C_{\text{pose}}=14$个关节点（包括头部、胸部、肩部、肘部、腕部、臀部、膝部和踝部），使用Kinect SDK提供的人体关节点坐标作为标签。对于动作识别任务，预定义$C_{\text{act}}=8$个常见的动作类别（包括站立、行走、跳跃、挥手、伸展、拳击、坐下和躺下），并根据光学图像中出现的实际情况标记所有目标。对于身份识别任务，人工标记在场景中可能重复出现的$C_{\text{id}}=10$个目标身份，在数据集中这些目标的衣着和发型可能会发生变化。

为了提供对雷达信号的直观理解，一个雷达回波样本及其对应的雷达图像投影结果如图所示，且回波的快时间维度已等效地转换为距离向维度。从回波中可以观察到，目标出现在距离雷达系统约2 m和5 m的地方。但是回波信号只提供了收发天线和目标反射点之间的相对位置关系，因此不可能直接获得更复杂的人体信息，例如人体关节点坐标。当提供天线阵列的空间排布时，可以通过BP成像算法将雷达回波转换为雷达图像。直接生成的三维雷达图像不容易可视化，因此沿方位向维度对其进行投影，对应的二维高度向-距离向投影图像如图所示。从雷达图像中可以观察到更多关于目标的信息，例如有效地区分目标身体的一部分。也可以按此方式对高度向-方位向投影图像或方位向-距离向投影图像做类似分析。通常雷达回波和雷达图像之间的转换可以表示为由雷达阵列属性参数化的非线性函数。这项工作试图在神经网络的优化过程中对其进行隐式建模。

![](https://pic.imgdb.cn/item/674eeb41d0e0a243d4dcce1d.png)

## （2）评估指标与方法

在定量评估过程中，雷达回波被转换为相关人体姿态任务的预测结果，并与预先提供的标签集合进行比较。使用**MPJPE**指标来计算人体三维关节点坐标的回归精度。**MPJPE**指标计算每个关节点坐标的均方误差，并取所有关节点误差的平均值。使用平均准确率（**mean Accuracy，mAcc**）指标来评估动作识别任务中每个动作类别的识别准确程度。对于身份识别任务，从评估样本中随机选择一个包含每个身份的信号来构建查询集，并将其他信号保留为库集。解码器将查询集信号和库集信号解码为身份特征，计算每个查询集信号特征和每个库集信号特征之间的余弦相似度距离并对其进行排序，以检索每个查询集信号特征的前K个最接近的库集信号特征。使用**mAP**指标和累积匹配曲线（**Cumulative Matching Curve，CMC**）来评估身份识别性能。**CMC**指标表示查询集身份出现在不同大小的库集候选身份列表中的概率。无论库集中有多少个匹配身份，在**CMC**指标计算中通常只考虑第一个匹配（**CMC rank-1**）或前五个匹配（**CMC rank-5**）。除了性能指标外，计算成本也是衡量方法质量的重要指标。这项工作选择了两个计算成本指标：参数量和**FLOPs**指标。参数量统计模型中参数的总数。参数越大的模型占用的内存越多，消耗的空间资源也越多。**FLOPs**指标统计模型每秒算术运算次数。通常具有较大**FLOPs**的模型具有较高的计算复杂度，并且消耗更多的计算资源。

将所提方法与以下几种最近提出的人体姿态估计方法进行比较，以进一步研究不同方法的性能，包括几种基于光学信号的方法和基于雷达信号的方法。所有的比较方法都是在相同的实验环境中训练和评估的。

- 关节点定位任务的对比方法：
1. **OpenPose**模型是一种光学人体关节点定位方法，它为每个关节点进行编码生成关节热图，并为肢体部位生成部位亲和场。将同步采集的光学图像作为该方法的输入数据；
2. **RF-Pose3D**模型是一种基于**WiFi**阵列的多目标关节点定位方法，它部署了两个正交的**WiFi**线性阵列来收集整个检测区域的三维图像。将BP成像算法应用于采集的雷达回波数据集，并将感兴趣区域离散为尺寸为$160\times 160\times 160$的三维雷达图像作为该方法的输入数据；
3. **JGLNet**模型是一种基于毫米波雷达的单目标关节点定位方法，它通过全局分支和局部分支直接从雷达回波中提取空间-时间特征，这两个分支都由**CNN**和长短期记忆网络（**Long Short-Term Memory，LSTM**）组成的。
- 动作识别任务的对比方法：
1. **DVCNN**模型是一种基于雷达点云的人体动作识别方法，它通过双视图**CNN**学习动作特征。将**BP**成像算法应用于采集的雷达回波数据集，并将生成的雷达图像进一步离散化为雷达点云作为该方法的输入数据；
2. **ST-GCN**模型也是一种基于雷达点云的人体动作识别方法，它使用图卷积网络（**Graph Convolutional Network，GCN**）直接从雷达点云坐标中提取动作时空信息。该方法的输入数据与**DVCNN**模型相同；
3. **Split BiRNN**模型首先使用**CNN**直接提取雷达回波信号中的低级局部特征，然后用门控递归单元（**Gated Recurrent Unit，GRU**）提取人体动作的高级全局特征。
- 身份识别任务的对比方法：
1. **ReID-baseline**模型是一种光学人体身份重识别方法，它通过结合多种方法的结构设计和训练技巧实现令人满意的人体身份识别表现。将同步采集的光学图像作为该方法的输入数据；
2. **RF-ReID**模型是一种基于雷达的长期人体身份识别方法，它从雷达高度向-距离向热图和方位向-距离向热图中提取人体目标的长期特征，并驱动模型学习具有足够信息的人体身份特征。将**BP**成像算法应用于采集的雷达回波数据集，并沿着相应的维度进行投影构造尺寸为$160\times 160$的二维雷达图像作为该方法的输入数据。

## （3）实验设置

模型堆叠$N_E=6$个编码器模块和$N_D=6$个解码器模块。多头自注意力头的数量设置为$N_H=8$，特征维度设置为$d=256$。模型中所有全连接层的神经元数量设置为$2048$。对于关节点定位任务，输出MLP采用三个全连接层，隐藏层维度分别为256、256和42，前两层使用ReLU激活。对于动作识别任务和身份识别任务，输出MLP都是简单的Softmax激活全连接层，输出维度分别为8和10。

对模型训练500个训练轮数。数据批量大小设置为256。使用AdamW优化器训练模型，权重衰减率设置为$0.0001$。初始学习率设置为$0.0001$，并在400个训练轮数后减少$50\%$。所有实验均通过PyTorch实现，并在Nvidia RTX3090 GPU平台上进行加速。

## （4）人体姿态估计任务的性能分析

所有关节点定位方法在训练时的定位误差曲线如图所示。从图中可以看出，随着训练迭代次数的增加，不同方法的定位误差均有不同程度地减小。所提方法的定位误差随着训练周期的增加逐渐收敛到较小值，平均定位误差在大约300个周期后低于所有其他方法。

![](https://pic.imgdb.cn/item/674eec35d0e0a243d4dcd050.png)

所有关节点定位方法的性能和计算成本列于表中。由于为光学图像设计的OpenPose模型经历过精细的模型结构迭代，并且具有更高分辨率的光学图像可以提供更丰富的人体信息，因此光学方法的结果通常被视为基于雷达的方法的参考上界。在本实验中，处理两种模态数据的方法存在≥6.3 mm的性能差距。RF-Pose3D模型从二维雷达图像投影中提取信息，JGLNet模型通过CNN直接处理雷达回波；这两种模型的数据处理方式阻碍了三维关节点坐标的定位准确性，尽管它们减小了卷积神经网络的参数量。所提方法实现定位精度和计算成本的最佳平衡。在当前实验环境下，雷达回波的单次前向传播仅需约88 ms的处理时间，能够满足系统的实时性要求。

![](https://pic.imgdb.cn/item/674eec7ad0e0a243d4dcd0e5.png)

上表还报告了预定义的14个关节点的定位性能。当定位反射面积较大的关节点时，所提方法的误差较小，如胸部关节点的定位误差是28.9mm。这些关节点部位通常移动缓慢，很容易被雷达信号捕捉到。当定位具有较小反射面积和灵活运动趋势的关节点时误差相对较大，如腕部关节点的定位误差是45.3 mm。在单帧雷达回波中，这些较小的关节点很容易被忽略。尽管模型试图从整个数据集中学习这些关节点的统计特征，但这部分关节点的性能仍然比其他关节点差。

## （5）人体动作识别任务的性能分析

所有动作识别方法的性能和计算成本列于表中。DVCNN模型性能受限的主要原因是从雷达回波到雷达图像再到雷达点云的多次转换中累积的误差。此外，该模型的计算成本大于直接处理雷达点云的模型。ST-GCN模型的性能与基于CNN的方法相当，并且计算成本更低。Split BiRNN模型在基于雷达的动作识别任务中取得了令人满意的性能，但由于其所使用的递归单元无法并行化，因此具有较高的计算延迟（2.10 G FLOPs）。结果表明，所提基于快慢时间自注意力的方法在所有动作识别方法中取得了最好的性能，mAcc指标达到0.890。这是因为自注意力模型可以捕获特征的全局信息，这使得该方法能够从雷达信号中提取有判别性的人体动作特征。此外所提方法具有较快的推理速度，FLOPs仅有0.38G，这也得益于自注意力计算的高效并行性。

![](https://pic.imgdb.cn/item/674eecc5d0e0a243d4dcd152.png)

上表还报告了这些方法对于不同人体动作类别的识别精度。结果表明，这些方法对于身体变化差异较小的动作（如挥手和拳击）的识别能力较弱。对于有明显区别的动作（如跳跃），方法的识别精度相对稳定。至于躺下这种动作，其对应雷达信号的分布与其他动作的信号差距较大，因此这些方法很容易将其区分开。此外，所有方法通常在区分站立这个动作时具有最高的相对识别精度。

所提方法在人体动作识别任务上的平均混淆矩阵如图所示。混淆矩阵中的每一行表示真实的动作类别，每一列表示预测的动作类别，颜色深浅表示识别准确度。由于电磁波的辐射特性，不同动作类别的雷达信号之间的混淆程度通常取决于人体目标在雷达平面上的等效投影面积。结果表明，挥手和拉伸等动作的识别准确率相对较低（约86\%），并且它们往往被错误地识别为彼此。这是因为这两种动作的人体状态相似，只是目标手臂的伸展程度略有不同。目标的细微差异可能不会导致雷达信号的明显变化，因此方法很难区分它们。至于站立和躺下这两种很容易根据人体躯干的重心进行物理区分的动作，识别精度要高得多，90\%以上的类别样本被正确识别。

![](https://pic.imgdb.cn/item/674eecf9d0e0a243d4dcd1d0.png)

## （6）人体身份识别任务的性能分析

所有身份识别方法的性能列于表中。对于身份识别任务，基于雷达的方法性能显著优于基于光学的方法。这是因为基于光学的方法侧重于提取目标的外观特征，如服装颜色或发型，这些特征在不同环境的变化中很容易失效。而基于雷达的方法由于无法感知衣服等非金属物体，因此它们倾向于学习目标的风格特征，这些特征是长期有效的，例如身体形状和运动习惯。所提方法进一步击败了基于CNN的方法。这归功于自注意力的全局交互性，它可以通过充分结合上下文信息来提取目标风格。相比之下，卷积层提取局部特征模式，这不足以支持长期的特征判别。

![](https://pic.imgdb.cn/item/674eed18d0e0a243d4dcd208.png)

为了对所提方法学习到的特征空间进行可视化解释，使用t-SNE算法将解码器的输出特征投影到二维平面上，如图所示。图中的每个点对应于模型为目标提取的特征向量，根据目标身份对其进行着色。结果表明特征分布在空间上是聚集的，进一步证明了所提方法成功地学习到具有空间判别性的目标特征，从而提高了在身份识别任务上的性能。

![](https://pic.imgdb.cn/item/674eed2dd0e0a243d4dcd235.png)

## （7）可视化分析

所提方法在验证集数据中的人体感知结果如图所示。结果表明，所提方法在不同的环境中能够准确估计不同目标的身份、动作和关节点信息。例如右图的两个结果显示，所提方法能够成功区分目标4和目标9，尽管他们在信号采集过程中被告知需要相互交换外套。

![](https://pic.imgdb.cn/item/674eed7cd0e0a243d4dcd2fc.png)


所提方法在测试集数据中的人体姿态估计结果如图所示。在这些墙壁遮挡或低能见度场合中光学系统完全失效，而所提方法仍然能够准确进行人体姿态估计。尽管在训练中从未见过这些场合的雷达回波数据，但所提方法在足够的训练样本中学习到人体不同关节点之间的相关性以及不同目标身体的统计约束，因此可以从未经训练场景采集的雷达回波信号中预测人体目标的姿态细粒度信息。

![](https://pic.imgdb.cn/item/674eed99d0e0a243d4dcd346.png)

预测结果中存在的几种错误情况如图所示。左图显示检测空间中存在四个目标。然而由于目标之间的相互遮挡，当在信号的同一传播方向上存在多个目标时，被遮挡的目标会被遗漏，导致结果只报告了三个目标。右图显示两个目标在检测空间中非常接近，导致目标的手臂被混淆。这是因为雷达信号缺乏精细的空间分辨率，并且目标之间的反射信号也可能导致结果混淆并损害性能。通过改进穿墙雷达系统以及改进数据集的质量，能够在一定程度上缓解这些问题。

![](https://pic.imgdb.cn/item/674eedafd0e0a243d4dcd377.png)

## （8）消融实验

为了更好地研究所提方法的有效性，对方法的关键组件进行了广泛的消融实验：编码器层的数量、解码器模块的数量以及快慢时间自注意力。所有消融实验均在相同的实验环境中进行。

编码器通过全局场景推断来感知雷达回波中所有可能存在的目标。通过改变编码器模块数来评估全局特征提取的重要性。具体来说，构造了八个不同模块数（从1到8）的编码器模型。这些模型对应的关节点定位MPJPE指标和FLOPs指标如图所示。结果表明，模型性能在在堆叠七个编码器模块时达到饱和。另一方面，当编码器模块数线性增加时，计算量几乎以相同的趋势增加。为了平衡性能和计算成本，在这项工作中设置了编码器模块数$N_E=6$。

解码器将目标查询向量和编码器提取的全局特征相结合，以进一步生成每个可能存在目标的特征。图显示关节点定位误差在解码器堆叠每个模块之后都会降低，在第一个模块和最后一个模块之间总共有非常显著的9.4 mm的定位误差改善。这是因为较浅的解码器模块不足以捕捉被查询目标之间的相关性，并且很容易重复预测同一目标。随着解码器模块数的增加，隐含地允许模型抑制重复的预测结果。然而过深的解码器模块数设置会减慢推理过程，损害模型的实时性能。$N_D=8$的模型推断时间大约是$N_D=1$的模型推断时间的1.5倍。在实验中设置解码器模块数$N_D=6$。

![](https://pic.imgdb.cn/item/674eedf3d0e0a243d4dcd3e9.png)

为了协同提取雷达回波中的时序与空间信息，所提方法设计了一种快慢时间自注意力模型。该模型首先通过当前雷达回波与历史回波的交叉互相关提取时序信息，然后通过每个当前回波的自相关提取空间信息。为了验证该模型的有效性，将其与仅执行雷达回波自相关的标准自注意力进行了比较，并在表中报告了三个下游任务的结果。结果表明，应用快慢时间自注意力后所有任务的性能都得到了提高，尤其是在动作识别等高度依赖时间信息的任务中。

![](https://pic.imgdb.cn/item/674eee17d0e0a243d4dcd42e.png)