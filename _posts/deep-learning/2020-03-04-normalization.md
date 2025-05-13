---
layout: post
title: '深度学习中的归一化方法(Normalization)'
date: 2020-03-04
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e79ca179dbe9d88c5e2e0b0.png'
tags: 深度学习
---

> Normalization in Deep Learning.

输入数据的特征通常具有不同的量纲和取值范围，使得不同特征的**尺度（scale）**差异很大。不同机器学习模型对数据特征尺度的敏感程度不同。如果一个机器学习算法在数据特征缩放前后不影响其学习和预测，则称该算法具有**尺度不变性（scale invariance）**，表示为$f(\lambda x)=f(x)$。理论上神经网络具有尺度不变性，但是输入特征的不同尺度会增加训练的困难：
1. **参数初始化困难**：当使用具有饱和区的激活函数$a=f(WX)$时，若特征$X$的不同维度尺度不同，对参数$W$的初始化不合适容易使激活函数陷入饱和区，产生**vanishing gradient**现象。
2. 梯度下降法的**效率下降**：如下图所示，左图是数据特征尺度不同的损失函数等高线，右图是数据特征尺度相同的损失函数等高线。由图可以看出，前者计算得到的梯度方向并不是最优的方向，需要迭代很多次才能收敛；后者的梯度方向近似于最优方向，大大提高了训练效率。![](https://pic.imgdb.cn/item/64167f52a682492fcc24777a.jpg)
3. **内部协方差偏移**(**Internal Covariance Shift**)：训练深度网络时，神经网络隐层参数更新会导致网络输出层输出数据的分布发生变化，而且随着层数的增加，这种偏移现象会逐渐被放大。神经网络本质学习的是数据分布，如果数据分布变化了，神经网络又不得不学习新的分布，当前后的要求不同时，可能会影响结果。

![](https://pic.downk.cc/item/5ea14edbc2a9a83be5cea36c.jpg)

**归一化（Normalization）**泛指把数据特征的不同维度转换到相同尺度的方法。深度学习中常用的归一化方法包括：
1. 基础归一化方法：最小-最大值归一化、标准化、白化、逐层归一化
2. 深度学习中的特征归一化：局部响应归一化**LRN**、批归一化**BN**、层归一化**LN**、实例归一化**IN**、组归一化**GN**、切换归一化**SN**
3. 改进特征归一化：（改进**BN**）**Batch Renormalization**, **AdaBN**, **L1-Norm BN**, **GBN**, **SPADE**；（改进**LN**）**RMS Norm**, **Pre-LN**, **Mix-LN**, **LayerNorm Scaling**；（改进**IN**）**FRN**, **AdaIN**
4. 深度学习中的参数归一化：权重归一化**WN**、余弦归一化**CN**、谱归一化**SN**
5. 不使用归一化的方法：**Fixup**, **SkipInit**, **ReZero**, **DyT**, **DyISRU**



# 1. 基础归一化方法

## （1）最小-最大值归一化 Min-Max Normalization

**最小-最大值归一化**是指将每个特征的取值范围归一到$[0,1]$之间。记共有$N$个样本，每个样本含有$D$个特征，其中第$n$个样本表示为$x_n=(x_{n1},...,x_{nD})$；则最小-最大值归一化表示为：

$$ x_{nd}\leftarrow \frac{x_{nd}-\min_d(x_{nd})}{\max_d(x_{nd})-\min_d(x_{nd})} $$

## （2）标准化 Standardization

**标准化**又叫**Z值归一化**（**Z-Score Normalization**），是指将每个特征调整为均值为**0**，方差为**1**：

$$
\begin{aligned}
μ_d&= \frac{1}{N} \sum_{n=1}^{N} {x_{nd}} \\
σ_d^2&= \frac{1}{N} \sum_{n=1}^{N} {(x_{nd}-μ_d)^2}\\
x_{nd}&\leftarrow \frac{x_{nd}-μ_d}{σ_d}
\end{aligned}
$$

## （3）白化 Whitening

**白化**在调整特征取值范围的基础上消除了不同特征之间的相关性，降低输入数据特征的冗余。具体地，将输入数据在特征方向上被特征值相除，使数据独立同分布(**i.i.d.**)，实现输入数据的零均值(**zero mean**)、单位方差(**unit variance**)、去相关(**decorrelated**)。

实现步骤：
1. 零均值：
$ \hat{X}=X-E(X) $
1. 计算协方差：
$ Cov(X)=E(XX^T)-E(X)(E(X))^T $
1. 去相关：
$ Cov(X)^ {-\frac{1}{2}} \hat{X} $

![](https://pic.downk.cc/item/5e7d917a504f4bcb04345594.png)

白化的主要缺点是对所有特征一视同仁，可能会放大不重要的特征和噪声；此外，对于深度学习，隐藏层使用白化时反向传播困难。

## （4）逐层归一化 Layer-wise Normalizaiton

**逐层归一化**是指将归一化方法应用于深度神经网络中，对神经网络每一个隐藏层的输入特征都进行归一化，从而提高训练效率。

逐层归一化的优点：
1. 更好的尺度不变性：通过对每一层的输入进行归一化，不论低层的参数如何变化，高层的输入保持相对稳定，网络具有更好的尺度不变性，可以更高效地进行参数初始化和超参数选择。
2. [更平滑的损失函数](https://arxiv.org/abs/1806.02375)：可以使神经网络的损失函数更平滑，使梯度变得更稳定，可以使用更大的学习率，提高收敛速度。
3. [隐形的正则化方法](https://arxiv.org/abs/1809.00846)：可以提高网络的泛化能力，避免过拟合。

# 2. 深度学习中的特征归一化

## （1）局部响应归一化 Local Response Normalization
- paper：[ImageNet Classification with Deep Convolutional Neural Networks](http://stanford.edu/class/cs231m/references/alexnet.pdf)

**局部响应归一化**受生物学中“[侧抑制](https://baike.baidu.com/item/%E4%BE%A7%E6%8A%91%E5%88%B6/10397049?fr=aladdin)”的启发，即活跃的神经元对于相邻的神经元具有抑制的作用。

**LRN**通常应用在**CNN**中，且作用于激活函数之后，对邻近的特征映射（表现为邻近的**通道**）进行归一化。假设一个卷积层的特征图为$$X \in \mathbb{R}^{C×H×W}$$，$H$和$W$是特征图的高度和宽度，$C$为通道数。指定$n$为归一化考虑的邻域通道数量，则**LRN**表示为：

$$
X^c \leftarrow \frac{X^c}{\left(k+\frac{α}{n}\sum_{c'=\max(1,c-\frac{n}{2})}^{\min(C,c+\frac{n}{2})} (X^{c'})^2\right)^β}
$$

超参数的取值：$k=1, α=0.0001, β=0.75$。

```python
torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)
```

## （2）批归一化 Batch Normalization
- paper：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

**批归一化（BN）**是指对神经网络每一个隐藏层的输入特征使用每一批次数据的统计量进行标准化。**BN**独立的对每一个特征维度计算统计量，并用**mini batch**的统计量作为总体统计量的估计（假设每一**mini batch**和总体数据近似同分布）。对每一个**mini batch**，计算每个特征维度的均值和（有偏的）方差，并对输入做标准化操作，其中$ε$保证了数值稳定性：

$$
y = \frac{x - E[x]}{\sqrt{Var[x]+\epsilon}}*\gamma + \beta
$$

注意到当使用具有饱和性质的激活函数（如**Sigmoid**）时，标准化操作会将几乎所有数据映射到激活函数的非饱和区（线性区），从而降低了神经网络的非线性表达能力。为了保证模型的表达能力不因标准化而下降，引入可学习的**rescale**和**reshift**操作$γ,β$。

**BN**一般应用在网络层（通常是仿射变换）后、激活函数前，此时仿射变换不再需要**bias**参数（$f(BN(WX+b))=f(BN(WX))$）；测试时，使用总体均值和方差的无偏估计进行标准化（有时也用训练时均值和方差的滑动平均值代替）:

$$
\begin{aligned}
\overline{\mu} &\leftarrow (1-m)*\overline{\mu}+m*E[x]\\
&+= m*(E[x]-\overline{\mu}) \qquad \text{in-place form} \\
\overline{\sigma}^2 &\leftarrow (1-m)*\overline{\sigma}^2+m*Var[x]\\
&+= m*(Var[x]-\overline{\sigma}^2)  \quad \text{in-place form}\\
\end{aligned}
$$

**BN**的作用包括：
1. 调整每一层输入特征的分布，减缓了**vanishing gradient**，可以使用更大的学习率;
2. **BN**具有权重缩放不变性，减少对参数初始化的敏感程度: $BN((λW)X) = BN(WX)$
3. 与总体分布差距较小的**mini batch**分布可以看作为模型训练引入了噪声，可以增加模型的鲁棒性，带有正则化效果；
4. [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604): 对损失函数的**landscape**增加了平滑约束，从而可以更平稳地进行训练。

**BN**适用于**mini batch**比较大、与总体数据分布比较接近的场合。在进行训练之前，要做好充分的**shuffle**。**BN**在运行过程中需要计算每个**mini batch**的统计量，因此不适用于动态的网络结构和**RNN**网络，也不适合**Online Learning**（**batchsize = 1**）。

### ⚪ BatchNorm1d：应用于MLP

记网络某一层的输入$$X=(x_{nd}) \in \mathbb{R}^{N×D}$$，$N$为**batch**维度，$D$为该层特征数（神经元个数），则**BN**表示为：

$$
\begin{aligned}
μ_d &= \frac{1}{N} \sum_{n=1}^{N} {x_{nd}} \\
σ_d^2&= \frac{1}{N} \sum_{n=1}^{N} {(x_{nd}-μ_d)^2} \\
\hat{x}_{nd}&= \frac{x_{nd}-μ_d}{\sqrt{σ_d^2+\epsilon}} \\
y_{nd} &= γ \hat{x}_{nd} + β
\end{aligned}
$$

```python
torch.nn.BatchNorm1d(
    num_features, eps=1e-05,
    momentum=0.1, affine=True,
    track_running_stats=True,
    device=None, dtype=None)
```

此时**BN**沿着特征维度$D$进行归一化，沿着批量维度$N$计算统计量因此也被称为时序**BN**（**Temporal Batch Normalization**）。

### ⚪ BatchNorm2d：应用于CNN

记网络某一层的输入$$X=(x_{nchw}) \in \mathbb{R}^{N×C×H×W}$$，$N$为**batch**维度，$C$为通道维度，$H,W$为空间维度，则**BN**表示为：

$$
\begin{aligned}
μ_c&= \frac{1}{NHW} \sum_{n=1}^{N} \sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}} \\
σ_c^2&= \frac{1}{NHW} \sum_{n=1}^{N} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {(x_{nchw}-μ_c)^2}}}\\
\hat{x}_{nchw}&= \frac{x_{nchw}-μ_c}{\sqrt{σ_c^2+ε}}\\
y_{nchw} &= γ \hat{x}_{nchw} + β
\end{aligned}
$$

```python
torch.nn.BatchNorm2d(
    num_features, eps=1e-05,
    momentum=0.1, affine=True,
    track_running_stats=True,
    device=None, dtype=None)
```

### ⚪ BatchNorm2d from sctratch

如果要实现类似 **BN** 滑动平均的操作，在 **forward** 函数中要使用原地（**inplace**）操作给滑动平均赋值。


```python
class BatchNorm2d(nn.Module):
    def __init__(self, dim, eps = 1e-5, momentum=0.1,):
        super(BatchNorm2d, self).__init__()\
        self.dim = dim
        self.eps = eps
        self.m = momentum
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            with torch.no_grad():
                self.running_mean += self.m * (mean - self.running_mean)
                self.running_var += self.m * (var * self.dim/(self.dim-1) - self.running_var) 
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).sqrt()
        return x_norm * self.gamma + self.beta
```

## （3）层归一化 Layer Normalization
- paper：[Layer Normalizaiton](https://arxiv.org/abs/1607.06450)

**层归一化（LN）**适用于序列模型（如**RNN,LSTM,Transformer**），最初提出是用来解决**BN**无法应用在**RNN**网络的问题。

**BN**沿**batch**维度计算统计量；而在**RNN**网络中，每一个样本句子的长度不固定，需要补零来统一长度，此时对于某个特征维度，有些样本可能是无意义的零填充，因此沿**batch**维度计算统计量是没有意义的。**LN**针对每一个训练样本计算统计量，即计算每个样本所有特征的均值和方差。

记网络某一层的输入$$X=(x_{nd}) \in \mathbb{R}^{N×D}$$，$N$为**batch**维度，$D$为该层特征数（神经元个数），则**LN**表示为：

$$
\begin{aligned}
μ_n &= \frac{1}{D} \sum_{d=1}^{D} {x_{nd}} \\
σ_n^2&= \frac{1}{D} \sum_{d=1}^{D} {(x_{nd}-μ_n)^2} \\
\hat{x}_{nd}&= \frac{x_{nd}-μ_n}{\sqrt{σ_n^2+\epsilon}} \\
y_{nd} &= γ \hat{x}_{nd} + β
\end{aligned}
$$


**LN**也包含可学习的**re-scale**和**re-center**参数$\gamma,\beta$，并且参数与单个样本的特征维度相同（作用于每个特征位置）；此外**LN**不需要在训练过程中动态地保存**mini batch**的均值和方差，节省了额外的存储空间。

**LN**的适用场合如下：
1. **LN**针对单个训练样本进行，不依赖于其他样本，适用小**mini batch**、动态网络和**RNN**，特别是**NLP**领域；可以**Online Learning**；
2. **LN**对同一个样本的所有特征进行相同的转换，如果不同输入特征含义不同（比如颜色和大小），那么**LN**的处理可能会降低模型的表达能力；
3. **LN**假设同一层的所有**channel**对结果具有相似的贡献，而**CNN**中每个通道提取不同模式的特征，因此**LN**不适用于**CNN**。

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
        dim, H, W = normalized_shape
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, H, W))
        self.b = nn.Parameter(torch.zeros(1, dim, H, W))

    def forward(self, x):
        mean = x.mean([1, 2, 3], keepdim = True)
        var = x.var([1, 2, 3], unbiased = False, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

torch.nn.LayerNorm(
    normalized_shape, eps=1e-05, # normalized_shape指定计算统计量的维度，如[C,H,W]
    elementwise_affine=True,     # 仿射参数默认作用于每个元素
    device=None, dtype=None
    )
```

## （4）实例归一化 Instance Normalization
- paper：[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)

**实例归一化（IN）**适用于生成模型（**GAN**），最初是在图像风格迁移任务中提出的。

在生成模型中，每一个样本实例之间是独立的，对**batch**维度计算统计量是不合适的；并且每个图像样本的每个通道之间通常也是独立的。**IN**计算每个样本在每个通道上的统计量，不仅可以加速模型收敛，并且可以保持每个实例及其通道之间的独立性。

记网络某一层的输入$$X=(x_{nchw}) \in \mathbb{R}^{N×C×H×W}$$，$N$为**batch**维度，$C$为通道维度，$H,W$为空间维度，则**IN**表示为：

$$
\begin{aligned}
μ_{nc}&= \frac{1}{HW} \sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}} \\
σ_{nc}^2&= \frac{1}{HW} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {(x_{nchw}-μ_{nc})^2}}}\\
\hat{x}_{nchw}&= \frac{x_{nchw}-μ_{nc}}{\sqrt{σ_{nc}^2+ε}}\\
\end{aligned}
$$

**IN**通常不引入额外的仿射变换。**IN**应用于**CNN**时假设每个样本的每个通道是独立的，这可能会忽略部分通道之间的相关性。

```python
class InstanceNorm2d(nn.Module):
    def __init__(self, dim, affine=False, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        mean = x.mean([2, 3], keepdim = True)
        var = x.var([2, 3], unbiased = False, keepdim = True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        if affine:
            return x_norm * self.g + self.b
        else:
            return x_norm

torch.nn.InstanceNorm2d(
    num_features, eps=1e-05, # normalized_shape指定计算统计量的维度，如[H,W]
    momentum=0.1, affine=False,
    track_running_stats=False,
    device=None, dtype=None
    )
```

## （5）组归一化 Group Normalization
- paper：[Group Normalization](https://arxiv.org/abs/1803.08494)

**组归一化（GN）**是**LN**和**IN**的一般形式：**LN**认为所有通道对输出的贡献是相似的，对每个样本的所有通道一起计算统计量；**IN**认为每个通道是独立的，对每个样本的每个通道分别计算统计量。

**GN**将每个样本的通道分成若干组$G$（默认$G=32$），假设组内通道具有相关性、组间通道是独立的，在每组通道内计算统计量。当$G=1$时**GN**退化为**LN**，当$G=C$时**GN**退化为**IN**。

![](https://pic.imgdb.cn/item/64b8f2f41ddac507cc90eebf.jpg)

记网络某一层的输入$$X=(x_{nchw}) \in \mathbb{R}^{N×C×H×W}$$，$N$为**batch**维度，$C$为通道维度，将$C$分成$G$个组，$H,W$为空间维度，则**GN**表示为：

$$
\begin{aligned}
μ_{ng}&= \frac{1}{HWC/G}  \sum_{c \in g} \sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}} \\
σ_{ng}^2&= \frac{1}{HWC/G}  \sum_{c \in g} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {(x_{nchw}-μ_{ng})^2}}}\\
\hat{x}_{nchw}&= \frac{x_{nchw}-μ_{ng}}{\sqrt{σ_{ng}^2+ε}}\\
y_{nchw} &= γ \hat{x}_{nchw} + β
\end{aligned}
$$

作者通过实验发现**GN**相比于**BN**更容易优化，但损失了一定的正则化能力。**GN**对不同**batch size**具有很好的鲁棒性，尤其适合**batch size**较小的计算机视觉任务中（如目标检测，分割）。

```python
torch.nn.GroupNorm(
    num_groups, num_channels,
    eps=1e-05, affine=True,
    device=None, dtype=None
    )
```

## （6）切换归一化 Switchable Normalization
- paper：[Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/abs/1806.10779)

**BN**、**LN**、**IN**分别是**minibatch-wise**、**layer-wise**和**channel-wise**的归一化操作。**切换归一化（SN）**同时应用这三种方法，学习三种方法的权重，从而适应各种深度学习任务。

如下图所示，不同的深度学习任务具有不同的权重，代表不同归一化方法对不同任务的适合程度。
![](https://pic.downk.cc/item/5e7db2c7504f4bcb044fe556.png)

**SN**的实现：

$$ y_{nchw}=\frac{x_{nchw}-\sum_{k \in Ω}^{} {w_kμ_k}}{\sqrt{\sum_{k \in Ω}^{} {w'_kσ^2_k}+ε}}*γ+β $$

其中$Ω={in,ln,bn}$，注意到三种方法的统计量是相关的，可计算如下：

$$ μ_{in}=\frac{1}{HW}\sum_{h,w}^{H,W} {x_{nchw}}，  σ^2_{in}=\frac{1}{HW}\sum_{h,w}^{H,W} {(x_{nchw}-μ_{in})^2} \\ μ_{ln}=\frac{1}{C}\sum_{c=1}^{C} {μ_{in}}，  σ^2_{ln}=\frac{1}{C}\sum_{c=1}^{C} {(σ^2_{in}+μ^2_{in})}-μ^2_{ln} \\ μ_{bn}=\frac{1}{N}\sum_{n=1}^{N} {μ_{in}}，  σ^2_{bn}=\frac{1}{N}\sum_{n=1}^{N} {(σ^2_{in}+μ^2_{in})}-μ^2_{bn} $$

$w_k$和$w_k'$是三种方法对应的权重系数，用参数$λ_{in},λ_{ln},λ_{bn},λ_{in}',λ_{ln}',λ_{bn}'$控制：

$$ w_k=\frac{e^{λ_k}}{\sum_{z \in Ω}^{} {e^{λ_z}}},\quad  w'_k=\frac{e^{λ'_k}}{\sum_{z \in Ω}^{} {e^{λ'_z}}} $$


# 3. 改进特征归一化

## （1）改进Batch Norm

### ⚪ Synchronized-BatchNorm (SyncBN)

当使用`torch.nn.DataParallel`将代码运行在多张 **GPU** 卡上时，**PyTorch** 的 **BN** 层默认操作是各卡上数据独立地计算均值和标准差。**同步BN (SyncBatchNorm)**使用所有卡上的数据一起计算 **BN** 层的均值和标准差，缓解了当批量大小比较小时对均值和标准差估计不准的情况，是在目标检测等任务中一个有效的提升性能的技巧。

```python
torch.nn.SyncBatchNorm(
    num_features, eps=1e-05,
    momentum=0.1, affine=True,
    track_running_stats=True,
    process_group=None,
    device=None, dtype=None
    )
```

### ⚪ Batch Renormalization
- paper：[Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/abs/1702.03275)

**BN**假设每一**mini batch**和总体数据近似同分布，用**mini batch**的统计量作为总体统计量的估计。实际上**mini batch**和总体的分布存在偏差，**Batch Renormalization**用一个仿射变换修正这一偏差。

记总体均值为$μ$，方差为$σ^2$；某一**mini batch**计算的均值为$μ_B$，方差为$σ_B^2$，引入仿射变换：

$$ \frac{x-μ}{σ}=\frac{x-μ_B}{σ_B} r+d $$

可以得到一组变换参数为：

$$  r=\frac{σ_B}{σ} , d=\frac{μ_B-μ}{σ} $$

当$σ=E(σ_B),μ=E(μ_B)$时，有$E(r)=1,E(d)=0$，这便是**BN**的假设。注意$r$和$d$是与**mini batch**有关的常数，并不参与训练，并对上下限进行了裁剪：

$$
\begin{aligned}
r&=\text{Clip}_{[1/r_{max},r_{max}]}(\frac{σ_B}{σ})\\
d&=\text{Clip}_{[-d_{max},d_{max}]}(\frac{μ_B-μ}{σ})
\end{aligned}
$$

在实际使用时，先使用**BN**（设置$r=1,d=0$）训练得到一个相对稳定的滑动平均，作为总体均值$μ$和方差$σ^2$的近似，再逐渐放松约束。

### ⚪ Adaptive Batch Normalization (AdaBN)
- paper：[Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)

**Domain adaptation (transfer learning)**希望能够将在一个训练集上训练的模型应用到一个类似的测试集上。此时训练集和测试集的分布是不同的，应用**BN**时由训练集得到的统计量不再适合测试集。

**AdaBN**的思想是用所有测试集数据计算预训练网络每一层的**BN**统计量（均值和方差），测试时用这些统计量代替由训练得到的原**BN**统计量:

$$
\begin{aligned}
μ^l &= \frac{1}{N} \sum_{n=1}^{N} {x^l_{test,n}}\\
σ^l &= \sqrt{\frac{1}{N} \sum_{n=1}^{N} {(x^l_{test,n}-μ^l)^2}+ε}
\end{aligned}
$$

### ⚪ L1-Norm Batch Normalization (L1-Norm BN)
- paper：[L1-Norm Batch Normalization for Efficient Training of Deep Neural Networks](https://arxiv.org/abs/1802.09769)

**BN**中存在平方和开根号运算，增加了计算量，需要额外的内存，减慢训练的速度；部署到资源限制的硬件系统（如**FPGA**）时有困难。

**L1-norm BN**把**BN**运算中的**L2-norm variance**替换成**L1-norm variance**：

$$ σ_B= \frac{1}{N} \sum_{n=1}^{N} {\mid x_n-μ_B \mid} $$

可以证明，（在正态分布假设下）通过**L1-norm**计算得到的$σ'=E(\|X-E(X)\|)$和通过**L2-norm**计算得到的$σ$仅相差一常数：

$$ \frac{σ}{E(\mid X-E(X) \mid)}=\sqrt{\frac{\pi}{2}} $$

这个常数可以由**rescale**时的$γ$参数学习到，所以不显式地引入算法中。

### ⚪ Generalized Batch Normalization （Generalized BN）
- paper：[Generalized Batch Normalization: Towards Accelerating Deep Neural Networks](https://arxiv.org/abs/1812.03271)

**BN**使用的是均值和方差统计量，在**Generalized BN**中使用更一般的统计量$S$和$D$:

$$ \hat{x}_n= \frac{x_n-S(x_n)}{D(x_n)} $$

广义偏差测度(**Generalized deviation measures**)提供了选择$D$和相关统计量$S$的方法。


### ⚪ Spatially-Adaptive Denormalization (SPADE)
- paper：[<font color=Blue>Semantic Image Synthesis with Spatially-Adaptive Normalization</font>](https://0809zheng.github.io/2022/05/18/gaugan.html)

**SPADE (Spatially-adaptive denormalization)**采用的归一化形式为**BatchNorm**，即沿着特征的每一个通道维度进行归一化。仿射变换参数$\gamma,\beta$不是标量，而是与空间位置有关的向量$\gamma_{c,x,y},\beta_{c,x,y}$，并由输入语义**mask**图像通过两层卷积层构造。

![](https://pic.imgdb.cn/item/639a8b11b1fccdcd36d3c37d.jpg)

向网络中加入**SPADE**层的参考代码实现如下：

```python
#   SPADE module
class SPADE2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(SPADE2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None # [1, c, h, w]
        self.bias = None # [1, c, h, w]
        self.bn = nn.BatchNorm2d(
            self.num_features, eps=1e-5,
            momentum=0.1, affine=False,
            )

    def forward(self, x):
        # Apply batch norm
        out = self.bn(out)
        return out*self.weight + self.bias


#            Model
class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        # 定义包含SPADE的主体网络
        self.model = nn.Sequential()
        # 定义生成SPADE参数的网络
        num_spade_params = self.get_num_spade_params()
        self.conv = ConvLayer(input_channel, num_spade_params)

    def get_num_spade_params(self):
        """Return the number of SPADE parameters needed by the model"""
        num_spade_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "SPADE2d":
                num_spade_params += 2 * m.num_features
        return num_spade_params

    def assign_spade_params(self, spade_params):
        """Assign the spade_params to the SPADE layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "SPADE2d":
                # Extract weight and bias predictions
                m.weight = spade_params[:, : m.num_features, :, :].contiguous()
                m.bias = spade_params[:, m.num_features : 2 * m.num_features, :, :].contiguous()
                # Move pointer
                if spade_params.size(1) > 2*m.num_features:
                    spade_params = spade_params[:, 2*m.num_features:, :, :]

    def forward(self, main_input, cond_input):
        # Update SPADE parameters by ConvLayer prediction based off conditional input
        self.assign_spade_params(self.conv(cond_input))
        out = self.model(main_input)
        return out
```


## （2）改进Layer Norm

### ⚪ Root Mean Square Layer Normalization (RMSNorm)
- paper：[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

**RMS Norm**去掉了**LN**中的均值和**reshift**操作，相当于对每个样本进行了**L2**归一化，相比于**LN**减少了计算负担，并且具有相似的效果。


$$
\begin{aligned}
σ_n^2&= \frac{1}{D} \sum_{d=1}^{D} {x_{nd}^2} \\
\hat{x}_{nd}&= \frac{x_{nd}}{\sqrt{σ_n^2+\epsilon}} \\
y_{nd} &= γ \hat{x}_{nd}
\end{aligned}
$$

**center**操作（减均值或**reshift**操作）类似于全连接层的**bias**项，储存到的是关于预训练任务的一种先验分布信息；而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。

### ⚪ Pre-LN
- paper：[<font color=Blue>On Layer Normalization in the Transformer Architecture</font>](https://0809zheng.github.io/2020/11/26/preln.html)

**Post-LN**是指把**LayerNorm**放在自注意力+残差连接之后：

$$
x_{t+1} = \text{LayerNorm}(x_t + \text{SelfAttn}_t(x_t))
$$

而**Pre-LN**是指把**LayerNorm**放在自注意力之前：

$$
x_{t+1} = x_t + \text{SelfAttn}_t(\text{LayerNorm}(x_t))
$$

**Pre-LN**结构通常更容易训练，但最终效果一般比**Post-LN**差。

### ⚪ Mix-LN
- paper：[<font color=Blue>Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN</font>](https://0809zheng.github.io/2024/12/18/mixln.html)

**Mix-LN**在模型的早期层（前$aL$层）应用**Post-LN**，在深度层（后$(1-a)L$层）应用**Pre-LN**。

这样做的目的是利用**Post-LN**在深度层增强梯度流动的优势，同时利用**Pre-LN**在早期层稳定梯度的优势。通过这种方式，**Mix-LN**在中间和深度层实现了更健康的梯度范数，促进了整个网络的平衡训练，从而提高了模型的整体性能。

![](https://pic1.imgdb.cn/item/67e501590ba3d5a1d7e50f60.png)

### ⚪ LayerNorm Scaling
- paper：[<font color=Blue>The Curse of Depth in Large Language Models</font>](https://0809zheng.github.io/2025/02/09/cod.html)

**LayerNorm Scaling**通过按深度的平方根对**Layer Normalization**的输出进行缩放，有效控制了深度层输出方差的增长，确保了所有层都能有效地参与学习:

![](https://pic1.imgdb.cn/item/67e5104d0ba3d5a1d7e51533.png)

## （3）改进Instance Norm

### ⚪ Filter Response Normalization (FRN)
- paper：[Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)

**FRN**类似于**IN**，也是对每个样本的每个通道进行的操作。不同于**IN**，**FRN**使用二阶矩代替了方差统计量，即计算方差时没有考虑均值。


记网络某一层的输入$$X=(x_{nchw}) \in \mathbb{R}^{N×C×H×W}$$，$N$为**batch**维度，$C$为通道维度，$H,W$为空间维度，则**FRN**表示为：

$$
\begin{aligned}
μ_{nc}&= \frac{1}{HW} \sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}} \\
v^2&= \frac{1}{HW} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}^2}}}\\
\hat{x}_{nchw}&= \frac{x_{nchw}-μ_{nc}}{\sqrt{v^2+ε}}\\
y_{nchw} &= γ \hat{x}_{nchw} + β
\end{aligned}
$$

### ⚪ Adaptive Instance Normalization (AdaIN)
- paper：[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

本文作者指出，**IN**通过将特征统计量标准化来实现图像风格的标准化，即**IN**的仿射参数$\gamma,\beta$设置不同的值可以将特征统计信息标准化到不同的分布，从而将输出图像转换到不同的风格。**AdaIN**可以实现从内容图像$c$到风格图像$s$的风格迁移：

![](https://pic.imgdb.cn/item/64bddde81ddac507cc2bce31.jpg)

向网络中加入**AdaIN**层的参考代码实现如下：

```python
#   AdaIN module
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        # fixed init
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)
        out = F.batch_norm(
            x_reshaped, running_mean, running_var,
            None, None, True,
            self.momentum, self.eps
        )
        return out

#            Model
class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        # 定义包含AdaIN的主体网络
        self.model = nn.Sequential()
        # 定义生成AdaIN参数的网络
        num_adain_params = self.get_num_adain_params()
        self.conv = nn.Conv2d(input_channel, num_adain_params, 1)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract weight and bias predictions
                weight = adain_params[:, : m.num_features]
                bias = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = bias.contiguous().view(-1)
                m.weight = weight.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, main_input, cond_input):
        # Update AdaIN parameters by ConvLayer prediction based off conditional input
        self.assign_adain_params(self.conv(cond_input))
        out = self.model(main_input)
        return out
```


# 4. 深度学习中的参数归一化

之前介绍的归一化方法都是针对网络层中的特征进行的操作，也可以把归一化应用到网络权重上。

## （1）权重归一化 Weight Normalization
- paper：[Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868v1)

**权重归一化（WN）**对权重$W$使用长度标量$g$和方向向量$v$进行重参数化：

$$ W=g\frac{v}{\mid\mid v \mid\mid} $$

其中$g= \mid\mid W \mid\mid$，向量$v$由反向传播更新。

由于神经网络中权重经常是共享的，因此这种方法计算开销小于对特征进行归一化的方法，且不依赖于**mini batch**的统计量。

## （2）余弦归一化 Cosine Normalization
- paper：[Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks](https://arxiv.org/abs/1702.05870)

对数据进行归一化的原因是因为数据经过神经网络的计算后可能变得很大，导致分布的方差爆炸，而这一问题的根源就是采用的计算方式(点积)，向量点积是无界的。

向量点积是衡量两个向量相似度的方法之一。类似的度量方式还有很多。夹角余弦就是其中一个且有确定界。余弦归一化将点积运算替换为计算余弦相似度，将输出控制在$[-1,1]$之间。

$$
Norm(W\cdot X) = \frac{W·X}{\mid\mid W \mid\mid · \mid\mid X \mid\mid} $$

## （3）谱归一化 Spectral Normalization
- paper：[<font color=Blue>Spectral Normalization for Generative Adversarial Networks</font>](https://0809zheng.github.io/2022/02/08/sngan.html)


**谱归一化(Spectral Normalization)**是指使用**谱范数(spectral norm)**对网络参数进行归一化：

$$ W \leftarrow \frac{W}{||W||_2^2} $$

谱归一化精确地使网络满足[Lipschitz连续性](https://0809zheng.github.io/2022/10/11/lipschitz.html)。**Lipschitz**连续性保证了函数对于**输入扰动的稳定性**，即函数的输出变化相对输入变化是缓慢的。

谱范数是一种由向量范数诱导出来的矩阵范数，作用相当于向量的模长：

$$ ||W||_2 = \mathop{\max}_{x \neq 0} \frac{||Wx||}{||x||} $$

谱范数$\|\|W\|\|_2$的平方的取值为$W^TW$的最大特征值。

```python
model = Model()
def add_sn(m):
        for name, layer in m.named_children():
             m.add_module(name, add_sn(layer))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
             return nn.utils.spectral_norm(m)
        else:
             return m
model = add_sn(model)
```

值得一提的是，谱归一化是对模型的每一层权重都进行的操作，使得网络的每一层都满足**Lipschitz**约束；这种约束有时太过强硬，通常只希望整个模型满足**Lipschitz**约束，而不必强求每一层都满足。

# 5. 不使用归一化的方法

近些年有一些方法尝试不引入归一化策略来训练深度学习模型。

### ⚪ Fixup
- paper：[<font color=Blue>Fixup Initialization: Residual Learning Without Normalization</font>](https://0809zheng.github.io/2020/11/09/fixup.html)

在没有归一化的残差网络中，输出方差会随着深度呈指数增长，从而导致梯度爆炸。

**Fixup**的核心思想是通过重新调整残差分支的权重初始化，使得每个残差分支对网络输出的更新幅度与网络深度无关。具体步骤如下：
1. 初始化分类层和残差分支的最后一层权重为$0$：这有助于稳定训练初期的输出。
2. 对残差分支内的权重层进行重新缩放：具体来说，将残差分支内的权重层按 $L^{-\frac{1}{2(m-2)}}$ 缩放，其中 $L$ 是网络深度，$m$ 是残差分支内的层数。这种缩放方式可以确保每个残差分支对网络输出的更新幅度为 $Θ(η/L)$，从而使得整个网络的更新幅度为 $Θ(η)$。
3. 添加标量乘数和偏置：在每个残差分支中添加一个标量乘数（初始化为$1$），并在每个卷积层、线性层和激活层前添加一个标量偏置（初始化为$0$）。这些参数有助于进一步调整网络的表示能力。

![](https://pic1.imgdb.cn/item/67e5178f0ba3d5a1d7e51957.png)

### ⚪ SkipInit
- paper：[<font color=Blue>Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks</font>](https://0809zheng.github.io/2021/02/01/skipinit.html)

由于归一化操作，残差分支的输出方差被抑制到接近$1$，从而使得残差块的输出主要由跳跃连接决定，即网络函数接近恒等函数。这种特性确保了网络在初始化时具有良好的梯度传播，便于训练。

基于上述分析，作者提出了**SkipInit**初始化方法。该方法的核心思想是在每个残差分支的末尾引入一个可学习的标量乘数$α$，并在初始化时将其设置为$0$或一个较小的常数$1/\sqrt{d}$（$d$是残差块的数量）。这样在初始化时，残差分支的贡献被显著缩小，使得残差块的输出接近跳跃连接，从而实现了与批归一化类似的效果。

$$
x_{t+1} = x_t + \alpha \cdot F_t(x_t)
$$

![](https://pic1.imgdb.cn/item/67e51c7c0ba3d5a1d7e51d91.png)

### ⚪ ReZero
- paper：[<font color=Blue>ReZero is All You Need: Fast Convergence at Large Depth</font>](https://0809zheng.github.io/2021/03/13/rezero.html)

与**SkipInit**类似，**ReZero**通过在每个残差连接处引入一个初始化为零的可训练参数，实现了动态等距性，从而显著加速了深度网络的训练。

$$
x_{t+1} = x_t + \alpha_t \cdot F_t(x_t)
$$

动态等距性要求网络的输入-输出雅可比矩阵的所有奇异值接近1，即输入信号的所有扰动都能在网络中以相似的方式传播。**ReZero**通过将每个残差块的初始输出设置为输入本身，确保了在训练开始时网络的雅可比矩阵的奇异值为1。

![](https://pic1.imgdb.cn/item/67e520230ba3d5a1d7e5210e.png)


### ⚪ Dynamic Tanh（DyT）
- paper：[<font color=Blue>Transformers without Normalization</font>](https://0809zheng.github.io/2025/03/13/dyt.html)

作者发现，在训练好的**Transformer**模型中，归一化层的输入-输出映射呈现出类似**tanh**函数的**S**形曲线。这种映射不仅对输入激活值进行了缩放，还对极端值进行了压缩。

**DyT**的核心思想是通过一个可学习的标量参数$α$和**tanh**函数来动态调整输入激活值，以代替网络中的**LayerNorm**：

$$
\text{DyT}(x)=γ⋅\tanh(αx)+β
$$

![](https://pic1.imgdb.cn/item/67e532c10ba3d5a1d7e52758.png)

### ⚪ Dynamic Inverse Square Root Unit (DyISRU)
- paper：[<font color=Blue>The Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions</font>](https://0809zheng.github.io/2025/03/27/dyisru.html)

作者从梯度近似的角度设计了**RMSNorm**归一化的替代函数。**RMSNorm**的梯度可以表示为：

$$
\begin{aligned}
\nabla_{\mathbf{x}} \mathbf{y} &= \frac{\sqrt{d}}{||\mathbf{x}||}\left(I - \frac{\mathbf{y} \mathbf{y}^\top}{d} \right)
\end{aligned}
$$

寻找一个函数$\mathbf{y}=f(\mathbf{x})$近似**RMSNorm**的梯度，则$f$能够替代归一化层的使用，从而实现在网络中去掉归一化层的目标。假设$\mathbf{y}=f(\mathbf{x})$是逐元素操作，即$y_i=f(x_i)$，则$f$的梯度需满足：

$$
\frac{d y_i}{d x_i} = \sqrt{d} / ||\mathbf{x}|| \left( 1 - \frac{y_i^2}{d} \right)
$$

若假设$\rho=\sqrt{d} / \|\|\mathbf{x}\|\|$为常数，求解上述微分方程可得到**DyT**的形式：

$$
y_i = \sqrt{d} \tanh \left( \frac{x_i}{\rho \sqrt{d}} \right)
$$

直接求解原微分方程可得到：

$$
y_i = \frac{\sqrt{d} x_i}{\sqrt{x_i^2+C}} 
$$

