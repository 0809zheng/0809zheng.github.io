---
layout: post
title: '深度学习中的Normalization'
date: 2020-03-04
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e79ca179dbe9d88c5e2e0b0.png'
tags: 深度学习
---

> Normalization in Deep Learning.

1. Background
2. Normalization
3. Local Response Normalization
4. Batch Normalization
5. Batch Renormalization
6. Adaptive Batch Normalization
7. L1-Norm Batch Normalization
8. Generalized Batch Normalization
9. Layer Normalization
10. Instance Normalization
11. Group Normalization
12. Switchable Normalization
13. Filter Response Normalization
14. Weight Normalization
15. Cosine Normalization


# 1. Background
输入数据的特征通常具有不同的量纲、取值范围，使得不同特征的尺度（scale）差异很大。

不同机器学习模型对数据特征尺度的敏感程度不同。如果一个机器学习算法在数据特征缩放前后不影响其学习和预测，则称该算法具有尺度不变性（scale invariance）。理论上神经网络具有尺度不变性，但是输入特征的不同尺度会增加训练的困难：

(1) 参数初始化困难

当使用具有饱和区的激活函数$ a=f(WX) $时，若特征X的尺度不同，对参数W的初始化不合适容易使激活函数陷入饱和区，产生vanishing gradient现象。

(2) 梯度下降法的效率下降

如下图所示，左图是数据特征尺度不同的损失函数等高线，右图是数据特征尺度相同的损失函数等高线。由图可以看出，前者计算得到的梯度方向并不是最优的方向，需要迭代很多次才能收敛；后者的梯度方向近似于最优方向，大大提高了训练效率。
![](https://pic.downk.cc/item/5e7d8443504f4bcb042a29d3.png)

# 2. Normalization
归一化（Normalization）泛指把数据特征转换为相同尺度的方法。

(1) Min-Max Normalization

最小-最大值归一化（Min-Max Normalization）将每个特征的取值范围归一到[0,1]之间：

记含有N个样本、每个样本含有P个特征的输入数据为$ X=(x_{np}) \in {\scr R}^{N×P} $，n=1,2,...,N为sample axis，p=1,2,...,P为feature axis，

$$ \hat{x}_{np}= \frac{x_{np}-min_p(x_{np})}{max_p(x_{np})-min_p(x_{np})} $$

(2) Standardization

标准化（Standardization）又叫Z值归一化（Z-Score Normalization），将每个特征调整为均值为0，方差为1：

记含有N个样本、每个样本含有P个特征的输入数据为$ X=(x_{np}) \in {\scr R}^{N×P} $，n=1,2,...,N为sample axis，p=1,2,...,P为feature axis，

$$ μ_p= \frac{1}{N} \sum_{n=1}^{N} {x_{np}} $$

$$ σ_p^2= \frac{1}{N} \sum_{n=1}^{N} {(x_{np}-μ_p)^2} $$

$$ \hat{x}_{np}= \frac{x_{np}-μ_p}{σ_p} $$

(3) Whitening

白化（Whitening）在调整特征取值范围的基础上消除了不同特征之间的相关性，降低输入数据特征的冗余。
![](https://pic.downk.cc/item/5e7d917a504f4bcb04345594.png)
具体地，将输入数据在特征方向上被特征值相除，使数据独立同分布(i.i.d.)，实现输入数据的zero mean、unit variance、decorrelated。

实现步骤：
1. 零均值：
$ \hat{X}=X-E(X) $
2. 计算协方差：
$ Cov(X)=E(XX^T)-EX(EX)^T $
3. Whiten：
$ Cov(X)^ {-\frac{1}{2}} \hat{X} $

Whiten对所有特征一视同仁，可能会放大不重要的特征和噪声；此外，对于深度学习，隐藏层使用Whiten时反向传播困难。

(4) Layer-wise Normalizaiton

逐层归一化（Layer-wise Normalizaiton）是将Normalizaiton应用于深度神经网络中，对神经网络隐藏层的输入进行归一化，从而提高训练效率。

逐层归一化的优点：

i). 更好的尺度不变性

使用梯度下降更新参数时，神经网络每一层的输入分布发生变化，导致后续层的输出分布变化，这种现象叫做**Internal Covariance Shift**。

一种对ICS形象的解释如下图，当前后的要求不同时，可能会影响结果。

![](https://pic.downk.cc/item/5ea14edbc2a9a83be5cea36c.jpg)

通过对每一层的输入进行归一化，不论低层的参数如何变化，高层的输入保持相对稳定，网络具有更好的尺度不变性，可以更高效地进行参数初始化和超参数选择。

ii). 更平滑的损失函数

可以[使神经网络的损失函数更平滑](https://arxiv.org/abs/1806.02375v1)，使梯度变得更稳定，可以使用更大的学习率，提高收敛速度。

iii). 隐形的正则化方法

可以提高网络的泛化能力，[避免过拟合](https://arxiv.org/abs/1809.00846)。

# 3. Local Response Normalization
- paper:ImageNet Classification with Deep Convolutional Neural Networks
- pdf:[http://stanford.edu/class/cs231m/references/alexnet.pdf](http://stanford.edu/class/cs231m/references/alexnet.pdf)

Local Response Normalization(LRN)受生物学中“[侧抑制](https://baike.baidu.com/item/%E4%BE%A7%E6%8A%91%E5%88%B6/10397049?fr=aladdin)”的启发，即活跃的神经元对于相邻的神经元具有抑制的作用。

LRN通常应用在CNN中，且作用于激活函数之后，对邻近的特征映射（表现为邻近的通道）进行归一化：

假设一个卷积层的feature map为$ Y \in {\scr R}^{H×W×C} $，H和W是特征映射的高度和宽度，C为通道数，

$$ \hat{Y}^c = \frac{Y^c}{(k+α\sum_{j=max(1,p-\frac{1}{2})}^{min(P,p+\frac{1}{2})} {(Y^j)^2)^β}} $$

n为归一化考虑的局部特征窗口大小，超参数的取值：n=5，k=2，α=0.001，β=0.75。


# 4. Batch Normalization
- paper:Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- arXiv:[https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

Batch Normalization（BN）简化了Whiten操作，可以应用于神经网络每一个隐藏层的输入特征。

BN对Whiten的两个简化：

第一，独立的对每一个特征维度计算统计量（没有进行decorrelate），不考虑不同特征间的相关性；

第二，用mini batch的统计量作为总体统计量的估计（假设每一mini batch和总体数据近似同分布）。

## (1) BN的实现
（a） Vanilla NN

记网络某一层的输入$ X=(x_{np}) \in {\scr R}^{N×P} $，N为batch axis，P为该层特征数（神经元个数），

$$ μ_p= \frac{1}{N} \sum_{n=1}^{N} {x_{np}} $$

$$ σ_p^2= \frac{1}{N} \sum_{n=1}^{N} {(x_{np}-μ_p)^2} $$

$$ \hat{x}_{np}= \frac{x_{np}-μ_p}{σ_p} $$

$$ y_{np} = γ \hat{x}_{np} + β $$

说明：
1. 对每一个mini batch，计算每个特征维度的均值和方差，并对输入做标准化操作，其中ε保证了数值稳定性；
2. 当使用具有饱和性质的激活函数（如Sigmoid）时，第一步的标准化会将几乎所有数据映射到激活函数的非饱和区（线性区），从而降低了神经网络的表达能力。为了保证模型的表达能力不因标准化而下降，引入可学习的scale和shift操作γ、β，β可以代替affine层的bias；特别地，当$ γ=\sqrt{σ_p^2+ε} $，$ β=μ_p $时，相当于identity transformation；
3. BN一般应用在仿射变换后、激活函数前，此时仿射变换不再需要bias（$ f(BN(WX+b))=f(BN(WX)) $）；
4. 测试时，使用总体均值和方差的无偏估计（有时也用训练时均值和方差的滑动平均值）。

（b） CNN

记网络某一层的输入$ X=(x_{nchw}) \in {\scr R}^{N×C×H×W} $，N为batch axis，C为channel axis，H、W为the spatial height and width axes。

$$ μ_c= \frac{1}{NHW} \sum_{n=1}^{N} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}}} $$

$$ σ_c^2= \frac{1}{NHW} \sum_{n=1}^{N} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {(x_{nchw}-μ_c)^2}}} $$

$$ \hat{x}_{nchw}= \frac{x_{nchw}-μ_c}{\sqrt{σ_c^2+ε}} $$

$$ y_{nchw} = γ \hat{x}_{nchw} + β $$

## (2) BN的作用
1. 调整每一层输入的分布，减缓了vanishing gradient，可以使用更大的学习率;
2. BN具有权重伸缩不变性，减少对参数初始化的敏感程度: $ BN((λW)X) = BN(WX) $
3. 分布差距较小的mini batch可以看作为模型训练引入了噪声，可以增加模型的鲁棒性，带有正则化效果。

值得一提的是近期的论文表明BN主要解决的是对目标函数空间增加了平滑约束，从而可以利用更大的学习率获得更好的局部最优解。

## (3) BN的适用场合
1. BN适用于mini batch比较大、数据分布比较接近的场合。在进行训练之前，要做好充分的shuffle；
2. BN在运行过程中需要计算每个mini batch的统计量，因此不适用于动态的网络结构和RNN网络，也不适合Online Learning（batchsize = 1）。


# 5. Batch Renormalization
- paper:Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
- arXiv:[https://arxiv.org/abs/1702.03275](https://arxiv.org/abs/1702.03275)

BN假设每一mini batch和总体数据近似同分布，用mini batch的统计量作为总体统计量的估计。实际上mini batch和总体的分布存在偏差，用一个仿射变换修正这一偏差：

记总体均值为$ μ $，方差为$ σ^2 $；某一mini batch计算的均值为$ μ_B $，方差为$ σ_B^2 $，引入仿射变换：

$$ \frac{x-μ}{σ}=\frac{x-μ_B}{σ_B} r+d ，\quad where \quad r=\frac{σ_B}{σ} , d=\frac{μ_B-μ}{σ} $$

当$ σ=E(σ_B) $、$ μ=E(μ_B) $时，有$ E(r)=1 $、$ E(d)=0 $，这便是BN的假设。

注意r和d是与mini batch有关的常数，并不参与训练，并对上下限进行了裁剪：

$$ r=Clip_{1/r_{max},r_{max}}(\frac{σ_B}{σ}) $$

$$ d=Clip_{-d_{max},d_{max}}(\frac{μ_B-μ}{σ}) $$

在实际使用时，先使用BN（设置r=1，b=0）训练得到一个相对稳定的滑动平均，再逐渐放松约束。

Batch Renormalization(BR)的实现：

（1）Training

$$ μ_B= \frac{1}{N} \sum_{n=1}^{N} {x_n} $$

$$ σ_B= \sqrt{\frac{1}{N} \sum_{n=1}^{N} {(x_n-μ_B)^2}+ε} $$

$$ r=Clip_{1/r_{max},r_{max}}(\frac{σ_B}{σ}) $$

$$ d=Clip_{-d_{max},d_{max}}(\frac{μ_B-μ}{σ}) $$

$$ \hat{x}_n= \frac{x_n-μ_B}{σ_B} r+d $$

$$ y_n = γ \hat{x}_n + β $$

$$ μ=(1-α)μ+αμ_B $$

$$ σ=(1-α)σ+ασ_B $$

（2）Inference

$$ y = γ \frac{x-μ}{σ} + β $$

# 6. Adaptive Batch Normalization
- paper:Revisiting Batch Normalization For Practical Domain Adaptation
- arXiv:[https://arxiv.org/abs/1603.04779](https://arxiv.org/abs/1603.04779)

Domain adaptation(transfer learning)希望能够将在一个训练集上训练的模型应用到一个类似的测试集上。此时训练集和测试集的分布是不同的，应用BN时由训练集得到的统计量不再适合测试集。

Adaptive Batch Normalization（AdaBN）的思想是用所有测试集计算预训练网络每一层的BN统计量（均值和方差），测试时用这些统计量代替原BN统计量（由训练得到）:

$$ μ^l = \frac{1}{N} \sum_{n=1}^{N} {x^l_{test,n}} $$

$$ σ^l = \sqrt{\frac{1}{N} \sum_{n=1}^{N} {(x^l_{test,n}-μ^l)^2}+ε} $$

# 7. L1-Norm Batch Normalization
- paper:L1-Norm Batch Normalization for Efficient Training of Deep Neural Networks
- arXiv:[https://arxiv.org/abs/1802.09769](https://arxiv.org/abs/1802.09769)

BN中存在square和root运算，增加了计算量，需要额外的内存，减慢训练的速度；部署到资源限制的硬件系统（如FPGA）时有困难。

L1-norm BN把BN运算中的L2-norm variance替换成L1-norm variance：

$$ σ_B= \frac{1}{N} \sum_{n=1}^{N} {\mid x_n-μ_B \mid} $$

可以证明，（在正态分布假设下）通过L1-norm计算得到的σ和通过L2-norm计算得到的σ仅相差一常数：

$$ \frac{σ}{E(\mid X-E(X) \mid)}=\sqrt{\frac{\pi}{2}} $$

这个常数可以由rescale时的γ学习到，所以不显式地引入算法中。

L1-norm BN把BN中的square和root操作换成对硬件更友好的absolute和signum操作，使计算更高效。

# 8. Generalized Batch Normalization
- paper:Generalized Batch Normalization: Towards Accelerating Deep Neural Networks
- arXiv:[https://arxiv.org/abs/1812.03271](https://arxiv.org/abs/1812.03271)

BN使用的是均值和方差统计量，在Generalized BN中使用更一般的统计量S和D:

$$ \hat{x}_n= \frac{x_n-S(x_n)}{D(x_n)} $$

Generalized deviation measures提供了选择D和相关统计量S的方法。

# 9. Layer Normalization
- paper:Layer Normalizaiton
- arXiv:[https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

Layer Normalizaiton（LN）适用于sequence model（RNN,LSTM），最初提出是用来解决BN无法应用在RNN网络的问题。

BN针对每一mini batch计算统计量，而在RNN网络中，每一个样本的统计量都是不同的（每一个句子的长度不固定）。

LN针对每一个训练样本计算统计量，即计算每个样本所有特征的均值和方差，此时每个样本的统计量是标量。

## (1) LN的实现
记X网络某一层的输入$ X=(x_{np}) \in {\scr R}^{N×P} $，N为batch axis，P为该层特征数（神经元个数），

$$ μ_n = \frac{1}{P} \sum_{p=1}^{P} {x_{np}} $$

$$ {σ_n}^2 = \frac{1}{P} \sum_{p=1}^{P} {(x_{np}-μ_n)^2} $$

$$ \hat{x}_n = \frac{x_n-μ_n}{\sqrt{ {σ_n}^2 +ε}} $$

$$ y_n = γ \hat{x}_n + β $$

说明：
1. LN也包含re-center和re-scale操作，参数可学习；
2. LN 不需要保存 mini batch 的均值和方差，节省了额外的存储空间。

（a） RNN

$$ a^t=W_{hh}h^{t-1}+W_{xh}x_t $$

$$ h^t=LN(a^t) $$

（b） LSTM

$$ \begin{pmatrix} i^t \\ f^t \\ o^t \\ g^t \\ \end{pmatrix} = \begin{pmatrix} sigmoid \\ sigmoid \\ sigmoid \\ tanh \\ \end{pmatrix} (LN(W_hh_{t-1})+LN(W_xx_{t})+b) $$

$$ c_t=f_t \bigodot c_{t-1}+i_t \bigodot g_t $$

$$ h_t=o_t \bigodot tanh(LN(c_t)) $$

## (2) LN的适用场合
1. LN针对单个训练样本进行，不依赖于其他样本，适用小mini batch、动态网络和RNN，特别是NLP领域；可以Online Learning；
2. LN对同一个样本的所有特征进行相同的转换，如果不同输入特征含义不同（比如颜色和大小），那么LN的处理可能会降低模型的表达能力；
3. LN假设同一层的所有channel对结果具有相似的贡献，而CNN中每一个神经元的receptive field不同，因此LN不适用于CNN。


# 10. Instance Normalization
- paper:Instance Normalization: The Missing Ingredient for Fast Stylization
- arXiv:[https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022)

Instance Normalization（IN）适用于generative model（GAN），最初是在图像风格迁移任务中提出的。

在生成模型中，每一个样本实例之间是独立的，对mini batch计算统计量是不合适的；作者进一步假设每个样本的每个通道是独立的。

IN计算每个样本在每个通道上的统计量，不仅可以加速模型收敛，并且可以保持每个实例及其通道之间的独立性。

记网络某一层的输入$ X=(x_{nchw}) \in {\scr R}^{N×C×H×W} $，N为batch axis，C为channel axis，H、W为the spatial height and width axes。

$$ μ_{nc}= \frac{1}{HW} \sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}} $$

$$ σ_{nc}^2= \frac{1}{HW} \sum_{h=1}^{H} {\sum_{w=1}^{W} {(x_{nchw}-μ_{nc})^2}} $$

$$ \hat{x}_{nchw}= \frac{x_{nchw}-μ_{nc}}{\sqrt{σ_{nc}^2+ε}} $$

$$ y_{nchw} = γ \hat{x}_{nchw} + β $$

IN应用于CNN时假设每个样本的每个通道是独立的，可能会忽略部分通道之间的相关性。


# 11. Group Normalization
- paper:Group Normalization
- arXiv:[https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)

Group Normalization（GN）是LN和IN的一般形式。

LN认为所有通道对输出的贡献是相似的，对每个样本的所有通道一起计算统计量；IN认为每个通道是独立的，对每个样本的每个通道分别计算统计量；

GN将每个样本的通道分成若干组G（G=32 by default），假设组内通道具有相关性、组间通道是独立的，在每组通道内计算统计量。当G=1时GN退化为LN，当G=C时GN退化为IN。

记网络某一层的输入$ X=(x_{nchw}) \in {\scr R}^{N×C×H×W} $，N为batch axis，C为channel axis，H、W为the spatial height and width axes，将C分成G个组。

$$ μ_{ng}= \frac{1}{HWC/G} \sum_{c \in g}^{} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {x_{nchw}}}} $$

$$ σ_{ng}^2= \frac{1}{HWC/G} \sum_{c \in g}^{} {\sum_{h=1}^{H} {\sum_{w=1}^{W} {(x_{nchw}-μ_{nc})^2}}} $$

$$ \hat{x}_{nghw}= \frac{x_{nghw}-μ_{ng}}{\sqrt{σ_{ng}^2+ε}} $$

$$ y_{nghw} = γ \hat{x}_{nghw} + β $$

作者通过实验发现GN相比于BN更容易优化，但损失了一定的正则化能力。GN对不同batch size具有很好的鲁棒性，尤其适合batch size较小的计算机视觉任务中（如目标检测，分割）。


# 12. Switchable Normalization
- paper:Differentiable Learning-to-Normalize via Switchable Normalization
- arXiv:[https://arxiv.org/abs/1806.10779](https://arxiv.org/abs/1806.10779)

BN、LN、IN分别是minibatch-wise、layer-wise和channel-wise的操作。Switchable Normalization（SN）同时应用这三种方法，学习三种方法的权重，从而适应各种深度学习任务。

如下图所示，不同的任务具有不同的权重，代表不同归一化方法对不同任务的适合程度。
![https://pic.downk.cc/item/5e7db2c7504f4bcb044fe556.png](https://pic.downk.cc/item/5e7db2c7504f4bcb044fe556.png)

SN的实现：

$$ y_{nchw}=γ\frac{x_{nchw}-\sum_{k \in Ω}^{} {w_kμ_k}}{\sqrt{\sum_{k \in Ω}^{} {w'_kσ^2_k}+ε}}+β $$

其中Ω={in,ln,bn}，注意到三种方法的统计量是相关的，可计算如下：

$$ μ_{in}=\frac{1}{HW}\sum_{i,j}^{H,W} {x_{nchw}}，  σ^2_{in}=\frac{1}{HW}\sum_{i,j}^{H,W} {(x_{nchw}-μ_{in})^2} $$

$$ μ_{ln}=\frac{1}{C}\sum_{c=1}^{C} {μ_{in}}，  σ^2_{ln}=\frac{1}{C}\sum_{c=1}^{C} {(σ^2_{in}+μ^2_{in})}-μ^2_{ln} $$

$$ μ_{bn}=\frac{1}{N}\sum_{n=1}^{N} {μ_{in}}，  σ^2_{bn}=\frac{1}{N}\sum_{n=1}^{N} {(σ^2_{in}+μ^2_{in})}-μ^2_{bn} $$

$ w_k $和$ w_k $'是三种方法对应的权重系数，用参数$ λ_{in} $,$ λ_{ln} $,$ λ_{bn} $,$ λ_{in} $',$ λ_{ln} $',$ λ_{bn} $'控制：

$$ w_k=\frac{e^{λ_k}}{\sum_{z \in Ω}^{} {e^{λ_z}}}\quad and\quad k \in Ω $$

$$ w'_k=\frac{e^{λ'_k}}{\sum_{z \in Ω}^{} {e^{λ'_z}}}\quad and\quad k \in Ω $$

# 13. Filter Response Normalization
- paper:Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks
- arXiv:[https://arxiv.org/abs/1911.09737](https://arxiv.org/abs/1911.09737)

Filter Response Normalization(FRN)类似于IN，也是对每个样本的每个通道进行的操作。不同于IN，FRN没有计算均值。

记网络某一层的输入$ X=(x_{nchw}) \in {\scr R}^{N×C×H×W} $，N为batch axis，C为channel axis，H、W为the spatial height and width axes。

$$ v^2= \frac{1}{HW} \sum_{h=1}^{H} {\sum_{w=1}^{W} {x^2_{nchw}}} $$

$$ \hat{x}_{nchw}= \frac{x_{nchw}-μ_{nc}}{\sqrt{v^2+ε}} $$

$$ y_{nchw} = γ \hat{x}_{nchw} + β $$

FRN用TLU（Thresholded Linear Unit）代替了激活函数ReLU：

$$ z = max(y, \tau) $$

其中τ是可学习的参数。

# 14. Weight Normalization
- paper:Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
- arXiv:[https://arxiv.org/abs/1602.07868v1](https://arxiv.org/abs/1602.07868v1)

之前的归一化方法是针对特征的操作，Weight Normalization（WN）是针对权重的操作。

WN对权重W进行重参数化（reparameterization），引入标量g和向量v：

$$ W=g\frac{v}{\mid\mid v \mid\mid} $$

注意到$ \mid\mid w \mid\mid = g $，W的方向为$ \frac{v}{\mid\mid v \mid\mid} $，即把权重W独立的分解成长度和方向。

由于神经网络中权重经常是共享的，因此这种方法计算开销小于对特征进行归一化的方法，且不依赖于mini batch的统计量。

# 15. Cosine Normalization
- paper:Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks
- arXiv:[https://arxiv.org/abs/1702.05870](https://arxiv.org/abs/1702.05870)

对于深度网络隐藏层的输入W·X，BN等方法实现了对X的归一化，WN等方法实现了对W的归一化。

注意到这里存在点积（dot product）运算，而点积运算是无界的；Cosine Normalization（CN）将点积运算替换为计算余弦相似度，将输出控制在[-1,1]之间。

$$ \frac{W·X}{\mid\mid W \mid\mid · \mid\mid X \mid\mid} $$
