---
layout: post
title: 'Bag of Tricks for Image Classification with Convolutional Neural Networks'
date: 2020-01-29
author: 郑之杰
cover: 'https://pic.downk.cc/item/5fa28c341cd1bbb86b21a21c.jpg'
tags: 论文阅读
---

> 一些使用卷积神经网络进行图像分类的技巧.

- paper：Bag of Tricks for Image Classification with Convolutional Neural Networks
- arXiv：[link](https://arxiv.org/abs/1812.01187)

这篇论文总结了图像分类任务中能够提高模型准确率的技巧。

# 1. Baseline
作者首先训练了一些**baseline**模型，作为准确率比较的基准。这些**baseline**模型训练的细节如下：

训练时的数据预处理：
1. 图像按照$32$位浮点数读入，像素值$0$~$255$;
2. 在$8\%$~$100\%$之间的图像范围内随机裁剪一个长宽比在$3/4$~$4/3%$之间的矩形，将其尺寸调整为$224 \times 224$；
3. 按照$0.5$的概率随机水平翻转；
4. 随机将色调、饱和度和亮度调节为$0.6$~$1.4$倍；
5. 随机加入**PCA**噪声$\mathcal{N}(0,0.01)$；
6. **RGB**通道减去均值$(123.68,116.779,103.939)$，除以方差$(58.393,57.12,57.375)$（均为**ImageNet**数据集的平均值）。

验证时的数据预处理：
1. 把图像的尺寸缩放到其短边为$256$；
2. 从图像中心裁剪$224 \times 224$；
3. **RGB**通道减去均值$(123.68,116.779,103.939)$，除以方差$(58.393,57.12,57.375)$；
4. 验证时不进行任何随机增强。

模型参数采用**Xavier**初始化，即随机地将卷积层和全连接层的权重参数设置为$\[- \sqrt{6/(d_{in}+d_{out})}, \sqrt{6/(d_{in}+d_{out})} \]$内的某一值，其中$d_{in}$和$d_{out}$是输入与输出的通道大小；偏置参数都设置为$0$。对于**BatchNorm**层，缩放因子$\gamma$设置为$1$，偏移因子$\beta$设置为$0$。

模型训练采用**Nesterov**加速梯度下降算法，模型在$8$块**Nvidia V100 GPU**上训练$120$轮，**batch size**设置为$256$。初始学习率为$0.1$，在第$30$,$60$和$90$轮除以$10$。

# 2. Training Tricks

## (1) Efficient Training
本节介绍一些允许在较低的数值精度和较大的**batch size**条件下兼顾训练精度和训练速度的技巧。

### a. Large-batch training
使用较大的**batch size**进行训练，会减缓训练过程，使模型收敛更慢。实践中发现对于相同的训练轮数，更大的**batch size**往往带来更低的模型准确率。下面是一些在模型训练中能够增大**batch size**的方法：
- **linear scaling learning rate**：更大的**batch size**能够减小随机梯度的方差，降低了噪声，因此可以增加学习率。如对于**batch size**$=256$时设置学习率为$0.1$；将**batch size**增大为$b$，则学习率调整为$0.1 \times b/256$。
- **learning rate warmup**：训练初期使用过大的学习率会导致数值不稳定，因此训练时可以设置几个**epoch**进行**warmup**，即将学习率从$0$线性增大到初始学习率。
- **zero** $ \gamma$：初始化时将所有残差块后面的**BatchNorm**的缩放因子$\gamma$设置为$0$，这样残差块输出为原始的输入，在前期更容易训练。
- **no bias decay**：权重衰减经常被用在权重参数和偏置参数上，等价于$L2$正则化。建议只对权重参数进行衰减，其余可学习的参数不进行衰减。

![](https://img.imgdb.cn/item/60497cde5aedab222cc377ba.jpg)

### b. Low-precision training
通常神经网络使用**float32**数值精度。一些新的硬件已经支持更低的数值精度，将数值精度从**float32**调整为**float16**后训练速度有明显的提高。但是降低精度可能会导致数值超过存储范围，从而干扰训练。因此可以用**float16**的数值计算梯度，使用**float32**的数值作为参数进行更新。也可以对损失函数乘以一个标量，将其数值对齐到**float16**精度。

![](https://img.imgdb.cn/item/60497cf75aedab222cc38430.jpg)

## (2) Model Tweaks
模型调整是指对网络结构进行微小的调整(如改变某一卷积层的步长)，这种调整几乎不会影响计算复杂度，但对模型精度会有显著的影响。本节在**ResNet**的基础上进行模型调整。

### a. ResNet Architecture

![](https://img.imgdb.cn/item/604981035aedab222cc57e0c.jpg)

典型的**ResNet**结构如上图所示，包括一个输入分支、四个中间分支和一个输出分支。输入分支使用通道数为$64$、步长为$2$的$7 \times 7$卷积，再使用步长为$2$的$3 \times 3$最大池化。中间分支首先使用一个下采样模块，再堆叠若干残差块。

### b. ResNet Tweaks

![](https://img.imgdb.cn/item/604abd205aedab222c7c300d.jpg)

上图是**ResNet**结构的一些改进。
- **ResNet-B**：调整了**ResNet**中的下采样模块。原模块中的路径$A$首先使用了步长为$2$的$1 \times 1$卷积，损失了大量信息；将其调整为步长为$1$，则保留了较多信息；中间的$3 \times 3$卷积步长调整为$3$，使得整体路径的输出尺寸不变。
- **ResNet-C**：将$7 \times 7$卷积拆分成三个$3 \times 3$卷积，从而降低模型复杂度。
- **ResNet-D**：在**ResNet-B**的基础上修改，原模块中的路径$B$使用了步长为$2$的$1 \times 1$卷积，也损失了大量信息。引入步长为$2$的$2 \times 2$平均池化，缓解了这一问题。

![](https://img.imgdb.cn/item/604abd3e5aedab222c7c3a86.jpg)

## (3) Training Refinements
本节介绍一些能够提高模型准确率的训练阶段改进方法。

### a. Cosine Learning Rate Decay
学习率的余弦衰减是指训练时按照余弦公式衰减学习率。忽略**warmup**阶段，假设共进行$T$轮训练，则第$t$轮的学习率表示为：

$$ \eta_t = \frac{1}{2} (1+cos(\frac{t \pi}{T})) \eta $$

相比于阶梯衰减学习率，余弦衰减在开始和结束时衰减较慢，中间接近线性衰减，提高了训练效率。

![](https://img.imgdb.cn/item/604abd5e5aedab222c7c45bc.jpg)

### b. Label Smoothing
图像分类的最后一层通常是全连接层，经过**softmax**函数计算每一类别$i$的概率$p_i$:

$$ p_i = \frac{exp(z_i)}{\sum_{j=1}^{K} exp(z_j)} $$

在训练集的标注中，正确类别$y$的概率标注为$q_y=1$，其余类别的概率标注$q_i=0$，模型最终的目标函数表示为负交叉熵损失：

$$ \mathcal{l}(p,q) = - \sum_{i=1}^{K} q_ilogp_i = -logp_y  \\ = -log\frac{exp(z_i)}{\sum_{j=1}^{K} exp(z_j)} = -z_y + log(\sum_{j=1}^{K} exp(z_j)) $$

由于真实标注倾向于正确类别$q_i=1$，会使正确类别的得分$z_y$趋近于$+∞$，可能导致过拟合。**标签平滑**改变了真实标注的概率：

$$ q_i = \begin{cases} 1- \epsilon \quad \text{if } i=y \\ \frac{\epsilon}{K-1} \quad \text{otherwise} \end{cases} $$

此使网络学习得到的类别得分会拟合下式：

$$ z_i^* = \begin{cases} log(\frac{(K-1)(1-\epsilon)}{\epsilon}) + \alpha \quad \text{if } i=y \\ \alpha \quad \text{otherwise} \end{cases} $$

其中$\alpha$是任意实数，这使得网络学习有限的输出，具有更好的泛化性。此使正确类别与错误类别之间的得分具有一个**gap**，下图是当$\epsilon$改变时**gap**的变化。

![](https://img.imgdb.cn/item/604abd875aedab222c7c57e7.jpg)

当$\epsilon=0$时，退化为一般的分类损失，此使网络需要学习的正确类别得分$z_y$趋近于$+∞$，**gap**也趋近于$+∞$。当$\epsilon$逐渐增大时，**gap**逐渐减小。当$\epsilon=\frac{K-1}{K}$时，标签中每个类别的概率都是$\frac{1}{2}$，**gap**为$0$。

### c. Knowledge Distillation
知识蒸馏是指用教师模型帮助训练学生模型。教师模型通常是准确率较高的预训练模型。实验中选用**ResNet-152**为教师模型，**ResNet-50**为学生模型。训练中采用蒸馏损失衡量学生模型的输出值$s$和教师模型的输出值$t$之间的差异，记真实值为$p$，则损失可以表示为：

$$ \mathcal{l}(p,softmax(s)) + T^2 \mathcal{l}(softmax(t/T),softmax(s/T)) $$

其中$T$是温度超参数，确保**softmax**函数的输出更平滑。

### d. Mixup Training
**mixup**数据增强方法是指随机选择两个样本$(x_i,y_i)$和$(x_j,y_j)$，根据线性插值随机地构造一个新的样本：

$$ \hat{x} = \lambda x_i + (1-\lambda) x_j $$

$$ \hat{y} = \lambda y_i + (1-\lambda) y_j $$

其中参数$\lambda$随机地从$\beta$分布中采样。训练时只用新构造的样本$(\hat{x},\hat{y})$。

![](https://img.imgdb.cn/item/604abdb75aedab222c7c71be.jpg)

# 3. Transfer Learning
迁移学习是指把训练好的分类网络应用到下游任务中。上述训练技巧对于目标检测、语义分割等下游视觉任务也有一定的提升。

![](https://img.imgdb.cn/item/604abdc75aedab222c7c769d.jpg)


