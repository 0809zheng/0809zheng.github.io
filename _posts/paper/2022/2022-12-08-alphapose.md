---
layout: post
title: 'AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time'
date: 2022-12-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/668f8ce0d9c307b7e9ee80d0.png'
tags: 论文阅读
---

> AlphaPose: 全身区域实时多人姿态估计与跟踪.

- paper：[AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time](https://arxiv.org/abs/2211.03375)

**AlphaPose** 针对人体姿态估计任务中的训练训练损失和、归一化方法和训练数据进行了改进。

## 1. 训练损失

在**Heatmap-based**方法中，对预测热图解码时是把模型输出的高斯概率分布图用**Argmax**得到最大相应点坐标。由于**Argmax**操作最的结果只能是整数，这就导致了经过下采样的特征图永远不可能得到输入图片尺度的坐标精度，因此产生了**量化误差(quantization error)**。

**IPR**把从预测热图中取最大值操作修改为取期望操作：关节被估计为热图中所有位置的积分，由它们的归一化概率加权。从而避开了 **Heatmap** 取 **Argmax** 只能得到整数坐标的问题。

$$
\hat{\mu} = \sum x \cdot p_x
$$

坐标通常采用**L1**损失进行监督：

$$
L_{reg} = ||\mu - \hat{\mu}||_1
$$

分析每一个热图位置的梯度：

$$
\frac{\partial L_{reg}}{\partial p_x} = x \cdot sign(\hat{\mu} - \mu)
$$

结果表明在反向传播时梯度形式是不对称的。**IPR** 在计算时会对每个像素乘上它自己的坐标值，因此坐标值较大的像素的梯度会有更大幅度的变动，这违背了模型预测平移不变性的假设，导致模型性能下降。

**AlphaPose** 提出了一种 **IPR** 的改进方案，称为 **Symmetric Integral Keypoints Regression (SIKR)**。**SIKR**直接对反向传播时的梯度进行修改：

$$
\frac{\partial L_{reg}}{\partial p_x} = A_{grad} \cdot sign(x-\hat{\mu}) \cdot sign(\hat{\mu} - \mu)
$$

其中 $A_{grad}$ 是一个人为设置的梯度幅度值。作者对修改前后梯度的 **Lipschitz** 常数进行分析，结果表明当$A_{grad}$取热图尺度的$1/8$时，修改后梯度的 **Lipschitz** 常数是前者的四分之一，这说明它的梯度空间更加平滑容易优化。

```python
class IngetralCoordinate(torch.autograd.Function):
    ''' Symmetry integral regression function.
    '''
    AMPLITUDE = 2

    @staticmethod
    def forward(ctx, input):
        assert isinstance(
            input, torch.Tensor), 'IngetralCoordinate only takes input as torch.Tensor'
        input_size = input.size()
        weight = torch.arange(
            input_size[-1], dtype=input.dtype, layout=input.layout, device=input.device)
        ctx.input_size = input_size
        output = input.mul(weight)
        ctx.save_for_backward(weight, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, output = ctx.saved_tensors
        output_coord = output.sum(dim=2, keepdim=True)
        weight = weight[None, None, :].repeat(
            output_coord.shape[0], output_coord.shape[1], 1)
        weight_mask = torch.ones(weight.shape, dtype=grad_output.dtype,
                                 layout=grad_output.layout, device=grad_output.device)
        weight_mask[weight < output_coord] = -1
        weight_mask[output_coord.repeat(
            1, 1, weight.shape[-1]) > ctx.input_size[-1]] = 1
        weight_mask *= IngetralCoordinate.AMPLITUDE
        return grad_output.mul(weight_mask)
```

## 2. 归一化方法

在 **Heatmap-based** 方法中，关键点的置信度取值是对 **Heatmap** 进行 **Softmax** 归一化后，直接取最大响应值点的分数。由于人体不同部位在图片上的尺度是不同的，较大尺度的部位在特征图上的响应区域也自然而然会更大，因此在进行 **Softmax** 归一化后，强制总概率之和为1，就会导致大尺度关键点的最大响应值偏小。

**AlphaPose** 针对 **Soft-Argmax** 的特性提出了一种新的归一化方式，改善了关键点置信度预测的质量。改进方案是将归一化变成二阶段的，在第一阶段得到大小稳定的分数作为关键点置信度，在第二阶段才完成归一化。

具体做法是先对特征图取 **Sigmoid**，使得每个像素上的响应值在0-1之间，满足置信度的要求，又不需要受总和为 1 的约束。而后续的 **IPR** 要求输入的特征图是离散概率图，因此再对 **Sigmoid** 后的结果进行归一化，使之满足总和为 1 。

![](https://pic.imgdb.cn/item/668fa065d9c307b7e90bfca9.png)

## 3. 训练数据

**AlphaPose**引入了多个公开数据集进行联合训练。除了作者团队自己提出的 **Halpe FullBody** 数据集和常用的 **COCO-WholeBody**，另外引入了三个公开数据集：人脸数据集 **300Wface**、人手数据集 **FreiHand** 和 **InterHand**。

这些数据集的数据量和坐标分布是不一致的。针对这些问题，**AlphaPose** 设置了不同数据集的采样比例，并提出了一个 **Part-Guided Proposal Generator (PGPG)**，根据不同部位的位置分布来生成训练样本。

本文作者人为指定一个数据 **batch** 中，1/3 的数据来自 **Halpe-FullBody**， 1/3 来自 **COCO-WholeBody**，剩下部分从 **300Wface/FreiHand/Interhand** 等比例采样。**PGPG** 则是统计 **Halpe-FullBody** 中各个部位的坐标分布，当训练样本来自局部数据集时，将样本放到图中合适的位置上。

![](https://pic.imgdb.cn/item/668fa273d9c307b7e90f11a1.png)