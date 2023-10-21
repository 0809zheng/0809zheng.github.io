---
layout: post
title: 'VarifocalNet: An IoU-aware Dense Object Detector'
date: 2021-05-25
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6531d9b4c458853aef65a297.jpg'
tags: 论文阅读
---

> VarifocalNet：交并比感知的密集目标检测器.

- paper：[VarifocalNet: An IoU-aware Dense Object Detector](https://arxiv.org/abs/2008.13367)

本文指出目前目标检测最大瓶颈依然是分类分支和回归分支分值不一致问题，为此作者提出了两个改进：
- 提出了正负样本不对称加权的 **Varifocal Loss**
- 提出星型**bbox**特征提取**refine**网络，对输出初始**bbox**进行**refine**

作者对[<font color=blue>ATSS</font>](https://0809zheng.github.io/2021/05/23/atss.html)的推理过程进行了详细分析：
- 在没有**centerness**分支时**map**是**38.5**，训练时候加入**centerness**即**w/ctr**(标准结构)从而得到**39.2**。
- **gt_ctr**是指在**baseline**基础上，测试时把**centerness**分支替换为对应**label**值，可以发现**map**仅仅提高到**41.1**。说明**centerness**的作用其实非常不明显，引入这个额外分支无法完全解决分类和回归分支不一致性问题。
- 如果将**centerness**分支的输出变成预测**bbox**和**gt bbox**的**iou**值(**gt_ctr_iou**)，此时**map**是**43.5**。
- 把预测**bbox**回归分支值全部替换为真实**gt bbox**，**map**没有很大改变。说明影响**mAP**性能的主要因素不是**bbox**预测不准确，**centerness**的作用非常微弱。
- 将分类分支的输出对应真实类别设置为**1**，也就是类别预测完全正确(**gt_cls**)，此时**map**是**43.1**，加入**centerness**后提升到**58.1**，说明在类别预测完全正确的情况下，**centerness**可以在一定程度上区分准确和不准确的边界框。
- 如果把分类分值的输出中对应真实类别设置为预测**bbox**和真实**bbox**的**iou**(**gt_cls_iou**)，即使不用**centernss**也可以达到**74.7**，说明目前的目标检测**bbox**分支输出的密集**bbox**中存在非常精确的预测框，关键是没有好的预测分值来选择出来。

![](https://pic.imgdb.cn/item/6531e4dcc458853aef7e93d0.jpg)

通过上述分析，可以总结下：
- **centernss**作用还不如**iou**分支
- 单独引入一条**centernss**或者**iou**分支，作用非常有限
- 目前目标检测性能瓶颈不在于bbox预测不准确，而在于没有一致性极强的分值排序策略选择出对应**bbox**
- 将**iou**感知功能压缩到分类分支中是最合适的，理论**mAP**上限最高。

## 1. varifocal Loss

广义**focal loss**将**focal loss**只能支持离散**label**的限制推广到了连续**label**，并且强制将分类分支对应类别处的预测值变成了**bbox**预测准确度。

$$
QFL(p) = -|y-p|^\beta \left( y\log p + (1-y) \log(1-p) \right)
$$

其中$y$为$0$~$1$的质量标签，来自预测的**bbox**和**gt bbox**的**iou**值，注意如果是负样本，则$y$直接等于$0$；$p$是分类分支经过**sigmoid**后的预测值。

**focal loss**和**Generalized Focal Loss**都是对称的。**Varifocal Loss**主要改进是提出了非对称的加权操作，在正样本中也存在不等权问题，突出正样本的主样本:

$$
VFL(p) = 
\begin{cases}
-y \left( y\log p + (1-y) \log(1-p) \right) & y > 0 \\
-\alpha p^\gamma \log(1-p) & y = 0
\end{cases}
$$

正样本时没有采用**focal loss**，而是普通的**bce loss**，只不过多了一个自适应**iou**加权，用于突出主样本。而为负样本时是标准的**focal loss**。

```python
def varifocal_loss(pred, target, alpha=0.75, gamma=2.0, iou_weighted=True):
    """
        pred (torch.Tensor): 预测的分类分数，形状为 (B,N,C) , N 表示 anchor 数量， C 表示类别数
        target (torch.Tensor): 经过对齐度归一化后的 IoU 分数，形状为 (B,N,C)，数值范围为 0~1
        alpha (float, optional): 调节正负样本之间的平衡因子，默认 0.75.
        gamma (float, optional): 负样本 focal 权重因子， 默认 2.0.
        iou_weighted (bool, optional): 正样本是否用 IoU 加权
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        # 计算权重，正样本(target > 0)中权重为 target,
        # 负样本权重为 alpha*pred_simogid^2
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    # 计算二值交叉熵后乘以权重
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
```

## 2. bbox refinement

本文还提出**bbox refinement**进一步提高性能。其整个网络结构如下：

![](https://pic.imgdb.cn/item/65322a81c458853aef40525e.jpg)

**bbox refinement**的基本流程是：
1. 任何一个**head**，对分类和回归分支特征图堆叠一系列卷积，输出通道全部统一为$256$
2. 对回归分支特征图进行回归预测，得到初始**bbox**预测值，输出通道是$4$，代表**lrtb**预测值，然后对预测值进行还原得到原图尺度的**lrtb**值
3. 利用**lrtb**预测值对每个特征图上面点生成$9$个**offset**坐标：$(x, y), (x-l’, y), (x, y-t’), (x+r’, y), (x, y+b’), (x-l’, y-t’),(x+l’, y-t’), (x-l’, y+b’)$ 和 $(x+r’, y+b’)$
4. 将**offset**作为变形卷积的**offset**输入，然后进行可变形卷积操作，此时每个点的特征感受野就可以和初始时刻预测值重合，也就是常说的特征对齐操作，此时得到**refine**后的**bbox**预测输出值$\Delta l,\Delta r,\Delta t,\Delta b$，将该**refine**输出值和初始预测值相乘即可得到**refine**后的真实**bbox**值
5. 对分类分支也是同样处理，加强分类和回归分支一致性

![](https://pic.imgdb.cn/item/65322bc0c458853aef43eb1c.jpg)