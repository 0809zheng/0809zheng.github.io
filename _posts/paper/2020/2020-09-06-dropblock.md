---
layout: post
title: 'DropBlock: A regularization method for convolutional networks'
date: 2020-09-06
author: 郑之杰
cover: 'https://pic2.imgdb.cn/item/645b65d60d2dde57777f593b.jpg'
tags: 论文阅读
---

> DropBlock：一种卷积神经网络的正则化方法.

- paper：[DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)

**DropBlock**是一种用于**CNN**的正则化方法。普通的**DropOut**只是随机屏蔽掉一部分特征，而**DropBlock**是随机屏蔽掉一部分连续区域，如下图所示。图像是一个**2D**结构，像素或者特征点之间在空间上存在依赖关系，这样普通的**DropOut**在屏蔽语义就不够有效，但是**DropBlock**这样屏蔽连续区域块就能有效移除某些语义信息，比如狗的头，从而起到有效的正则化作用。**DropBlock**和**CutOut**有点类似，只不过**CutOut**是用于图像的一种数据增强方法，而**DropBlock**是用在**CNN**的特征上的一种正则化手段。

![](https://pic2.imgdb.cn/item/645b663d0d2dde57778035de.jpg)

**DropBlock**的原理很简单，它和**DropOut**的最大区别是就是屏蔽的地方是一个连续的方块区域，其伪代码如下所示：

![](https://pic2.imgdb.cn/item/645b66970d2dde577780ef12.jpg)

**DropBlock**有两个主要参数：**block_size**和$\gamma$，其中**block_size**为方块区域的边长，而$\gamma$控制被屏蔽的特征数量大小。对于**DropBlock**，首先要用参数为$\gamma$的伯努利分布生成一个**center mask**，这个**center mask**产生的是要屏蔽的**block**的中心点，然后将**mask**中的每个点扩展到**block_size**大小的方块区域，从而生成最终的**block mask**。

假定输入的特征大小为$(N,C,H,W)$，那么**center mask**的大小应该为$$(N,C,H-\text{block size}+1,W-\text{block size}+1)$$，而**block mask**的大小为$$(N,C,\text{block size},\text{block size})$$。

![](https://pic2.imgdb.cn/item/645b68ab0d2dde5777837943.jpg)

对于**DropBlock**，往往像**DropOut**那样直接设置一个**keep_prob**（或者**drop_prob**），这个概率值控制特征被屏蔽的量。此时需要将**keep_prob**转换为$\gamma$，两个参数带来的效果应该是等价的，所以有：

$$
(1- \text{keep prob}) \times \text{feat size}^2 = \gamma \times \text{block size}^2 \times (\text{feat size}-\text{block size}+1)^2
$$

其中**feat size**是特征图的大小，给定**keep_prob**后可以算出$\gamma$：

$$
\gamma = \frac{(1- \text{keep prob}) \times \text{feat size}^2}{\text{block size}^2 \times (\text{feat size}-\text{block size}+1)^2}
$$

**DropBlock**往往采用较大的**keep_prob**，如下图所示采用**0.9**的效果是最好的。另外，论文中发现对**keep_prob**采用一个线性递减的**scheduler**可以进一步增加效果：**keep_prob**从**1.0**线性递减到设定值如**0.9**。

![](https://pic2.imgdb.cn/item/645b6a1a0d2dde5777849ad4.jpg)


对于**block_size**，实验发现采用**block_size=7**效果是最好的，如下所示： ​

![](https://pic2.imgdb.cn/item/645b6a410d2dde577784bf2f.jpg)


在实现上，可以先对**center mask**进行**padding**，然后用一个**kernel_size**为**block_size**的**max pooling**来得到**block mask**。最后将特征乘以**block mask**即可，不过和**DropOut**类似，为了保证训练和测试的一致性，还需要对特征进行归一化：乘以**count(block mask)/count_ones(block mask)**。

```python
class DropBlock2d(nn.Module):
    """
    Args:
        p (float): Probability of an element to be dropped.
        block_size (int): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``
    """

    def __init__(self, p: float, block_size: int, inplace: bool = False) -> None:
        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        if not self.training:
            return input

        N, C, H, W = input.size()
        # compute the gamma of Bernoulli distribution
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size // 2)
        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})"
        return s
```