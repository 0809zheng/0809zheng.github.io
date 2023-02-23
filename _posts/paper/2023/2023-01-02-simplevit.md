---
layout: post
title: 'Better plain ViT baselines for ImageNet-1k'
date: 2023-01-02
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63f5d794f144a0100725133f.jpg'
tags: 论文阅读
---

> 在ImageNet-1k数据集上更好地训练视觉Transformer.

- paper：[Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580)

**ViT**模型如果不在更大的数据集上预训练，而只用**ImageNet1K**数据是表现较差的，通常需要设计较重的数据增强来训练。本文作者提出，只对原生**ViT**的训练策略做了少许修改就能大幅度提升**ViT**的性能：训练**90 epoch**时性能达到$76.5\%$，和**ResNet50**相当（$76.2\%$），而训练**300 epoch**，性能可以达到$80.0\%$，和**DeiT**性能相当（$79.8\%$）。

![](https://pic.imgdb.cn/item/63f5d944f144a01007277cb1.jpg)

本文选取**ViT-S/16**模型来进行实验，**ViT-S/16**的**patch**大小为$16\times 16$，而参数量（22M vs 25M）和**FLOPs**（4.6G vs 4.1G）和**ResNet50**基本相同。

在模型方面的改动：
1. 位置编码采用固定的**sincos2d**，而不是可学习的参数；
2. 最后的分类特征采用全局平均池化，而不是训练一个**class token**；
3. 分类头采用单层**linear**分类层，而不是**MLP**。
4. 移除了**Dropout**。

在训练策略方面的改动：
1. 采用的数据增强是**level=10**的**RandAug**和概率为$0.2$的**MixUP**；
2. 训练的**batch size**采用$1024$，而不是$4096$。

这些改动的消融实验如下：

![](https://pic.imgdb.cn/item/63f5db10f144a010072a78ba.jpg)

**SimpleViT**的完整实现可参考[vit-pytorch](https://github.com/lucidrains/vit-pytorch)。其中**sincos2d**位置编码实现如下：

$$
\left[
\sin \left(\frac{x}{10000^{\Omega}}\right),
\cos \left(\frac{x}{10000^{\Omega}}\right),
\sin \left(\frac{y}{10000^{\Omega}}\right),
\cos \left(\frac{y}{10000^{\Omega}}\right)
\right]
$$

```python
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)
```

**Transformer**与[<font color=Blue>ViT</font>](https://0809zheng.github.io/2020/12/30/vit.html)采用相同的**pre-norm**形式，并移除了**dropout**：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
```

**SimpleViT**的主体结构如下，相比于**ViT**移除了**CLS token**，分类时采用全局平均池化，并且使用线性层输出类别概率。

```python
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
```

实例化一个**SimpleViT**的例子如下：

```python
import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,        # Last dimension of output tensor after linear transformation
    depth = 6,         # Number of Transformer blocks
    heads = 16,        # Number of heads in Multi-head Attention layer
    mlp_dim = 2048     # Dimension of the MLP (FeedForward) layer
)

img = torch.randn(1, 3, 256, 256)
preds = v(img) # (1, 1000)
```