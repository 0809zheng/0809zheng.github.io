---
layout: post
title: 'Segmenter: Transformer for Semantic Segmentation'
date: 2023-01-17
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6416d034a682492fccba52da.jpg'
tags: 论文阅读
---

> Segmenter：为语义分割设计的视觉Transformer.

- paper：[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)

语义分割方法通常依赖于卷积编码器-解码器架构，其中编码器生成低分辨率图像特征，解码器对特征进行上采样，以逐像素地分割图像。最先进的方法依赖于可学习的堆叠卷积，可以捕获语义丰富的信息。然而卷积滤波器的局部特性限制了对图像中全局信息的访问。

本文将语义分割问题定义为序列到序列问题，介绍了一种用于语义分割的 **Transformer**模型：**Segmenter**。与基于卷积的方法相比，**Segmenter**允许在第一层和整个网络中建模全局上下文。**Segmenter**依赖于与图像 **patch** 对应的输出嵌入，并使用逐点线性解码器（**point-wise linear decoder**）或一个 **mask transformer** 解码器从这些嵌入中获得类标签。

![](https://pic.imgdb.cn/item/6416d374a682492fccbf3cd1.jpg)

**Segmenter**完全基于**Transformer**的编码器-解码器架构。它将一系列 **patch** 嵌入映射到像素级的类别**mask**。**patch** 序列由**Transformer**编码器编码，并由逐点线性映射或 **mask Transformer** 解码。模型采用逐像素交叉熵损失进行端到端训练；在推理时，对类别**mask**上采样后应用 **argmax** 获得每个像素的类别。

![](https://pic.imgdb.cn/item/6416d47da682492fccc0d56d.jpg)

### ⚪ 编码器

一个图像$x∈R^{H×W×C}$被分割成一个块序列$\mathbf{x}=[x_1，...，x_N]∈R^{N×P^2×C}$，其中$(P,P)$是划分的块的大小，$N = H W / P^2$是块的数量，$C$是通道的数量。每个块被展平成一个一维向量，然后线性投影到一个**patch embeddings**，产生一个块嵌入序列$x_0=[E_{x_1}，...，E_{x_N}]∈R^{N×D}$，其中$E∈R^{D×(P^2C)}$。 为了获取位置信息，将可学习的位置嵌入$\text{pos}= [ pos_1 ， . . . ， pos_N ] ∈ R^{N × D}$添加到块序列中，得到**token** $z_0=x_0+pos$的输入序列。将由$L$层组成的**transformer**编码器应用于标记$z_0$的序列，生成上下文化编码$z_L∈R^{N×D}$序列。

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model)
        )
        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
```

### ⚪ 解码器

解码器将来自编码器的图像块编码序列$z_L∈R^{N×D}$映射到**patch**级类别分数，然后通过双线性插值上采样到分割映射$s∈R^{H×W×K}$，其中$K$为类别数。解码器有两种形式，分别为一个轻量级的线性解码器和一个表现更好的**Mask Transformer**。

逐点线性解码器对块的编码$z_L∈R^{N×D}$应用逐点线性层，产生块级类别数$z_{lin}∈R^{N×K}$，然后将序列重塑为**2D**特征图$s_{lin}∈R^{H/P×W/P×K}$，并提前上采样到原始图像大小$s∈R^{H×W×K}$，然后在类别维度上应用**softmax**，得到最终的分割映射。

```python
class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.head = nn.Linear(self.d_encoder, n_cls)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        return x
```

**Mask Transformer**则是引入了一组$K$个可学习的类嵌入$c=[cls_1，...，cls_K]∈R^{K×D}$，其中$K$是类的数量。每个类的嵌入都被随机初始化，并分配给一个语义类。它将用于生成类别掩码。解码器是一个由$M$层组成的**transformer**编码器，通过计算解码器输出的$L_2$标准化**patch**嵌入$z^{\prime}_M∈R^{N×D}$与类嵌入$c∈R^{K×D}$之间的乘积来生成$K$个掩码。类别掩码的集合计算如下：

$$
\operatorname{Masks}\left(z_M^{\prime}, c\right)=z_M^{\prime} c^T
$$

其中，$Masks(z_M,c)∈R^{N×K}$是一组图像块序列。然后将每个**mask**序列重塑为二维**mask**，形成$s_{mask}∈R^{H/P×W/P×K}$, 并上采样到原始图像大小，获得特征图$s∈R^{H×W×K}$。然后在类别维度上应用**softmax**和层归一化，得到像素级的类别分数，形成最终的分割图。


```python
class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)


    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        return masks
```