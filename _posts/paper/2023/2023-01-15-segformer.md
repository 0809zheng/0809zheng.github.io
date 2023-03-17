---
layout: post
title: 'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers'
date: 2023-01-15
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/641417d4a682492fcc36c846.jpg'
tags: 论文阅读
---

> SegFormer：为语义分割设计的简单高效的Transformer模型.

- paper：[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

**SegFormer**是一种为语义分割任务设计的**Transformer**模型，主要有以下几个特点：
- **ViT**做**patch embedding**时，每个**patch**都是独立的，而**SegFormer**对**patch**设计成有重叠的，保证局部连续性。
- 使用了多尺度特征融合。**Encoder**输出多尺度的特征，**Decoder**将多尺度的特征融合在一起。模型能够同时捕捉高分辨率的粗略特征和低分辨率的细小特征，优化分割结果。
- 舍弃了**ViT**中的**position embedding**位置编码，取而代之的是**Mix FFN**。在测试图片大小与训练集图片大小不一致时，不需要再对位置向量做双线性插值。
- 轻量级的**Decoder**：使得**Decoder**的计算量和参数量非常小，从而使得整个模型可以高效运行，简单直接。并且通过聚合不同层的信息，结合了局部和全局注意力。

![](https://pic.imgdb.cn/item/6414188aa682492fcc38e8ce.jpg)

**SegFormer**可以分为两个部分：用于生成多尺度特征的分层**Encoder**和轻量级的**All-MLP Decoder**，融合多层特征并上采样，最终解决分割任务。输入一张大小$H \times W \times 3$的图片，首先将其划分为大小$4 \times 4$的**patches**。对比**ViT**中使用的大小$16 \times 16$的**patches**，使用更小的**patches**有利于进行分割任务。使用这些**patches**作为**Encoder**的输入，获取大小为$\frac{H}{4} \times \frac{W}{4} \times C_1$、$\frac{H}{8} \times \frac{W}{8} \times C_2$、$\frac{H}{16} \times \frac{W}{16} \times C_3$、$\frac{H}{32} \times \frac{W}{32} \times C_4$的多尺度的特征图。将这些多尺度特征输入到解码器中，经过一系列**MLP**和上采样操作，最终输出大小$\frac{H}{4} \times \frac{W}{4} \times N_{cls}$的特征图，其中$N_{cls}$是类别个数。

## （1） 编码器

**Encoder**是由**Transformer Block**堆叠起来的，其中包含**Efficient Self-Attention**、**Mix-FFN**和**Overlap Patch Embedding**三个模块。

### ⚪ Overlapped Patch Merging

为了产生类似于**CNN backbone**的多尺度特征图，**SegFormer**使用了**patch merging**的方法，通过$H \times W \times 3$的输入图像，得到大小$\frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}} \times C_i$的多尺度特征图，其中$$i \in \{1,2,3,4\}$$，并且$C_{i+1}>C_i$。

**ViT**中的**patch merging**可以将$2 \times 2 \times C_i$的特征图合并成为$1 \times 1 \times C_{i+1}$的向量来达到降低特征图分辨率的目的。**SegFormer**同样使用这种方法，将分层特征从$F_1 (\frac{H}{4} \times \frac{W}{4} \times C_1)$缩小到$F_2 (\frac{H}{8} \times \frac{W}{8} \times C_2)$，同样的方法可以得到$F_3,F_4$。但是由于**ViT**中的**patch**是不重叠的，会丢失**patch**边界的连续性，因此**SegFormer**在切割**patch**时采用了重叠的**patch**。切割方法类似于卷积核在**feature map**上的移动卷积，源代码中也是采用卷积来实现，设置卷积核大小$K$、步距$S$、填充大小$P$。

第一个**Transformer Block**的**Patch Merging**设置为$K=7,S=4,P=3$，这样输出特征图大小变成输入特征图大小的$\frac{1}{4}$。之后三个**Transformer Block**的**Patch Merging**设置为$K=3,S=2,P=1$，输出特征图大小变为输入特征图大小的$\frac{1}{2}$。这样最终就得到了分辨率分别是$\frac{H}{4} \times \frac{W}{4} \times C_1$、$\frac{H}{8} \times \frac{W}{8} \times C_2$、$\frac{H}{16} \times \frac{W}{16} \times C_3$、$\frac{H}{32} \times \frac{W}{32} \times C_4$的多尺度的特征图。

```python
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W 
```

### ⚪ Efficient Self-Attention

**Transformer**的计算量之所以大，主要是因为其**Self-Attention**的计算。对应**multi-head self-attention**来说，每一个**head**的**Q、K、V**都是相同维度$N \times C$，其中$N=H \times W$是序列的长度。这个过程的计算复杂度是$O(N^2)$，对于高分辨率的图像来说这是无法计算的。

为了减少计算量，作者采用了**spatial reduction**操作。输入维度$N \times C$的**K**和**V**矩阵通过**Reshape**变成$\frac{N}{R} \times (C \cdot R)$的大小。然后通过线性变换，将$(C \cdot R)$的维度变为$C$。这样输出的大小就变成了$\frac{N}{R} \times C$。计算复杂度就变成了$O(\frac{N^2}{R})$。论文中四个**Transformer Block**分别将$R$设置成了$[64,16,4,1]$。源代码中使用卷积实现。

$$
\begin{aligned}
\hat{K} & =\operatorname{Reshape}\left(\frac{N}{R}, C \cdot R\right)(K) \\
K & =\operatorname{Linear}(C \cdot R, C)(\hat{K})
\end{aligned}
$$

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### ⚪ Mix-FFN

作者认为在语义分割任务中实际上并不需要**position encoding**，采用**Mix-FFN**替代。**Mix-FFN**假设**0 padding**操作可以汇入位置信息，直接用**0 padding**的$3 \times 3$卷积来达到这一目的：

$$
\mathbf{x}_{\text {out }}=\operatorname{MLP}\left(\operatorname{GELU}\left(\operatorname{Conv}_{3 \times 3}\left(\operatorname{MLP}\left(\mathbf{x}_{i n}\right)\right)\right)\right)+\mathbf{x}_{i n}
$$

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

### ⚪ Encoder

完整的编码器实现如下：

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs
```

## （2） 解码器

**SegFormer**的**Decoder**是一个仅由**MLP**层组成的轻量级**Decoder**，之所以能够使用这种简单结构，关键在于分层**Transformer**编码器具有比传统**CNN**编码器更大的有效感受野。

**All-MLP Decoder**包含四个主要步骤：

① 来自**Encoder**的四个不同分辨率的特征图$F_i$分别经过**MLP**层使得通道维度相同;

$$
\hat{\mathrm{F}}_{\mathrm{i}}=\operatorname{Linear}\left(\mathrm{C}_{\mathrm{i}}, \mathrm{C}\right)\left(\mathrm{F}_{\mathrm{i}}\right), \forall \mathrm{i}
$$

② 将特征图分别进行双线性插值上采样到原图的$\frac{1}{4}$，并拼接在一起;

$$
\hat{\mathrm{F}}_{\mathrm{i}}=\mathrm{U} \text { psample }\left(\frac{\mathrm{W}}{4} \times \frac{\mathrm{W}}{4}\right)\left(\hat{\mathrm{F}}_{\mathrm{i}}\right), \forall \mathrm{i}
$$

③ 使用**MLP**层来融合级联特征；

$$
\mathrm{F}=\operatorname{Linear}(4 \mathrm{C}, \mathrm{C})\left(\operatorname{Concat}\left(\hat{\mathrm{F}}_{\mathrm{i}}\right)\right), \forall \mathrm{i}
$$

④ 使用另外一个**MLP**层采用融合特征图输出最终$\frac{H}{4} \times \frac{W}{4} \times N_{cls}$的预测特征图。

$$
\mathrm{M}=\operatorname{Linear}\left(\mathrm{C}, \mathrm{N}_{\mathrm{cls}}\right)(\mathrm{F})
$$

```python
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x
```
