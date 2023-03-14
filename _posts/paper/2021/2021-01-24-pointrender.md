---
layout: post
title: 'PointRend: Image Segmentation as Rendering'
date: 2021-01-24
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/640ec455f144a01007338e02.jpg'
tags: 论文阅读
---

> PointRend: 把图像分割建模为渲染.

- paper：[PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)


**PointRend**将渲染领域的操作引入到分割领域，本身可以理解为一个新颖的**上采样**操作。

在图像分割问题中，边缘的恢复、精确分割是比较麻烦的问题。在比较经典的一些语义分割模型中，在模型后都是接一个**8**倍或**4**倍的上采样操作来恢复图像，这对于物体边缘的预测自然是不利的，上采样会损失一些边缘信息。因此**PointRend**就是为上采样过程中精确恢复物体边缘的任务而生。

![](https://pic.imgdb.cn/item/640ec590f144a01007377475.jpg)

**PointRend**包含了两个阶段的特征处理，分别是**fine-grained features**和**coarse prediction**部分，如果主干网络是**ResNet**，那么**fine-grained features**就是**ResNet**的**stage2**输出，也就是**4**倍下采样时的精细分割结果，而**coarse prediction**就是检测头的预测结果（还未上采样还原成原图的结果）。

![](https://pic.imgdb.cn/item/640ec603f144a01007388399.jpg)

从**coarse prediction**中挑选**N**个“难点”，也就是结果很有可能和周围点不一样的点（比如物体边缘的点）。对于每一个难点，获取他的“特征向量”，对于点特征向量（**point features**），主要由两部分组成，分别是**fine-grained features**的对应点和**coarse prediction**的对应点的特征向量，将这两个特征向量拼接成一个向量。接着通过一个**MLP**网络对这个“特征向量”进行预测，更新**coarse prediction**。也就相当于对这个难点进行新的预测，对像素进行分类。

![](https://pic.imgdb.cn/item/640ecacef144a010074121e0.jpg)

如上图所示，对于一个**coarse prediction**(**4x4**大小)，将其上采样两倍(**8x8**大小，这里可以理解为检测头的输出)后，取了一些难分割的点（大多是边缘部分），取这些点的特征向量输入到**MLP**网络中，进行**point prediction**，得到每一个点的新类别，最后结果输出(**8x8**大小，边缘更加精确的结果)。

**PointRend**是基于点的预测，那么这些点该如何采样？如果对全局点进行采样，那么计算量就过大。如果只想对预测困难点（物体边界）进行采样，则点采样过程需要对模型的**Train**过程和**Inference**过程做区分。

在**Inference**过程中，每个区域都通过迭代**coarse-to-fine**的方式来渲染。在每一次迭代过程中，**PointRend**都使用双线性差值将上一次的**segmentation result**进行上采样，然后在这个结果中选择**N**个不确定的点（分类概率接近**0.5**的点，也就是模型认为模棱两可的点）提取特征向量，经过**MLP**进行分类，得到新的**segmentation result**。

对于**Train**过程的点采样操作，遵循**Inference**中的子采样对于梯度的传播不太友好，于是选择随机采样的方式来进行采样。首先依据均匀分布随机取$kN$个点($k>1$)；然后上采样后，预测估计这些点的结果，再从$kN$个点中选取$βN$个点($0<β<1$)。

![](https://pic.imgdb.cn/item/640eccf3f144a01007446e44.jpg)

```python
def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # F.grid_sample(a, grid)按照grid采样，grid归一化为[-1,1]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, _, H, W = mask.shape
    # C维度存储每个类别的分类概率
    mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        # 用前两个分类最大概率之差的负值衡量分类不确定度
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)
    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)
```

**PointRend**对于物体的边缘恢复效果是很不错的，而且很灵活，可以作为上采样操作放置在很多分割网络后面。

```python
class PointHead(nn.Module):
    def __init__(self,num_classes, in_c=512, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c+num_classes, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)

        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation)
        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096

        while out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(out, num_points, training=self.training)

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(res2, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1).scatter_(2, points_idx, rend).view(B, C, H, W))

        return {"fine": out}


class PointRend(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        result = self.backbone(x)
        result.update(self.head(x, result["res2"], result["coarse"]))
        return result

if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    net = PointRend(deeplabv3(False,num_classes=33), PointHead(num_classes=33))
    out = net(x)
    for k, v in out.items():
        print(k, v.shape)
    # {"coarse":, "rend":, "points"}
```

在**inference**阶段，**PointRend**直接输出**render**之后的精细分割结果；在**training**阶段，**PointRend**提供粗略的分割结果和**render**点集的预测结果，并以此构造两个损失：

```python
...

result = net(X)
pred = F.interpolate(result["coarse"], X.shape[-2:], mode="bilinear", align_corners=True)

seg_loss = F.cross_entropy(pred, gt, ignore_index=255)
gt_points = point_sample(
    gt.float().unsqueeze(1),
    result["points"],
    mode="nearest",
    align_corners=False
).squeeze_(1).long()
points_loss = F.cross_entropy(result["rend"], gt_points, ignore_index=255)  

loss_sum = seg_loss + points_loss

...
```