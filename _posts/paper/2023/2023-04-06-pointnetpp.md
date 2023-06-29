---
layout: post
title: 'PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space'
date: 2023-04-06
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/649d3e991ddac507cc46c6fd.jpg'
tags: 论文阅读
---

> PointNet++：度量空间中点集上的深层特征学习.

- paper：[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)

[<font color=blue>PointNet</font>](https://0809zheng.github.io/2023/04/05/pointnet.html)可以直接从点云数据中提取特征，然而它只能捕捉全局信息。本文在其基础上设计了**PointNet++**，具有编码器-解码器结构，能够捕捉局部信息。

**PointNet++**的编码器模块引入了**set abstracion level**来进行局部信息聚合。该结构接受一个点云作为输入，把点云划分成若干局部区域，输出点数更少的点云。其分为这几个模块：
- **Sampling layer**：用来定义局部区域中心。在接受的$n$个点中，根据度量函数，挑选$m$个互相之间距离最远的点。
- **Grouping layer**：对于$m$个中心点对应的局部区域，找到每个区域内的点有哪些。
- **PointNet layer**：对于$m$个区域，使用**PointNet**得到$m$个新点；其中特征向量是**PointNet**给出，空间坐标为原本的中心点。

对于分类任务，解码器模块使用**PointNet**提取全局特征，然后通过**MLP**进行分类；对于分割任务，整体结构类似于**UNet**，其中上采样通过找附近的点用距离的倒数作为权重插值，使用$1\times 1$卷积进行通道降维。

![](https://pic.imgdb.cn/item/649d3eb91ddac507cc47010a.jpg)

## （1）Sampling layer：最远点采样

**Sampling layer**是在所有的点云数据中（假设有$N$个点）采样$N'$个点；假设网络的输入尺寸为$N\times d$，其中$N$是点云数据的数据点数量，$d$为点上的特征；通过**Sampling layer**后网络的输出变成了$N'\times d$。

从$N$个点云数据中采样$N'$个点，而且希望这$N'$个点能够包含尽可能多的有用信息，可以通过**最远点采样(farthest point sampling, FPS)**算法实现。

最远点采样算法的流程如下：
1. 随机选择一个点作为初始点并加入已选择采样点集；
2. 计算未选择采样点集中每个点与已选择采样点之间的距离，将距离最大的那个点加入已选择采样点集；
3. 更新未选择采样点集中每个点距离为与所有已选择采样点之间距离的最小值；
4. 循环迭代**2,3**，直至获得了目标数量的采样点。

```python
def farthest_point_sample(point_xyz, n_point):
    """
    :param point_xyz: points xyz coord
    :param n_point: sample num
    """
    device = point_xyz.device
    B, N, C = point_xyz.shape

    sample_points_index = torch.zeros([B, n_point], dtype=torch.long).to(device)
    distance = torch.ones([B, N], dtype=torch.long).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for ii in range(n_point):
        sample_points_index[:, ii] = farthest
        sample_point = point_xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((point_xyz - sample_point) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return sample_points_index
```

## （2）Grouping layer：Ball Query

**Grouping layer**是以之前采样的$N'$个点为中心找到结构相同的$N'$个子区域，每个子区域包含$K$个点，每个点的维度为$d$。通过**Grouping layer**后，网络的输出变成了$N'\times K\times d$。

**Ball Query**的实现过程：
1. 预设搜索区域的半径$R$与子区域的点数$K$；
2. 以之前采样的$N'$个点为中心画半径为$R$的球体作为搜索区域；
3. 在每个球体内搜索离中心最近的$K$个点。如果球体内点的数量小于$K$，则直接对最近点重采样凑够规模$K$;
4. 获取所有$N'$个中心点对应的$N'$个子区域，每个子区域包含$K$个点。


```python
def index_points(points, idx):
    """
    Index points while keep the data dim as [B,N',C] type
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, M, S]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, M, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def compute_dist_square(src, tar):
    """
    :param src: source point [B, N, C]
    :param tar: target point [B, S, C]
    :return: dis pair [B, N, S]
    """
    B, N, _ = src.shape
    _, S, _ = tar.shape
    dist = -2 * torch.matmul(src, tar.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(tar ** 2, -1).view(B, 1, S)
    return dist

def raidus_nn_sample(point_xyz, query_xyz, radius, k):
    """
    :param point_xyz:  all point [B, N, 3]
    :param query_xyz:  query point [B, M, 3]
    :param radius: search radius
    :param k: radius nn points num limit K
    :return: nn idx [B, M, K]
    """
    device = point_xyz.device
    B, N, C = point_xyz.shape
    _, M, _ = query_xyz.shape
    # dist [B, M, N]
    dist = compute_dist_square(point_xyz, query_xyz)
    dist = dist.permute(0, 2, 1)
    # nn_idx [B, M, N]
    nn_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    nn_idx[dist > radius ** 2] = N
    # sort, make label N lie on the end of each line，leave the first k elements
    nn_idx = torch.sort(nn_idx, dim=-1)[0][:, :, :k]
    # replace non radius nn points use nearest radius nn point
    nn_nearest = nn_idx[:, :, 0].view(B, M, 1).repeat([1, 1, k])
    mask = nn_idx == N
    nn_idx[mask] = nn_nearest[mask]
    return nn_idx

def sample_and_group(points_xyz, points_feature, n_point, radius, nn_k, returnfps=False):
    """
    :param points_xyz: input point xyz [B, N, 3]
    :param points_feature: input point point-wise feature [B, N, D]
    :param n_point: sample centriod points num
    :param radius: nn radius
    :param nn_k: nn k
    :param returnfps: if return  fps_idx
    :return:
        sample_points: xyz [B, n_point, nn_k, 3]
        group_points: norm_xyz+feature [B, npoint, nn_k, 3+D]
    """
    # get shape
    B, N, C = points_xyz.shape
    # fps sampling
    fps_points_idx = farthest_point_sample(points_xyz, n_point)
    torch.cuda.empty_cache()
    # index points
    sample_points = index_points(points_xyz, fps_points_idx)
    torch.cuda.empty_cache()
    # grouping
    nn_idx = raidus_nn_sample(points_xyz, sample_points, radius, nn_k)
    torch.cuda.empty_cache()
    # index points
    group_points_xyz = index_points(points_xyz, nn_idx)  # [B, n_point, nn_k, C]
    torch.cuda.empty_cache()
    # group normalization
    group_points_xyz_norm = group_points_xyz - sample_points.view(B, n_point, 1, C)

    # concatenate feature
    if points_feature is not None:
        group_points_featrue = index_points(points_feature, nn_idx)
        group_points = torch.cat([group_points_xyz_norm, group_points_featrue], dim=-1)
    else:
        group_points = group_points_xyz_norm
    if returnfps:
        return sample_points, group_points, group_points_xyz, fps_points_idx
    else:
        return sample_points, group_points
```

## （3）PointNet layer

**PointNet layer**使用**PointNet**从$N'$个子区域中提取全局特征。通过**PointNet layer**后，网络的输出变成了$N'\times d$。**PointNet**设计为简单的“$1\times 1$卷积+**BN**+**ReLU**”堆叠的形式，并通过最大池化提取全局特征。

## （4）Set Abstraction

**Sampling layer**+**Grouping layer**+**PointNet layer**被称为**Set Abstraction**，该模块把输入点云$N\times d$转换成数量更少的输出点云特征$N'\times d$，并且汇聚了局部特征。

```python
class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_points, radius, nn_k, in_channel, mlp, group_all=False):
        """
        :param n_points: sample points num
        :param radius: nn radius
        :param nn_k: nn num
        :param in_channel: input channel
        :param mlp: pointnet mlp
        :param group_all: if group all point
        """
        super(PointNetSetAbstraction, self).__init__()
        self.n_point = n_points
        self.radius = radius
        self.nn_k = nn_k
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm2d(out_channel))
            self.relus.append(nn.ReLU())
            last_channel = out_channel

    def forward(self, points_xyz, points_features):
        """
        :param point_xyz: [B, C, N]
        :param point_features: [B, D, N]
        :return:
            out_xyz: [B, C, M]
            out_features: [B, D', M]
        """
        points_xyz = points_xyz.permute(0, 2, 1)
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)

        if not self.group_all:
            out_xyz, group_points = sample_and_group(points_xyz, points_features, self.n_point, self.radius, self.nn_k)
        else:
            out_xyz, group_points = sample_and_group_all(points_xyz, points_features)

        group_points = group_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        out_xyz = out_xyz.permute(0, 2, 1)
        x = group_points
        for mpl_conv, bn, relu in zip(self.mlp_convs, self.bns, self.relus):
            x = mpl_conv(x)
            x = bn(x)
            x = relu(x)

        x = torch.max(x, dim=2)[0]
        return out_xyz, x
```

## （5）特征融合

在点云不均匀的时候，在密集区域学习出来的特征可能不适合稀疏区域。因此作者提出了两种特征融合方式。

![](https://pic.imgdb.cn/item/649d543f1ddac507cc6d569e.jpg)

### ⚪ Multi-scale grouping (MSG)

**MSG**是指对不同半径的子区域进行特征提取后进行特征堆叠。

```python
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, n_points, radius_list, nn_k_list, in_channel, mlp_list):
        """
        :param n_points: sample num
        :param radius_list: radius list [r1,r2,...]
        :param nn_k_list: nn_k list [n1,n2,...]
        :param in_channel: input channel
        :param mlp_list: pointnet mlp for each scale
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.n_point = n_points
        self.radius_list = radius_list
        self.nn_k_list = nn_k_list
        self.mlp_convs_blocks = nn.ModuleList()
        self.bns_blocks = nn.ModuleList()
        self.relus_blocks = nn.ModuleList()
        for mlp in mlp_list:
            last_channel = in_channel + 3
            mlp_convs = nn.ModuleList()
            bns = nn.ModuleList()
            relus = nn.ModuleList()
            for out_channel in mlp:
                mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                relus.append(nn.ReLU())
                last_channel = out_channel
            self.mlp_convs_blocks.append(mlp_convs)
            self.bns_blocks.append(bns)
            self.relus_blocks.append(relus)

    def forward(self, points_xyz, points_features):
        """
        :param point_xyz: [B, C, N]
        :param point_features: [B, D, N]
        :return:
            out_xyz: [B, C, M]
            out_features: [B, D', M]
        """
        points_xyz = points_xyz.permute(0, 2, 1)
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)
        B, N, C = points_xyz.shape
        S = self.n_point
        out_xyz = index_points(points_xyz, farthest_point_sample(points_xyz, self.n_point))
        x_list = []
        for radius, nn_k, mlp_convs, bns, relus in zip(self.radius_list, self.nn_k_list, self.mlp_convs_blocks,
                                                       self.bns_blocks, self.relus_blocks):
            nn_idx = raidus_nn_sample(points_xyz, out_xyz, radius, nn_k)
            # index points
            group_points_xyz = index_points(points_xyz, nn_idx)  # [B, n_point, nn_k, C]
            # group normalization
            group_points_xyz_norm = group_points_xyz - out_xyz.view(B, S, 1, C)

            # concatenate feature
            if points_features is not None:
                group_points_featrue = index_points(points_features, nn_idx)
                group_points = torch.cat([group_points_xyz_norm, group_points_featrue], dim=-1)
            else:
                group_points = group_points_xyz_norm

            group_points = group_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
            x = group_points
            for mpl_conv, bn, relu in zip(mlp_convs, bns, relus):
                x = mpl_conv(x)
                x = bn(x)
                x = relu(x)

            x = torch.max(x, dim=2)[0]
            x_list.append(x)
            
        out_xyz = out_xyz.permute(0, 2, 1)
        x = torch.cat(x_list, dim=1)
        return out_xyz, x
```

### ⚪ Multi-resolution grouping (MRG)

**MSG**方法计算量太大，**MRG**用两个**Pointnet**对连续的两层分别做特征提取与聚合，然后再进行特征拼接。

## （6）分类网络

对于分类任务，由编码器提取的点云特征经过一个全局**Pointnet**提取全局特征，并通过**MLP**进行分类。

```python
def sample_and_group_all(points_xyz, points_feature):
    """
    Equivalent to sample_and_group with input parameter n_point = 1 ,radius = inf, nn_k = N
    Input:
        points_xyz: input points position data, [B, N, 3]
        points_feature: input points data, [B, N, D]
    Return:
        sample_points: sampled points position data, [B, 1, 3]
        group_points: sampled points data, [B, 1, N, 3+D]
    """
    device = points_xyz.device
    B, N, C = points_xyz.shape

    # sample point is [0, 0, 0]
    sample_points = torch.zeros(B, 1, C).to(device)

    # grouping all points
    group_points_xyz = points_xyz.view(B, 1, N, C)
    if points_feature is not None:
        group_points = torch.cat([group_points_xyz, points_feature.view(B, 1, N, -1)], dim=-1)
    else:
        group_points = group_points_xyz
    return sample_points, group_points

class PointNetpp_cls(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(PointNetpp_cls, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.reg = nn.LogSoftmax(dim=-1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.reg(x)
        return x, l3_points
```

## （7）分割网络

点云数据的分割任务实际上就是为原始点云中的每个点分配一个语义标签。作者提出了一种利用基于距离插值的分层特征传播（**Feature Propagation**）策略，将已经进行特征提取的点通过上采样的方式，将特征传播给在下采样过程中丢失的点（未参与特征提取的点）。

分层特征传播是基于**k**近邻的反向距离加权平均的插值方式，实现了丢失点（待插值点）特征的求解。假设丢失点$x$的待求解特征为$f^{(j)}(x)$，并假设其$k$个特征已知的近邻点特征为$f_i^{(j)},i=1,...,k$。则有

$$
f^{(j)}(x) = \frac{\sum_{i=1}^kd(x,x_i)^{-p}f_i^{(j)}}{\sum_{i=1}^kd(x,x_i)^{-p}}
$$

```python
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
```

完整的分割过程为：
1. 通过分层特征传播实现点云特征的插值；
2. 将插值特征与编码器中的对应特征（两者具有相同数量的特征点）通过跳跃连接的结构连接后进行特征堆叠；
3. 堆叠的特征被输入到一个**unit pointnet**网络（类似于$1\times 1$卷积）中实现特征的进一步提取。
4. 重复**1,2,3**若干次；
5. 利用$1\times 1$卷积+**BN**+**ReLU**输出分割预测结果。

```python
class PointNetpp_seg(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(PointNetpp_seg, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points
```