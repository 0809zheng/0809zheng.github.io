---
layout: post
title: '二值图像的距离变换(Distance Transform)'
date: 2023-03-22
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/641c0a6aa682492fcc7d1da6.jpg'
tags: 数学
---

> Distance Transform Algorithm of Binary Images.

**距离变换(Distance Transform)**是一种针对二值图像（背景: $0$, 前景: $1$）的变换算法，把图像中的每个像素值替换为该像素到前景像素的最近距离。通过距离变换能够基本找出二值图像中前景形状的骨架。

![](https://pic.imgdb.cn/item/641c0a6aa682492fcc7d1da6.jpg)

对于距离的度量可以选择：

- 欧氏距离：在事实上比较直观，但是平方根计算比较费时，且距离可能不是整数。

$$ D_E =\sqrt{(i-k)^2+(j-l)^2} $$

- 城市街区距离：在只允许横向和纵向运动的情况下，从起点到终点的移动步数。该距离中像素点是4邻接的，即每个点只与它的上、下、左、右相邻的4个点之间的距离为1。

$$ D_4 =|i-k|+|j-l| $$

- 棋盘距离：允许横向、纵向和沿对角线方向移动的情况下，从起点到终点的移动步数。在这种定义下，像素点是8邻接的，即每个点只与它的上、下、左、右、四个对角线方向相邻的8个点之间的距离为1。

$$ D_8 =\max \left \{ |i-k|,|j-l| \right \} $$

# 1. 距离变换的算法实现

距离变换算法可以通过广度优先搜索或动态规划实现（对应[LeetCode 542. 01 矩阵](https://leetcode.cn/problems/01-matrix/description/)），也可以直接调用第三方库。

## ⚪ 通过广度优先搜索实现距离变换

对于矩阵中的每一个元素，如果它的值为1，那么离它最近的1就是它自己。如果它的值为0，那么我们就需要找出离它最近的1，并且返回这个距离值。

我们可以从1的位置开始进行**广度优先搜索**。广度优先搜索可以找到从起点到其余所有点的最短距离，因此如果我们从1开始搜索，每次搜索到一个0，就可以得到1到这个0的最短距离，也就是离这个0最近的1的距离了。

在进行广度优先搜索的时候会使用到队列，我们在搜索前会把所有的1的位置加入队列（对应距离为0）。

```python
_ _ _ _
_ 0 _ _
_ _ 0 _
_ _ _ _
```

随后我们进行广度优先搜索，找到所有距离为1的0：

```python
_ 1 _ _
1 0 1 _
_ 1 0 1
_ _ 1 _
```

接着重复步骤，直到搜索完成：

```python
_ 1 _ _         2 1 2 _         2 1 2 3
1 0 1 _   ==>   1 0 1 2   ==>   1 0 1 2
_ 1 0 1         2 1 0 1         2 1 0 1
_ _ 1 _         _ 2 1 2         3 2 1 2
```

```python
def DistanceTransform(mat):
    m, n =len(mat), len(mat[0])
    res = [[0]*n for _ in range(m)]

    from collections import deque
    queue = deque([(i, j) for i in range(m) for j in range(n) if mat[i][j]==1])
    visited = set(queue)

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_x, new_y = x+dx, y+dy
            if 0<=new_x<m and 0<=new_y<n and (new_x,new_y) not in visited:
                res[new_x][new_y] = res[x][y]+1
                queue.append((new_x,new_y))       
                visited.add((new_x,new_y))                 

    return res
```

## ⚪ 通过动态规划实现距离变换

距离矩阵中任意一个元素0最近的元素1只可能出现在四个方向：左上、左下、右上、右下。因此我们可以进行四次动态搜索：
- 水平向左移动 和 竖直向上移动；
- 水平向左移动 和 竖直向下移动；
- 水平向右移动 和 竖直向上移动；
- 水平向右移动 和 竖直向下移动。

以「水平向左移动」和「竖直向上移动」为例，用 $f(i, j)$ 表示位置 $(i, j)$ 到最近的 1 的距离，那么我们可以向上移动一步，再移动 $f(i - 1, j)$ 步到达某一个 1；也可以向左移动一步，再移动 $f(i, j - 1)$ 步到达某一个 1。因此可以写出如下的状态转移方程：

$$
f(i, j) = \begin{cases} 1 + \min\big(f(i - 1, j), f(i, j - 1)\big) &, \text{位置 } (i, j) \text{ 的元素为 } 0 \\ 0 &, \text{位置 } (i, j) \text{ 的元素为 } 1 \end{cases}
$$
​
通过这种遍历，我们搜索到任意位置$x$左上角的元素1，并作为最近距离的候选。

```python
_ _ _ _         o o o _
_ _ _ _   ==>   o o o _
_ _ x _         o o x _
_ _ _ _         _ _ _ _
```

对于另外三种移动方法，我们也可以写出类似的状态转移方程，得到四个 $f(i, j)$ 的值，那么其中最小的值就表示位置 $(i, j)$ 到最近的 1 的距离。

```python
def DistanceTransform(mat):
    m, n =len(mat), len(mat[0])
    # 初始化动态规划的数组，所有的距离值都设置为一个很大的数
    dp = [[1e9]*n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 1:
                dp[i][j] = 0

    # 只有 水平向左移动 和 竖直向上移动，注意动态规划的计算顺序
    for i in range(m):
        for j in range(n):
            if i>0:
                dp[i][j] = min(dp[i][j], 1+dp[i-1][j])
            if j>0:
                dp[i][j] = min(dp[i][j], 1+dp[i][j-1])

    # 只有 水平向左移动 和 竖直向下移动，注意动态规划的计算顺序
    for i in range(m-1,-1,-1):
        for j in range(n):
            if i<m-1:
                dp[i][j] = min(dp[i][j], 1+dp[i+1][j])
            if j>0:
                dp[i][j] = min(dp[i][j], 1+dp[i][j-1])

    # 只有 水平向右移动 和 竖直向上移动，注意动态规划的计算顺序
    for i in range(m):
        for j in range(n-1,-1,-1):
            if i>0:
                dp[i][j] = min(dp[i][j], 1+dp[i-1][j])
            if j<n-1:
                dp[i][j] = min(dp[i][j], 1+dp[i][j+1])
    
   # 只有 水平向右移动 和 竖直向下移动，注意动态规划的计算顺序
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if i<m-1:
                dp[i][j] = min(dp[i][j], 1+dp[i+1][j])
            if j<n-1:
                dp[i][j] = min(dp[i][j], 1+dp[i][j+1])                        

    return dp
```

### ⭐ 进一步化简

我们发现上述方法中的代码有一些重复计算的地方。实际上，只需要保留
- 只有 水平向左移动 和 竖直向上移动；
- 只有 水平向右移动 和 竖直向下移动。

这两者即可（或者另外两者），下面尝试说明。按照之前的思路，进行两次动态搜索后，某个元素$x$左上角和右下角的候选元素1已经被找到。

```python
_ _ _ _         o o o _
_ _ _ _   ==>   o o o _
_ _ x _         o o x o
_ _ _ _         _ _ o o
```

接下来考察左下角和右上角元素。在这给出一个性质： 假如距离$x=(i,j)$最近的1在右上角$(i-a,j+b),a>0,b>0$，则距离$(i,j+b)$最近的1也在$(i-a,j+b)$。

该性质可以采用反证法证明： 如果距离$(i,j+b)$最近的点$(x,y)$不在$(i-a,j+b)$，则$(i,j+b)$和$(x,y)$距离$d<a$，这时点$(i,j)$和$(x,y)$的距离$d'<=b+d<a+b$，与假设矛盾。

利用这个性质，如果距离$(i,j)$最近的1在右上角$(i-a,j+b),a>0,b>0$，在第一次动态搜索时$(i,j)$没有取得最优值，但在搜索中$(i,j+b)$取得最优值(因为这个最优值在他正上方)；在第二次动态搜索时$(i,j)$可以搜索到$(i,j+b)$，进而间接地访问到原本位于其右上角的1。

```python
def DistanceTransform(mat):
    m, n =len(mat), len(mat[0])
    # 初始化动态规划的数组，所有的距离值都设置为一个很大的数
    dp = [[1e9]*n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 1:
                dp[i][j] = 0

    # 只有 水平向左移动 和 竖直向上移动，注意动态规划的计算顺序
    for i in range(m):
        for j in range(n):
            if i>0:
                dp[i][j] = min(dp[i][j], 1+dp[i-1][j])
            if j>0:
                dp[i][j] = min(dp[i][j], 1+dp[i][j-1])
    
   # 只有 水平向右移动 和 竖直向下移动，注意动态规划的计算顺序
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if i<m-1:
                dp[i][j] = min(dp[i][j], 1+dp[i+1][j])
            if j<n-1:
                dp[i][j] = min(dp[i][j], 1+dp[i][j+1])                        

    return dp
```

## ⚪ 通过`scipy.ndimage.distance_transform_edt`实现距离变换

`scipy.ndimage.distance_transform_edt`的作用是计算一张图上每个前景像素点$1$到背景$0$的最近距离，并且支持多通道输入。

```python
import numpy as np
from scipy.ndimage import distance_transform_edt
 
a = np.array((([0, 1, 1, 1, 1],
              [0, 0, 1, 1, 1],
              [0, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 1, 1, 0, 0]),
             ([0, 1, 1, 1, 1],
              [0, 0, 1, 1, 1],
              [0, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 1, 1, 0, 0]))
             )
 
y1 = distance_transform_edt(a)
print(y1.shape)  # (2, 5, 5)
print(y1)
# [[[0.         1.         1.41421356 2.23606798 3.        ]
#   [0.         0.         1.         2.         2.        ]
#   [0.         1.         1.41421356 1.41421356 1.        ]
#   [0.         1.         1.41421356 1.         0.        ]
#   [0.         1.         1.         0.         0.        ]]
#  [[0.         1.         1.41421356 2.23606798 3.        ]
#   [0.         0.         1.         2.         2.        ]
#   [0.         1.         1.41421356 1.41421356 1.        ]
#   [0.         1.         1.41421356 1.         0.        ]
#   [0.         1.         1.         0.         0.        ]]]
```

# 2. 距离变换的应用

## (1) 构造分割任务的损失函数

分割任务的真实标签为多通道的二值图像，因此可以通过构造真实标签的距离变换图为每个像素生成距离目标轮廓边界的距离，并进一步根据距离信息对不同像素的损失进行加权，从而使模型更加关注分割的轮廓边界区域。

### ⚪ [Distance Map Penalized CE Loss](https://arxiv.org/abs/1908.03679)

距离图惩罚交叉熵损失通过由真实标签计算的[距离变换图](https://0809zheng.github.io/2023/03/22/distancetransfrom.html)对交叉熵进行加权，引导网络重点关注难以分割的边界区域。

$$
L_{D P C E}=-\frac{1}{N} \sum_{c=1}^c\left(1+D^c\right) \circ \sum_{i=1}^N g_i^c \log s_i^c
$$

其中$D^c$是类别$c$的距离惩罚项，通过取真实标签的距离变换图的倒数来生成。通过这种方式可以为边界上的像素分配更大的权重。

```python
from einops import rearrange
from scipy.ndimage import distance_transform_edt

class DisPenalizedCE(torch.nn.Module):
    def __init__(self):
        super(DisPenalizedCE, self).__init__()

    @torch.no_grad()
    def one_hot2dist(self, seg):
        res = np.zeros_like(seg)
        for c in range(seg.shape[1]):
            posmask = seg[:,c,...]
            if posmask.any():
                negmask = 1.-posmask
                pos_edt = distance_transform_edt(posmask)
                pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
                neg_edt =  distance_transform_edt(negmask)
                neg_edt = (np.max(neg_edt)-neg_edt)*negmask        
                res[:,c,...] = pos_edt + neg_edt
        return res

    def forward(self, result, gt):
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 h w')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)
        dist = torch.from_numpy(self.one_hot2dist(y_onehot.cpu().numpy())+1).float()

        result = torch.softmax(result, dim=1)
        result_logs = torch.log(result)

        loss = -result_logs * y_onehot
        weighted_loss = loss*dist
        return weighted_loss.mean()
```

### ⚪ [<font color=Blue>Boundary Loss</font>](https://0809zheng.github.io/2021/03/25/boundary.html)

在**Boundary Loss**中，每个点$q$的**softmax**输出$s_{\theta}(q)$通过$ϕ_G$进行加权。$ϕ_G:Ω→R$是真实标签边界$∂G$的水平集表示：如果$q∈G$则$ϕ_G(q)=−D_G(q)$否则$ϕ_G(q)=D_G(q)$。$D_G:Ω→R^+$是一个相对于边界$∂G$的[距离变换图](https://0809zheng.github.io/2023/03/22/distancetransfrom.html)。

$$ \mathcal{L}_B(\theta) = \int_{\Omega} \phi_G(q) s_{\theta}(q) d q $$

```python
from einops import rearrange, einsum
from scipy.ndimage import distance_transform_edt

class BDLoss(nn.Module):
    def __init__(self):
        super(BDLoss, self).__init__()

    @torch.no_grad()
    def one_hot2dist(self, seg):
        res = np.zeros_like(seg)
        for c in range(seg.shape[1]):
            posmask = seg[:,c,...]
            if posmask.any():
                negmask = 1.-posmask
                neg_map = distance_transform_edt(negmask)
                pos_map = distance_transform_edt(posmask)
                res[:,c,...] = neg_map * negmask - (pos_map - 1) * posmask
        return res

    def forward(self, result, gt):
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 h w')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        bound = torch.from_numpy(self.one_hot2dist(y_onehot.cpu().numpy())).float()
        # only compute the loss of foreground
        pc = result[:, 1:, ...]
        dc = bound[:, 1:, ...]
        multipled = pc * dc
        return multipled.mean()
```

### ⚪ [Hausdorff Distance Loss](https://arxiv.org/abs/1904.10030)

豪斯多夫距离损失通过[距离变换图](https://0809zheng.github.io/2023/03/22/distancetransfrom.html)来近似并优化真实标签和预测分割之间的[Hausdorff距离](https://0809zheng.github.io/2021/03/03/distance.html#-%E8%B1%AA%E6%96%AF%E5%A4%9A%E5%A4%AB%E8%B7%9D%E7%A6%BB-hausdorff-distance)：

$$
L_{H D}=\frac{1}{N} \sum_{c=1}^c \sum_{i=1}^N\left[\left(s_i^c-g_i^c\right)^2 \circ\left(d_{G_i^c}^{\alpha}+d_{S_i^c}^{\alpha}\right)\right]
$$

其中$d_G,d_S$分别是真实标签和预测分割的距离变换图，计算每个像素与目标边界之间的最短距离。

```python
from einops import rearrange
from scipy.ndimage import distance_transform_edt

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def one_hot2dist(self, seg):
        res = np.zeros_like(seg)
        for c in range(seg.shape[1]):
            posmask = seg[:,c,...]
            if posmask.any():
                negmask = 1.-posmask
                pos_edt = distance_transform_edt(posmask)
                neg_edt = distance_transform_edt(negmask)      
                res[:,c,...] = pos_edt + neg_edt
        return res

    def forward(self, result, gt):
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 h w')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        pred_dt = torch.from_numpy(self.one_hot2dist(result.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.one_hot2dist(y_onehot.cpu().numpy())).float()

        pred_error = (result - y_onehot) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        return dt_field.mean()
```