---
layout: post
title: '提高非极大值抑制算法的效率'
date: 2021-05-11
author: 郑之杰
cover: ''
tags: 深度学习
---

> Improve efficiency of the NMS algorithm.

**非极大值抑制(non-maximum suppression,NMS)**算法是目标检测等任务中常用的后处理方法，能够过滤掉多余的检测边界框。本文介绍**NMS**算法及其计算效率较低的原因，并介绍一些能够提高**NMS**算法效率的算法。目录如下：

1. **NMS**算法
2. **CUDA NMS**
3. **Fast NMS**
4. **Cluster NMS**
5. **Matrix NMS**

# 1. NMS算法
**NMS**算法的流程如下：
- 输入边界框集合$$\mathcal{B}=\{(B_n,c_n)\}_{n=1,...,N}$$，其中$c_n$是边界框$B_n$的置信度；
- 选中集合$$\mathcal{B}$$中置信度最大的边界框$B_i$，将其从集合$$\mathcal{B}$$移动至输出边界框集合$$\mathcal{O}$$中；
- 遍历集合$$\mathcal{B}$$中的其余所有边界框$B_j$，计算边界框$B_i$和边界框$B_j$的交并比$\text{IoU}(B_i,B_j)$。若$\text{IoU}(B_i,B_j)≥\text{threshold}$，则删除边界框$B_j$；
- 重复上述步骤，直至集合$$\mathcal{B}$$为空集。

上述算法存在一些问题，如下：
1. 算法采用顺序处理的模式，运算效率低；
2. 根据阈值删除边界框的机制缺乏灵活性；
3. 阈值通常是人工根据经验选定的；
4. 评价标准是交并比**IoU**，只考虑两框的重叠面积。

# 2. CUDA NMS
**CUDA NMS**是**NMS**的**GPU**版本，旨在将**IoU**的计算并行化，并通过矩阵运算加速。

**NMS**中的**IoU**计算是顺序处理的。假设图像中一共有$N$个检测框，每一个框都需要和其余所有框计算一次**IoU**，则计算复杂度是$O(\frac{N(N-1)}{2})=O(N^2)$。

若将边界框集合$$\mathcal{B}$$按照置信度得分从高到低排序，即$B_1$是得分最高的框，$B_N$是得分最低的框；则可以计算**IoU**矩阵：

$$ X=\text{IoU}(B,B)= \begin{pmatrix} x_{11} & x_{12} & ... & x_{1N} \\ x_{21} & x_{22} & ... & x_{2N} \\ ... & ... & ... & ... \\ x_{N1} & x_{N2} & ... & x_{NN} \\ \end{pmatrix} , \quad x_{ij}=\text{IoU}(B_i,B_j) $$

通过**GPU**的并行加速能力，可以一次性得到**IoU**矩阵的全部计算结果。

下面是**CUDA NMS**计算**IoU**矩阵的一种简单实现。许多深度学习框架也已将**CUDA NMS**作为基本函数使用，如**Pytorch**在**torchvision 0.3**版本中正式集成了**CUDA NMS**。

```
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
```

计算得到**IoU**矩阵后，需要利用它抑制冗余框。可以采用矩阵查询的方法（仍然需要顺序处理，但计算**IoU**本身已经被并行加速）；也可以使用下面提出的一些算法。

# 3. Fast NMS
- paper：[YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

根据**IoU**矩阵的计算规则可以得出$\text{IoU}(B_i,B_j)=\text{IoU}(B_j,B_i)$，且计算$\text{IoU}(B_i,B_i)$是没有意义的。因此**IoU**矩阵$X$是对称矩阵。**Fast NMS**算法首先对矩阵$X$使用**pytorch**提供的`triu`函数进行上三角化，得到上三角矩阵：

$$ X=\text{IoU}(B,B)= \begin{pmatrix} 0 & x_{12} & ... & x_{1N} \\ 0 & 0 & ... & x_{2N} \\ ... & ... & ... & ... \\ 0 & 0 & ... & 0 \\ \end{pmatrix} $$

若按照**NMS**的规则，应该按行依次遍历矩阵$X$，如果某行$i$中元素$x_{ij}=\text{IoU}(B_i,B_j),j＞i$超过阈值，则应剔除边界框$B_j$，且不再考虑$j$所对应的行与列。

**Fast NMS**则对上述规则进行了化简。对矩阵$X$执行按列取最大值的操作，得到一维向量$b=\[b_1,b_2,...,b_N\]$，$b_n$代表矩阵$X$的第$n$列中元素的最大值。然后使用阈值二值化，$b$中元素小于阈值对应保留的边界框，$b$中元素大于阈值对应冗余框。

**Fast NMS**的思路是只要边界框$B_j$与任意边界框$B_i$重合度较大(超过阈值)，则认为其是冗余框，将其剔除。这样做容易删除更多边界框，因为假如边界框$B_j$与边界框$B_i$重合度较大，但边界框$B_i$已经被剔除，则边界框$B_j$还是有可能会被保留的。一个简单的例子如下，注意到边界框$B_4$被错误地剔除了。

$$ X= \begin{pmatrix} 0 & 0.6 & 0.1 & 0.3 & 0.8 \\   & 0 & 0.2 & 0.72 & 0.1 \\   &   & 0 & 0.45 & 0.12 \\   &   &   & 0 & 0.28 \\   &   &   &   & 0 \\ \end{pmatrix} \\ (\text{按列取最大值}) \\ b = [0,0.6,0.2,0.72,0.8] \\ (\text{选定阈值为0.5}) \\ b = [1,0,1,0,0] \\ (\text{保留边界框1和边界框3}) $$

使用**pytorch**实现**Fast NMS**算法如下：

```
def fast_nms(self, boxes, scores, NMS_threshold:float=0.5):
    '''
    Arguments:
        boxes (Tensor[N, 4])
        scores (Tensor[N, 1])
    Returns:
        Fast NMS results
    '''
    scores, idx = scores.sort(1, descending=True)
    boxes = boxes[idx]   # 对框按得分降序排列
    iou = box_iou(boxes, boxes)  # IoU矩阵
    iou.triu_(diagonal=1)  # 上三角化
    keep = iou.max(dim=0)[0] < NMS_threshold  # 列最大值向量，二值化

    return boxes[keep], scores[keep]
```

**Fast NMS**算法比**NMS**算法运算速度更快，但由于其会抑制更多边界框，会导致性能略微下降。

# 4. Cluster NMS
- paper：[Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/2005.03572)

**Cluster NMS**算法旨在弥补**Fast NMS**算法性能下降的问题，同时保持较快的运算速度。

定义边界框的**cluster**，若边界框$B_i$属于该**cluster**，则边界框$B_i$与集合内任意边界框$B_j$的交并比$\text{IoU}(B_i,B_j)$均超过阈值，且边界框$B_i$与不属于该集合的任意边界框$B_k$的交并比$\text{IoU}(B_i,B_k)$均不低于阈值。通过定义**cluster**将边界框分成不同的簇，如下图可以把边界框分成三组**cluster**：

![](https://pic.imgdb.cn/item/609b797bd1a9ae528fb1bd9f.jpg)

**Cluster NMS**算法本质上是**Fast NMS**算法的迭代式。算法前半部分与**Fast NMS**算法相同，都是按降序排列边界框、计算**IoU**矩阵、矩阵上三角化、按列取最大值、阈值二值化得到一维向量$b$。不同于**Fast NMS**算法直接根据向量$b$输出结果，**Cluster NMS**算法将向量$b$按对角线展开为对角矩阵，将其左乘**IoU**矩阵。然后再对新的矩阵按列取最大值、阈值二值化，得到新的向量$b$，再将其按对角线展开后左乘**IoU**矩阵，直至两次迭代中向量$b$不再变化。

![](https://pic.imgdb.cn/item/609b7a55d1a9ae528fb87c53.jpg)

矩阵左乘相当于进行**行变换**。向量$b$的对角矩阵左乘**IoU**矩阵，若$b$的第$n$项为$0$，代表对应的边界框$B_n$是冗余框，则不应考虑该框对其他框产生的影响，因此将**IoU**矩阵的第$n$行置零；反之若$b$的第$n$项为$1$，代表对应的边界框$B_n$不是冗余框，因此保留**IoU**矩阵的第$n$行。由数学归纳法可证，**Cluster NMS**算法的收敛结果与**NMS**算法相同。

使用**pytorch**实现**Cluster NMS**算法如下：

```
def cluster_nms(self, boxes, scores, NMS_threshold:float=0.5, epochs:int=200):
    '''
    Arguments:
        boxes (Tensor[N, 4])
        scores (Tensor[N, 1])
    Returns:
        Fast NMS results
    '''
    scores, idx = scores.sort(1, descending=True)
    boxes = boxes[idx]   # 对框按得分降序排列
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    C = iou
    for i in range(epochs):    
        A=C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float().unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    keep = maxA < NMS_threshold  # 列最大值向量，二值化

    return boxes[keep], scores[keep]
```

**NMS**算法顺序处理每一个边界框，会在所有**cluster**上迭代，在计算时重复计算了不同**cluster**之间的边界框。**Cluster NMS**算法通过行变换使得迭代进行在拥有框数量最多的**cluster**上，其迭代次数不超过图像中最大**cluster**所拥有的**边界框个数**。因此**Cluster NMS**算法适合图像中有很多**cluster**的场合。

实践中又提出了一些**Cluster NMS**的变体：

### (1) 引入得分惩罚机制
**得分惩罚机制(score penalty mechanism, SPM)**是指每次迭代后根据计算得到的**IoU**矩阵对边界框的置信度得分进行惩罚，即与该边界框重合度高的框越多，该边界框的置信度越低：

$$ c_j = c_j \cdot \prod_{i}^{} e^{-\frac{x_{ij}^2}{\sigma}} $$

每轮迭代后，需要根据置信度得分对边界框顺序进行重排，使用**pytorch**实现如下：

```
def SPM_cluster_nms(self, boxes, scores, NMS_threshold:float=0.5, epochs:int=200):
    '''
    Arguments:
        boxes (Tensor[N, 4])
        scores (Tensor[N, 1])
    Returns:
        Fast NMS results
    '''
    scores, idx = scores.sort(1, descending=True)
    boxes = boxes[idx]   # 对框按得分降序排列
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    C = iou
    for i in range(epochs):    
        A=C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float().unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    scores = torch.prod(torch.exp(-C**2/0.2),0)*scores    #惩罚得分
    keep = scores > 0.01    #得分阈值筛选
    return boxes[keep], scores[keep]
```

### (2) 引入中心点距离
将交并比**IoU**替换成**DIoU**，即在**IoU**的基础上加上中心点的归一化距离，能够更好的表达两框的距离，计算如下：

$$ \text{DIoU} = \text{IoU} - R_{DIoU} $$

其中惩罚项$R_{DIoU}$设置为：

$$ R_{DIoU} = \frac{ρ^2(b_{pred},b_{gt})}{c^2} $$

其中$b_{pred}$,$b_{gt}$表示边界框的中心点，$ρ$是欧式距离，$c$表示最小外接矩形的对角线距离，如下图所示：

![](https://img.imgdb.cn/item/6017d99c3ffa7d37b3ec6a3e.jpg)

将计算**IoU**矩阵替换为**DIoU**矩阵，并对得分惩罚机制进行修改：

$$ c_j = c_j \cdot \prod_{i}^{} \mathop{\min} \{ e^{-\frac{x_{ij}^2}{\sigma}} + (1-\text{DIoU})^{\beta}, 1 \} $$

上式中$\beta$用于控制中心点距离惩罚的程度，$min$避免惩罚因子超过$1$。

### (3) 加权平均法（weighted NMS）
在计算**IoU**矩阵时考虑边界框置信度得分的影响，即每次迭代时将**IoU**矩阵与边界框的置信度得分向量按列相乘。

使用**pytorch**实现如下：

```
def Weighted_cluster_nms(self, boxes, scores, NMS_threshold:float=0.5, epochs:int=200):
    '''
    Arguments:
        boxes (Tensor[N, 4])
        scores (Tensor[N, 1])
    Returns:
        Fast NMS results
    '''
    scores, idx = scores.sort(1, descending=True)
    n = scores.shape[0]
    boxes = boxes[idx]   # 对框按得分降序排列
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    C = iou
    for i in range(epochs):    
        A=C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float().unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    keep = maxA < NMS_threshold  # 列最大值向量，二值化
    weights = (C*(C>0.8).float() + torch.eye(n).cuda()) * (scores.reshape((1,n)))
    xx1 = boxes[:,0].expand(n,n)
    yy1 = boxes[:,1].expand(n,n)
    xx2 = boxes[:,2].expand(n,n)
    yy2 = boxes[:,3].expand(n,n)

    weightsum=weights.sum(dim=1)         # 坐标加权平均
    xx1 = (xx1*weights).sum(dim=1)/(weightsum)
    yy1 = (yy1*weights).sum(dim=1)/(weightsum)
    xx2 = (xx2*weights).sum(dim=1)/(weightsum)
    yy2 = (yy2*weights).sum(dim=1)/(weightsum)
    boxes = torch.stack([xx1, yy1, xx2, yy2], 1)
    return boxes[keep], scores[keep]
```

# 5. Matrix NMS
- paper：[SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152)

**Matrix NMS**算法与**Fast NMS**算法的思想是一样的，不同之处在于前者针对的是图像分割中的**mask IoU**。其伪代码如下：

![](https://pic.imgdb.cn/item/609b8b94d1a9ae528f3b9f42.jpg)
