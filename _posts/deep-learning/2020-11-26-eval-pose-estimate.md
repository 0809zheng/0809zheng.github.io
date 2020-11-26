---
layout: post
title: '人体姿态估计的评估指标'
date: 2020-11-26
author: 郑之杰
cover: 'https://pic.downk.cc/item/5fbf12c215e7719084dc05ab.jpg'
tags: 深度学习
---

> Pose Estimation Evaluation.

人体姿态估计中常用的评估指标包括：
1. **PCK**：**Percentage of Correct Keypoints**
2. **OKS**：**Object Keypoint Similarity**
3. **AP**：**Average Precision**
4. **mAP**：**mean Average Precision**

# 1. Percentage of Correct Keypoints
**PCK**指标衡量正确估计出的关键点比例。这是比较老的人体姿态估计指标，在$2017$年比较广泛使用，现在基本不再使用。但是在工程项目中，使用该指标评价训练模型的好坏还是蛮方便的。

第$i$个关键点的**PCK**指标计算如下：

$$ PCK_{i}^{k} = \frac{\sum_{p}^{} {\delta (\frac{d_{pi}}{d_{p}^{def}} ≤ T_k)}}{\sum_{p}^{} {1}} $$

其中：
- $p$表示第$p$个人
- $T_k$表示人工设定的阈值，$T_k \in \[0:0.01:0.1\]$
- $k$表示第$k$个阈值
- $d_{pi}$表示第$p$个人的第$i$个关键点预测值与人工标注值之间的欧氏距离
- $d_{p}^{def}$表示第$p$个人的尺度因子，不同数据集中此因子的计算方法不一样。**FLIC**数据集是以当前人的躯干直径作为尺度因子，即左肩到右臀的欧式距离或者右肩到左臀的欧式距离；**MPII**数据集是以当前人的头部直径作为尺度因子，即头部左上点与右下点的欧式距离，使用此尺度因子的姿态估计指标也称**PCKh**。
- $\delta$表示如果条件成立则为$1$，否则为$0$

算法的**PCK**指标是对所有关键点计算取平均：

$$ PCK_{mean}^{k} = \frac{\sum_{p}^{} {\sum_{i}^{} {\delta (\frac{d_{pi}}{d_{p}^{def}}} ≤ T_k)}}{\sum_{p}^{} {\sum_{i}^{} {1}}} $$

**PCK**指标计算参考代码：

```
def compute_pck_pckh(dt_kpts,gt_kpts,refer_kpts):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,k]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,k]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部对角线的欧式距离；
    :return: 相关指标
    """
    
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)
    assert(len(refer_kpts)==2)
    assert(dt.shape[0]==gt.shape[0])
    ranges=np.arange(0.0,0.1,0.01)
    kpts_num=gt.shape[2]
    ped_num=gt.shape[0]
	
    # compute dist
    scale=np.sqrt(np.sum(np.square(gt[:,:,refer_kpts[0]]-gt[:,:,refer_kpts[1]]),1))
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
	
    # compute pck
    pck = np.zeros([ranges.shape[0], gt.shape[2]+1])
    for idh,trh in enumerate(list(ranges)):
        for kpt_idx in range(kpts_num):
            pck[idh,kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= trh)
        # compute average pck
        pck[idh,-1] = 100*np.mean(dist <= trh)
    return pck
```

# 2. Object Keypoint Similarity
**OKS**是目前常用的人体骨骼关键点检测算法的评估指标，该指标受目标检测中的**IoU**指标启发，目的是计算关键点预测值和标注真值的相似度。

第$p$个人的**OKS**指标计算如下：

$$ OKS_p = \frac{\sum_{i}^{} {exp\{-d_{pi}^{2}/2S_{p}^{2} \sigma_{i}^{2}\} \delta (v_{pi} > 0)}}{\sum_{i}^{} {\delta (v_{pi} > 0)}} $$

其中：
- $i$表示第$i$个关键点
- $d_{pi}$表示第$p$个人的第$i$个关键点预测值与人工标注值之间的欧氏距离
- $S_{p}$表示第$p$个人的尺度因子，其值为行人检测框面积的平方根$S_{p}=\sqrt{wh}S，$w$、$h$为检测框的宽和高
- $\sigma_{i}$表示第$i$个关键点的归一化因子，该因子是通过对所有的样本集中关键点由人工标注与真实值存在的标准差，$\sigma$越大表示此类型的关键点越难标注。对**COCO**数据集中的$5000$个样本统计出$17$类关键点的归一化因子，取值为：{**鼻子：0.026，眼睛：0.025，耳朵：0.035，肩膀：0.079，手肘：0.072，手腕：0.062，臀部：0.107，膝盖：0.087，脚踝：0.089**}，此值可以看作常数，如果使用的关键点类型不在此当中，则需要统计方法计算
- $v_{pi}$表示第$p$个人的第$i$个关键点的可见性，对于人工标注值，$v_{pi}=0$表示关键点未标记（图中不存在或不确定在哪里），$v_{pi}=1$表示关键点无遮挡且已标注，$v_{pi}=2$关键点有遮挡但已标注。对于预测关键点，$v_{pi}'=0$表示没有预测出，$v_{pi}'=1$表示预测出
- $\delta$表示如果条件成立则为$1$，否则为$0$

**OKS**指标计算参考代码：

```
sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
variances = (sigmas * 2)**2
def compute_kpts_oks(dt_kpts, gt_kpts, area):
    """
    this function only works for computing oks with keypoints，
    :param dt_kpts: 关键点检测结果　dt_kpts.shape=[3,k],dt_kpts[0]表示横坐标值，dt_kpts[1]表示纵坐标值，dt_kpts[2]表示可见性，
    :param gt_kpts:　关键点标记结果　gt_kpts.shape=[3,k],gt_kpts[0]表示横坐标值，gt_kpts[1]表示纵坐标值，gt_kpts[2]表示可见性，
    :param area:　groundtruth中当前一组关键点所在人检测框的面积
    :return:　两组关键点的相似度oks
    """
	
    g = np.array(gt_kpts)
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    assert(np.count_nonzero(vg > 0) > 0)
    d = np.array(dt_kpts)
    xd = d[0::3]
    yd = d[1::3]
    dx = xd - xg
    dy = yd - yg
    e = (dx**2 + dy**2) /variances/ (area+np.spacing(1)) / 2　#加入np.spacing()防止面积为零
    e=e[vg > 0]
    return np.sum(np.exp(-e)) / e.shape[0]
```

# 3. Average Precision
对于**单人姿态估计**，首先计算**OKS**指标，然后人为给定一个阈值$T$，通过所有图像计算**AP**指标：

$$ AP = \frac{\sum_{p}^{} {\delta (oks_p) > T}}{\sum_{p}^{} {1}} $$

对于**多人姿态估计**，如果采用的检测方法是**自顶向下**，先把所有的人找出来再检测关键点，那么其**AP**计算方法同上；

如果采用的检测方法是**自底向上**，先把所有的关键点找出来再组成人，假设一张图片中共有$M$个人，预测出$N$个人，由于不知道预测出的$N$个人与标记的$M$个人之间的对应关系，因此需要计算标记的每个人与预测的$N$个人的**OKS**指标，得到一个大小为${M}\times{N}$的矩阵，矩阵的每一行为标记的一个人与预测结果的$N$个人的**OKS**指标，然后找出每一行中**OKS**指标最大的值作为当前标记人的**OKS**指标。最后每一个标记人都有一个**OKS**指标，然后人为给定一个阈值$T$，通过所有图像计算**AP**指标：

$$ AP = \frac{\sum_{m}^{} \sum_{p}^{} {\delta (oks_p) > T}}{\sum_{m}^{} \sum_{p}^{} {1}} $$

# 4. mean Average Precision
**mAP**是给**AP**指标中的人工阈值$T$设定不同的值，对这些阈值下得到的**AP**求平均得到的结果。

$$ T \in [0.5:0.05:0.95] $$
