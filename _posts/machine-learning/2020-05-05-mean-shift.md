---
layout: post
title: 'Mean-Shift'
date: 2020-05-05
author: 郑之杰
cover: 'https://pic.downk.cc/item/5eb3bf40c2a9a83be5282b0b.jpg'
tags: 机器学习
---

> Mean-Shift Clustering.

**Mean-Shift**是基于核密度估计的聚类算法，沿着密度上升的方向寻找属于同一个簇的数据点。

算法是在特征空间上实现的，如对于**RGB**图像，定义特征空间为坐标空间和颜色空间的组合：

$$ (x, y, r, g, b) $$

通用的**Mean-Shift**算法的过程如下：（不用指定簇的数目）
1. 在未被标记的数据点中随机选择一个点作为中心$center$；
2. 找出离$center$距离在$bandwidth$之内的所有点，记做集合$M$，认为这些点属于簇$c$;
3. 以$center$为中心点，计算从$center$开始到集合$M$中每个元素的向量，将这些向量加权求和，得到$shift$向量;
4. $center = center + shift$。即$center$沿着$shift$的方向移动，移动距离是$$\mid\mid shift \mid\mid$$;
5. 重复步骤$$2,3,4$$直到$shift$向量的大小很小（接近收敛），记住此时的$center$。注意，这个迭代过程中遇到的点都应该归类到簇$c$;
6. 如果收敛时当前簇$c$的$center$与其它已经存在的簇$c2$中心的距离小于阈值，那么把$c2$和$c$合并。否则，把$c$作为新的聚类，增加$1$类;
7. 重复$$1,2,3,4,5$$直到所有的点都被标记访问。

![](https://pic.downk.cc/item/5eb3b9b0c2a9a83be5223456.jpg)

计算$shift$向量，直接计算向量和：

$$ shift = \frac{1}{k} \sum_{x_i \in S_k}^{} {(x-x_i)} $$

引入核函数$K$，使得随着样本与被偏移点的距离不同，其偏移量对均值偏移向量的贡献也不同：

$$ shift = \frac{1}{k} \sum_{x_i \in S_k}^{} {K(x,x_i)} = \frac{\sum_{i=1}^{k} {x_ig(\mid\mid \frac{x-x_i}{h} \mid\mid^2)}}{\sum_{i=1}^{k} {g(\mid\mid \frac{x-x_i}{h} \mid\mid^2)}} - x $$

其中$h$是窗口的大小$bandwidth$，$$g(x) = -K'(x)$$