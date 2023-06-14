---
layout: post
title: '目标检测(Object Detection)'
date: 2020-05-08
author: 郑之杰
cover: ''
tags: 深度学习
---

> Object Detection.

**目标检测(Object Detection)**任务是指在图像中检测出可能存在的目标；包括**定位**和**分类**两个子任务：其中定位是指确定目标在图像中的具体位置，分类是确定目标的具体类别。

本文目录：
1. 传统的目标检测算法
2. 基于深度学习的目标检测算法
3. 目标检测的评估指标
4. 非极大值抑制算法

# 1. 传统的目标检测算法

传统的目标检测算法主要有三个步骤，
1. 在图像中生成候选区域(**proposal region**);
2. 对每个候选区域提取特征向量，这一步通常是用人工精心设计的特征描述子(**feature descriptor**)提取图像的特征;
3. 对每个候选区域提取的特征进行分类，确定对应的类别。


## （1）生成候选区域

常用的候选区域生成方法包括滑动窗口、**Felzenszwalb**算法、选择搜索算法。

### ⚪ 滑动窗口 Sliding Window

候选区域通常是由**滑动窗口**实现的；结合不同尺度的图像以及不同尺度的窗口可生成大量的候选区域。

### ⚪ Felzenszwalb算法

- paper：[Efficient graph-based image segmentation](http://cvcl.mit.edu/SUNSeminar/Felzenszwalb_IJCV04.pdf)

**Felzenszwalb**算法通过基于图的方法把图像分割成一些相似的区域。使用无向图$G=(V,E)$表示输入图像，其中的一个节点$v_i \in V$代表一个像素，一条边$e_{ij} = (v_i,v_j) \in V$连接两个节点$i,j$，每条边$e_{ij}$有一个权重$w_{ij}$衡量两个节点$i,j$的不相似程度（可以通过颜色、位置、强度值来衡量）。则一个分割结果$S$是把节点$V$分配到多个内部相互连接的子集$$\{C\}$$中。相似的像素应该属于通过一个子集，不相似的像素被分配到不同的子集。

构造一个图像的图的方法有两种：
- **网格图 (Grid Graph)**：每个像素只和其邻域内的像素连接（比如周围的$8$个像素），边的连接权重是像素的强度值之差的绝对值。
- **最近邻图 (Nearest Neighbor Graph)**：把每个像素表示为位置+颜色特征空间中的一个点$(x,y,r,g,b)$，两个像素之间的边权重是像素特征向量的欧氏距离。

首先定义以下几个概念：
- **内部差异 (Internal difference)**：$Int(C)=\max_{e \in MST(C,E)} w(e)$。其中**MST**表示最小生成树。当移除所有权重小于$Int(C)$的边时，集合$C$仍然保持连接。
- 两个集合之间的**差异 (difference)**：$Diff(C_1,C_2)=\min_{v_i\in C_1,v_j \in C_2,(v_i,v_j) \in E} w(v_i,v_j)$。若两个集合之间没有边连接，则定义$Diff(C_1,C_2)=\infty$。
- **最小内部差异 (Minimum internal difference)**：$MInt(C_1,C_2)=\min(Int(C_1)+k/\|C_1\|,Int(C_2)+k/\|C_2\|)$。阈值$k$越大则则倾向于选择较大的子集。

给定两个区域$C_1,C_2$，则当下列判据成立时才把这两个区域视为两个独立的子集；否则分割过于细致，应该把两个区域合并。

$$
D(C_1,C_2) =
\begin{cases}
True & Diff(C_1,C_2) > MInt(C_1,C_2) \\
False & Diff(C_1,C_2) \leq MInt(C_1,C_2)
\end{cases}
$$

**Felzenszwalb**算法采用自底向上的处理流程。给定具有$\|V\|=n$个节点和$\|E\|=m$条边的无向图$G=(V,E)$：
- 把边按照权重升序排列$e_1,e_2,...,e_m$；
- 初始化时把每个像素看作一个单独的区域，则一共有$n$个区域；
- 重复以下操作$k=1,...,m$： 
1. 第$k-1$步的分割结果为$S^{k-1}$；
2. 选取排序后的第$k$条边$e_k=(v_i,v_j)$；
3. 如果$v_i,v_j$在分割$S^{k-1}$中属于同一个区域，则直接令$S^k=S^{k-1}$；
4. 如果$v_i,v_j$在分割$S^{k-1}$中属于不同的区域$C_i^{k-1},C_j^{k-1}$，若$w(v_i,v_j) \leq MInt(C_i^{k-1},C_j^{k-1})$则将两个区域合并；否则不做处理。

```python
import scipy
import skimage.segmentation

img = scipy.misc.imread("image.jpg", mode="L")
segment_mask = skimage.segmentation.felzenszwalb(img, scale=100)
```

### ⚪ 选择搜索算法 Selective Search

选择搜索算法是一种常用的提取潜在目标的区域提议算法。该算法建立在图像分割结果上（如**Felzenszwalb**算法的输出），并使用基于区域的特征执行自底向上的层次划合并。选择搜索算法的工作流程如下：
- 初始化阶段，使用**Felzenszwalb**算法生成图像的分割区域集合；
- 使用贪心算法迭代地执行区域合并：
1. 首先计算所有相邻区域之间的相似度；
2. 两个最相似的区域被合并为一组，然后计算新的区域和其邻域区域之间的相似度。
- 重复执行相似区域之间的合并，直至整个图像变成单个区域。
- 上述过程中产生的所有区域都被视为可能存在目标的区域。

在评估两个区域之间的相似性时，采用四种互补的相似性度量：
- 颜色
- 纹理：通过**SIFT**算子提取特征
- 尺寸：鼓励较小的区域尽早合并
- 形状：理想情况下，一个区域可以填补另一个区域的空白

通过调整**Felzenszwalb**算法的阈值$k$、改变颜色空间、选择不同的相似性度量组合，可以制定一套多样化的选择搜索策略。能够产生具有最高质量的提议区域的选择搜索策略为：不同初始分割结果的混合+多种颜色空间的混合+所有相似性度量的组合。算法也需要在质量（模型复杂性）和速度之间取得平衡。

## （2）设计特征描述子

常用的特征描述子包括图像梯度向量、方向梯度直方图**HOG**、尺度不变特征变换**SIFT**、可变形部位模型**DPM**。

### ⚪ 图像梯度向量 Image Gradient Vector

**图像梯度向量**定义为每个像素沿着$x$轴和$y$轴的像素颜色变化度量。若$f(x,y)$表示像素位置$(x,y)$处的颜色，则像素$(x,y)$的梯度向量定义为：

$$
\nabla f(x,y) = \begin{bmatrix} g_x \\ g_y \end{bmatrix}= \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}= \begin{bmatrix} f(x+1,y)-f(x-1,y) \\ f(x,y+1)-f(x,y-1) \end{bmatrix}
$$

梯度向量的幅值(**magnitude**) $g$定义为向量的**L2**范数；梯度向量的方向(**direction**) $\theta$定义为两个偏导数方向之间的夹角。

$$
g = \sqrt{g_x^2+g_y^2}, \quad \theta = \arctan(\frac{g_y}{g_x})
$$

![](https://pic.imgdb.cn/item/6480463d1ddac507ccd73a8e.jpg)

如上图中像素$(x,y)$的梯度向量为：

$$
\nabla f = \begin{bmatrix} f(x+1,y)-f(x-1,y) \\ f(x,y+1)-f(x,y-1) \end{bmatrix} = \begin{bmatrix} 55-105 \\ 90-40 \end{bmatrix} = \begin{bmatrix} -50 \\ 50 \end{bmatrix}
$$


### ⚪ 方向梯度直方图 Histogram of Oriented Gradients

- paper：[Histograms of oriented gradients for human detection](https://inria.hal.science/file/index/docid/548512/filename/hog_cvpr2005.pdf)


**方向梯度直方图 (HOG)** 的构造过程如下：
1. 预处理图像，包括尺寸调整和像素值归一化；
2. 计算每个像素的梯度向量，并进一步计算其幅值和方向；
3. 把图像划分成一系列$8\times 8$的图像块。对于每个图像块，把其中$64$个像素梯度向量的幅值累加到直方图的$9$个桶中，直方图根据梯度向量的方向绝对值($0-180°$)经验性地划分成$9$个均匀的桶；
4. 如果一个像素的梯度向量的方向恰好位于两个桶之间，则将其按比例划分到两个桶中；比如大小为$8$的梯度向量方向为$15°$，则分别把$6$和$2$划分到$0°$桶和$20°$桶中；这种分配方式对图像较小的失真具有鲁棒性；
5. 在图像上滑动$2\times 2$个图像块（对应$16\times 16$像素）。每个滑动窗口中的$4$个直方图被连接后进行归一化。最终的**HOG**特征向量是所有滑动窗口向量的级联。
6. **HOG**特征向量可以被输入到分类器中，用于学习目标识别任务。

![](https://pic.imgdb.cn/item/648056fe1ddac507ccf358a7.jpg)


### ⚪ 尺度不变特征变换 Scale Invariant Feature Transform

- paper：[Object Recognition from Local Scale-Invariant Features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)



**尺度不变特征变换 (SIFT)**是一种非常稳定的局部特征，具有旋转不变性、尺度不变性、亮度变化保持不变性。**SIFT**特征的构造过程如下：
1. **DoG尺度空间的极值检测**：为了使算子具有**尺度不变性**，先构造高斯差分尺度空间 **(Difference of Gaussina, DOG)**：通过不同$\sigma$值的高斯滤波和降采样构造图像的高斯尺度空间，再通过每组高斯空间中的相邻图像相减得到**DoG**图像。对于**DoG**中的每个像素点，将其与所在图像$3×3$邻域$8$个像素点以及同一尺度空间中上下两层图像$3×3$邻域$18$个像素点进行比较。当其值大于（或者小于）所有比较点时，该点为候选极值点；![](https://pic.imgdb.cn/item/648061cd1ddac507cc0321b5.jpg)
2. **删除不稳定的极值点**：对**DoG**尺度空间函数进行二次泰勒展开求极值点，去掉局部曲率非常不对称的点。剔除的点主要有两种：低对比度的特征点和不稳定的边缘响应点；
3. **确定特征点的主方向**：计算以特征点为中心的邻域内各个像素点的**HOG**直方图（以$45°$划分直方图，则共有$8$个桶），直方图中最高峰所对应的方向即为特征点的方向。![](https://pic.imgdb.cn/item/648064cf1ddac507cc073bb2.jpg)
4. **生成特征点的描述子**：为了使算子具有**旋转不变性**，首先将坐标轴旋转为特征点的方向。以特征点为中心划分$4×4$的图像块，每个图像块包括$4×4$像素。对每个图像块构造**HOG**直方图（长度为$8$的向量），一个特征点可以产生$128$维的**SIFT**特征向量。为了使算子具有**亮度变化保持不变性**，对**SIFT**特征向量进行归一化处理。![](https://pic.imgdb.cn/item/648067111ddac507cc09a287.jpg)


### ⚪ Deformable Parts Model (DPM)

- paper：[Cascade object detection with deformable part models](https://ieeexplore.ieee.org/document/5539906/)

可变形部位模型(**DPM**)使用可变形部位的混合图模型（马尔可夫随机场）来识别目标。**DPM**主要由三个部分组成：
- 一个**根滤波器(root filter)**近似定义覆盖整个目标的检测窗口，并指定区域特征向量的权重；
- 多个**部位滤波器(part filter)**覆盖目标的较小部位，其学习分辨率设置为根滤波器的两倍。
- 一个**空间模型(spatial model)**对部位滤波器相对于根滤波器的位置进行评分。

![](https://pic.imgdb.cn/item/648660ba1ddac507ccbac9a5.jpg)

检测目标的质量是通过滤波器的得分减去变形成本来衡量的。令$x$为输入图像，$y$是$x$的一个子区域，$\beta_{root},\beta_{part}$分别代表根滤波器和部位滤波器，代价函数$cost()$衡量部位偏离其相对于根的理想位置的惩罚。则模型匹配图像$x$的得分$f()$计算为：

$$
f(model, x) = f(\beta_{root},x) + \sum_{\beta_{part}} \max_y [f(\beta_{part},y)-cost(\beta_{part},x,y)]
$$

得分模型$f()$通常设置为滤波器$\beta$与区域特征向量$\Phi(x)$的点积：$f(\beta,x) = \beta\cdot \Phi(x)$。区域特征向量$\Phi(x)$可以由**HOG**算子构造。根滤波器中得分较高的位置检测出包含目标可能性高的区域，而部位滤波器中得分较高的位置证实了已识别的物体假设。





# 2. 基于深度学习的目标检测模型

在传统的方法中，经常会使用集成、串联学习、梯度提升等方法来提高目标检测的准确率；但是传统的方法逐渐暴露出很多问题，比如检测准确率有限、需要人工设计特征描述子等。近些年来深度学习的引入使得目标检测的精度和速度有了很大的提升，[卷积神经网络](https://0809zheng.github.io/2020/03/06/CNN.html)能够提取图像的深层语义特征，省去了人工设计和提取特征的步骤。

目前主流的目标检测模型分成两类。
- **两阶段（Two-Stage）**的目标检测模型：首先在图像中生成可能存在目标的候选区域，然后对这些候选区域进行预测。这些方法精度高，速度相对慢一些；
- **单阶段（One-Stage）**的目标检测模型：把图像中的每一个位置看作潜在的候选区域，直接进行预测。这些方法速度快，精度相对低一些。

![](https://pic.downk.cc/item/5facf3b81cd1bbb86b4e1145.jpg)

上图是目前大部分目标检测模型的主要流程图。一个目标检测系统主要分成三部分，如图中的**backbone**、**neck**和**regression**部分。
1. **backbone**部分通常是一个卷积网络，把图像转化成对应的**特征映射(feature map)**；
2. **neck**部分通常是对特征映射做进一步的增强处理；
3. **regression**部分通常把提取的特征映射转换成**边界框(bounding box)**和**类别(class)**信息。

单阶段的目标检测模型直接在最后的特征映射上进行预测；而两阶段的方法先在特征映射上生成若干候选区域，再对候选区域进行预测。由此可以看出单阶段的方法所处理的候选区域是**密集**的；而两阶段的方法由于预先筛选了候选区域，最终处理的候选区域相对来说是**稀疏**的。

下面介绍一些常用的目标检测模型：
- 两阶段的目标检测模型：
- 单阶段的目标检测模型：**OverFeat**
- 基于Transformer的目标检测模型

## （1）两阶段的目标检测模型

### ⚪ R-CNN
- paper：[<font color=blue>Rich feature hierarchies for accurate object detection and semantic segmentation</font>](https://0809zheng.github.io/2021/03/01/rcnn.html)

**R-CNN**首先应用选择搜索算法提取感兴趣区域，对于每个区域进行尺寸调整后通过预训练的卷积神经网络提取特征向量，并通过二元支持向量机对每个预测类别进行二分类。为进一步提高检测框的定位精度，训练一个回归器进行边界框的位置和尺寸修正。

![](https://pic.imgdb.cn/item/648678031ddac507ccd6f5f5.jpg)

### ⚪ Fast R-CNN
- paper：[<font color=blue>Fast R-CNN</font>](https://0809zheng.github.io/2021/03/07/fastrcnn.html)

**Fast R-CNN**是对**R-CNN**的改进：
- **Fast R-CNN**首先将图像通过卷积网络提取特征映射，在原始图像中生成的候选区域被投影到在特征映射的对应位置；
- **Fast R-CNN**把特征区域的各向异性放缩替换为**RoI Pooling**层，即通过最大池化把候选区域的特征映射转换成固定尺寸的特征。
- **Fast R-CNN**把用于目标分类的分类损失和用于目标边界框坐标修正的回归损失结合起来一起训练。

![](https://pic.imgdb.cn/item/6486bcbd1ddac507cc5ff741.jpg)


### ⚪ Faster R-CNN
- paper：[<font color=blue>Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks</font>](https://0809zheng.github.io/2021/03/09/fasterrcnn.html)

**Faster R-CNN**改进了**R-CNN**和**Fast RCNN**中的候选区域生成方法，使用**区域提议网络(Region Proposal Network, RPN)**代替了选择搜索算法。**RPN**在预训练卷积网络输出特征映射每个元素对应的输入像素位置预设一系列具有不同尺寸和长宽比的**anchor**，并进行**anchor**分类和边界框回归；只有正类**anchor**被视为可能存在目标的**proposals**，并通过**RoI Pooling**和预测头执行目标分类和目标边界框坐标修正。

![](https://pic.imgdb.cn/item/6486c8f51ddac507cc862f79.jpg)



### 2. SPPNet

![](https://pic.downk.cc/item/5facf6701cd1bbb86b4ef017.jpg)

由于预测的卷积网络需要固定尺寸的输入，**R-CNN**处理的方法是直接对特征映射进行各向异性的缩放，这一过程会引入失真和畸变，损失了图像的部分文本信息。

**SPPNet**提出了一种**空间金字塔池化(Spatial Pyramid Pooling)**层，对于任意大小的输入尺寸，均可形成相同长度的输出特征向量，且这个长度是预先定义的，与输入图像的尺寸无关。当给定一个输入特征映射时，把这个映射划分成$s×s$个子区域，如果不能整除就做近似。对每一个子区域应用最大池化可以得到一个标量，那么最终就能把输入转化成一个向量。当选择的$s$不同时，所得向量的长度也不同。在这篇文章中$s$取$1$、$2$、$4$，最终得到$21$维的向量。




### 5. FPN

![](https://pic.downk.cc/item/5facf8ae1cd1bbb86b4f97df.jpg)

之前的几个网络都是将图像喂入网络得到对应的特征向量。卷积网络的浅层映射特征通常具有较强的空间信息，据具有较高的分辨率，适合检测小目标；而深层的特征映射具有较强的语义信息，具有范围较大的感受野，适合检测较大的目标。

**特征金字塔网络(Feature Pyramid Network, FPN)**就是将卷积网络的浅层映射和深层映射结合起来，首先通过前向传播得到由浅入深的特征映射，再将深层的特征映射通过转置卷积增加尺寸，并结合每一层特征映射的信息，最终可以得到一个空间信息和语义信息都很丰富的特征映射。将该特征映射用于下游的任务，最终能够提高目标检测的准确率。

### 6. Cascade R-CNN

![](https://pic.downk.cc/item/5facf96f1cd1bbb86b4fcd81.jpg)

网络在检测目标时会设置一个**交并比阈值(IoU Threshold)**，当预测的边界框和真实目标框的交并比超过该阈值时，才认为边界框检测到了目标。通常该阈值设置越高，能够检测出的边界框越准确，但是由于提高了阈值，导致正样本的数量呈指数级降低，容易过拟合。在预测时，该阈值选取得不同会导致候选区域的样本分布发生变化，从而影响最终的结果。

**Cascade R-CNN**是一种串联的两阶段目标检测算法，串联网络的思想是使用不同交并比的阈值训练多个级联的检测器，在原文中作者使用了四个网络，第一个网络提取特征映射，之后分别使用阈值为$0.5$、$0.6$、$0.7$的检测网络，通过这样一种串联的学习获得了较高的目标检测精度。

## （2）单阶段的目标检测模型

### ⚪ OverFeat
- paper：[OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229)

**OverFeat**使用同一个卷积神经网络同时执行目标检测、定位和分类任务，其主要思想是：构建一个卷积神经网络（采用**AlexNet**结构：作为特征提取器的全卷积网络+作为分类器的全连接层），以滑动窗口的方式在不同尺度的图像的不同区域位置上进行图像分类，然后通过把分类器调整为回归器进行边界框位置预测。

**OverFeat**的训练流程：
1. 在图像分类任务上训练卷积神经网络；
2. 冻结特征提取器，把分类器替换为回归器，在每个空间位置和尺度上预测每个类别边界框的坐标$(x_{left},x_{right},y_{top},y_{bottom})$。

**OverFeat**的检测流程：
1. 使用预训练的卷积网络在每个位置上执行分类；
2. 对于生成的所有已分类区域预测目标边界框；
3. 合并重叠的边界框以及可能来自同一个目标的边界框。

注意到全卷积网络可以并行地以滑动窗口的方式提取图像特征。**OverFeat**采用$5\times 5$卷积核实现滑动窗口，相当于在输入图像上设置$14 \times 14$的窗口。

![](https://pic.imgdb.cn/item/648672f01ddac507cccfcec2.jpg)




### 1. YOLO系列

![](https://pic.downk.cc/item/5facfa001cd1bbb86b50019d.jpg)

**YOLO**算法可能是目前最知名的单阶段目标检测算法，它的基本思想是用卷积网络实现滑动窗口。

当一张图像喂入卷积网络后，可以得到尺寸缩小的特征映射，比如$7×7$的映射。映射的每一个子区域都能对应到原图像中的一个子区域，假设原图像的这个子区域内含有目标，则通过网络把相关信息编码到特征映射的对应区域上。

在原始的网络中，每一个子区域预设一些边界框用来检测该区域可能出现的目标，由此可以看出，单阶段的检测方法在每个子区域都会预测很多边界框，因此所处理的候选区域是非常密集的，所以会出现大量的负样本，造成目标检测中正负样本的比例极其不均衡，这也是影响单阶段目标检测算法的主要问题。

**YOLOv2**相比于**YOLO**进行了很多改进，包括引入了**BatchNorm**、新的网络结构和**anchor**机制。作者还引入了许多提高检测精度的训练和测试方法。后来的**YOLOv3**等也主要是使用了大量同时期的模型训练技巧，通过对照和消融实现选择能够最大程度提升检测性能的一些方法。

### 2. SSD

![](https://pic.downk.cc/item/5facfa9c1cd1bbb86b503ae1.jpg)

**SSD**网络也是一种单阶段的目标检测器。之前提到卷积网络的不同层次的特征映射具有不同的空间和语义信息，**SSD**网络考虑使用网络中的多层特征映射，在每一层映射上设置不同尺寸的**anchor**用来检测不同尺度的目标，取得了很好的检测效果。

## （3） 基于Transformer的目标检测模型

# 目标检测领域的论文清单

## 综述 Survey

### Deep Learning for Generic Object Detection: A Survey
- arXiv:[https://arxiv.org/abs/1809.02165](https://arxiv.org/abs/1809.02165)

### Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks
- arXiv:[https://arxiv.org/abs/1809.03193](https://arxiv.org/abs/1809.03193)

### Object Detection in 20 Years: A Survey
- arXiv:[https://arxiv.org/abs/1905.05055](https://arxiv.org/abs/1905.05055)

### A Survey of Deep Learning-based Object Detection
- arXiv:[https://arxiv.org/abs/1907.09408](https://arxiv.org/abs/1907.09408)

### Recent Advances in Deep Learning for Object Detection
- arXiv:[https://arxiv.org/abs/1908.03673](https://arxiv.org/abs/1908.03673)

### Imbalance Problems in Object Detection: A Review
- arXiv:[https://arxiv.org/abs/1909.00169](https://arxiv.org/abs/1909.00169)

## 传统检测方法 Traditional CV Methods

### Selective Search for Object Recognition
- intro:Selective Search
- paper:[https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)

### Discriminatively Trained Deformable Part Models
- intro:DPM
- paper:[http://www.computervisiononline.com/software/discriminatively-trained-deformable-part-models](http://www.computervisiononline.com/software/discriminatively-trained-deformable-part-models)

## 两阶段的检测器 Two-Stage Detectors

### Rich feature hierarchies for accurate object detection and semantic segmentation
- intro:R-CNN
- arXiv:[http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- github(official):[https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)

### Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
- intro:SPP-net
- arXiv:[http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github(official):[https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)

### Fast R-CNN
- intro:Fast R-CNN
- arXiv:[http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- github(official):[https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- intro:Faster R-CNN
- arXiv:[http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- github(official, Matlab):[https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)

### R-FCN: Object Detection via Region-based Fully Convolutional Networks
- intro:R-FCN
- arXiv:[http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)

### Feature Pyramid Networks for Object Detection
- intro:FPN
- arXiv:[https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)

### Mask R-CNN
- intro:Mask R-CNN
- arXiv:[http://arxiv.org/abs/1703.06870](http://arxiv.org/abs/1703.06870)

### A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection
- intro:
- arXiv:[https://arxiv.org/abs/1704.03414](https://arxiv.org/abs/1704.03414)

### Light-Head R-CNN: In Defense of Two-Stage Object Detector
- intro:Light-Head R-CNN
- arXiv:[https://arxiv.org/abs/1711.07264](https://arxiv.org/abs/1711.07264)
- github(official):[https://github.com/zengarden/light_head_rcnn](https://github.com/zengarden/light_head_rcnn)

### Cascade R-CNN: Delving into High Quality Object Detection
- intro:Cascade R-CNN
- arXiv:[https://arxiv.org/abs/1712.00726](https://arxiv.org/abs/1712.00726)

## 单阶段的检测器 One-Stage Detectors

### Scalable Object Detection using Deep Neural Networks
- intro:MultiBox
- arXiv:[https://arxiv.org/abs/1312.2249](https://arxiv.org/abs/1312.2249)



### You Only Look Once: Unified, Real-Time Object Detection
- intro:YOLO
- arXiv:[http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code:[https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)

### SSD: Single Shot MultiBox Detector
- intro:SSD
- arXiv:[http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- github(official):[https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)

### YOLO9000: Better, Faster, Stronger
- intro:YOLOv2
- arXiv:[https://arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- code:[https://pjreddie.com/darknet/yolov2/](https://pjreddie.com/darknet/yolov2/)

### DSSD : Deconvolutional Single Shot Detector
- intro:DSSD
- arXiv:[https://arxiv.org/abs/1701.06659](https://arxiv.org/abs/1701.06659)

### DSOD: Learning Deeply Supervised Object Detectors from Scratch
- intro:DSOD
- arXiv:[https://arxiv.org/abs/1708.01241](https://arxiv.org/abs/1708.01241)

### Focal Loss for Dense Object Detection
- intro:RetinaNet
- arXiv:[https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

### Single-Shot Refinement Neural Network for Object Detection
- intro:RefineNet
- arXiv:[https://arxiv.org/abs/1711.06897](https://arxiv.org/abs/1711.06897)

### MegDet: A Large Mini-Batch Object Detector
- intro:MegDet
- arXiv:[https://arxiv.org/abs/1711.07240](https://arxiv.org/abs/1711.07240)

### FSSD: Feature Fusion Single Shot Multibox Detector
- intro:FSSD
- arXiv:[https://arxiv.org/abs/1712.00960](https://arxiv.org/abs/1712.00960)

### Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network
- intro:ESSD
- arXiv:[https://arxiv.org/abs/1801.05918](https://arxiv.org/abs/1801.05918)

### YOLOv3: An Incremental Improvement
- intro:YOLOv3
- arXiv:[https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- github(official):[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

### DetNet: A Backbone network for Object Detection
- intro:DetNet
- arXiv:[https://arxiv.org/abs/1804.06215](https://arxiv.org/abs/1804.06215)

### Pelee: A Real-Time Object Detection System on Mobile Devices
- intro:Pelee
- arXiv:[https://arxiv.org/abs/1804.06882](https://arxiv.org/abs/1804.06882)

### MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects
- intro:MDSSD
- arXiv:[https://arxiv.org/abs/1805.07009](https://arxiv.org/abs/1805.07009)

### You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery
- intro:YOLT
- arXiv:[https://arxiv.org/abs/1805.09512](https://arxiv.org/abs/1805.09512)

### Fire SSD: Wide Fire Modules based Single Shot Detector on Edge Device
- intro:Fire SSD
- arXiv:[https://arxiv.org/abs/1806.05363](https://arxiv.org/abs/1806.05363)

### CornerNet: Detecting Objects as Paired Keypoints
- intro:CornerNet
- arXiv:[https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244)

### M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network
- intro:M2Det
- arXiv:[https://arxiv.org/abs/1811.04533](https://arxiv.org/abs/1811.04533)

### EfficientDet: Scalable and Efficient Object Detection
- intro:EfficientDet
- arXiv:[https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)



## ⚪ 目标检测

- [You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery](https://0809zheng.github.io/2020/10/12/yolt.html)：(arXiv1805)YOLT：高分辨率大尺寸卫星图像的目标检测。

- [CornerNet: Detecting Objects as Paired Keypoints Learning](https://0809zheng.github.io/2020/07/20/cornernet.html)：(arXiv1808)CornerNet：检测目标框的左上角和右下角位置。

- [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://0809zheng.github.io/2020/04/03/mmdetection.html)：(arXiv1906)商汤科技和香港中文大学开源的基于Pytorch实现的深度学习目标检测工具箱。

- [Recent Advances in Deep Learning for Object Detection](https://0809zheng.github.io/2020/05/17/paper-recent.html)：(arXiv1908)深度学习中目标检测最近的进展综述。

- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://0809zheng.github.io/2020/06/13/yolov4.html)：(arXiv2004)YOLO的第四个版本。

- [Cross-Regional Oil Palm Tree Detection](https://0809zheng.github.io/2021/05/14/oilpalm.html)：(CVPR2020)跨区域的油棕树检测。

- [Cross-regional oil palm tree counting and detection via a multi-level attention domain adaptation network](https://0809zheng.github.io/2021/05/15/oilpalmv2.html)：(arXiv2008)通过多层次注意力域自适应网络进行跨区域的油棕树计数和检测。

- [OneNet: Towards End-to-End One-Stage Object Detection](https://0809zheng.github.io/2020/12/26/onenet.html)：(arXiv2012)OneNet：无需**NMS**的**One-stage**端到端目标检测方法。

- [YOLOX: Exceeding YOLO Series in 2021](https://0809zheng.github.io/2021/08/01/yolox.html)：(arXiv2107)YOLOX：**Anchor-free**的**YOLO**检测器。

# 3. 目标检测的评估指标

在目标检测中，所有置信度(**Confidence Score**)大于阈值的检测框都被视为检测到的目标样本。根据检测框的位置以及其中目标的类别，结果可以被划分到以下几类中的一个：
- 真阳性（**True Positive, TP**）：被正确检测的目标样本。需要满足两个条件：目标边界框与**Ground Truth**的交并比大于阈值；目标预测类别与标签类别匹配。
- 假阳性（**False Positive, FP**）：被错误检测为目标样本的非目标样本。通常为边界框与**Ground Truth**的交并比小于阈值的样本（定位错误），或目标预测类别与标签类别不匹配的样本（分类错误）。
- 假阴性（**False Negative, FN**）：没有被检测出的目标样本。通常为没有检测出的**Ground Truth**区域。
- 真阴性（**True Negative, TN**）：检测出的非目标样本，在目标检测中通常不关心。

在划分检测框的类别时需要计算两者的**交并比 (Intersection over Union, IoU)**，即计算检测区域与**Ground Truth**区域的交集与并集之比：

![](https://pic.imgdb.cn/item/6482dd001ddac507ccacf5a1.jpg)

```python
def IoU(x1_pred, y1_pred, x2_pred, y2_pred, x1_gt, y1_gt, x2_gt, y2_gt):
    W = min(x2_pred, x2_gt) - max(x1_pred, x1_gt)
    H = min(y2_pred, y2_gt) - max(y1_pred, y1_gt)
    if W <= 0 or H <= 0:
        return 0
    
    area_pred = (x2_pred-x1_pred) * (y2_pred-y1_pred)
    area_gt = (x2_gt-x1_gt) * (y2_gt-y1_gt)
    intersection = W * H
    union = area_pred + area_gt - intersection
    return intersection / union
```

目标检测的常用评估指标包括准确率、召回率、**F-score**、**P-R**曲线、平均准确率**AP**、类别平均准确率**mAP**。

### ⚪ 准确率 Precision

**准确率**也叫查准率，是指在所有识别出的物体中**TP**样本所占的比例：

$$
Precision = \frac{TP}{TP+NP} = \frac{TP}{n_{pred}}
$$

### ⚪ 召回率 Recall

**召回率**是指在所有**Ground Truth**物体中**TP**样本所占的比例：

$$
Precision = \frac{TP}{TP+FN} = \frac{TP}{n_{gt}}
$$

### ⚪ F-Score

准确率和召回率随着交并比的阈值而变化。当逐渐减小阈值时，倾向于把更多检测到的样本视为**TP**样本，则准确率逐渐下降，召回率逐渐增加。

**F-Score**是准确率和召回率的的调和平均数 (**harmonic mean**)：

$$
F_{score} = \frac{(B^2+1)PR}{B^2P+R}
$$

其中权重$B$调整准确率和召回率的比重关系。$B$越大则越重视召回率，$B$越小则越重视准确率。特别地，当$B=1$时称为**F1-Score**。

### ⚪ P-R曲线

对于某个指定的预测类别，可以根据给定的交并比阈值$a$把该类别下的所有预测区域划分到**TP**样本和**FP**样本。按照置信度对这些样本倒序排列后，依次计算前$i$个样本的准确率和召回率。例如：

$$
\begin{array}{l|llll}
    \text{编号} & \text{置信度} & \text{类别}& \text{准确率}& \text{召回率} \\
    \hline
    1 & 88.9 & \text{TP} & 1 & 0.2 \\
    2 & 88.0 & \text{TP} & 1 & 0.4 \\
    3 & 86.5 & \text{TP} & 1 & 0.6 \\
    4 & 82.3 & \text{FP} & 0.75 & 0.6 \\
    5 & 77.2 & \text{FP} & 0.6 & 0.6 \\
    6 & 75.3 & \text{TP} & 0.667 & 0.8 \\
    7 & 67.3 & \text{FP} & 0.571 & 0.8 \\
    8 & 64.4 & \text{TP} & 0.625 & 1 \\
\end{array}
$$

此时召回率按照非单调递减的规律变化，而准确率大致按照递减规律变化（有时也会增加）。则可以绘制**准确率-召回率曲线 (P-R plot)**，召回率为横轴，准确率为纵轴：

![](https://pic.imgdb.cn/item/6482e86f1ddac507ccc14cee.jpg)

### ⚪ 平均准确率 Average Precision (AP)

某个类别的**平均准确率 (AP)**定义为该类别对应的**P-R**曲线之下的面积：

$$
AP = \int_0^1P(r)dr
$$

**Precision**与**Recall**的值域在 $0$ 到 $1$ 之间，所以 **AP** 的值域也是在 $[0, 1]$ 范围。由于**P-R**曲线是由离散点构成的曲线，直接计算积分是不可行的。因此对**P-R**曲线进行平滑处理：给定任意一个 **Recall** 值，它对应的 **Precision** 值就等于它右侧的 **Precision** 中最大的值。

$$
p_{interp}(r) = \mathop{\max}_{\overline{r}\geq r}P(\overline{r})
$$

![](https://pic.imgdb.cn/item/6482ea611ddac507ccc472a7.jpg)


在**Pascal VOC 2009**之前计算**AP**值时采用差值方法：在平滑处理的**P-R**曲线上，取横轴 $0.2-1$ 的 $9$ 等分点的 **Precision** 的值，计算其平均值为最终 **AP** 的值。如上图中的**AP**值为：

$$
AP = \frac{1}{9}(1\times 5 + 0.667\times 2 + 0.625\times 2) = 0.843
$$

上述计算方法使用的采样点较少，会有精度损失；并且在比较较小的**AP**值时显著性较差。在**Pascal VOC 2010**之后采用精度更高的方式：绘制出平滑后的**P-R**曲线后，用积分的方式计算平滑曲线下方的面积作为最终的 **AP** 值。

![](https://pic.imgdb.cn/item/6482f6691ddac507ccd9f486.jpg)

如上图中的**AP**值为：

$$
AP = 1\times(0.6-0.2) + 0.667\times(0.8-0.6) + 0.625\times(1-0.8) = 0.658
$$

### ⚪ 类别平均准确率 mean Average Precision (mAP)

**类别平均准确率**是指计算所有类别的**AP**值的平均值。