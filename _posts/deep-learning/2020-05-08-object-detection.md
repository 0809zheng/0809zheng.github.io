---
layout: post
title: '目标检测(Object Detection)'
date: 2020-05-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/64899e3d1ddac507ccb6d5cb.jpg'
tags: 深度学习
---

> Object Detection.

**目标检测(Object Detection)**任务是指在图像中检测出可能存在的目标；包括**定位**和**分类**两个子任务：其中定位是指确定目标在图像中的具体位置，分类是确定目标的具体类别。

本文目录：
1. 传统的目标检测算法
2. 基于深度学习的目标检测算法
3. 目标检测的评估指标
4. 非极大值抑制算法
5. 目标检测中的损失函数

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
- 两阶段的目标检测模型：**R-CNN**, **Fast RCNN**, **Faster RCNN**, **SPP-Net**, **FPN**, **Cascade RCNN**, 
- 单阶段的目标检测模型：**OverFeat**, **YOLOv1-3**, **SSD**, **RetinaNet**, **YOLT**, 
- **Anchor-Free**的目标检测模型：
- 基于**Transformer**的目标检测模型：**DETR**

### ⭐ 扩展阅读
- [<font color=blue>Recent Advances in Deep Learning for Object Detection</font>](https://0809zheng.github.io/2020/05/17/paper-recent.html)：(arXiv1908)深度学习中目标检测最近的进展综述。

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

### ⚪ SPP-Net

- paper：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](http://arxiv.org/abs/1406.4729)

在目标检测模型中需要把尺寸和形状各异的**proposal**调整到固定的尺寸以进行后续的分类和边界框回归任务。**SPP-Net**提出了一种**空间金字塔池化(Spatial Pyramid Pooling, SPP)**层，能够把任意不同尺寸和不同长宽比的图像特征转换为固定尺寸大小的输出特征向量。在实现时分别把特征划分成$k_i \times k_i$的栅格，然后应用最大池化操作构造长度为$\sum_i k_i^2c$的输出特征。

![](https://pic.imgdb.cn/item/63abf68f08b6830163947507.jpg)


### ⚪ FPN

- paper：[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

卷积网络不同层输出的特征映射具有不同的感受野和分辨率，适合检测不同尺度的目标：浅层特征具有较高的分辨率，适合检测小目标；深层特征具有范围较大的感受野，适合检测大目标。

**特征金字塔网络 (Feature Pyramid Network, FPN)**通过转置卷积和特征融合生成不同尺度的特征，并在这些空间信息和语义信息都很丰富的特征映射上同时执行目标检测任务。

![](https://pic.imgdb.cn/item/648a798e1ddac507cc9f9cb3.jpg)

### ⚪ Cascade R-CNN
- paper：[<font color=blue>Cascade R-CNN: Delving into High Quality Object Detection</font>](https://0809zheng.github.io/2021/03/10/cascadercnn.html)

在**Faster R-CNN**等网络中，在训练时会为**RPN**网络设置**IoU**阈值，以区分**proposal**是否包含目标，进而只对**positive proposal**进行边界框回归；通常该阈值设置越高，生成的**proposal**越准确，但是正样本的数量降低，容易过拟合。而在推理时所有**proposal**都用于边界框回归。这导致这导致了在训练和测试阶段中，**proposal**的分布不匹配问题。

为了提高检测精度，产生更高质量的**proposal**，**Cascade R-CNN**使用不同**IoU**阈值（$0.5$、$0.6$、$0.7$）训练多个级联的检测器，通过串联的学习获得较高的目标检测精度。

![](https://pic.imgdb.cn/item/648a6d5a1ddac507cc8cd09c.jpg)


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

### ⚪ YOLO
- paper：[<font color=blue>You Only Look Once: Unified, Real-Time Object Detection</font>](https://0809zheng.github.io/2021/03/16/yolo.html)

**YOLO**模型把图像划分成$S\times S$个网格（对应尺寸为$S\times S$的特征映射），每个网格预测$B$个边界框的坐标和置信度得分，以及当边界框中存在目标时的$K$个类别概率。网络的输出特征尺寸为$S\times S \times (5B+K)$。

![](https://pic.imgdb.cn/item/648ab13b1ddac507cc26b9d9.jpg)

### ⚪ YOLOv2/YOLO9000
- paper：[<font color=blue>YOLO9000: Better, Faster, Stronger</font>](https://0809zheng.github.io/2021/03/17/yolov2.html)

**YOLOv2**相比于**YOLO**进行以下改进：
- 网络结构的改进：增加**BatchNorm**、引入跳跃连接、轻量级网络
- 检测方式的改进：引入**anchor**、通过**k-means**设置**anchor**
- 训练过程的改进：增大图像分辨率、多尺度训练

**YOLO9000**结合小型目标检测数据集（**COCO**的$80$类）与大型图像分类数据集（**ImageNet**的前$9000$类）联合训练目标检测模型。如果输入图像来自分类数据集，则只会计算分类损失。

### ⚪ YOLOv3
- paper：[<font color=blue>YOLOv3: An Incremental Improvement</font>](https://0809zheng.github.io/2021/03/19/yolov3.html)

**YOLOv3**相比于**YOLOv2**进行以下改进：
- 网络结构的改进：特征提取网络为**DarkNet53**、使用多层映射进行多尺度检测、构造特征金字塔网络增强特征
- 损失函数的改进：边界框回归损失采用**GIoU**损失、置信度分类与类别分类损失采用逻辑回归（二元交叉熵）


### ⚪ SSD
- paper：[<font color=blue>SSD: Single Shot MultiBox Detector</font>](https://0809zheng.github.io/2021/03/20/ssd.html)

**SSD**模型提取包含不同尺度的图像的特征金字塔表示，并在每个尺度上执行目标检测。在每一层特征映射上，**SSD**设置不同尺寸的**anchor**来检测不同尺度的目标。对于每一个特征位置，模型对$k=6$个**anchor**分别预测$4$个边界框位置偏移量与$c$个类别概率。则对于$m\times n$的特征图，模型输出特征尺寸为$m\times n\times k(c+4)$。

![](https://pic.imgdb.cn/item/648acd401ddac507cc7a016f.jpg)


### ⚪ RetinaNet
- paper：[<font color=blue>Focal Loss for Dense Object Detection</font>](https://0809zheng.github.io/2021/03/21/retinanet.html)

**RetinaNet**的两个关键组成部分是**Focal Loss**和特征图像金字塔。**Focal Loss**用于缓解边界框的正负样本类别不平衡问题；特征图像金字塔通过**FPN**构造多尺度特征进行预测。

![](https://pic.imgdb.cn/item/648d115a1ddac507ccc3c57e.jpg)

### ⚪ YOLT
- paper：[<font color=blue>You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery</font>](https://0809zheng.github.io/2020/10/12/yolt.html)

对于高分辨率大尺寸图像（如遥感图像）中的目标检测问题，**YOLT**提出了一种两阶段的检测框架：首先把输入图像划分成重叠的子图像，对每张子图像分别进行检测；再通过全局的非极大值抑制算法获得最终的检测结果。

![](https://pic.imgdb.cn/item/649bce1c1ddac507cc8a1d4f.jpg)

## （3） Anchor-Free的目标检测方法

目标检测模型的主流是**Anchor-Based**的方法。这种方法在特征图的每一个像素点预设几个不同尺度和长宽比的边界框，称之为**anchor**。网络对每一个 **anchor** 进行分类，并对正类的 **anchor** 进行回归（位置及大小调整）。这类方法的主要优点是：
1. 很大程度上减少了计算量，并将 **proposal** 数量放到可控范围内以便后面的计算和筛选；
2. 通过调整不同的 **anchor** 设置可以覆盖尽可能多的物体，也可针对不同任务设置不同的 **anchor** 尺度范围；
3. 由于 **anchor** 的尺度是人工定义的，物体的定位是通过 **anchor** 的回归来实现，通过计算偏移量而不是物体的位置大大降低了优化难度。

然而 **anchor** 的设置也有着它自身的缺点。单纯通过使用更多不同大小和长宽比的 **anchor** 以及更多的训练技巧就可以达到更好的效果，然而这种通过增加算力而改进网络的方法很难落实到实际的应用中。并且 **anchor** 的设定需要人为设定大量的参数，且离散的 **anchor** 尺度设定会导致一些物体无法很好的匹配，从而导致遗漏。

**Anchor-Free**的目标检测方法没有采用“预设**anchor**+偏移量回归”的检测流程，而是把目标检测任务视作关键点检测等其它形式的任务，直接对目标的位置进行预测。

### ⚪ CornerNet
- paper：[<font color=blue>CornerNet: Detecting Objects as Paired Keypoints Learning</font>](https://0809zheng.github.io/2020/07/20/cornernet.html)

**CornerNet**检测目标框的左上角和右下角位置，通过**Corner Pooling**提取角点特征，通过预测角点嵌入进行角点匹配，并进一步预测角点位置的偏移量。

![](https://pic.imgdb.cn/item/64c0da391ddac507ccf07702.jpg)

### ⚪ CenterNet
- paper：[<font color=blue>Objects as Points</font>](https://0809zheng.github.io/2021/03/23/centernet.html)

**CenterNet**直接检测目标的中心点、目标的大小以及中心点的位置偏移。

![](https://pic.imgdb.cn/item/64c382301ddac507cc461a53.jpg)

### ⚪ FCOS
- paper：[<font color=blue>FCOS: A Simple and Strong Anchor-free Object Detector</font>](https://0809zheng.github.io/2021/03/30/fcos.html)

**FCOS**预测特征图上各点的类别，再预测各点到**bbox**左侧、右侧、顶端和底部的距离，以及各点的**center-ness score**。

![](https://pic.imgdb.cn/item/64c4bfc11ddac507cc3bf5b2.jpg)


## （4） 基于Transformer的目标检测模型




## 两阶段的检测器 Two-Stage Detectors


### R-FCN: Object Detection via Region-based Fully Convolutional Networks
- intro:R-FCN
- arXiv:[http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)



### Mask R-CNN
- intro:Mask R-CNN
- arXiv:[http://arxiv.org/abs/1703.06870](http://arxiv.org/abs/1703.06870)


## 单阶段的检测器 One-Stage Detectors


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

### DetNet: A Backbone network for Object Detection
- intro:DetNet
- arXiv:[https://arxiv.org/abs/1804.06215](https://arxiv.org/abs/1804.06215)


### EfficientDet: Scalable and Efficient Object Detection
- intro:EfficientDet
- arXiv:[https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)



## ⚪ 目标检测


- [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://0809zheng.github.io/2020/04/03/mmdetection.html)：(arXiv1906)商汤科技和香港中文大学开源的基于Pytorch实现的深度学习目标检测工具箱。

- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://0809zheng.github.io/2020/06/13/yolov4.html)：(arXiv2004)YOLO的第四个版本。



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
def IoU(pred, tar, epsilon=1e-12): # [n, 4]  (x1,y1,x2,y2)
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    tar_area  = (tar[:, 2]  - tar[:, 0])  * (tar[:, 3]  - tar[:, 1])
    inter_lt = np.maximum(pred[:, :2], tar[:, :2])
    inter_rb = np.minimum(pred[:, 2:], tar[:, 2:])
    inter_wh = np.maximum(inter_rb - inter_lt, 0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    return inter_area / (pred_area + tar_area - inter_area + epsilon)
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

# 4. 非极大值抑制算法

**非极大值抑制 (non-maximum suppression,NMS)**算法是目标检测等任务中常用的后处理方法，能够过滤掉多余的检测边界框。

**NMS**算法的流程如下：
- 输入边界框集合$$\mathcal{B}=\{(B_n,c_n)\}_{n=1,...,N}$$，其中$c_n$是边界框$B_n$的置信度；
- 选中集合$$\mathcal{B}$$中置信度最大的边界框$B_i$，将其从集合$$\mathcal{B}$$移动至输出边界框集合$$\mathcal{O}$$中；
- 遍历集合$$\mathcal{B}$$中的其余所有边界框$B_j$，计算边界框$B_i$和边界框$B_j$的交并比$\text{IoU}(B_i,B_j)$。若$\text{IoU}(B_i,B_j)≥\text{threshold}$，则删除边界框$B_j$；
- 重复上述步骤，直至集合$$\mathcal{B}$$为空集。

```python
# NMS from scratch
def NMS(dets, thresh):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]

    areas = (x2-x1+1) * (y2-y1+1)  # +1是把像素数量转换为长宽
    order = scores.argsort()[::-1] # 置信度降序排列

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        idxs = np.where(iou <= thresh)[0]
        order = order[idxs + 1] # +1是修正order[1:]的偏移
    return keep


# NMS via torchvision
from torchvision.ops import nms
keep = nms(
    boxes,         # Tensor[N, 4], (x1, y1, x2, y2) format
    scores,        # Tensor[N]
    iou_threshold  # float
)                  # 返回按置信度降序排列的候选框索引
boxes = boxes[keep]
```

**NMS**算法的处理精度有限、计算效率较低，主要原因包括：
1. 算法采用顺序处理的模式，运算效率低；
2. 根据阈值删除边界框的机制缺乏灵活性；
3. 阈值通常是人工根据经验选定的；
4. 评价标准是交并比**IoU**，只考虑两框的重叠面积。

## （1）提高NMS算法的精度

提高**NMS**算法精度的方法包括**Soft-NMS**, **IoU-Guided NMS**, **Weighted NMS**, **Softer-NMS**, **Adaptive NMS**, **DIoU-NMS**。

### ⚪ Soft-NMS

- paper：[Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503)

对于**IoU≥NMS**阈值的检测框，**NMS**算法直接剔除这些检测框（将其置信度置零），这种严格的剔除机制对于存在遮挡的目标检测情况不友好。**Soft-NMS**引入得分惩罚机制，对于**IoU≥**阈值的检测框，使用一个与**IoU**正相关的惩罚函数$f(\cdot)$降低它们的置信度（而不是直接置零）。

$$
s_i = \begin{cases} s_if\left( \text{IoU}(M,B_i) \right), & \text{IoU}(M,B_i) \geq \text{thresh} \\ s_i, & \text{IoU}(M,B_i) < \text{thresh} \end{cases}
$$

惩罚函数可以设置为线性惩罚或高斯惩罚：

$$
f\left( \text{IoU}(M,B_i) \right) = 1-\text{IoU}(M,B_i)  \quad \text{or} \quad e^{-\frac{\text{IoU}(M,B_i) ^2}{\sigma}}
$$

在对所有边界框的迭代终止之后，**Soft-NMS**依据预先设定的得分阈值（如$0.0001$）来保留最终的检测框。

**Soft-NMS**的主要缺点是处理边界框时存在定位与得分不一致的情况，即在存在遮挡目标的情况下，定位较好而得分低的框相比于定位较差而得分高的框具有更低的最终得分。

### ⚪ IoU-Guided NMS

- paper：[Acquisition of Localization Confidence for Accurate Object Detection](https://arxiv.org/abs/1807.11590)

考虑到边界框的定位与得分可能出现不一致的情况，**IoU-Guided NMS**在网络中额外引入了**IoU**预测分支来学习定位的置信度，进而使用定位置信度来引导**NMS**。

![](https://pic.imgdb.cn/item/648fbe071ddac507ccb68fa0.jpg)

在实现时，**IoU-Guided NMS**使用定位置信度作为**NMS**的筛选依据，每次迭代挑选出具有最大定位置信度的候选框，然后将**IoU≥NMS**阈值的冗余框剔除，并把候选框的分类得分更新为所有冗余框及其自身的得分最大值。最终输出的框必定是同时具有最大分类得分与最大定位置信度的框。

**IoU-Guided NMS**有助于提高严格指标下的精度，如**AP75**, **AP90**；但需要额外添加**IoU**预测分支，造成计算开销。

### ⚪ Weighted NMS

- paper：[Inception Single Shot MultiBox Detector for object detection](https://ieeexplore.ieee.org/document/8026312)

**NMS**每次迭代所选出得分最大的框未必是精确定位的，而冗余框有可能是定位良好的。**Weighted NMS**是对边界框的坐标进行加权平均，加权平均的对象包括每次选出的候选框$M$以及**IoU≥NMS**阈值的冗余框。

$$
M \leftarrow \frac{\sum_i w_i B_i}{\sum_i w_i}, B_i \in \{B\mid \text{IoU}(M,B)\geq \text{thresh}\} ∪ \{M\}
$$

其中加权权重通过对应框的得分与**IoU**乘积计算：$$w_i=s_i\text{IoU}(M,B_i)$$。

**Weighted NMS**通常能够获得更高的**Precision**和**Recall**；但其加权权重通过得分与**IoU**计算，前者存在得分与定位不一致问题，后者只考虑两框重叠部分，位置关系描述不够全面。

### ⚪ Softer-NMS

- paper：[Bounding Box Regression with Uncertainty for Accurate Object Detection](https://arxiv.org/abs/1809.08545)

**Softer-NMS**也是对边界框的坐标进行加权平均，其加权平均形式为：

$$
M \leftarrow \frac{\sum_i w_i B_i/\sigma_i^2}{\sum_i w_i/\sigma_i^2}, B_i \in \{B\mid \text{IoU}(M,B)\geq \text{thresh}\} ∪ \{M\}
$$

其中权重$w_i$计算只与**IoU**相关：

$$
w_i = e^{-\frac{(1-\text{IoU}(M,B_i) )^2}{\sigma_t}}
$$

而$\sigma_i$是为每个边界框引入的定位不确定度，通过网络学习得到：

![](https://pic.imgdb.cn/item/648fc28b1ddac507ccbd6e69.jpg)

**Softer-NMS**可以与标准的**NMS**或**Soft-NMS**结合使用，以稳定提升检测性能。但是需要修改模型来预测方差。

### ⚪ Adaptive NMS

- paper：[Adaptive NMS: Refining Pedestrian Detection in a Crowd](https://arxiv.org/abs/1904.03629)

**NMS**算法认为与当前候选框**IoU**越大的框越可能是冗余框，然而当目标之间存在严重遮挡时可能存在**IoU**大的独立框。通常期望当目标分布稀疏时，**NMS**可选用小阈值以剔除更多冗余框；而当目标分布密集时，**NMS**选用大阈值以获得更高的召回。**Adaptive NMS**为网络引入了一个密度预测模块，用于学习每一个检测框的密度。

![](https://pic.imgdb.cn/item/648ff7671ddac507cc10018f.jpg)

对于一个**Ground Truth**边界框$B_i$，其密度标签定义为：

$$
d_i = \max_j \text{IoU}(B_i,B_j)
$$

此时预测边界框的标签修改为$(x,y,w,h,s,d)$。密度$d$越大，代表该框所处位置的目标分布越密集，越有可能是遮挡严重的地方；反之密度越小，代表该框所处位置的目标分布越稀疏，不太可能有遮挡。

**Adaptive NMS**将每次迭代的**NMS**阈值更改为：

$$
thresh = \max(min\_thresh, d_M)
$$

**Adaptive NMS**可以灵活地结合到各种**NMS**算法中，对存在目标遮挡的检测任务更加友好；但是需要额外添加密度预测模块，造成计算开销。


### ⚪ DIoU-NMS

- paper：[Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)

考虑到在**IoU**相同的情况下，一个框的中心点越靠近当前候选框$M$的中心点，则前者更有可能是冗余框。

![](https://pic.imgdb.cn/item/648ffc801ddac507cc176aad.jpg)

因此使用**DIoU**替代**IoU**作为**NMS**的评判准则：

$$
\text{DIoU} = \text{IoU} - \frac{d^2}{c^2}
$$

![](https://pic.imgdb.cn/item/648ffcd81ddac507cc17fb3f.jpg)

进一步引入参数$\beta$控制中心点偏移的惩罚幅度：

$$
\text{DIoU} = \text{IoU} - \left(\frac{d^2}{c^2}\right)^\beta
$$

当$\beta \to +\infty$时，**DIoU**退化为**IoU**；当$\beta \to 0$时，几乎所有中心点不重合的框都会被保留。

在保持**NMS**阈值不变的情况下，**DIoU**计算得到的结果会小一些，因此会保留更多的框，从而实现更大的召回率，有助于缓解遮挡案例；然而每次迭代剩余更多的框会增加迭代轮数，进一步降低运算效率。

## （2）提高NMS算法的效率

**NMS**中的**IoU**计算是顺序处理的。假设图像中一共有$N$个检测框，每一个框都需要和其余所有框计算一次**IoU**，则计算复杂度是$O(\frac{N(N-1)}{2})=O(N^2)$。

提高**NMS**算法效率的方法包括**CUDA NMS**, **Fast NMS**, **Cluster NMS**, **Matrix NMS**。

### ⚪ CUDA NMS
**CUDA NMS**是**NMS**的**GPU**版本，旨在将**IoU**的计算并行化，并通过矩阵运算加速。

若将边界框集合$$\mathcal{B}$$按照置信度得分从高到低排序，即$B_1$是得分最高的框，$B_N$是得分最低的框；则可以计算**IoU**矩阵：

$$ X=\text{IoU}(B,B)= \begin{pmatrix} x_{11} & x_{12} & ... & x_{1N} \\ x_{21} & x_{22} & ... & x_{2N} \\ ... & ... & ... & ... \\ x_{N1} & x_{N2} & ... & x_{NN} \\ \end{pmatrix} , \quad x_{ij}=\text{IoU}(B_i,B_j) $$

通过**GPU**的并行加速能力，可以一次性得到**IoU**矩阵的全部计算结果。

许多深度学习框架已将**CUDA NMS**作为基本函数使用，如**Pytorch**在**torchvision 0.3**版本中正式集成了**CUDA NMS**。下面是**CUDA NMS**计算**IoU**矩阵的一种简单实现。

```python
def box_iou(boxes1, boxes2): # input: [M, 4], [N, 4]

    def box_area(box): # (x1, y1, x2, y2)
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t()) # [M,]
    area2 = box_area(boxes2.t()) # [N,]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [M,N,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [M,N,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [M, N]
    return inter / (area1[:, None] + area2 - inter) # [M, N]
```

计算得到**IoU**矩阵后，需要利用它抑制冗余框。可以采用矩阵查询的方法（仍然需要顺序处理，但计算**IoU**本身已经被并行加速）；也可以使用下面提出的一些算法。

### ⚪ Fast NMS
- paper：[YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

根据**IoU**矩阵的计算规则可以得出$\text{IoU}(B_i,B_j)=\text{IoU}(B_j,B_i)$，且计算$\text{IoU}(B_i,B_i)$是没有意义的。因此**IoU**矩阵$X$是对称矩阵。

**Fast NMS**算法首先对矩阵$X$使用**pytorch**提供的`triu`函数进行上三角化，得到上三角矩阵：

$$ X=\text{IoU}(B,B)= \begin{pmatrix} 0 & x_{12} & ... & x_{1N} \\ 0 & 0 & ... & x_{2N} \\ ... & ... & ... & ... \\ 0 & 0 & ... & 0 \\ \end{pmatrix} $$

若按照**NMS**的规则，应该按行依次遍历矩阵$X$，如果某行$i$中元素$x_{ij}=\text{IoU}(B_i,B_j),j＞i$超过阈值，则应剔除边界框$B_j$，且不再考虑$j$所对应的行与列。

**Fast NMS**则对上述规则进行了化简，其思路是只要边界框$B_j$与任意边界框$B_i$重合度较大(超过阈值)，则认为其是冗余框，将其剔除。对矩阵$X$执行按列取最大值的操作，得到一维向量$b=[b_1,b_2,...,b_N]$，$b_n$代表矩阵$X$的第$n$列中元素的最大值（即边界框$B_j$与其余边界框的最大**IoU**）。然后进行阈值二值化：$b$中元素小于阈值对应保留的边界框，$b$中元素大于阈值对应冗余框。

**Fast NMS**容易删除更多边界框，因为假如边界框$B_j$与边界框$B_i$重合度较大，但边界框$B_i$已经被剔除，则边界框$B_j$还是有可能会被保留的。一个简单的例子如下，注意到边界框$B_4$被错误地剔除了。

$$
\begin{aligned}
X&= \begin{pmatrix} 0 & 0.6 & 0.1 & 0.3 & 0.8 \\   & 0 & 0.2 & 0.72 & 0.1 \\   &   & 0 & 0.45 & 0.12 \\   &   &   & 0 & 0.28 \\   &   &   &   & 0 \\ \end{pmatrix} \\
& \downarrow (\text{按列取最大值}) \\
b &= [0, 0.6,0.2,0.72,0.8] \\
& \downarrow (\text{选定阈值为0.5}) \\
b &= [1,0,1,0,0] \\
&(\text{保留边界框1和边界框3})
\end{aligned}
$$

使用**pytorch**实现**Fast NMS**算法如下：

```python
def fast_nms(boxes, scores, NMS_threshold:float=0.5):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx]   # 对框按得分降序排列
    iou = box_iou(boxes, boxes)  # IoU矩阵
    iou.triu_(diagonal=1)  # 上三角化
    keep = iou.max(dim=0)[0] < NMS_threshold  # 列最大值向量，二值化
    return boxes[keep], scores[keep]
```

**Fast NMS**算法比**NMS**算法运算速度更快，但由于其会抑制更多边界框，会导致性能略微下降。

### ⚪ Cluster NMS
- paper：[Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/2005.03572)

**Cluster NMS**算法旨在弥补**Fast NMS**算法性能下降的问题，同时保持较快的运算速度。

定义边界框的**cluster**，若边界框$B_i$属于该**cluster**，则边界框$B_i$与集合内任意边界框$B_j$的交并比$\text{IoU}(B_i,B_j)$均超过阈值，且边界框$B_i$与不属于该集合的任意边界框$B_k$的交并比$\text{IoU}(B_i,B_k)$均低于阈值。通过定义**cluster**将边界框分成不同的簇，如下图可以把边界框分成三组**cluster**：

![](https://pic.imgdb.cn/item/609b797bd1a9ae528fb1bd9f.jpg)

**Cluster NMS**算法本质上是**Fast NMS**算法的迭代式。算法前半部分与**Fast NMS**算法相同，都是按降序排列边界框、计算**IoU**矩阵、矩阵上三角化、按列取最大值、阈值二值化得到一维向量$b$。

不同于**Fast NMS**算法直接根据向量$b$输出结果，**Cluster NMS**算法将向量$b$按列复制后将其右乘到**IoU**矩阵中。然后再对新的矩阵按列取最大值、阈值二值化，得到新的向量$b$，再将其扩展后右乘**IoU**矩阵，直至相邻迭代中向量$b$不再变化。

![](https://pic.imgdb.cn/item/609b7a55d1a9ae528fb87c53.jpg)

矩阵右乘相当于进行**行变换**。向量$b$扩展后右乘到**IoU**矩阵，若$b$的第$n$项为$0$，代表对应的边界框$B_n$是冗余框，则不应考虑该框对其他框产生的影响，因此将**IoU**矩阵的第$n$行置零；反之若$b$的第$n$项为$1$，代表对应的边界框$B_n$不是冗余框，因此保留**IoU**矩阵的第$n$行。由数学归纳法可证，**Cluster NMS**算法的收敛结果与**NMS**算法相同。

使用**pytorch**实现**Cluster NMS**算法如下：

```python
def cluster_nms(boxes, scores, NMS_threshold:float=0.5, epochs:int=200):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx].view_as(boxes)   # 对框按得分降序排列
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    C = iou
    for i in range(epochs):    
        A = C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float() # 二值化
        E = E.unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    keep = maxA < NMS_threshold
    return boxes[keep], scores[keep]
```

**NMS**算法顺序处理每一个边界框，会在所有**cluster**上迭代，在计算时重复计算了不同**cluster**之间的边界框。**Cluster NMS**算法通过行变换使得迭代进行在拥有框数量最多的**cluster**上，其迭代次数不超过图像中最大**cluster**所拥有的**边界框个数**。因此**Cluster NMS**算法适合图像中有很多**cluster**的场合。

实践中又提出了一些**Cluster NMS**的变体：

### ⭐ Cluster NMS + Soft-NMS：引入得分惩罚机制
**得分惩罚机制(score penalty mechanism, SPM)**是指每次迭代后根据计算得到的**IoU**矩阵对边界框的置信度得分进行惩罚，即与该边界框重合度高的框越多，该边界框的置信度越低：

$$ s_j = s_j \cdot \prod_{i}^{} e^{-\frac{c_{ij}^2}{\sigma}} $$

```python
def SPM_cluster_nms(boxes, scores, NMS_threshold:float=0.5, epochs:int=200):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx].view_as(boxes)   # 对框按得分降序排列
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    C = iou
    for i in range(epochs):    
        A = C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float() # 二值化
        E = E.unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    scores = torch.prod(torch.exp(-C**2/0.2),0)*scores  #惩罚得分
    keep = scores > 0.1    #得分阈值筛选
    return boxes[keep], scores[keep]
```

### ⭐ Cluster NMS + DIoU-NMS：引入中心点距离修正
将**IoU**替换成**DIoU**，即在**IoU**的基础上加上中心点的归一化距离，能够更好的表达两框的距离；并对得分惩罚机制进行修改：

$$ s_j = s_j \cdot \prod_{i}^{} \mathop{\min} \{ e^{-\frac{c_{ij}^2}{\sigma}} + (1-\text{DIoU})^{\beta}, 1 \} $$

上式中$\beta$用于控制中心点距离惩罚的程度，$\min$避免惩罚因子超过$1$。

### ⭐ Cluster NMS + Weighted NMS：引入加权平均法
在计算**IoU**矩阵时考虑边界框置信度得分的影响，即每次迭代时将**IoU**矩阵与边界框的置信度得分向量按列相乘。

```python
def Weighted_cluster_nms(boxes, scores, NMS_threshold:float=0.5, epochs:int=200):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx].view_as(boxes)   # 对框按得分降序排列
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    C = iou
    for i in range(epochs):    
        A = C
        maxA = A.max(dim=0)[0]   # 列最大值向量
        E = (maxA < NMS_threshold).float() # 二值化
        E = E.unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
        C = iou.mul(E)     # 按元素相乘
        if A.equal(C)==True:     # 终止条件
            break
    keep = maxA < NMS_threshold  # 列最大值向量，二值化
    weights = C*(C>NMS_threshold).float() + torch.eye(C.shape[0])
    weights = weights * scores.reshape((1,-1))
    xx1 = boxes[:,0].unsqueeze(1)
    yy1 = boxes[:,1].unsqueeze(1)
    xx2 = boxes[:,2].unsqueeze(1)
    yy2 = boxes[:,3].unsqueeze(1)

    weightsum = weights.sum(dim=1)         # 坐标加权平均
    xx1 = (xx1*weights).sum(dim=1)/(weightsum)
    yy1 = (yy1*weights).sum(dim=1)/(weightsum)
    xx2 = (xx2*weights).sum(dim=1)/(weightsum)
    yy2 = (yy2*weights).sum(dim=1)/(weightsum)
    boxes = torch.stack([xx1, yy1, xx2, yy2], 1)
    return boxes[keep], scores[keep]
```

### ⚪ Matrix NMS
- paper：[SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152)

**Matrix NMS**算法是对**Soft-NMS**算法的并行实现。在**Soft-NMS**算法中，对于每个边界框$B_j$，如果存在与其**IoU**超过阈值的边界框$B_i$，则说明$B_j$可能是潜在的冗余框，对其置信度得分$s_j$施加一定的惩罚：

$$
s_j \leftarrow s_j f\left(\text{IoU}(B_i,B_j)\right)
$$

**Soft-NMS**算法在计算惩罚系数时是串行实现的。为了并行地计算惩罚系数，**Matrix NMS**考虑以下两点：
- 计算每个满足$s_i>s_j$的边界框$B_i$对边界框$B_j$的惩罚；
- 计算边界框$B_i$被其余边界框抑制的概率。

**Matrix NMS**对边界框$B_j$的惩罚系数计算为：

$$
w_j = \mathop{\min}_{\forall s_i > s_j} \frac{f\left(\text{IoU}(B_i,B_j)\right)}{\mathop{\min}_{\forall s_k > s_i}f\left(\text{IoU}(B_i,B_k)\right)}
$$

其中惩罚函数的设置与**Soft-NMS**相同，可以设置为线性惩罚或高斯惩罚：

$$
f\left( \text{IoU}(M,B_i) \right) = 1-\text{IoU}(M,B_i)  \quad \text{or} \quad e^{-\frac{\text{IoU}(M,B_i) ^2}{\sigma}}
$$

```python
def matrix_nms(boxes, scores, method='gauss', sigma=0.5):
    iou = box_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
    iou_cmax = iou.max(dim=0)[0] # 计算Bj的最大IoU
    iou_cmax = iou_cmax.expand(iou.shape[0], iou.shape[0]).T
    if method == 'gauss':
        decay = torch.exp(-(iou**2-iou_cmax**2)/sigma)
    else:
        decay = (1 - iou) / (1 - iou_cmax)
    decay = decay.min(dim=0)[0]
    return scores * decay
```

# 5. 目标检测中的损失函数

目标检测中的损失函数包括边界框的**分类**损失和**回归**损失。其中分类损失用于区分边界框的类别，即边界框内目标的类别，对于两阶段的检测方法还包含边界框的正负类别；常用的分类损失函数包括**Cross-Entropy loss**, **Focal loss**, **Poly loss**, **Cross-Entropy loss**, **Cross-Entropy loss**, 。

而回归损失衡量预测边界框坐标$x_{pred}$和**GT**边界框坐标$x_{gt}$之间的差异，常用的回归损失函数包括**L1 / L2 loss**, **smooth L1 loss**, **IoU loss**, **GIoU loss**, **DIoU loss**, **CIoU loss**, **EIoU loss**, **MPDIoU loss**。

## （1）常用的分类损失

### ⚪ Cross-Entropy loss

通用的分类损失采用交叉熵损失(**Cross-Entropy loss**)。给定网络输出边界框的预测类别概率分布$p=(p_0,...,p_K)$和标签类别$c$ ($0$表示背景类)，则可以构造多元交叉熵损失：

$$
\begin{aligned}
\mathcal{L}_{cls}(p,c) &= -\log p_c \\
\end{aligned}
$$

对于一些两阶段的目标检测算法（如**Faster RCNN**），需要对**RPN**提取的**proposal**边界框进行二分类（区分是否为包含目标的边界框），此时需要构造二元交叉熵损失：

$$
\begin{aligned}
\mathcal{L}_{cls}(\hat{p},p) &= -p \log \hat{p} - (1-p) \log (1-\hat{p}) \\
\end{aligned}
$$

### ⚪ [<font color=blue>Focal Loss</font>](https://0809zheng.github.io/2021/03/21/retinanet.html)

目标检测中的边界框分类问题是一种类别不平衡分类，即大部分边界框都是无目标的背景框。

**Focal loss**显式地引入了权重因子$(1-p_t)^{\gamma},\gamma \geq 0$，使得$p_t$（目标类别的预测置信度）越大时权重越小，即对容易分类的样本减少权重。此外为了更好地控制损失函数的形状，额外引入一个权重系数$\alpha$。

$$
\mathcal{L}_{\text{focal}}(p,t) = -\alpha(1-p_t)^\gamma \log p_t
$$

![](https://pic.imgdb.cn/item/648d08ef1ddac507cc9777f4.jpg)

### ⚪ [<font color=blue>Poly Loss</font>](https://0809zheng.github.io/2022/07/07/poly.html)

交叉熵损失可以被泰勒展开为一系列多项式函数的线性组合：

$$
\begin{aligned}
\mathcal{L}_{CE}(p,t) &= -\log p_t = \sum_{j=1}^\infty \frac{1}{j} (1-p_t)^j \\
&= (1-p_t)+ \frac{1}{2} (1-p_t)^2 +\frac{1}{3} (1-p_t)^3 + \cdots \\
\end{aligned}
$$

其中低阶部分倾向于得到正确的预测结果，高阶部分倾向于防止预测结果出错（如缓解类别不平衡问题）。**Poly Loss**为前$N$个多项式项引入扰动$\epsilon_1,...,\epsilon_N$，用于对更精确的任务目标进行更精确的调整。

$$
\begin{aligned}
\mathcal{L}_{\text{Poly-N}}(p,t) &= (\epsilon_1+1)(1-p_t)+ \cdots +\left(\epsilon_N+\frac{1}{N}\right) (1-p_t)^N + \sum_{j=N+1}^\infty \frac{1}{j} (1-p_t)^j \\
&= -\log p_t + \sum_{j=1}^N \epsilon_j (1-p_t)^j
\end{aligned}
$$

## （2）常用的回归损失

### ⚪ L1 / L2 Loss
通用的回归损失采用**L1 / L2 loss**，计算如下：

$$ L1 = |x| \qquad L2 = x^2 $$

这两种损失函数存在缺点：
- **L1 Loss**的导数为常数，在训练后期，真实值与预测值的差值$x=x_{gt}-x_{pred}$很小时，如果学习率不变，损失函数会在稳定值附近波动，难以收敛到更高精度；
- **L2 Loss**在差值很大时，其导数非常大，故在训练初期不稳定。

### ⚪ smooth L1 loss
- paper：[<font color=blue>Fast R-CNN</font>](https://0809zheng.github.io/2021/03/07/fastrcnn.html)

针对**L1 / L2 loss**存在的问题，修正后得到**smooth L1 loss**：

$$ \text{smooth-L1}(x) = \begin{cases} |x|-0.5, & x ≥ 1 \\ 0.5x^2, &x < 1 \end{cases} $$

该损失函数在差值$x$较大时是**L1 Loss**，在其较小时是**L2 Loss**。也可以引入一个方差系数：

$$ \text{smooth-L1}(x, \sigma) = \begin{cases} |x|-0.5/\sigma^2, & x ≥ 1/\sigma^2 \\ 0.5\sigma^2x^2, &x < 1/\sigma^2 \end{cases} $$


```python
def smoothL1Loss(pred_loc, gt_loc, sigma):
    sigma_squared = sigma ** 2
    regression_diff = (gt_loc - pred_loc)
    regression_diff = regression_diff.abs().float()
    regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
    return regression_loss.mean()
```

上述介绍的损失函数存在共通的缺点：
- 这些损失函数独立地计算每一个坐标分量(如$x$,$y$,$h$,$w$)的差异，然后相加得到最终的损失。这样做忽略了不同坐标分量之间的联系(如$x$,$y$靠近图像边缘时，$h$,$w$会受到限制)；
- 目标检测中，评估边界框定位质量的指标是**交并比 IoU**，与上述损失函数的计算是不匹配的(具有相同损失值的不同坐标组合可能具有不同的**IoU**)。

### ⚪ IoU loss
- paper：[UnitBox: An Advanced Object Detection Network](https://arxiv.org/abs/1608.01471)

为使得边界框的定位与其评估指标**IoU**相匹配，不妨直接把**IoU**设置为损失函数（**IoU loss**）：

$$ \text{IoU loss} = -\log\frac{\text{intersection}(x_{gt},x_{pred})}{\text{union}(x_{gt},x_{pred})} $$

实际计算中也简化为：

$$ \text{IoU loss} = 1-\frac{\text{intersection}(x_{gt},x_{pred})}{\text{union}(x_{gt},x_{pred})} $$

该损失将边界框的不同坐标分量(如$x$,$y$,$h$,$w$)联系起来，具有尺度不变性。

```python
def IoULoss(b1, b2): # [n, 4]  format: xywh
    #   求出预测框左上角右下角
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half
    
    #   求出真实框左上角右下角
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #   求真实框和预测框所有的iou
    intersect_mins  = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / union_area
    return 1-iou
```

**IoU loss**的缺点：
- 当预测边界框和实际边界框不相交时，**IoU**为$0$。此时**IoU loss**是不可导的，不能反映两个边界框的远近程度，无法优化；
- 假设预测框和目标框的大小都确定，只要两个框的相交值是确定的，其**IoU**值是相同时，不能反映两个框是如何相交的。如下图所示，通常认为右边的边界框更好，但两者**IoU**值相同：

![](https://img.imgdb.cn/item/60177c843ffa7d37b3a8976c.jpg)

### ⚪ GIoU loss
- paper：[Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)

**IoU**不能区分一些边界框的相交情况。作者提出了评估指标**GIoU (Generalized IoU)**，不仅关注重叠区域，还关注非重叠区域，能更好的反映两者的重合度。计算如下：

$$ \text{GIoU} = \text{IoU} - \frac{|C-(A∪B)|}{|C|} $$

其中$A$、$B$表示两个边界框区域，$C$表示$A$和$B$的外接矩形。**GIoU**的取值范围是$-1 \sim 1$，当两框重合时取$1$，当两框未相交时取$-1$。

通过**GIoU**可以定义**GIoU loss**：

$$ \text{GIoU loss} = 1-\text{GIoU} $$

```python
def GIoULoss(b1, b2): # [n, 4]  format: xywh
    #   求出预测框左上角右下角
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half
    
    #   求出真实框左上角右下角
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #   求真实框和预测框所有的iou
    intersect_mins  = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / union_area

    #   找到真实框和预测框的外接矩形
    enclose_mins    = torch.min(b1_mins, b2_mins)
    enclose_maxes   = torch.max(b1_maxes, b2_maxes)
    enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    
    #   计算GIoU
    enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou            = iou - (enclose_area - union_area) / enclose_area
    return 1-giou
```

**GIoU loss**的缺点：
- 当一个边界框完全包含另一个边界框时，**GIoU**退化为**IoU**，无法区分两者的相对位置关系。如下图三者的**GIoU**相同，但我们认为右边的结果更好：

![](https://img.imgdb.cn/item/60177d613ffa7d37b3a92f2f.jpg)

### ⚪ DIoU loss
- paper：[Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)

作者分析边界框回归的三个重要因素：**中心点距离**、**重叠面积**和**长宽比**。笔者的理解如下，对于给定的**GT**边界框，中心点距离能够对预测框进行粗定位，重叠面积能够进一步筛选预测框，长宽比能够最终确定预测框。

**IoU**只衡量重叠面积，需要进一步考虑预测框和**GT**框的惩罚项，损失函数的通用范式如下：

$$ L = 1-IoU+R(B_{pred},B_{gt}) $$

作者提出了评估指标**DIoU (Distance IoU)**，将惩罚项设置为：

$$ R_{DIoU} = \frac{ρ^2(b_{pred},b_{gt})}{c^2} $$

其中$b_{pred}$,$b_{gt}$表示边界框的中心点，$ρ$是欧式距离，$c$表示最小外接矩形的对角线距离，如下图所示：

![](https://img.imgdb.cn/item/6017d99c3ffa7d37b3ec6a3e.jpg)

**DIoU**是在**IoU**的基础上加上中心点的归一化距离惩罚，能够更好的表达两框之间的距离，计算如下：

$$ \text{DIoU} = \text{IoU} - R_{DIoU} $$

通过**DIoU**可以定义**DIoU loss**：

$$ \text{DIoU loss} = 1-\text{DIoU} $$

```python
def DIoULoss(b1, b2): # [n, 4]  format: xywh
    #   求出预测框左上角右下角
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half
    
    #   求出真实框左上角右下角
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #   求真实框和预测框所有的iou
    intersect_mins  = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / union_area

    #   找到真实框和预测框的外接矩形
    enclose_mins    = torch.min(b1_mins, b2_mins)
    enclose_maxes   = torch.max(b1_maxes, b2_maxes)
    enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    
    #   计算中心的距离
    center_wh       = b1_xy - b2_xy
    center_distance     = torch.sum(torch.pow(center_wh, 2), axis=-1)
    #   计算对角线距离
    enclose_diagonal    = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    #   计算DIoU
    diou                = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min = 1e-6)
    return 1-diou
```

**DIoU loss**的缺点是没有考虑边界框的长宽比的影响。

### ⚪ CIoU loss
- paper：[Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)

**DIoU**的作者又进一步提出了**CIoU (Complete IoU)**，在惩罚项上增加长宽比影响因子$\alpha v$：

$$ R_{CIoU} = \frac{ρ^2(b_{pred},b_{gt})}{c^2} + \alpha v $$

其中$v$衡量边界框长宽比的一致性，$\alpha$用于平衡$v$的值，计算如下：

$$
\begin{aligned}
v &= \frac{4}{\pi^2} (\arctan\frac{w^{gt}}{h^{gt}}-\arctan\frac{w}{h})^2 \\
\alpha &= \frac{v}{(1-\text{IoU})+v}
\end{aligned}
$$

通过**CIoU**可以定义**CIoU loss**：

$$ \text{CIoU loss} = 1-\text{CIoU} $$

```python
def CIoULoss(b1, b2): # [n, 4]  format: xywh
    #   求出预测框左上角右下角
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half
    
    #   求出真实框左上角右下角
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #   求真实框和预测框所有的iou
    intersect_mins  = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / union_area

    #   找到真实框和预测框的外接矩形
    enclose_mins    = torch.min(b1_mins, b2_mins)
    enclose_maxes   = torch.max(b1_maxes, b2_maxes)
    enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    
    #   计算中心的距离
    center_wh       = b1_xy - b2_xy
    center_distance     = torch.sum(torch.pow(center_wh, 2), axis=-1)
    #   计算对角线距离
    enclose_diagonal    = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    diou                = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min = 1e-6)

    v       = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min = 1e-6))), 2)
    alpha   = v / torch.clamp((1.0 - iou + v), min = 1e-6)
    ciou     = diou - alpha * v
    return 1-ciou
```

**CIoU loss**的主要缺点如下：
- 衡量长宽比的$v$计算过于复杂，减缓了收敛速度；
- 推导可得$\frac{\partial v}{\partial w}=-\frac{h}{w}\frac{\partial v}{\partial h}$，$w$和$h$的优化是相反的。

### ⚪ EIoU loss
- paper：[Focal and Efficient IOU Loss for Accurate Bounding Box Regression](https://arxiv.org/abs/2101.08158)

为简化**CIoU**中长宽比影响因子$\alpha v$的计算，作者提出**EIoU (Efficient IoU)**，用下式代替长宽比的影响因子$\alpha v$：

$$ \frac{ρ^2(w_{pred},w_{gt})}{c_w^2} + \frac{ρ^2(h_{pred},h_{gt})}{c_h^2} $$

**EIoU**的定义如下：

$$ \text{EIoU} = \text{IoU} - \frac{ρ^2(b_{pred},b_{gt})}{c^2}-\frac{ρ^2(w_{pred},w_{gt})}{c_w^2} - \frac{ρ^2(h_{pred},h_{gt})}{c_h^2} $$

通过**EIoU**可以定义**EIoU loss**：

$$ \text{EIoU loss} = 1-\text{EIoU} $$

```python
def EIoULoss(b1, b2): # [n, 4]  format: xywh
    #   求出预测框左上角右下角
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half
    
    #   求出真实框左上角右下角
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #   求真实框和预测框所有的iou
    intersect_mins  = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / union_area

    #   找到真实框和预测框的外接矩形
    enclose_mins    = torch.min(b1_mins, b2_mins)
    enclose_maxes   = torch.max(b1_maxes, b2_maxes)
    enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    
    #   计算EIoU
    enclose_squared_w    = enclose_wh[..., 0] ** 2
    enclose_squared_h    = enclose_wh[..., 1] ** 2
    enclose_squared_diag = enclose_squared_w + enclose_squared_h
    center_squared_w     = (b1_xy[..., 0] - b2_xy[..., 0]) ** 2
    center_squared_h     = (b1_xy[..., 1] - b2_xy[..., 1]) ** 2
    center_squared_diag  = center_squared_w + center_squared_h
    eiou                 = iou - center_squared_w / enclose_squared_w - center_squared_h / enclose_squared_h - center_squared_diag / enclose_squared_diag
    return 1-eiou
```

### ⚪ MPDIoU loss
- paper：[<font color=blue>MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression</font>](https://0809zheng.github.io/2023/07/14/mpdiou.html)

**MPDIoU**直接最小化预测边界框与实际标注边界框之间的左上角和右下角点距离。

$$ \text{MPDIoU} = \text{IoU} - \frac{d_1^2}{w^2+h^2}- \frac{d_2^2}{w^2+h^2} $$

![](https://pic.imgdb.cn/item/64c07dab1ddac507cc48af59.jpg)

通过**MPDIoU**可以定义**MPDIoU loss**：

$$ \text{MPDIoU loss} = 1-\text{MPDIoU} $$