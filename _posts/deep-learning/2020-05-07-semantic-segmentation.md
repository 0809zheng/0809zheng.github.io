---
layout: post
title: '图像分割(Image Segmentation)'
date: 2020-05-07
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63f2c620f144a01007e3c370.jpg'
tags: 深度学习
---

> Image Segmentation.

**图像分割 (Image Segmentation)**是对图像中的每个像素进行分类，可以细分为：
- **语义分割 (semantic segmentation)**：注重类别之间的区分，而不区分同一类别的不同个体；
- **实例分割 (instance segmentation)**：注重类别以及同一类别的不同个体之间的区分；
- **全景分割 (panoptic segmentation)**：对于可数的对象实例(如行人、汽车)做实例分割，对于不可数的语义区域(如天空、地面)做语义分割。

![](https://pic.imgdb.cn/item/63f2c620f144a01007e3c370.jpg)

本文目录：
1. 图像分割模型
2. 图像分割的评估指标
3. 图像分割的损失函数
4. 常用的图像分割数据集

# 1. 图像分割模型

图像分割的任务是使用深度学习模型处理输入图像，得到带有语义标签的相同尺寸的输出图像。

![](https://pic.imgdb.cn/item/63f2e1aef144a0100707c297.jpg)

图像分割模型通常采用**编码器-解码器(encoder-decoder)**结构。编码器从预处理的图像数据中提取特征，解码器把特征解码为分割热图。图像分割模型的发展趋势可以大致总结为：
- 全卷积网络：**FCN**, **SegNet**, **U-Net**, 
- 多尺度特征：**DeepLab v1,2,3,3+**, **PSPNet**, 
- 基于**Transformer**：

## (1) 基于全卷积网络的图像分割模型

标准卷积神经网络包括卷积层、下采样层和全连接层。因此早期基于深度学习的图像分割模型旨在解决如何更好从卷积下采样中恢复丢掉的信息损失。

这一阶段的模型为生成与输入图像尺寸一致的分割结果，丢弃了全连接层，并引入一系列上采样操作。逐渐形成了以**U-Net**为核心的对称编码器-解码器结构。

### ⚪ [<font color=Blue>FCN</font>](https://0809zheng.github.io/2021/02/08/fcn.html)

**FCN**提出用全卷积网络来处理语义分割问题，通过全卷积网络进行特征提取和下采样，通过双线性插值进行上采样。

![](https://pic.imgdb.cn/item/63f3294ff144a010076aeec8.jpg)

### ⚪ [<font color=Blue>SegNet</font>](https://0809zheng.github.io/2021/02/11/segnet.html)

**SegNet**采用编码器-解码器结构，通过反池化进行上采样。

![](https://pic.downk.cc/item/5ebb64bcc2a9a83be59a49f5.jpg)


### ⚪ [<font color=Blue>RefineNet</font>](https://0809zheng.github.io/2021/02/19/refinenet.html)

**RefineNet**把编码器产生的多个分辨率特征进行一系列卷积、融合、池化。

![](https://pic.downk.cc/item/5ebcea7ac2a9a83be531a81b.jpg)

### ⚪ [<font color=Blue>U-Net</font>](https://0809zheng.github.io/2021/02/13/unet.html)

**U-Net**使用对称的U型网络设计，在对应的下采样和上采样之间引入跳跃连接。

![](https://pic.imgdb.cn/item/63f32f2ff144a01007724bfb.jpg)

### ⚪ [<font color=Blue>V-Net</font>](https://0809zheng.github.io/2021/06/05/vnet.html)

**V-Net**是**3D**版本的**U-Net**，下采样使用步长为$2$的卷积。

![](https://pic.imgdb.cn/item/63f96706f144a01007a6219c.jpg)

### ⚪ [<font color=Blue>M-Net</font>](https://0809zheng.github.io/2021/06/06/mnet.html)

**M-Net**在**U-Net**的基础上引入了**left leg**和**right leg**。**left leg**使用最大池化不断下采样数据，**right leg**则对数据进行上采样并叠加到每一层次的输出后。

![](https://pic.imgdb.cn/item/60db00195132923bf85b72b1.jpg)


### ⚪ [<font color=Blue>W-Net</font>](https://0809zheng.github.io/2021/06/07/wnet.html)

**W-Net**通过堆叠两个**U-Net**实现无监督的图像分割。编码器**U-Net**提取分割表示，解码器**U-Net**重构原始图像。

![](https://pic.imgdb.cn/item/60dbc55a5132923bf89ebb75.jpg)

### ⚪ [<font color=Blue>Y-Net</font>](https://0809zheng.github.io/2021/06/08/ynet.html)

**Y-Net**在**U-Net**的编码位置后增加了一个概率图预测结构，在分割任务的基础上额外引入了分类任务。

![](https://pic.imgdb.cn/item/60dc51415132923bf82c49dd.jpg)

### ⚪ [<font color=Blue>UNet++</font>](https://0809zheng.github.io/2021/06/29/unetpp.html)

**UNet++**通过跳跃连接融合了不同深度的**U-Net**，并为每级**U-Net**引入深度监督。

![](https://pic.imgdb.cn/item/63f97021f144a01007b12a99.jpg)

### ⚪ [<font color=Blue>Attention U-Net</font>](https://0809zheng.github.io/2021/02/20/attunet.html)

**Attention U-Net**通过引入**Attention gate**模块将空间注意力机制集成到**U-Net**的跳跃连接和上采样模块中。

![](https://pic.imgdb.cn/item/63f97532f144a01007b9c89a.jpg)

## (2) 基于多尺度特征的图像分割模型

多尺度问题是指当图像中的目标对象存在不同大小时，分割效果不佳的现象。比如同样的物体，在近处拍摄时物体显得大，远处拍摄时显得小。解决多尺度问题的目标就是不论目标对象是大还是小，网络都能将其分割地很好。

随着图像分割模型的效果不断提升，分割任务的主要矛盾逐渐从恢复像素信息逐渐演变为如何更有效地利用上下文(**context**)信息，并基于此设计了一系列用于提取多尺度特征的网络结构。

### ⚪ [<font color=Blue>Deeplab</font>](https://0809zheng.github.io/2021/02/14/deeplab.html)

**Deeplab**引入空洞卷积进行图像分割任务，并使用全连接条件随机场精细化分割结果。

![](https://pic.imgdb.cn/item/63f333ecf144a010077a1a93.jpg)

### ⚪ [<font color=Blue>DeepLab v2</font>](https://0809zheng.github.io/2021/02/15/deeplab2.html)

**Deeplab v2**引入了**空洞空间金字塔池化层 ASPP**，即带有不同扩张率的空洞卷积的金字塔池化。

![](https://pic.imgdb.cn/item/63f724f6f144a010074d13e4.jpg)

### ⚪ [<font color=Blue>DeepLab v3</font>](https://0809zheng.github.io/2021/02/16/deeplab3.html)

**Deeplab v3**对**ASPP**模块做了升级，把扩张率调整为$[1, 6, 12, 18]$，并增加了全局平均池化：

![](https://pic.downk.cc/item/5ebcde6bc2a9a83be525b262.jpg)


### ⚪ [<font color=Blue>DeepLab v3+</font>](https://0809zheng.github.io/2021/02/17/deeplab3+.html)

**Deeplab v3+**采用了编码器-解码器结构。

![](https://pic.downk.cc/item/5ebce009c2a9a83be5274019.jpg)

上述**Deeplab**模型的对比如下：

![](https://pic.imgdb.cn/item/63f729f8f144a0100755b990.jpg)

### ⚪ [<font color=Blue>PSPNet</font>](https://0809zheng.github.io/2021/02/18/pspnet.html)

**PSPNet**引入了**金字塔池化模块 PPM**。**PPM**模块并联了四个不同大小的平均池化层，经过卷积和上采样恢复到原始大小。

![](https://pic.imgdb.cn/item/63f86f67f144a010072e9a47.jpg)



## (3) 基于Transformer的图像分割模型

















### ⚪ 参考文献
- [<font color=Blue>Fully Convolutional Networks for Semantic Segmentation</font>](https://0809zheng.github.io/2021/02/08/fcn.html)：(arXiv1411)FCN: 语义分割的全卷积网络。
- [<font color=Blue>Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs</font>](https://0809zheng.github.io/2021/02/14/deeplab.html)：(arXiv1412)DeepLab: 通过深度卷积网络和全连接条件随机场实现图像语义分割。
- [<font color=Blue>U-Net: Convolutional Networks for Biomedical Image Segmentation</font>](https://0809zheng.github.io/2021/02/13/unet.html)：(arXiv1505)U-Net: 用于医学图像分割的卷积网络。
- [<font color=Blue>SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation</font>](https://0809zheng.github.io/2021/02/11/segnet.html)：(arXiv1511)SegNet: 图像分割的深度卷积编码器-解码器结构。
- [<font color=Blue>V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation</font>](https://0809zheng.github.io/2021/06/05/vnet.html)：(arXiv1606)V-Net：用于三维医学图像分割的全卷积网络。
- [<font color=Blue>DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs</font>](https://0809zheng.github.io/2021/02/15/deeplab2.html)：(arXiv1606)DeepLab v2: 通过带有空洞卷积的金字塔池化实现图像语义分割。
- [<font color=Blue>RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation</font>](https://0809zheng.github.io/2021/02/19/refinenet.html)：(arXiv1611)RefineNet: 高分辨率语义分割的多路径优化网络。
- [<font color=Blue>Pyramid Scene Parsing Network</font>](https://0809zheng.github.io/2021/02/18/pspnet.html)：(arXiv1612)PSPNet: 金字塔场景解析网络。
- [<font color=Blue>M-Net: A Convolutional Neural Network for Deep Brain Structure Segmentation</font>](https://0809zheng.github.io/2021/06/06/mnet.html)：(ISBI 2017)M-Net：用于三维脑结构分割的二维卷积神经网络。
- [<font color=Blue>Rethinking Atrous Convolution for Semantic Image Segmentation</font>](https://0809zheng.github.io/2021/02/16/deeplab3.html)：(arXiv1706)DeepLab v3: 重新评估图像语义分割中的扩张卷积。
- [<font color=Blue>W-Net: A Deep Model for Fully Unsupervised Image Segmentation</font>](https://0809zheng.github.io/2021/06/07/wnet.html)：(arXiv1711)W-Net：一种无监督的图像分割方法。
- [<font color=Blue>Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation</font>](https://0809zheng.github.io/2021/02/17/deeplab3+.html)：(arXiv1802)DeepLab v3+: 图像语义分割中的扩张可分离卷积。
- [<font color=Blue>Attention U-Net: Learning Where to Look for the Pancreas</font>](https://0809zheng.github.io/2021/02/20/attunet.html)：(arXiv1804)Attention U-Net: 向U-Net引入注意力机制。
- [<font color=Blue>Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images</font>](https://0809zheng.github.io/2021/06/08/ynet.html)：(arXiv1806)Y-Net：乳腺活检图像的分割和分类。
- [<font color=Blue>UNet++: A Nested U-Net Architecture for Medical Image Segmentation</font>](https://0809zheng.github.io/2021/06/29/unetpp.html)：(arXiv1807)UNet++：用于医学图像分割的巢型UNet。



# 2. 图像分割的评估指标 

图像分割任务本质上是一种图像像素分类任务，可以使用常见的分类评价指标来评估模型的好坏。图像分割中常用的评估指标包括：
- 像素准确率 (**pixel accuracy, PA**)
- 类别像素准确率 (**class pixel accuracy, CPA**)
- 类别平均像素准确率 (**mean pixel accuracy, MPA**)
- 交并比 (**Intersection over Union, IoU**)
- 平均交并比 (**mean Intersection over Union, MIoU**)
- 频率加权交并比 (**Frequency Weighted Intersection over Union, FWIoU**)
- **Dice**系数 (**Dice Coefficient**)

上述评估指标均建立在**混淆矩阵**的基础之上，因此首先介绍混淆矩阵，然后介绍这些评估指标的计算。

## ⚪ 混淆矩阵
图像分割问题本质上是对图像中的像素的分类问题。

### (1) 二分类
以**二分类**为例，图像中的每个像素可能属于**正例(Positive)**也可能属于**反例(Negative)**。根据像素的实际类别和模型的预测结果，可以把像素划分为以下四类中的某一类：
- **真正例 TP(True Positive)**：实际为正例，预测为正例
- **假正例 FP(False Positive)**：实际为反例，预测为正例
- **真反例 TN(True Negative)**：实际为反例，预测为反例
- **假反例 FN(False Negative)**：实际为正例，预测为反例

绘制分类结果的**混淆矩阵(confusion matrix)**如下：

$$ \begin{array}{l|cc} \text{真实情况\预测结果} & \text{正例} & \text{反例} \\ \hline  \text{正例} & TP & FN \\  \text{反例} & FP & TN \\ \end{array} $$

根据混淆矩阵可做如下计算：
- **准确率(accuracy)**，定义为模型分类正确的像素比例：

$$ \text{Accuracy} = \frac{TP+TN}{TP+FP+TN+FN} $$

- **查准率(precision)**，定义为模型预测为正例的所有像素中，真正为正例的像素比例：

$$ \text{Precision} = \frac{TP}{TP+FP} $$

- **查全率(recall)**,又称**召回率**，定义为所有真正为正例的像素中，模模型预测为正例的像素比例：

$$ \text{Recall} = \frac{TP}{TP+FN} $$

### (2) 多分类

图像分割通常是**多分类**问题，也有类似结论。对于多分类问题，**混淆矩阵**表示如下：

$$ \begin{array}{l|ccc} \text{真实情况\预测结果} & \text{类别1} & \text{类别2} & \text{类别3} \\ \hline  \text{类别1} & a & b & c \\  \text{类别2} & d & e & f \\ \text{类别3} & g & h & i \\ \end{array} $$

对于多分类问题，也可计算：
- **准确率**：

$$ \text{Accuracy} = \frac{a+e+i}{a+b+c+d+e+f+g+h+i} $$

- **查准率**，以类别$1$为例：

$$ \text{Precision} = \frac{a}{a+d+g} $$

- **查全率**，以类别$1$为例：

$$ \text{Recall} = \frac{a}{a+b+c} $$

### (3) 计算混淆矩阵
对于图像分割的预测结果`imgPredict`和真实标签`imgLabel`，可以使用[np.bincount](https://0809zheng.github.io/2020/09/11/bincount.html)函数计算混淆矩阵，计算过程如下：

```python
import numpy as np

def genConfusionMatrix(numClass, imgPredict, imgLabel):
    '''
    Parameters
    ----------
    numClass : 类别数(不包括背景).
    imgPredict : 预测图像.
    imgLabel : 标签图像.
    '''
    # remove classes from unlabeled pixels in gt image and predict
    mask = (imgLabel >= 0) & (imgLabel < numClass)
    
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass**2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

imgPredict = np.array([[0,1,0],
                 [2,1,0],
                 [2,2,1]])
imgLabel = np.array([[0,2,0],
                  [2,1,0],
                  [0,2,1]])
print(genConfusionMatrix(3, imgPredict, imgLabel))

###
[[3 0 1]
 [0 2 0]
 [0 1 2]]
###
```

## ⚪ 像素准确率 PA
**像素准确率** (**pixel accuracy, PA**) 衡量所有类别预测正确的像素占总像素数的比例，相当于分类任务中的**准确率(accuracy)**。

**PA**计算为混淆矩阵对角线元素之和比矩阵所有元素之和，以二分类为例：

$$ \text{PA} = \frac{TP+TN}{TP+FP+TN+FN} $$

```python
def pixelAccuracy(confusionMatrix):
    # return all class overall pixel accuracy
    #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
    acc = np.diag(confusionMatrix).sum() /  confusionMatrix.sum()
    return acc
```

## ⚪ 类别像素准确率 CPA
**类别像素准确率** (**class pixel accuracy, CPA**) 衡量在所有预测类别为$i$的像素中，真正属于类别$i$的像素占总像素数的比例，相当于分类任务中的**查准率(precision)**。

第$i$个类别的**CPA**计算为混淆矩阵第$i$个对角线元素比矩阵该列元素之和。以二分类为例，第$0$个类别的**CPA**计算为：

$$ \text{CPA} = \frac{TP}{TP+FP} $$

```python
def classPixelAccuracy(confusionMatrix):
    # return each category pixel accuracy(A more accurate way to call it precision)
    # acc = (TP) / TP + FP
    classAcc = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return classAcc # 返回一个列表，表示各类别的预测准确率
```

## ⚪ 类别平均像素准确率 MPA
**类别平均像素准确率** (**mean pixel accuracy, MPA**) 计算为所有类别的**CPA**的平均值:

$$ \text{MPA} = \text{mean}(\text{CPA}) $$

```python
def meanPixelAccuracy(confusionMatrix):
    classAcc = classPixelAccuracy(confusionMatrix)
    meanAcc = np.nanmean(classAcc) # np.nanmean表示遇到Nan类型，其值取为0
    return meanAcc 
```

## ⚪ 交并比 IoU
**交并比** (**Intersection over Union, IoU**) 又称**Jaccard index**，衡量预测类别为$i$的像素集合$A$和真实类别为$i$的像素集合$B$的交集与并集之比。

$$ \text{IoU} = \frac{|A ∩ B |}{|A ∪ B|}= \frac{|A ∩ B |}{|A|+| B |-|A ∩ B |} $$

预测类别为$i$的像素集合是指所有预测为类别$i$的像素，用混淆矩阵第$i$列元素之和表示。真实类别为$i$的像素集合是指所有实际类别$i$的像素，用混淆矩阵第$i$行元素之和表示。

第$i$个类别的**IoU**计算为混淆矩阵第$i$个对角线元素比矩阵该列元素与该行元素的并集。以二分类为例，第$0$个类别的**IoU**计算为：

$$ \text{IoU} = \frac{TP}{TP+FP+FN} $$

```python
def IntersectionOverUnion(confusionMatrix):
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix) 
    IoU = intersection / union  
    return IoU # 返回列表，其值为各个类别的IoU
```

## ⚪ 平均交并比 MIoU
**平均交并比** (**mean Intersection over Union, MIoU**) 计算为所有类别的**IoU**的平均值:

$$ \text{MIoU} = \text{mean}(\text{IoU}) $$

```python
def meanIntersectionOverUnion(confusionMatrix):
    IoU = IntersectionOverUnion(confusionMatrix)
    mIoU = np.nanmean(IoU) # 求各类别IoU的平均
    return mIoU
```

## ⚪ 频率加权交并比 FWIoU
**频率加权交并比** (**Frequency Weighted Intersection over Union, FWIoU**) 按照真实类别为$i$对应像素占所有像素的比例对类别$i$的**IoU**进行加权。

第$i$个类别的**FWIoU**首先计算混淆矩阵第$i$行元素求和比矩阵所有元素求和，再乘以第$i$个类别的**IoU**。以二分类为例，第$0$个类别的**FWIoU**计算为：

$$ \text{FWIoU} = \frac{TP+FN}{TP+FP+FN+TN} \cdot \frac{TP}{TP+FP+FN} $$

最终给出的**FWIoU**应为所有类别**FWIoU**的平均值。

```python
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) +
            np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU
```

## ⚪ Dice Coefficient
**Dice Coefficient**衡量预测类别为$i$的像素集合$A$和真实类别为$i$的像素集合$B$之间的相似程度。

预测类别为$i$的像素集合是指所有预测为类别$i$的像素，用混淆矩阵第$i$列元素之和表示。真实类别为$i$的像素集合是指所有实际类别$i$的像素，用混淆矩阵第$i$行元素之和表示。

**Dice Coefficient**的计算相当于**IoU**的分子分母同时加上两个集合的交集。

$$ \text{Dice} = \frac{2|A ∩ B |}{|A|+| B |} = \frac{2\text{IoU}}{1+\text{IoU}} $$


第$i$个类别的**Dice**计算为混淆矩阵第$i$个对角线元素的两倍比矩阵该列元素与该行元素之和。以二分类为例，第$0$个类别的**Dice**计算为：

$$ \text{Dice} = \frac{2TP}{2TP+FP+FN} = \text{F1-score} $$

因此**Dice**系数等价于分类指标中的**F1-Score**。

```python
def Dice(confusionMatrix):
    # Dice = 2*TP / (TP + FP + TP + FN)
    intersection = np.diag(confusionMatrix)
    Dice = 2*intersection / (
        np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0))
    return Dice # 返回列表，其值为各个类别的Dice
```


# 3. 图像分割的损失函数


### ⚪ [<font color=Blue>Dice Loss</font>](https://0809zheng.github.io/2021/06/05/vnet.html)



# 4. 常用的图像分割数据集

图像分割任务广泛应用在自动驾驶、遥感图像分析、医学图像分析等领域，其中常用的图像分割数据集包括：

![](https://pic.imgdb.cn/item/63f3386ff144a010077ff89b.jpg)

### ⚪ [Cityscapes](https://www.cityscapes-dataset.com/)

**Cityscapes**是最常用的语义分割数据集之一，它是专门针对城市街道场景的数据集。整个数据集由 50 个不同城市的街景组成，数据集包括 5,000 张精细标注的图片和 20,000 张粗略标注的图片。

关于测试集的表现，**Cityscapes** 数据集 **SOTA** 结果近几年鲜有明显增长，**SOTA mIoU** 数值在 80 ~ 85 之间。目前 **Cityscapes** 数据集主要用在一些应用型文章如实时语义分割。

![](https://pic.imgdb.cn/item/63f2dc32f144a01007ffb60a.jpg)

### ⚪ [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

**ADE20K** 同样是最常用的语义分割数据集之一。它是一个有着 20,000 多张图片、150 种类别的数据集，其中训练集有 20,210 张图片，验证集有 2,000 张图片。近两年，大多数新提出的研究型模型（特别是 **Transformer**类的模型）都是在 **ADE20K** 数据集上检验其在语义分割任务中的性能的。

关于测试集的表现，**ADE20K** 的 **SOTA mIoU** 数值仍然在被不停刷新，目前在 55~60 之间，偏低的指标绝对值主要可以归于以下两个原因：
- **ADE20K** 数据集类别更多（150类），**mIoU** 的指标容易被其中的长尾小样本类别拖累，因而指标偏低。
- **ADE20K** 数据集图片数量更多（训练集 20,210 张，验证集 2,000 张），对算法模型性能的考验更高。

![](https://pic.imgdb.cn/item/63f2dca0f144a0100700e378.jpg)

### ⚪ [SYNTHIA](http://synthia-dataset.net)

**SYNTHIA**是计算机合成的城市道路驾驶环境的像素级标注的数据集。是为了在自动驾驶或城市场景规划等研究领域中的场景理解而提出的。提供了**11**个类别物体（分别为天空、建筑、道路、人行道、栅栏、植被、杆、车、信号标志、行人、骑自行车的人）细粒度的像素级别的标注。

![](https://pic.downk.cc/item/5ebb5eb7c2a9a83be58f1d5e.jpg)

### ⚪ [APSIS](http://xiaoyongshen.me/webpage_portrait/index.html)

人体肖像分割数据库**(Automatic Portrait Segmentation for Image Stylization, APSIS)**。

![](https://pic.downk.cc/item/5ebb5e07c2a9a83be58e8e68.jpg)



