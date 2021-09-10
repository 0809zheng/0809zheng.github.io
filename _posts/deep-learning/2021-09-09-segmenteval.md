---
layout: post
title: '图像分割的评估指标'
date: 2021-09-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/613aae6b44eaada7393f78c1.jpg'
tags: 深度学习
---

> Evaluation metrics for image segmentations.

图像分割中常用的评估指标包括：
- 像素准确率 (**pixel accuracy, PA**)
- 类别像素准确率 (**class pixel accuracy, CPA**)
- 类别平均像素准确率 (**mean pixel accuracy, MPA**)
- 交并比 (**Intersection over Union, IoU**)
- 平均交并比 (**mean Intersection over Union, MIoU**)
- 频率加权交并比 (**Frequency Weighted Intersection over Union, FWIoU**)
- **Dice Coefficient**

上述评估指标均建立在**混淆矩阵**的基础之上，因此首先介绍混淆矩阵，然后介绍这些评估指标的计算。

# 0. 混淆矩阵
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

# 1. 像素准确率 PA
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

# 2. 类别像素准确率 CPA
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

# 3. 类别平均像素准确率 MPA
**类别平均像素准确率** (**mean pixel accuracy, MPA**) 计算为所有类别的**CPA**的平均值:

$$ \text{MPA} = \text{mean}(\text{CPA}) $$

```python
def meanPixelAccuracy(confusionMatrix):
    classAcc = classPixelAccuracy(confusionMatrix)
    meanAcc = np.nanmean(classAcc) # np.nanmean表示遇到Nan类型，其值取为0
    return meanAcc 
```

# 4. 交并比 IoU
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

# 5. 平均交并比 MIoU
**平均交并比** (**mean Intersection over Union, MIoU**) 计算为所有类别的**IoU**的平均值:

$$ \text{MIoU} = \text{mean}(\text{IoU}) $$

```python
def meanIntersectionOverUnion(confusionMatrix):
    IoU = IntersectionOverUnion(confusionMatrix)
    mIoU = np.nanmean(IoU) # 求各类别IoU的平均
    return mIoU
```

# 6. 频率加权交并比 FWIoU
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

# 7. Dice Coefficient
**Dice Coefficient**衡量预测类别为$i$的像素集合$A$和真实类别为$i$的像素集合$B$之间的相似程度。

预测类别为$i$的像素集合是指所有预测为类别$i$的像素，用混淆矩阵第$i$列元素之和表示。真实类别为$i$的像素集合是指所有实际类别$i$的像素，用混淆矩阵第$i$行元素之和表示。

**Dice Coefficient**的计算相当于**IoU**的分子分母同时加上两个集合的交集。

$$ \text{Dice} = \frac{2|A ∩ B |}{|A|+| B |} = \frac{2\text{IoU}}{1+\text{IoU}} $$


第$i$个类别的**Dice**计算为混淆矩阵第$i$个对角线元素的两倍比矩阵该列元素与该行元素之和。以二分类为例，第$0$个类别的**Dice**计算为：

$$ \text{Dice} = \frac{2TP}{TP+FP+TN+FN} $$

```python
def Dice(confusionMatrix):
    # Dice = 2*TP / (TP + FP + TP + FN)
    intersection = np.diag(confusionMatrix)
    Dice = 2*intersection / (
        np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0))
    return Dice # 返回列表，其值为各个类别的Dice
```

