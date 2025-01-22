---
layout: post
title: '分类任务的常用性能指标'
date: 2020-02-07
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/678e40b3d0e0a243d4f5f1f7.png'
tags: 机器学习
---

> Performance measures for classification tasks.

本文介绍监督学习的分类任务中常用的性能评价指标，包括：
1. 错误率与精度
1. 基于混淆矩阵的指标
1. 查准率-查全率(**P-R**)曲线与**F1**得分
1. 宏查准率/查全率与微查准率/查全率
1. 受试者工作特性曲线**ROC**与曲线下面积**AUC**
1. 代价敏感错误率与代价曲线

# 1. 错误率与精度
**错误率(error rate)**与**精度(accuracy)**是分类任务中最常用的性能度量指标。
- 错误率是指分类错误的样本数占总样本数的比例：

$$ E(f;D) = \frac{1}{m} \sum_{i=1}^{m} \Bbb{I}(f(x_i)≠y_i) $$

- 精度是指分类正确的样本数占总样本数的比例：

$$ acc(f;D) = \frac{1}{m} \sum_{i=1}^{m} \Bbb{I}(f(x_i)=y_i) = 1-E(f;D) $$

# 2. 基于混淆矩阵的指标
以二分类为例，每个样本可能属于**正样本(阳性,Positive)**也可能属于**负样本(阴性,Negative)**。根据样本的实际属性和模型的预测结果，可以把样本划分为以下四类中的某一类：
- **真阳性 TP(True Positive)**：实际为正样本，预测也为正样本
- **假阳性 FP(False Positive)**：实际为负样本，预测为正样本；也称为**类型I错误(Type I error)**
- **真阴性 TN(True Negative)**：实际为负样本，预测也为负样本
- **假阴性 FN(False Negative)**：实际为正样本，预测为负样本；也称为**类型II错误(Type II error)**

进而可以绘制分类结果的**混淆矩阵(confusion matrix)**如下：

$$ \begin{array}{l|cc} \text{真实情况\预测结果} & \text{正样本} & \text{负样本} \\ \hline  \text{正样本} & TP & FN \\  \text{负样本} & FP & TN \\ \end{array} $$

![](https://pic1.imgdb.cn/item/678e40b3d0e0a243d4f5f1f7.png)

- **查准率(Precision)**,又称**准确率**或**正预测值(Positive predictive value, PPV)**，定义为模型预测为正样本的所有样本中，真正为正样本的样本比例：

$$ P = \frac{TP}{TP+FP} $$

- **查全率(Recall)**,又称**召回率**或**敏感性(Sensitivity)**或**真阳性率(true positive rate, TPR)**，定义为所有真正为正样本的样本中，模型预测为正样本的样本比例：

$$ R = \frac{TP}{TP+FN} $$

查准率和查全率通常是相互矛盾的。查准率越高，即模型预测越保守，只会把有把握的少量样本分类为正样本(极端情况下，当模型只预测成功一个正样本样本时，查准率为$1$，但查全率相当低)；查全率越高，即模型预测越激进，倾向于把更多样本分类为正样本(极端情况下，当模型将所有样本预测为正样本，查全率为$1$，但查准率相当低)。

- **特异性(Specificity)**,又称**真阴性率(true negative rate, TNR)**或**选择率(selectivity)**，定义为所有真正为负样本的样本中，模型预测为负样本的样本比例：

$$ \text{TNR} = \frac{TN}{TN+FP} $$

- **假阳性率(false positive rate, FPR)**，又称**fall-out**，定义为所有真正为负样本的样本中，模型预测为正样本的样本比例：

$$ \text{FPR} = \frac{FP}{TN+FP} $$

- **假阴性率（False Negative Rate, FNR）**也称为**遗漏率(miss rate)**，定义为所有真正为正样本的样本中，模型预测为负样本的样本比例：

$$ \text{FNR} = \frac{FN}{TP+FN} $$

- **假发现率（False Discovery Rate, FDR）**定义模型预测为正样本的所有样本中，真正为负样本的样本比例：
  
$$ \text{FDR} = \frac{FP}{TP+FP} $$

- **假遗漏率（False Omission Rate, FOR）**定义模型预测为负样本的所有样本中，真正为正样本的样本比例：

$$ \text{FOR} = \frac{FN}{TN+FN} $$
​
- **负预测值（Negative Predictive Value, NPV）**定义模型预测为负样本的所有样本中，真正为负样本的样本比例：

$$ \text{NPV} = \frac{TN}{TN+FN} $$

对于多分类混淆矩阵，在评估某个类别的性能时可以将多分类混淆矩阵转化为二分类混淆矩阵，即将其他类别的样本都归为负样本。如下图所示，假设类别**0**是选定的正样本类别：

![](https://pic1.imgdb.cn/item/679055c4d0e0a243d4f657c8.png)
​
# 3. 查准率-查全率(P-R)曲线与F1得分
使用模型对所有样本进行预测，按照置信度(模型认为其属于正样本的概率)从高到低对所有样本进行排序。按照此顺序依次把样本作为正样本进行预测，每次可计算出当前位置的查准率$P$与查全率$R$。以查准率$P$为纵轴、查全率$R$为横轴可以绘制**查准率-查全率(P-R)曲线**。

![](https://pic.imgdb.cn/item/60e6988e5132923bf8fdbbc1.jpg)

最初模型把最有把握的样本预测为正样本，此时查准率$P$为$1$、查全率$R$接近$0$。按顺序逐个把样本看作正样本后查准率$P$逐渐下降、查全率$R$逐渐提高。最终所有样本都被模型认为是正样本，此时查全率$R$为$1$、查准率$P$接近$0$。

**P-R曲线**可以直观地显示模型在样本总体上的查准率和查全率。若模型$A$的**P-R曲线**能够完全“包住”模型$C$的**P-R曲线**，则认为模型$A$的性能优于模型$C$的性能。若模型$A$的**P-R曲线**和模型$B$的**P-R曲线**发生“交叉”，则很难直接比较模型$A$和模型$B$的性能。此时可以使用的判据包括：
- **P-R曲线**下面积的大小：计算比较困难
- **平衡点(Break-Even Point,BEP)**：查准率$=$查全率时的取值
- **F1-score**：查准率和查全率的**调和平均(harmonic mean)**:

$$ \frac{1}{F1} = \frac{1}{2} \cdot (\frac{1}{P}+\frac{1}{R}) $$

$$ F1 = \frac{2\times P \times R}{P+R} = \frac{2TP}{2TP+FP+FN} = \frac{2TP}{\#NUM+TP-TN}  $$

- **F$_{\beta}$-score**：查准率和查全率的**加权调和平均**；对查准率和查全率的重视程度不同，$\beta>0$衡量查全率对查准率的相对重要性：

$$ \frac{1}{F_{\beta}} = \frac{1}{1+\beta^2} \cdot (\frac{1}{P}+\frac{\beta^2}{R}) $$

$$ F_{\beta} = \frac{(1+\beta^2)\times P \times R}{\beta^2 \times P+R} $$

上式当$\beta=1$时退化为**F1-score**。$\beta>1$时(如$\beta→∞$时$F_{\beta}≈R$)查全率有更大影响；$\beta<1$时(如$\beta→0$时$F_{\beta}≈P$)查准率有更大影响。

# 4. 宏查准率/查全率与微查准率/查全率
有时需要在$n$个二分类混淆矩阵上综合考虑查准率和查全率。如通过多次训练和测试得到多个混淆矩阵，或者多分类任务中每两个类别的组合对应一个混淆矩阵。

一种做法是在每个混淆矩阵$i$上分别计算出查准率$P_i$和查全率$R_i$，再计算平均值。此时可以计算**宏查准率(macro-P)**,**宏查全率(macro-R)**和**宏F1(macro-F1)**得分：

$$ \text{macro-}P = \frac{1}{n} \sum_{i=1}^{n}P_i $$

$$ \text{macro-}R = \frac{1}{n} \sum_{i=1}^{n}R_i $$

$$ \text{macro-}F1 = \frac{2 \times \text{macro-}P \times \text{macro-}R}{\text{macro-}P+\text{macro-}R} $$

另一种做法是先对混淆矩阵的每个元素取平均，得到$\overline{TP},\overline{FP},\overline{TN},\overline{FN}$，再基于这些平均值计算**微查准率(micro-P)**,**微查全率(micro-R)**和**微F1(micro-F1)**得分：

$$ \text{micro-}P = \frac{\overline{TP}}{\overline{TP}+\overline{FP}}  $$

$$ \text{micro-}R = \frac{\overline{TP}}{\overline{TP}+\overline{FN}}  $$

$$ \text{micro-}F1 = \frac{2 \times \text{micro-}P \times \text{micro-}R}{\text{micro-}P+\text{micro-}R} $$

# 5. 受试者工作特性曲线ROC与曲线下面积AUC

### ⚪ ROC的绘制

与绘制**P-R**曲线时类似，使用模型对所有样本进行预测，按照置信度(模型认为其属于正样本的概率)从高到低对所有样本进行排序。按照此顺序依次把样本作为正样本进行预测，每次可计算出当前位置的**真阳性率(True Positive Rate,TPR)**和**假阳性率(False Positive Rate,FPR)**：
- 真阳性率$TPR$：所有真正为正样本的样本中，模型认为是正样本的样本比例(等价于查全率)：

$$ TPR = \frac{TP}{TP+FN} $$

- 假阳性率$FPR$：所有真正为负例的样本中，模型认为是正样本的样本比例：

$$ FPR = \frac{FP}{TN+FP} $$

以真阳性率$TPR$为纵轴、假阳性率$FPR$为横轴可以绘制**受试者工作特性(Receiver Operating Characteristic,ROC)曲线**。**ROC**曲线源于二战中用于敌机检测的雷达信号分析技术。假设有$m^+$个正样本和$m^-$个负样本，则真阳性率$TPR$和假阳性率$FPR$也可表示为：

$$ TPR = \frac{TP}{m^+}, \quad FPR = \frac{FP}{m^-} $$

绘制**ROC**曲线前首先需要对所有测试样本按照模型给出的预测结果按置信度从大到小进行排序，接着将分类阈值设为一个不可能取到的最大值，显然此时所有样本都被预测为负样本，真阳性率$TPR$和假阳性率$FPR$都为$0$，此时位于坐标原点$(0,0)$。接下来需要把分类阈值从大到小依次设为每个样本的预测值，然后每次计算真阳性率$TPR$和假阳性率$FPR$，并绘制下一个坐标点。设定**ROC**曲线的$x$轴步长为$\frac{1}{m^-}$，$y$轴步长为$\frac{1}{m^+}$，注意到每次绘制时会出现三种情况：
- 新增一个真正样本样本，此时$TP+=1$，$x$轴坐标不变，$y$轴坐标增加$\frac{1}{m^+}$，对应下图的绿色轨迹；
- 新增一个假正样本样本，此时$FP+=1$，$x$轴坐标增加$\frac{1}{m^-}$，$y$轴坐标不变，对应下图的红色轨迹；
- 多个样本的预测置信度相同，此时若有$i$个真正样本样本，$j$个假正样本样本，则$x$轴坐标增加$\frac{i}{m^-}$，$y$轴坐标增加$\frac{j}{m^+}$，对应下图的蓝色轨迹。

![](https://pic.imgdb.cn/item/60e6a4335132923bf85ac519.jpg)

### ⚪ AUC的计算

与**P-R**曲线相似，若模型$A$的**ROC**曲线能够完全“包住”模型$B$的**ROC**曲线，则认为模型$A$的性能优于模型$B$。若两个模型的**ROC**曲线发生“交叉”，则需要比较**ROC**曲线下的面积，即**AUC(Area Under Curve)**。

观察上图，**ROC**曲线是由坐标为$$\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$$的点按序连接而成的曲线。若将其沿$x$方向划分成若干梯形(矩形也是特殊的梯形)，则可按照梯形面积公式计算**AUC**：

$$ AUC = \sum_{i=1}^{m-1} \frac{1}{2} (y_i+y_{i+1}) \cdot (x_{i+1}-x_i) $$

### ⚪ 排序损失

理想中的模型预测应使得所有真正样本排在所有假正样本之后，此时的**ROC**曲线应该是$(0,0)→(0,1)→(1,1)$的折线。定义**排序损失**$\mathcal{l}_{rank}$为实际**ROC**曲线与理想**ROC**曲线的差异，即上图中**ROC**曲线与$y$轴围成的面积：

$$\mathcal{l}_{rank} = 1-AUC$$

同样地，把$\mathcal{l}_{rank}$拆分成一系列高为$\frac{1}{m^+}$的梯形面积之和。注意到只有每个真实例样本$x^+$才能够构成有效的梯形(对应图中绿色和蓝色路径)；此时梯形的下底为预测置信度大于该真实例样本$x^+$的假实例样本的数量乘以步长$\frac{1}{m^-}$：

$$ \frac{1}{m^-} \sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)<f(x^-)) $$

梯形的上底为预测置信度大于等于该真实例样本$x^+$的假实例样本的数量乘以步长$\frac{1}{m^-}$：

$$ \frac{1}{m^-} (\sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)<f(x^-))+\sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)=f(x^-))) $$

则每个真实例样本$x^+$对应的梯形面积为：

$$ \frac{1}{2} \cdot \frac{1}{m^+} \cdot [ \frac{2}{m^-} \sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)<f(x^-))+\frac{1}{m^-} \sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)=f(x^-))] $$

**排序损失**$\mathcal{l}_{rank}$因此计算为：

$$ \mathcal{l}_{rank} = \sum_{x^+ \in D^+}^{} \frac{1}{2} \cdot \frac{1}{m^+} \cdot [ \frac{2}{m^-} \sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)<f(x^-))+\frac{1}{m^-} \sum_{x^- \in D^-}^{} \Bbb{I} (f(x^+)=f(x^-))] \\ = \frac{1}{m^+m^-} \sum_{x^+ \in D^+}^{} \sum_{x^- \in D^-}^{} (\Bbb{I} (f(x^+)<f(x^-))+\frac{1}{2} \Bbb{I} (f(x^+)=f(x^-))) $$

# 6. 代价敏感错误率与代价曲线
实际应用中不同类型的错误所造成的后果往往是不同的。如安全检查中假正样本的危害要比假负样本更严重。因此针对分类任务设置**代价矩阵(cost matrix)**，其中$cost_{ij}$表示将第$i$类样本预测为第$j$类样本的代价：

$$
\begin{array}{l|cc}
    \text{真实情况\预测结果} & \text{负样本(第$0$类)} & \text{正样本(第$1$类)} \\
    \hline
    \text{负样本(第$0$类)} & 0 & cost_{01} \\
    \text{正样本(第$1$类)} & cost_{10} & 0 \\
\end{array}
$$

根据不同错误的代价，可以定义**代价敏感(cost-sensitive)错误率**：

$$ E(f;D;cost) = \frac{1}{m} (\sum_{x_i \in D^+}^{} \Bbb{I} (f(x_i)≠y_i) \times cost_{01} + \sum_{x_i \in D^-}^{} \Bbb{I} (f(x_i)≠y_i) \times cost_{10}) $$

**ROC**曲线是在均等代价（分类错误的损失代价相同）的前提下反映学习模型的泛化能力；而**代价曲线**是在非均等代价的前提下反映学习模型的期望总体代价。期望总体代价越小，则模型的泛化能力越强。

代价曲线的横轴为归一化的正样本概率代价，纵轴为归一化的损失代价。假设正样本概率为$p$，则归一化的正样本概率代价计算为：

$$
cost_{norm}^+ = \frac{cost_{10} \cdot p}{cost_{10} \cdot p + cost_{01} \cdot (1-p)}
$$

归一化的损失代价计算为：

$$
cost_{norm} = \frac{cost_{10} \cdot p \cdot \text{FNR} + cost_{01} \cdot (1-p) \cdot \text{FPR}}{cost_{10} \cdot p + cost_{01} \cdot (1-p)}
$$

**ROC**曲线上的每一点对应代价曲线平面上的一条线段，ROC曲线上的点$(\text{TPR},\text{FPR})$对应代价曲线上一条从$(0,\text{FPR})$到$(1,\text{FNR})$的线段，线段下的面积表示该条件下的期望总体代价。将**ROC**曲线上的每一点转化为代价平面上的一条线段，然后去所有线段的下界，围成的面积即为在所有条件下模型的期望总体代价。

![](https://pic1.imgdb.cn/item/678e3fc8d0e0a243d4f5f1bc.png)