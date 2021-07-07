---
layout: post
title: '绘制混淆矩阵'
date: 2021-06-24
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60d55b6a844ef46bb200dffc.jpg'
tags: Python
---

> Draw Confusion Matrix.

### 混淆矩阵简介
在机器学习领域，**混淆矩阵(confusion matrix)**通常用于评估监督学习中分类任务算法的性能（这个概念对应于非监督学习中的**匹配矩阵 matching matrix**）。混淆矩阵不仅能够显示每个类别的识别准确率，还能够显示将某个类别识别为其他类别的错误率。

### 混淆矩阵的获取
可以使用`sklearn`库计算混淆矩阵。其中`label_y`和`pred_y`都是$n \times 1$的张量，表示每个样本的真实类别和预测类别(用自然数表示)。

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label_y, pred_y,)
```

### 混淆矩阵的绘制
绘制混淆矩阵可以使用下列函数，其中`cm`是混淆矩阵的数值，`classes`是类别对应的名称列表。

```python
def confusion_metrix(cm, classes):
    # 对混淆矩阵进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 在混淆矩阵中显示每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=10, va='center', ha='center')

    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap='binary')

    # 显示渐变色条
    plt.colorbar()

    num_local = np.array(range(len(classes)))    
    plt.xticks(num_local, classes, rotation=0, fontproperties = 'Times New Roman', fontsize=10)    # 将标签印在x轴坐标上
    plt.yticks(num_local, classes, rotation=90, fontproperties = 'Times New Roman', fontsize=10)    # 将标签印在y轴坐标上，旋转90°
    
    # 把图像原点设置成左上角
    ax = plt.gca()  #获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  #将X坐标轴移到上面
    

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }

    # 显示x,y轴标签。x轴标签用图像标题代替，以此显示在上方。
    plt.ylabel('True label', font1)    
#    plt.xlabel('Predicted label', font1)
    plt.title('Predicted label', font1)
    
    fig = plt.gcf()
    plt.margins(0, 0)
    fig.tight_layout()
    
    plt.show()
    fig.savefig("./confusion_matrix.pdf", format='pdf', transparent=True, dpi=300, pad_inches=0)
```

通过上述函数绘制的混淆矩阵如下：

![](https://pic.imgdb.cn/item/60d55b6a844ef46bb200dffc.jpg)