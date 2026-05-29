---
layout: post
title: 'sklearn.model_selection中StratifiedKFold与train_test_split的区别'
date: 2026-01-30
author: 郑之杰
cover: ''
tags: Python
---

> Differences between sklearn.model_selection.StratifiedKFold and sklearn.model_selection.train_test_split.

在划分训练/测试集时，`sklearn.model_selection.StratifiedKFold`和`sklearn.model_selection.train_test_split`是两个经常用到的工具。两者的功能类似：
- `train_test_split`一次性地把数据集切分成一份训练集和一份测试集；
- `StratifiedKFold`则可以划分出多份训练集和测试集，用于交叉验证。

然而，即使设置了完全相同的划分比例与随机数种子，`train_test_split` 的结果也**不等于** `StratifiedKFold` 的第一次划分结果；这触及了这两个函数内部实现的细节。

## `train_test_split`的实现细节

`train_test_split`的划分过程如下：
1. 分组：首先根据提供的标签，将所有样本的索引按类别分开；
2. 打乱**样本索引**：使用提供的随机数种子，独立地对每个类别内的样本索引列表进行随机洗牌；
3. 开始划分：从每个类别洗好牌的索引列表的开头，抓取相应数量的索引放入训练集；
4. 完成划分：所有从各个类别中抽取出来的索引组成了最终的训练集索引；剩下的所有索引则组成了测试集索引。

在[源代码](https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/model_selection/_split.py#L2226)中，该过程是通过`StratifiedShuffleSplit`这个类实现的。

```python
for _ in range(self.n_splits):
    # ... (计算每个类别应该抽取多少样本到 train (n_i) 和 test (t_i)) ...

    train = []
    test = []

    for i in range(n_classes):
        # 1. 对当前类别的所有索引进行随机洗牌
        permutation = rng.permutation(class_counts[i])
        perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

        # 2. 从洗牌后的索引列表中，连续地抽取样本
        #    - 取开头 n_i[i] 个作为训练集
        train.extend(perm_indices_class_i[: n_i[i]])
        #    - 接着取后面 t_i[i] 个作为测试集
        test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

    # ... (最后再把整个 train 和 test 列表洗牌一次) ...

    yield train, test

```

举一个具体的例子。假设某个类别的6个样本$[c_0, c_1, c_2, c_3, c_4, c_5]$，对应的样本索引是$[0,1,2,3,4,5]$；固定随机数种子$42$后，打乱的样本索引是$[3,2,5,4,1,0]$，对应的样本顺序是$[c_3, c_2, c_5, c_4, c_1, c_0]$。


构建三折交叉验证，`train_test_split`创建第一折会这样做：从洗牌后的列表中取出前4个 $[c_3, c_2, c_5, c_4]$ 作为训练集，剩余4个 $[c_1, c_0]$ 作为测试集。

```python
import numpy as np

RANDOM_STATE = 42
array_unique = np.array([0, 1, 2, 3, 4, 5])

rng_A = np.random.default_rng(RANDOM_STATE)
rng_A.shuffle(array_unique)
# [3 2 5 4 1 0]
```


## `StratifiedKFold`的实现细节

`StratifiedKFold`的划分过程如下：
1. 分组：首先根据提供的标签，将所有样本的索引按类别分开；
2. 索引分配：按照**fold=K**个箱子和样本总数，为每个类别的样本分配$[0,K-1]$的**Fold**索引；
3. 打乱**Fold索引**：使用提供的随机数种子，独立地对每个类别内的**Fold**索引列表进行随机洗牌；
4. 完成划分：最终每个样本的**Fold**索引$k$对应于它是第$k$折的测试集（其余折的训练集）。

在[源代码](https://github.com/scikit-learn/scikit-learn/blob/d3898d9d5/sklearn/model_selection/_split.py#L688)中，该过程是通过`_make_test_folds`这个方法实现的。

```python
y_order = np.sort(y_encoded)
allocation = np.asarray(
    [
        np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
        for i in range(self.n_splits)
    ]
)
# allocation 是一个 (n_splits, n_classes) 的矩阵
# allocation[i, k] 表示第 i 个Fold中应该包含 k 类的样本数量

for k in range(n_classes):
    # ...
    folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
    # 生成一个数组，其中包含了分配给所有 k 类样本的Fold索引
    # 如果 allocation[:, k] 是 [3, 3, 2]，它会生成 [0, 0, 0, 1, 1, 1, 2, 2]
    if self.shuffle:
        rng.shuffle(folds_for_class)
    test_folds[y_encoded == k] = folds_for_class
```

继续举前述例子。假设某个类别的6个样本$[c_0, c_1, c_2, c_3, c_4, c_5]$；构建三折交叉验证，则生成的**Fold**索引是$[0,0,1,1,2,2]$，在相同的随机数种子$42$下打乱顺序是$[1,1,2,2,0,0]$

`StratifiedKFold`创建第一折会这样做：取出**Fold**索引为0的样本 $[c_4, c_5]$ 作为测试集，剩余 $[c_1, c_2, c_3, c_4]$ 作为训练集。


```python
import numpy as np

RANDOM_STATE = 42
array_duplicates = np.array([0, 0, 1, 1, 2, 2])

rng_B = np.random.default_rng(RANDOM_STATE)
rng_A.shuffle(array_duplicates)
# [1 1 2 2 0 0]
```

## 总结

`train_test_split` 与 `StratifiedKFold`的关键差异在于两者应用随机数种子的列表不同：`train_test_split`打乱了样本的索引列表，而`StratifiedKFold`打乱了分配给每个类别的**Fold**索引列表。

因为输入的列表不同，即使随机数种子相同，最终的排列结果和分配方式也几乎不可能导致完全一致的划分。因此在实验中切勿将两者混用，以免造成数据不一致或标签泄漏等问题的出现。
