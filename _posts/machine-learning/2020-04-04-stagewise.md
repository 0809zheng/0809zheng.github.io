---
layout: post
title: '前向逐步回归(Stagewise Regression)'
date: 2020-04-04
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/639efdc4b1fccdcd36af0918.jpg'
tags: 机器学习
---

> Stagewise Regression.

**前向逐步回归(Stagewise Regression)**属于一种贪心算法，通过对学习权重的逐步试探，得到局部最优的结果。

该方法不对需要求得损失函数的闭式解，可以应用于各种不同的损失函数上。

算法的主要流程如下：初始化权重$w$，在每轮迭代中预设回归误差为$+\infty$，对权重$w$的每一维度特征采用步长$\epsilon$进行增大或缩小，并计算回归误差。如果回归误差有所减小，则更新权重$w$。

```python
def stageWise(X, Y, eps=0.01, n_Iter=100):
    # 获取数据信息
    N, D = np.shape(X)
    ws = np.zeros((D, 1))
    ws_optim = ws.copy()
    for i in range(n_Iter):
        lowest_Error = np.inf
        # 每一个维度进行搜索
        for d in range(D):
            # 2个搜索方向
            for sign in [-1, 1]:
                ws_temp = ws.copy()
                ws_temp[d] += eps*sign
                Y_hat = np.dot(X, ws_temp)
                # 计算误差
                error = np.sum((Y_hat-Y)*(Y_hat-Y))/N # 线性回归
                # error = np.sum((Y_hat-Y)*(Y_hat-Y))/N +0.1*np.sum(np.abs(ws_temp)**2) # 岭回归
                # error = np.sum((Y_hat-Y)*(Y_hat-Y))/N +0.1*np.sum(np.abs(ws_temp))  # LASSO回归
                # 如果误差减小则进行w保存
                if error < lowest_Error:
                    lowest_Error = error
                    ws_optim = ws_temp
    return ws_optim
```

下图给出了某数据集上，在前向逐步回归迭代过程中部分网络权重参数的变化情况。从图中可以看出，第$7$号参数首先出现显著地变化，表明该参数对应的特征重要性较高；反之$6$号参数几乎没有改变(一直是$0$)，表明该位置对应的特征不重要。

![](https://pic.imgdb.cn/item/639efdc4b1fccdcd36af0918.jpg)
