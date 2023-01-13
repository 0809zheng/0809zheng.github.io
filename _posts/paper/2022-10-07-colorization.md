---
layout: post
title: 'Colorful Image Colorization'
date: 2022-10-07
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/63c1230abe43e0d30e00d85f.jpg'
tags: 论文阅读
---

> 通过彩色图像着色实现无监督特征学习.

- paper：[Colorful Image Colorization](https://arxiv.org/abs/1603.08511)

**着色（colorization）**是一种很强大的自监督学习任务：训练模型以对输入灰度图像进行着色；准确地说，将该图像映射到量化颜色值输出的分布上。

![](https://pic.imgdb.cn/item/63c123e8be43e0d30e01f05c.jpg)

本文中图像着色是在[**CIE Lab\***](https://en.wikipedia.org/wiki/CIELAB_color_space)颜色空间中进行的，相比于适用于物理设备的颜色空间**RGB**或**CMYK**，**Lab\***颜色更适合人类视觉；**Lab\***颜色空间包括：
- **L\***：匹配人眼对亮度的感知，$0$为黑色，$100$为白色；
- **a\***：负值代表绿色，正值代表品红色；
- **b\***：负值代表蓝色，正值代表黄色。

通常构造在量化颜色值上预测概率分布的交叉熵损失比构造原始颜色值的**L2**损失表现更好，因此对*ab*颜色空间进行量化。图a给出了栅格尺寸为$10$的*ab*量化颜色空间，则共有$313$种*ab*对的颜色取值；图b给出了*ab*值的经验概率分布；图c给出了根据*L*值决定的*ab*值的经验概率分布。

![](https://pic.imgdb.cn/item/63c12649be43e0d30e057fd9.jpg)

为了在常见颜色（通常对应较低*ab*值的常见背景如云、墙和灰尘）和罕见颜色（可能与图像中的关键对象有关）之间进行平衡，使用加权项重新平衡损失函数，以增加不常见颜色取值的损失。加权项被构造为$(1-λ)*$高斯核平滑经验概率分布$+λ*$均匀分布，其中两个分布都在量化*ab*颜色空间上定义。

实现图像着色的网络构造如下：

![](https://pic.imgdb.cn/item/63c129fdbe43e0d30e0b29d3.jpg)