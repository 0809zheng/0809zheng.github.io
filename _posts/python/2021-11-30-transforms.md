---
layout: post
title: '使用torchvision.transforms进行图像增强'
date: 2021-11-30
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/61a47e912ab3f51d915294d9.jpg'
tags: Python
---

> Image Augmentation with torchvision.transforms.

- 官方文档：[Pytorch Dcos](https://pytorch.org/vision/stable/transforms.html#)

`torchvision.transforms`提供了常用的图像变换方法，输入支持**PIL**图像或**tensor**图像。图像变换中存在一些随机性，使用下列语句设置随机数种子：

```python
import torch
torch.manual_seed(17)
```

可以同时设置多种数据增强方法，通过`Compose`方法实现：

```python
from torchvision import transforms
train_transform = transforms.Compose([,])
```

也可以设置使用增强方法的概率，通过`RandomApply`方法实现：

```python
train_transform = transforms.RandomApply(
    torch.nn.ModuleList([,]), p=0.3)
```

`torchvision.transforms`提供的图像增强方法可以分为**几何变换**和**像素变换**。

# 1. 常用的几何变换：
常用的几何变换包括：
- `transforms.CenterCrop(size)`：在图像中心进行裁剪
- `transforms.FiveCrop(size)`：裁剪图像的中心和四个角落
- `transforms.TenCrop(size)`：裁剪图像的中心和四个角落，并水平翻转
- `transforms.RandomCrop(size)`：随机裁剪图像的一部分
- `transforms.RandomHorizontalFlip(p=0.5)`：把图像水平翻转
- `transforms.RandomVerticalFlip(p=0.5)`：把图像垂直翻转
- `transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))`：随机裁剪图像的一部分并缩放到指定尺寸
- `transforms.RandomRotation(degrees)`：对图像进行随机旋转
- `transforms.RandomPerspective(distortion_scale=0.5, p=0.5)`：对图像进行透视变换
- `transforms.RandomAffine(degrees)`：以图像中心进行随机仿射变换

### ⚪ `transforms.CenterCrop(size)`

在图像中心进行裁剪。如果图像尺寸小于裁剪长度，则对图像填充$0$。主要参数如下：
- `size`：裁剪尺寸。可以输入`int`或`(h,w)`。

### ⚪ `transforms.FiveCrop(size)`

裁剪图像的中心和四个角落，返回一个图像元组。主要参数如下：
- `size`：裁剪尺寸。可以输入`int`或`(h,w)`。

### ⚪ `transforms.TenCrop(size)`

裁剪图像的中心和四个角落，再将它们水平翻转，返回一个图像元组。主要参数如下：
- `size`：裁剪尺寸。可以输入`int`或`(h,w)`。
- `vertical_flip`：使用垂直翻转代替水平翻转。

### ⚪ `transforms.RandomCrop(size)`

随机裁剪图像的一部分。主要参数如下：
- `size`：裁剪尺寸。可以输入`int`或`(h,w)`。

### ⚪ `transforms.RandomHorizontalFlip(p=0.5)`

把图像水平翻转，可以指定概率$p$。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a46e372ab3f51d91418c77.jpg) |

### ⚪ `transforms.RandomVerticalFlip(p=0.5)`

把图像垂直翻转，可以指定概率$p$。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a472bb2ab3f51d9145f8f0.jpg) |

### ⚪ `transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))`

随机裁剪图像的一部分并缩放到指定尺寸。主要参数如下：
- `size`：指定输出尺寸，可以输入`int`或`(h,w)`。
- `scale`：指定裁剪区域面积的下界和上界。数值为相对于原图的面积比例。
- `ratio`：指定裁剪区域高宽比的下界和上界。
  
| 原始图像 | 增强图像  |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed93.jpg)  |

### ⚪ `transforms.RandomRotation(degrees)`

对图像进行随机旋转。主要参数如下：
- `degrees`：设置旋转的角度范围。可以输入`float`或`(min,max)`；若输入`float`则取值为$[-\text{degrees},\text{degrees}]$。

| 原始图像 | 增强图像  |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a46a922ab3f51d913e1903.jpg)  

### ⚪ `transforms.RandomPerspective(distortion_scale=0.5, p=0.5)`

对图像进行透视变换，可以指定概率$p$。主要参数如下：
- `distortion_scale`：控制变形度，可取$[0,1]$，默认$0.5$。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a46f762ab3f51d9142b2a1.jpg) |

### ⚪ `transforms.RandomAffine(degrees)`

以图像中心进行随机仿射变换。主要参数如下：
- `degrees`：设置旋转的角度范围。可以输入`float`或`(min,max)`；若输入`float`则取值为$[-\text{degrees},\text{degrees}]$。
- `translate`：设置平移的范围$[0,1]$。可以输入`(a,b)`；则水平平移的范围取值为$[-\text{width}*a,\text{width}*a]$，垂直平移的范围取值为$[-\text{height}*b,\text{height}*b]$。
- `scale`：设置缩放的范围。可以输入`(a,b)`。
- `shear`：设置剪切的范围。若输入`a`，则沿**x**轴剪切的范围取值为$(-a,a)$；若输入`(a,b)`，则沿**x**轴剪切的范围取值为$(a,b)$；若输入`(a,b,c,d)`，则沿**x**轴剪切的范围取值为$(a,b)$，沿**y**轴剪切的范围取值为$(c,d)$。
- `fill`：设置像素填充，默认为$0$。

| 原始图像 | 旋转 | 平移 | 缩放 | 剪切 |
| :---: | :---:  | :---:  | :---:  | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a46a922ab3f51d913e1903.jpg) |![](https://pic.imgdb.cn/item/61a46a922ab3f51d913e191a.jpg) |![](https://pic.imgdb.cn/item/61a46a922ab3f51d913e1908.jpg)|![](https://pic.imgdb.cn/item/61a46a922ab3f51d913e1910.jpg)|



# 2. 常用的像素变换
常用的像素变换包括：
- `transforms.Grayscale(num_output_channels=1)`：把图像转换为灰度图
- `transforms.RandomGrayscale(p=0.1)`：按一定概率把图像转换为灰度图
- `transforms.RandomInvert(p=0.5)`：反转像素值
- `transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))`：使用高斯模糊进行图像平滑
- `transforms.RandomPosterize(bits, p=0.5)`：通过减少每个颜色通道的位数对图像进行色调分离
- `transforms.RandomSolarize(threshold, p=0.5)`：通过反转高于阈值的像素值对图像进行曝光
- `transforms.RandomAdjustSharpness(sharpness_factor, p=0.5)`：调整图像的锐化程度
- `transforms.RandomAutocontrast(p=0.5)`：自动调整图像的对比度
- `transforms.RandomEqualize(p=0.5)`：直方图均衡化
- `transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`：随机改变图像的颜色参数

### ⚪ `transforms.Grayscale(num_output_channels=1)`

把图像转换为灰度图，可以指定输出通道数(通常为$1$或$3$)。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a465322ab3f51d9138fc5d.jpg) |

### ⚪ `transforms.RandomGrayscale(p=0.1)`

按一定概率把图像转换为灰度图，可以指定概率$p$。

### ⚪ `transforms.RandomInvert(p=0.5)`

反转像素值，可以指定概率$p$。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a474e72ab3f51d914822cb.jpg) |

### ⚪ `transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))`

使用高斯模糊进行图像平滑。主要参数如下：
- `kernel_size`：高斯核的大小。
- `sigma`：高斯核的标准差，可以输入`int`或`(min,max)`。

| 原始图像 | 增强图像  |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a4746b2ab3f51d9147a382.jpg)  |

### ⚪ `transforms.RandomPosterize(bits, p=0.5)`

通过减少每个颜色通道的位数对图像进行色调分离，可以指定概率$p$。颜色通道位数越多，则能表示的颜色数量越多。主要参数如下：
- `bits`：每个颜色通道保留的位数($0$-$8$)。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a475b32ab3f51d914909d2.jpg) |

### ⚪ `transforms.RandomSolarize(threshold, p=0.5)`

通过反转高于阈值的像素值对图像进行曝光，可以指定概率$p$。主要参数如下：
- `threshold`：像素反转的阈值(超过阈值则反转)。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a4769d2ab3f51d9149fa5a.jpg) |

### ⚪ `transforms.RandomAdjustSharpness(sharpness_factor, p=0.5)`

调整图像的锐化程度，可以指定概率$p$。主要参数如下：
- `sharpness_factor`：锐化程度。取值非负，$1$为原图，$0$为模糊图像。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a477dc2ab3f51d914b4ee6.jpg) |

### ⚪ `transforms.RandomAutocontrast(p=0.5)`

自动调整图像的对比度，可以指定概率$p$。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a4787a2ab3f51d914bff43.jpg) |

### ⚪ `transforms.RandomEqualize(p=0.5)`

直方图均衡化，可以指定概率$p$。

| 原始图像 | 增强图像 |
| :---: | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a478c42ab3f51d914c50ce.jpg) |

### ⚪ `transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`

随机改变图像的颜色参数。主要参数如下：
- `brightness`：改变亮度。可以输入`float`或`(min,max)`；若输入`float`则取值为$[\max(0,1-\text{brightness}),1+\text{brightness}]$。
- `contrast`：改变对比度。可以输入`float`或`(min,max)`；若输入`float`则取值为$[\max(0,1-\text{contrast}),1+\text{contrast}]$。
- `saturation`：改变饱和度。可以输入`float`或`(min,max)`；若输入`float`则取值为$[\max(0,1-\text{saturation}),1+\text{saturation}]$。
- `hue`：改变色调。可以输入`float`或`(min,max)`；若输入`float`则取值为$[-\text{hue},\text{hue}]$，色调取值应为$[-0.5,0.5]$。


| 原始图像 | 亮度 | 对比度 | 饱和度 | 色调 |
| :---: | :---:  | :---:  | :---:  | :---:  |
| ![](https://pic.imgdb.cn/item/619f76de2ab3f51d9131ed8b.jpg) | ![](https://pic.imgdb.cn/item/61a463a02ab3f51d913733fc.jpg) |![](https://pic.imgdb.cn/item/61a463a02ab3f51d91373405.jpg) | ![](https://pic.imgdb.cn/item/61a463a02ab3f51d91373417.jpg)| ![](https://pic.imgdb.cn/item/61a463a02ab3f51d91373412.jpg)|
