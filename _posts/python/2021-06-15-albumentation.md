---
layout: post
title: 'Albumentations: 图像的数据增强库'
date: 2021-06-15
author: 郑之杰
cover: 'https://camo.githubusercontent.com/3bb6e4bb500d96ad7bb4e4047af22a63ddf3242a894adf55ebffd3e184e4d113/68747470733a2f2f686162726173746f726167652e6f72672f776562742f62642f6e652f72762f62646e6572763563746b75646d73617a6e687734637273646669772e6a706567'
tags: Python
---

> Albumentations: Fast and Flexible Image Augmentations.

- website：[link](https://albumentations.ai/)
- documents：[link](https://albumentations.ai/docs/)
- code：[github](https://github.com/albumentations-team/albumentations)
- paper：[Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125)

**Albumentations**是一个为图像的数据增强设计的**python**库，安装如下：

```
pip install albumentations
```

## 1. Albumentations中的数据增强方法
**Albumentations**中的数据增强方法可以分为**像素级的变换(pixel-level transforms)**和**空间级的变换(spatial-level transforms)**两类。

### ⚪ pixel-level transforms
像素级的变换只改变图像的整体像素值，不影响图像的标签（如**mask**,检测框,关键点）。适用于图像分类等任务。

### 格式转换
- [Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)：标准化像素值,`mean=(0.485, 0.456, 0.406)`像素均值;`std=(0.229, 0.224, 0.225)`像素标准差;`max_pixel_value=255.0`最大像素值
- [Downscale](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Downscale)：通过下采样再上采样降低图像的质量,`scale_min=0.25`下采样的最小尺寸;`scale_max=0.25`下采样的最大尺寸;`interpolation=0`插值方法,默认为最近邻差值
- [FromFloat](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FromFloat)：改变像素值的数据类型(通常下采样),`dtype='uint16'`目标数据类型;`max_value=None`输入像素值的最大值
- [ImageCompression](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ImageCompression)：降低图像的压缩质量,`quality_lower=99`图像质量的下界;`quality_upper=100`图像质量的上界
- [ToFloat](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToFloat)：改变像素值的数据类型(转变成**float32**),`max_value=None`输入像素值的最大值
- [ToGray](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToGray)：转变成灰度图

### 引入全局噪声
- [GaussNoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise)：增加高斯噪声,`var_limit=(10.0, 50.0)`噪声的方差范围;`mean=0`噪声的均值;`per_channel=True`每个通道独立增加噪声
- [GlassBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GlassBlur)：增加玻璃噪声,`sigma=0.7`高斯核的标准差;`max_delta=4`交换像素的最大距离;`iterations=2`迭代次数;`mode='fast'`计算模式,`fast`或`exact`
- [ISONoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ISONoise)：增加相机传感器噪声,`color_shift=(0.01, 0.05)`色调的改变方差;`intensity=(0.1, 0.5)`控制色彩强度和亮度的因子
- [RandomFog](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomFog)：图像加雾,`fog_coef_lower=0.3`加雾的最低程度;`fog_coef_upper=1`加雾的最高程度;`alpha_coef=0.08`加雾的透明度
- [RandomGamma](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma)：增加伽马噪声,`gamma_limit=(80, 120)`噪声的范围
- [RandomRain](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomRain)图像加雨,`slant_lower=-10`雨倾斜程度的最小值;`slant_upper=10`雨倾斜程度的最大值;`drop_length=20`雨的长度;`drop_width=1`雨的宽度;`drop_color=(200, 200, 200)`雨的颜色;`blur_value=7`雨的扰动程度;`brightness_coefficient=0.7`下雨时的阴天程度
- [RandomShadow](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomShadow)：图像加阴影,`shadow_roi=(0, 0.5, 1, 1)`图像出现阴影的区域;`num_shadows_lower=1`阴影区域的最小值;`num_shadows_upper=2`阴影区域的最大值;`shadow_dimension=5`阴影区域是几边形
- [RandomSnow](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSnow)：图像加雪,`snow_point_lower=0.1`最小降雪量;`snow_point_upper=0.3`最大降雪量;`brightness_coeff=2.5`雪的大小
- [RandomSunFlare](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSunFlare)：图像加太阳耀斑,`flare_roi=(0, 0, 1, 0.5))`图像出现耀斑的区域;`angle_lower=0`最小角度;`angle_upper=1`最大角度;`num_flare_circles_lower=6`最小耀斑数量;`num_flare_circles_upper=10`最大耀斑数量;`src_radius=400`耀斑半径;`src_color=(255, 255, 255)`耀斑颜色
- [RandomToneCurve](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomToneCurve)：通过色调曲线随机改变明亮区域和阴暗区域之间的关系,`scale=0.1`两个控制点之间的距离分布的标准差
- [ToSepia](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToSepia)：对图像应用棕褐色过滤器(减去较暗的色调)

### 平滑滤波：模糊图像
- [Blur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur)：随机平滑,`blur_limit=7`平滑核的最大尺寸
- [GaussianBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur)使用高斯核进行平滑,`blur_limit=(3, 7)`平滑核的尺寸范围;`sigma_limit=0`高斯核的标准差
- [MedianBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MedianBlur)：中值滤波,`blur_limit=7`平滑核的最大尺寸
- [MotionBlur](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MotionBlur)：为图像增加运动模糊,`blur_limit=7`平滑核的最大尺寸

### 锐化滤波：增强轮廓
- [Emboss](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Emboss)：使用**emboss**滤波器提取轮廓并与原图叠加;`alpha=(0.2, 0.5)`轮廓的可见度;`strength=(0.2, 0.7)`**emboss**的强度范围
- [Sharpen](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Sharpen)：锐化图像并与原图叠加;`alpha=(0.2, 0.5)`锐化图像的可见度;`lightness=(0.5, 1.0)`锐化图像的亮度

### 对比度变换
- [Equalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Equalize)：直方图均衡化,`by_channels=True`对不同通道分别进行,否则只进行Y通道;`mask=None`对指定区域的像素进行均衡化
- [CLAHE](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CLAHE)：限制对比度的自适应直方图均衡化,`clip_limit=(1,4)`对比度限制的范围;`tile_grid_size=(8,8)`网格大小
- [ColorJitter](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter)：随机改变图像的亮度,对比度和饱和度,`brightness=0.2`亮度改变的程度,比例随机从$\[max(0,1-brightness),1+brightness\]$采样;`contrast=0.2`对比度改变的程度,同上;`saturation=0.2`饱和度改变的程度,同上;`hue=0.2`色调改变的程度,比例随机从$\[-hue,hue\]$采样
- [HistogramMatching](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.HistogramMatching)：直方图匹配,`reference_images`直方图参考图像;`blend_ratio=(0.5, 1.0)`混合程度;`read_fn=lambda x: x`读取图像的函数
- [HueSaturationValue](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue)：随机改变图像的色调,饱和度和像素值,`hue_shift_limit=20`色调改变的范围;`sat_shift_limit=30`饱和度改变的范围;`val_shift_limit`像素值改变的范围
- [RandomBrightnessContrast](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)随机改变图像的亮度和对比度,`brightness_limit=0.2`亮度改变的范围;`contrast_limit=0.2`对比度改变的范围;`brightness_by_max=True`通过最大值还是均值调整对比度

### 颜色变换
- [ChannelDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelDropout)：随机丢弃图像的通道,`channel_drop_range=(1, 1)`丢弃的通道数量范围;`fill_value=0`对被丢弃的通道重新赋值
- [ChannelShuffle](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelShuffle)：对图像的通道进行随机重排
- [FancyPCA](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FancyPCA)：对图像所有像素增加在该像素集中通过PCA计算得到的固定值$\[p_1,p_2,p_3\]\[\alpha_1 \lambda_1,\alpha_2 \lambda_2,\alpha_3 \lambda_3\]^T$,`alpha=0.1`扰动特征值的程度,从$\mathcal{N}(0,\alpha)$中采样
- [InvertImg](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.InvertImg)：反转图像的像素值(255-当前值)
- [MultiplicativeNoise](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MultiplicativeNoise)：图像像素值乘以随机数,`multiplier=(0.9, 1.1)`随机数的范围;`per_channel=False`是否通道独立操作;`elementwise=False`是否像素独立操作
- [Posterize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Posterize)：减少每个颜色通道的bit数,;`num_bits=4`颜色存储的比特数
- [RGBShift](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift)：随机改变图像的像素颜色值,`r_shift_limit=20`红色分量改变的范围;`g_shift_limit=20`绿色分量改变的范围;`b_shift_limit=20`蓝色分量改变的范围
- [Solarize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Solarize)：反转图像中大于阈值像素的像素值,`threshold=128`像素阈值

### 下游任务
- [FDA](https://albumentations.ai/docs/api_reference/augmentations/domain_adaptation/#albumentations.augmentations.domain_adaptation.FDA)：通过傅里叶变换进行域自适应的风格迁移,`reference_images`目标风格图像;`beta_limit=0.1`方法的系数;`read_fn=lambda x: x`读取图像的函数
- [Superpixels](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Superpixels)：将图像转变成超像素(图像分割),`p_replace=0.1`每块超像素由其颜色均值替代的概率;`n_segments=100`超像素的估计数量;`max_size=128`图像增强的最大尺寸,`interpolation=1`插值方法



### ⚪ spatial-level transforms
空间级的变换同时改变图像及其标注(如**mask**,检测框,关键点)，适用于图像分割、目标检测、姿态估计等任务。

| Transform                                                                                                                                                                       | Image | Masks | BBoxes | Keypoints |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: | :-------: |
| [Affine](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Affine)                             | ✓     | ✓     | ✓      | ✓         |
| [CenterCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CenterCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [CoarseDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CoarseDropout)                                   | ✓     | ✓     |        |           |
| [Crop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.Crop)                                         | ✓     | ✓     | ✓      | ✓         |
| [CropAndPad](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CropAndPad)                             | ✓     | ✓     | ✓      | ✓         |
| [CropNonEmptyMaskIfExists](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CropNonEmptyMaskIfExists) | ✓     | ✓     | ✓      | ✓         |
| [ElasticTransform](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ElasticTransform)         | ✓     | ✓     |        |           |
| [Flip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Flip)                                                     | ✓     | ✓     | ✓      | ✓         |
| [GridDistortion](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GridDistortion)                                 | ✓     | ✓     |        |           |
| [GridDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GridDropout)                                       | ✓     | ✓     |        |           |
| [HorizontalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HorizontalFlip)                                 | ✓     | ✓     | ✓      | ✓         |
| [Lambda](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Lambda)                                                 | ✓     | ✓     | ✓      | ✓         |
| [LongestMaxSize](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.LongestMaxSize)                     | ✓     | ✓     | ✓      | ✓         |
| [MaskDropout](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MaskDropout)                                       | ✓     | ✓     |        |           |
| [NoOp](https://albumentations.ai/docs/api_reference/core/transforms_interface/#albumentations.core.transforms_interface.NoOp)                                                   | ✓     | ✓     | ✓      | ✓         |
| [OpticalDistortion](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.OpticalDistortion)                           | ✓     | ✓     |        |           |
| [PadIfNeeded](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.PadIfNeeded)                                       | ✓     | ✓     | ✓      | ✓         |
| [Perspective](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.Perspective)                   | ✓     | ✓     | ✓      | ✓         |
| [PiecewiseAffine](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PiecewiseAffine)           | ✓     | ✓     | ✓      | ✓         |
| [RandomCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [RandomCropNearBBox](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCropNearBBox)             | ✓     | ✓     | ✓      | ✓         |
| [RandomGridShuffle](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGridShuffle)                           | ✓     | ✓     |        |           |
| [RandomResizedCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomResizedCrop)               | ✓     | ✓     | ✓      | ✓         |
| [RandomRotate90](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.RandomRotate90)                     | ✓     | ✓     | ✓      | ✓         |
| [RandomScale](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.RandomScale)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomSizedBBoxSafeCrop)   | ✓     | ✓     | ✓      |           |
| [RandomSizedCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomSizedCrop)                   | ✓     | ✓     | ✓      | ✓         |
| [Resize](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.Resize)                                     | ✓     | ✓     | ✓      | ✓         |
| [Rotate](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate)                                     | ✓     | ✓     | ✓      | ✓         |
| [SafeRotate](https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.SafeRotate)                             | ✓     | ✓     | ✓      | ✓         |
| [ShiftScaleRotate](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate)         | ✓     | ✓     | ✓      | ✓         |
| [SmallestMaxSize](https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.SmallestMaxSize)                   | ✓     | ✓     | ✓      | ✓         |
| [Transpose](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Transpose)                                           | ✓     | ✓     | ✓      | ✓         |
| [VerticalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.VerticalFlip)                                     | ✓     | ✓     | ✓      | ✓         |

## 2. Albumentations的使用
**Albumentations**的简单使用如下：

```python
import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256), # 随机裁剪
    A.HorizontalFlip(p=0.5), # 随机水平翻转
    A.RandomBrightnessContrast(p=0.2), # 随机明亮对比度
    A.OneOf([
        A.Blur(blur_limit=3, p=0.1), # 使用随机大小的内核模糊图像
        A.MedianBlur(blur_limit=3, p=0.1), # 中值滤波
    ], p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
```

**Albumentations**的两个主要方法：
1. `A.Compose`：顺序执行内部的变换
2. `A.OneOf`：随机选择一种变换执行

**Albumentations**已经集成在**mmdetection**框架下。使用时直接修改`config`文件内的`train_pipeline`即可：

```python
albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='OneOf', transforms=[
            dict(type='Blur', blur_limit=3, p=0.5),
            dict(type='MedianBlur', blur_limit=3, p=0.5),
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Albu', transforms=albu_train_transforms),  # 数据增强
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
```
