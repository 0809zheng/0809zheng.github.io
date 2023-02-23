---
layout: post
title: 'Person-in-WiFi: Fine-grained Person Perception using WiFi'
date: 2021-03-04
author: 郑之杰
cover: 'https://img.imgdb.cn/item/60407db4360785be543f21fd.jpg'
tags: 论文阅读
---

> 使用一维WiFi阵列实现人体图像分割和姿态估计.

- paper：Person-in-WiFi: Fine-grained Person Perception using WiFi
- arXiv：[link](https://arxiv.org/abs/1904.00276)

**2D**和**3D**传感器(如**RGB**深度相机、雷达和激光雷达)已经被用于实现人体图像分割和姿态估计等感知任务。这些传感器捕获具有较高空间分辨率的二维像素或三维点云，并用卷积神经网络进行处理。作者提出了一种使用**1D**传感器，即现成的**WiFi**天线，实现人体感知任务的方法。使用**WiFi**天线比使用前述传感器更便宜、更节能，几乎没有隐私泄露问题。

作者分别使用包含$3$个天线的标准**WiFi**路由器作为发射天线和接收天线，**WiFi**信号的中心频率设置为$2.4GHz$，工作在**IEEE802.11n**协议。人体站在发射天线和接收天线之间，发射天线发射脉冲信号，在空间中穿透、折射和反射。这一过程收集环境中丰富的空间信息。信号被接收天线接收后用于人体的感知。

![](https://img.imgdb.cn/item/60408615360785be54439d7c.jpg)

然而**WiFi**天线接收的是**1D**信号，并不具有三维空间中的全部信息。直接从**1D**信号中恢复更高维度的信息是一个**ill-posed**问题。作者以$2.4GHz$为中心划分了$30$个电磁频率，即信号再接收天线上会产生$30$种不同的接收模式。之所以这样做是因为不同波长的信号可以感知不同尺度的物体。因此每帧数据会有$3 \times 3 \times 30$种不同的信号组合，输入数据表示为**通道统计信息(Channel State Information, CSI)**。

![](https://img.imgdb.cn/item/604087f6360785be5444a926.jpg)

作者使用一台固定到接收天线的**RGB**相机收集视频信号，并人工制作标签。对于人体图像分割的标签，作者使用**Mask R-CNN**生成尺寸为$1 \times 46 \times 82$的**分割掩膜(Segmentation Masks,SM)**。对于姿态估计的标签，作者使用**Openpose**生成尺寸为$26 \times 46 \times 82$的**关节点热图(Joint Heat Maps,JHMs)**，其中定义人体的$25$个关节和一个背景；使用**Openpose**生成尺寸为$52 \times 46 \times 82$的**部位亲和场(Part Affinity Fields, PAFs)**，其中$52$是人体的$26$个肢体坐标值。

网络的输入张量尺寸为$150 \times 3 \times 3$，包含$5$个样本。将其通过上采样将尺寸调整为$150 \times 96 \times 96$，再通过残差卷积、**U-Net**得到更丰富的特征图，再根据任务需求进行下采样，生成用于图像分割的**SM**和用于姿态估计的**JHMs**和**PAFs**。

![](https://img.imgdb.cn/item/60408d2e360785be5447c236.jpg)

网络的损失函数由三部分构成：

$$ \mathcal{L} = \lambda_1L_{SM} + \lambda_2L_{JHM} + \lambda_3L_{PAF} $$

权重$\lambda_i$分别设置为$0.1$、$1$和$1$。$L_{SM}$采用二元交叉熵损失。$L_{JHM}$和$L_{PAF}$直接使用$L_2$损失的效果并不好，这是因为人体关节在图像中只占有很少的像素比例，而$L_2$损失倾向于平均图像中所有像素的回归误差。因此作者在损失中引入了**马太权重(Matthew Weight)**，其命名启发于经济学中的**马太效应(Matthew Effect: the rich get richer, the poor get poorer.)**。数值越大的位置对应的损失权重越高。

$$ L_{JHM}^{(i,j,c)} = w_{(i,j,c)} \cdot || \hat{y}_{(i,j,c)}-y_{(i,j,c)} ||^2_2 $$

$$ w_{(i,j,c)} = k \cdot y_{(i,j,c)} + b \cdot \Bbb{I}(y_{(i,j,c)}) $$

![](https://img.imgdb.cn/item/6040955a360785be544c91d6.jpg)

![](https://img.imgdb.cn/item/60409605360785be544cf73c.jpg)

作者在具有$1$至$5$名测试者的环境中收集数据，收集的帧数如上表所示。对于人体分割，使用**mAP**和**mIoU**作为评估指标。对于姿态估计，使用**PCK**作为评估指标。之所以不使用**OKS**指标，是因为**OKS**指标考虑了人体的$18$个关节点，而作者使用了$25$个；**OKS**超参数是基于**COCO**数据集的统计信息，可能会引入评估偏执。

在不同的实际环境中，**WiFi**信号可能表现出不同的模式，从而对系统的部署和应用造成困难。作者提出了一种基于**GAN**的训练策略，尝试解决模型在未训练过的环境中的部署问题。首先训练一个环境判别器用于区分数据是从哪个环境采集的，再训练一个生成器把数据转化为尺寸相同的生成数据，通过对抗学习消除环境因素对数据的影响。进而将生成的数据送入训练好的系统执行任务。

![](https://img.imgdb.cn/item/6040994f360785be544ed595.jpg)
