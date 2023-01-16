---
layout: post
title: '人脸检测, 识别与验证(Face Detection, Recognition, and Verification)'
date: 2020-05-10
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ec24f09c2a9a83be5645938.jpg'
tags: 深度学习
---

> Face Detection, Recognition, and Verification.

⭐扩展阅读：[15分钟带你深入理解人脸识别](https://www.bilibili.com/video/BV1RK411H7fB/?spm_id_from=333.999.0.0&vd_source=8a9594660e9be61c53f58a256d91aa93)

**人脸验证(Face Verification)**是指判断给定的人脸图像和用户**ID**是否匹配，是一种二分类问题；**人脸识别(Face Recognition)**是指判断给定的人脸图像属于用户数据库中的哪个人（或没有匹配），是一种多分类问题。人脸验证和识别都需要提供人脸图像，通过**人脸检测(Face Detection)**实现。


# 1. 人脸检测 Face Detection

**人脸检测**是指对于任意一幅给定的图像，采用一定的策略对其进行搜索以确定其中是否含有人脸，如果是则返回人脸的位置、大小和姿态，是人脸验证、识别、表情分析等各种问题的关键步骤。

人脸检测常用的数据集包括：
- **WIDER FACE**：人脸检测的一个基准数据集，包含32,203张图像，393703个标注人脸，其中158,989张在训练集中，39,496张在验证集中，其余的在测试集中。验证集和测试集包含 “easy”“medium”, “hard” 三个子集，且依次包含，即“hard”数据集包含所有“easy”和“medium”的图像。
- **FDDB**：FDDB是全世界最具权威的人脸检测数据集之一，包含2845张图片，共有5171个人脸作为测试集。
- **Pascal Faces**：PASCAL VOC为图像识别和分类提供了一整套标准化的优秀的数据集，Pascal Faces是PASCAL VOC的子集，共851张包含已标注人脸的图像，在本论文中仅用于评估模型性能。

### ⚪ Eigenface
**Eigenface**是一种基于机器学习的人脸检测方法。**Eigenface**的思路是把人脸图像从像素空间变换到特征空间，在特征空间中进行相似度计算。具体地，使用**主成分分析**得到人脸数据矩阵的特征向量，这些特征向量可以看作组成人脸的特征，每个人脸都可以表示成这些特征向量的线性组合。**Eigenface**算法流程如下：
1. 将训练集表示成二维张量$A \in \Bbb{R}^{d \times N}$，其中$d$表示每张人脸图像的像素数，$N$表示训练集大小；
2. 求训练集中所有图像的平均人脸$\overline{A} = A.sum(dim=1)$，训练集减去该平均后得到差值图像的数据矩阵$\Phi = A - \overline{A}$；
3. 计算数据矩阵的协方差矩阵$C = \Phi \Phi^T$，对其进行特征值分解，得到特征向量$v$；
4. 对于测试图像$T$，将其投影到这些特征向量上，计算特征向量$v_k$相对于测试图像的权重$\omega_k = v_k^T(T-\overline{A})$，取前$M$个特征向量，得到权重矩阵$\omega_T = \[ \omega_1,\omega_2,...,\omega_M \]$；
5. 对训练集中的每一张人脸图像计算权重矩阵$\omega$，比较测试图像的权重矩阵与其之间的欧式距离$$\epsilon_T = \| \omega_T - \omega \|^2$$，若距离小于阈值，则认为测试图像和该训练图像对应同一个人脸。若测试图像与训练集的所有图像距离都大于阈值，则认为测试图像不包含人脸。

在实际中，通常人脸图像的像素数较大，而训练集的样本总数较少，即$d>>N$。在计算协方差矩阵$C = \Phi \Phi^T \in \Bbb{R}^{d \times d}$时会得到较大的矩阵，不利于后续的特征值分解。此时可以先计算另一个协方差矩阵，$C' = \Phi^T \Phi \in \Bbb{R}^{N \times N}$，计算其特征值为$u$，则矩阵$C$的特征值计算为$v=\Phi u$，推导如下：

$$ C' \cdot u = \lambda \cdot u \\ \Phi^T \Phi \cdot u = \lambda \cdot u \\ \Phi \Phi^T \Phi \cdot u = \lambda \cdot \Phi u \\ C \Phi \cdot u = \lambda \cdot \Phi u \\ C \cdot v = \lambda \cdot v $$

### ⚪ [SSH: Single Stage Headless Face Detector](https://arxiv.org/abs/1708.03979)

**SSH**是一个快速、轻量级的人脸检测器，直接从分类网络中的早期卷积层以单阶段方式检测人脸。

网络结构如图所示：

![](https://pic.downk.cc/item/5ecf4f8ac2a9a83be5eda2df.jpg)

网络使用**VGG16**作为**backbone**，分成$3$路进行不同尺度的检测，使得模型对于图像中不同尺寸大小脸的检测均具有良好的鲁棒性。

**检测模块（detection module）**分为**M1**、**M2**和**M3**，分别检测小、中、大尺寸的人脸，其中又使用了**上下文模块（context module）**。上下文模块通过$2$个$3 \times 3$的卷积层和$3$个$3 \times 3$的卷积层并联，用来读取图像的文本信息。

![](https://pic.imgdb.cn/item/63c4ceabbe43e0d30e447337.jpg)

# 2. 人脸识别 Face Recognition

传统的人脸识别认为不同人脸由不同特征组成，主要思路是设计特征提取器，再利用机器学习算法对提取的特征进行分类。

深度学习中的人脸识别方法通过深度网络把校正后的人脸图像编码到特征空间里，属于同一个人的人脸图像在特征空间中距离较近；不同人的特征图像在特征空间中距离较远；具体的实现采用[<font color=blue>度量学习</font>](https://0809zheng.github.io/2022/11/01/metric.html)。

![](https://pic.downk.cc/item/5ec2400fc2a9a83be54ed95d.jpg)

训练好模型之后，就可以实现人脸验证和识别。在实际使用中，预先存储已知人脸数据集的人脸对应特征向量；对于一张新的人脸图像，先提取特征再进行比较。

### ⚪ [<font color=blue>DeepFace</font>](https://0809zheng.github.io/2020/05/09/deepface.html)

**DeepFace**使用卷积神经网络进行人脸表示的特征提取：

![](https://pic.downk.cc/item/5eb13b9bc2a9a83be5cf4f77.jpg)

使用**孪生网络 Siamese Network**训练卷积网络。对于一张人脸的图像$$x^{(1)}$$，使用网络得到一个特征向量$$f(x^{(1)})$$；对于另一张人脸的图像$$x^{(2)}$$，喂入具有同样参数的网络，得到特征向量$$f(x^{(2)})$$；

![](https://pic.downk.cc/item/5eb15059c2a9a83be5e7a20e.jpg)

训练的目标是，若两张图像是同一个人，则两个特征向量越接近越好；否则差别越大越好。

### ⚪ 人脸识别数据集

![](https://pic.downk.cc/item/5ec24091c2a9a83be54f4672.jpg)

- 2007年发布的LFW是第一个在非限定环境下进行人脸识别的数据集。
- 2014年发布的CASIA-Webface是第一个被广泛使用的公共训练集，自此之后涌现了许多大规模的训练集，如包含260万张人脸的VGGFace。 
- 上图中玫红色框内多被用来作为大规模训练集，其余则作为不同任务和场景下的测试集。

### LFW（Labeled Faces in the Wild）
专为研究非受限人脸识别问题而设计的人脸照片数据库。包含从网络收集的超过13,000张人脸图像。 每张脸都标有人物的名字，有1680人有两张或更多张不同的照片。

人脸图片均来源于生活中的自然场景，因此识别难度会增大，尤其由于多姿态、光照、表情、年龄、遮挡等因素影响导致即使同一人的照片差别也很大。

有些照片中可能不止一个人脸出现（此时仅选择中心坐标的人脸作为目标，其余视为噪声）。

LFW数据集中的图片被两两分组，这两张图片可能来自同一个人，也可能来自不同的人。模型需要做的就是判断两张照片是否来自同一个人。

![](https://pic.downk.cc/item/5ec241dbc2a9a83be550990c.jpg)

### ⚪ 人脸识别的评估指标
- **准确率 accuracy**：正确检测出的样本数与总样本数之比。
- **误识率 False Accept Rate（FAR）**：假设不同人的两张照片的类间比较次数为$$N_{IRA}$$，判断其为同一人的错误接受次数为$$N_{FA}$$，则：

$$ FAR = \frac{N_{FA}}{N_{IRA}} $$

- **拒识率 False Reject Rate（FRR）**：假设同一个人的两张照片的类内比较次数为$$N_{GRA}$$，判断其为不同人的错误拒绝次数为$$N_{FR}$$，则：

$$ FRR = \frac{N_{FR}}{N_{GRA}} $$


# 3. 人脸验证 Face Verification
训练好的人脸识别模型可以应用到人脸验证任务中。下面介绍直接解决人脸验证问题的模型。

### ⚪ [DeepID: Deep learning face representation from predicting 10,000 classes](https://www.researchgate.net/publication/283749931_Deep_Learning_Face_Representation_from_Predicting_10000_Classes)

**DeepID**模型使用人脸上不同的区域训练多个单独的ConvNet，每个ConvNet的最后一个隐层为提取到的特征，称之为**DeepID(Deep hidden IDentity feature)**。

![](https://pic.downk.cc/item/5ec24b38c2a9a83be55f463e.jpg)

- ConvNet的结构由四层CNN和一层Softmax组成。
- 输入的不是整个人脸，而是人脸中的某个区域（Patch）。输入数据有两种，一种是$39 * 31 * k$, 对应矩形区域，一种是$31 * 31 * k$，对应正方形区域，其中$k$当输入是RGB图像时为3，灰度图时为1。
- 输出的特征DeepID是Softmax层之前的隐层，Softmax层只在训练时才有。
- 训练时使用10000个人的数据集，用识别任务训练。
- 最后将不同区域提取到的DeepID连接起来作为人脸的特征，用PCA降维到150维后送入Joint Bayesian分类器（也可以是其他分类器）进行人脸验证， 此时变为二分类任务。

### ⚪ [DeepID2: Deep Learning Face Representation by Joint Identification-Verification](https://arxiv.org/abs/1406.4773)

![](https://pic.downk.cc/item/5ec24bf2c2a9a83be56047e7.jpg)

相较于DeepID，**DeepID2**中ConvNet的输入尺寸更大。模型最大的改进在于同时使用**识别**信号和**验证**信号来监督学习过程，而在DeepID中只用到了识别信号。

识别信号使用交叉熵损失：

$$ Ident(f,t,θ_{id}) = \sum_{i=1}^{n} {-p_i\log p_i} = -\log p_t $$

验证信号使用L2损失：

$$ Verif(f_i,f_j,y_{ij},θ_{ve}) = \begin{cases} \frac{1}{2} \mid\mid f_i-f_j \mid\mid^2, & if \quad y_{ij}=1 \\ \frac{1}{2} max(0,m-\mid\mid f_i-f_j \mid\mid^2), & if \quad y_{ij}=-1 \end{cases} $$

可以看出，验证信号的计算需要两个样本，故每次迭代时需要随机抽取两个样本，然后计算误差。
