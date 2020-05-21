---
layout: post
title: '人脸识别'
date: 2020-05-10
author: 郑之杰
cover: 'https://pic.downk.cc/item/5ec24f09c2a9a83be5645938.jpg'
tags: 深度学习
---

> Face Recognition.

**Face Verification**：人脸验证，$1:1$问题
- input：image + name/ID
- output：图像和人名是否匹配（yes/no）

**Face Recognition**：人脸识别，$1:K$问题
- input：image + K persons database
- output：图像属于数据库中的哪个人（或没有匹配）

传统的人脸识别认为不同人脸由不同特征组成，主要思路是设计特征提取器，再利用机器学习算法对提取特征进行分类。

**Face Recognition**的主要步骤：
1. **Face Detect**：从一张图像中检测出人脸；
2. **Face Align**：对检测到的人脸进行矫正；
3. **Face Represent**：对校正后的人脸提取特征；
4. **Face Classify**：用特征进行识别。

# 1. Face Detect
参考[人脸检测]()。

# 2. Face Align
本节介绍**DeepFace**提出的Alignment方法。

- paper：[Deepface: Closing the gap to human-level performance in face verification](https://www.researchgate.net/publication/263564119_DeepFace_Closing_the_Gap_to_Human-Level_Performance_in_Face_Verification)

![](https://pic.downk.cc/item/5eb133bec2a9a83be5c702e9.jpg)

(a) 使用$LBP$ $Histograms$和$SVR$的方法检测出$6$个**初始基准点 initial fiducial points**:两个眼睛中心、鼻尖和嘴的位置；

(b) 拟合一个对基准点的**变换**(缩放$scale$、旋转$rotate$、平移$translate$)对图像进行**裁剪**；

(c) 用$SVR$对裁剪后的图像定位$67$个**基准点**，进行**三角剖分 Delaunay triangulation**，在人脸的轮廓上添加三角形避免**不连续性 discontinuities**；

(d) 用$3D$人脸库**USF Human-ID**构建一个平均$3D$人脸模型，手工标注$67$个**基准点**；

(e) 用**generalized least squares**学习$3D$人脸和$2D$人脸之间的**映射**，并对三角形进行可视化，颜色越深代表越不可见；

(f) 根据学习到的映射把原$2D$图像中的基准点转换成$3D$图像中的基准点；

(g) 得到**端正 frontalized**的人脸图像；

(h) 把最终图像转换成$3D$模型(not used in this paper)。

# 3. Face Represent
把人脸图像编码到特征空间里，属于同一个人的人脸图像在特征空间中距离较近；不同人的特征图像在特征空间中距离较远：

![](https://pic.downk.cc/item/5ec2400fc2a9a83be54ed95d.jpg)

如何实现编码过程？传统方法：设计特征描述子。深度学习的方法：使用卷积神经网络。

本节介绍[**DeepFace**](https://www.researchgate.net/publication/263564119_DeepFace_Closing_the_Gap_to_Human-Level_Performance_in_Face_Verification)提出的特征提取网络。

使用卷积神经网络进行人脸表示的特征提取：

![](https://pic.downk.cc/item/5eb13b9bc2a9a83be5cf4f77.jpg)

C1：卷积层，输入通道数$3$，输出通道数$32$，卷积核大小$11×11$；

M2：最大池化层；

C3：卷积层，输入通道数$32$，输出通道数$16$，卷积核大小$9×9$；

L4：局部卷积层，输入通道数$16$，输出通道数$16$，卷积核大小$9×9$；

L5：局部卷积层，输入通道数$16$，输出通道数$16$，卷积核大小$7×7$；

L6：局部卷积层，输入通道数$16$，输出通道数$16$，卷积核大小$5×5$；

F7：全连接层，输出未标准化的4096维人脸特征向量；

F8：全连接层，$Softmax$分类，用来进行$Face$ $recognition$，4300维是数据库中的人数。

- **局部卷积层**：卷积核参数不共享，基于人脸的不同区域会有不同的统计特征假设；局部卷积层会导致更大的参数量，需要更多的数据支持。

如何训练这个网络呢？使用**孪生网络 Siamese Network**。

对于一张人脸的图像$$x^{(1)}$$，使用网络得到一个特征向量$$f(x^{(1)})$$；

对于另一张人脸的图像$$x^{(2)}$$，喂入具有同样参数的网络，得到特征向量$$f(x^{(2)})$$；

![](https://pic.downk.cc/item/5eb15059c2a9a83be5e7a20e.jpg)

训练的目标是，若两张图像是同一个人，则两个特征向量越接近越好；否则差别越大越好；使用一种**相似度度量 metric**衡量这种差异。

具体的实现采用**度量学习 Metric Learning**，常用的**Metric**包括：

**(1).binary classification**

选取两张人脸图像，计算出特征向量之后进行二分类，判断这两个图像是否代表同一个人：

记两张人脸图像$$x^{(i)}$$和$$x^{(j)}$$，得到的特征向量为$$f(x^{(i)})$$和$$f(x^{(j)})$$，特征向量为$k$维，

![](https://pic.downk.cc/item/5eb15453c2a9a83be5ec1087.jpg)

使用逻辑回归：

$$ \hat{y} = σ(W\mid f(x^{(i)})-f(x^{(j)}) \mid + b) $$

**(2).Weighted χ2 distance**

使用$χ^2$**相似度**代替差值：

$$ \hat{y} = σ(W\frac{(f(x^{(i)})-f(x^{(j)}))^2}{f(x^{(i)})+f(x^{(j)})} + b) $$

**(3).Triplet Loss**

- 参考论文：[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

![](https://pic.downk.cc/item/5ec2398ec2a9a83be547c98a.jpg)

**三元组损失函数 Triplet Loss**是指每次训练使用三张人脸的图像：
- **anchor**：记为$A$，经过网络得到特征向量$$f(A)$$;
- **positive**：与$anchor$是同一个人，记为$P$,经过网络得到特征向量$$f(P)$$;
- **negative**：与$anchor$不是同一个人，记为$N$,经过网络得到特征向量$$f(N)$$;

希望同一个人的特征向量接近，而不同人的特征向量差别大：

$$ \mid\mid f(A)-f(P) \mid\mid^2 ≤ \mid\mid f(A)-f(N) \mid\mid^2 $$

通常加上一个**margin**$α$:

$$ \mid\mid f(A)-f(P) \mid\mid^2 + α ≤ \mid\mid f(A)-f(N) \mid\mid^2 $$

如果没有$α$，所有的距离将会收敛到0。

则**Triplet Loss**定义为：

$$ L(A,P,N) = max(\mid\mid f(A)-f(P) \mid\mid^2 + α - \mid\mid f(A)-f(N) \mid\mid^2, 0) $$

**Triplet Hard Loss**:

在选取训练数据时，通常选择难训练的三元组(**Hard Sample Mining**)，具体可参考论文:
- [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)

**(4).Improved Triplet Loss**

- 参考论文：[Person re-identification by multi-channel parts-based CNN with improved triplet loss function](https://www.researchgate.net/publication/311610684_Person_Re-identification_by_Multi-Channel_Parts-Based_CNN_with_Improved_Triplet_Loss_Function)

![](https://pic.downk.cc/item/5ec23a04c2a9a83be5485919.jpg)

**Improved Triplet Loss**是在**Triplet Loss**的基础上，对**anchor**和**positive**的距离加上约束$β$，定义为：

$$ L(A,P,N) = max(\mid\mid f(A)-f(P) \mid\mid^2 + α - \mid\mid f(A)-f(N) \mid\mid^2, 0) \\ + max(\mid\mid f(A)-f(P) \mid\mid^2 - β, 0) $$

**(5).Quadruplet Loss**

- 参考论文：[Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/abs/1704.01719)

![](https://pic.downk.cc/item/5ec23be0c2a9a83be54a3bb6.jpg)

**Quadruplet Loss**每次训练选择四张图像，分别是属于同一个人的**anchor**和**positive**，以及另外两个人**negative1**和**negative2**。

$$ L(A,P,N_1,N_2) = max(\mid\mid f(A)-f(P) \mid\mid^2 + α - \mid\mid f(A)-f(N_1) \mid\mid^2, 0) \\ + max(\mid\mid f(A)-f(P) \mid\mid^2 + α - \mid\mid f(N_2)-f(N_1) \mid\mid^2, 0) $$


# 4. Face Classify
训练好模型之后，就可以实现人脸验证和识别。

在实际使用中，预先存储已知人脸数据集的人脸对应特征向量；对于一张新的人脸图像，先提取特征再进行比较。

# 5. 人脸识别数据集

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

# 6. 评估指标
- **准确率 accuracy**：正确检测出的样本数与总样本数之比。
- **误识率 False Accept Rate（FAR）**：假设不同人的两张照片的类间比较次数为$$N_{IRA}$$，判断其为同一人的错误接受次数为$$N_{FA}$$，则：

$$ FAR = \frac{N_{FA}}{N_{IRA}} $$

- **拒识率 False Reject Rate（FRR）**：假设同一个人的两张照片的类内比较次数为$$N_{GRA}$$，判断其为不同人的错误拒绝次数为$$N_{FR}$$，则：

$$ FRR = \frac{N_{FR}}{N_{GRA}} $$

# 7. 人脸验证
前述方法主要解决的是人脸识别问题，训练好的网络可以应用到人脸验证任务中。下面介绍一个直接解决人脸验证问题的模型。

### DeepID
- 论文：[Deep learning face representation from predicting 10,000 classes](https://www.researchgate.net/publication/283749931_Deep_Learning_Face_Representation_from_Predicting_10000_Classes)

**DeepID**主要针对的是人脸验证任务。模型使用人脸上不同的区域训练多个单独的ConvNet，每个ConvNet的最后一个隐层为提取到的特征，称之为**DeepID(Deep hidden IDentity feature)**。

![](https://pic.downk.cc/item/5ec24b38c2a9a83be55f463e.jpg)

- ConvNet的结构由四层CNN和一层Softmax组成。
- 输入的不是整个人脸，而是人脸中的某个区域（Patch）。输入数据有两种，一种是$39 * 31 * k$, 对应矩形区域，一种是$31 * 31 * k$，对应正方形区域，其中$k$当输入是RGB图像时为3，灰度图时为1。
- 输出的特征DeepID是Softmax层之前的隐层，Softmax层只在训练时才有。
- 训练时使用10000个人的数据集，用识别任务训练。
- 最后将不同区域提取到的DeepID连接起来作为人脸的特征，用PCA降维到150维后送入Joint Bayesian分类器（也可以是其他分类器）进行人脸验证， 此时变为二分类任务。

### DeepID2
- 论文：[Deep Learning Face Representation by Joint Identification-Verification](https://arxiv.org/abs/1406.4773)

![](https://pic.downk.cc/item/5ec24bf2c2a9a83be56047e7.jpg)

相较于DeepID，**DeepID2**中ConvNet的输入尺寸更大。模型最大的改进在于同时使用**识别**信号和**验证**信号来监督学习过程，而在DeepID中只用到了识别信号。

识别信号使用交叉熵损失：

$$ Ident(f,t,θ_{id}) = \sum_{i=1}^{n} {-p_ilogp_i} = -logp_t $$

验证信号使用L2损失：

$$ Verif(f_i,f_j,y_{ij},θ_{ve}) = \begin{cases} \frac{1}{2} \mid\mid f_i-f_j \mid\mid^2, & if \quad y_{ij}=1 \\ \frac{1}{2} max(0,m-\mid\mid f_i-f_j \mid\mid^2), & if \quad y_{ij}=-1 \end{cases} $$

可以看出，验证信号的计算需要两个样本，故每次迭代时需要随机抽取两个样本，然后计算误差。