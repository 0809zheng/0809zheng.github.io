---
layout: post
title: '图像分割(Image Segmentation)'
date: 2020-05-07
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-000-63f2c620.jpg'
tags: 深度学习
---

> Image Segmentation.

**图像分割（image segmentation）**为图像中的像素分配区域归属。常见任务可从输出结构、交互方式和标签空间等不同维度描述：
- **语义分割（semantic segmentation）**：为每个像素预测语义类别，不区分同类对象的不同实例；
- **实例分割（instance segmentation）**：同时预测对象类别，并区分同一类别的不同实例；
- **全景分割（panoptic segmentation）**：对可数的**thing**类别区分实例，对不可数的**stuff**类别预测语义区域，并要求每个有效像素只属于一个输出区域；
- **提示分割（promptable segmentation）**：根据点、框、掩码、文本或参考图像等提示返回目标区域，输出不一定包含语义类别；
- **开放词汇分割（open-vocabulary segmentation）**：利用视觉—语言表示识别训练标签集合之外的文本类别。

本文目录：
1. 图像分割模型
2. 图像分割中的通用技巧
3. 图像分割的评估指标
4. 图像分割的损失函数
5. 常用的图像分割数据集

# 1. 图像分割模型

图像分割模型处理输入图像并输出与像素或区域对应的预测；根据任务不同，输出可以是逐像素类别、带类别的实例掩码，或由提示指定的类无关掩码。

给定图像$x\in\mathbb R^{H\times W\times 3}$，传统语义分割器为位置$u$预测类别分布$p_\theta(c\mid x,u)$，输出为：

$$\hat y_u=\mathop{\arg\max}_{c\in\{1,\ldots,C\}}p_\theta(c\mid x,u)$$

实例分割和现代通用分割器更常预测无序的区域集合$\{(p_i,m_i)\}_{i=1}^{N}$，其中$p_i$是类别分布，$m_i\in[0,1]^{H\times W}$是掩码。语义分割可将这些查询组合为逐像素类别得分，实例与全景分割则保留区域身份。这两种输出表示对应两条发展路线：一条不断改进逐像素分类的特征分辨率与上下文，另一条把分割改写为区域集合预测。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-001-63f2e1ae.jpg)

图像分割模型通常采用**编码器—解码器（encoder-decoder）**结构：编码器提取多尺度视觉特征，解码器将其恢复为像素、区域或掩码预测。模型的发展主线可以大致总结为：
- 全卷积与实时网络：**FCN、SegNet、RefineNet、U-Net、V-Net、UNet++、Attention U-Net、nnU-Net、ICNet、BiSeNet、DFANet、SegNeXt**；
- 上下文与多尺度模块：**DeepLab、PSPNet、UPerNet、EncNet、PSANet、APCNet、DMNet、HRNet、OCRNet、PointRend、K-Net**；
- 实例与全景分割：**Mask R-CNN、YOLACT、SOLO、Panoptic FPN、Panoptic-DeepLab**；
- **Transformer**与掩码分类：**SETR、TransUNet、SegFormer、Segmenter、MaskFormer、Mask2Former、OneFormer**；
- 提示与基础模型：**DEXTR、SAM、SegGPT、SEEM、SAM 2、SAM 3**；
- 开放词汇与语言驱动分割：**CLIPSeg、MaskCLIP、OpenSeg、LSeg、GroupViT、SAN、ODISE、CAT-Seg、FC-CLIP、LISA、Grounded SAM**。

## (1) 基于全卷积网络的图像分割模型

标准卷积神经网络包括卷积层、下采样层和全连接层。早期基于深度学习的图像分割模型为生成与输入图像尺寸一致的分割结果，丢弃了全连接层，并引入一系列上采样操作。因此这一阶段的模型旨在解决如何更好从卷积下采样中恢复丢掉的信息损失，逐渐形成了以**U-Net**为核心的对称编码器-解码器结构。

### ⚪ **FCN**：把分类网络改造成端到端逐像素预测器
- **paper**：[**Fully Convolutional Networks for Semantic Segmentation**](https://arxiv.org/abs/1411.4038)

**FCN**把分类网络末端的全连接层改写为卷积层，使网络能够接收任意尺寸图像并输出保留空间布局的类别热图。设骨干网络的总输出步幅为$s$，低分辨率类别得分先通过可学习的转置卷积上采样$s$倍；转置卷积可用双线性插值核初始化，并与整个网络一起端到端训练。

只从最深层恢复分辨率的**FCN-32s**具有强语义但边界粗糙。**FCN-16s**先在**pool4**特征上接$1\times1$卷积，得到同样$C$通道的类别得分，再与$2$倍上采样后的深层得分逐元素相加；**FCN-8s**继续以同样方式融合**pool3**的类别得分。新增的得分层以零初始化，使训练从粗预测平滑过渡。这里的跳跃连接融合的是类别得分，而后来的**U-Net**通常在特征维度执行拼接。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-002-63f3294f.jpg)

### ⚪ **SegNet**：用最大池化索引恢复空间结构
- **paper**：[**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**](https://arxiv.org/abs/1511.00561)

**SegNet**采用与**VGG16**卷积部分对应的编码器—解码器。编码器在每次$2\times2$最大池化时保存窗口内最大值的位置索引；解码器将低分辨率激活放回这些索引位置，其余位置补零，再通过卷积把稀疏特征变为稠密特征。

这种反池化没有需要学习的上采样参数，并比保存整张跳跃特征图节省推理内存。代价是非最大激活已在池化时丢失，索引只能提供几何位置，不能像**U-Net**跳跃连接那样直接传递浅层纹理。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-003-5ebb64bc.jpg)


### ⚪ **RefineNet**：逐级融合多分辨率残差特征
- **paper**：[**RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation**](https://arxiv.org/abs/1611.06612)

**RefineNet**把预训练**ResNet**四个阶段的特征组织成多路径级联：最深层先形成低分辨率语义表示，随后每个**RefineNet block**接收上一阶段输出和当前分辨率的骨干特征，逐级恢复高分辨率预测。块内先用**Residual Convolution Unit**适配各输入，再把低分辨率分支上采样到共同尺寸并求和。

融合后的**Chained Residual Pooling**串联多个池化分支，每一级都从前一级结果继续扩大感受野，并通过残差求和保留原特征。所有组件都采用恒等映射式残差路径，使预训练特征能够沿长距离直接传播，实现了高分辨率细节与深层上下文的渐进融合。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-004-5ebcea7a.jpg)

### ⚪ **U-Net**：用同尺度跳跃连接补回定位细节
- **paper**：[**U-Net: Convolutional Networks for Biomedical Image Segmentation**](https://arxiv.org/abs/1505.04597)

**U-Net**由收缩路径和对称的扩张路径组成。收缩路径重复卷积与最大池化以获得语义和上下文；扩张路径用上采样卷积提高分辨率，并与收缩路径中相同尺度的特征沿通道维拼接，再通过卷积联合处理。若编码器第$l$层特征为$E_l$、解码器上一层为$D_{l+1}$，其基本更新可写为：

$$D_l=\phi_l\!\left([E_l,\operatorname{Up}(D_{l+1})]\right)$$

跳跃特征为解码器提供池化前的边界与纹理信息，深层分支则提供判断区域类别所需的上下文。原始模型使用无填充卷积，因此拼接前需要裁剪编码器特征；现代实现多使用保持尺寸的填充。它在少量医学图像上配合弹性形变增强获得良好效果。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-005-63f32f2f.jpg)

### ⚪ **V-Net**：用三维卷积与Dice目标分割体数据
- **paper**：[**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**](https://arxiv.org/abs/1606.04797)

**V-Net**直接接收三维医学体数据。每个阶段使用一到三层$5\times5\times5$卷积建模相邻切片的体上下文，并学习残差函数；阶段之间以$2\times2\times2$、步长为$2$的卷积代替最大池化完成下采样，激活函数为**PReLU**。解码端使用转置卷积恢复体素分辨率，编码端特征也通过跳跃连接传给对应解码阶段。

为缓解前景器官远少于背景体素的问题，论文直接最大化可微分的**Dice**系数。设$p_i$为体素$i$的前景概率、$g_i\in\{0,1\}$为标签：

$$D=\frac{2\sum_i p_ig_i}{\sum_i p_i^2+\sum_i g_i^2}$$

该目标按前景重叠归一化，不需要手工设置类别权重。三维网络能利用层间结构，但显存开销随体积尺寸迅速增加，实践中常需要裁块训练、各向异性卷积或二维/三维混合设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-006-63f96706.jpg)

### ⚪ **M-Net**：用相邻切片与多尺度侧路分割脑结构
- **paper**：[**M-Net: A Convolutional Neural Network for Deep Brain Structure Segmentation**](https://ieeexplore.ieee.org/document/7950555)

**M-Net**面向三维脑部影像，但没有直接运行完整三维编码器。模型把目标切片及其相邻切片输入一次三维卷积，将层间上下文压缩成二维特征，再交给二维**U-Net**主体处理，从而在体上下文和显存开销之间折中。

网络左侧的**left leg**对原始输入连续池化，并把不同尺度图像注入相应编码阶段；右侧的**right leg**把各解码尺度的预测上采样到原始分辨率后融合，形成多尺度深度监督。它同时增强输入尺度和输出尺度，但仍依赖固定切片方向，不能完全替代各向同性三维建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-007-60db0019.jpg)


### ⚪ **W-Net**：以重建与图割目标学习无监督分割
- **paper**：[**W-Net: A Deep Model for Fully Unsupervised Image Segmentation**](https://arxiv.org/abs/1711.08506)

**W-Net**串联两个**U-Net**：第一个网络把图像映射为$K$通道的像素归属概率，第二个网络从该软分割重建原图。仅用重建误差会允许模型把颜色和纹理任意编码到通道中，因此训练还加入可微的**soft normalized cut**目标，使相似且空间邻近的像素倾向于进入同一区域、弱连接区域倾向于分离。

训练交替优化分割网络的**soft-Ncut**和整个网络的重建误差，推理后再用全连接**CRF**与层次区域合并细化结果。输出通道是无监督区域编号。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-008-60dbc55a.jpg)

### ⚪ **Y-Net**：联合组织分割与病理诊断
- **paper**：[**Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images**](https://arxiv.org/abs/1806.01313)

**Y-Net**以共享编码器同时解决乳腺活检图像的组织分割和诊断分类。模型以图像块为输入：分割解码器输出像素级组织标签图；另一条分支接在编码器瓶颈处，判断该图像块是否具有诊断判别性。一条编码路径分叉到两端，网络因此呈现“**Y**”形。

诊断阶段不直接用编码特征分类整张切片，而是只保留判别性图像块的分割结果，统计各组织类型的频率与共现关系，形成切片级描述子，再由多层感知机输出良性、非典型增生、原位癌和浸润癌等诊断类别。分割图因此成为可解释的中间证据。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-009-60dc5141.jpg)

### ⚪ **UNet++**：用嵌套稠密跳跃连接缩小语义差距
- **paper**：[**UNet++: A Nested U-Net Architecture for Medical Image Segmentation**](https://arxiv.org/abs/1807.10165)

**UNet++**认为直接拼接浅层编码特征与深层解码特征时，两者语义层级差异过大，因此在跳跃路径中加入一串卷积节点。记$x^{i,j}$为第$i$个尺度、跳跃路径第$j$个节点，则$j>0$时它同时聚合同尺度此前节点与更低分辨率节点的上采样结果：

$$x^{i,j}=\mathcal H\!\left([x^{i,0},\ldots,x^{i,j-1},\operatorname{Up}(x^{i+1,j-1})]\right)$$

这种嵌套稠密连接让特征在送入解码器前逐步获得更强语义。多个不同深度的输出头提供深度监督；推理时既可平均这些输出，也可剪去较深路径以交换精度与速度。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-010-63f97021.jpg)

### ⚪ **Attention U-Net**：在跳跃连接中抑制无关区域
- **paper**：[**Attention U-Net: Learning Where to Look for the Pancreas**](https://arxiv.org/abs/1804.03999)

**Attention U-Net**在编码特征进入跳跃连接前加入**Attention Gate**。门控同时接收高分辨率编码特征$x$和来自更深解码层的门控信号$g$，二者经线性投影、相加和非线性变换后产生空间系数：

$$\alpha=\sigma\!\left(\psi^\top\operatorname{ReLU}(W_xx+W_gg+b)\right),\qquad \hat x=\alpha\odot x$$

深层门控信号提供“当前需要寻找什么”的语义，高分辨率分支提供“目标具体在哪里”的细节。门控调制传给解码器的空间位置，它尤其适合目标较小、背景结构复杂的医学分割。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-011-63f97532.jpg)

### ⚪ **GRUU-Net**：用卷积门控循环单元迭代细化细胞掩码
- **paper**：[**GRUU-Net: Integrated convolutional and gated recurrent neural network for cell segmentation**](https://www.sciencedirect.com/science/article/pii/S1361841518306753)

**GRUU-Net**把卷积特征提取与**ConvGRU**循环状态结合起来，使模型在多次迭代中保留并修正此前的分割先验。其**Full-Resolution Dense Unit（FRDU）**始终维护全分辨率循环特征，同时接收下采样**CNN**分支提供的多尺度语义；门控机制决定旧状态保留多少、当前证据写入多少。

密集连接和**U**形路径继续提供局部到全局特征，循环状态则把一次前向预测变为迭代细化过程。它适合边界模糊、相邻细胞粘连的显微图像，但循环展开增加推理延迟，循环次数也是额外计算—精度权衡。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-012-640ae613.jpg)

### ⚪ **nnU-Net**：自动配置医学图像分割流程
- **paper**：[**nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation**](https://arxiv.org/abs/1809.10486)、[**nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation**](https://www.nature.com/articles/s41592-020-01008-z)

**nnU-Net**把医学分割中依赖经验的预处理、网络拓扑和训练设置系统化。它先从数据集中提取“指纹”，包括图像尺寸、体素间距、模态和强度分布等，再把配置分为三类：
- 固定参数：基础**U-Net**模板、优化器、学习率策略、数据增强和损失函数；
- 规则参数：根据数据指纹和显存预算自动确定重采样间距、强度归一化、裁块尺寸、批量大小、下采样次数和卷积核形状；
- 经验参数：通过交叉验证在**2D U-Net**、**3D full-resolution U-Net**与**3D cascade**之间选择或集成，并决定是否保留最大连通域等后处理。

训练使用**Dice**与交叉熵的组合，并对解码器多个分辨率施加深度监督。**nnU-Net**在大量医学分割挑战中表现突出，说明配置合理的标准**U-Net**往往比局部结构改动更重要。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-056-nnunet.png)

### ⚪ **ICNet**：以多分辨率级联平衡速度与精度
- **paper**：[**ICNet for Real-Time Semantic Segmentation on High-Resolution Images**](https://arxiv.org/abs/1704.08545)

**ICNet**按分辨率分配计算。原图$1/4$分辨率的输入通过较完整的**PSPNet**式骨干获取语义上下文；$1/2$分辨率的输入只经过骨干前几层，并与低分辨率分支共享权重；全分辨率输入只经过三层轻量步长卷积，提供边界细节。三路特征相对原图的输出步幅依次为$32、16、8$。

**Cascade Feature Fusion（CFF）**单元把低分辨率特征$F_1$上采样$2$倍并经空洞卷积对齐，再与经$1\times1$卷积投影的高分辨率特征$F_2$相加：

$$F'=\operatorname{ReLU}\!\left(\operatorname{BN}(\operatorname{DilConv}(\operatorname{Up}(F_1)))+\operatorname{BN}(\operatorname{Conv}_{1\times1}(F_2))\right)$$

训练时各级融合输出都用相应尺度的标签监督，形成由粗到细的**cascade label guidance**。重计算只发生在小图上，高分辨率分支只负责细化，因此模型能在$1024\times2048$的城市场景图像上实时运行。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-057-icnet.png)

### ⚪ **BiSeNet**：分离空间细节与语义上下文路径
- **paper**：[**BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation**](https://arxiv.org/abs/1808.00897)

**BiSeNet**把实时分割中互相冲突的需求拆成两条路径。**Spatial Path**只使用三层步长卷积，把图像降到$1/8$分辨率并保留较宽通道，以维护边缘和位置；**Context Path**采用轻量骨干快速降采样，通过全局平均池化和**Attention Refinement Module**补充大感受野语义。

**Feature Fusion Module**先拼接两路特征，再通过全局池化生成通道权重，对融合结果执行残差式重标定。训练还在上下文路径的中间层加入辅助监督。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-013-640981f1.jpg)

### ⚪ **BiSeNet V2**：用引导聚合连接细节与语义分支
- **paper**：[**BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation**](https://arxiv.org/abs/2004.02147)

**BiSeNet V2**重新设计了不依赖预训练骨干的双分支网络。宽而浅的**Detail Branch**在较高分辨率上使用普通卷积保存局部结构；窄而深的**Semantic Branch**通过**Stem Block、Gather-and-Expansion Layer**和**Context Embedding Block**低成本提取上下文，其中深度可分离卷积减少计算量。

**Bilateral Guided Aggregation Layer**让语义分支生成空间门控来筛选细节特征，同时让细节分支调制语义特征，最后相加融合。多个语义阶段的辅助头只在训练时使用。这种互相引导比固定融合更能抑制细节分支中的纹理噪声。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-014-6409878f.jpg)

### ⚪ **DFANet**：跨轻量分支聚合深层特征
- **paper**：[**DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation**](https://arxiv.org/abs/1904.02216)

**DFANet**采用通道数更小的修改版**Xception**作为轻量骨干，并级联三个骨干。**Sub-network Aggregation**把前一个骨干的最终输出上采样后作为下一个骨干的输入，相当于以较低成本反复细化高层特征；**Sub-stage Aggregation**把前一个骨干各阶段的特征传给下一个骨干的对应阶段，使同一分辨率的表示跨骨干复用并继续扩大感受野。

每个骨干末端接全连接注意力模块（**fc attention**），用全局向量重标定通道。解码端把各骨干较浅阶段的特征融合为细节表示，再与各骨干上采样后的深层输出融合并预测。两种聚合以较少通道和较低分辨率换取深层特征复用，使模型适合实时分割。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-015-63fc01e5.jpg)

### ⚪ **SegNeXt**：用卷积注意力建模多尺度上下文
- **paper**：[**SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation**](https://arxiv.org/abs/2209.08575)

**SegNeXt**的分层**MSCAN**编码器先以深度卷积混合局部空间信息，再在**Multi-Scale Convolutional Attention（MSCA）**中并联多组条带深度卷积。较长的$1\times k$与$k\times1$卷积近似大二维卷积核，以较低参数量覆盖不同尺度；融合结果经$1\times1$卷积生成空间注意力并与输入逐元素相乘：

$$Y=\operatorname{Conv}_{1\times1}\!\left(\sum_j\operatorname{Scale}_j(\operatorname{DWConv}(X))\right)\odot X$$

轻量级**Hamburger**解码头统一后三个阶段的通道与分辨率，并用矩阵分解式全局上下文模块进一步聚合区域信息。它说明大感受野和内容自适应权重并不必然依赖二次复杂度的自注意力。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-016-642ff205.jpg)


## (2) 基于上下文模块的图像分割模型

多尺度问题是指当图像中的目标对象存在不同大小时，分割效果不佳的现象。比如同样的物体，在近处拍摄时物体显得大，远处拍摄时显得小。解决多尺度问题的目标就是不论目标对象是大还是小，网络都能将其分割地很好。

随着图像分割模型的效果不断提升，分割任务的主要矛盾逐渐从恢复像素信息逐渐演变为如何更有效地利用上下文(**context**)信息，并基于此设计了一系列用于提取多尺度特征的网络结构。

这一时期的分割网络的基本结构为：首先使用预训练模型(如**ResNet**)提取图像特征(通常$8 \times$下采样)，然后应用精心设计的**上下文模块**增强多尺度特征信息，最后对特征应用上采样(通常为$8 \times$上采样)和$1\times 1$分割头生成分割结果。

有一些方法把自注意力机制引入图像分割任务，通过自注意力机制的全局交互性来捕获视觉场景中的全局依赖，并以此构造上下文模块。对于这些方法的讨论详见[<font color=Blue>卷积神经网络中的自注意力机制</font>](https://0809zheng.github.io/2020/11/21/SAinCNN.html)。

### ⚪ **DeepLab v1**：用空洞卷积维持特征分辨率
- **paper**：[**Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**](https://arxiv.org/abs/1412.7062)

**DeepLab v1**去掉分类网络后部的部分下采样，并把后续卷积改为空洞卷积。对一维信号，扩张率$r$的空洞卷积为：

$$y[i]=\sum_k x[i+r\cdot k]w[k]$$

它在不增加参数量和不继续降低特征分辨率的情况下扩大感受野，使骨干输出步幅可从$32$降到$8$。粗分割经双线性插值恢复到原图后，再用全连接**CRF**依据颜色与位置相似性建立像素对势，修正边界。空洞卷积负责语义特征，**CRF**负责低层外观一致性。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-017-63f333ec.jpg)

### ⚪ **DeepLab v2**：用ASPP并行采样多尺度上下文
- **paper**：[**DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs**](https://arxiv.org/abs/1606.00915)

**DeepLab v2**把单一扩张率推广为**Atrous Spatial Pyramid Pooling（ASPP）**：在同一特征图上并联扩张率$6、12、18、24$的$3\times3$空洞卷积，各分支从不同有效感受野观察目标，再把类别得分相加。网络骨干由**VGG**升级为残差网络，并继续使用全连接**CRF**后处理。

**ASPP**与图像金字塔的区别是它只计算一次骨干特征，再改变卷积采样间隔；因此计算更省，但当扩张率过大时，有效采样点可能落到特征边界之外并产生栅格效应。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-018-63f724f6.jpg)

### ⚪ **DeepLab v3**：把局部空洞采样与全局图像上下文结合
- **paper**：[**Rethinking Atrous Convolution for Semantic Image Segmentation**](https://arxiv.org/abs/1706.05587)

**DeepLab v3**把残差网络后续阶段改造成不同扩张率的级联模块，并重新设计**ASPP**。新模块包含$1\times1$卷积、扩张率$6、12、18$的$3\times3$空洞卷积以及图像级全局平均池化分支；所有分支拼接后再用$1\times1$卷积融合。全局分支弥补大扩张率在小特征图上退化为近似$1\times1$卷积的问题。

模型使用**Batch Normalization**稳定各分支训练，并移除**CRF**后处理。输出步幅可在$16$和$8$之间调整：更小步幅保留更多空间细节，但显著增加计算与显存。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-019-5ebcde6b.jpg)


### ⚪ **DeepLab v3+**：用轻量解码器恢复边界
- **paper**：[**Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**](https://arxiv.org/abs/1802.02611)

**DeepLab v3+**把**DeepLab v3**视为编码器：深层**ASPP**先提取多尺度语义，再上采样$4$倍；与此同时，骨干中的浅层高分辨率特征用$1\times1$卷积压缩通道。两者拼接后经过两层$3\times3$卷积细化，最后再上采样到输入尺寸。浅层分支只提供有限通道，避免低层纹理淹没深层语义。

论文还把**Xception**中的深度可分离卷积与空洞卷积结合为**atrous separable convolution**，将空间卷积和通道混合拆开以降低计算。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-020-5ebce009.jpg)

上述**Deeplab**模型的对比如下：

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-021-63f729f8.jpg)

### ⚪ **PSPNet**：用金字塔池化注入场景先验
- **paper**：[**Pyramid Scene Parsing Network**](https://arxiv.org/abs/1612.01105)

**PSPNet**的**Pyramid Pooling Module（PPM）**把骨干特征分别自适应平均池化为$1\times1、2\times2、3\times3、6\times6$网格。每个分支经$1\times1$卷积压缩通道并双线性上采样到原尺寸，再与原特征拼接。$1\times1$分支提供整幅图像的场景先验，较细网格保留子区域布局，使像素分类同时参考局部、区域和全局上下文。

主干采用空洞卷积维持$1/8$分辨率，并在中间残差阶段添加权重较小的辅助分割损失以改善深层优化。**PPM**按固定网格聚合，计算稳定，但网格本身不随对象形状变化，这也促使后续方法研究自适应区域上下文。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-022-63f86f67.jpg)

### ⚪ **UPerNet**：统一解析场景、对象、部件与材质
- **paper**：[**Unified Perceptual Parsing for Scene Understanding**](https://arxiv.org/abs/1807.10221)

**UPerNet**在最深骨干特征上应用**Pyramid Pooling Module**获取全局场景信息，再通过**FPN**自顶向下融合多尺度特征。对象和部件分割使用融合后的高分辨率语义特征，材质与纹理等更依赖局部细节的任务可连接较浅层特征，场景分类则从全局表示预测。

不同数据集往往只标注部分任务，模型因此为场景、对象、部件、材质和纹理分别设置输出头，并仅对当前样本拥有的标签计算损失。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-024-64082f5f.jpg)

### ⚪ **EncNet**：用可学习视觉字典编码全局上下文
- **paper**：[**Context Encoding for Semantic Segmentation**](https://arxiv.org/abs/1803.08904)

**EncNet**的**Context Encoding Module**学习$K$个码字$d_k$及其平滑因子$s_k$。对局部特征$x_i$，模型根据其与码字的距离计算软分配$a_{ik}$，并聚合残差：

$$
e_k=\sum_i a_{ik}(x_i-d_k),\\ a_{ik}=\frac{\exp(-s_k\|x_i-d_k\|^2)}{\sum_j\exp(-s_j\|x_i-d_j\|^2)}
$$

所有编码残差汇总成全局场景向量，用于生成通道缩放系数并重标定原特征。额外的**Semantic Encoding Loss（SE-loss）**从该向量预测图像中有哪些类别，以多标签监督强化场景先验。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-025-63fb12bc.jpg)

### ⚪ **PSANet**：为每个像素学习全图信息传播权重
- **paper**：[**PSANet: Point-wise Spatial Attention Network for Scene Parsing**](https://hszhao.github.io/projects/psanet/)

**PSANet**用卷积为每个位置预测长度为$H\!W$的注意力向量，显式描述该位置与所有位置的关系。**Collect**分支回答“当前像素应从哪些位置收集信息”，**Distribute**分支回答“当前像素应把信息发送到哪些位置”；两条路径分别聚合后拼接，形成双向全局上下文。

与固定网格池化相比，**Point-wise Spatial Attention（PSA）**可按图像内容选择远距离区域；代价是注意力张量随空间位置数近似平方增长，因此通常在低分辨率特征上使用。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-026-63fea9de.jpg)


### ⚪ **APCNet**：让金字塔上下文随像素内容自适应
- **paper**：[**Adaptive Pyramid Context Network for Semantic Segmentation**](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.html)

**APCNet**在多个池化尺度上建立**Adaptive Context Module（ACM）**。每个模块先把特征池化为一组子区域，再由**Global-guided Local Affinity（GLA）**结合像素特征和全局图像表示，预测当前像素对各子区域的关联权重；池化特征按这些权重求和，得到位置相关的上下文向量。

多个尺度的自适应上下文与原特征融合，组成**Adaptive Pyramid Context**。与**PSPNet**把同一网格描述广播给所有像素不同，**APCNet**允许道路像素、人物像素从不同区域收集信息，但额外关联矩阵也增加了内存和计算。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-027-63fd5b40.jpg)


### ⚪ **DMNet**：为每幅图像生成多尺度动态滤波器
- **paper**：[**Dynamic Multi-Scale Filters for Semantic Segmentation**](https://openaccess.thecvf.com/content_ICCV_2019/html/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.html)

**DMNet**认为固定卷积核无法针对图像中的实际对象尺度调整上下文。每个**Dynamic Convolution Module（DCM）**先把输入池化到指定网格，再从池化特征生成样本专属的逐通道卷积核；该动态核作用于原特征后产生一个尺度的上下文表示。

多个不同池化网格的**DCM**并联，输出与原特征拼接。池化尺寸决定感受范围，动态核决定当前图像应提取什么模式，因此同一模块可随输入内容变化。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-028-63fd5395.jpg)

### ⚪ **HRNet**：全程保持高分辨率表示
- **paper**：[**High-Resolution Representations for Labeling Pixels and Regions**](https://arxiv.org/abs/1904.04514)

多数分割骨干先降到低分辨率，再由解码器恢复细节。**HRNet**则始终保留$1/4$分辨率主分支，并逐阶段并联加入$1/8、1/16、1/32$分辨率的新分支。每个阶段后执行多分辨率融合：

$$Y_r=\sum_{r'}f_{r'\to r}(X_{r'})$$

其中低分辨率到高分辨率使用$1\times1$卷积和双线性上采样，高分辨率到低分辨率使用若干步长为$2$的$3\times3$卷积，同分辨率使用恒等映射。重复交换让高分辨率分支持续获得深层语义，低分辨率分支也持续获得空间细节。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-058-hrnet.png)

**HRNetV1**只输出最高分辨率分支，主要用于姿态估计；用于分割的**HRNetV2**把所有分支上采样到$1/4$尺度后拼接，再由$1\times1$卷积预测；面向检测的**HRNetV2p**再从该表示构造特征金字塔。

### ⚪ **OCRNet**：让像素从对象区域表示中读取上下文
- **paper**：[**Object-Contextual Representations for Semantic Segmentation**](https://arxiv.org/abs/1909.11065)

**OCRNet**先用辅助头得到粗分割概率，再以每个类别的概率为权重聚合像素特征，形成类别区域表示$f_k$。对像素$x_i$，模型计算其与所有区域表示的关系权重，并回聚对象上下文：

$$f_k=\sum_i \tilde m_{ki}x_i,\qquad y_i=\rho\!\left(\sum_k w_{ik}\,\delta(f_k)\right)$$

其中$\tilde m_{ki}$由粗分割概率在空间维归一化，$w_{ik}$由像素—区域相似度归一化。最终将$y_i$与原像素特征融合再分类。它把昂贵的像素—像素关系压缩为像素—类别区域关系，并让同一对象内的像素共享语义；若粗分割严重错误，区域表示也会受到污染。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-029-642fd5bf.jpg)


### ⚪ **PointRend**：只在不确定位置执行高分辨率预测
- **paper**：[**PointRend: Image Segmentation as Rendering**](https://arxiv.org/abs/1912.08193)

**PointRend**把分割视为由粗到细的图像渲染。模型先产生低分辨率粗掩码，再选择最不确定的点：二分类中通常取概率最接近$0.5$的位置，多分类中可取前两类得分差最小的位置。对每个点，从高分辨率骨干特征双线性采样**fine-grained feature**，并拼接粗预测后送入共享**MLP**重新分类。

训练时先随机过采样候选点，再按不确定性与随机比例选取有限点计算损失；推理时反复把掩码上采样$2$倍，并只更新当前最不确定的若干点。这样计算集中在边界和细结构，而不必在整张高分辨率网格运行重型解码器；若粗预测完全漏掉小对象，仅细化已有难点也难以恢复。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-030-640ec603.jpg)

### ⚪ **K-Net**：以可迭代动态内核统一分割输出
- **paper**：[**K-Net: Towards Unified Image Segmentation**](https://arxiv.org/abs/2106.14855)

**K-Net**把每个分割槽表示为一个动态卷积核$K_i$。给定像素特征$F$，所有内核一次卷积即可产生一组软掩码：

$$M=\sigma(K*F)$$

模型再按$M_i$对像素特征加权汇聚，得到每个区域的**group feature**；**Kernel Update Head**用门控机制融合区域证据与旧内核，并通过内核间自注意力交换上下文。更新后的内核重新生成掩码，重复多阶段迭代即可逐步细化区域。语义分割可让内核对应类别，实例分割可让内核对应候选实例，全景分割则同时使用两组内核。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-031-641021f3.jpg)

## (3) 实例分割与全景分割模型

语义分割只需为像素给出类别，实例与全景分割还必须区分同类对象。早期方法沿两条路线发展：**自上而下（top-down）**方法先检测目标框，再在框内预测掩码；**自下而上（bottom-up）**方法先预测像素级中心、偏移或嵌入，再把像素聚成实例。全景分割进一步要求可数的**thing**实例与不可数的**stuff**区域互不重叠，因此还需要统一的合并规则。

### ⚪ **Mask R-CNN**：在两阶段检测器上并联掩码分支
- **paper**：[**Mask R-CNN**](https://arxiv.org/abs/1703.06870)

**Mask R-CNN**在**Faster R-CNN**的分类与边框回归分支旁并联一个小型全卷积掩码分支。**RoIAlign**在每个子区域的规则采样点上用双线性插值读取特征，避免**RoIPool**两次取整造成的错位。对每个候选区域，掩码分支为每个类别输出一张$m\times m$（常用$28\times28$）的掩码；若该区域的真实类别为$k$，掩码损失只计算第$k$个通道：

$$\mathcal L=\mathcal L_{\text{cls}}+\mathcal L_{\text{box}}+\mathcal L_{\text{mask}},\\ \mathcal L_{\text{mask}}=\frac1{m^2}\sum_{u}\operatorname{BCE}\!\left(\sigma(m_k(u)),y(u)\right)$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-059-maskedrcnn.png)

每类掩码独立使用**sigmoid**，类别由分类分支决定，因此不同类别的掩码之间不发生竞争。这种类别与掩码解耦的设计简单有效，但掩码受限于检测框和低分辨率网格，重叠目标和细边界需额外处理。

### ⚪ **YOLACT**：用原型掩码与系数实现实时实例分割
- **paper**：[**YOLACT: Real-time Instance Segmentation**](https://arxiv.org/abs/1904.02689)

**YOLACT**把实例分割拆成两个并行任务。**Protonet**从**FPN**高分辨率层生成与实例无关的$k$个原型掩码$P\in\mathbb R^{h\times w\times k}$；检测头除类别和边框外，还为每个锚框输出$k$维掩码系数，并用**tanh**允许正负组合。设$C\in\mathbb R^{n\times k}$为**NMS**后保留实例的系数，则实例掩码为：

$$M=\sigma\!\left(PC^\top\right)$$

组合后的掩码再用边框裁剪，并以二值交叉熵训练。论文还提出**Fast NMS**：一次计算同类检测的**IoU**矩阵，只保留上三角并按列取最大值决定是否抑制，以矩阵运算替代顺序循环。原型共享与矩阵化后处理使模型达到实时。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-060-yolact.png)

### ⚪ **SOLO / SOLOv2**：按位置类别直接分割实例
- **paper**：[**SOLO: Segmenting Objects by Locations**](https://arxiv.org/abs/1912.04488)

**SOLO**既不依赖检测框，也不需要像素嵌入聚类。它把图像划分为$S\times S$网格：若目标中心落在网格$(i,j)$，该网格负责预测此实例的语义类别和整幅图像上的掩码。类别分支输出$S\times S\times C$，掩码分支输出$H\times W\times S^2$，第$k=iS+j$个通道对应该位置的实例；掩码分支加入坐标通道（**CoordConv**）以获得位置敏感性。实例因此被转化为“位置类别”。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-061-solo.png)

- **paper**：[**SOLOv2: Dynamic and Fast Instance Segmentation**](https://arxiv.org/abs/2003.10152)

**SOLOv2**把$S^2$个掩码通道改写为动态卷积：核分支为每个网格预测卷积核$G_{ij}$，特征分支从**FPN**融合出统一掩码特征$F$，实例掩码为$M_{ij}=G_{ij}*F$。它还提出并行的**Matrix NMS**，按更高分掩码的重叠程度衰减当前得分：

$$\operatorname{decay}_j=\min_{i:\,s_i>s_j}\frac{f(\operatorname{IoU}_{i,j})}{f(\operatorname{IoU}_{\cdot,i})},\qquad f(\operatorname{IoU})=1-\operatorname{IoU}\ \text{或}\ \exp(-\operatorname{IoU}^2/\sigma)$$

其中$\operatorname{IoU}_{\cdot,i}$是掩码$i$与更高分掩码的最大**IoU**。中心落入同一网格的多个目标仍会冲突，因此**SOLO**依赖多层**FPN**为不同大小的对象分配网格。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-062-solov2.png)

### ⚪ **Panoptic FPN**：在特征金字塔上增加语义分支
- **paper**：[**Panoptic Feature Pyramid Networks**](https://arxiv.org/abs/1901.02446)

**Panoptic FPN**复用检测器的**Feature Pyramid Network（FPN）**。金字塔每一层先经过$3\times3$卷积，再通过若干“卷积—归一化—上采样”模块统一到$1/4$分辨率；各层结果逐元素相加并预测逐像素语义。高层提供大感受野，低层提供小对象和边界细节。

语义分支负责不可数的**stuff**区域，已有**Mask R-CNN**分支负责可数的**thing**实例，最后通过启发式规则消解重叠并合成为全景结果。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-023-64083b39.jpg)

### ⚪ **Panoptic-DeepLab**：用中心与偏移自下而上组成实例
- **paper**：[**Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation**](https://arxiv.org/abs/1911.10194)

**Panoptic-DeepLab**在共享骨干后接两套独立的**ASPP**与解码器：语义分支预测所有类别的逐像素得分；实例分支预测类无关的对象中心热图和每个像素指向所属中心的二维偏移$O$。中心热图以二维高斯为目标并用均方误差训练，偏移只在**thing**像素上用$L_1$损失训练。

推理时，对中心热图做关键点式**NMS**得到中心集合$\{c_k\}$，再把每个**thing**像素$u$分配给偏移后最近的中心：

$$\hat k(u)=\mathop{\arg\min}_k\left\|c_k-\left(u+O(u)\right)\right\|_2$$

每个实例的类别由其像素语义预测多数投票决定，最后与**stuff**语义区域合并。该方法不需要检测框和**RoI**操作，结构简单且速度较快。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-063-panoptic-deeplab.png)

## (4) 基于Transformer与掩码分类的图像分割模型

**Transformer**通过自注意力建立远距离依赖，因此被引入分割任务。但图像块化会损失细边界，全局注意力的计算和显存还会随词元数平方增长。因此实际分割模型通常采用分层编码器、多尺度特征或局部/稀疏注意力。

另一条更重要的变化是从逐像素分类转向**掩码分类（mask classification）**。模型预测固定数量的掩码查询，每个查询同时给出二值掩码与类别；语义分割再把查询结果合成为逐像素标签。这种表示更自然地连接了语义、实例和全景分割。

### ⚪ **SETR**：把语义分割改写为序列到序列预测
- **paper**：[**Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**](https://arxiv.org/abs/2012.15840)

**SETR**把图像切成不重叠块并线性嵌入为词元序列，使用标准**ViT**编码器在固定$1/16$尺度上建立全局关系；从第一层自注意力开始，每个图像块都能与全图交互。编码后的序列重新排列为空间特征图再解码。

论文比较三种解码器：**Naive**直接卷积并一次上采样；**Progressive UPsampling（PUP）**交替卷积和$2$倍上采样；**Multi-Level feature Aggregation（MLA）**从多个**Transformer**层取特征并逐级融合。**SETR**证明纯序列编码可用于密集预测，但单尺度块化会损失细边界，全局注意力成本也随图像块数平方增长。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-032-6412d409.jpg)


### ⚪ **TransUNet**：把卷积局部特征送入全局Transformer编码器
- **paper**：[**TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation**](https://arxiv.org/abs/2102.04306)

**TransUNet**先用**ResNet50**前三个阶段提取局部纹理与多尺度特征，再把最深特征图划分为块、线性投影并送入$12$层**ViT**。这样**Transformer**处理的是卷积特征词元，在建立长距离依赖前已注入局部归纳偏置。

编码序列还原为空间特征后，解码器连续上采样，并与**ResNet**较浅阶段的特征执行**U-Net**式跳跃拼接。**ViT**负责全局器官关系，卷积跳跃连接负责精细定位。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-033-6422a162.jpg)


### ⚪ **SegFormer**：用分层高效注意力配合轻量MLP解码器
- **paper**：[**SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**](https://arxiv.org/abs/2105.15203)

**SegFormer**的**MiT**编码器通过重叠块嵌入逐级下采样，产生$1/4、1/8、1/16、1/32$四个尺度。**Efficient Self-Attention**先把键和值的序列长度从$N$压缩为$N/R$，将注意力复杂度由$O(N^2)$降为$O(N^2/R)$；**Mix-FFN**在两层线性映射之间加入$3\times3$深度卷积，以局部空间混合隐式表达位置，因此无需固定尺寸的位置编码。

**All-MLP Decoder**把四级特征分别线性投影到相同通道、上采样到$1/4$尺度并拼接，再用线性层融合和预测。编码器承担多尺度与上下文建模，解码器保持轻量；这使模型能适应不同测试分辨率。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-034-6414188a.jpg)


### ⚪ **Segmenter**：以类别词元和图像块相似度生成掩码
- **paper**：[**Segmenter: Transformer for Semantic Segmentation**](https://arxiv.org/abs/2105.05633)

**Segmenter**使用标准**ViT**把图像编码为块词元。最简单的线性解码器独立分类每个词元；更完整的**Mask Transformer**加入$K$个可学习类别嵌入，与图像词元一起经过若干自注意力层，使类别表示和图像内容双向交互。归一化后的块表示$z$与类别表示$c$点积得到类别掩码：

$$\operatorname{Masks}(z,c)=zc^\top$$

掩码上采样后形成逐像素预测。类别嵌入相当于固定语义类别的查询，但每个类别只有一个输出通道，不区分同类实例。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-035-6416d47d.jpg)

### ⚪ **MaskFormer**：把语义分割改写为掩码分类
- **paper**：[**Per-Pixel Classification is Not All You Need for Semantic Segmentation**](https://arxiv.org/abs/2107.06278)

**MaskFormer**不再让每个像素独立选择类别。骨干网络和像素解码器产生逐像素嵌入$E_{\mathrm{pixel}}$，**Transformer**解码器把$N$个可学习查询变换为掩码嵌入$E_{\mathrm{mask}}$；二者点积生成掩码概率$m_i(h,w)=\sigma(E_{\mathrm{mask},i}^{\top}E_{\mathrm{pixel},h,w})$，另一个线性分类器输出类别概率$p_i(c)$。语义分割得分可由查询结果组合：

$$s_c(h,w)=\sum_{i=1}^{N}p_i(c)m_i(h,w)$$

训练时通过二分图匹配把预测查询与真实掩码一一配对，再联合优化类别交叉熵、掩码**Focal Loss**和**Dice Loss**，从而避免固定查询顺序。逐像素分类只改变每个位置的类别，而掩码分类显式表示“哪个区域属于哪个查询”；这不仅改善语义分割，也为统一实例与全景分割奠定了表示基础。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-036-642e7af5.jpg)

### ⚪ **Mask2Former**：用掩码注意力统一图像分割
- **paper**：[**Masked-attention Mask Transformer for Universal Image Segmentation**](https://arxiv.org/abs/2112.01527)

**Mask2Former**让查询的交叉注意力只访问当前预测掩码覆盖的区域。设第$l$层查询特征为$X_l$，$Q_l=f_Q(X_{l-1})$，$K_l,V_l$是图像特征经$f_K,f_V$得到的投影；上一层预测掩码二值化后给出注意力掩码$\mathcal M_{l-1}$，则掩码交叉注意力为：

$$X_l=\operatorname{softmax}\!\left(\mathcal M_{l-1}+Q_lK_l^\top\right)V_l+X_{l-1}$$

其中$\mathcal M_{l-1}(x,y)$在前景位置取$0$、在其他位置取$-\infty$，因此掩码外位置不会参与当前查询的特征聚合。解码器轮流使用$1/32、1/16、1/8$多尺度特征，并把自注意力放在交叉注意力之后。训练时，二分图匹配在所有预测与真实掩码上共享同一组均匀采样点，最终掩码损失则在偏向不确定位置的重要性采样点上计算，从而降低高分辨率训练成本。掩码约束既减少无关背景干扰，也让查询逐层聚焦于对应区域；同一架构可用于语义、实例和全景分割。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-046-mask2former.png)

### ⚪ **OneFormer**：用任务条件实现一次训练、多种分割
- **paper**：[**OneFormer: One Transformer to Rule Universal Image Segmentation**](https://arxiv.org/abs/2211.06220)

**OneFormer**在掩码分类器中加入“语义分割”“实例分割”或“全景分割”等任务条件。任务文本首先编码为任务词元并调制对象查询；训练时，真实语义标签也会根据当前任务转换为相应的掩码集合，例如语义任务合并同类区域，实例任务只保留可数对象，全景任务同时保留**thing**实例与**stuff**区域。查询—文本对比学习进一步约束查询表达对应任务语义。它解决了统一架构仍需分别训练的问题：同一组参数可以根据任务提示产生不同粒度的输出。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-047-oneformer.png)

## (5) 提示分割与分割基础模型

提示分割把分割条件从固定类别扩展为点、框、已有掩码、参考图像或上下文示例。其核心评估对象是“输出掩码是否符合提示意图”；同一提示存在多个合理区域时，模型还需要表达输出歧义。

### ⚪ **DEXTR (Deep Extreme Cut)**：基于极值点的交互式分割
- **paper**：[**Deep Extreme Cut: From Extreme Points to Object Segmentation**](https://arxiv.org/abs/1711.09081)

**DEXTR**用目标的四个极值点（最上、最下、最左、最右）引导分割。模型先用极值点围成的边界框（外扩若干像素）裁剪目标并缩放到固定尺寸，再把稀疏点击$p_i$转换为以各点为中心的二维高斯热图$H$：

$$
H(x, y) = \max_{i=1}^{4} \mathcal{N}((x, y) | \mu=p_i, \Sigma)
$$

热图作为第四个通道与**RGB**图像拼接（$\text{Image}_{\text{RGB}} \oplus H$），送入**DeepLab-v2**式的**ResNet-101**骨干与**PSP**头，并以类别平衡的二值交叉熵训练前景掩码。极值点既给出目标范围，又落在物体边界上，比普通框提供更强的形状线索。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-049-dextr.png)

**DEXTR**输出类无关的前景掩码，已经体现了“稀疏交互→密集引导→掩码”的提示分割思路；但它只支持固定形式的四点提示、一次只输出一个掩码，并在常规分割数据上训练，不具备**SAM**那样统一多种提示、表达歧义并在大规模数据上预训练的接口。

### ⚪ **Segment Anything (SAM)**：可提示的通用分割基础模型
- **paper**：[**Segment Anything**](https://arxiv.org/abs/2304.02643)

**SAM**定义了**promptable segmentation**任务：给定任意点、框、粗掩码或文本提示，模型都应返回至少一个合理掩码。架构由三部分解耦组成：
- 图像编码器：经**MAE**预训练的**ViT-H**把$1024\times1024$图像编码为$64\times64$的嵌入$E_{\text{image}}$，每张图像只需计算一次；
- 提示编码器：点和框以位置编码加类型嵌入表示，粗掩码经卷积下采样后与图像嵌入逐元素相加；论文也用**CLIP**文本编码器初步探索了文本提示；
- 掩码解码器：两层改造的**Transformer**解码器在提示词元、输出词元和图像嵌入之间执行双向交叉注意力，再由输出词元动态预测掩码。

$$
\{M_k, \widehat{\text{IoU}}_k\}_{k=1}^{3} = \text{MaskDecoder}(E_{\text{image}}, E_{\text{prompt}})
$$

单个点可能对应部件、对象或整体，因此解码器同时输出$3$个掩码及各自的预测**IoU**；训练只对损失最小的掩码反传，避免把多个合理答案平均成模糊结果。掩码损失为**Focal Loss**与**Dice Loss**的线性组合，训练中还模拟多轮点击交互。

训练数据来自三阶段数据引擎：辅助人工标注、半自动补全，以及规则网格点提示下的全自动生成；最终的**SA-1B**含约$1100$万张图像和$11$亿个掩码。重型编码与轻型交互分离，使提示解码可在浏览器中约$50$毫秒内完成。**SAM**输出类无关掩码，零样本边界质量强，但不提供语义类别；需要类别时通常还要与检测器或视觉—语言模型组合。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-037-642e6ea6.jpg)

### ⚪ **SegGPT**：把分割建模为视觉上下文学习
- **paper**：[**SegGPT: Towards Segmenting Everything in Context**](https://arxiv.org/abs/2304.03284)

**SegGPT**沿用**Painter**的视觉上下文框架，把多种分割任务统一为“看示例、给查询图像着色”。训练时，每个样本的掩码按语义类别或实例映射为随机颜色，得到彩色目标图；同一类别或实例在示例与查询中使用同一种颜色，但不同样本之间颜色随机变化，迫使模型从上下文推断“哪些区域应与示例一致”，而不是记住固定颜色对应的类别。

输入由两张拼接画布组成：一张拼接示例图像$x_e$与查询图像$x_q$，另一张拼接示例彩色掩码$y_e$与被遮挡的查询目标。**ViT**编码器联合处理两张画布并补全查询的彩色掩码：

$$
\hat{y}_q = f_\theta\!\left([x_e, x_q], [y_e, \varnothing]\right)
$$

训练目标是预测与真实彩色掩码之间的**smooth-L1**损失。推理时只需更换示例，就能执行语义、实例、部件和视频对象等分割；多个示例可以空间拼接后集成，也可以在若干层平均查询特征；还可以只调一组可学习提示，实现针对特定任务的上下文调优。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-048-seggpt.png)

### ⚪ **SEEM**：统一多种提示与分割粒度
- **paper**：[**Segment Everything Everywhere All at Once**](https://arxiv.org/abs/2304.06718)

**SEEM**在**X-Decoder**式的通用分割解码器上统一多种提示。文本提示由文本编码器得到$P_t$；点、框、涂鸦、掩码以及另一张图像中的参考区域都被视为图像上的区域，由**visual sampler**从图像特征$Z$中池化或插值得到视觉提示$P_v$；多轮交互中上一轮的掩码与查询信息写成记忆提示$P_m$。解码器把可学习查询$Q_h$与这些提示一起处理：

$$
\langle O^m_h, O^c_h\rangle = \text{Decoder}\!\left(Q_h;\langle P_t, P_v, P_m\rangle \mid Z\right)
$$

其中$O^m_h$生成掩码，$O^c_h$投影到与文本对齐的联合语义空间进行开放词汇分类。视觉提示与文本嵌入因此可以在同一空间中匹配，不同类型的提示也能同时输入；解码器通过注意力掩码控制提示与查询之间的交互。没有提示时，同一模型执行通用的语义、实例和全景分割；有提示时，它执行交互式或指代分割。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-050-seem.png)

### ⚪ **SAM 2**：用流式记忆统一图像与视频分割
- **paper**：[**SAM 2: Segment Anything in Images and Videos**](https://arxiv.org/abs/2408.00714)

**SAM 2**把可提示分割从图像扩展到视频：用户可在任意帧给出点、框或掩码，模型生成并在整段视频中传播时空掩码（**masklet**）。图像被视为只有一帧、记忆为空的视频，因此同一模型同时支持图像与视频分割。

每帧先由经**MAE**预训练的**Hiera**分层编码器计算一次图像嵌入$E_t$。**memory attention**让当前帧嵌入对记忆库$\mathcal B_t$执行自注意力和交叉注意力，再交给与**SAM**相似的提示编码器和掩码解码器：

$$
\tilde E_t = \text{MemAttn}(E_t, \mathcal B_t),\qquad M_t = \text{MaskDecoder}(\tilde E_t, P_t)
$$

**memory encoder**把预测掩码与当前帧嵌入融合为新的空间记忆并写回记忆库。记忆库以先进先出队列保存最近若干帧，同时保留若干被提示过的帧，并存储由解码器输出词元得到的轻量**object pointer**。解码器还增加遮挡预测头，判断目标在当前帧是否可见。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-051-sam2.png)

训练数据来自模型与标注者循环迭代的数据引擎，最终构建了**SA-V**视频分割数据集。流式记忆避免了对整段视频反复计算全局注意力，适合在线交互。

### ⚪ **SAM 3**：按概念检测、分割并跟踪全部实例
- **paper**：[**SAM 3: Segment Anything with Concepts**](https://arxiv.org/abs/2511.16719)

**SAM 3**提出**Promptable Concept Segmentation（PCS）**任务：给定简短名词短语（如“a penguin”）、图像示例框或二者组合，模型需要在图像或视频中找出该概念的所有实例，输出掩码并在视频中保持实例身份。**SAM**和**SAM 2**的点、框提示指定“这一个对象”，概念提示则指定“这一类对象”；**SAM 3**同时保留了前者的视觉提示分割能力。

模型由共享**Perception Encoder（PE）**视觉骨干的检测器和跟踪器组成。检测器采用**DETR**式集合预测：文本和示例提示先与图像特征融合，再由对象查询输出候选框、掩码与匹配分数；示例既可以是正例，也可以是负例。进一步加入全局**presence head**，把“图像中是否存在该概念”与“哪个查询定位到实例”分开估计：

$$
s_i = p(\text{present}\mid I, P)\cdot p(q_i\ \text{matches}\mid I, P, \text{present})
$$

这种分解缓解了对象查询同时承担识别与定位时的冲突，也减少了难负例短语造成的假阳性。在视频中，检测器逐帧发现新实例，继承**SAM 2**设计的跟踪器借助记忆库传播已有掩码，二者的结果再匹配、合并并更新实例身份。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-052-sam3.png)

训练依赖结合人工与模型标注的大规模数据引擎，论文同时发布了覆盖大量概念的**SA-Co**基准。**SAM 3**本身只处理短名词短语式概念；需要推理的复杂指令由**SAM 3 Agent**交给多模态大语言模型分解，后者迭代调用**SAM 3**并检查结果。

## (6) 开放词汇与语言驱动分割

开放词汇分割进一步把固定的$C$类分类器替换为视觉—文本相似度。设像素或掩码表示为$z_v$，类别文本表示为$z_t(c)$，常用分类得分为：

$$
p(c\mid z_v)=\frac{\exp(\operatorname{sim}(z_v,z_t(c))/\tau)}{\sum_{c'\in\mathcal C}\exp(\operatorname{sim}(z_v,z_t(c'))/\tau)}
$$

### ⚪ **CLIPSeg**：用文本或图像提示条件化CLIP解码器
- **paper**：[**Image Segmentation Using Text and Image Prompts**](https://arxiv.org/abs/2112.10003)

**CLIPSeg**在冻结的**CLIP ViT-B/16**上增加轻量**Transformer**解码器，使同一模型可按文本或图像提示输出二值分割。解码器从**CLIP**视觉编码器第$3、6、9$层读取激活，经线性投影后逐层加入解码器残差流，以复用不同深度的语义与空间信息。

提示先由**CLIP**文本编码器或图像编码器得到条件向量$e$，图像提示可以是突出目标区域的参考图像。条件是在解码器输入处使用**FiLM**调制：

$$
\operatorname{FiLM}(h;e)=\gamma(e)\odot h+\beta(e)
$$

最后由转置卷积恢复到输入分辨率并输出前景概率。训练使用扩展的**PhraseCut**数据集，并随机插值文本与视觉提示嵌入，使两种提示共享条件空间。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-053-clipseg.png)

### ⚪ **MaskCLIP**：不训练地从冻结CLIP读取密集预测
- **paper**：[**Extract Free Dense Labels from CLIP**](https://arxiv.org/abs/2112.01071)

**CLIP**图像编码器最后通过注意力池化只输出全局嵌入；直接把图像块的输出词元与文本比较，空间位置与文本嵌入并不对齐。**MaskCLIP**去掉最后一层注意力池化中的**query-key**交互，把每个位置的**value**嵌入经原有输出投影后作为该位置的图像特征，这两个线性层可等价改写为$1\times1$卷积。类别提示的文本嵌入$t_c$直接充当分类器权重：

$$
s_c(u)=\left\langle W_o\,v(u),\,t_c\right\rangle
$$

该过程无需训练即可产生零样本分割，并可配合提示去噪与键平滑改善结果。**MaskCLIP+**再以这些预测作为伪标签训练标准分割网络，通过自训练进一步提升未见类别性能。它说明冻结**CLIP**中已有可用的局部语义。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-064-maskclip.png)

### ⚪ **OpenSeg**：用区域—词语对齐扩展分割词汇
- **paper**：[**Scaling Open-Vocabulary Image Segmentation with Image-Level Labels**](https://arxiv.org/abs/2112.12143)

**OpenSeg**先学习类无关的掩码提议，再把区域特征与文本中的词语对齐。模型从骨干与特征金字塔得到像素特征，由掩码提议头预测$N$个类无关掩码，再对每个掩码执行掩码池化得到区域表示$z_i$。对图像描述中的$K$个词嵌入$w_k$，区域—词语相似度与图文匹配分数为：

$$
s_{ik}=\cos(z_i,w_k),\qquad G(I,T)=\frac1K\sum_{k=1}^{K}\sum_{i=1}^{N}\operatorname{softmax}_i(s_{ik})\,s_{ik}
$$

**region-word grounding loss**在批内对比匹配与不匹配的图文对，使每个词自动选择最相关的区域，而不需要词与掩码的人工对应。掩码提议用**COCO**中的类无关掩码训练，区域—词语对齐则利用**COCO Captions**和**Localized Narratives**等图像级文本扩展词汇。

推理时把任意类别名编码为文本嵌入，与区域表示比较后把掩码得分投回像素。论文在**ADE20K-150/847**和**PASCAL Context-59/459**等不同词表大小上评测，展示了图像级文本对大词汇分割的扩展作用。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-065-openseg.png)

### ⚪ **LSeg**：用像素嵌入对齐冻结文本空间
- **paper**：[**Language-driven Semantic Segmentation**](https://arxiv.org/abs/2201.03546)

**LSeg**用冻结的**CLIP**文本编码器把$N$个候选标签编码为$T\in\mathbb R^{N\times C}$，并训练基于**DPT**的图像编码器，输出与之同维的密集嵌入$I\in\mathbb R^{\tilde H\times\tilde W\times C}$。每个位置与所有标签做内积，得到逐像素标签得分：

$$
F_{hwk} = I_{hw}\cdot T_k
$$

训练在已有分割标注上执行带温度的逐像素**softmax**交叉熵，让像素嵌入靠近对应标签的文本嵌入、远离其他标签；空间正则化模块再对得分图做轻量卷积并上采样。推理时替换标签集合即可改变输出类别，也可以加入同义词或“其他”类别。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-054-lseg.png)

### ⚪ **GroupViT**：从文本监督中涌现的语义分割
- **paper**：[**GroupViT: Semantic Segmentation Emerges from Text Supervision**](https://arxiv.org/abs/2202.11094)

**GroupViT**只用图文对训练，不使用任何像素标注。它在**ViT**中加入可学习的分组词元：第一阶段$64$个分组词元与图像块词元一起经过若干**Transformer**层，再由**Grouping Block**把每个图像块分配到一个分组；第二阶段在得到的片段上继续使用$8$个分组词元合并更大区域。设分组词元为$g_i$、片段词元为$s_j$，分配矩阵为：

$$
A_{ij}=\frac{\exp\!\big((W_qg_i)^\top(W_ks_j)+\gamma_i\big)}{\sum_{k}\exp\!\big((W_qg_k)^\top(W_ks_j)+\gamma_k\big)}
$$

其中$\gamma$为**Gumbel**噪声。前向时每个片段取**one-hot**硬分配，反向用直通估计传递梯度；合并后的新片段由对应分组词元加上所分配片段的加权平均得到。

最终片段平均后与文本嵌入做图文对比学习。除完整描述外，模型还从描述中抽取名词，构造“**a photo of a {noun}**”提示并执行多标签对比，使不同区域对应不同语义。推理时把每个最终片段与类别文本嵌入比较，并沿两阶段分配关系把标签投回像素。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-055-groupvit.png)

### ⚪ **SAN**：用侧适配网络保留视觉—语言模型能力
- **paper**：[**Side Adapter Network for Open-Vocabulary Semantic Segmentation**](https://arxiv.org/abs/2302.12242)

**SAN**保持**CLIP**完全冻结，在旁边训练一个轻量**ViT**侧网络。侧网络接收图像块与可学习查询，并在若干层融合**CLIP**浅层的中间特征。它同时输出两类信息：查询与像素特征点积得到类无关掩码提议；另一组查询嵌入生成供**CLIP**深层使用的注意力偏置。

识别时复制**CLIP**的**[CLS]**词元，形成与掩码一一对应的区域识别词元；在**CLIP**最后若干层中，注意力偏置$B_i$使第$i$个识别词元只关注对应掩码区域，其输出嵌入$r_i$再与类别文本嵌入比较：

$$
p_i(c)=\operatorname{softmax}_c\!\left(\langle r_i,t_c\rangle/\tau\right)
$$

掩码提议与识别分支解耦，因此不必对每个裁剪区域重复运行**CLIP**：整张图像只经过一次**CLIP**前向，侧网络也能经由冻结的**CLIP**端到端获得识别梯度。少量可训练参数既保留了**CLIP**的开放词汇能力，又避免微调破坏预训练对齐。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-066-san.png)

### ⚪ **ODISE**：借助文本到图像扩散模型进行开放词汇全景分割
- **paper**：[**Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models**](https://arxiv.org/abs/2303.04803)

**ODISE**认为，文本到图像扩散模型为了生成图像，必须学习对象位置、边界与语义之间的对应，因此其内部特征适合开放词汇全景分割。模型给输入图像加入少量噪声，只运行一次冻结的**Stable Diffusion UNet**去噪步骤并提取多尺度中间特征。测试时没有图像描述，**implicit captioner**便用冻结的**CLIP**图像编码器生成图像嵌入，再经可学习映射作为**UNet**交叉注意力的文本条件。

扩散特征送入**Mask2Former**式掩码生成器，得到类无关掩码与掩码嵌入。训练既可以使用类别标签监督，即与类别名的文本嵌入做分类；也可以使用图像描述监督，通过区域—词语对齐学习更大的词汇。推理时，模型还对冻结**CLIP**特征做掩码池化得到另一组类别概率，并与扩散分支做几何平均：

$$
p_i(c)\propto p^{\text{diff}}_i(c)^{\lambda}\,p^{\text{clip}}_i(c)^{1-\lambda}
$$

最终结果按全景分割规则合成。论文在**COCO**上训练，并在**ADE20K**等数据集上做开放词汇评测。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-067-odise.png)

### ⚪ **CAT-Seg**：聚合图像—文本代价体进行开放词汇分割
- **paper**：[**CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation**](https://arxiv.org/abs/2303.11797)

**CAT-Seg**不先生成掩码提议，而是直接比较**CLIP**的密集图像嵌入$D^V(i)$与每个类别文本嵌入$D^L(n)$，构造逐像素、逐类别的余弦相似度代价体：

$$
C(i,n)=\frac{D^V(i)\cdot D^L(n)}{\|D^V(i)\|\,\|D^L(n)\|}
$$

代价体经卷积嵌入后交替执行两类聚合：**spatial aggregation**在每个类别的代价图内使用**Swin Transformer**式局部注意力，平滑噪声并利用空间结构；**class aggregation**在每个位置跨类别执行不带位置编码的注意力，让模型考虑类别间的竞争，并对类别顺序和数量不敏感。聚合后的代价再由上采样解码器结合骨干浅层特征逐步恢复分辨率。

训练时不完全冻结**CLIP**，只微调图像与文本编码器注意力层中的部分投影参数，使密集特征适配分割，同时尽量保持开放词汇对齐。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-068-catseg.png)

### ⚪ **FC-CLIP**：用单个冻结卷积CLIP同时生成并识别掩码
- **paper**：[**Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP**](https://arxiv.org/abs/2308.02487)

许多两阶段方法先用独立网络生成掩码，再把掩码区域送入**CLIP**分类。**FC-CLIP**改用一个冻结的卷积**CLIP**（**ConvNeXt-L**）作为共享骨干，因为卷积网络在高分辨率输入下仍能保持较好的密集特征与开放词汇对齐。**Mask2Former**式像素解码器和掩码解码器在该骨干上生成类无关掩码，并训练一个**in-vocabulary classifier**识别训练类别；另一个**out-of-vocabulary classifier**直接对冻结**CLIP**特征做掩码池化后与文本嵌入比较，无需训练。

推理时，两类得分按类别是否属于训练集合使用不同权重做几何集成：

$$
p_i(c)=\begin{cases}p^{\text{in}}_i(c)^{1-\alpha}\,p^{\text{out}}_i(c)^{\alpha}, & c\in\mathcal C_{\text{train}}\\[2pt] p^{\text{in}}_i(c)^{1-\beta}\,p^{\text{out}}_i(c)^{\beta}, & c\notin\mathcal C_{\text{train}}\end{cases}
$$

共享骨干使整个系统只需提取一次特征，训练成本和推理延迟都明显低于“掩码生成器+独立**CLIP**”的设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-069-fcclip.png)

### ⚪ **LISA**：用多模态大语言模型输出分割词元
- **paper**：[**LISA: Reasoning Segmentation via Large Language Model**](https://arxiv.org/abs/2308.00692)

**LISA**提出**reasoning segmentation**任务：查询可以是需要常识或推理的隐式描述，例如“图中能补充维生素C的食物”，模型必须先理解意图再输出掩码。它在**LLaVA**的词表中加入特殊的`<SEG>`词元；多模态大语言模型生成回答时一旦输出`<SEG>`，就取其最后一层隐藏状态$h_{\text{seg}}$，经**MLP**投影为提示嵌入，再与**SAM**视觉骨干的图像特征一起送入掩码解码器：

$$
M=\operatorname{Dec}\!\left(\operatorname{Enc}_{\text{SAM}}(x),\ \operatorname{MLP}(h_{\text{seg}})\right)
$$

训练目标由文本生成的自回归交叉熵与掩码的二值交叉熵、**Dice Loss**组成；视觉骨干冻结，语言模型用**LoRA**微调，解码器与投影层端到端训练。训练数据混合了语义分割、指代分割和视觉问答，论文还构建了**ReasonSeg**基准。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-070-lisa.png)

### ⚪ **Grounded SAM**：组合开放集检测器与可提示分割模型
- **paper**：[**Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks**](https://arxiv.org/abs/2401.14159)

**Grounded SAM**是一条模块化流水线，**Grounded DINO**先根据文本输出开放集边界框及其对应短语，**SAM**再以这些框为提示生成掩码：

$$
\{(b_i,\text{phrase}_i)\}=\operatorname{GroundingDINO}(x,T),\\ M_i=\operatorname{SAM}(x,b_i)
$$

在此基础上还可接入**RAM**或**BLIP**自动生成标签与描述，用**Stable Diffusion**按掩码修复编辑图像，或接入跟踪器扩展到视频。模块组合能快速利用各自最强的预训练模型，支持文本驱动的检测、分割、自动标注和编辑。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-071-grounded-sam.png)

# 2. 图像分割中的通用技巧

分割模型的最终效果不只由网络结构决定，还依赖训练信号、数据增强、上采样算子、后处理、少标注学习和跨域适应等环节。这些技巧通常与具体骨干和分割头解耦，可以作为插件加入不同模型。

## (1) 训练监督与数据增强

像素级监督只作用在最终输出上时，深层网络的中间特征难以直接获得任务梯度；实例级标注又往往集中在少数常见场景中。训练阶段的常见做法是增加中间监督，或用增强构造更多样的对象组合。

### ⚪ **Deep Supervision**：用辅助输出直接约束中间特征
- **paper**：[**Deeply-Supervised Nets**](https://arxiv.org/abs/1409.5185)、[**Training Deeper Convolutional Networks with Deep Supervision**](https://arxiv.org/abs/1505.02496v1)

**深度监督（Deep Supervision）**在若干隐藏层后增加只在训练时使用的辅助分类器，使中间特征直接获得任务梯度并具备判别性。若主输出损失为$\mathcal L_0$，辅助头损失为$\mathcal L_s^{(j)}$，总目标通常写为：

$$
\mathcal L=\mathcal L_0+\sum_j\alpha_j(t)\mathcal L_s^{(j)}
$$

权重$\alpha_j(t)$可以固定，也可以随训练衰减，避免后期辅助任务过度限制最终表示。一个带有深度监督的八层卷积网络可在**Conv4**后添加辅助分类器：**Conv4**特征既继续进入**Conv5**，也直接产生预测。分割网络常把辅助头接在不同尺度，但每个辅助标签必须按明确插值规则缩放；辅助输出在推理时通常删除。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-038-61274a2b.jpg)

### ⚪ **Copy-Paste**：把对象实例粘贴到新背景中
- **paper**：[**Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation**](https://arxiv.org/abs/2012.07177)

**Copy-Paste**从一张图像中随机选取部分实例，连同其掩码粘贴到另一张图像上。设$I_1$为源图像、$I_2$为背景图像、$\alpha$为被粘贴实例的二值掩码，则合成图像为：

$$
I=\alpha\odot I_1+(1-\alpha)\odot I_2
$$

被完全遮挡的原有实例从标注中移除，部分遮挡实例的掩码与边界框同步更新。两张图像先分别做随机翻转和**Large Scale Jittering**：随机缩放到原尺寸的$0.1$至$2.0$倍后裁剪或填充，变化范围远大于常用的$0.8$至$1.25$倍尺度抖动。论文发现无需建模周围上下文，随机选择粘贴位置即可显著提升实例分割，对**LVIS**中样本稀少的长尾类别尤其有效，并且能与伪标签自训练叠加。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-072-copy-paste.png)


## (2) 上采样与特征对齐

编码器输出步幅通常为$8$至$32$，解码器需要把低分辨率表示恢复到原图尺寸。双线性插值与转置卷积使用固定或与内容无关的核，跨尺度融合时还会出现特征错位；以下方法分别从标签结构、内容自适应重组和语义流对齐改进这一环节。

### ⚪ **DUpsampling**：利用标签冗余进行数据相关上采样
- **paper**：[**Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation**](https://arxiv.org/abs/1903.02120)

**DUpsampling**注意到分割标签在局部高度冗余：一个$r\times r$图像块内的$C$类**one-hot**标签展平后为$v\in\{0,1\}^{r^2C}$，却只分布在一个低维子空间中。论文先在训练标签上学习线性投影$P$与重建矩阵$W$：

$$
\min_{P,W}\sum_v\|v-WPv\|_2^2
$$

网络只需在$1/16$或$1/32$分辨率上为每个位置预测低维向量$\tilde v$，**DUpsampling**再用$W\tilde v$恢复整个$r\times r\times C$标签块，相当于一个由数据学到的$1\times1$卷积加像素重排。由于最终分辨率不再依赖解码器逐级上采样，浅层特征可以先下采样到最低分辨率再融合，融合方式更灵活、计算也更低。论文还配合可学习温度的**softmax**，缓解重建输出尺度与交叉熵之间的不匹配。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-073-dupsampling.png)

### ⚪ **CARAFE**：按内容预测上采样重组核
- **paper**：[**CARAFE: Content-Aware ReAssembly of FEatures**](https://arxiv.org/abs/1905.02188)

**CARAFE**为每个输出位置预测一个独立的重组核。设上采样倍率为$\sigma$，核预测模块先用$1\times1$卷积压缩输入通道，再用$k_{enc}\times k_{enc}$卷积输出$\sigma^2k_{up}^2$个通道，重排为每个输出位置的$k_{up}\times k_{up}$核，并在核内做**softmax**归一化。输出位置$l'=(i',j')$对应输入位置$l=(\lfloor i'/\sigma\rfloor,\lfloor j'/\sigma\rfloor)$，其特征由该位置邻域加权重组：

$$
X'_{l'}=\sum_{n=-r}^{r}\sum_{m=-r}^{r}W_{l'}(n,m)\,X_{(i+n,\,j+m)}
$$

其中$r=\lfloor k_{up}/2\rfloor$，默认设置为$\sigma=2$、$k_{up}=5$、$k_{enc}=3$。与双线性插值相比，重组核随语义内容变化；与转置卷积相比，它在较大邻域内聚合信息，而参数量和计算量都很小。**CARAFE**可直接替换**FPN、UPerNet**以及**Mask R-CNN**掩码头中的上采样算子。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-074-carafe.png)

### ⚪ **SFNet**：用语义流对齐相邻层级特征
- **paper**：[**Semantic Flow for Fast and Accurate Scene Parsing**](https://arxiv.org/abs/2002.10120)

**FPN**式解码器融合相邻层级时直接对深层特征做双线性上采样，但多次下采样后深浅层特征在空间上并不严格对应，直接相加会使边界模糊。**SFNet**借鉴光流估计，提出**Flow Alignment Module（FAM）**：先把低分辨率特征$F_l$上采样并与高分辨率特征$F_{l-1}$拼接，用$3\times3$卷积预测二维**semantic flow** $\Delta_{l-1}$；高分辨率网格上的点$p_{l-1}$按流偏移映射回低分辨率坐标，再用可微双线性采样读取特征：

$$
\tilde F_l(p_{l-1})=\operatorname{Sample}\!\left(F_l,\ \frac{p_{l-1}+\Delta_{l-1}(p_{l-1})}{2}\right)
$$

对齐后的$\tilde F_l$再与$F_{l-1}$相加。**FAM**只增加少量卷积，却能让**ResNet-18**等轻量骨干在**Cityscapes**上兼顾实时速度与较高精度。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-075-sfnet.png)

## (3) 后处理与测试时推理

卷积网络的输出步幅与平滑插值会让边界模糊，单一尺度推理也难以同时照顾大物体和细结构。后处理利用低层外观信息修正边界，测试时推理则在多个尺度、翻转或滑窗上集成预测。最常用的基线是把多个缩放比例与水平翻转的概率图插值回原图后取平均；大尺寸图像则用重叠滑窗推理，以避免显存溢出。

### ⚪ **DenseCRF**：用全连接条件随机场细化边界
- **paper**：[**Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials**](https://arxiv.org/abs/1210.5644)

**DenseCRF**在所有像素对之间建立连接，并最小化能量：

$$
E(\mathbf x)=\sum_i\psi_u(x_i)+\sum_{i<j}\psi_p(x_i,x_j)
$$

一元势$\psi_u(x_i)=-\log P(x_i)$来自分割网络的概率；二元势由**Potts**兼容函数$\mu(x_i,x_j)=[x_i\neq x_j]$与两个高斯核组成：

$$
\begin{aligned}
\psi_p(x_i,x_j)=\mu(x_i,x_j)\bigg[w_1\exp\!\left(-\frac{\|p_i-p_j\|^2}{2\theta_\alpha^2}-\frac{\|I_i-I_j\|^2}{2\theta_\beta^2}\right) \\
+w_2\exp\!\left(-\frac{\|p_i-p_j\|^2}{2\theta_\gamma^2}\right)\bigg]
\end{aligned}
$$

其中$p$为位置、$I$为颜色。外观核鼓励位置相近且颜色相似的像素取相同标签，平滑核去除孤立的小区域。精确推理不可行，论文用平均场近似迭代更新各像素的标签分布；每轮消息传递等价于特征空间中的高斯滤波，可借助**permutohedral lattice**在像素数的线性时间内完成。**DeepLab v1/v2**即以它作为后处理，其核参数需要在验证集上搜索。

### ⚪ **CRF-RNN**：把平均场推理展开为可训练网络
- **paper**：[**Conditional Random Fields as Recurrent Neural Networks**](https://arxiv.org/abs/1502.03240)

**CRF-RNN**把**DenseCRF**的一次平均场迭代拆成若干可微算子：以高斯滤波完成消息传递，以$1\times1$卷积对各滤波结果加权，以可学习的类别兼容矩阵替代固定的**Potts**模型，再与一元势合并并经**softmax**归一化。设$Q^{(t)}$为第$t$轮的标签分布、$U$为一元势，单次迭代可写为：

$$
Q^{(t+1)}=\operatorname{softmax}\!\left(-U-\mu\Big(\sum_m w_m\,k_m\otimes Q^{(t)}\Big)\right)
$$

其中$k_m\otimes Q$表示对$Q$在空间上做第$m$个高斯滤波，$\mu$作用于类别维。把这一迭代重复$T$次并共享参数，就得到一个循环网络。它接在**FCN**之后端到端训练，核权重和兼容矩阵均由反向传播学习；论文训练时展开$5$次、测试时展开$10$次。**CRF**由此从独立的后处理变为网络中的一层。

### ⚪ **SegFix**：用内部像素的预测替换边界像素
- **paper**：[**SegFix: Model-Agnostic Boundary Refinement for Segmentation**](https://arxiv.org/abs/2007.04269)

**SegFix**观察到分割模型在物体内部的预测通常可靠，误差集中在边界附近。它训练一个与分割模型无关的独立网络，预测两张图：二值**boundary map**指出哪些像素位于边界带；**direction map**为每个像素给出指向所属物体内部的方向，其训练目标由各类区域距离变换的梯度得到，并离散化为若干方向类别。推理时，只对边界像素沿预测方向移动若干像素，用该内部位置的标签替换原预测：

$$
\tilde y_p=\begin{cases}y_{p+\Delta p}, & p\in\mathcal B\\ y_p, & \text{otherwise}\end{cases}
$$

其中$\mathcal B$为预测的边界像素集合，$\Delta p$为由方向得到的偏移。由于只依赖输入图像和已有分割结果，**SegFix**可以离线应用到**DeepLab v3、HRNet**等任意模型的输出上，并在**Cityscapes、ADE20K**等数据集上稳定改善边界精度。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-076-segfix.png)

### ⚪ **Hierarchical Multi-Scale Attention**：学习相邻尺度之间的融合权重
- **paper**：[**Hierarchical Multi-Scale Attention for Semantic Segmentation**](https://arxiv.org/abs/2005.10821)

简单平均多尺度预测会把小尺度在细结构上的错误和大尺度在大物体上的错误一并带入结果。**Hierarchical Multi-Scale Attention**让网络在较低尺度额外预测逐像素注意力$\alpha$，表示该位置更应相信低分辨率还是高分辨率输入的结果。对相邻尺度$0.5$与$1.0$，融合后的类别得分为：

$$L_{1.0}^{\text{fused}}=\operatorname{Up}\!\left(\alpha_{0.5}\odot L_{0.5}\right)+\left(1-\operatorname{Up}(\alpha_{0.5})\right)\odot L_{1.0}$$

训练只使用一对相邻尺度，学到的是相对权重；推理时把同一规则逐级串联到$0.25、0.5、1.0、2.0$等更多尺度。相比训练时同时输入所有尺度的注意力方法，它的训练显存更低，测试时也可以自由增减尺度。论文还用该模型为**Cityscapes**粗标注图像自动生成精细伪标签，再加入训练。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-077-hmsa.png)

## (4) 半监督、弱监督与噪声标签

像素级标注成本高，实践中常只有少量精标与大量无标注图像，或只有图像级标签与带噪掩码。半监督方法让模型在扰动下保持一致并利用伪标签；弱监督方法从分类激活图出发扩散出像素标签；噪声标签方法则在训练中逐步修正标注。

### ⚪ **CutMix与ClassMix**：用区域混合构造一致性扰动
- **paper**：[**Semi-supervised semantic segmentation needs strong, varied perturbations**](https://arxiv.org/abs/1906.01916)

半监督分类常依赖“决策边界应位于低密度区域”的聚类假设，但分割中相邻像素对应的图像块高度相关，这一假设在输入空间并不成立。论文因此采用**CutMix**式强扰动：用矩形掩码$M$混合两张无标注图像，要求学生网络对混合图像的预测，与教师网络对两张原图的预测按同一掩码拼接后的结果一致：

$$\mathcal L_{\text{unsup}}=\ell\!\left(f_\theta(M\odot x_A+(1-M)\odot x_B),\ M\odot f_{\bar\theta}(x_A)+(1-M)\odot f_{\bar\theta}(x_B)\right)$$

其中教师参数$\bar\theta$是学生参数的指数滑动平均。

- **paper**：[**ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning**](https://arxiv.org/abs/2007.07936)

**ClassMix**把矩形改为语义形状：先用教师网络预测$x_A$，随机选择其中一半预测类别，令这些类别覆盖的像素构成$M$，再把它们粘贴到$x_B$上，并以按同一掩码混合的伪标签监督学生。掩码沿预测的物体边界切割，混合图像比矩形拼接更接近真实场景，同时保留了强扰动的效果。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-078-classmix.png)

### ⚪ **CPS**：两个网络互相提供伪标签
- **paper**：[**Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision**](https://arxiv.org/abs/2106.01226)

**Cross Pseudo Supervision（CPS）**使用两个结构相同、初始化不同的分割网络$f_{\theta_1}$与$f_{\theta_2}$。对同一输入，两者分别输出概率$P_1,P_2$，取逐像素**argmax**得到**one-hot**伪标签$Y_1,Y_2$，再让每个网络拟合另一个网络的伪标签：

$$\mathcal L_{\text{cps}}=\frac{1}{|\mathcal D|}\sum_{x\in\mathcal D}\frac{1}{WH}\sum_{i=1}^{WH}\Big[\ell_{\text{ce}}(p_{1i},y_{2i})+\ell_{\text{ce}}(p_{2i},y_{1i})\Big]$$

该损失同时作用于有标注和无标注图像，总目标为$\mathcal L=\mathcal L_s+\lambda\mathcal L_{\text{cps}}$。不同初始化使两个网络犯不同的错误，交叉监督既扩充了训练数据，也鼓励两者在无标注图像上达成一致；论文进一步结合**CutMix**扰动。推理时只需使用其中一个网络。

### ⚪ **UniMatch**：扩展弱到强一致性的扰动空间
- **paper**：[**Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation**](https://arxiv.org/abs/2208.09910)

**FixMatch**式方法对无标注图像先做弱增强得到预测$p^w$，只保留置信度高于阈值$\tau$的像素作为伪标签，再监督强增强视图的预测$p^s$：

$$\mathcal L_u=\frac1{|\Omega|}\sum_{i\in\Omega}\mathbb 1\!\left(\max p^w_i\ge\tau\right)H\!\left(p^w_i,p^s_i\right)$$

**UniMatch**指出这一框架在分割中已是强基线，并从两个方向扩大扰动：其一是**feature perturbation**，在弱视图的编码器特征上施加**Dropout**后送入解码器，得到预测$p^{fp}$，使扰动不局限于图像空间；其二是**dual-stream perturbation**，从同一弱视图生成两个强增强视图，并都由$p^w$监督，以更充分地利用伪标签。无标注损失为：

$$\mathcal L_u=\lambda\,\mathcal L_{fp}+\frac{\mu}{2}\left(\mathcal L_{s_1}+\mathcal L_{s_2}\right)$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-079-unimatch.png)

### ⚪ **CAM与AffinityNet**：从图像级标签扩散出像素伪标签
- **paper**：[**Learning Deep Features for Discriminative Localization**](https://arxiv.org/abs/1512.04150)

弱监督语义分割只给出“图像中有哪些类别”。**Class Activation Map（CAM）**在分类网络最后的特征图$f(x)$上接全局平均池化和线性分类器，训练后把类别$c$的分类权重直接作用于每个位置：

$$
M_c(x)=w_c^\top f(x)
$$

响应高的位置即对分类最有贡献的区域。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-custom-003-5fd47f68.jpg)

- **paper**：[**Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation**](https://arxiv.org/abs/1803.10464)

**CAM**通常只覆盖物体最具判别性的部分，因此**AffinityNet**进一步学习像素间的语义亲和度：先从**CAM**中取高置信前景与背景区域，对半径$\gamma$内的像素对生成亲和标签（同类为$1$、异类为$0$、不确定区域忽略），再训练网络输出特征$f^{\text{aff}}$，亲和度定义为：

$$
a_{ij}=\exp\!\left(-\|f^{\text{aff}}_i-f^{\text{aff}}_j\|_1\right)
$$

以逐元素$\beta$次幂$A^{\circ\beta}$构造转移矩阵$T=D^{-1}A^{\circ\beta}$，**CAM**通过随机游走在同一物体内部扩散：$\operatorname{vec}(M^\ast)=T^t\operatorname{vec}(M)$。修正后的激活图生成像素伪标签，再用于训练常规分割网络。这一“种子定位→亲和传播→伪标签训练”的流程是许多弱监督分割方法的基本框架。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-080-affinitynet.png)

### ⚪ **Self-Correction**：用模型与标签聚合缓解标注噪声
- **paper**：[**Self-Correction for Human Parsing**](https://arxiv.org/abs/1910.09777)

图像分割任务的标签可能存在噪声。**自校正（self-correction）**从带噪标签出发，通过聚合当前模型和前一阶段模型的参数推断更可靠的标签，再用更新后的标签训练模型。

设当前阶段模型参数为$\hat{w}$，前一阶段聚合参数为$\hat{w}_{m-1}$，则模型聚合为：

$$ \hat{w}_m = \frac{m}{m+1}\hat{w}_{m-1} + \frac{1}{m+1}\hat{w} $$

标签也可通过当前预测$\hat{y}$与前一阶段标签$\hat{y}_{m-1}$聚合：

$$ \hat{y}_m = \frac{m}{m+1}\hat{y}_{m-1} + \frac{1}{m+1}\hat{y} $$

训练先在原始带噪标签上预热，随后采用循环学习率；每个循环结束时执行一次模型聚合，并在训练集上重新估计聚合模型的**Batch Normalization**统计量，再用聚合模型的预测更新标签。模型聚合使预测更稳定，更好的标签又反过来改善下一轮训练，二者在循环中相互促进。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-039-6231958a.jpg)

## (5) 跨域自适应

分割标注昂贵，常用合成数据（如**GTA5、SYNTHIA**）训练模型，再部署到真实城市场景（如**Cityscapes**），但纹理、光照与传感器差异会造成明显的域偏移。**无监督域自适应（unsupervised domain adaptation, UDA）**只使用带标注的源域与无标注的目标域，主要依靠对抗对齐和自训练两类策略。

### ⚪ **AdaptSegNet**：在输出空间对齐源域与目标域
- **paper**：[**Learning to Adapt Structured Output Space for Semantic Segmentation**](https://arxiv.org/abs/1802.10349)

**AdaptSegNet**认为，不同域的图像外观差异很大，但分割输出的空间布局与局部语义结构（如天空在上、道路在下）相似，因此在**softmax**输出空间做对抗对齐比在高维特征空间更容易。判别器$D$接收分割概率图$P$并区分其来自源域还是目标域；分割网络在源域上优化交叉熵，同时让目标域输出骗过判别器：

$$
\begin{aligned}
\mathcal L(I_s,I_t)&=\mathcal L_{\text{seg}}(I_s)+\lambda_{adv}\mathcal L_{adv}(I_t),\\ \mathcal L_{adv}(I_t)&=-\sum_{h,w}\log D\big(P_t\big)^{(h,w,1)}
\end{aligned}
$$

论文还在中间层增加辅助分割输出及对应判别器，形成多层级对抗学习，并在**GTA5→Cityscapes**与**SYNTHIA→Cityscapes**上验证。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-081-adaptsegnet.png)

### ⚪ **DAFormer**：用Transformer与自训练策略改进域自适应
- **paper**：[**DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation**](https://arxiv.org/abs/2111.14887)

**DAFormer**指出，早期**UDA**方法多沿用**DeepLab v2**，网络结构本身限制了自适应效果。它以**SegFormer**的**MiT-B5**为编码器，并设计融合多层级特征与多扩张率上下文的解码器。训练沿用**DACS**式自训练：教师网络为学生的指数滑动平均，在目标域图像上产生伪标签，并按高置信像素比例为伪标签损失加权；源域与目标域图像再按**ClassMix**方式混合。论文还提出三项稳定训练的策略：
- **Rare Class Sampling**：按类别频率$f_c$以概率$P(c)\propto\exp\!\left((1-f_c)/T\right)$采样含该类的源域图像，使稀有类别更早进入自训练；
- **Thing-Class ImageNet Feature Distance**：约束可数物体类别上的瓶颈特征接近**ImageNet**预训练模型的特征，减轻对合成域的过拟合；
- **Learning Rate Warmup**：逐步提高学习率，避免早期的大幅更新破坏预训练特征。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-082-daformer.png)

# 3. 图像分割的评估指标

图像分割任务本质上是一种图像像素分类任务，可以使用常见的分类评价指标来评估模型的好坏。图像分割中常用的评估指标包括：
- 像素准确率 (**pixel accuracy, PA**)
- 类别像素准确率 (**class pixel accuracy, CPA**)
- 类别平均像素准确率 (**mean pixel accuracy, MPA**)
- 交并比 (**Intersection over Union, IoU**)
- 平均交并比 (**mean Intersection over Union, MIoU**)
- 频率加权交并比 (**Frequency Weighted Intersection over Union, FWIoU**)
- **Dice**系数 (**Dice Coefficient**)

上述评估指标均建立在**混淆矩阵**的基础之上，因此首先介绍混淆矩阵，然后介绍这些评估指标的计算。

## ⚪ 混淆矩阵
图像分割问题本质上是对图像中的像素的分类问题。

### (1) 二分类
以**二分类**为例，图像中的每个像素可能属于**正例(Positive)**也可能属于**反例(Negative)**。根据像素的实际类别和模型的预测结果，可以把像素划分为以下四类中的某一类：
- **真正例 TP(True Positive)**：实际为正例，预测为正例
- **假正例 FP(False Positive)**：实际为反例，预测为正例
- **真反例 TN(True Negative)**：实际为反例，预测为反例
- **假反例 FN(False Negative)**：实际为正例，预测为反例

绘制分类结果的**混淆矩阵(confusion matrix)**如下：

$$ \begin{array}{l|cc} \text{真实情况\预测结果} & \text{正例} & \text{反例} \\ \hline  \text{正例} & TP & FN \\  \text{反例} & FP & TN \\ \end{array} $$

根据混淆矩阵可做如下计算：
- **准确率(accuracy)**，定义为模型分类正确的像素比例：

$$ \text{Accuracy} = \frac{TP+TN}{TP+FP+TN+FN} $$

- **查准率(precision)**，定义为模型预测为正例的所有像素中，真正为正例的像素比例：

$$ \text{Precision} = \frac{TP}{TP+FP} $$

- **查全率(recall)**,又称**召回率**，定义为所有真正为正例的像素中，模型预测为正例的像素比例：

$$ \text{Recall} = \frac{TP}{TP+FN} $$

- **F1分数(F1-Score)**，定义为查准率和召回率的调和平均数。

$$ \text{F}_1 = 2\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision}+\text{Recall}} $$

### (2) 多分类

图像分割通常是**多分类**问题，也有类似结论。对于多分类问题，**混淆矩阵**表示如下：

$$ \begin{array}{l|ccc} \text{真实情况\预测结果} & \text{类别1} & \text{类别2} & \text{类别3} \\ \hline  \text{类别1} & a & b & c \\  \text{类别2} & d & e & f \\ \text{类别3} & g & h & i \\ \end{array} $$

对于多分类问题，也可计算：
- **准确率**：

$$ \text{Accuracy} = \frac{a+e+i}{a+b+c+d+e+f+g+h+i} $$

- **查准率**，以类别$1$为例：

$$ \text{Precision} = \frac{a}{a+d+g} $$

- **查全率**，以类别$1$为例：

$$ \text{Recall} = \frac{a}{a+b+c} $$

### (3) 计算混淆矩阵
对于图像分割的预测结果`imgPredict`和真实标签`imgLabel`，可以使用[np.bincount](https://0809zheng.github.io/2020/09/11/bincount.html)函数计算混淆矩阵，计算过程如下：

<details markdown="1">
<summary>代码实现</summary>

```python
import numpy as np

def genConfusionMatrix(numClass, imgPredict, imgLabel):
    '''
    Parameters
    ----------
    numClass : 类别数(不包括背景).
    imgPredict : 预测图像.
    imgLabel : 标签图像.
    '''
    # remove classes from unlabeled pixels in gt image and predict
    mask = (imgLabel >= 0) & (imgLabel < numClass)
    
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass**2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

imgPredict = np.array([[0,1,0],
                 [2,1,0],
                 [2,2,1]])
imgLabel = np.array([[0,2,0],
                  [2,1,0],
                  [0,2,1]])
print(genConfusionMatrix(3, imgPredict, imgLabel))

###
[[3 0 1]
 [0 2 0]
 [0 1 2]]
###
```

</details>

## ⚪ 像素准确率 PA
**像素准确率** (**pixel accuracy, PA**) 衡量所有类别预测正确的像素占总像素数的比例，相当于分类任务中的**准确率(accuracy)**。

**PA**计算为混淆矩阵对角线元素之和比矩阵所有元素之和，以二分类为例：

$$ \text{PA} = \frac{TP+TN}{TP+FP+TN+FN} $$

<details markdown="1">
<summary>代码实现</summary>

```python
def pixelAccuracy(confusionMatrix):
    # return all class overall pixel accuracy
    #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
    acc = np.diag(confusionMatrix).sum() /  confusionMatrix.sum()
    return acc
```

</details>

## ⚪ 类别像素准确率 CPA
**类别像素准确率（class pixel accuracy, CPA）**衡量真实属于类别$i$的像素中被正确预测的比例，等价于该类别的**召回率（recall）**。若混淆矩阵以行为真实类别、以列为预测类别，则第$i$类的**CPA**为：

$$ \text{CPA}_i = \frac{n_{ii}}{\sum_j n_{ij}} $$

以二分类正类为例：

$$ \text{CPA} = \frac{TP}{TP+FN} $$

<details markdown="1">
<summary>代码实现</summary>

```python
def classPixelAccuracy(confusionMatrix):
    denominator = confusionMatrix.sum(axis=1)
    return np.divide(
        np.diag(confusionMatrix),
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator != 0,
    )
```

</details>

若以列为真实类别，则行列方向应相应交换。原始计数约定必须在公式和代码中保持一致；按预测类别归一化得到的是**查准率（precision）**，不是标准语义分割协议中的类别准确率。

## ⚪ 类别平均像素准确率 MPA
**类别平均像素准确率（mean pixel accuracy, MPA）**是有效类别**CPA**的宏平均：

$$ \text{MPA} = \frac{1}{|\mathcal C_{\mathrm{valid}}|}\sum_{i\in\mathcal C_{\mathrm{valid}}}\text{CPA}_i $$

<details markdown="1">
<summary>代码实现</summary>

```python
def meanPixelAccuracy(confusionMatrix):
    return np.nanmean(classPixelAccuracy(confusionMatrix))
```

</details>

`np.nanmean`会忽略没有真实像素的类别产生的`NaN`，而不是把它置为$0$。是否排除背景和未出现类别必须随评测结果一并说明。

## ⚪ 交并比 IoU
**交并比** (**Intersection over Union, IoU**) 又称**Jaccard index**，衡量预测类别为$i$的像素集合$A$和真实类别为$i$的像素集合$B$的交集与并集之比。

$$ \text{IoU} = \frac{|A ∩ B |}{|A ∪ B|}= \frac{|A ∩ B |}{|A|+| B |-|A ∩ B |} $$

预测类别为$i$的像素集合是指所有预测为类别$i$的像素，用混淆矩阵第$i$列元素之和表示。真实类别为$i$的像素集合是指所有实际类别$i$的像素，用混淆矩阵第$i$行元素之和表示。

第$i$个类别的**IoU**计算为混淆矩阵第$i$个对角线元素比矩阵该列元素与该行元素的并集。以二分类为例，第$0$个类别的**IoU**计算为：

$$ \text{IoU} = \frac{TP}{TP+FP+FN} $$

<details markdown="1">
<summary>代码实现</summary>

```python
def IntersectionOverUnion(confusionMatrix):
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix) 
    IoU = intersection / union  
    return IoU # 返回列表，其值为各个类别的IoU
```

</details>

## ⚪ 平均交并比 MIoU
**平均交并比** (**mean Intersection over Union, MIoU**) 计算为所有类别的**IoU**的平均值:

$$ \text{MIoU} = \text{mean}(\text{IoU}) $$

<details markdown="1">
<summary>代码实现</summary>

```python
def meanIntersectionOverUnion(confusionMatrix):
    IoU = IntersectionOverUnion(confusionMatrix)
    mIoU = np.nanmean(IoU) # 求各类别IoU的平均
    return mIoU
```

</details>

## ⚪ 频率加权交并比 FWIoU
**频率加权交并比** (**Frequency Weighted Intersection over Union, FWIoU**) 按照真实类别为$i$对应像素占所有像素的比例对类别$i$的**IoU**进行加权。

第$i$个类别的**FWIoU**首先计算混淆矩阵第$i$行元素求和比矩阵所有元素求和，再乘以第$i$个类别的**IoU**。以二分类为例，第$0$个类别的**FWIoU**计算为：

$$ \text{FWIoU} = \frac{TP+FN}{TP+FP+FN+TN} \cdot \frac{TP}{TP+FP+FN} $$

最终给出的**FWIoU**应为所有类别**FWIoU**的求和。

<details markdown="1">
<summary>代码实现</summary>

```python
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) +
            np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU
```

</details>

## ⚪ Dice Coefficient
**Dice Coefficient**衡量预测类别为$i$的像素集合$A$和真实类别为$i$的像素集合$B$之间的重叠程度：

$$ \text{Dice} = \frac{2|A \cap B|}{|A|+|B|} $$

对于二值硬掩码，**Dice**与**IoU**可以相互换算：

$$ \text{Dice}=\frac{2\text{IoU}}{1+\text{IoU}},\qquad \text{IoU}=\frac{\text{Dice}}{2-\text{Dice}} $$

这一等价关系不应直接推广到训练中不同定义、不同平滑项和不同类别聚合方式的**soft Dice**与**soft IoU**。


第$i$个类别的**Dice**计算为混淆矩阵第$i$个对角线元素的两倍比矩阵该列元素与该行元素之和。以二分类为例，第$0$个类别的**Dice**计算为：

$$ \text{Dice} = \frac{2TP}{2TP+FP+FN} = \text{F1-score} $$

因此**Dice**系数等价于分类指标中的**F1-Score**。

<details markdown="1">
<summary>代码实现</summary>

```python
def Dice(confusionMatrix):
    # Dice = 2*TP / (TP + FP + TP + FN)
    intersection = np.diag(confusionMatrix)
    Dice = 2*intersection / (
        np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0))
    return Dice # 返回列表，其值为各个类别的Dice
```

</details>

特别地，对于二值分割问题，**Dice**系数可以直接通过$$\{0,1\}$$预测矩阵和标签矩阵计算：

<details markdown="1">
<summary>代码实现</summary>

```python
def dice_coef(pred, target):
    smooth = 1.
    m1 = pred.view(-1).float()
    m2 = target.view(-1).float()
    intersection = (m1 * m2).sum().float()
    dice = (2. * intersection + smooth) / (torch.sum(m1*m1) + torch.sum(m2*m2) + smooth)
    return dice
```

</details>

## ⚪ 实例分割与全景分割指标

实例分割需要同时判断类别、掩码质量与实例匹配，通常在多个掩码**IoU**阈值下计算**Average Precision（AP）**。只有类别一致且掩码**IoU**超过阈值的预测才算正确匹配，因此语义区域大致正确但把两个实例粘连在一起，仍会产生漏检和误检。

全景分割常用**Panoptic Quality（PQ）**。在**IoU**大于$0.5$的唯一匹配集合$TP$上，它定义为：

$$
\operatorname{PQ}=\frac{\sum_{(p,g)\in TP}\operatorname{IoU}(p,g)}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FN|}=\operatorname{SQ}\times\operatorname{RQ}
$$

其中**Segmentation Quality（SQ）**是匹配区域的平均**IoU**，**Recognition Quality（RQ）**衡量区域识别与匹配质量。**PQ**同时适用于**thing**和**stuff**类别，但评测时通常还应分别报告两者结果。

## ⚪ 边界质量与校准指标

区域指标可能掩盖细长结构和轮廓偏移：一个对象内部占据大量像素时，即使边界错开数个像素，**IoU**仍可能很高。需要关注轮廓质量时，可补充**Boundary IoU**、边界**F-score**、平均表面距离或**Hausdorff Distance**，并明确边界容忍半径和物理像素间距。

像素概率还可用**NLL、Brier score**或**ECE**评估校准。高**mIoU**只说明离散标签重叠较好，不保证$0.9$的预测置信度真的对应约$90\%$的正确率。对于医疗决策、自动驾驶和交互标注，置信度校准与失败检测往往和平均重叠指标同样重要。


# 4. 图像分割的损失函数

图像分割损失不仅衡量预测与标注的差异，还规定如何平衡易像素与难像素、大区域与小区域、内部与边界。常见损失可划分为：
- 基于分布的损失：**Cross-Entropy Loss、Weighted Cross-Entropy Loss、TopK/OHEM、Focal Loss、Distance Map Penalized CE Loss**；
- 基于区域的损失：**Sensitivity-Specificity Loss、soft IoU、Lovász-Softmax、Dice Loss、Tversky Loss、Focal Tversky Loss、Generalized Dice Loss**；
- 基于边界的损失：**Boundary Loss、Hausdorff Distance Loss**；
- 基于集合匹配的损失：为掩码查询与真实区域执行二分图匹配，再联合优化类别、掩码交叉熵或**Focal Loss**与**Dice Loss**。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-040-641e59a6.jpg)

## (1) 基于分布的损失 Distribution-based Loss

基于分布的损失函数旨在最小化两种分布之间的差异。

### ⚪ Cross-Entropy Loss

交叉熵损失是由**KL**散度导出的，衡量数据分布$P$和预测分布$Q$之间的差异：

$$
\begin{aligned}
D_{\mathrm{KL}}(P \parallel Q) & =\sum_i p_i \log \frac{p_i}{q_i} \\
& =-\sum_i p_i \log q_i+\sum_i p_i \log p_i \\
& =H(P, Q)-H(P)
\end{aligned}
$$

注意到数据分布$P$通常是已知的，因此最小化数据分布$P$和预测分布$Q$之间的**KL**散度等价于最小化交叉熵$H(P,Q)$。对于分割任务，指定$g_i^c$是像素$i$是否属于标签$c$的二元指示符，$s_i^c$是对应的预测结果，则交叉熵损失定义为：

$$
L_{C E}=-\frac{1}{N} \sum_{c=1}^C \sum_{i=1}^N g_i^c \log s_i^c
$$

<details markdown="1">
<summary>代码实现</summary>

```python
ce_loss = torch.nn.CrossEntropyLoss()
# result无需经过Softmax，gt为整型
loss = ce_loss(result, gt)
```

</details>

### ⚪ Weighted Cross-Entropy Loss

为缓解类别不平衡问题，加权交叉熵损失为每个类别指定一个权重$w_c$。权重$w_c$通常与类别出现频率成反比，比如设置为训练集中类别出现频率的倒数。

$$
L_{W C E}=-\frac{1}{N} \sum_{c=1}^C \sum_{i=1}^N w_c g_i^c \log s_i^c
$$

<details markdown="1">
<summary>代码实现</summary>

```python
wce_loss = torch.nn.CrossEntropyLoss(weight=weight)
loss = wce_loss(result, gt)
```

</details>

### ⚪ **TopK与OHEM**：集中优化高损失像素
- **paper**：[**Loss Max-Pooling for Semantic Image Segmentation**](https://arxiv.org/abs/1704.02966)

**TopK**损失或**在线难例挖掘（online hard example mining, OHEM）**按当前损失或置信度选择难像素，只在选中的集合$\mathcal K$上反向传播：

$$
L_{\text{TopK}}=-\frac{1}{|\mathcal K|}\sum_{i\in\mathcal K}\sum_{c=1}^{C}g_i^c\log s_i^c
$$

硬选择可以避免大量简单背景主导梯度，但$k$过小会放大错误标注和异常像素。与此不同，**Focal Loss**保留全部像素，只连续降低易例权重。

<details markdown="1">
<summary>代码实现</summary>

```python
class TopKLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, k=10):
        super(TopKLoss, self).__init__()
        self.k = k
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, result, gt):
        res = self.ce_loss(result, gt)
        num_pixels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_pixels * self.k / 100), sorted=False)
        return res.mean()
```

</details>

### ⚪ **Focal Loss**：连续降低易分类像素的权重
- **paper**：[**Focal Loss for Dense Object Detection**](https://arxiv.org/abs/1708.02002)

**Focal Loss**在交叉熵上乘以$(1-p_t)^\gamma$，让高置信度易例的梯度快速衰减；$\alpha_t$用于类别平衡，$\gamma$控制对难例的聚焦强度：

$$
L_{\text{Focal}}=-\frac{1}{N}\sum_{i=1}^{N}\alpha_{t_i}(1-p_{t_i})^\gamma\log p_{t_i}
$$

它最初面向稠密目标检测，也常用于前景—背景极不平衡的分割和掩码分类。**Focal Loss**并不会自动区分边界错误与区域错误，错误标签也可能因持续保持低置信度而被放大。

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        pt = (y_onehot * result).sum(1)
        logpt = pt.log()

        gamma = self.gamma
        loss = -1 * torch.pow((1 - pt), gamma) * logpt
        return loss.mean()
```

</details>

### ⚪ [Distance Map Penalized CE Loss](https://arxiv.org/abs/1908.03679)

距离图惩罚交叉熵损失通过由真实标签计算的[距离变换图](https://0809zheng.github.io/2023/03/22/distancetransform.html)对交叉熵进行加权，引导网络重点关注难以分割的边界区域。

$$
L_{D P C E}=-\frac{1}{N} \sum_{c=1}^C\left(1+D^c\right) \circ \sum_{i=1}^N g_i^c \log s_i^c
$$

其中$D^c$是类别$c$的距离惩罚项，通过取真实标签的距离变换图的倒数来生成。通过这种方式可以为边界上的像素分配更大的权重。

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange
from scipy.ndimage import distance_transform_edt

class DisPenalizedCE(torch.nn.Module):
    def __init__(self):
        super(DisPenalizedCE, self).__init__()

    @torch.no_grad()
    def one_hot2dist(self, seg):
        res = np.zeros_like(seg)
        for c in range(seg.shape[1]):
            posmask = seg[:,c,...]
            if posmask.any():
                negmask = 1.-posmask
                pos_edt = distance_transform_edt(posmask)
                pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
                neg_edt =  distance_transform_edt(negmask)
                neg_edt = (np.max(neg_edt)-neg_edt)*negmask        
                res[:,c,...] = pos_edt + neg_edt
        return res

    def forward(self, result, gt):
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 h w')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)
        dist = torch.from_numpy(self.one_hot2dist(y_onehot.cpu().numpy())+1).float()

        result = torch.softmax(result, dim=1)
        result_logs = torch.log(result)

        loss = -result_logs * y_onehot
        weighted_loss = loss*dist
        return weighted_loss.mean()
```

</details>

## (2) 基于区域的损失 Region-based Loss

基于区域的损失函数旨在最小化真实标签$G$和预测分割$S$之间的不匹配程度或者最大化两者之间的重叠区域。

### ⚪ [Sensitivity-Specificity Loss](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1)

敏感性-特异性损失通过加权敏感性与特异性来解决类别不平衡问题：

$$
\begin{aligned}
L_{S S}= & w \frac{\sum_{c=1}^C \sum_{i=1}^N\left(g_i^c-s_i^c\right)^2 g_i^c}{\sum_{c=1}^C \sum_{i=1}^N g_i^c+\epsilon} \\
& +(1-w) \frac{\sum_{c=1}^C \sum_{i=1}^N\left(g_i^c-s_i^c\right)^2\left(1-g_i^c\right)}{\sum_{c=1}^C \sum_{i=1}^N\left(1-g_i^c\right)+\epsilon}
\end{aligned}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum

class SSLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(SSLoss, self).__init__()
        self.smooth = smooth
        self.r = 0.1 # weight parameter in SS paper

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - result)**2
        specificity_numerator = einsum(squared_error, y_onehot, 'b c n, b c n -> b c')
        specificity_denominator = einsum(y_onehot, 'b c n -> b c')+self.smooth
        specificity_part = einsum(specificity_numerator, 'b c -> b')/einsum(specificity_denominator, 'b c -> b')
        sensitivity_numerator = einsum(squared_error, bg_onehot, 'b c n, b c n -> b c')
        sensitivity_denominator = einsum(bg_onehot, 'b c n -> b c')+self.smooth
        sensitivity_part = einsum(sensitivity_numerator, 'b c -> b')/einsum(sensitivity_denominator, 'b c -> b')

        ss = self.r * specificity_part + (1-self.r) * sensitivity_part
        return ss.mean()
```

</details>

### ⚪ [IoU Loss](https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22)

**IoU Loss**直接优化**IoU index**。由于预测热图和真实标签都可以表示为$[0,1]$矩阵，因此集合运算可以直接通过对应元素计算：

$$
L_{I O U}=1- \frac{|A ∩ B |}{|A|+| B |-|A ∩ B |}=1-\frac{\sum_{c=1}^C \sum_{i=1}^N g_i^c s_i^c}{\sum_{c=1}^C \sum_{i=1}^N\left(g_i^c+s_i^c-g_i^c s_i^c\right)}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        union = einsum(result, "b c n -> b c") + einsum(y_onehot, "b c n -> b c") - einsum(result, y_onehot, "b c n, b c n -> b c")
        divided = 1 - (einsum(intersection, "b c -> b") + self.smooth) / (einsum(union, "b c -> b") + self.smooth)
        return divided.mean()
```

</details>

### ⚪ [Lovász Loss](https://arxiv.org/abs/1705.08790)

**Lovász-Softmax Loss**采用[**Lovász**延拓](https://0809zheng.github.io/2023/03/25/submodularity.html)把离散的集合函数扩展为连续、凸且分段线性的**IoU**代理。它通常几乎处处可微，但不是处处光滑；核心操作是按像素误差排序，再按集合交并比的边际变化加权。

首先定义类别$c$的误分类像素集合$M_c$：

$$
\mathbf{M}_c\left(\boldsymbol{y}^*, \tilde{\boldsymbol{y}}\right)=\left\{\boldsymbol{y}^*=c, \tilde{\boldsymbol{y}} \neq c\right\} \cup\left\{\boldsymbol{y}^* \neq c, \tilde{\boldsymbol{y}}=c\right\}
$$

则**IoU Loss**可以写成集合$M_c$的函数：

$$
\Delta_{J_c}: \mathbf{M}_c \in\{0,1\}^N \mapsto \frac{\left|\mathbf{M}_c\right|}{\left|\left\{\boldsymbol{y}^*=c\right\} \cup \mathbf{M}_c\right|}
$$

定义类别$c$的像素误差向量$m(c) \in [0,1]^N$：

$$
m_i(c) = \begin{cases} 1-s_i^c, & \text{if }c=\boldsymbol{y}^*_i \\ s_i^c, & \text{otherwise} \end{cases}
$$

则$$\Delta_{J_c}(\mathbf{M}_c)$$的**Lovász**延拓$$\overline{\Delta_{J_c}}(m(c))$$根据定义可表示为：

$$
\overline{\Delta_{J_c}}: m \in R^N \mapsto \sum_{i=1}^N m_{\pi(i)} g_i(m)
$$

其中$$g_i(m)=\Delta_{J_c}(\{\pi_1,...,\pi_i\})-\Delta_{J_c}(\{\pi_1,...,\pi_{i-1}\})$$，$\pi$是$m$中元素的一个按递减顺序排列：$m_{\pi_1} \geq m_{\pi_2} \geq \cdots \geq m_{\pi_N}$。

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    n = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if n > 1:  # cover 1-pixel case
        jaccard[1:n] = jaccard[1:n] - jaccard[0:-1]
    return jaccard

class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            input_c = inputs[:, c]
            loss_c = (target_c - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, lovasz_grad(target_c_sorted)))
        losses = torch.stack(losses)
        return losses.mean()

    def forward(self, inputs, targets):
        # inputs.shape = (batch size, class_num, h, w)
        # targets.shape = (batch size, h, w)
        inputs = rearrange(inputs, 'b c h w -> (b h w) c')
        targets = targets.view(-1)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
```

</details>


### ⚪ **Dice Loss**：直接优化前景区域重叠
- **paper**：[**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**](https://arxiv.org/abs/1606.04797)

**Dice Loss**与**IoU Loss**类似，直接优化**Dice Coefficient**。由于预测热图和真实标签都可以表示为$[0,1]$矩阵，因此集合运算可以直接通过对应元素计算：

$$
L_{\text {Dice }}=1-\frac{2|A ∩ B |}{|A|+| B |}=1-\frac{2 \sum_{c=1}^C \sum_{i=1}^N g_i^c s_i^c}{\sum_{c=1}^C \sum_{i=1}^N g_i^c+\sum_{c=1}^C \sum_{i=1}^N s_i^c}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum
   
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        union = einsum(result, "b c n -> b c") + einsum(y_onehot, "b c n -> b c")
        divided = 1 - 2 * (einsum(intersection, "b c -> b") + self.smooth) / (einsum(union, "b c -> b") + self.smooth)
        return divided.mean()
```

</details>

### ⚪ [Tversky Loss](https://arxiv.org/abs/1706.05721)

**Dice Loss**可以被视为查准率和召回率的调和平均值，它对假阳性和假阴性样本的权重相等。**Tversky Loss**在**Dice Loss**的分母中调整了假阳性和假阴性样本的权重，以实现查准率和召回率之间的权衡。

$$
\begin{aligned}
L_{\text {Tversky }}= & 1-\left(\sum_{c=1}^C \sum_{i=1}^N g_i^c s_i^c\right) /\left(\sum_{c=1}^C \sum_{i=1}^N g_i^c s_i^c\right. \\
& \left.+\alpha \sum_{c=1}^C \sum_{i=1}^N\left(1-g_i^c\right) s_i^c+\beta \sum_{c=1}^C \sum_{i=1}^N g_i^c\left(1-s_i^c\right)\right)
\end{aligned}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum

class TverskyLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        FP = einsum(result, 1-y_onehot, "b c n, b c n -> b c")
        FN = einsum(1-result, y_onehot, "b c n, b c n -> b c")
        denominator = intersection + self.alpha * FP + self.beta * FN
        divided = 1 - einsum(intersection, "b c -> b") / einsum(denominator, "b c -> b").clamp(min=self.smooth)
        return divided.mean()
```

</details>

### ⚪ [Focal Tversky Loss](https://arxiv.org/abs/1810.07842)

**Focal Tversky Loss**对整体**Tversky**误差施加幂变换，而不是像**Focal Loss**那样逐像素调权。原论文采用：

$$ L_{\text{FTL}}=(1-\operatorname{TI})^{1/\gamma}=(L_{\text{Tversky}})^{1/\gamma} $$

一些实现改用$(1-\operatorname{TI})^\gamma$。两种约定在$\gamma>1$时的曲率相反，对高、低误差区域的梯度权重也不同，因此不能仅凭“**focal**”名称推断其优化效果，报告实验时必须写清公式与参数范围。

### ⚪ [Asymmetric Similarity Loss](https://ieeexplore.ieee.org/document/8573779)

**Asymmetric Similarity Loss**和**Tversky Loss**的动机类似，也是调整假阳性和假阴性样本的权重，以平衡查准率和召回率。该损失相当于设置**Tversky Loss**中$\alpha+\beta=1$：

$$
\begin{aligned}
L_{\text {Asym }}= & 1-\left(\sum_{c=1}^C \sum_{i=1}^N g_i^c s_i^c\right) /\left(\sum_{c=1}^C \sum_{i=1}^N g_i^c s_i^c\right. \\
& \left.+\frac{\beta^2}{1+\beta^2} \sum_{c=1}^C \sum_{i=1}^N\left(1-g_i^c\right) s_i^c+\frac{1}{1+\beta^2} \sum_{c=1}^C \sum_{i=1}^N g_i^c\left(1-s_i^c\right)\right)
\end{aligned}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum

class AsymLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(AsymLoss, self).__init__()
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        weight = (self.beta**2)/(1+self.beta**2)
        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        FP = einsum(result, 1-y_onehot, "b c n, b c n -> b c")
        FN = einsum(1-result, y_onehot, "b c n, b c n -> b c")
        denominator = intersection + weight * FP + (1-weight) * FN
        divided = 1 - einsum(intersection, "b c -> b") / einsum(denominator, "b c -> b").clamp(min=self.smooth)
        return divided.mean()
```

</details>

### ⚪ [Generalized Dice Loss](https://arxiv.org/abs/1707.03237)

**Generalized Dice Loss**是**Dice Loss**的多类别扩展，其中每个类别的权重与标签频率成反比：$w_c=1/(\sum_{i=1}^Ng_i^c)^2$。

$$
L_{\text {GD }}=1-\frac{2 \sum_{c=1}^C w_c \sum_{i=1}^N g_i^c s_i^c}{\sum_{c=1}^C w_c \sum_{i=1}^N (g_i^c+s_i^c)}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum

class GDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(GDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        w = 1 / (einsum(y_onehot, "b c n -> b c") + 1e-10)**2
        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        union = einsum(result, "b c n -> b c") + einsum(y_onehot, "b c n -> b c")
        divided = 1 - 2 * (einsum(intersection, w, "b c, b c -> b") + self.smooth) / (einsum(union, w, "b c, b c -> b") + self.smooth)
        return divided.mean()
```

</details>

### ⚪ [Penalty Loss](https://openreview.net/forum?id=H1lTh8unKN)

**Penalty Loss**把**Tversky Loss**中调整假阳性和假阴性样本权重的思想引入**Generalized Dice Loss**。

$$
\begin{aligned}
L_{\text {Penalty }}= & 1-2\left(\sum_{c=1}^C  w_c \sum_{i=1}^N g_i^c s_i^c\right) /\left(\sum_{c=1}^C  w_c \sum_{i=1}^N (g_i^c+ s_i^c)\right. \\
& \left.+k \sum_{c=1}^C  w_c \sum_{i=1}^N\left(1-g_i^c\right) s_i^c+k \sum_{c=1}^C  w_c \sum_{i=1}^N g_i^c\left(1-s_i^c\right)\right)
\end{aligned}
$$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum

class PenaltyLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(PenaltyLoss, self).__init__()
        self.smooth = smooth
        self.k = 2.5

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        w = 1 / (einsum(y_onehot, "b c n -> b c") + 1e-10)**2
        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        union = einsum(result+y_onehot, "b c n -> b c")
        FP = einsum(result, 1-y_onehot, "b c n, b c n -> b c")
        FN = einsum(1-result, y_onehot, "b c n, b c n -> b c")
        denominator = einsum(union, w, "b c, b c -> b") + self.k * einsum(FP, w, "b c, b c -> b") + self.k * einsum(FN, w, "b c, b c -> b")
        divided = 1 - 2 * einsum(intersection, w, "b c, b c -> b") / denominator.clamp(min=self.smooth)
        return divided.mean()
```

</details>


## (3) 基于边界的损失 Boundary-based Loss

基于边界的损失是指在目标的轮廓空间而不是区域空间上采用距离度量的形式定义的损失函数，衡量真实标签和预测分割中目标边界之间的距离。

有两种不同的框架来计算两个边界之间的距离。一种是**微分**框架，它通过计算每个点沿边界曲线法线上的速度来评估每个点的运动情况。另一种是**积分**框架，它通过计算两个边界的不匹配区域的积分来近似距离。

在训练神经网络时，边界损失通常应该与基于区域的损失相结合，以减少训练的不稳定性。

### ⚪ **Boundary Loss**：用有符号距离场近似边界距离
- **paper**：[**Boundary loss for highly unbalanced segmentation**](https://arxiv.org/abs/1812.07032)

在**Boundary Loss**中，每个点$q$的**softmax**输出$s_{\theta}(q)$通过$\phi_G$进行加权。$\phi_G:\Omega\to\mathbb R$是真实标签边界$\partial G$的有符号距离表示：如果$q\in G$则$\phi_G(q)=-D_G(q)$，否则$\phi_G(q)=D_G(q)$。$D_G:\Omega\to\mathbb R^+$是相对于边界$\partial G$的[距离变换图](https://0809zheng.github.io/2023/03/22/distancetransform.html)。边界上的值为$0$，错误地把概率质量放到目标外部会产生正代价，把质量放到目标内部则产生负贡献。

$$ \mathcal{L}_B(\theta) = \int_{\Omega} \phi_G(q) s_{\theta}(q) d q $$

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange, einsum
from scipy.ndimage import distance_transform_edt

class BDLoss(nn.Module):
    def __init__(self):
        super(BDLoss, self).__init__()

    @torch.no_grad()
    def one_hot2dist(self, seg):
        res = np.zeros_like(seg)
        for c in range(seg.shape[1]):
            posmask = seg[:,c,...]
            if posmask.any():
                negmask = 1.-posmask
                neg_map = distance_transform_edt(negmask)
                pos_map = distance_transform_edt(posmask)
                res[:,c,...] = neg_map * negmask - (pos_map - 1) * posmask
        return res

    def forward(self, result, gt):
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 h w')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        bound = torch.from_numpy(self.one_hot2dist(y_onehot.cpu().numpy())).float()
        # only compute the loss of foreground
        pc = result[:, 1:, ...]
        dc = bound[:, 1:, ...]
        multipled = pc * dc
        return multipled.mean()
```

</details>

### ⚪ [Hausdorff Distance Loss](https://arxiv.org/abs/1904.10030)

豪斯多夫距离损失通过[距离变换图](https://0809zheng.github.io/2023/03/22/distancetransform.html)来近似并优化真实标签和预测分割之间的[Hausdorff距离](https://0809zheng.github.io/2021/03/03/distance.html#-%E8%B1%AA%E6%96%AF%E5%A4%9A%E5%A4%AB%E8%B7%9D%E7%A6%BB-hausdorff-distance)：

$$
L_{H D}=\frac{1}{N} \sum_{c=1}^C \sum_{i=1}^N\left[\left(s_i^c-g_i^c\right)^2 \circ\left(d_{G_i^c}^{\alpha}+d_{S_i^c}^{\alpha}\right)\right]
$$

其中$d_G,d_S$分别是真实标签和预测分割的距离变换图，计算每个像素与目标边界之间的最短距离。

<details markdown="1">
<summary>代码实现</summary>

```python
from einops import rearrange
from scipy.ndimage import distance_transform_edt

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def one_hot2dist(self, seg):
        res = np.zeros_like(seg)
        for c in range(seg.shape[1]):
            posmask = seg[:,c,...]
            if posmask.any():
                negmask = 1.-posmask
                pos_edt = distance_transform_edt(posmask)
                neg_edt = distance_transform_edt(negmask)      
                res[:,c,...] = pos_edt + neg_edt
        return res

    def forward(self, result, gt):
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 h w')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.data, 1)

        pred_dt = torch.from_numpy(self.one_hot2dist(result.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.one_hot2dist(y_onehot.cpu().numpy())).float()

        pred_error = (result - y_onehot) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        return dt_field.mean()
```

</details>

## (4) 基于集合匹配的损失 Set-based Loss

**MaskFormer**一类模型输出无序掩码集合，训练前必须决定哪个查询对应哪个真实区域。设预测集合为$\{(p_i,m_i)\}_{i=1}^{N}$，真实集合为$\{(c_j,y_j)\}_{j=1}^{M}$，二分图匹配寻找总代价最小的注入映射：

$$
\hat\sigma=\arg\min_{\sigma}\sum_{j=1}^{M}\mathcal C\big((c_j,y_j),(p_{\sigma(j)},m_{\sigma(j)})\big)
$$

匹配代价通常组合类别代价、掩码**Focal/BCE**代价和**Dice**代价。匹配完成后，掩码损失只作用于配对查询；分类损失还会监督未配对查询预测“无对象”类别。由于全分辨率逐点匹配成本很高，实际实现常随机采样像素估计掩码损失。

# 5. 常用的图像分割数据集

图像分割任务广泛应用于场景理解、自动驾驶、遥感和医学影像。数据集不仅在图像数量和类别数上不同，还可能采用语义、实例、全景、粗标注、弱标注或提示掩码等不同协议。模型迁移时尤其要检查类别映射、`ignore`标签、训练分辨率和许可证。

### ⚪ **PASCAL VOC**：经典通用目标语义分割基准
- **dataset**：[**The PASCAL Visual Object Classes Challenge**](http://host.robots.ox.ac.uk/pascal/VOC/)

**PASCAL VOC 2012**包含$20$个前景类别和背景类，官方分割训练集较小，实践中常加入**Semantic Boundaries Dataset（SBD）**扩展标注。它适合验证经典方法和低数据训练。

### ⚪ **COCO-Stuff**：在实例对象之外标注场景材质区域
- **paper**：[**COCO-Stuff: Thing and Stuff Classes in Context**](https://arxiv.org/abs/1612.03716)

**COCO-Stuff**在**COCO**的对象实例标注之外加入天空、墙面、道路等**stuff**类别，为语义与全景场景理解提供更丰富的长尾类别。使用时要区分不同版本的类别合并和训练划分。

### ⚪ [**Cityscapes**](https://www.cityscapes-dataset.com/)：精细城市街景分割

**Cityscapes**面向城市街景理解，包含来自$50$个城市的$5{,}000$张精细标注图像和约$20{,}000$张粗标注图像。常用语义协议评估$19$个类别，并提供实例与全景标注。高分辨率、细小交通参与者和类别不平衡使它也常用于实时分割、域适应与鲁棒性研究。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-042-63f2dc32.jpg)

### ⚪ **Mapillary Vistas**：覆盖多地区与多天气的街景
- **paper**：[**The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes**](https://arxiv.org/abs/1711.11533)

**Mapillary Vistas**收集来自不同国家、设备、季节、天气和拍摄条件的高分辨率街景，类别粒度比**Cityscapes**更细。它适合检验跨地区泛化和长尾道路类别。

### ⚪ [**ADE20K**](http://groups.csail.mit.edu/vision/datasets/ADE20K/)：覆盖多样室内外场景

**ADE20K**的常用语义分割协议包含$150$个类别、$20{,}210$张训练图像和$2{,}000$张验证图像。类别丰富、长尾明显且场景构成复杂，因此成为评估通用场景解析、**Transformer**和视觉基础模型的重要基准。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-043-63f2dca0.jpg)

### ⚪ [SYNTHIA](http://synthia-dataset.net)

**SYNTHIA**是计算机合成的城市道路驾驶环境的像素级标注的数据集。是为了在自动驾驶或城市场景规划等研究领域中的场景理解而提出的。提供了**11**个类别物体（分别为天空、建筑、道路、人行道、栅栏、植被、杆、车、信号标志、行人、骑自行车的人）细粒度的像素级别的标注。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-044-5ebb5eb7.jpg)

### ⚪ [APSIS](http://xiaoyongshen.me/webpage_portrait/index.html)

**人体肖像分割数据库（Automatic Portrait Segmentation for Image Stylization, APSIS）**面向前景人物与背景的二值分割，适合研究肖像抠图。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/dl-seg-045-5ebb5e07.jpg)
