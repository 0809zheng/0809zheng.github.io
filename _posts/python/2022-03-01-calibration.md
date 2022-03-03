---
layout: post
title: '使用opencv-python(cv2)库进行相机标定'
date: 2022-03-01
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/621dcba52ab3f51d912c6fd8.jpg'
tags: Python
---

> Camera Calibration with opencv-python.

本文首先建立相机成像模型，然后介绍相机的畸变问题，之后定义相机的标定并介绍张正友标定法，最后介绍张氏标定法的**python**实现。

# 1. 相机成像模型

为描述相机模型，需要建立空间中的四种坐标系：
1. **世界坐标系**：代表物体在真实世界中的三维坐标，坐标系用$x_w,y_w,z_w$(单位：毫米)表示。
2. **相机坐标系**：代表以相机光学中心为原点的坐标系，相机光轴与$z$轴重合，坐标系用$x_c,y_c,z_c$(单位：毫米)表示。
3. **图像坐标系**：代表相机感光板(即成像平面)上的坐标系，原点$O_n$为相机光轴与成像平面的交点(图像的中心点)，坐标系用$x_n,y_n$(单位：毫米)表示。
4. **像素坐标系**：代表图像上的点在图像存储矩阵中的像素位置，图像左上角为坐标原点$O_u$，坐标系用$x_u,y_u$(单位：像素)表示。

![](https://pic.imgdb.cn/item/621dcb802ab3f51d912c269e.jpg)

则相机成像过程可以表示为三次坐标系变换的过程：

![](https://pic.imgdb.cn/item/621dcba52ab3f51d912c6fd8.jpg)

### ⚪ 世界坐标系→相机坐标系：刚体变换

世界坐标系是真实世界的基准坐标系，而相机坐标系是以相机为中心建立的坐标系。我们需要知道真实世界中的点$(x_w,y_w,z_w)$在相机坐标系中的位置$(x_c,y_c,z_c)$，这可以通过一次齐次坐标变换(刚体变换)得到：

$$ \begin{bmatrix} x_c \\ y_c \\ z_c \\ 1 \end{bmatrix} =  \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_w \\ y_w \\ z_w \\ 1 \end{bmatrix} $$

其中$R \in \Bbb{R}^{3 \times 3}$是正交旋转矩阵，$t \in \Bbb{R}^{3 \times 1}$是平移矢量。

### ⚪ 相机坐标系→图像坐标系：透视投影

把相机看作小孔成像模型(对于广角相机不适用)，则相机坐标系中的三维点$(x_c,y_c,z_c)$可以通过简单的中心透视投影变换为图像中的二维点$(x_n,y_n)$：

$$ \begin{cases} x_n = \frac{f}{z_c}x_c \\ y_n = \frac{f}{z_c}y_c \end{cases} \\ \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix}  = \begin{bmatrix} \frac{f}{z_c} & 0 & 0 & 0 \\0 & \frac{f}{z_c} & 0 & 0 \\ 0 & 0 & \frac{1}{z_c} & 0 \end{bmatrix} \begin{bmatrix} x_c \\ y_c \\ z_c \\ 1 \end{bmatrix} $$

其中$f$是相机的焦距。

### ⚪ 图像坐标系→像素坐标系：仿射变换
若记图像中每个像素的物理长度和宽度为$dx,dy$，图像中心的像素坐标为$(u_0,v_0)$，则图像中的二维点$(x_n,y_n)$在像素坐标系下的位置可通过线性仿射变换得到：

$$ \begin{cases} x_u = u_0+\frac{x_n}{dx} \\ y_u = v_0+\frac{y_n}{dy} \end{cases} \\ \begin{bmatrix} x_u \\ y_u \\ 1 \end{bmatrix}  = \begin{bmatrix} \frac{1}{dx} & 0 & u_0\\0 & \frac{1}{dy} & v_0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} $$

### ⚪ 相机成像模型

相机成像模型可以写作：

$$  \begin{bmatrix} x_u \\ y_u \\ 1 \end{bmatrix}  = \begin{bmatrix} \frac{1}{dx} & 0 & u_0\\0 & \frac{1}{dy} & v_0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} \frac{f}{z_c} & 0 & 0 & 0 \\0 & \frac{f}{z_c} & 0 & 0 \\ 0 & 0 & \frac{1}{z_c} & 0 \end{bmatrix} \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_w \\ y_w \\ z_w \\ 1 \end{bmatrix} \\ = \begin{bmatrix} \frac{f}{dxz_c} & 0 & \frac{u_0}{z_c} & 0 \\0 & \frac{f}{dyz_c} & \frac{v_0}{z_c} & 0 \\ 0 & 0 & \frac{1}{z_c} & 0 \end{bmatrix} \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_w \\ y_w \\ z_w \\ 1 \end{bmatrix} $$

不失一般性地假设$z_c=1$，并引入相机感光板横边和纵边的角度误差$\theta$ ($90$°表示无误差)，则单点无畸变的相机成像模型如下：

$$  \begin{bmatrix} x_u \\ y_u \\ 1 \end{bmatrix}  = \begin{bmatrix} \frac{f}{dx} & -\frac{f \cot \theta}{dx} & u_0 & 0 \\0 & \frac{f}{dy \sin \theta} & v_0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_w \\ y_w \\ z_w \\ 1 \end{bmatrix} $$

其中矩阵$$\begin{bmatrix} \frac{f}{dx} & -\frac{f \cot \theta}{dx} & u_0 & 0 \\0 & \frac{f}{dy \sin \theta} & v_0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$称为相机的**内参(intrinsic)矩阵**，其参数由相机的内部参数决定，主要包括焦距$f$,感光板夹角$\theta$,像素在相机感光板上的物理长度$dx,dy$和感光板中心在像素坐标系下的坐标$(u_0,v_0)$。

矩阵$$\begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$称为相机的**外参(extrinsic)矩阵**，外参矩阵取决于相机坐标系和世界坐标系的相对位置。在标定时由于每张照片的标定平面不同，外参矩阵也不同。

# 2. 相机畸变

相机成像时会引入非线性畸变。常见的畸变包括由于相机光学系统的透镜组不完善造成的**径向畸变**(**枕形畸变**+**桶形畸变**)、由于不正确的镜头组合造成的**离心畸变**、由于相机装配不完善造成的**透镜畸变**。

### ⚪ 径向畸变

由于透镜系统光轴附近的放大率与远光轴区域的放大率存在差异，导致导致图像上的点会向内或向外偏离光轴中心，从而造成径向畸变。

如果远光轴区域的放大率比光轴附近的大，则距离光轴中心越远的点的距离会变得更远，从而造成**枕形畸变**；反之，如果远光轴区域的放大率小于光轴附近的放大率，则造成**桶形畸变**。

![](https://pic.imgdb.cn/item/621de69f2ab3f51d91621e92.jpg)

由径向畸变造成的前两阶畸变量可以表示为：

$$ \begin{cases} \delta_x = k_1x_u(x_u^2+y_u^2) + k_2x_u(x_u^2+y_u^2)^2 \\ \delta_y = k_1y_u(x_u^2+y_u^2) + k_2y_u(x_u^2+y_u^2)^2 \end{cases} $$

### ⚪ 离心畸变
离心畸变是指实际相机的光学系统的光学中心和镜头各器件的光学中心不一致带来的畸变，包括径向畸变和切向畸变。

由离心畸变造成的前两阶畸变量可以表示为：

$$ \begin{cases} \delta_x = p_1(3x_u^2+y_u^2) + 2p_2x_uy_u \\ \delta_y = p_2(x_u^2+3y_u^2) + 2p_1x_uy_u \end{cases} $$

### ⚪ 透镜畸变
透镜畸变是指透镜设计和加工不完善、安装误差导致的畸变，也包括径向畸变和切向畸变，实际影响较小。

由透镜畸变造成的一阶畸变量可以表示为：

$$ \begin{cases} \delta_x = q_1(x_u^2+y_u^2)  \\ \delta_y = q_2(x_u^2+y_u^2) \end{cases} $$

### ⚪ 畸变模型

高阶畸变通常影响较小，在实际中畸变模型仅考虑包含一、二阶径向畸变和一、二阶离心畸变即可。考虑畸变后的像素坐标系中的位置$(x_u,y_u)$修正为：

$$ \begin{cases} x_u = x_u + x_u(k_1r_u^2 + k_2r_u^4) + p_1(r_u^2+2x_u^2) + 2p_2x_uy_u \\ y_u = y_u+y_u(k_1r_u^2 + k_2r_u^4) + p_2(r_u^2+2y_u^2) + 2p_1x_uy_u \end{cases} $$

其中$r_u^2=x_u^2+y_u^2$。

# 3. 相机的标定
相机成像可以进行视觉测量，即通过相机捕捉的图像信息获取真实三维世界中的位置信息。相机标定的目的是建立真实世界中的物理距离与图像中像素坐标的映射关系。为使测量结果可信，标定过程需要达成两个目的：
1. 建立世界坐标系和像素坐标系之间的映射关系：获取相机的内参矩阵($f,\theta,dx,dy,u_0,v_0$)和外参矩阵($R,t$)。
2. 校正畸变：获取相机的畸变参数($k_1,k_2,p_1,p_2$)。

值得一提的是，仅仅通过单目相机标定的结果是无法直接从像素坐标转化到物理坐标的，因为透视投影时丢失了一个维度的坐标($z$轴)。因此使用相机对真实空间进行测距需要双目相机。

# 4. 张正友标定法

[张正友标定法](https://ieeexplore.ieee.org/document/888718)使用如下图所示的棋盘格标定板。将世界坐标系固定于棋盘格平面上($z_w=0$)，由于标定板上的棋盘格距离已知，可以得到每一个角点在世界坐标系下的物理坐标$(U,V,0)$。使用相机获得标定板的图像后，可以利用相应的图像检测算法检测到每个角点的像素坐标$(u,v)$。根据相机成像模型，可以获得相机的内外参矩阵和畸变参数。

![](https://pic.imgdb.cn/item/62207a595baa1a80aba81b10.jpg)

首先假设相机无畸变，则相机成像模型为：

$$  Z\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}  = \begin{bmatrix} \frac{f}{dx} & -\frac{f \cot \theta}{dx} & u_0  \\0 & \frac{f}{dy \sin \theta} & v_0  \\ 0 & 0 & 1  \end{bmatrix} \begin{bmatrix} R_1 & R_2 & t  \end{bmatrix} \begin{bmatrix} U \\ V  \\ 1 \end{bmatrix} \\= A \begin{bmatrix} R_1 & R_2 & t  \end{bmatrix} \begin{bmatrix} U \\ V  \\ 1 \end{bmatrix} $$

其中$Z$为尺度因子，为简化计算提取出来。记$A$为内参矩阵，$R_1,R_2$为旋转矩阵$R$的前两列。对于同一个相机，内参矩阵$A$为定值。对于同一张图片，内参矩阵$A$和外参矩阵$$\begin{bmatrix} R_1 & R_2 & t  \end{bmatrix}$$为定值。对于同一张图片中的单点，内参矩阵$A$、外参矩阵$$\begin{bmatrix} R_1 & R_2 & t  \end{bmatrix}$$和尺度因子$Z$为定值。

## ⚪ 求解内参矩阵与外参矩阵的积

记内参矩阵与外参矩阵的乘积为如下单应性(**homography**)矩阵$H$：

$$ H= A \begin{bmatrix} R_1 & R_2 & t  \end{bmatrix} = \begin{bmatrix} H_1 & H_2 & H_3  \end{bmatrix}= \begin{bmatrix} H_{11} & H_{12} & H_{13} \\ H_{21} & H_{22} & H_{23} \\ H_{31} & H_{32} & H_{33}  \end{bmatrix} $$

则成像模型可写为：

$$  \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}  = \frac{1}{Z} H \begin{bmatrix} U \\ V  \\ 1 \end{bmatrix} = \frac{1}{Z} \begin{bmatrix} H_{11} & H_{12} & H_{13} \\ H_{21} & H_{22} & H_{23} \\ H_{31} & H_{32} & H_{33}  \end{bmatrix} \begin{bmatrix} U \\ V  \\ 1 \end{bmatrix} $$

上式消去尺度因子$Z$，可得：

$$ u = \frac{H_{11}U+H_{12}V+H_{13}}{H_{31} U+ H_{32} V+ H_{33}},\quad v = \frac{H_{21}U+H_{22}V+H_{23}}{H_{31} U+ H_{32} V+ H_{33}} $$

其中角点的世界坐标$(U,V,0)$和像素坐标$(u,v)$是已知的。矩阵$H$是齐次矩阵，共有$8$个独立未知元素。标定板上的每个角点能够提供两个约束方程，因此当标定板上的角点数量至少为$4$个时，能够通过最小二乘法求得矩阵$H$。

## ⚪ 求解内参矩阵
记相机的内参矩阵$A$：

$$  A = \begin{bmatrix} \frac{f}{dx} & -\frac{f \cot \theta}{dx} & u_0  \\0 & \frac{f}{dy \sin \theta} & v_0  \\ 0 & 0 & 1  \end{bmatrix} = \begin{bmatrix} \alpha & \gamma & u_0  \\0 & \beta & v_0  \\ 0 & 0 & 1  \end{bmatrix} $$

已知矩阵$$H=A\begin{bmatrix} R_1 & R_2 & t  \end{bmatrix}$$，下面求内参矩阵$A$。
注意到旋转矩阵$R$的两列$R_1,R_2$存在单位正交关系：

$$ R_1^TR_2 = 0 \\ R_1^TR_1 = R_2^TR_2 = 1 $$

根据$R_1=A^{-1}H_1,R_2=A^{-1}H_2$可得：

$$ H_1^TA^{-T}A^{-1}H_2 = 0 \\ H_1^TA^{-T}A^{-1}H_1 = H_2^TA^{-T}A^{-1}H_2 = 1 $$

注意到上述三个约束方程中均存在矩阵$A^{-T}A^{-1}$，记对称矩阵$B=A^{-T}A^{-1}$：

$$ B=A^{-T}A^{-1} = \begin{bmatrix} \frac{1}{\alpha} & 0 &  0 \\-\frac{\gamma}{\alpha\beta} & \frac{1}{\beta} & 0  \\ \frac{\gamma v_0-\beta u_0}{\alpha \beta} & -\frac{v_0}{\beta} & 1  \end{bmatrix} \begin{bmatrix} \frac{1}{\alpha} & -\frac{\gamma}{\alpha\beta} & \frac{\gamma v_0-\beta u_0}{\alpha \beta}  \\0 & \frac{1}{\beta} & -\frac{v_0}{\beta}  \\ 0 & 0 & 1  \end{bmatrix} \\ = \begin{bmatrix} \frac{1}{\alpha^2} & -\frac{\gamma}{\alpha^2\beta} &  \frac{\gamma v_0-\beta u_0}{\alpha^2 \beta} \\-\frac{\gamma}{\alpha^2\beta} & \frac{1}{\beta^2}+\frac{\gamma^2}{\alpha^2\beta^2} & \frac{\gamma(\beta u_0-\gamma v_0)}{\alpha^2 \beta^2}-\frac{v_0}{\beta^2}  \\ \frac{\gamma v_0-\beta u_0}{\alpha^2 \beta} & \frac{\gamma(\beta u_0-\gamma v_0)}{\alpha^2 \beta^2}-\frac{v_0}{\beta^2} & \frac{(\gamma v_0-\beta u_0)^2}{\alpha^2 \beta^2}+\frac{v_0^2}{\beta^2}+1  \end{bmatrix} \\ =  \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33}  \end{bmatrix} $$

引入矩阵$B$后约束方程为：

$$ H_1^TBH_2 = 0 \\ H_1^TBH_1 = H_2^TBH_2 = 1 $$

为求解矩阵$B$，首先计算矩阵$H_i^TBH_j$：

$$ H_i^TBH_j = \begin{bmatrix} H_{1i} & H_{2i} & H_{3i}  \end{bmatrix} \begin{bmatrix} B_{11} & B_{12} & B_{13} \\ B_{12} & B_{22} & B_{23} \\ B_{13} & B_{23} & B_{33}  \end{bmatrix} \begin{bmatrix} H_{1j} \\ H_{2j} \\ H_{3j}  \end{bmatrix} \\ =  \begin{bmatrix} H_{1i}H_{1j} \\ H_{1i}H_{2j}+H_{2i}H_{1j} \\ H_{2i}H_{2j} \\ H_{1i}H_{3j}+H_{3i}H_{1j} \\  H_{2i}H_{3j}+H_{3i}H_{2j} \\ H_{3i}H_{3j}  \end{bmatrix}^T \begin{bmatrix} B_{11} \\B_{12} \\ B_{22} \\ B_{13} \\ B_{23} \\ B_{33} \end{bmatrix}  $$

记上式两个相乘的向量分别为$v_{ij}$和$b$，则有$H_i^TBH_j=v_{ij}^Tb$；对应的约束方程为：

$$ v_{12}^Tb = 0 \\ v_{11}^Tb = v_{22}^Tb = 1 $$

上式又写作：

$$ \begin{bmatrix} v_{12}^T \\ v_{11}^T-v_{22}^T  \end{bmatrix}b =vb=0 $$

其中矩阵$v$是由矩阵$H$构成的，因此是已知的。对于上式，只需要求解出向量$b$，即可得到矩阵$B$。矩阵$B$是对称矩阵，共有$6$个独立未知元素。每张标定板的图片能够提供$vb=0$的两个约束方程，因此当标定板图片数量至少为$3$张时，能够通过最小二乘法求得向量$b$(即矩阵$B$)。

求得矩阵$B$后，便可根据对应关系求得相机的内参矩阵：

$$ \alpha = \sqrt{\frac{1}{B_{11}}}, \quad \beta=\sqrt{\frac{B_{11}}{B_{11}B_{22}-B_{12}^2}}, \quad \gamma=-B_{12}\alpha^2\beta \\ v_0=\frac{B_{12}B_{13}-B_{11}B_{23}}{B_{11}B_{22}-B_{12}^2} , \quad u_0 = \frac{\gamma v_0}{\beta}-B_{13}\alpha^2 $$


## ⚪ 求解外参矩阵
外参矩阵反映了标定板和相机的位置关系，对于不同的图片，标定板和相机的位置关系发生变化，因此每一张图片对应的外参矩阵都是不同的。

在关系式$$H= A \begin{bmatrix} R_1 & R_2 & t  \end{bmatrix}$$中，已经求得矩阵$H$和矩阵$A$，因此外参矩阵计算为：

$$ \begin{bmatrix} R_1 & R_2 & t  \end{bmatrix} = A^{-1}H $$

由于世界坐标系建立在标定板平面上，因此棋盘格上任意一点的坐标$z=0$。旋转矩阵$R$的第三列$R_3$在坐标系变换中没有起作用，由于$R_3$与旋转矩阵的另外两列$R_1,R_2$正交，因此可以由该两列向量的叉乘得到：$R_3=R_1 \times R_2$。

## ⚪ 求解畸变参数

上面的推导均假设不存在畸变参数。如果考虑相机的畸变参数，则畸变后的像素坐标系中的位置$(u,v)$修正为：

$$ \begin{cases} \hat{u} = u + u(k_1r^2 + k_2r^4) + p_1(r^2+2u^2) + 2p_2uv \\ \hat{v} = v+v(k_1r^2 + k_2r^4) + p_2(r^2+2v^2) + 2p_1uv \end{cases} $$

其中$r^2=u^2+v^2$。只需知道理想无畸变的像素坐标$(u,v)$和畸变后的像素坐标$(\hat{u},\hat{v})$便可以建立畸变参数的方程组。

畸变后的像素坐标$(\hat{u},\hat{v})$可以直接通过识别标定板的角点获得。理想无畸变的像素坐标$(u,v)$可通过成像模型近似求得：已知角点的世界坐标$(U,V,0)$，通过前述过程中计算得到的内参矩阵$A$和外参矩阵$$\begin{bmatrix} R_1 & R_2 & t  \end{bmatrix}$$进行投影可以得到近似正确的像素坐标：

$$ u = \frac{H_{11}U+H_{12}V+H_{13}}{H_{31} U+ H_{32} V+ H_{33}},\quad v = \frac{H_{21}U+H_{22}V+H_{23}}{H_{31} U+ H_{32} V+ H_{33}} $$

理论上每个角点能够提供关于畸变参数的两个方程。如果考虑更高阶的畸变，则需要更多的约束方程。

## ⚪ 非线性优化
注意到在求解内参矩阵和外参矩阵时，假设成像过程不存在畸变；而在求解畸变参数时，假设内参矩阵和外参矩阵无误差。因此通过上面的过程求解得到的内参矩阵、外参矩阵和畸变参数是不准确的，需要通过非线性优化(如**L-M**算法)对各参数进行迭代优化。

假设共收集$m$张标定板图像，每张图像上有$n$个标定板角点。记$m_{ij}$为直接通过识别标定板的角点获得的畸变后的像素坐标$(\hat{u},\hat{v})$，$\hat{m}$为通过反投影得到的理想无畸变的像素坐标$(u,v)$的近似。建立优化目标函数：

$$ \mathop{\min} \sum_{i=1}^{m}\sum_{j=1}^{n}||m_{ij}-\hat{m}(A;R,t;k_1.k_2,p_1,p_2)||^2 $$

优化上述目标函数，即可得到的内参矩阵、外参矩阵和畸变参数的较为准确的值。

# 5. 使用**opencv-python**库标定相机

使用张正友标定法进行相机标定的步骤如下：
1. 准备一个棋盘格图片，棋盘格大小已知(见文末)；使用相机对棋盘格图片进行不同角度的拍摄，实践中通常收集$15$-$20$张标定板图片。
2. 使用图像处理算法检测图像中的棋盘格角点，得到角点的像素坐标。根据已知的棋盘格大小和世界坐标系原点，计算角点的物理坐标。
3. 求解内参矩阵和外参矩阵。
4. 求解畸变参数。
5. 使用**L-M**算法对上述参数进行进一步优化。

使用[**opencv-python**](http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/calib3d/doc/calib3d.html)库可以方便地实现相机的标定。

### ⚪ cv2库中的标定相关函数

- `cv2.findChessboardCorners(image, patternSize)`：寻找棋盘格的角点像素位置。输入棋盘格图像`image`和角点尺寸`patternSize`；返回检测返回值`retval`和角点像素坐标`corners`。
- `cv2.drawChessboardCorners(image, patternSize, corners, patternWasFound)`：绘制检测到的角点。输入棋盘格图像`image`、角点尺寸`patternSize`、角点像素坐标`corners`和检测返回值`patternWasFound`。
- `cv2.calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix=None, distCoeffs=None)`:标定相机并返回标定参数。输入角点的物理坐标`objectPoints`和像素坐标`imagePoints`、图像尺寸`imageSize`以及相机内参`cameraMatrix`和畸变参数`distCoeffs`的初始值；返回检测返回值`retval`、相机内参矩阵`cameraMatrix`、畸变参数`distCoeffs`、外参旋转矢量`rvecs`和外参平移矢量`tvecs`。

### ⚪相机标定的完整程序

定义棋盘格的角点数`inter_corner_shape`（方格数减$1$）和格距`size_per_grid`，将格式为`img_type`的标定板图片放在路径`img_dir`下。标定函数如下：

```python
import os
import glob
import numpy as np
import cv2

def calib(inter_corner_size=(7,5), size_per_grid=0.02, img_dir, img_type='png'):
    # 计算角点的世界坐标
    w, h = inter_corner_shape
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    cp_world = cp_int*size_per_grid

    obj_points = [] # 存储角点的物理坐标
    img_points = [] # 存储角点的像素坐标
    # 使用glob库遍历标定板图像，os.sep表示适应系统的分隔符
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用findChessboardCorners寻找角点的像素位置
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w,h))
        # ret=Ture表示寻找到角点
        if ret == True:
            obj_points.append(cp_world)
            img_points.append(cp_img)
            # 使用drawChessboardCorners绘制棋盘格图像中的角点
            cv2.drawChessboardCorners(img, (w,h), cp_img, ret)
            cv2.imshow('FoundCorners', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    # 使用获得相机标定参数
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(
        obj_points, img_points, gray_img.shape[::-1], None, None)
    print('ret:',ret)
    print('internal matrix:\n',mat_inter) # 内参矩阵
    print('distortion cofficients:\n',coff_dis) # 畸变参数，形式(k_1,k_2,p_1,p_2,p_3)
    print('rotation vectors:\n',v_rot) # 外参：旋转向量
    print('translation vectors:\n',v_trans) # 外参：平移向量
```

可以使用**重投影(reproject)**误差衡量标定结果的好坏。假设共收集$m$张标定板图像，每张图像上有$n$个标定板角点。记$(u,v)$为直接通过识别标定板的角点获得的畸变后的像素坐标，$(\hat{u},\hat{v})$为通过投影得到的理想无畸变的像素坐标的近似。则重投影误差计算为：

$$ \frac{1}{mn} \sum_{i=1}^{m}\sum_{j=1}^{n}\sqrt{(\hat{u}_{ij}-u_{ij})^2+(\hat{v}_{ij}-v_{ij})^2} $$

使用`projectPoints`可以根据角点的物理坐标`obj_points`和标定得到的参数计算角点的投影像素坐标，并进一步计算重投影误差：

```python
total_error = 0
for i in range(len(obj_points)):
    img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
    error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
    total_error += error
print('average error of reproject:',total_error/len(obj_points))
```

附：棋盘格图片（角点数$7\times 5$，格距$0.02$m）

![](https://pic.imgdb.cn/item/622028b05baa1a80ab69a3ad.png)