---
layout: post
title: '全色锐化(Panchromatic Sharpening)'
date: 2024-10-08
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/673af40cd29ded1a8c6d8d11.png'
tags: 深度学习
---

> Panchromatic Sharpening.

由于受遥感卫星传感器设计、遥感成像机理等因素的影响，单源遥感图像在空间、光谱分辨率等方面相互制约，一般遥感卫星只能获得单幅低空间分辨率的**多光谱（Multispectral, MS）**图像或高空间分辨率的**全色（Panchromatic, PAN）**图像。

**全色锐化 (Panchromatic Sharpening)**是指将全色图像的高分辨率空间细节信息与多光谱图像的丰富光谱信息进行融合，得到高质量、理想的**高空间分辨率多光谱（High Spatial Resolution Multispectral, HRMS）**图像。

![](https://pic.imgdb.cn/item/673af40cd29ded1a8c6d8d11.png)

像素级融合是直接在原始遥感图像各像素上的直接融合处理，其目的是为了获得质量更高的融合图像，如提升观测图像的分辨率、增强原始图像的清晰度等。像素级全色图像锐化方法通常分为:
1. 成分替换法(**CS-based**)：使用全色图像对多光谱图像的成分进行替换，如**Brovey**变换, **PCA**变换, **IHS**变换, **GS**变换, **GSA**, **CNMF**, **GFPCA**。
2. 多分辨率分析法(**MRA-based**)：对全色图像和多光谱图像不同尺度的高、低频成份进行融合，如**SFIM**变换, **Wavelet**变换, **MTF-GLP**, **MTF-GLP-HPM**。
3. 模型优化法(**MO-based**)：建立并优化融合图像与全色图像和多光谱图像之间的能量函数，如**SFIM**, **Wavelet**。
4. 深度学习方法(**DL-based**)：使用深度学习模型自动学习图像特征，从而实现图像分辨率的提升，如**PNN**, **PanNet**。



### 👉 参考文献
- [基于深度学习的像素级全色图像锐化研究综述](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20211325/)
- [Awesome-Pansharpening](https://github.com/Lihui-Chen/Awesome-Pansharpening)
- [Rewrite some pansharpening methods with python](https://github.com/codegaj/py_pansharpening/tree/master)

## 1. 成分替换法 Component Substitution

成分替换法(**CS-based**)先将多光谱图像转换到一个新的空间，然后在新的映射空间用全色图像对转换后的多光谱图像空间信息成份进行替换，其不足在于光谱失真较为明显。

### ⚪ Brovey变换
- A. R. Gillespie, A. B. Kahle, and R. E. Walker, “Color enhancement of highly correlated images-II. Channel ratio and “Chromaticity” Transform techniques,”         Remote Sensing of Environment, vol. 22, no. 3, pp. 343–365, August 1987.

**Brovey**变换基于光谱建模，旨在提高数据直方图高端和低端的视觉对比度。假定全色图像所跨越的光谱范围与多光谱通道覆盖的范围相同，该变换所采用的方法是将各个重采样的多光谱像素乘以相应全色像素亮度与所有多光谱亮度总和的比值：

$$
c_r = \frac{c_r}{(c_r+c_b+c_g)} \cdot PAN
$$

```python
def Brovey(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    u_hs = upsample(hs, ratio)
    
    I = np.mean(u_hs, axis=-1)
    
    image_hr = (pan-np.mean(pan))*(np.std(I, ddof=1)/np.std(pan, ddof=1))+np.mean(I)
    image_hr = np.squeeze(image_hr)

    I_Brovey=[]
    for i in range(C):
        temp = image_hr*u_hs[:, :, i]/(I+1e-8)
        temp = np.expand_dims(temp, axis=-1)
        I_Brovey.append(temp)
        
    I_Brovey = np.concatenate(I_Brovey, axis=-1) 
    
    #adjustment
    I_Brovey[I_Brovey<0]=0
    I_Brovey[I_Brovey>1]=1
    
    return np.uint8(I_Brovey*255)
```

### ⚪ PCA变换
- P. S. Chavez Jr. and A. W. Kwarteng, “Extracting spectral contrast in Landsat Thematic Mapper image data using selective principal component analysis,”         Photogrammetric Engineering and Remote Sensing, vol. 55, no. 3, pp. 339–348, March 1989.

**PCA**变换将多光谱图像转换到主成分空间，利用多波段数据的协方差来确定主要信息的方向；其第一个主成分通常代表了最大的信息变异，可以被视为亮度信息。通过替换第一个主成分为高分辨率的全色图像，提高多光谱图像的空间分辨率。将修改后的主成分数据逆变换回原来的多光谱波段，得到增强后的多光谱图像。

```python
def PCA(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    u_hs = upsample_interp23(hs, ratio)

    image_hr = pan
    
    p = princomp(n_components=C)
    pca_hs = p.fit_transform(np.reshape(u_hs, (M*N, C)))
    
    pca_hs = np.reshape(pca_hs, (M, N, C))
    
    I = pca_hs[:, :, 0]
    
    image_hr = (image_hr - np.mean(image_hr))*np.std(I, ddof=1)/np.std(image_hr, ddof=1)+np.mean(I)
    
    pca_hs[:, :, 0] = image_hr[:, :, 0]
    
    I_PCA = p.inverse_transform(pca_hs)
    
    #equalization
    I_PCA = I_PCA-np.mean(I_PCA, axis=(0, 1))+np.mean(u_hs)
    
    #adjustment
    I_PCA[I_PCA<0]=0
    I_PCA[I_PCA>1]=1
    
    return np.uint8(I_PCA*255)
```

### ⚪ IHS变换
- W. Carper, T. Lillesand, and R. Kiefer, “The use of Intensity-Hue-Saturation transformations for merging SPOT panchromatic and multispectral image data,”         Photogrammetric Engineering and Remote Sensing, vol. 56, no. 4, pp. 459–467, April 1990.

**IHS（Intensity-Hue-Saturation）**变换将原始多光谱图像从 **RGB** 色彩空间转换为 **IHS** 空间。此步骤能够分离出亮度信息（**Intensity**），色调（**Hue**），以及饱和度（**Saturation**）。使用高分辨率的全色图像替换转换后的 **IHS** 图像的亮度通道，这样可以增强图像的空间分辨率。将修改后的 **IHS** 图像转换回 **RGB** 色彩空间，得到增强后的多光谱图像。

```python
def IHS(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    u_hs = upsample_interp23(hs, ratio)
    
    I = np.mean(u_hs, axis=-1, keepdims=True)
    
    P = (pan - np.mean(pan))*np.std(I, ddof=1)/np.std(pan, ddof=1)+np.mean(I)
    
    I_IHS = u_hs + np.tile(P-I, (1, 1, C))
    
    #adjustment
    I_IHS[I_IHS<0]=0
    I_IHS[I_IHS>1]=1
    
    return np.uint8(I_IHS*255)
```

### ⚪ GS变换
- C. A. Laben and B. V. Brower, “Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening,” Eastman Kodak Company, Tech. Rep. US Patent # 6,011,875, 2000.

**Gram-Schmidt（GS）**变换基于与经典的正交化过程类似的思想。
1. 将多光谱波段线性组合成一个模拟全色图像，该图像在空间上与真实的全色图像尽可能相似。
2. 使用模拟全色图像作为 **Gram-Schmidt** 正交化过程的第一基向量；将模拟全色图像从各个多光谱波段中去除，生成一组新的互相正交的波段（主成分）。
3. 用真实的高分辨率全色图像替换模拟全色图像，对正交化后的波段逆变换，得到增强的多光谱图像。

```python
def GS(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    u_hs = upsample_interp23(hs, ratio)
    
    #remove means from u_hs
    means = np.mean(u_hs, axis=(0, 1))
    image_lr = u_hs-means
    
    #sintetic intensity
    I = np.mean(u_hs, axis=2, keepdims=True)
    I0 = I-np.mean(I)
    
    image_hr = (pan-np.mean(pan))*(np.std(I0, ddof=1)/np.std(pan, ddof=1))+np.mean(I0)
    
    #computing coefficients
    g = []
    g.append(1)
    
    for i in range(C):
        temp_h = image_lr[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(c[0,1]/np.var(I0))
    g = np.array(g)
    
    #detail extraction
    delta = image_hr-I0
    deltam = np.tile(delta, (1, 1, C+1))
    
    #fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    
    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    
    g = np.tile(g, (M, N, 1))
    
    V_hat = V+ g*deltam
    
    I_GS = V_hat[:, :, 1:]
    
    I_GS = I_GS - np.mean(I_GS, axis=(0, 1))+means
    
    #adjustment
    I_GS[I_GS<0]=0
    I_GS[I_GS>1]=1
    
    return np.uint8(I_GS*255)
```

### ⚪ GSA
- B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, October 2007.

**GSA (Generalized Spatial and Attribute) **通过结合空间和属性信息来实现全色图像和多光谱图像的融合。该方法的核心思想是通过计算每个像素的空间和光谱权重，并在融合过程中应用这些权重。
1. 分析全色图像和多光谱图像在空间域上的梯度或边缘信息，计算每个像素的空间权重。通常，高梯度或高边缘强度区域赋予较高权重，因为这些区域包含重要的空间细节。
2. 分析多光谱图像的光谱属性，计算每个像素的光谱权重。光谱权重用于保持多光谱图像的光谱信息。
3. 使用计算得到的空间权重和光谱权重，按照一定的融合规则将全色图像和多光谱图像的像素进行组合，生成具有高空间分辨率的多光谱图像。

相关程序可参考[GSA.py](https://github.com/codegaj/py_pansharpening/blob/master/methods/GSA.py)。

### ⚪ CNMF
- N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion," IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.

**CNMF**（**Coupled Non-negative Matrix Factorization**，耦合非负矩阵分解）的核心思想是将多光谱图像和全色图像表示为非负矩阵，并通过非负矩阵分解（**NMF**）技术分解为低维特征矩阵和系数矩阵。通过约束和耦合两个图像的分解过程，能够实现图像的融合和细节提升。
1. 将多光谱图像和全色图像表示为二维矩阵，其中每列代表一个像素，每行代表一个波段或空间分量。
2. 对多光谱图像矩阵执行 **NMF** 分解，得到两个低维非负矩阵：基矩阵和系数矩阵；对全色图像执行类似的分解操作。
3. 在非负矩阵分解过程中，通过加入耦合约束条件，使得全色图像和多光谱图像在低维特征空间上的表示尽可能一致。
4. 使用耦合后的低维特征矩阵和系数矩阵，重构出高分辨率的多光谱图像，从而实现图像的全色锐化。

相关程序可参考[CNMF.py](https://github.com/codegaj/py_pansharpening/blob/master/methods/CNMF.py)。

### ⚪ GFPCA
- W. Liao et al., "Two-stage fusion of thermal hyperspectral and visible RGB image by PCA and guided filter," 2015 7th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), Tokyo, 2015, pp. 1-4.

**GFPCA**（**Guided Filter Principal Component Analysis**，导向滤波器主成分分析）结合了主成分分析（**PCA**）和导向滤波（**Guided Filtering**），通过利用导向滤波器的边缘保持特性来优化 **PCA** 的结果，从而在融合过程中实现更精细的图像质量。
1. 对多光谱图像进行主成分分析（**PCA**），提取出主要的成分，并将其转换为不同的主成分空间。
2. 使用全色图像作为导向图，对**PCA**所得的第一个主成分执行导向滤波。导向滤波有助于保留边缘，同时去除噪声，从而在融合过程中保持空间细节。
3. 用经过导向滤波处理的主成分替代原始**PCA**主成分，这样做可以将全色图像的细节信息更好地渗透到多光谱图像中。
4. 通过逆**PCA**变换，将调整后的主成分转换回原始的多光谱波段，从而形成具有高空间分辨率的多光谱图像。

```python
from sklearn.decomposition import PCA as princomp
from cv2.ximgproc import guidedFilter

def GFPCA(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))

    p = princomp(n_components=C)
    pca_hs = p.fit_transform(np.reshape(hs, (m*n, C)))
    
    pca_hs = np.reshape(pca_hs, (m, n, C))
    
    pca_hs = upsample_interp23(pca_hs, ratio)
    
    gp_hs = []
    for i in range(C):
        temp = guidedFilter(np.float32(pan), np.float32(np.expand_dims(pca_hs[:, :, i], -1)), 8, eps = 0.001**2)
        temp = np.expand_dims(temp ,axis=-1)
        gp_hs.append(temp)
        
    gp_hs = np.concatenate(gp_hs, axis=-1)
    
    I_GFPCA = p.inverse_transform(gp_hs)
    
    #adjustment
    I_GFPCA[I_GFPCA<0]=0
    I_GFPCA[I_GFPCA>1]=1
    
    return np.uint8(I_GFPCA*255)
```

## 2. 多分辨率分析法 Multi-Resolution Analysis

多分辨率分析法(**MRA-based**)首先利用多尺度变换方法，如小波变换或者金字塔变换等，将源图像分解获得高、低频成份，再运用适当的融合规则对不同尺度的高、低频成份进行融合，最后将融合后的高、低频成份反变换获得融合图像，其不足在于空间细节失真较为严重。


### ⚪ SFIM变换
- J. Liu, “Smoothing filter based intensity modulation: a spectral preserve image fusion technique for improving spatial details,” International Journal of Remote Sensing, vol. 21, no. 18, pp. 3461–3472, December 2000.

**SFIM（Smoothing Filter-based Intensity Modulation）**基于平滑滤波和强度调制来融合多光谱图像和全色图像。
1. 对高空间分辨率的全色图像进行平滑滤波，以提取其低频分量，通常使用均值滤波器或高斯滤波器；
2. 使用全色图像的高频分量对多光谱图像进行调制，从而增强多光谱图像的空间细节；
3. 对每个波段的调制结果可以进行幅值调整，以确保最终结果得到增强的视觉效果，同时减轻可能的过度增强。

$$
     MS_{\text{sharp}}(i,j) = \frac{PAN(i,j)}{PAN_{smooth}(i,j)} \times MS(i,j)
$$

```python
from scipy import signal

def SFIM(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    u_hs = upsample_interp23(hs, ratio)
    
    if np.mod(ratio, 2)==0:
        ratio = ratio + 1
        
    pan = np.tile(pan, (1, 1, C))
    
    pan = (pan - np.mean(pan, axis=(0, 1)))*(np.std(u_hs, axis=(0, 1), ddof=1)/np.std(pan, axis=(0, 1), ddof=1))+np.mean(u_hs, axis=(0, 1))
    
    kernel = np.ones((ratio, ratio))
    kernel = kernel/np.sum(kernel)
    
    I_SFIM = np.zeros((M, N, C))
    for i in range(C):
        lrpan = signal.convolve2d(pan[:, :, i], kernel, mode='same', boundary = 'wrap')
        I_SFIM[:, :, i] = u_hs[:, :, i]*pan[:, :, i]/(lrpan+1e-8)

    #adjustment
    I_SFIM[I_SFIM<0]=0
    I_SFIM[I_SFIM>1]=1    
    
    return np.uint8(I_SFIM*255)
```

### ⚪ Wavelet变换
- King R L, Wang J. A wavelet based algorithm for pan sharpening Landsat 7 imagery [C]//IGARSS 2001. Scanning the Present and Resolving the Future. Proceedings.  IEEE 2001 International Geoscience and Remote Sensing Symposium (Cat. No. 01CH37217). IEEE, 2001, 2: 849-851.

小波变换（**Wavelet Transform**）通过分解图像为不同频率的子带，以融合多光谱图像和全色图像，实现高空间分辨率和多光谱信息的结合。
1. 对多光谱图像和全色图像分别进行小波分解。这一过程将图像分解为不同频带的子图像（例如，低频子带和高频子带）。
2. 将全色图像中的高频子带与多光谱图像的低频子带相结合，以实现增强的空间分辨率，同时尽量保留多光谱信息；
3. 将融合后的子带进行逆小波变换，重建高分辨率的多光谱图像。

```python
import pywt

def Wavelet(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
    u_hs = upsample_interp23(hs, ratio)
    
    pan = np.squeeze(pan)
    pc = pywt.wavedec2(pan, 'haar', level=2)
    
    rec=[]
    for i in range(C):
        temp_dec = pywt.wavedec2(u_hs[:, :, i], 'haar', level=2)
        
        pc[0] = temp_dec[0]
        
        temp_rec = pywt.waverec2(pc, 'haar')
        temp_rec = np.expand_dims(temp_rec, -1)
        rec.append(temp_rec)
        
    I_Wavelet = np.concatenate(rec, axis=-1)
    
    #adjustment
    I_Wavelet[I_Wavelet<0]=0
    I_Wavelet[I_Wavelet>1]=1
    
    return np.uint8(I_Wavelet*255)
```

### ⚪ MTF-GLP
- B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October 2002.

**MTF-GLP（Modulation Transfer Function Generalized Laplacian Pyramid）**是一种结合调制传递函数（**MTF**）和广义拉普拉斯金字塔（**GLP**）的全色锐化方法。该方法通过模拟传感器的调制传递函数特性来对图像进行滤波，并采用金字塔分解技术来增强多光谱图像的空间分辨率。
1. 使用调制传递函数来模拟传感器的光学和电子特性，生成一个低分辨的多光谱图像，该图像旨在匹配实际传感器对图像空间细节的响应。
2. 对低分辨多光谱图像和全色图像进行拉普拉斯金字塔分解，以分离高频和低频信息。拉普拉斯金字塔是一种多分辨率结构，通过逐层减去图像的高斯模糊版本形成。
3. 将从全色图像中获取的高频细节与多光谱图像的低频部分相结合。这一过程增强了多光谱图像的空间分辨率。
4. 通过拉普拉斯金字塔的逆变换过程，重建出具有高空间分辨率的多光谱图像。

相关程序可参考[MTF_GLP.py](https://github.com/codegaj/py_pansharpening/blob/master/methods/MTF_GLP.py)。

### ⚪ MTF-GLP-HPM
- B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,” Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.

**MTF-GLP-HPM（Modulation Transfer Function Generalized Laplacian Pyramid with High-pass Modulation）**是**MTF-GLP**方法的扩展版本，通过结合调制传递函数和高通调制技术，实现对图像空间细节的精细增强，同时保持多光谱图像的光谱特性。
1. 使用调制传递函数来模拟传感器的特性，生成一个低分辨率的多光谱图像，该过程旨在模拟传感器对图像空间细节的响应，通常使用高斯滤波模拟。
2. 对低分辨多光谱图像和全色图像进行拉普拉斯金字塔分解，以分离高频和低频信息，获取各层次的图像细节。
3. 通过将来自全色图像的高频信息与多光谱图像的低频成分相结合，增强多光谱图像空间细节。高通调制进一步细化了**MTF-GLP**的处理，通过添加一个乘法因子来强调来自全色图像的边缘与细节信息。
4. 通过对经过增强的图像在金字塔结构上进行逆变换，重建出具有高空间分辨率以及保留原始光谱信息的多光谱图像。

相关程序可参考[MTF_GLP_HPM.py](https://github.com/codegaj/py_pansharpening/blob/master/methods/MTF_GLP_HPM.py)。


## 3. 模型优化法 Model Optimization

模型优化法(**MO-based**)根据理想的融合图像$X$与全色图像$P$、多光谱图像$M$之间的关系建立能量函数，并通过最优化求解获得高分辨率多光谱融合图像，但其计算较为复杂。

### ⚪ SIRF
- [SIRF: Simultaneous Satellite Image Registration and Fusion in a Unified Framework. (IEEE TIP 2015)](https://ieeexplore.ieee.org/document/7156141)

**SIRF**把高分辨率多光谱图像（目标图像）的优化问题建模为最小化最小二乘拟合项和动态梯度稀疏正则器的线性组合。前者用于保留多光谱图像的精确光谱信息，而后者用于保留高分辨率全色图像的清晰边缘。模型的优化函数包含两个主要方面：

1. 光谱保持：下采样的高分辨率多光谱图像应接近原始多光谱图像，以保持准确的光谱信息：

$$
f_1(X,M) = \frac{1}{2} \left\| \text{DownSample}(X) - M \right\|_F^2
$$

2. 动态梯度稀疏：将不同波段上具有相同空间位置的像素分配到一个组中，它们的梯度（对应于陆地物体的边缘）往往位于相同的空间位置，因此通过$l_{2,1}$范数促进组内稀疏性：

$$
f_2(X,P) = \sum_{i,j}\sqrt{\sum_d\sum_{q=1,2}(\nabla_q X_{i,j,d} - \nabla_q P_{i,j})^2}
$$

### ⚪ PSFG$\text{S}^2$LR
- [A Variational Pan-Sharpening Method Based on Spatial Fractional-Order Geometry and Spectral–Spatial Low-Rank Priors. (IEEE TGRS 2018)](https://ieeexplore.ieee.org/document/8167324)

**PSFG**$\text{S}^2$**LR**结合了空间分数阶几何和光谱空间低秩先验，充分利用空间分数阶几何先验的空间细节和纹理表达能力以及低秩先验的光谱空间相关性保持能力。模型的优化函数包含三个主要方面：

1. 数据生成保真度项：模拟多光谱图像和高分辨率多光谱图像（目标图像）之间的退化关系以强制执行几何和光谱保持约束 （$D$是模拟模糊和下采样矩阵）：


$$
f_1(X,M) = \frac{1}{2} \left\| DX - M \right\|_2^2
$$

2. 基于分数阶全变分的空间分数阶几何先验项：利用全色图像和目标图像之间的空间分数阶梯度特征一致性，将全色图像的空间结构信息转移到目标图像中：

$$
f_2(X,P) = \sum \limits _{i = 1}^{N} {{{\| {\nabla ^\alpha {{\mathbf{X}}_{i}} - {\nabla ^\alpha }{\mathbf{P}}} \|}_{1,2}}}
$$

3. 基于加权核范数的光谱空间低秩先验项：利用目标图像和多光谱图像中基于非局部块的低秩结构稀疏性，进一步保留图像的空间结构和光谱信息：

$$
f_3(X,M) = \sum \limits _{i = 1}^{N} {\sum \limits _{j = 1}^{B} {{{\| {{{\tilde {\mathbf{R}}}_{j}}({{\mathbf{X}}_{i}} - \text{UpSample}(M)_i)} \|}_{{{\omega }}, * }}} }
$$


### ⚪ LGC
- [A Variational Pan-Sharpening With Local Gradient Constraints. (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_A_Variational_Pan-Sharpening_With_Local_Gradient_Constraints_CVPR_2019_paper.pdf)

**LGC (Local Gradient Constraints)**是一种基于局部梯度约束的变分模型，通过考虑全色图像和高分辨率多光谱图像（目标图像）在不同局部区域和波段中的梯度差异，实现了更准确的空间信息保持。模型的优化函数包含两个主要方面：

1. 光谱保持：下采样的高分辨率多光谱图像应接近原始多光谱图像，以保持准确的光谱信息：

$$
f_1(X,M) = \frac{1}{2} \left\| \text{DownSample}(X) - M \right\|_2^2
$$

2. 空间保持：引入局部梯度约束，高分辨率多光谱图像不同波段和局部区域中的梯度应呈线性关系$\nabla x \approx a\nabla p + b$：

$$
f_2(X,A,C,P) = \sum_{b=1}^B  \sum_k \sum_{i \in \omega_k}(\nabla x_{b,i} - a_{b,k}\nabla p_i -c_{b,k})^2
$$

### ⚪ PGCP-PS
- [PAN-Guided Cross-Resolution Projection for Local Adaptive Sparse Representation- Based Pansharpening. (IEEE TGRS 2019)](https://ieeexplore.ieee.org/document/8643394)

**PGCP-PS**的基本思想是从模拟的全色图像超分辨率场景中估计用于锐化多光谱图像的跨分辨率投影和偏移量，并注入高分辨率多光谱图像（目标图像）超分辨率重建过程。把目标图像$X$建模为：

$$
X=\text{UpSample}(M) +\widehat {\boldsymbol {x}}_{k,i}^{\mathrm {HM}}+\boldsymbol {O}_{k,i}^{\mathrm {HM}}
$$

其中$\widehat {\boldsymbol {x}}_{k,i}^{\mathrm {HM}}$是$X$的高频分量，可表示为字典基$D_h$和稀疏系数$\alpha_{k,i}$的组合$\widehat {\boldsymbol {x}}_{k,i}^{\mathrm {HM}}=D_h\alpha_{k,i}$；$\boldsymbol {O}_{k,i}^{\mathrm {HM}}$是多光谱图像超分辨率的偏移量。$D_h,\alpha_{k,i},\boldsymbol {O}_{k,i}^{\mathrm {HM}}$通过构造全色图像的超分辨率模型来近似。


### ⚪ BPSM
- [Bayesian Pan-Sharpening With Multiorder Gradient-Based Deep Network Constraints. (IEEE JSTARS 2020)](https://ieeexplore.ieee.org/document/9020056)

**BPSM (Bayesian pan-sharpening model)**是一种基于贝叶斯理论的全色锐化模型，该模型涉及三个假设：1）多光谱图像是通过模糊核卷积从高分辨率多光谱图像（目标图像）中抽取的；2）使用多尺度递归块组成的卷积神经网络保留全色图像的空间信息；3）在多阶梯度域中引入各向异性的全变分先验以重建更好的图像边缘和细节。模型的后验概率建模为：

$$
\begin{align*} p(X|M,f)=&N(M|X,\sigma _{1}^{2})N(\nabla _{1}f|\nabla _{1}X,\sigma _{21}^{2})N(\nabla _{2}f|\nabla _{2}X,\sigma _{22}^{2})\\ & \times L(\nabla _{1}X|0,s_{1})L(\nabla _{2}X|0,s_{2}) \end{align*}
$$

其中$f=f_{MCNN}(M,P)$设计为**MCNN**卷积网络。


### ⚪ F-BMP
- [Fast and High-Quality Blind Multi-Spectral Image Pansharpening. (IEEE TGRS 2021)](https://ieeexplore.ieee.org/document/9491792)

**F-BMP**通过计算具有最小总广义变差的核系数来估计高分辨率多光谱图像（目标图像）下采样的模糊核，并使用局部拉普拉斯先验 (**LLP**) 估计目标图像的每个通道与全色图像之间的关系。模型的优化函数包含三个主要方面：

1. 数据保真度项：迫使经过模糊和下采样的高分辨率多光谱图像接近多光谱图像，其中$B(u)$是实现高分辨率多光谱图像与模糊核 $u$ 的卷积的 **Toeplitz** 矩阵：

$$
f_1(X,M) = \frac{1}{2} \left\| \text{DownSample}(B(u)X) - M \right\|_F^2
$$

2. 模糊核$u$的正则化项：采用**TGV2**作为正则化器，保留模糊核的高阶平滑度，同时拒绝远离峰值的非平凡系数：

$$
f_2(u) = \min _{\mathbf {p}} \left \{ \alpha_1\|\mathbf {\nabla } \mathbf {u}-\mathbf {p}\|_{2,1}+\alpha_2\|\mathcal {E}(\mathbf {p})\|_{2,1}\right \}+\mathbf {I}_{\mathbb {S}}(\mathbf {u})
$$

3. **HRMS** 图像和 **PAN** 图像之间的正则化项：根据受局部线性模型启发的变分观点，最小化来自另一个通道的每个高频分量块的最接近的线性仿射函数来近似目标通道的每个高频分量块的总体损失 ($\mathcal{L}$是拉普拉斯算子)：

$$
f_3(X, P) = \frac {\lambda }{2}\sum _{i,j}\sum _{k\in w_{j}}\bigg [\big ([{\mathcal{L}}(\mathbf {X}_{i})]_{j,k}- a_{i,j}[{\mathcal{L}}(\mathbf {P})]_{j,k}-c_{i,j}\big )^{2}+ \epsilon a^{2}_{i,j}\bigg ]
$$

## 4. 基于深度学习的方法 Deep Learning

深度学习方法(**DL-based**)是指使用深度学习模型进行全色融合。早期**DL-based**全色锐化方法大多参考了[图像超分辨率](https://0809zheng.github.io/2020/08/27/SR.html)的概念，通过构建卷积神经网络（**CNN**）来自动学习图像特征，从而实现图像分辨率的提升。

### ⚪ [<font color=blue>PNN</font>](https://0809zheng.github.io/2024/10/09/pcnn.html)
- (Remote Sensing 2016) Pansharpening by Convolutional Neural Networks

**PNN**采用了较为简单的**CNN**架构以避免过拟合和训练困难。在特征融合阶段采用了逐元素相加的方式，将多光谱图像的特征与全色图像的特征进行结合，以生成具有丰富光谱信息和高空间分辨率的融合图像。

![](https://pic.imgdb.cn/item/673c70e2d29ded1a8cd5198e.png)

### ⚪ [<font color=blue>PanNet</font>](https://0809zheng.github.io/2024/10/10/pannet.html)
- (ICCV 2017) PanNet: A Deep Network Architecture for Pan-Sharpening

**PanNet**遵循**ResNet**框架，引入了光谱映射和高通输入的改进：
- 光谱映射是指将低分辨率的多光谱图像进行上采样，并通过跳跃连接将其添加到网络的目标函数中。这允许网络专注于学习图像中的细节信息，同时保留光谱内容。
- 高通输入是指网络输入的是全色图像和低分辨率多光谱图像的高通分量，这使得网络能够学习如何将全色图像中的空间信息映射到高分辨率多光谱图像中。

![](https://pic.imgdb.cn/item/673c743dd29ded1a8cd97a52.png)

### ⚪ [<font color=blue>MSDCNN</font>](https://0809zheng.github.io/2024/10/11/msdcnn.html)
- (arXiv1712) A Multi-Scale and Multi-Depth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening

**MSDCNN**架构结合了多尺度特征提取和深层网络结构，以更好地捕捉遥感影像中的空间细节和光谱信息。由两个主要部分组成：
- 特征提取网络：采用基本的卷积神经网络结构，用于从输入的全色图像和多光谱图像中提取初步特征。
- 多尺度特征提取网络：包含多个多尺度卷积层块，每个块由不同尺度的卷积核组成，用于捕捉不同尺度的空间细节。这些块通过串联和卷积操作融合不同尺度的特征，以生成更丰富的特征表示。

![](https://pic.imgdb.cn/item/673c78b7d29ded1a8cddb32e.png)


### ⚪ [<font color=blue>SRPPNN</font>](https://0809zheng.github.io/2024/10/12/srppnn.html)
- (IEEE TGRS 2021) Super-Resolution-Guided Progressive Pansharpening Based on a Deep Convolutional Neural Network

**SRPPNN**模型在超分辨率过程中引入了渐进式全色锐化和高通残差模块。
- 渐进式全色锐化：将整个全色锐化网络分解为一系列子网络，每个子网络负责执行特征提取和2倍上采样。这种方法有助于逐步改善图像的空间分辨率，同时考虑尺度效应。
- 高通残差模块：用于直接注入**PAN**图像中的空间细节，进一步增强融合结果的空间分辨率。

![](https://pic.imgdb.cn/item/673c7c07d29ded1a8ce084b7.png)

### ⚪ [<font color=blue>INNformer</font>](https://0809zheng.github.io/2024/10/13/innformer.html)
- (AAAI 2022) Pan-Sharpening with Customized Transformer and Invertible Neural Network

**INNformer**模型采用两流独立卷积编码器，在特征提取之后利用定制化**Transformer**捕捉长距离依赖关系，利用卷积网络捕捉局部特征；最后使用可逆神经网络融合模块将**MS**和**PAN**图像的特征进行融合，生成高分辨率的多光谱图像。

![](https://pic.imgdb.cn/item/673c8023d29ded1a8ce43a8b.png)

### ⚪ [<font color=blue>PanFormer</font>](https://0809zheng.github.io/2024/10/14/panformer.html)
- (arXiv2203) PanFormer: a Transformer Based Model for Pan-sharpening

**PanFormer**是一个基于**Transformer**的全色锐化模型。它首先通过自注意力机制的模态特定编码器提取全色图像和多光谱图像的模态特定特征，然后使用交叉注意力机制的跨模态融合模块来合并光谱和空间特征，最后使用卷积网络的图像恢复模块生成全色锐化图像。

![](https://pic.imgdb.cn/item/673da72cd29ded1a8ccccb10.png)


### ⚪ [<font color=blue>SFIIN</font>](https://0809zheng.github.io/2024/10/15/sfiin.html)
- (ECCV 2022) Spatial-Frequency Domain Information Integration for Pan-Sharpening

**SFIIN**通过结合空间域和频率域的信息进一步提升全色锐化的性能。其核心模块是**SFIB**，包含三个关键组件：空间域信息分支、频率域信息分支和双域信息交互。
- 空间域信息分支：使用卷积层提取**PAN**和**MS**图像在空间域的局部信息。
- 频率域信息分支：在傅里叶空间使用卷积层提取频率域的全局信息。
- 双域信息交互：通过空间注意力和通道注意力学习空间域和频率域信息的互补表示。

![](https://pic.imgdb.cn/item/673dad30d29ded1a8cd3430e.png)


### ⚪ [<font color=blue>MIDPS</font>](https://0809zheng.github.io/2024/10/16/mutnet.html)
- (CVPR 2022) Mutual Information-driven Pan-sharpening

**MIDPS**是一种基于互信息最小化的全色锐化框架，显式地鼓励**MS**和**PAN**之间的互补信息学习，减少信息冗余。模型架构包含三个模块：模态感知特征提取、互信息约束和后融合模块。
- 模态感知特征提取：使用两个独立的卷积层特征提取分支，将**PAN**和**MS**图像投影到模态感知特征图；
- 互信息约束：将**PAN**特征和**MS**特征转换为低维特征向量，并引入互信息最小化来显式鼓励两种模态之间的互补信息学习；
- 后融合模块：基于可逆神经网络的后融合模块将经过互信息最小化处理的特征向量投影到最终的融合图像。

![](https://pic.imgdb.cn/item/673dbfa7d29ded1a8cefc924.png)

