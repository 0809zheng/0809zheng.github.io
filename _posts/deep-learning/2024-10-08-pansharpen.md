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
1. 成分替换法(**CS-based**)：使用全色图像对多光谱图像的成分进行替换，如。
2. 多分辨率分析法(**MRA-based**)：
3. 模型优化法(**MO-based**)：
4. 深度学习方法(**DL-based**)：

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

模型优化法(**MO-based**)根据理想的融合图像与全色图像、多光谱图像之间的关系建立能量函数，并通过最优化求解获得高分辨率多光谱融合图像，但其计算较为复杂。

## 4. 基于深度学习的方法 Deep Learning

深度学习方法(**DL-based**)是指使用深度学习模型进行全色融合。


