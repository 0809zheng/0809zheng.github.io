---
layout: post
title: '初始损失的健全性检查(Sanity Check)'
date: 2026-02-15
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/math-sanity-check-cover.png'
tags: 数学
---

> Sanity Check for the Initial Loss.

深度学习模型在训练开始时的权重是随机初始化的。如果输出层的初始化尺度足够小，那么模型在送入最后一个激活函数（如 **Softmax** 或 **Sigmoid**）之前输出的 **logits** 近似全为 0，模型对所有类别的预测是**均等**的，相当于“随机猜测”。

在这种均匀预测下，各种常见损失函数的期望值都可以直接算出来。把第一个训练步的损失与理论值比较，就能在大规模训练之前发现初始化、数据加载、损失实现或归约方式（**reduction**）上的错误，这就是初始损失的**健全性检查(Sanity Check)**。

本文先给出这类计算的一般原理（以及 **logits** 方差不可忽略时的修正），再逐一推导分类、回归、分割、对比学习与生成模型中常用损失的理论初始值，最后介绍按标签先验初始化输出偏置的做法。

#### TL;DR:

| 任务类型 | 损失函数 | 输出层 | 理论初始损失 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **二分类** | **BCE** | **Sigmoid** | $\log 2 \approx 0.693$ | 初始预测概率为 0.5 |
| **二分类** | **Focal Loss** | **Sigmoid** | $\alpha_t\, 2^{-\gamma}\log 2$ | $\gamma$ = 调制因子，$\alpha_t$ = 类别权重 |
| **多分类** | **Cross-Entropy** | **Softmax** | $\log C$ | $C$ = 类别数；标签平滑不改变该值 |
| **知识蒸馏** | **KL** 散度 | **Softmax** | $\log C - H(p_{\text{teacher}})$ | 教师分布的熵越大，初始值越小 |
| **回归** | **MSE** | 线性 | $\text{Var}(\hat{y})+\text{Var}(y)+(\mathbb{E}\hat{y}-\mathbb{E}y)^2$ | $\hat{y}\approx 0$ 时约为 $\mathbb{E}[y^2]$ |
| **回归** | **MAE** | 线性 | $\mathbb{E}\vert y\vert$ | 标准化高斯目标约为 $0.798$ |
| **分割** | **Dice Loss** | **Sigmoid** | $\frac{1}{1+2f}$ | $f$ = 前景像素比例 |
| **对比学习** | **InfoNCE** | **L2** 归一化 | $\log(N+1)+\frac{1}{2d\tau^2}$ | $N$ = 负样本数，$d$ = 特征维度，$\tau$ = 温度 |
| **生成模型** | **VAE (ELBO)** | — | $D\log 2$（**Bernoulli** 重建）$+\,0$（**KL**） | $D$ = 像素数 |

# 1. 一般原理：均匀预测与凸性下界

## 1.1 均匀预测下的损失

几乎所有分类类损失都可以写成 **logits** 向量 $z$ 的函数 $\mathcal{L}(z)$。当输出层初始化尺度很小时 $z\approx \mathbf{0}$，此时 **Softmax** 给出均匀分布、**Sigmoid** 给出 $0.5$，损失值 $\mathcal{L}(\mathbf{0})$ 只与类别数、负样本数等结构参数有关，与数据无关。

## 1.2 Jensen 下界

常用的 **BCE**、交叉熵和 **InfoNCE** 都是 **logits** 的**凸函数**。以交叉熵为例：

$$
\mathcal{L}(z) = -\log\frac{e^{z_y}}{\sum_{j=1}^{C}e^{z_j}} = \log\sum_{j=1}^{C}e^{z_j} - z_y
$$

其中 $\log\sum e^{z_j}$（**log-sum-exp**）是凸函数，$-z_y$ 是线性函数，两者之和仍是凸函数。若随机初始化下每个 **logit** 的期望为 0（不要求相互独立），由 **Jensen** 不等式：

$$
\boxed{\mathbb{E}\left[\mathcal{L}(z)\right] \ge \mathcal{L}\left(\mathbb{E}[z]\right) = \mathcal{L}(\mathbf{0})}
$$

因此均匀预测给出的理论值是零均值随机初始化下初始损失期望的**下界**：**logits** 的方差越大，实际初始损失越高。

## 1.3 logits 方差不可忽略时的修正

仍以交叉熵为例：设 $z_1,\dots,z_C$ 独立同分布于 $N(0,s^2)$，则

$$
\begin{aligned}
\mathbb{E}\left[\mathcal{L}\right] &=\mathbb{E}\left[\log\sum_{j=1}^{C}e^{z_j} - z_y\right] \\
&= \log C + \mathbb{E}\left[\log\left(\frac{1}{C}\sum_{j=1}^{C}e^{z_j}\right)\right] - \mathbb{E}[z_y],\quad C>2 \\
&= \log 2 + \mathbb{E}\left[\log\left(\frac{1+e^z}{2}\right)\right] - \mathbb{E}[yz],\quad C=2, \text{取}z_1=0,z_2=z,y=\{0,1\} \\
\end{aligned}
$$

当 $C$ 较大时，由大数定律 $\frac{1}{C}\sum_j e^{z_j}\to \mathbb{E}[e^{z}]=e^{s^2/2}$（正态分布的矩母函数），且 $\mathbb{E}[z_y]=0$，于是

$$
\boxed{\mathbb{E}\left[\mathcal{L}_{\text{CE}}\right] \approx \log C + \frac{s^2}{2}}
$$

对于二分类 **BCE**，把 $\log(\frac{1+e^z}{2})$ 在 $z=0$ 处展开：$\log(\frac{1+e^z}{2}) \approx \frac{1}{2}\log 2-\frac{z}{4}+\frac{z^2}{8}$，因此 $s$ 较小时

$$
\boxed{\mathbb{E}\left[\mathcal{L}_{\text{BCE}}\right] \approx \log 2 + \frac{s^2}{8}}
$$

蒙特卡洛模拟的结果如下（每组数万次采样）：

| 设置 | 模拟值 | 均匀预测值 | 近似公式 |
| :--- | :--- | :--- | :--- |
| 交叉熵，$C=10$，$s=1$ | 2.719 | 2.303 | 2.803 |
| 交叉熵，$C=1000$，$s=1$ | 7.404 | 6.908 | 7.408 |
| 交叉熵，$C=1000$，$s=0.3$ | 6.946 | 6.908 | 6.953 |
| **BCE**，$s=0.5$ | 0.725 | 0.693 | 0.724 |

可以看到 $C$ 越大近似越准；当 $s=1$ 时初始损失比 $\log C$ 高出约 0.5。若希望初始损失贴近 $\log C$，可以把输出层权重的初始化尺度取小。

# 2. 二分类问题 (Binary Classification)

## 2.1 二进制交叉熵

二分类问题的损失函数是二进制交叉熵损失 (**Binary Cross-Entropy Loss, BCELoss**)，网络最后一层激活函数通常是**Sigmoid**。模型输出一个 **logit** $z$，**Sigmoid** 函数将其转换为正类概率：

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

当 $z \approx 0$ 时 $e^{-z}\approx 1$，初始预测概率为 $p\approx \frac{1}{2}$。对于真实标签为 $y\in\{0,1\}$ 的样本，**BCELoss** 的定义是：

$$
\mathcal{L} = -\left[y \log p + (1-y) \log(1-p)\right]
$$

无论 $y=1$（$\mathcal{L}=-\log p$）还是 $y=0$（$\mathcal{L}=-\log(1-p)$），代入 $p=\frac{1}{2}$ 都得到相同的值，因此该结论与正负样本比例无关：

$$
\boxed{\mathcal{L}_{\text{init}} = \log 2 \approx 0.693}
$$

## 2.2 多标签分类

多标签分类对 $K$ 个标签各使用一个独立的 **Sigmoid** 输出和 **BCE** 损失。每个标签的初始损失都是 $\log 2$，因此按标签取平均（**reduction='mean'**）时初始损失为 $\log 2$，按标签求和时为 $K\log 2$。

## 2.3 Focal Loss

**Focal Loss** 在 **BCE** 的基础上引入调制因子 $(1-p_t)^\gamma$ 来降低易分样本的权重，其中 $p_t$ 是模型对真实类别的预测概率，$\alpha_t$ 是类别权重（正样本取 $\alpha$，负样本取 $1-\alpha$）：

$$
\mathcal{L}_{\text{FL}} = -\alpha_t (1-p_t)^\gamma \log p_t
$$

在 $p_t=\frac{1}{2}$ 时：

$$
\boxed{\mathcal{L}_{\text{FL,init}} = \alpha_t \, 2^{-\gamma}\log 2}
$$

取常用的 $\gamma=2$、$\alpha=0.25$，正样本的初始损失为 $0.25\times\frac{1}{4}\times 0.693\approx 0.043$，负样本为 $0.75\times\frac{1}{4}\times 0.693\approx 0.130$。

# 3. 多分类问题 (Multi-Class Classification)

## 3.1 交叉熵

多分类问题的损失函数是交叉熵损失 (**Cross-Entropy Loss**)，网络最后一层激活函数通常是**Softmax**。模型输出 **logits** $z = [z_1, z_2, \dots, z_C]$，其中 $C$ 是类别总数。**Softmax** 函数将 **logits** 转换为概率分布：

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

当所有 $z_i \approx 0$ 时 $e^{z_i}\approx 1$，每个类别的预测概率为 $p_i \approx \frac{1}{C}$。对于真实类别为 $y$ 的样本，交叉熵损失只保留真实类别对应的一项：

$$
\mathcal{L} = -\sum_{i=1}^{C} y_i \log p_i = -\log p_y
$$

代入 $p_y = \frac{1}{C}$ 得到

$$
\boxed{\mathcal{L}_{\text{init}} = -\log\frac{1}{C} = \log C}
$$

例如图像分类中 **CIFAR-10**（10 类）的初始损失应在 $\log 10 \approx 2.303$ 附近，**ImageNet**（1000 类）应在 $\log 1000 \approx 6.908$ 附近（忽略**logits** 方差的影响）。同样的结论适用于逐元素计算交叉熵的任务：语义分割的逐像素交叉熵初始值为 $\log C$；语言模型的下一个 **token** 预测是在整个词表上做分类，词表大小为 $V$ 时初始损失为 $\log V$，例如 **GPT-2** 的词表 $V=50257$ 对应 $\log 50257 \approx 10.82$。

## 3.2 标签平滑

**标签平滑（Label Smoothing）**把 **one-hot** 目标替换为 $q_i = (1-\varepsilon)\,\mathbb{1}[i=y] + \frac{\varepsilon}{C}$。由于 $\sum_i q_i = 1$，在均匀预测下

$$
\mathcal{L} = -\sum_{i=1}^{C} q_i \log\frac{1}{C} = \log C \sum_{i=1}^{C} q_i = \log C
$$

即标签平滑不改变初始损失的理论值。

## 3.3 知识蒸馏中的 KL 散度

知识蒸馏用 **KL** 散度让学生分布 $p$ 逼近教师分布 $p_{\text{t}}$。学生在初始时给出均匀分布 $p_i=\frac{1}{C}$，于是

$$
\boxed{\text{KL}\left(p_{\text{t}}\,\Vert\,p\right) = \sum_{i=1}^{C} p_{\text{t},i}\log\frac{p_{\text{t},i}}{1/C} = \log C - H(p_{\text{t}})}
$$

其中 $H(p_{\text{t}})=-\sum_i p_{\text{t},i}\log p_{\text{t},i}$ 是教师分布的熵。教师输出越“软”（如使用较高的蒸馏温度），熵越大，**KL** 的初始值越接近 0；教师输出接近 **one-hot** 时，初始值接近 $\log C$。

# 4. 回归问题 (Regression)

## 4.1 均方误差

回归问题最常用的损失函数是均方误差损失 (**Mean Squared Error, MSELoss**)，最后一层通常是线性层（无激活）：

$$
\mathcal{L} = (\hat{y} - y)^2
$$

将初始预测 $\hat{y}$ 与真实值 $y$ 都视为随机变量。随机初始化的模型尚未学到输入与目标之间的关系，可以认为 $\hat{y}$ 与 $y$ 不相关，即 $\mathbb{E}[\hat{y}y]=\mathbb{E}[\hat{y}]\,\mathbb{E}[y]$。利用 $\mathbb{E}[W^2]=\text{Var}(W)+(\mathbb{E}W)^2$：

$$
\boxed{\mathbb{E}\left[\mathcal{L}_{\text{init}}\right] = \text{Var}(\hat{y}) + \text{Var}(y) + \left(\mathbb{E}[\hat{y}] - \mathbb{E}[y]\right)^2}
$$

其中 $\text{Var}(\hat{y}-y)=\text{Var}(\hat{y})+\text{Var}(y)$ 用到了两者不相关。三项分别对应：模型初始输出的方差、目标值的方差、初始预测均值与目标均值的偏差。若输出层初始化使 $\hat{y}\approx 0$，则 $\text{Var}(\hat{y})\approx 0$、$\mathbb{E}[\hat{y}]\approx 0$，初始损失约为 $\text{Var}(y)+(\mathbb{E}[y])^2=\mathbb{E}[y^2]$，即目标值的二阶矩；对目标做了标准化（均值 0、方差 1）时约为 $1$。

需要注意，当目标值未标准化且均值远离 0 时，偏差项 $(\mathbb{E}[y])^2$ 会主导初始损失。例如目标均值为 3、标准差为 2，而初始输出近似为 0 时，初始 **MSE** 约为 $4+9=13$，其中 9 来自均值偏差。

## 4.2 平均绝对误差

平均绝对误差 (**Mean Absolute Error, MAE / L1 Loss**) 定义为 $\mathcal{L}=\vert\hat{y}-y\vert$。当 $\hat{y}\approx 0$ 时初始损失约为 $\mathbb{E}\vert y\vert$。若目标服从标准正态分布 $N(0,1)$：

$$
\mathbb{E}\vert y\vert = 2\int_0^{\infty} y\,\frac{1}{\sqrt{2\pi}}e^{-y^2/2}\,dy = \frac{2}{\sqrt{2\pi}}\left[-e^{-y^2/2}\right]_0^{\infty} = \sqrt{\frac{2}{\pi}} \approx 0.798
$$

# 5. 图像分割 (Image Segmentation)

除逐像素交叉熵（初始值为 $\log C$，见 3.1 节）之外，分割任务常用 **Dice Loss** 处理前景与背景的不平衡。设 $p_k\in[0,1]$ 是像素 $k$ 的前景预测概率，$y_k\in\{0,1\}$ 是标签，共 $n$ 个像素：

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_k p_k y_k}{\sum_k p_k + \sum_k y_k}
$$

初始时 **Sigmoid** 输出 $p_k=\frac{1}{2}$。记前景像素比例为 $f=\frac{1}{n}\sum_k y_k$，则 $\sum_k p_k y_k = \frac{nf}{2}$、$\sum_k p_k = \frac{n}{2}$、$\sum_k y_k = nf$，代入得

$$
\boxed{\mathcal{L}_{\text{Dice,init}} = 1 - \frac{nf}{\frac{n}{2} + nf} = \frac{1}{1+2f}}
$$

**Dice Loss** 的初始值依赖于前景比例：$f=0.5$ 时为 $0.5$，$f=0.1$ 时为 $0.833$，$f=0.01$ 时为 $0.980$。实现中常在分子分母加入平滑项以避免除零，当平滑项远小于像素数时对上述结果影响很小。

# 6. 对比学习与度量学习 (Contrastive and Metric Learning)

## 6.1 InfoNCE

在自监督对比学习 (**SimCLR**, **MoCo**) 中常用的损失函数是 **InfoNCE**。模型为查询 $q$、正样本 $k_+$ 以及 $N$ 个负样本 $\{k_i\}_{i=1}^{N}$ 输出经过 **L2** 归一化的 $d$ 维特征向量，相似度取余弦相似度 $\text{sim}(u,v)=u^\top v$，$\tau$ 为温度系数：

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\exp(\text{sim}(q, k_+) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(q, k_i) / \tau)}
$$

若所有相似度都为 0，则每一项指数都等于 1，得到均匀预测下的值：

$$
\mathcal{L}(\mathbf{0}) = -\log \frac{1}{1 + N} = \log(N+1)
$$

然而温度系数会放大相似度的随机波动。设 $u,v$ 是 $d$ 维单位球面上独立的均匀随机向量。固定 $v=e_1$（由旋转对称性不失一般性），则 $u^\top v=u_1$；由于 $\sum_{i=1}^{d}u_i^2=1$ 且各分量同分布，有

$$
\mathbb{E}[u^\top v]=0,\qquad \text{Var}(u^\top v)=\mathbb{E}[u_1^2]=\frac{1}{d}
$$

因此每个 **logit** $\text{sim}/\tau$ 的方差为 $s^2=\frac{1}{d\tau^2}$。按照 1.3 节的论证（负样本数较多时），

$$
\boxed{\mathbb{E}\left[\mathcal{L}_{\text{InfoNCE,init}}\right] \approx \log(N+1) + \frac{1}{2d\tau^2}}
$$

修正项在小温度下并不小。蒙特卡洛模拟的结果如下：

| $d$ | $\tau$ | $N$ | 模拟值 | $\log(N+1)$ | 修正公式 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 128 | 0.5 | 63 | 4.167 | 4.159 | 4.175 |
| 128 | 0.1 | 63 | 4.552 | 4.159 | 4.550 |
| 128 | 0.07 | 4095 | 9.055 | 8.318 | 9.115 |

上述推导假设正样本与负样本在初始特征空间中不可区分。实际上同一图像的两个增强视图经过随机网络后通常仍比不同图像更相似，这会使初始损失低于公式值；随机初始化的网络也可能把所有输入映射到相近的特征上，使各相似度接近一个共同的值，此时损失更接近 $\log(N+1)$。

## 6.2 常见对比学习框架中的负样本数

不同框架中“负样本数” $N$ 的含义不同，代入 6.1 节时需要区分：
*   **SimCLR**（**NT-Xent** 损失）：批大小为 $B$ 时共有 $2B$ 个增强视图，每个视图以另一个视图为正样本，其余 $2B-2$ 个视图为负样本，因此均匀预测值为 $\log(2B-1)$。
*   **MoCo**：负样本来自长度为 $K$ 的队列，均匀预测值为 $\log(K+1)$。
*   **CLIP**：批大小为 $B$ 时，每张图像在 $B$ 段文本中识别配对文本（反之亦然），图像到文本与文本到图像两个方向的交叉熵取平均，均匀预测值为 $\log B$。

## 6.3 Triplet Loss

度量学习中的三元组损失（**Triplet Loss**）为 $\mathcal{L}=\max\left(0,\ d(a,p)-d(a,n)+m\right)$，其中 $a,p,n$ 分别是锚点、正样本和负样本，$m$ 是间隔。初始时正负样本到锚点的距离分布相同，$\mathbb{E}[d(a,p)-d(a,n)]=0$；由于 $\max(0,\cdot+m)$ 是凸函数，由 **Jensen** 不等式初始损失的期望不小于 $m$，距离差的波动越小越接近 $m$。

# 7. 生成模型：变分自编码器 (VAE)

**VAE** 最小化负 **ELBO**，由重建项和 **KL** 项组成：

$$
\mathcal{L} = -\mathbb{E}_{q(z|x)}\left[\log p(x|z)\right] + \text{KL}\left(q(z|x)\,\Vert\, N(0,I)\right)
$$

编码器输出均值 $\mu$ 和对数方差 $\log\sigma^2$。初始时两者都接近 0，即 $q(z\vert x)\approx N(0,I)$，**KL** 项

$$
\text{KL} = \frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right) \approx 0
$$

重建项取决于解码器的似然形式。若对 $D$ 个像素使用 **Bernoulli** 似然（即逐像素 **BCE**），初始输出 $0.5$ 时每个像素贡献 $\log 2$，按像素求和得到

$$
\boxed{\mathcal{L}_{\text{VAE,init}} \approx D\log 2}
$$

例如 **MNIST**（$D=784$）约为 $543.4$。若使用固定方差的高斯似然，重建项退化为（按比例缩放的）**MSE**，按 4.1 节计算。

# 8. 按标签先验初始化输出偏置

前文都假设输出层偏置为 0。当类别不平衡时，更好的做法是把输出层偏置初始化为与标签先验一致，使初始预测直接等于各类别的先验频率。

对于二分类，设正样本比例为 $\pi$，令偏置 $b=\log\frac{\pi}{1-\pi}$，则 **sigmoid** $(b)=\pi$。初始损失的期望为

$$
\boxed{\mathbb{E}\left[\mathcal{L}_{\text{init}}\right] = -\pi\log\pi - (1-\pi)\log(1-\pi) = H(\pi)}
$$

即标签分布的熵。

对于多分类，令 $b_c=\log\pi_c$，**Softmax** 输出恰好为先验 $\pi_c$，初始损失为 $H(\pi)=-\sum_c\pi_c\log\pi_c\le\log C$，等号当且仅当先验均匀时成立。

对于回归，把输出偏置设为训练集目标均值 $\mathbb{E}[y]$，可以消去 4.1 节中的均值偏差项，初始 **MSE** 降为 $\text{Var}(y)+\text{Var}(\hat{y})$。

这种初始化可以避免模型在训练最初的若干步中只学习先验（损失曲线开头出现一段陡降），在极度不平衡时还能减轻训练初期不稳定。目标检测中的 **RetinaNet** 将分类分支的偏置初始化为 $b=-\log\frac{1-\pi}{\pi}$、$\pi=0.01$，使所有锚框的初始前景概率为 0.01，就是这一思想的应用。
