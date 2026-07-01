---
layout: post
title: 'Modeling Uncertainty in Multi-Modal Fusion for Lung Cancer Survival Analysis'
date: 2026-04-17
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/69d7523915908d0c2fa8760b.png'
tags: 论文阅读
---

> 建模非癌生存分析多模态融合中的不确定性.

- paper：[Modeling Uncertainty in Multi-Modal Fusion for Lung Cancer Survival Analysis](https://ieeexplore.ieee.org/document/9433823)

# 0. TL; DR

本文扩展了联合标签融合（**joint label fusion**）方法，在估计不同模态预测之间的相关性时，明确地考虑了模型不确定性（**model uncertainty**）。

为了验证该方法，作者在对接受手术切除的**non-small cell lung cancer** (**NSCLC**) 患者进行生存预测的实验研究中，使用了**imaging**（影像）和**genetic**（基因）数据。实验结果显示，所提出的方法取得了有希望的性能，其在一年生存预测任务上的**AUC**达到了0.728，显著优于所有基线方法，这证明了在多模态融合中考虑模型不确定性的重要性。


# 1. 背景介绍

在医疗保健应用中，多模态数据变得越来越重要，例如在诊断、干预和预后分析中。由于不同的模态可能捕捉到互补的信息，多模态融合旨在通过整合多模态数据来提升分析性能。

多模态融合可以大致分为**early fusion**（早期融合）、**late fusion**（后期融合）和**intermediate fusion**（中间层融合）。
*   **Early fusion**：通过拼接来组合多模态数据，适用于那些已经对齐且处于相同语义特征空间的模态。
*   **Intermediate fusion**：主要用于处理异构的多模态数据（如影像和基因数据），它首先将多模态数据转换到一个**common semantic space**（共同语义空间），然后再进行基于拼接的融合。
*   **Late fusion**（也称**decision fusion**，决策融合）：为每个单模态独立训练分类器，然后在测试时通过组合这些单模态分类器的预测来得出共识预测。

然而，现有的**late fusion**方法存在一个局限性：它们没有考虑到在学习单模态模型时产生的**model uncertainty**（模型不确定性）。许多机器学习问题是欠定的（**under-determined**），即对于给定的学习模型和训练集，可能存在许多不同的模型参数都能很好地拟合训练数据。这些具有不同参数的训练模型在未见过的测试数据上可能会有不同的泛化表现，因此任何一个特定的训练模型都可能是不可靠的。如果每个单模态学习中的不确定性没有得到妥善处理，那么整体的融合性能可能会受到损害。

为了解决这个问题，作者提出了一种新的**joint multimodal late fusion**算法。该算法扩展了**joint label fusion**方法，在进行多模态融合时，明确地考虑了不同模态预测之间的模态间相关性和每个模态自身的**model uncertainty**。


# 2. 方法介绍

作者首先从理论上分析了单模态下的模型不确定性，然后提出了一种能够捕捉这种不确定性以及模态间相关性的融合方法。该方法的流程图如图所示。

![](https://pic1.imgdb.cn/item/69d753ba15908d0c2fa877e6.png)

## 2.1 单模态中的模型不确定性

对于一个单模态，给定训练数据 $$D = \{x_i, y_i\}_{i=1}^n$$ 和测试数据 $$T = \{t_j\}_{j=1}^m$$ 。模型不确定性源于能够完美拟合训练数据的模型参数$\theta$不唯一的问题。

对于一个测试样本$t_j$，在贝叶斯框架下，其期望预测可以通过对所有可能模型参数$\theta$的预测进行积分得到：

$$
p(y=l|t_j, D, U) = \int_{\theta \in \Theta_D} \delta(U(t_j, \theta)=l) p(\theta|D, U)
$$

其中，$\Theta_D$是所有能拟合训练数据的模型参数集合，$U$是学习模型，$\delta(\cdot)$是指示函数。期望预测误差$e(t_j, D, U)$则定义为所有“坏”的模型参数（即那些不能正确预测测试样本的参数）所贡献的误差的期望。

$$
e(t_j, D, U) = \int_{\theta \in \Theta_D \setminus \Theta_{D,T}} \delta(U(t_j, \theta) \neq y(t_j)) p(\theta|D, U)
$$

## 2.2 联合多模态融合 (Joint Multimodal Fusion)

作者扩展了**joint label fusion**方法，将其应用于多模态融合。对于一个包含$K$个模态的测试样本$t_j$，其融合后的标签分布通过对每个模态的预测进行加权平均得到：

$$
p(y=l|t_j) = \sum_{k=1}^{K} w_k p(y=l|t_j^k, D_k, U_k)
$$

其中，$w_k$是分配给第$k$个模态的权重，且$\sum_{k=1}^K w_k = 1$。融合的目标是选择一组权重$w$，使得在整个测试集$T$上的期望误差的平方和最小：

$$
w^* = \arg\max_w \sum_j e(t_j)^2 = \arg\max_w w^T M w
$$

其中，$M$是一个捕捉不同模态预测之间成对相关性的矩阵，其元素定义为：

$$
M[k_1, k_2] = \sum_j e(t_{j}^{k_1}, D_{k_1}, U_{k_1}) e(t_{j}^{k_2}, D_{k_2}, U_{k_2})
$$

上述优化问题有一个闭式解：$w^* = \frac{M^{-1} \mathbf{1}_K}{\mathbf{1}_K^T M^{-1} \mathbf{1}_K}$，其中$\mathbf{1}_K$是一个全为1的$K$维向量。

## 2.3 估计相关性矩阵M

在训练时，由于测试数据的真实标签是未知的，因此无法直接计算期望误差$e$和相关性矩阵$M$。作者提出在验证数据上估计$M$。

此外，为了估计期望误差，需要对模型参数空间$\Theta_D$进行采样。作者采用了深度集成方法：对于每个模态，通过随机初始化独立地训练$N$个分类器。这$N$个分类器可以看作是对模型参数空间$\Theta_D$的$N$次随机采样。对于一个验证样本$t_j$，其期望预测误差可以通过这$N$个分类器预测错误的比例来估计：

$$
e(t_j, D, U) = \frac{\sum_{i=1}^{N} \delta(U(t_j, \theta_i) \neq y(t_j))}{N}
$$

在验证集上计算出每个样本的期望误差后，就可以根据公式来估计相关性矩阵$M$。得到$M$后，即可通过闭式解计算出最佳的融合权重$w^*$。

# 3. 实验分析

作者在一项**NSCLC**患者术后一年生存预测的任务上进行了实验，数据包含**imaging**（影像）和**genetic**（基因）两种模态。

## 3.1 数据与预处理

使用了**NSCLC Radiogenomics**数据集，共包含107名接受了手术切除但未接受辅助/新辅助治疗的**NSCLC**患者。
*   基因表达数据：包含5268个基因的**RNA-sequencing**数据。
*   **CT**影像数据：基于提供的肿瘤**mask**，使用标准的**radiomics**库提取了107维的纹理和强度特征。

## 3.2 基线方法
对于基因表达和**CT radiomics**数据，都使用了一个包含两个隐藏层的全连接神经网络（**NN**）。
*   **NN-based intermediate fusion**：将每个模态的特征分别通过一个全连接层转换到一个50维的共同空间，然后拼接这些特征进行融合。
*   **NN-based late fusion**：将两个单模态模型的输出拼接起来，然后通过一个逻辑回归层进行融合。
*   **Deep Ensemble**：为了公平比较，作者也为所有基线方法实现了**deep ensemble**版本，即训练50个随机初始化的模型并对结果进行平均。

## 3.3 实验设置
采用4折交叉验证，数据按60-15-25的比例划分为训练、验证和测试集。对于作者提出的方法，在验证集上使用$N=50$次随机模型采样来估计相关性矩阵$M$。使用**area under curve** (**AUC**) of the **ROC** curve作为评估指标。

## 3.4 结果

|  | **w/o uncertainty** | **w uncertainty** |
|:---:|:---:|:---:|
| CT radiomics | 0.514 | 0.531 |
| gene-expression | 0.589 | 0.663 |
| NN intermediate fusion | 0.586 | 0.591 |
| NN late fusion | 0.529 | 0.613 |
| **proposed** | N/A | **0.728** |

**gene-expression**特征（**AUC**=0.589）比**CT radiomics**特征（**AUC**=0.514）更具判别性。

**NN intermediate fusion**和**NN late fusion**的性能都未能超越仅使用**gene-expression**特征的单模态模型。作者认为，这可能是因为训练数据量较小，使得基于学习的融合更具挑战性。

通过引入**deep ensemble**来考虑模型不确定性，所有基线方法的性能都得到了一致的提升。例如，**gene-expression**模型的**AUC**从0.589提升到了0.663。这证实了由于模型不确定性，任何单个训练的分类器都可能不可靠，而考虑所有潜在的分类器能带来更鲁棒的性能。

作者提出的后期融合方法取得了最佳的AUC值，达到了0.728。这一结果显著优于所有基线方法，包括它们的**deep ensemble**版本。这表明，通过显式地建模不同模态预测之间的相关性并考虑模型不确定性，该方法即使在训练数据较小的情况下也能表现出色。