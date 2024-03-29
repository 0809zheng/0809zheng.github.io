---
layout: post
title: '机器学习中的假设检验(Hypothesis Test)'
date: 2021-07-09
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/60e807c55132923bf84e8654.jpg'
tags: 数学
---

> Hypothesis Test in Machine Learning.

# 1. 假设检验的基本概念
在统计学中，总体分布往往是未知的，只能从中进行有限的抽样从而获得部分样本的信息。有时需要对总体的特征做出某种假设，如何判断该假设是正确的还是错误的？需要借助**假设检验(hypothesis test)**。

假设检验的核心思想是**小概率事件**和**反证法**。首先提出一个待检验的假设(该假设通常是想要去否定的)，通过统计方法试图证明该假设是小概率事件，利用**小概率事件**原理(小概率事件在少量实验中是几乎不可能出现的)证明该假设是错误的，从而**反证**假设的对立面很可能是正确的。

待检验的假设称为**原假设(zero hypothesis)**，记为$H_0$，该假设一般设置为消极的(如方法没效果或特征不显著)。原假设的对立面称之为**备择假设(alternative hypothesis)**，记为$H_1$，如果拒绝了原假设，则可以接受备择假设。

通过统计学方法，在原假设$H_0$的基础上构造**检验统计量**，可以得到其统计分布。给定**显著性水平(significance level)** $\alpha$作为将事件划分到“小概率事件”的概率，则“小概率事件”对应的分布区间称为**拒绝域(critical region)**；拒绝域之外的区域称为**置信区间(confidence interval)**。对于某次检验统计量的观测值，如果其落入拒绝域中，则认为原假设$H_0$不成立，从而拒绝原假设$H_0$，接受备择假设$H_1$。

假设检验的流程如下：
1. 提出关于总体的备择假设$H_1$，收集样本数据；
2. **反证法**，假设原假设$H_0$是正确的，构造检验统计量及其分布；
3. 计算样本数据的检验统计量观测值在分布中的位置；
4. 将结果与临界值进行比较；
5. 若为**小概率事件**，接受备择假设$H_1$；反之拒绝备择假设$H_1$。

# 2. 机器学习中的假设检验
在机器学习任务中，经常会遇到“性能”比较：
- 希望比较模型的**泛化性能**，但只能获得测试集上的**测试性能**，两者的对比结果可能不相同；
- 测试集上的性能与测试集本身的选择有关，不同的测试集会有不同的测试结果；
- 有些机器学习算法具有随机性，即使使用相同参数的模型在同一个测试集上也可能得到不同的结果。

若想比较模型的性能，可以借助统计假设检验。假设检验中的“假设”通常是对模型泛化错误率分布的猜想。几种常用的比较机器学习性能的假设检验方法如下：
1. 二项检验：通过单次实验(留出法)检验单个模型的泛化性能
2. **t**检验：通过多次实验(交叉验证法)检验单个模型的泛化性能
3. 交叉验证**t**检验：通过多次实验(交叉验证法)检验两个模型的性能差异
4. **McNemar**检验：通过单次实验(留出法)检验两个模型的性能差异
5. **Friedman**检验：检验多个模型的性能差异
6. **Nemenyi**后续检验：检验多个模型的性能差异

### (1) 二项检验
- 假设：单个模型的泛化错误率$\epsilon$不超过给定值$\epsilon_0$：$\epsilon ≤ \epsilon_0$

以二分类任务为例。若记模型通过留出法评估的测试错误率为$\hat{\epsilon}$，假设共有$m$个样本，则共有$\hat{\epsilon} \times m$个样本被错误分类。泛化错误率$\epsilon$应服从**二项(binomial)分布**：

$$ P(\hat{\epsilon};\epsilon) = \text{C}_{m}^{\hat{\epsilon} \times m} \epsilon^{\hat{\epsilon} \times m}(1-\epsilon)^{m-\hat{\epsilon} \times m} $$

对假设$\epsilon ≤ \epsilon_0$可以应用**二项检验(binomial test)**，不加证明地给出显著性水平为$\alpha$时二项检验的临界值$\overline{\epsilon}$：

$$ \overline{\epsilon} = \mathop{\max}_{\epsilon} \text{ s.t. } \sum_{i=\epsilon_0 \times m+1}^{m} \text{C}_{m}^{i} \epsilon_0^{i}(1-\epsilon_0)^{m-i}<\alpha $$

### (2) t检验
- 假设：单个模型的泛化错误率$\epsilon$等于其通过$k$折交叉验证得到的多次测试错误率$$\hat{\epsilon}_1,...,\hat{\epsilon}_k$$的平均值$$\mu$$：$\epsilon = \mu =\frac{1}{k}\sum_{i=1}^{k}\hat{\epsilon}_i$

计算$k$次测试错误率$$\hat{\epsilon}_1,...,\hat{\epsilon}_k$$的平均值$$\mu$$和方差$$\sigma^2$$：

$$ \mu = \frac{1}{k}\sum_{i=1}^{k}\hat{\epsilon}_i $$

$$ \sigma^2 = \frac{1}{k-1}\sum_{i=1}^{k}(\hat{\epsilon}_i-\mu)^2 $$

构造统计量$\tau_t$：

$$ \tau_t = \frac{\sqrt{k}(\mu-\epsilon)}{\sigma} $$

上述统计量$\tau_t$服从自由度为$k-1$的**t**分布，则可以应用**t检验(t-test)**。

### (3) 交叉验证t检验
- 假设：模型**A**与模型**B**的性能相同，即在相同的训练/测试集上得到的测试错误率相同：$\epsilon_i^A = \epsilon_i^B$

对于模型**A**与模型**B**，通过$k$折交叉验证得到的测试错误率分别为$$\epsilon_1^A,...,\epsilon_k^A$$和$$\epsilon_1^B,...,\epsilon_k^B$$，其中$\epsilon_i^A$和$\epsilon_i^B$是在相同的第$i$折训练/测试集上得到的结果。采用**成对t检验(paired t-test)**进行检验。具体地，对每一对测试错误率求差$\Delta_i=\epsilon_i^A-\epsilon_i^B$，可得差值$\Delta_1,...,\Delta_k$。计算其平均值$$\mu$$和方差$$\sigma^2$$，构造统计量$\tau_t$：

$$ \tau_t = \frac{\sqrt{k}\mu}{\sigma} $$

上述统计量$\tau_t$服从自由度为$k-1$的**t**分布，则可以应用**t检验(t-test)**。

### (4) McNemar检验
- 假设：模型**A**与模型**B**的性能相同，即在一个模型上测试正确且在另一个模型上测试错误的样本数相同：$e_{01} = e_{10}$

对于二分类问题，使用留出法不仅可以计算模型**A**与模型**B**的测试错误率$\epsilon$，还可以统计在两个模型都正确/错误以及在一个模型正确另一个模型错误的样本数$e$，列出**列联表(contingency table)**：

$$
\begin{array}{l|cc}
    \text{模型B\模型A} & \text{正确} & \text{错误} \\
    \hline
    \text{正确} & e_{00} & e_{01} \\
    \text{错误} & e_{10} & e_{11} \\
\end{array}
$$

采用**McNemar检验**对上述假设进行检验。构造统计量：

$$ \tau_{\chi^2} = \frac{(|e_{01}-e_{10}|-1)^2}{e_{01}+e_{10}} $$

上述统计量$\tau_{\chi^2}$服从自由度为$1$的$\chi^2$分布。

### (5) Friedman检验
- 假设：多个模型的性能相同

当需要使用一组数据集对多个模型进行比较时，可以采用基于算法排序的**Friedman检验**。假设使用$N$个数据集对$k$个模型进行比较，首先在每个数据集上根据留出法或交叉验证法得到所有模型测试结果，根据测试性能对这些模型由好到坏排序，并赋予序值$1,2,...$，若算法的测试性能相同则平分序值。列出比较序值表，并计算平均序值：

$$
\begin{array}{c|ccc}
    \text{数据集} & \text{模型1} & ... & \text{模型k} \\
    \hline
    D_1 & 1 & 2 & 3 \\
    ... & 1.5 & 1.5 & 3 \\
    D_N & 1 & 2.5 & 2.5 \\ \hline
    \text{平均序值} & 1.18 & 2 & 2.83 \\
\end{array}
$$

若假设所有模型的性能相同，则这些模型的平均序值应该相同。采用**Friedman检验**进行假设检验。令$r_i$表示第$i$个模型的平均序值，(若不考虑平分序值的情况)$r_i$的均值和方差分别表示为$\frac{(k+1)}{2}$和$\frac{(k^2-1)}{12N}$。构造统计量：

$$ \tau_{\chi^2} = \frac{k-1}{k} \frac{12N}{k^2-1} \sum_{i=1}^{k} (r_i-\frac{k+1}{2})^2 = \frac{12N}{k(k+1)} (\sum_{i=1}^{k}r_i^2-\frac{k(k+1)^2}{4}) $$

在$k$和$N$都较大时，上述统计量$\tau_{\chi^2}$服从自由度为$k-1$的$\chi^2$分布。原始检验要求$k>30$，实际中通常使用统计量：

$$ \tau_{F} = \frac{(N-1)\tau_{\chi^2}}{N(k-1)-\tau_{\chi^2}} $$

述统计量$\tau_{F}$服从自由度为$k-1$和$(k-1)(N-1)$的**F**分布。

### (6) Nemenyi后续检验
若假设“所有模型的性能相同”被拒绝，则说明模型的性能显著不同。可通过**Nemenyi后续检验(post-hoc test)**进一步区分各模型。即计算每个模型平均序值的临界值域：

$$ CD = q_{\alpha}\sqrt{\frac{k(k+1)}{6N}} $$

其中$q_{\alpha}$是**Tukey**分布的临界值。根据每个模型的平均序值和临界值域，可以绘制**Friedman检验图**：

![](https://pic.imgdb.cn/item/60e9525b5132923bf8d14a3f.jpg)

若两个模型的横线段有交叠，则说明这两个模型没有显著差别；否则即说明有显著差别。

