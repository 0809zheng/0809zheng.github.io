---
layout: post
title: '期望最大算法'
date: 2020-03-26
author: 郑之杰
cover: 'https://img.imgdb.cn/item/6074ef4d8322e6675cfb7a34.jpg'
tags: 机器学习
---

> Expectation Maximization.

**期望最大(expectation maximization,EM)算法**是一种迭代计算的数值方法，用来解决含有**隐变量**的参数模型的极大似然估计。

本文目录：
1. **EM**算法
2. 算法收敛性
3. 广义**EM**算法


# 1. EM算法
概率模型通常含有**观测变量(observable variable)**，给定数据后可以通过**极大似然估计**方法或**贝叶斯估计**方法估计模型参数。如果模型还包含**隐变量(latent variable)**，则上述方法失效，需要使用期望最大算法进行迭代计算。

记观测变量为$x$，隐变量为$z$，它们对应的数据总称为**完全数据(complete data)**。假设模型参数为$\theta$，则极大化观测变量$x$关于参数$\theta$的对数似然函数：

$$ L(\theta) = logP(x ; \theta) = log \int_{z}^{} P(x,z ; \theta) dz $$

注意到上式是**intractable**的，因为对数函数中包含积分项，需要采用近似算法。

### (1) 使用条件概率和KL散度推导EM算法
对对数似然函数做如下变换：

$$ logP(x ; \theta) = log \frac{P(x,z ; \theta)}{P(z|x ; \theta)} \\ = logP(x,z ; \theta) - logP(z|x ; \theta) \\ = log\frac{P(x,z ; \theta)}{Q(z)} - log\frac{P(z|x ; \theta)}{Q(z)} $$

上式即使已知联合分布$P(x,z ; \theta)$，其条件分布$P(z\|x ; \theta)$也是未知的；引入一个新的分布$Q(z)$近似该分布。上式两端对$Q(z)$求期望：

$$ \text{左端} = \int_{z}^{} Q(z) logP(x ; \theta) dz = logP(x ; \theta) \int_{z}^{} Q(z) dz = logP(x ; \theta) $$

$$ \text{右端} = \int_{z}^{} Q(z) log\frac{P(x,z ; \theta)}{Q(z)} dz - \int_{z}^{} Q(z) log\frac{P(z|x ; \theta)}{Q(z)} dz \\ = \text{ELBO} + \text{KL}(Q(z) || P(z|x ; \theta)) $$

上式中**ELBO**为**证据下界(evidence lower bound)**，是对数似然函数的一个下界；**KL**散度是一个不小于零的值。即：

$$ logP(x ; \theta) = \text{ELBO} + \text{KL}(Q || P) ≥ \text{ELBO} $$

极大似然估计的原目标函数为：

$$ \theta_{\text{MLE}} = \mathop{\arg \max}_{\theta} logP(x ; \theta) $$

假设对于第$t$轮计算得到的$\theta^{(t)}$，**KL**散度可以忽略，即$\text{KL}(Q(z) \| \| P(z\|x ; \theta^{(t)})) ≈ 0$，此时$Q(z) ≈ P(z\|x ; \theta^{(t)})$，且目标函数近似替代为：

$$ \mathop{\arg \max}_{\theta} logP(x ; \theta) = \mathop{\arg \max}_{\theta} \text{ELBO} $$

此时**ELBO**可以表示为：

$$ \text{ELBO} = \int_{z}^{} Q(z) log\frac{P(x,z ; \theta)}{Q(z)} dz \\ = \int_{z}^{} P(z|x ; \theta^{(t)}) log\frac{P(x,z ; \theta)}{P(z|x ; \theta^{(t)})} dz \\ = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz - \int_{z}^{} P(z|x ; \theta^{(t)}) logP(z|x ; \theta^{(t)}) dz $$

注意到上式右端只有第一项包含参数$\theta$，因此最大化**ELBO**等价于最大化第一项，即：

$$ \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz $$

### (2) 使用边缘分布和Jenson不等式推导EM算法
对对数似然函数做如下变换：

$$ logP(x ; \theta) = log \int_{z}^{} P(x,z ; \theta) dz = log \int_{z}^{} \frac{P(x,z ; \theta)}{Q(z)} Q(z) dz \\ = log \Bbb{E}_{Q(z)} [\frac{P(x,z ; \theta)}{Q(z)}] $$

引入一个新的分布$Q(z)$，将上式表示成期望的形式，由于$log()$函数是**凹函数(concave)**，由**Jenson**不等式：

$$ log \Bbb{E}_{Q(z)} [\frac{P(x,z ; \theta)}{Q(z)}] ≥ \Bbb{E}_{Q(z)} [log \frac{P(x,z ; \theta)}{Q(z)}] $$

上式等号成立当且仅当$\frac{P(x,z ; \theta)}{Q(z)}$为常数$C$，此时$P(x,z ; \theta)=CQ(z)$，两端对$z$取积分得$C=P(x; \theta)$，故$Q(z)=\frac{P(x,z ; \theta)}{P(x; \theta)}=P(z\|x ; \theta)$。

极大似然估计的原目标函数为：

$$ \theta_{\text{MLE}} = \mathop{\arg \max}_{\theta} logP(x ; \theta) $$

假设对于第$t$轮计算得到的$\theta^{(t)}$，**Jenson**不等式可取等号，即$Q(z)=P(z\|x ; \theta^{(t)})$，此时目标函数近似替代为：

$$ \mathop{\arg \max}_{\theta} logP(x ; \theta) = \mathop{\arg \max}_{\theta} \Bbb{E}_{Q(z)} [log \frac{P(x,z ; \theta)}{Q(z)}] \\ = \mathop{\arg \max}_{\theta} \Bbb{E}_{P(z|x ; \theta^{(t)})} [log \frac{P(x,z ; \theta)}{P(z|x ; \theta^{(t)})}] \\ = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) log \frac{P(x,z ; \theta)}{P(z|x ; \theta^{(t)})} dz \\ = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) log P(x,z ; \theta) dz - \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) log P(z|x ; \theta^{(t)}) dz $$

注意到上式右端只有第一项包含参数$\theta$，因此最大化目标函数等价于：

$$ \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz $$

### (3) 理解EM算法
**EM**算法的表达式如下：

$$ \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz $$

**EM**算法在迭代时分成两步。第一步(**E-step**)，计算联合分布的对数似然函数$logP(x,z ; \theta)$关于给定参数$\theta^{(t)}$下的条件分布$P(z\|x ; \theta^{(t)})$的期望；第二步(**M-step**)，计算期望的最大值。参数的初值可以任意选择，但需注意**EM**算法对初值是敏感的。

![](https://img.imgdb.cn/item/6074ef4d8322e6675cfb7a34.jpg)

上图给出了**EM**算法的直观解释。对数似然函数表示为曲线$L(\theta)$，该曲线是无法直接求解的。因此在每次迭代$\theta^{(t)}$时寻找对数似然函数的一个下界(即**ELBO**,表示为$B(\theta,\theta^{(t)})$)，通过极大化该下界来近似极大化对数似然。由于$L(\theta)≥B(\theta,\theta^{(t)})$，保证每次迭代时对数似然都是增加的。


# 2. 算法收敛性
原问题的极大似然估计的目标函数为：

$$ \theta_{\text{MLE}} = \mathop{\arg \max}_{\theta} logP(x ; \theta) $$

注意到$P(x ; \theta)≤1$，因此函数$logP(x ; \theta)$有上界；只需要证明该函数对于参数$\theta$是单调的，则可说明该目标函数是收敛的。即证：

$$ logP(x ; \theta^{(t+1)}) ≥ logP(x ; \theta^{(t)}) $$

对目标函数做一些变换：

$$ logP(x ; \theta) = log \frac{P(x,z ; \theta)}{P(z|x ; \theta)} \\ = log P(x,z ; \theta) - log P(z|x ; \theta) $$

上式两端对$P(z\|x ; \theta^{(t)})$求期望：

$$ \text{左端} = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x ; \theta) dz = logP(x ; \theta) \int_{z}^{} P(z|x ; \theta^{(t)}) dz = logP(x ; \theta) $$

$$ \text{右端} = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz - \int_{z}^{} P(z|x ; \theta^{(t)}) logP(z|x ; \theta) dz $$

若记：

$$ Q(\theta) = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz $$

$$ H(\theta) = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(z|x ; \theta) dz $$

则有：

$$ logP(x ; \theta) = Q(\theta) - H(\theta) $$

若证$logP(x ; \theta^{(t+1)}) ≥ logP(x ; \theta^{(t)})$,不妨证$Q(\theta^{(t+1)}) ≥ Q(\theta^{(t)})$,$H(\theta^{(t+1)}) ≤ H(\theta^{(t)})$。

### 证明 $Q(\theta^{(t+1)}) ≥ Q(\theta^{(t)})$

由于$Q(\theta)$恰好是期望最大算法的第一步得到的期望值，即：

$$ \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz = \mathop{\arg \max}_{\theta} Q(\theta) $$

因此$\theta^{(t+1)}$是使得$Q(\theta)$取得最大值的参数：

$$ Q(\theta^{(t+1)}) = \mathop{max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz \\ ≥ \mathop{max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta^{(t)}) dz = Q(\theta^{(t)}) $$

### 证明 $H(\theta^{(t+1)}) ≤ H(\theta^{(t)})$

$$ H(\theta^{(t+1)}) - H(\theta^{(t)}) \\ = \int_{z}^{} P(z|x ; \theta^{(t)}) logP(z|x ; \theta^{(t+1)}) dz - \int_{z}^{} P(z|x ; \theta^{(t)}) logP(z|x ; \theta^{(t)}) dz \\ = \int_{z}^{} P(z|x ; \theta^{(t)}) log\frac{P(z|x ; \theta^{(t+1)})}{P(z|x ; \theta^{(t)})} dz \\ = \Bbb{E}_{P(z|x ; \theta^{(t)})}[log\frac{P(z|x ; \theta^{(t+1)})}{P(z|x ; \theta^{(t)})}] \\ \text{(由Jenson不等式)} \\ ≤ log\Bbb{E}_{P(z|x ; \theta^{(t)})}[\frac{P(z|x ; \theta^{(t+1)})}{P(z|x ; \theta^{(t)})}] \\ = log \int_{z}^{} P(z|x ; \theta^{(t)}) \frac{P(z|x ; \theta^{(t+1)})}{P(z|x ; \theta^{(t)})} dz \\ = log \int_{z}^{} P(z|x ; \theta^{(t+1)}) dz = log1 = 0 $$

# 3. 广义EM算法
**EM**算法的表达式如下：

$$ \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \int_{z}^{} P(z|x ; \theta^{(t)}) logP(x,z ; \theta) dz \\ = \mathop{\arg \max}_{\theta} \Bbb{E}_{P(z|x ; \theta^{(t)})}[logP(x,z ; \theta)] $$

在迭代中分为两步执行：

$$ \text{E-step：} P(z|x ; \theta^{(t)}) → \Bbb{E}_{P(z|x ; \theta^{(t)})}[logP(x,z ; \theta)] $$

$$ \text{M-step：} \theta^{(t+1)} = \mathop{\arg \max}_{\theta} \Bbb{E}_{P(z|x ; \theta^{(t)})}[logP(x,z ; \theta)] $$

上述算法要求条件概率$P(z\|x ; \theta^{(t)})$是可求的。若该条件概率不可求，则仍将对数似然函数看作**ELBO**和**KL**散度之和：

$$ logP(x ; \theta) = \int_{z}^{} Q(z) log\frac{P(x,z ; \theta)}{Q(z)} dz - \int_{z}^{} Q(z) log\frac{P(z|x ; \theta)}{Q(z)} dz \\ = \text{ELBO(Q,\theta)} + \text{KL}(Q(z) || P(z|x ; \theta)) $$

将$Q(z)$也看做未知量，在迭代求解时与参数$\theta$一起求解。即给定$\hat{\theta}$时，有：

$$ \hat{Q} = \mathop{\arg \min}_{Q} \text{KL}(Q(z) || P(z|x ; \theta)) = \mathop{\arg \max}_{Q} \text{ELBO(Q,\hat{\theta})} $$

给定$\hat{Q}$时，有：

$$ \hat{\theta} = \mathop{\arg \max}_{\theta} \text{ELBO(\hat{Q},\theta)} $$

上述算法被称作**广义期望最大(generalized expectation maximization,GEM)**算法，也被称作**极大-极大(maximization-maximization)**算法，算法流程如下：

$$ \text{E-step：} Q^{(t+1)} = \mathop{\arg \max}_{Q} \text{ELBO(Q,\theta^{(t)})} $$

$$ \text{M-step：} \theta^{(t+1)} = \mathop{\arg \max}_{Q} \text{ELBO(Q^{(t+1)},\theta)} $$


