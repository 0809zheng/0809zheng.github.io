---
layout: post
title: '多臂老虎机'
date: 2020-09-02
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f505cc9160a154a671bb05b.jpg'
tags: 机器学习
---

> Multi-armed bandit.

# 1. 模型介绍
**多臂老虎机（Multi-Armed Bandit，MAB）**是统计学中一个经典问题。设想一个赌徒面前有$N$个老虎机，事先他不知道每台老虎机的真实盈利情况，他如何根据每次玩老虎机的结果来选择下次拉哪台或者是否停止赌博，来最大化自己的从头到尾的收益。

❀关于多臂老虎机问题名字的来源，是因为老虎机在以前是有一个操控杆，就像一只手臂(**arm**)，而玩老虎机的结果往往是口袋被掏空，就像遇到了土匪(**bandit**)一样。

![](https://pic.downk.cc/item/5f506bda160a154a671ee1f3.jpg)

假设共有$K$台**arm**，进行$T$轮实验。对于第$t$轮实验，**代理（agent）**根据给定的**策略（policy）**采取**行动（action）**选择一台**arm** $a_t$，并获得**奖励（reward）** $r_t$。

若记最优的**arm**能够得到奖励$μ$，则目标是使所有实验中的累计**遗憾（regret）**最小化：

$$ R(T) = μT-\frac{1}{T} \sum_{t=1}^{T} {r_t(a_t)} $$

实际中假设状态空间有限但非常大，因此遍历所有**arm**是不现实的，因此需要权衡**探索和利用(Exploration and Exploitation,EE)**。
- **探索(Exploration)**指的是每次优先选择之前没有选择过的**arm**；
- **利用(Exploitation)**指的是优先选择已知奖励高的**arm**，利用当前已有的信息来最大化收益。

### ⚪ε-greedy 算法
当$K$台**arm**的**reward**已知或已经被估计出来时，可以通过**ε-greedy**算法进行搜索。
1. 以$ε$的概率在$K$台**arm**之间等概率随机选择；
2. 以$1-ε$的概率在已经探索过的**arm**中选择**reward**最高的。

### ⚪UCB(Upper Confidence Bound)算法
实际中$K$台**arm**的**reward**是未知的，需要进行估计。假设每台**arm**的**reward**服从某个概率分布，对**reward**的置信区间进行估计。

**UCB**算法是在利用阶段时，选择**reward**估计的置信上界最大的**arm**。

# 2. Contextual MAB
**Contextual MAB**需要结合上下文信息（**context**），即每台**arm**在每一时刻的**reward**是变化的。

对于第$t$轮实验，**agent**能够观察到当前**context** $x_t$，根据**policy**选择一个**arm** $a_t$，目标是使所有实验中的累计**遗憾（regret）**最小化：

$$ R(T) = \mathop{\max}_{\pi \in \Pi} \frac{1}{T} \sum_{t=1}^{T} {r_t(\pi (x_t))}-\frac{1}{T} \sum_{t=1}^{T} {r_t(a_t)} $$

### ⚪Disjoint Linear Model
在**Contextual MAB**问题中，每台**arm**的**reward**与**context**相关。一种简单的模型假设每台**arm**在第$t$轮的特征$x_{t,a}$与其奖励$r_{t,a}$呈线性关系：

$$ E[r_{t,a} \mid x_{t,a}] = x^T_{t,a}θ_a $$

其中参数$θ_a$可以通过岭回归等算法进行学习。



