---
layout: post
title: 'Investigating Human Priors for Playing Video Games'
date: 2020-07-09
author: 郑之杰
cover: 'https://pic.downk.cc/item/5f080a5314195aa594c8a08b.jpg'
tags: 论文阅读
---

> 探究电子游戏的人类先验知识.

- paper：Investigating Human Priors for Playing Video Games
- arXiv：[link](https://arxiv.org/abs/1802.10217v3)

本文主要阐述了人类在进行电子游戏时会引入一些先验知识帮助游戏的进行，当移除某些先验知识时会导致表现显著下降。

![](https://pic.downk.cc/item/5f080af314195aa594c8dbb3.jpg)

作者设计了八种游戏模式，分别是：
1. 原始的游戏；
2. 把游戏中的物体用像素块表示；
3. 把游戏中的贴图改成具有误导性的图形；
4. 把台阶上的物体和背景用像素块表示；
5. 把整个游戏用像素块表示；
6. 不同台阶使用不同的像素块；
7. 更改梯子的交互方式（需要左右键上升）；
8. 将游戏翻转90度。

作者通过对人类玩家进行测试，发现在这八种任务中玩家的平均通关时间、死亡次数（模2）和探索状态数（模1000）差异较大：

![](https://pic.downk.cc/item/5f080c6514195aa594c94efb.jpg)

