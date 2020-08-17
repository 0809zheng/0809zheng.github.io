---
layout: post
title: '适定问题与不适定问题'
date: 2020-08-16
author: 郑之杰
cover: ''
tags: 数学
---

> Well-posed problem ＆ Ill-posed problem.

**适定问题（Well-posed problem）**是指满足下列三个要求的问题:

1. a solution exists：解必须**存在**；
2. the solution is unique：解必须**唯一**；
3. the solution's behavior changes continuously with the initial conditions：解能根据初始条件连续变化，不会发生跳变，即解必须**稳定**。

上述三个要求中，只要有一个不满足，则称之为**不适定问题（ill-posed problems）**。

图像处理中**不适定问题（ill posed problem）**或称为**反问题（inverse Problem）**的研究从20世纪末成为国际上的热点问题，成为现代数学家、计算机视觉和图像处理学者广为关注的研究领域。典型的图像处理不适定问题包括：
- 图像去噪（Image De-nosing）
- 图像恢复（Image Restorsion）
- 图像放大（Image Zooming）
- 图像修补（Image Inpainting）
- 图像去马赛克（image Demosaicing）
- 图像超分辨（Image super-resolution）

**Jaeyoung**在**CVPR**的论文中这样描述**CV**中的不适定问题：
- In most cases, there are several possible output images corresponding to a given input image and the problem can be seen as a task of selecting the most proper one from all the possible outputs.
- 这种不适定问题就是：一个输入图像会对应多个合理输出图像，而这个问题可以看作是从多个输出中选出最合适的那一个。
