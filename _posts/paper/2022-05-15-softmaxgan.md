---
layout: post
title: 'Softmax GAN'
date: 2022-05-15
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/639977dbb1fccdcd36247e3b.jpg'
tags: 论文阅读
---

> 把生成对抗网络建模为Softmax函数.

- paper：[Softmax GAN](https://arxiv.org/abs/1705.07215v5)

本文作者把生成对抗网络建模为**Softmax**函数。具体地，若记共有$\|P\|=\|P_{data}\|+\|P_G\|$个样本，包括$\|P_{data}\|$个真实样本和$\|P_G\|$个生成样本。对于每个样本$x$，使用判别器$D(x)$计算**logits**，并通过**Softmax**函数进行建模：

$$ P(x) = \frac{e^{-D(x)}}{\sum_x e^{-D(x)}} = \frac{e^{-D(x)}}{Z_P} $$

对于判别器$D(x)$，希望其能正确地区分真实样本和生成样本。因此判别器将目标概率均等地分配给$\|P\|$中的所有真实样本，而生成样本的目标概率为$0$；则判别器学习的目标分布为：

$$ T(x) = \begin{cases} \frac{1}{|P_{data}|}, & \text{if  } x \in P_{data}(x) \\ 0, & \text{if  } x \in P_G(x) \end{cases} $$

构造交叉熵损失函数：

$$ \begin{aligned} L_D &= - \Bbb{E}_{x \text{~} P(x)} [T(x) \log P(x)] \\ &= - \Bbb{E}_{x \text{~} P(x)} [T(x) \log  \frac{e^{-D(x)}}{Z_P}] \\ &=- \Bbb{E}_{x \text{~} P_{data}(x)} [\frac{1}{|P_{data}|} \log  \frac{e^{-D(x)}}{Z_P}] \\ &= \frac{1}{|P_{data}|}\Bbb{E}_{x \text{~} P_{data}(x)} [ D(x)] + \log  Z_P \end{aligned} $$

对于生成器$G(x)$，希望其生成的样本足够接近真实样本。因此生成器将概率平均分配给所有样本；则生成器学习的目标分布为一个均匀分布：

$$ T(x) = \frac{1}{|P|}= \frac{1}{|P_{data}|+|P_{G}|}  $$

构造交叉熵损失函数：

$$ \begin{aligned} L_G &= - \Bbb{E}_{x \text{~} P(x)} [T(x) \log P(x)] \\ &= - \Bbb{E}_{x \text{~} P(x)} [\frac{1}{|P_{data}|+|P_{G}|} \log  \frac{e^{-D(x)}}{Z_P}] \\ &= \frac{1}{|P_{data}|+|P_{G}|}\Bbb{E}_{x \text{~} P(x)} [ D(x)] + \log  Z_P \\ &= \frac{1}{|P_{data}|+|P_{G}|}(\Bbb{E}_{x \text{~} P_{data}(x)} [ D(x)]+\Bbb{E}_{x \text{~} P_G(x)} [ D(x)] )+ \log  Z_P \end{aligned} $$

**Softmax GAN**的完整目标函数如下：

$$ \begin{aligned}  & \mathop{ \min}_{D}  \frac{1}{|P_{data}|}\Bbb{E}_{x \text{~} P_{data}(x)} [ D(x)] + \log  Z_P \\  & \mathop{ \min}_{G} \frac{1}{|P_{data}|+|P_{G}|}(\Bbb{E}_{x \text{~} P_{data}(x)} [ D(x)]+\Bbb{E}_{x \text{~} P_G(x)} [ D(x)] )+ \log  Z_P \end{aligned} $$


**Softmax GAN**的完整**pytorch**实现可参考[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/softmax_gan)，下面给出损失函数计算和参数更新过程：

```python
for epoch in range(opt.n_epochs):
    for i, real_imgs in enumerate(dataloader):
        batch_size = real_imgs.shape[0]
        # Adversarial ground truths
        g_target = 1 / (batch_size * 2)
        d_target = 1 / batch_size

        z = torch.randn(batch_size, opt.latent_dim) 
        gen_imgs = generator(z)  

        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs)                  

        # Partition function
        Z = torch.sum(torch.exp(-d_real)) + torch.sum(torch.exp(-d_fake))

        # 训练判别器
        optimizer_D.zero_grad()
        d_loss = d_target * torch.sum(d_real) + log(Z)
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = g_target * (torch.sum(d_real) + torch.sum(d_fake)) + log(Z)
        g_loss.backward()
        optimizer_G.step()
```


