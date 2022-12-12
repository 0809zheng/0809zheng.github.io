---
layout: post
title: 'Toward Multimodal Image-to-Image Translation'
date: 2022-03-18
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/63539c5616f2c2beb181763b.jpg'
tags: 论文阅读
---

> BicycleGAN：多模态图像翻译.

- paper：[Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

对于图像翻译任务(**Image-to-Image Translation**)，大多数方法获得的图像输出都是单一的。例如执行斑马到马的转换，被转换的同一马的照片将始终具有相同的外观和色调；这是因为这些方法的生成器不具有随机性。

本文设计了一种多样化图像翻译方法**BicycleGAN**，该方法在训练时依赖配对的数据集，因此该方法与**CycleGAN**并没有什么关系，而是与[<font color=Blue>Pix2Pix</font>](https://0809zheng.github.io/2022/03/10/p2p.html)比较相似。

# 1. BicycleGAN的整体结构

**BicycleGAN**整体采用**GAN**的结构，通过在生成器中引入噪声实现图像的多样化翻译。完成训练后通过调整输入生成器的噪声$z$，可以将图像$A$转换为具有不同风格和内容的图像$\hat{B}$。

![](https://pic1.imgdb.cn/item/63539f7c16f2c2beb18613ac.jpg)

**BicycleGAN**的训练过程采用双向的循环过程。

![](https://pic1.imgdb.cn/item/6353a01f16f2c2beb186f677.jpg)

一种过程采用[<font color=Blue>VAE-GAN</font>](https://0809zheng.github.io/2022/02/17/vaegan.html)的形式。将图像$B$通过一个编码器$E(\cdot)$编码为隐变量$z$，与图像$A$共同输入生成器重构图像$\hat{B}$。建立图像$B$和图像$\hat{B}$之间的重构损失和判别损失，并且构造隐变量$z$的**KL**散度。

另一种过程采用[<font color=Blue>CGAN</font>](https://0809zheng.github.io/2022/03/02/cgan.html)的形式。将图像$A$和随机噪声$z$输入生成器构造图像$\hat{B}$，将其与图像$B$共同构造判别损失。并且使用编码器$E(\cdot)$将图像$\hat{B}$编码为隐变量$z$，构造其与输入隐变量之间的重构损失。

根据上面的讨论，**BicycleGAN**由生成器、判别器和编码器三个网络组成。在实践时两种训练过程分别使用一个判别器能够提高生成图像的质量。

# 2. BicycleGAN的生成器

**BicycleGAN**的生成器采用**UNet**结构，将隐变量$z$引入生成器时也有两种实现形式，即与输入图像连接或者与下采样中的所有层特征连接：

![](https://pic1.imgdb.cn/item/6353a2b816f2c2beb18aa98a.jpg)

```python
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape

        self.fc = nn.Linear(latent_dim, self.h * self.w)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)
        d1 = self.down1(torch.cat((x, z), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)
```

# 3. BicycleGAN的判别器

**BicycleGAN**的判别器采用[<font color=Blue>Pix2Pix</font>](https://0809zheng.github.io/2022/03/10/p2p.html)提出的**PatchGAN**结构，输出为一个$N \times N$矩阵，其中的每个元素对应输入图像的一个子区域，用来评估该子区域的真实性。

```python
class MultiDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
```

# 4. BicycleGAN的编码器

**BicycleGAN**的编码器采用**VAE**形式的概率编码器，把输入图像编码为正态分布的均值和对数方差（以便后续重参数化）。

```python
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Encoder, self).__init__()
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar
```


# 5. BicycleGAN的目标函数

根据前面的讨论，**BicycleGAN**的目标函数包括：
- **VAE-GAN**过程：图像$B$和图像$\hat{B}$之间的重构损失和判别损失、隐变量$z$的**KL**散度。
- **CGAN**过程：图像$\hat{B}$与图像$B$之间的判别损失、隐变量$\hat{z}$与输入隐变量$z$之间的重构损失。

**BicycleGAN**的完整**pytorch**实现可参考[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/bicyclegan)，下面给出其损失函数的计算和参数更新过程：

```python
# Losses
gan_loss = torch.nn.BCELoss()
mae_loss = torch.nn.L1Loss()

# Initialize model
generator = Generator(opt.latent_dim, input_shape)
encoder = Encoder(opt.latent_dim, input_shape)
D_VAE = MultiDiscriminator(input_shape)
D_LR = MultiDiscriminator(input_shape)

# Optimizers
optimizer_GE = torch.optim.Adam(
    itertools.chain(encoder.parameters(), generator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = torch.randn(mu.size(0), opt.latent_dim)
    z = sampled_z * std + mu
    return z

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 3, opt.img_width // 2 ** 3)

for epoch in range(opt.n_epochs):
    for i, (real_A, real_B) in enumerate(dataloader):
        # Adversarial ground truths
        valid = torch.ones(real_A.shape[0], *patch).requires_grad_(False)
        fake = torch.zeros(real_A.shape[0], *patch).requires_grad_(False)

        # forward propogation of cVAE-GAN
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)  

        # forward propogation of cLR-GAN      
        sampled_z = torch.randn(real_A.size(0), opt.latent_dim)
        fake_B_ = generator(real_A, sampled_z)
        mu_, logvar_ = encoder(fake_B_)
        encoded_z_ = reparameterization(mu_, logvar_)

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------
        optimizer_D_VAE.zero_grad()
        loss_D_VAE = gan_loss(real_B, valid) + gan_loss(fake_B.detach(), fake)
        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------
        optimizer_D_LR.zero_grad()
        loss_D_LR = gan_loss(real_B, valid) + gan_loss(fake_B_.detach(), fake)
        loss_D_LR.backward()
        optimizer_D_LR.step()

        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------
        optimizer_GE.zero_grad()

        # ----------
        # cVAE-GAN
        # ----------

        # Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = gan_loss(fake_B, valid)

        # ---------
        # cLR-GAN
        # ---------

        # cLR Loss: Adversarial loss
        loss_LR_GAN = gan_loss(fake_B_, valid)
        # Latent L1 loss
        loss_latent = mae_loss(sampled_z, sampled_z_)

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------
        loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel 
                  + opt.lambda_kl * loss_kl + opt.lambda_latent * loss_latent

        loss_GE.backward()
        optimizer_GE.step()
```