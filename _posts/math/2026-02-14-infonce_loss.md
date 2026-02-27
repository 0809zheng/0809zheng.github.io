---
layout: post
title: '对比学习的损失动力学(InfoNCE Loss Dynamics)'
date: 2026-02-14
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/69902991b411fb96b92a7caa.png'
tags: 数学
---

> InfoNCE Loss Dynamics.

如果你使用过[对比学习（**Contrastive Learning**）](https://0809zheng.github.io/2022/10/01/self.html#2-%E5%9F%BA%E4%BA%8E%E5%AF%B9%E6%AF%94%E5%AD%A6%E4%B9%A0%E7%9A%84%E8%87%AA%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95)，尤其是像 **SimCLR、MoCo** 或基于 **Memory Bank** 的实例判别方法，你很可能观察到一个反直觉的现象：在训练初期，损失函数的值不仅不下降，反而会显著上升，达到一个峰值后才开始稳步回落。

如果从数学的角度深入剖析这一现象背后的损失动力学，会发现这不仅正常，而且是一个非常健康的信号，标志着模型正在从一个“混沌”的随机状态，通过一个“坍缩”的中间态，最终进入一个“有序”的特征学习阶段。

# 一、对比学习与 InfoNCE 损失

回顾一下对比学习的核心：**InfoNCE** 损失函数。对于一个给定的“查询”（**query**）样本 $q$，有一个与之配对的“正样本”（**positive key**）$k_+$，以及一个包含 $N$ 个“负样本”（**negative keys**）的集合 $\{k_i\}$。**InfoNCE** 损失的目标是让 $q$ 与 $k_+$ 的相似度远大于它与所有负样本的相似度。其数学形式如下：

$$
\mathcal{L}_q = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\exp(\text{sim}(q, k_+) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(q, k_i) / \tau)}
$$

其中 $\text{sim}(u, v)$ 通常是余弦相似度 $u·v / (\|\|u\|\| \|\|v\|\|)$。在实践中通常会对特征向量进行 **L2** 归一化，所以 $\text{sim}(u, v) = u·v$。$τ$ 是**温度系数 (temperature)**，一个控制分布锐度的超参数。

为了简化分析，定义两个关键的对数几率（**logits**）：
*   **正样本对数几率**:  $z_+ = \text{sim}(q, k_+) / \tau$
*   **负样本对数几率**:  $z_{-, i} = \text{sim}(q, k_i) / \tau$

那么损失函数可以写成 **CrossEntropy** 形式：

$$
\mathcal{L}_q = -\left( z_+ - \log(e^{z_+} + \sum_{i=1}^{N} e^{z_{-, i}}) \right)
$$

接下来让我们分析在训练的不同阶段，这些 $z$ 值是如何变化的，以及它们如何驱动损失曲线的动态。

# 二、损失动力学的三阶段分析

可以将对比学习的整个训练初期划分为三个连续的阶段：**随机初始化阶段**、**对齐与坍缩阶段**、以及**发散与聚类阶段**。

## 阶段一：随机初始化 (t=0)

在训练开始时，神经网络的权重是随机初始化的。这意味着：
1.  **特征向量是随机的**: 对于任何输入，模型输出的特征向量 $q$, $k_+$, $\{k_i\}$ 在高维单位超球面上是随机分布的。
2.  **相似度期望为零**: 高维空间中两个随机向量的点积（余弦相似度）的期望值非常接近于 0。因此， $z_+$ 和所有的 $z_{-,i}$ 都约等于 0。

此时的初始损失值：

$$
\mathcal{L}_{\text{init}} \approx -\log \frac{e^0}{e^0 + \sum_{i=1}^{N} e^0} = -\log \frac{1}{1+N} = \log(1+N)
$$

**结论1**: 训练的初始损失值约等于 $\log($负样本数量 $+ 1)$。这是一个非常有用的 **sanity check**！

## 阶段二：对齐与坍缩 (损失上升期)

训练开始后，模型接收到第一个梯度信号。这个信号告诉它：“你需要提高 $sim(q, k_+)$”。模型为了快速降低损失，会采取一个最“偷懒”的策略：**对齐 (Alignment)**。

模型开始学习输出一些非随机的、结构化的特征。它发现，只要让所有输出的特征向量都变得彼此相似（即都指向表示空间中的相似方向），那么 $sim(q, k_+)$ 就会增加。这个过程也被称为**特征坍缩 (Feature Collapse)** 的早期阶段。

由于特征坍缩，不仅正样本对的相似度 $sim(q, k_+)$ 从 0 开始增加，所有负样本对的平均相似度 $\text{mean}(\text{sim}(q, k_i))$ 也会从 0 开始显著增加。对于损失函数 $\mathcal{L}_q$ 的梯度，关于负样本对数几率 $z_{-,i}$ 的偏导数是：

$$
\frac{\partial \mathcal{L}_q}{\partial z_{-,i}} = \frac{e^{z_{-,i}}}{e^{z_+} + \sum_{j=1}^{N} e^{z_{-,j}}} > 0
$$

这个梯度是正的，意味着增加负样本的相似度会直接导致损失增加。在训练初期，模型为了增加 $z_+$ 而采取的对齐策略，不可避免地也增加了所有的 $z_{-,i}$。

由于分母中的求和项 $\sum e^{z_{-,i}}$ 包含了更多负样本，即使每个 $z_{-,i}$ 的增长幅度不大，它们的总和效应也会非常剧烈，其增长速度会超过分子项 $e^{z_+}$ 的增长速度。

**结论2**: 在训练初期，模型为了对齐正样本而采取的特征坍缩策略，会导致负样本相似度系统性地增加，从而使总损失上升。这个阶段标志着模型正在摆脱随机状态，开始学习结构化信息。

## 阶段三：发散与聚类 (损失下降期)

当损失持续上升，梯度信号会变得越来越惩罚性。模型会意识到，仅仅让所有特征都变得一样是行不通的。此时，对比学习的真正威力开始显现：**发散 (Uniformity)**。

梯度现在强烈地“惩罚”高的负样本相似度。模型被迫学习一种新的策略：在拉近正样本对的同时，必须将所有负样本对在表示空间中推开，使它们均匀地分布在超球面上。这个“推”和“拉”的动态过程，最终使得来自同一个实例的不同视图（$q$ 和 $k_+$）在表示空间中形成紧密的聚类，而不同实例的聚类之间则相互远离。

经过这个阶段，模型学到了有意义的判别性特征。正样本对的相似度会持续走向 1，而负样本对的平均相似度则被抑制在一个较低的水平（$sim(q, k_+)$ >> $sim(q, k_i)$）。

此时，损失函数中的分子项 $e^{z_+}$ 将远大于分母中的求和项 $\sum e^{z_{-,i}}$，整个分数项趋近于 1，损失值因此开始稳步下降，并最终收敛。

**结论3**: 损失从峰值开始下降，标志着模型已经学会了“发散”策略，开始形成有效的实例级特征表示。

# 三、对比损失中的温度系数 $\tau$

回顾 **InfoNCE** 损失函数：

$$
\mathcal{L}_q = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\exp(\text{sim}(q, k_+) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(q, k_i) / \tau)}
$$

温度系数 $τ$ 位于相似度得分 $\text{sim}()$ 的分母上。它本质上是在应用 **Softmax** 函数之前，对所有的 **logits** ($\text{sim}/τ$) 进行缩放。
*   低 $τ$ **(e.g., 0.05, 0.07)**: 放大 **logits** 之间的差异。一个 $\text{sim}$ 值上的微小差别，在除以一个很小的 $τ$ 后，会被急剧放大。这使得 Softmax 分布变得非常“尖锐” (**peaky**)。模型被强迫去关注那些最困难的负样本（即与 $q$ 最相似的 $k_i$），因为它们的 $\exp(\text{sim}/τ)$ 值会显著高于其他负样本。这是一种**困难负样本挖掘 (Hard Negative Mining)** 的隐式机制。
*   高 $τ$ **(e.g., 0.2, 0.5)**: 缩小 **logits** 之间的差异，使得 **Softmax** 分布变得“平滑” (**smooth**)。模型会平等地对待所有负样本，任务变成了简单地将正样本与所有负样本的平均水平区分开。

简而言之，$τ$ 决定了任务的难度。低 $τ$ 是困难模式，高 $τ$ 是简单模式。下文可以看到，通过对$τ$在损失动力学中的作用进行详细分析，可以得到结论：
- 较小的$τ$学习过程更苛刻，梯度更大，可能导致损失曲线的峰值更高，但一旦越过峰值，收敛可能更快、更有效。
- 较大的$τ$学习过程更温和，梯度更小，训练更稳定。损失上升的幅度可能较小，甚至不明显，但模型可能因为缺乏对困难样本的区分压力而学不到足够精细的特征。

### 阶段一：随机初始化 (t=0)

此阶段不受 $τ$ 的影响。由于所有 $\text{sim}()$ 都约等于 0，**logits** $\text{sim}/τ$ 也约等于 0。因此，初始损失始终接近 $\log(N+1)$。

### 阶段二：对齐与坍缩 (损失上升期) & $τ$ 的放大效应

当模型进入这个阶段时，$τ$ 的作用开始显现。损失关于一个负样本相似度 $\text{sim}_i$ 的梯度（简化形式）：

$$
\frac{\partial \mathcal{L}_q}{\partial \text{sim}_i} \propto \frac{1}{\tau} \cdot \text{softmax}(\text{sim}_j / \tau)
$$

可以看到，梯度与 $1/τ$ 成正比。当 $τ$ 很低时，这个梯度被显著放大。在这个阶段，$\text{mean}(\text{sim}(q, k_i))$ 开始从 0 增加。一个较低的 $τ$ 会急剧放大这个增加所带来的惩罚（即损失的增量）。即使负样本的相似度只增加了 $ε$，损失的增加幅度也与 $ε/τ$ 相关。

因为低 $τ$ 使得模型对负样本相似度的增加更加敏感，它会导致损失在这一阶段爬升得更快、更高。它迫使模型更快地意识到坍缩策略是行不通的，因为它带来的惩罚（损失上升）非常巨大。

### 阶段三：发散与聚类 (损失下降期) & $τ$ 的判别压力

当模型被迫进入这个阶段，$τ$ 的作用转变为施加判别压力。一个低的 $τ$ 意味着，为了让损失下降，模型不仅需要让 $\text{sim}(q, k+)$ 大于 $\text{mean}(\text{sim}(q, k_i))$，它还必须让 $\text{sim}(q, k+)$ 显著大于最相似的那个负样本 $\text{max}(\text{sim}(q, k_i))$。

这种对困难负样本的关注，迫使模型学习更具判别力的特征，从而在表示空间中形成更清晰、更紧凑的聚类。聚类间的边界被推得更开。如果模型能够成功应对这种压力，它最终会收敛到一个特征表示更优的状态，这通常对应着更低的最终损失值和更好的下游任务性能。

低温 $τ$ 在损失下降阶段充当了一个严格的“考官”，通过强迫模型区分困难负样本，驱动它学习更鲁棒的特征表示。


# 四、对比损失的Toy Model

下面通过一个实际例子来检验对比损失的动力学。构造一些假数据和模型，并定义**InfoNCE**损失：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --- 1. Simulation Dataset ---
class ToyDataset(Dataset):
    def __init__(self, num_samples, data_dim):
        # 创建 num_samples 个独立的、高斯分布的“原型”数据
        self.prototypes = torch.randn(num_samples, data_dim)

    def __len__(self):
        return len(self.prototypes)

    def __getitem__(self, idx):
        # 对同一个原型添加两次不同的噪声，生成两个视图
        prototype = self.prototypes[idx]
        view1 = prototype + torch.randn_like(prototype) * 0.1 # view1
        view2 = prototype + torch.randn_like(prototype) * 0.1 # view2
        return view1, view2, idx

# --- 2. Simple MLP Model ---
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
        
# --- 3. Contrastive Loss with Memory Bank ---
class NCECriterion(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, x, targets, memory_bank, temperature):
        sim = torch.einsum('bd,nd->bn', x, memory_bank) / temperature
        loss = F.cross_entropy(sim, targets)
        return loss
```

定义超参数和训练过程：

```python
# --- Hyperparameters & Setup ---
torch.manual_seed(42)
np.random.seed(42)

# 数据参数
NUM_SAMPLES = 2048   # 数据集大小，也是 Memory Bank 大小
DATA_DIM = 512       # 原始数据维度
PROJECTION_DIM = 128 # 对比学习的特征维度

# 训练参数
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07   # 对比学习温度

def run_experiment():
    # 初始化数据
    dataset = ToyDataset(NUM_SAMPLES, DATA_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型、Memory Bank 和损失函数
    model = MLPEncoder(DATA_DIM, 512, PROJECTION_DIM)
    memory_bank = F.normalize(torch.randn(NUM_SAMPLES, PROJECTION_DIM), dim=1)
    criterion = NCECriterion(NUM_SAMPLES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 用于记录损失
    step_losses = []
    
    print("Starting toy model training...")
    global_step = 0
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for view1, view2, indices in progress_bar:
            model.train()
            
            # --- 前向传播 ---
            q = model(view1) # Query
            k = model(view2) # Key
            
            # 归一化
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)
            
            # --- 计算损失 ---
            loss1 = criterion(q, indices, memory_bank, TEMPERATURE)
            loss2 = criterion(k, indices, memory_bank, TEMPERATURE)
            loss = (loss1 + loss2) / 2
            
            # --- 反向传播与优化 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- 更新 Memory Bank ---
            with torch.no_grad():
                memory_bank[indices] = k.detach()
            
            # --- 记录与显示 ---
            step_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

    print("Training finished.")
    return step_losses
```

绘制损失曲线：

```python
training_losses = run_experiment()

# 绘制损失曲线
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# 原始损失
ax.plot(training_losses, alpha=1, label='Raw Step Loss')

# 理论初始损失
initial_loss_theory = np.log(NUM_SAMPLES)
ax.axhline(y=initial_loss_theory, color='red', linestyle='--', label=f'Theoretical Initial Loss: log({NUM_SAMPLES}) ≈ {initial_loss_theory:.2f}')

ax.set_title('Contrastive Loss Curve', fontsize=16, pad=20)
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('InfoNCE Loss', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
```

![](https://pic1.imgdb.cn/item/69902991b411fb96b92a7caa.png)