---
layout: post
title: 'Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology'
date: 2025-10-21
author: 郑之杰
cover: 'https://pic1.imgdb.cn/item/6901d2213203f7be00b075df.png'
tags: 论文阅读
---

> Additive MIL：用于病理学的本质可解释多实例学习.

- paper：[Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology](https://arxiv.org/abs/2206.01794)


# 0. TL; DR

这篇论文提出了一种名为**可加性多实例学习（Additive Multiple Instance Learning, Additive MIL）**的本质可解释的**MIL**范式。对标准的**Attention MIL**模型进行一个简单改动——先对每个实例进行分类级别的贡献度计算，然后再将所有实例的贡献度相加，得到最终的预测结果。这种可加性结构使得包级别预测可以被精确地分解为每个实例对每个类别的边际贡献度。

在**Camelyon16, TCGA-NSCLC**和**TCGA-RCC**三个大规模**WSI**数据集上，**Additive MIL**模型的预测性能与非可加性模型相当甚至更优。其热力图能够提供类别相关的、以及**促进（excitatory）/抑制（inhibitory）**的贡献度信息。


# 1， 背景介绍

在计算病理学领域，基于多实例学习（**MIL**）的深度学习模型，特别是以**Attention MIL (ABMIL)**为代表的方法，在全切片图像（**WSI**）的自动诊断、分级、预后预测等方面取得了巨大成功。这些模型能够仅通过切片级别的标签，学习到强大的分类能力。

**ABMIL**的注意力权重$α_i$可以被可视化为热力图，高亮出模型认为“重要”的区域。这被广泛地用作一种可解释性工具。作者指出了依赖注意力热力图进行空间归因（**spatial credit assignment**）的根本性缺陷：
1.  **必要不充分**: 高注意力权重只意味着这个**patch**的特征被模型需要，但不代表它对最终的阳性预测有贡献。例如模型可能高度关注某个良性区域，因为它能提供与癌变区域的对比信息，但这不意味着这个良性区域是致癌的。
2.  **正负不分 (Excitatory vs. Inhibitory)**: 注意力权重是非负的。它无法区分一个**patch**是在提供存在癌症的证据（促进性），还是在提供不是癌症的证据（抑制性）。例如炎性细胞聚集可能获得高注意力，因为它帮助模型排除了癌症，但热力图上它看起来和真正的癌细胞一样亮。
3.  **类别不分 (Class-agnostic)**: 在多分类任务中（如癌症亚型分类），注意力权重是类别无关的。一个**patch**获得高注意力，无从知晓它是在为A亚型、B亚型还是C亚型提供证据。
4.  **忽略交互**: 注意力权重是独立计算的，它无法反映**patch**之间的交互作用对最终分类的影响。例如两个中等权重的**patch**，在分类层可能共同提供了强有力的证据。


# 2. Additive MIL

**Additive MIL**的核心思想源于广义可加模型（**GAMs**），旨在通过一个简单的结构性约束，实现模型的内在可解释性。

标准的**Attention MIL**模型的计算流程是特征提取 -> 注意力加权 -> 求和聚合 -> 分类。**Additive MIL**的核心改动，就是交换最后两个操作的顺序：特征提取 -> 注意力加权 -> （对每个实例）计算类别贡献度 -> 求和聚合。

$$ p_{\text{Additive}}(x) = \sum_{i=1}^N \psi_p(m_i(x)) $$

$\psi_p(m_i(x))$是实例$i$对最终预测的边际贡献度。它是一个向量，维度等于类别数$C$。对于一个$C$分类任务，$\psi_p$会为每个加权后的实例表示$m_i(x)$输出一个$C$维的向量，代表这个实例分别对每个类别的**logit**贡献了多少。将所有实例的贡献度向量按位相加，就得到了整个包（**WSI**）对于所有类别的最终**logits**。

![](https://pic1.imgdb.cn/item/6901d2903203f7be00b077d3.png)

这个方法施加了一个**可加性约束（additive constraint）**。最终的预测结果被强制定义为所有实例贡献度的线性总和。这意味着可以精确地、无损地将最终的预测分数，反向分解到每一个实例的贡献上。

论文证明了**Additive MIL**与博弈论中的**Shapley**值之间的联系。**Shapley**值是解决合作博弈中如何公平分配收益问题的经典方法。在机器学习中，它可以用来计算每个特征对于一个模型预测的“边际贡献”。作者指出，**Additive MIL**模型计算出的实例贡献度$g(x_i)$，与该实例的**Shapley**值$φ_i$成正比。

$$ g(x_i) \propto \phi_i(V, x) $$

这意味着**Additive MIL**提供的实例贡献度在数学上是公平且最优的信用分配方案，为**Additive MIL**的可解释性提供了强有力的理论。
1.  精确的边际贡献: 贡献度分数与最终预测直接相加，反映模型决策。
2.  类别相关的贡献: 对于多分类任务，可以为每个类别生成一张独立的贡献度热力图。
3.  区分促进与抑制: 贡献度分数有正有负。正值代表该**patch**促进了某个类别的预测（**excitatory**），负值代表抑制（**inhibitory**）。
4.  内在可解释: 无需任何后处理（**post-hoc**）方法，解释性是模型结构自带的属性。

# 3. 实验分析

在**Camelyon16, TCGA-NSCLC**和**TCGA-RCC**三个大规模**WSI**数据集上，将**Mean-pooling, ABMIL, TransMIL**等模型改造为**Additive**版本后，其预测性能与原始的、更复杂的非可加性模型相当，甚至在某些情况下更优。证明了通过一个合理的结构约束，可以在不牺牲（甚至提升）性能的前提下，获得完全的可解释性。性能的提升可能源于可加性约束作为一种有效的正则化，在小样本的病理数据上限制了模型过拟合。

![](https://pic1.imgdb.cn/item/6901f8173203f7be00b18508.png)

比较**Additive MIL**热力图与传统**Attention MIL**热力图在定位癌变区域上的优劣。在**Camelyon16**测试集上，使用病理学家手工标注的肿瘤区域作为金标准，评估两种热力图的**AUPRC**。**Additive MIL**热力图（AUPRC 0.47）在定位精度上显著优于**Attention MIL**热力图（AUPRC 0.36）。**Attention**热力图产生了更多的假阳性。作者还进行了一项专家评估。在绝大多数情况下（如**Camelyon16**上**49/50**的比例），病理学家都更倾向于使用**Additive MIL**热力图作为辅助诊断工具，因为它更准确、假阳性更少。

![](https://pic1.imgdb.cn/item/6901f8543203f7be00b18736.png)

**Additive MIL**热力图提供了传统注意力热力图无法企及的解释深度。在RCC亚型分类中，**Attention**热力图只能模糊地高亮出一片区域，而**Additive**热力图可以清晰地显示出：这片区域的一部分对**KIRC**亚型有强促进作用（青色），而另一部分对**KIRP**亚型有强促进作用（绿色）。

![](https://pic1.imgdb.cn/item/6901f9c63203f7be00b192a7.png)
![](https://pic1.imgdb.cn/item/6901f9dd3203f7be00b193c8.png)

**Additive**热力图可以用不同颜色（如红色vs蓝色）清晰地展示一个区域是在促进某个类别的预测，还是在抑制它。

![](https://pic1.imgdb.cn/item/6901fa463203f7be00b197b7.png)

当模型出错时，**Additive MIL**成为了强大的调试工具。在一个将良性**WSI**误判为恶性的案例中，**Attention**热力图毫无反应，而**Additive**热力图精确地高亮出了导致误判的元凶：一个异常增生的生发中心（**germinal center**），它在形态上与肿瘤有相似之处。在一个将**KIRP**亚型误判为**KICH**的案例中，**Additive**热力图显示，模型虽然正确识别出了**KIRP**区域（对**KIRP**贡献为正），但它错误地将一张罕见的肾上腺组织（**adrenal gland**）当成了**KICH**的强阳性信号，导致最终预测错误。

![](https://pic1.imgdb.cn/item/6901fa553203f7be00b19820.png)