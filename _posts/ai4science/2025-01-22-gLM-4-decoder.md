---
layout: post
title: '基因组基础模型IV：解码器框架'
date: 2025-01-22
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm4-decoder-000-cover.png'
tags: AI4S
---

> Genome Foundation Model part IV: Decoder Architectures.

**基因组基础模型（Genome Foundation Model, gFM）**，又称**基因组语言模型（Genomic Language Model, gLM）**，将 **DNA** 序列映射为长度为 $T$ 的离散词元序列 $z_{1:T}$，通过在大规模无标注基因组语料上拟合边缘概率或条件概率分布 $p_\theta(z)$，学习具备跨任务迁移能力的通用基因组表征。

本文是**基因组基础模型**系列的第四篇。全系列按“**分词 → 预训练 → 编码器 → 解码器**”的脉络组织：
- [I. DNA 分词策略](https://0809zheng.github.io/2025/01/01/gLM-1-token.html)：如何将碱基序列切分为词元。
- [II. 预训练策略](https://0809zheng.github.io/2025/01/08/gLM-2-object.html)：如何构造自监督信号从无标注基因组中学习。
- [III. 编码器框架](https://0809zheng.github.io/2025/01/15/gLM-3-encoder.html)：双向骨干网络如何在长序列上混合上下文信息。
- **IV. 解码器框架**：生成式骨干网络如何评估序列似然并从头生成序列。

**解码器框架（Decoder Architecture）**通常以自回归方式将序列联合概率分解为条件概率乘积 $p_\theta(x) = \prod_{t=1}^T p_\theta(x_t \mid x_{<t})$，或通过离散扩散过程建模数据分布的变分下界。与仅能计算局部伪似然的掩码编码器不同，解码器不仅能直接评估任意完整序列或复杂插入缺失突变（**Indel**）的对数似然差 $\log p_\theta(x^{\mathrm{alt}}) - \log p_\theta(x^{\mathrm{ref}})$，还具备从头（*de novo*）生成调控元件与人工基因组、或基于上下文提示补全序列的能力；其深层隐状态同样可通过线性探针或微调迁移至判别式任务。

- **1. Transformer 解码器框架（Transformer Decoders）**
  - [GenSLMs：密码子级原核预训练与潜空间扩散层次规划](#p-006)
  - [megaDNA：三级多尺度因果 Transformer 的噬菌体语言模型](#p-001)
  - [GENERator：功能基因座长上下文预训练的真核生成式模型](#p-002)
  - [GENERator-v2：分解式单核苷酸监督与基因组压缩预训练](#p-003)
  - [Genos：基于泛基因组与稀疏混合专家的百万碱基级模型](#p-004)
  - [AntigenLM：全基因组八节段保序预训练的流感演化预测模型](#p-005)
- **2. 隐式长卷积与混合框架（Convolutional & Hybrid Decoders）**
  - [HyenaDNA：基于隐式长卷积与数据门控的单核苷酸模型](#p-007)
  - [Evo：长卷积与注意力混合架构的全基因组生成基础模型](#p-008)
  - [Evo 2：多尺度卷积混合架构的跨生命域基因组基础模型](#p-009)
- **3. 状态空间混合框架（State-Space Hybrid Decoders）**
  - [HybriDNA：结构化状态空间对偶与因果注意力混合的长程模型](#p-010)
- **4. 离散扩散生成框架（Discrete Diffusion Decoders）**
  - [D3LM：连续掩码率加权去噪的离散扩散基因组语言模型](#p-011)

<a id="dec-transformer"></a>
# 1. Transformer 解码器框架（Transformer Decoders）

**自回归 Transformer 解码器**通过下三角因果注意力掩码 $M \in \{0, -\infty\}^{T \times T}$（当 $j > i$ 时 $M_{ij} = -\infty$）严格阻断未来信息回流，使位置 $i$ 的隐状态仅聚合前缀上下文 $x_{\le i}$：

$$
\mathrm{CausalAttn}(Q, K, V) = \mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
$$

在基因组建模中，直接套用标准自回归 **Transformer** 面临三项核心矛盾：（1）单碱基分辨率下，全基因组或长基因座序列极易触发二次方复杂度与键值缓存（**KV Cache**）显存瓶颈；（2）真核基因组中大段低复杂度非编码背景序列会主导自回归交叉熵损失，稀释编码区与顺式调控元件的梯度信号；（3）若采用 $k$-**mer** 分词压缩序列长度，扁平的 $4^k$ 词表分类损失会忽略近邻词元间的碱基重叠相似性，且难以直接导出单核苷酸分辨率的突变似然。

针对上述问题，基因组 **Transformer** 解码器发展出多尺度层次化注意力（**MegaByte** 架构）、分组查询注意力（**GQA**）配合稀疏混合专家（**MoE**）、基因中心语料压缩拼接，以及将 $k$-**mer** 预测分布边缘化投影回单碱基空间的分解式监督机制。

<a id="p-006"></a>
### ⚪ **GenSLMs**：密码子级原核预训练与潜空间扩散层次规划
- **paper**：[**GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics**](https://doi.org/10.1177/10943420231201154)（*IJHPCA* 2023）

**GenSLMs** 采用密码子级分词在跨属原核生物基因上预训练 **GPT** 解码器，并结合潜空间连续扩散高层规划与贝叶斯优化引导的奖励束搜索，在全基因组尺度建模并预测 **SARS-CoV-2** 变异株的演化轨迹。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-088-genslm.png)

**密码子级分词与两阶段全基因组迁移：** 模型将每三个连续核苷酸编码为一个密码子词元（词表大小 $64$），将约 $30\text{ kbp}$ 的病毒全基因组压缩为最高 $10{,}240$ 个密码子词元。为克服单一病毒序列高同源性导致的多样性不足，训练采取两阶段迁移：先在跨属原核生物同源蛋白家族的非冗余单基因序列上预训练基础自回归模型；再将窗口扩展至全长，在完整 **SARS-CoV-2** 基因组上微调，以捕获跨开放阅读框（**ORF**）的长程共进化耦合。

**基于潜空间扩散的句子级层次化生成：** 针对词元级自回归模型在超长序列生成中易丢失全局 **ORF** 结构连贯性的问题，模型将全基因组每 $512$ 个密码子切分为一个“句子”，分三步耦合全局规划与局部解码：
1. **对比学习句子编码器**：利用对比损失训练句子编码器，以前置与后置句子为条件预测中间缺失句子的表征，将每个 $512$ 密码子片段映射为高层潜表征。
2. **潜空间去噪扩散模型**：在基因组的句子级潜表征序列上训练去噪扩散概率模型，学习从高斯噪声 $z_T$ 逐步恢复全局高层表征序列 $z_0 = (z_{01}, z_{02}, \ldots)$。
3. **高层规划条件化滑动窗口解码**：微调解码器，使其在预测当前密码子时同时以前置局部窗口内的密码子序列 $r$ 与全局高层规划 $z_0$ 为条件，即建模 $p(x_t \mid r, z_0)$。

**奖励引导束搜索：** 为定向生成具有特定属性（如与参考序列 $y$ 的全局比对得分）的候选突变序列，模型在束搜索中融合阶段性奖励函数 $R(x) = \sum_{t=1}^L r_t(x_t)$。给定条件语言模型 $p_\theta$、长度惩罚系数 $\alpha$ 与奖励混合超参数 $\mu \in [0, 1]$，完整序列的修正评分函数与单步词元选择准则定义为：

$$
\begin{aligned}
&S(x) = \frac{\mu}{L^\alpha}\sum_{t=1}^L \log p_\theta(x_t \mid x_0, \ldots, x_{t-1}, c) + (1-\mu)R(x), \\
&s_t(x_t) = \mu\, p_\theta(x_t \mid \mathcal{X}_{\mathrm{beams}}^i, c) + (1-\mu)\, r_t(x_t),
\end{aligned}
$$

其中 $c$ 为前置推理上下文，混合系数 $\mu$ 由外层高斯过程代理模型通过贝叶斯优化动态调优。生成的候选全基因组经系统发育树放置后，可用于预警潜在的高风险关切变异株（**VOC**）。

<a id="p-001"></a>
### ⚪ **megaDNA**：三级多尺度因果 Transformer 的噬菌体语言模型
- **paper**：[**A long-context language model for deciphering and generating bacteriophage genomes**](https://doi.org/10.1038/s41467-024-53759-4)（*Nature Communications* 2024）

**megaDNA** 采用三级多尺度因果 **Transformer** 架构，在无注释完整噬菌体基因组上进行自回归预训练，实现高达 $96\text{ kbp}$ 的全基因组从头生成、零样本必需基因识别与跨尺度功能表征提取。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-087-megadna.png)

**单核苷酸分词与三级多尺度解码器：** 模型直接采用单核苷酸分词，将四个标准碱基映射为离散词元，在筛选后的高完整度噬菌体基因组语料上训练。为克服单碱基长序列的自注意力瓶颈，主干网络引入 **MegaByte** 式三级层次化解码器架构，各级配置递增的隐藏维度与感受野尺度：
1. **局部层（Local Layer）**：在 $16\text{ bp}$ 局部窗口内处理单核苷酸嵌入，捕获密码子与短调控基序等细粒度特征；
2. **中间层（Middle Layer）**：接收局部层输出，将有效感受野扩展至 $1{,}024\text{ bp}$，建模单个开放阅读框与顺式调控区的基因级结构；
3. **全局层（Global Layer）**：汇总中间层特征，将上下文窗口扩展至整个输入序列（最高达 $96\text{ kbp}$），捕获全基因组尺度的长程依赖。

推理阶段通过温度采样，模型可从头自回归生成包含启动子、核糖体结合位点（**RBS**）及蛋白编码区的完整候选噬菌体基因组序列。

**零样本计算机诱变与多尺度探针：** 在零样本必需基因与位点约束预测中，模型在基因组上执行滑动窗口计算机饱和诱变，通过计算突变序列相对于野生型序列的自回归损失增量，直接评估基因编码区以及起止密码子的功能必需性。在有监督下游任务中，模型分别提取局部层、中间层与全局层的内部激活向量并拼接为多尺度联合嵌入，接入线性探针即可预测蛋白质适应度、翻译效率与分类学归属。

<a id="p-002"></a>
### ⚪ **GENERator**：功能基因座长上下文预训练的真核生成式模型
- **paper**：[**GENERator: A Long-Context Generative Genomic Foundation Model**](https://arxiv.org/abs/2502.07272)（*arXiv* 2025）

**GENERator** 在真核生物基因中心功能语料上结合随机相位偏移的 $6$-**mer** 分词与 **LLaMA** 因果解码器架构，在 $98\text{k}$ 碱基窗口内实现序列恢复、词元概率边缘化的单碱基零样本变异效应预测以及提示词引导的顺式调控元件设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-030-6891d386.png)

**基因中心功能语料与随机相位 $6$-mer 分词：** 为防止自回归预训练过拟合基因间区低复杂度重复背景，模型仅提取注释基因边界内的完整功能基因座（涵盖外显子、内含子、非翻译区及邻近启动子与终止子）构建跨物种真核语料。模型采用定长 $6$-**mer** 分词，主干采用配备 **RoPE** 与分组查询注意力（**GQA**）的 **LLaMA** 解码器；预训练时在每个周期为每条序列在 $[0, 5]\text{ nt}$ 内均匀随机采样起始截断偏移量，使模型遍历同一底层序列的不同 $6$-**mer** 切分相位，消除分词边界伪影。

**词元概率边缘化的单碱基变异效应预测：** 在无比对零样本单核苷酸变异（**SNV**）评估中，为解决 $6$-**mer** 词表与单碱基突变的粒度失配，模型将变异位点置于因果序列末端，通过聚合覆盖目标位点的所有 $k$-**mer** 词元概率，将预测分布边缘化投影回单核苷酸 $4$ 分类空间。给定参考序列 $S = (s_1, \ldots, s_L)$，位点 $i$ 处出现碱基 $X \in \{\mathrm{A, T, C, G}\}$ 的边际概率以及参考等位基因 $R$ 突变为替代等位基因 $A$ 的对数似然比定义为：

$$
p(s_i = X) = \sum_{t \in \mathcal{T}_i} p(t)\,\mathbb{I}(t_j = X), \qquad
\mathrm{VEP}(R \to A) = \log \frac{p(s_i = R \mid S_{\setminus i})}{p(s_i = A \mid S_{\setminus i})},
$$

其中 $\mathcal{T}_i$ 为覆盖位点 $i$ 的词元集合，$j$ 为位点 $i$ 在词元 $t$ 内的相对偏移量。

**提示词引导的顺式调控元件设计与归因：** 在顺式调控元件（**CRE**）可控设计中，框架解耦为两个微调组件：（1）**活性条件化生成器**根据实验活性分位数将训练集离散化为高、中、低区间，在序列前端拼接专用前缀提示词元（`<high>`、`<mid>`、`<low>`）进行微调；（2）**定量活性预测器**将主干微调为回归器 $f(\cdot)$ 对生成候选集排序，并通过将位点 $i$ 替换为其余三种替代碱基得到突变序列 $S_{i \to x}$，计算单碱基活性贡献度：

$$
C(i) = f(S) - \frac{1}{3} \sum_{\substack{x \in \{\mathrm{A, T, C, G}\} \\ x \neq s_i}} f(S_{i \to x}).
$$

<a id="p-003"></a>
### ⚪ **GENERator-v2**：分解式单核苷酸监督与基因组压缩预训练
- **paper**：[**Functional In-Context Learning in Genomic Language Models with Nucleotide-Level Supervision and Genome Compression**](https://doi.org/10.64898/2026.01.27.702015)（*bioRxiv* 2026）

**GENERator-v2** 在保持 **Transformer** 解码器与 $6$-**mer** 分词高吞吐量的基础上，提出分解式单核苷酸监督（**FNS**）与基因组压缩预训练（**GCP**），解决了非重叠 $k$-**mer** 分类损失忽略近邻碱基相似性以及真核基因组功能元件稀疏导致跨基因互作欠拟合的问题。

**分解式单核苷酸监督（FNS）与相位不变训练：** 标准 $k$-**mer** 下一词元预测将大小为 $4^k$ 的词表视为无结构原子类别，使仅相差单个碱基的预测与完全错误预测承受相同惩罚。**FNS** 在不改变分词器或网络输出维度的前提下，将词元分布 $p(v)$ 边缘化投影为块内各单碱基位点的条件分布。设词表 $\mathcal{V} = \mathcal{B}^k$，对于词元内部相对位点 $j \in \{1, \ldots, k\}$ 与碱基 $b \in \mathcal{B}$，诱导出的单碱基边际概率及对应真实目标 $y = (y^{(1)}, \ldots, y^{(k)})$ 的单步训练损失定义为：

$$
p^{(j)}(b) = \sum_{v \in \mathcal{V}:\, v^{(j)} = b} p(v), \qquad
\mathcal{L}_{\mathrm{FNS}} = -\sum_{j=1}^k \log p^{(j)}\!\big(y^{(j)}\big),
$$

其中 $v^{(j)}$ 表示词元 $v$ 的第 $j$ 个碱基。该目标等价于在共享隐状态条件下并行预测连续 $k$ 个条件独立的单碱基词元。配合遍历全部偏移量 $\delta \in \{0, \ldots, k-1\}$ 的分词移位增强，除序列两端边界位点外，聚合后的总单碱基监督信号对分词相位完全不变：

$$
\sum_{\delta=0}^{k-1} \mathcal{L}_\delta(S) = \sum_{t=1}^n \ell(b_t) + E_{\mathrm{bdry}}(S).
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-031-699abe35.png)

**基因组压缩预训练（GCP）与双层语义解耦：** 为使有限的上下文窗口覆盖多个真核功能单元，模型提取每条染色体 $C$ 上的注释功能区间（编码基因座、非编码 **RNA** 与调控区），剔除基因间背景序列后，按原始坐标顺序将各区间序列 $x_i$ 以特殊分隔词元 `<s>` 拼接为连续压缩流：

$$
X_C = x_1 \,\|\, \text{<s>} \,\|\, x_2 \,\|\, \text{<s>} \,\|\, \cdots.
$$

预训练在压缩流上执行密集因果自注意力（屏蔽 `<s>` 位置的损失），在数据层面实现了跨兆碱基功能单元的共现建模与“下一基因预测”。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-032-699abf04.png)

推理时，模型展现出局部语法与全局语义的解耦：提取序列末位内容词元的隐状态可编码正反链方向与密码子相位；而提取序列末端追加的分隔符 `<s>` 隐状态，则会自动坍缩局部链方向与相位差异，聚合出反映物种分类学层级的基因级摘要表征。

<a id="p-004"></a>
### ⚪ **Genos**：基于泛基因组与稀疏混合专家的百万碱基级模型
- **paper**：[**Genos: a human-centric genomic foundation model**](https://doi.org/10.1093/gigascience/giaf132)（*GigaScience* 2025）

**Genos** 基于人类泛基因组单倍型解析组装构建两阶段渐进式语料，并在集成超大基频 **RoPE** 与 **GQA** 的 **Transformer** 解码器中引入稀疏混合专家（**MoE**）架构，实现兼顾人类群体多样性与条件计算效率的百万碱基级（$1\text{ Mb}$）基因组建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-033-699bbadf.png)

**人类泛基因组组装与两阶段渐进式语料：** 模型采用单核苷酸分词，训练语料整合自人类泛基因组参考联盟（**HPRC**）、结构变异联盟（**HGSVC**）的高质量单倍型从头组装及参考基因组。数据流遵循两阶段策略：（1）**第一阶段**将样本切分为 $8\text{k}$ 至 $1\text{M}\text{ bp}$ 四个尺度，在短尺度片段中剔除远离基因边界的基因间区序列，并按长度由短至长渐进训练；（2）**第二阶段（持续预训练）**取消基因间区过滤，将全尺度与正反链样本混合，使模型充分学习远端基因间区、片段重复与转座元件等背景结构。

**稀疏混合专家（MoE）与百万碱基解码器：** 为支撑 $1\text{ Mb}$ 单碱基窗口，注意力层集成基频达 $5\times 10^7$ 的旋转位置编码与分组查询注意力（**GQA**）。前馈网络升级为稀疏 **MoE** 层，路由器根据当前序列上下文为每个词元动态激活 Top-$2$ 个 $\mathrm{SwiGLU}$ 专家子网络，自适应分配算力以区分简单重复区域与复杂调控元件。针对仅有 $4$ 类碱基词元易引发的路由坍缩，训练目标联合优化下一词元预测损失、专家负载均衡损失与路由器对数几率 **Z-loss** 惩罚。下游应用支持挂载一维扩张卷积头回归单碱基 **RNA-seq** 表达轨迹，或将提取的基因表征投影至科学大语言模型中进行遗传疾病跨模态诊断。

<a id="p-005"></a>
### ⚪ **AntigenLM**：全基因组八节段保序预训练的流感演化预测模型
- **paper**：[**AntigenLM: Structure-Aware DNA Language Modeling for Influenza**](https://openreview.net/forum?id=AntigenLM)（*ICLR* 2026）

**AntigenLM** 在长上下文 **GPT-2** 解码器架构下提出两阶段功能单元编码策略，通过在无截断完整甲型流感基因组上执行保序自回归预训练，并结合多时间窗口时序提示与哨兵词元微调，统一实现未来流行株抗原序列预测与病毒亚型判别。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-034-699bcad8.png)

**全基因组八节段保序拼接预训练：** 模型采用单核苷酸分词。预训练阶段严格保留各基因节段编码序列（**CDS**）的完整性，将同一分离株的全部八个基因节段按由长至短的固定顺序（$\mathrm{PB2}\to\mathrm{PB1}\to\mathrm{PA}\to\mathrm{HA}\to\mathrm{NP}\to\mathrm{NA}\to\mathrm{MP}\to\mathrm{NS}$）无间隔拼接为单条连续序列（约 $13\text{k}$ 词元），使自注意力在全基因组尺度捕获聚合酶复合体与表面抗原之间的跨节段共进化及代偿突变信号，优化标准因果语言建模损失 $\mathcal{L}_{\mathrm{CLM}}$。

**时序块提示与哨兵词元引导的演化预测：** 下游适配阶段引入界定功能区边界的哨兵词元（如 `<subtype>`、`<HA>`、`<NA>`、`<sep>`）。在未来优势毒株预测任务中，模型将同一地理区域内连续三个历史时间窗口的血凝素（$\mathrm{HA}$）与神经氨酸酶（$\mathrm{NA}$）序列块与待预测未来时间块拼接为单条时序提示序列：

$$
\begin{aligned}
&\underbrace{\mathrm{block}^{(1)}}_{\text{past}}\;\underbrace{\mathrm{block}^{(2)}}_{\text{past}}\;\underbrace{\mathrm{block}^{(3)}}_{\text{past}}\;\underbrace{\mathrm{block}^{(\star)}}_{\text{future}}, \\ 
&\mathrm{block}^{(i)} = \text{<subtype><HA>}\mathrm{HA}^{(i)}\text{<NA>}\mathrm{NA}^{(i)}\text{<sep>}\quad (i=1,2,3),
\end{aligned}
$$

其中未来目标块 $\mathrm{block}^{(\star)}$ 省略前导 `<subtype>` 词元。推理时输入前三个历史块并从目标 `<HA>` 词元触发自回归解码，直至生成 `<sep>` 终止符，直接输出未来流行株的完整 $\mathrm{HA}$ 与 $\mathrm{NA}$ 序列；而在亚型分类任务中，则直接提取 `<HA>`$\mathrm{HA}$`<NA>`$\mathrm{NA}$`<sep>` 末端 `<sep>` 位置的隐状态接入线性分类头。

<a id="dec-conv"></a>
# 2. 隐式长卷积与混合框架（Convolutional & Hybrid Decoders）

为在保持单核苷酸分辨率的同时突破二次方注意力瓶颈，**隐式长卷积框架（Implicit Long Convolution，如 Hyena）**利用参数化的全局因果卷积核替代自注意力矩阵。给定输入投影序列 $v \in \mathbb{R}^L$ 与由小型神经网络隐式生成的因果滤波器 $h \in \mathbb{R}^L$，其核心操作由下三角 **Toeplitz** 矩阵乘法与逐元素数据门控交替构成，并可通过快速傅里叶变换（**FFT**）在 $\mathcal{O}(L \log_2 L)$ 时间内完成全局上下文聚合：

$$
y_t = (h * v)_t = \sum_{\tau=0}^{t} h_{t-\tau} v_\tau = \mathrm{iFFT}\big(\mathrm{FFT}(h) \odot \mathrm{FFT}(v)\big)_t.
$$

尽管纯隐式长卷积在百万碱基尺度上具备显著的吞吐量优势，但在需要精确跨位置信息检索（Associative Recall）的复杂序列生成中仍存在容量瓶颈。因此，新一代全基因组生成基础模型（如 **Evo** 与 **Evo 2**）演进为**多尺度卷积—注意力混合架构（StripedHyena）**：以占多数比例的多尺度输入依赖卷积层高效处理局部基序与长程背景信号，同时周期性交错插入少量配备旋转位置编码（**RoPE**）的自注意力层，兼顾百万碱基上下文效率与高保真序列设计。

<a id="p-007"></a>
### ⚪ **HyenaDNA**：基于隐式长卷积与数据门控的单核苷酸模型
- **paper**：[**HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793246797c7-Abstract-Conference.html)（*NeurIPS* 2023）

**HyenaDNA** 以隐式长卷积与数据控制门控构成的 **Hyena** 算子完全替代稠密自注意力，结合序列长度渐进预热与软提示上下文学习，在 $\mathcal{O}(L\log_2 L)$ 复杂度下实现百万词元级单碱基自回归建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-037-680afb15.png)

**隐式长卷积与数据门控 Hyena 算子：** 模型采用单核苷酸分词。给定长度为 $L$ 的输入序列，单通道 $2$ 阶 **Hyena** 算子先通过线性层与短卷积将输入映射为三个分支投影 $x_1, x_2, v \in \mathbb{R}^L$，随后交替施加逐元素数据门控与全局因果长卷积：

$$
(x_1, x_2, v) \mapsto H(x_1, x_2)v, \qquad H(x_1, x_2) = D_{x_2} T_h D_{x_1},
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-036-680afb8f.png)

其中 $D_{x_1}, D_{x_2}$ 为以 $x_1, x_2$ 为对角线的门控矩阵，$T_h$ 为由因果长卷积核 $h$ 构造的下三角 **Toeplitz** 矩阵。为使参数量与序列长度 $L$ 解耦，滤波器 $h_t = \gamma_\theta(t)$ 由接收位置编码、带正弦激活与窗口调制的小型 **MLP** 隐式生成，并通过 **FFT** 高效计算。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-035-680afb55.png)

**长度渐进预热与软提示上下文学习：** 预训练在人类参考基因组上优化下一核苷酸预测损失，并采用序列长度预热调度：从极短窗口（$L=64$）出发分阶段将序列长度逐级翻倍至百万碱基，以稳定早期梯度方差。依托宽裕的上下文窗口，模型支持免更新主干权重的**可调软提示（Soft Prompting）**：给定长度为 $T$ 的下游任务序列 $x_p$，在嵌入空间向序列前端拼接 $N$ 个可学习软提示向量 $\theta \in \mathbb{R}^{N \times D}$：

$$
x \leftarrow \mathrm{concat}[\mathrm{embed}(x_p),\, \theta], \qquad x \in \mathbb{R}^{D \times (T+N)}.
$$

冻结预训练主干，仅优化软提示参数 $\theta$，并直接借用单碱基字符（如 **A** 与 **N**）映射下游分类标签。

<a id="p-008"></a>
### ⚪ **Evo**：长卷积与注意力混合架构的全基因组生成基础模型
- **paper**：[**Sequence modeling and design from molecular to genome scale with Evo**](https://www.science.org/doi/10.1126/science.ado9336)（*Science* 2024）

**Evo** 采用将模态范式 **Hyena** 算子与旋转位置编码自注意力交错堆叠的 **StripedHyena** 混合架构，在大规模原核与噬菌体全基因组语料（**OpenGenome**）上以单核苷酸分辨率训练，实现了跨 **DNA**、**RNA** 与蛋白质模态的零样本功能预测、多组件分子复合物可控生成以及百千碱基尺度的基因组设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-038-680b267c.png)

**StripedHyena 混合架构与两阶段上下文扩展：** 模型采用单核苷酸分词。主干网络由数据控制 **Hyena** 层与等间隔插入的少量（占比约 $10\%$）**RoPE** 多头自注意力层交错构成，兼顾长卷积的滤波效率与自注意力的长程精确召回。在 **Hyena** 层中，输入经线性投影与短卷积生成查询、键、值流 $q, k, v$，随后通过模态范式状态空间隐式生成全局滤波器 $h = \mathrm{ImplicitFilter}(L)$，依次执行逐元素门控与 **FFT** 长卷积：

$$
v \leftarrow k \odot v, \qquad v_t \leftarrow \mathrm{FFTConv}(h, v)_t, \qquad y \leftarrow q \odot v.
$$

预训练语料整合自细菌、古菌、原核病毒及质粒序列（严格剔除感染真核宿主的病毒）。训练分两阶段进行：先在 $8\text{k}$ 窗口内完成基础自回归预训练，再通过对 **RoPE** 施加缩放因子为 $16$ 的线性位置插值，将上下文窗口平滑扩展至 $131\text{k}$。

**跨中心法则零样本推断与多组件系统生成：** 依托原核基因组中操纵子共定位与跨模态共进化特性，模型直接计算突变型与野生型序列的自回归对数似然比 $\log [p(x_{\mathrm{mut}}) / p(x_{\mathrm{wt}})]$，即可零样本评估蛋白质、非编码 **RNA** 突变效应及核糖体结合位点驱动的表达水平；通过在基因起始端插入提前终止密码子并引入长侧翼上下文，可根据似然下降幅度执行全基因组必需基因筛查。在生成任务中，模型经条件控制符微调后可协同设计序列发散但结构匹配的 **CRISPR-Cas** 蛋白—**RNA** 复合物，或直接从物种提示符自回归生成保持天然编码密度与操纵子排布的数百 **kb** 基因组序列。

<a id="p-009"></a>
### ⚪ **Evo 2**：多尺度卷积混合架构的跨生命域基因组基础模型
- **paper**：[**Genome modelling and design across all domains of life with Evo 2**](https://doi.org/10.1038/s41586-026-10176-5)（*Nature* 2026）

**Evo 2** 将训练语料扩展至跨越全生命域的 **OpenGenome2**，提出由短显式、中正则化与长隐式卷积及旋转注意力交错构成的 **StripedHyena 2** 多重混合架构，在单核苷酸分辨率与百万碱基（$1\text{ Mb}$）窗口下统一实现全谱系临床变异零样本评估、稀疏自编码器机制可解释性解析、真核染色体级序列生成以及推理时引导的染色质可及性设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-039-680b357e.png)

**StripedHyena 2 多重混合架构与百万碱基训练：** 不同于前代仅使用单一长隐式卷积，**StripedHyena 2** 将输入依赖卷积拆解为针对不同距离尺度的三类算子：短显式卷积（**SE**）、中距离正则化卷积（**MR**）与长隐式卷积（**LI**），并以 **SE-MR-LI** 组合块与 **RoPE** 自注意力层周期性交错堆叠。训练遵循两阶段范式：（1）**短窗口功能区富集预训练**在 $8\text{k}$ 窗口内优先采样真核启动子、外显子、剪接位点及基因中心窗口，学习核心遗传与调控语法；（2）**多阶段长上下文中期训练（Midtraining）**逐步将窗口从 $8\text{k}$ 分级翻倍扩展至 $1\text{M}$ 单核苷酸词元，纳入完整染色体级长序列以学习长程基因组组成规律。

**全谱系零样本变异评估与稀疏自编码器（SAE）可解释性：** 模型无需多序列比对（**MSA**），直接计算自回归对数似然变化（$\Delta\text{Likelihood}$），即可统一评估编码区与非编码区的单碱基变异（**SNV**）及插入、缺失等复杂变异。为解析内部黑盒表征，研究在模型中间层激活上无监督训练 **Batch-TopK** 稀疏自编码器（**SAE**），将稠密隐状态分解为高维稀疏潜在特征；无需先验标签，**SAE** 的潜在维度即可自发对齐外显子—内含子精确剪接边界、移码突变响应、转录因子结合基序、蛋白质二级结构以及前噬菌体与 **CRISPR** 间隔序列等生物学特征。

**跨生命域基因组生成与推理时引导的表观遗传设计：** 模型可根据提示生成保持天然基因共线性与四核苷酸使用偏差的人类线粒体基因组、最小原核基因组及酵母真核染色体序列。针对预训练未显式包含的表观遗传状态，**Evo 2** 作为先验提议分布与序列到功能预测器（**Enformer / Borzoi**）结合，构建推理时引导生成框架：按每 $128\text{ bp}$ 块迭代自回归采样，在每步利用预测器评估候选序列在目标细胞系中的染色质可及性轨迹，通过束搜索保留与目标开闭峰值模式匹配度最高的候选块，实现哺乳动物染色质可及性图谱的从头可编程设计。

<a id="dec-ssm"></a>
# 3. 状态空间混合框架（State-Space Hybrid Decoders）

**状态空间解码器（State-Space Decoders）**利用选择性状态空间模型（如 **Mamba2**）的线性递归形式，在自回归生成中以 $\mathcal{O}(1)$ 的常数级单步显存与计算开销维护无限长历史上下文，消除了 **Transformer** 解码器随生成长度线性膨胀的 **KV Cache** 瓶颈。基于结构化状态空间对偶性（**Structured State Space Duality, SSD**），其线性状态递推可精确等价为半可分矩阵（Semiseparable Matrix）对偶变换：

$$
h_t = a_t h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t \quad \Longleftrightarrow \quad y = M x, \quad M_{ij} = \mathbb{I}(j \le i)\, C_i^\top A_{j:i+1} B_j.
$$

然而，两项固有限制制约了纯状态空间解码器在基因组中的通用性：一是固定维度的递归隐状态对超长距离下的单核苷酸精细互作存在有损压缩，需交错混入少量自注意力层以恢复精确上下文检索；二是单向因果掩码使词元表征缺失右侧下游序列信息，在迁移至启动子或增强子分类等判别式理解任务时存在结构性劣势，需配合序列重放（如回声嵌入）等机制补全双向视野。

<a id="p-010"></a>
### ⚪ **HybriDNA**：结构化状态空间对偶与因果注意力混合的长程模型
- **paper**：[**HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model**](https://arxiv.org/abs/2502.10807)（*ICLR 2025 Workshop*）

**HybriDNA** 将 **Mamba2** 结构化状态空间对偶块与标准 **Transformer** 因果自注意力块按 $7:1$ 比例交错堆叠，并结合序列重复的回声嵌入（**Echo Embedding**）判别式微调与指令提示生成式微调，在 $131\text{ kb}$ 单核苷酸窗口下统一支持长程基因组理解与顺式调控元件设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-040-6891c3a3.png)

**7:1 交错混合架构与 SSD 算子：** 模型采用单核苷酸分词，每 $8$ 层由 $7$ 个 **Mamba2** 块与 $1$ 个标准 **Transformer** 解码器块交错构成；由于首层 **Mamba2** 的短卷积已具备隐式位置感知能力，网络完全省略了显式位置编码。**Mamba2** 内部的多头结构化状态空间对偶（**SSD**）层将状态转移矩阵简化为标量与单位阵的乘积 $A_t = a_t I$，其半可分矩阵可分解为低秩项与下三角项之和 $M = U V^\top + K$，从而在 $\mathcal{O}(NL)$ 线性复杂度下利用矩阵乘法单元完成全局上下文聚合。模型在跨物种基因组语料上通过多阶段序列长度预热（从 $8\text{k}$ 扩展至 $131\text{k}$）优化下一词元预测损失。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-041-6891c3ef.png)

**回声嵌入（Echo Embedding）判别式微调：** 为弥补因果解码器无法将右侧未来词元信息编码至当前位置的缺陷，模型在下游分类任务中引入回声嵌入机制。给定输入序列 $x$，将其沿序列维度直接重复拼接构造回声输入 $x_{\mathrm{echo}} = [x, x]$；在单向因果掩码下，后半段重复序列中的每个词元均可完整关注前半段中的全部原始序列上下文。模型仅提取后半段重复序列对应的末层隐状态执行均值池化得到全局表征 $h_\theta(x_{\mathrm{echo}})$，再送入线性分类头：

$$
P(y \mid x) = \mathrm{Softmax}\big(h_\theta(x_{\mathrm{echo}})W + b\big).
$$

**任务提示引导的生成式微调：** 在顺式调控元件设计任务中，模型将编码目标活性等级的离散标签映射为专用提示词元 $z_0, \ldots, z_{k-1}$，拼接于真实调控序列 $x_{0:T}$ 前端，最小化条件自回归损失：

$$
\mathcal{L}_{\mathrm{generative}}(\theta) = -\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid z_0, \ldots, z_{k-1}, x_0, \ldots, x_{t-1}).
$$

推理时以高活性标签提示词元为前缀，通过束搜索即可解码生成目标活性的合成调控序列。

<a id="dec-diffusion"></a>
# 4. 离散扩散生成框架（Discrete Diffusion Decoders）

自回归解码器强制要求按自左向右的固定顺序逐词元生成序列，但真实的 **DNA** 顺式调控元件往往受双向上下文、回文对称性（如部分转录因子结合位点）以及全局 **GC** 含量平衡的无向约束。**离散掩码扩散语言模型（Discrete Masked Diffusion Language Models, MDLM）**将吸收态（Absorbing-state）连续时间马尔可夫链引入双向编码器主干，打破了因果生成顺序的限制。

设干净序列为 $x_0$，前向加噪过程在连续时间步（即掩码率）$t \in [0, 1]$ 下，独立地以概率 $t$ 将各位置词元替换为吸收态掩码符号 `[M]` 得到受损序列 $x_t$。反向生成过程则利用双向去噪网络 $p_\theta(x_0 \mid x_t)$ 从全掩码序列 $x_1$ 出发，分多步并行去掩码恢复完整序列。其训练目标等价于在连续掩码率 $t \sim \mathcal{U}(0, 1)$ 上对掩码位置交叉熵做 $1/t$ 加权积分，构成了真实数据对数似然的严格变分下界（**ELBO**）：

$$
-\log p_\theta(x_0) \le \mathbb{E}_{t \sim \mathcal{U}(0,1),\, x_t \sim q(\cdot \mid x_0)}\!\left[\frac{1}{t}\sum_{i:\, x_t^i = \mathrm{[M]}} -\log p_\theta(x_0^i \mid x_t)\right].
$$

该范式将固定掩码率（如 $t=0.15$）的 **BERT** 式表征学习视为其单一时间切片特例，从而在同一双向网络中自然统一了零掩码（$t=0$）表征提取与非顺序并行序列生成。

<a id="p-011"></a>
### ⚪ **D3LM**：连续掩码率加权去噪的离散扩散基因组语言模型
- **paper**：[**D3LM: A Discrete DNA Diffusion Language Model for Bidirectional DNA Understanding and Generation**](https://arxiv.org/abs/2603.01780)（*ICLR 2026 Workshop*）

**D3LM** 在双向 **Transformer** 架构下引入离散掩码扩散语言建模范式，通过连续掩码率加权交叉熵目标，将双向基因组表征学习与非顺序迭代去噪生成统一于单一模型。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-089-d3lm.png)

**非重叠 $6$-mer 分词与连续掩码率扩散训练：** 模型复用配备 **RoPE** 与 **SwiGLU** 的 **NT-v2** 双向 **Transformer** 主干，采用非重叠 $6$-**mer** 分词。给定长度为 $L$ 的干净词元序列 $x_0 \sim q(x)$，前向加噪过程从均匀分布 $t \in [0, 1]$ 中采样掩码率，将每个位置独立以概率 $t$ 替换为掩码词元 `[M]` 生成受损序列 $x_t$。反向过程通过双向网络同步预测所有被遮蔽位点的原始词元，优化 $1/t$ 加权交叉熵变分上界：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\!\left[\frac{1}{t}\sum_{i=1}^L \mathbb{I}[x_t^i = \mathrm{[M]}]\log p_\theta(x_0^i \mid x_t)\right].
$$

**逆向去掩码生成采样：** 在从头序列生成阶段，模型将逆向去噪离散化为 $T$ 个时间步，从全掩码序列 $x_1 = (\mathrm{[M]}, \ldots, \mathrm{[M]})$ 出发迭代恢复序列。在每一步 $t$，记当前处于掩码状态的位点集合为 $\mathcal{M}_t$，模型对所有 $i \in \mathcal{M}_t$ 预测带温度 $\tau$ 缩放的条件分布 $p_\theta^{\mathrm{temp}}(x_0^i \mid x_t) \propto p_\theta(x_0^i \mid x_t)^{1/\tau}$，并按照预设调度（均匀随机或基于置信度/信息熵）从 $\mathcal{M}_t$ 中选取子集 $\mathcal{U}_t \subset \mathcal{M}_t$ 执行去掩码转移：

$$
x_{t-1}^i = \begin{cases} \text{sample from } p_\theta^{\mathrm{temp}}(x_0^i \mid x_t), & \text{if } i \in \mathcal{U}_t, \\ \mathrm{[M]}, & \text{if } i \in \mathcal{M}_t \setminus \mathcal{U}_t, \\ x_t^i, & \text{otherwise}. \end{cases}
$$

通过逐步并行揭露并协同修正位点，模型无需遵循自左向右的因果顺序即可满足全局调控约束。

**零掩码双向表征提取与微调：** 由于训练覆盖了 $t \to 0$ 的全连续谱掩码水平，推理时可直接将未加噪的原始序列（设 $t=0$）输入双向主干提取无偏移的上下文表征 $h(x) = \mathrm{D3LM}_\theta(x, t=0)$，配合 **LoRA** 与轻量级任务头即可进行下游判别式微调。
