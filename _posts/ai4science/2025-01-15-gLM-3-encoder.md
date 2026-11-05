---
layout: post
title: '基因组基础模型(Genome Foundation Model) III：编码器框架'
date: 2025-01-15
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm3-encoder-000-cover.png'
tags: AI4S
---

> Genome Foundation Model part III: Encoder Architectures.

**基因组基础模型（Genome Foundation Model, gFM）**，又称**基因组语言模型（Genomic Language Model, gLM）**，将 **DNA** 序列映射为长度为 $T$ 的离散词元序列 $z_{1:T}$，通过在大规模无标注基因组语料上拟合边缘或条件概率分布 $p_\theta(z)$，学习具备跨任务迁移能力的通用基因组表征。

本文是**基因组基础模型**系列的第三篇。全系列按“**分词 → 预训练 → 编码器 → 解码器 → 多模态**”的脉络组织：
- [I. DNA 分词策略](https://0809zheng.github.io/2025/01/01/gLM-1-token.html)：如何将碱基序列切分为词元。
- [II. 预训练策略](https://0809zheng.github.io/2025/01/08/gLM-2-object.html)：如何构造自监督信号从无标注基因组中学习。
- **III. 编码器框架**：模型如何在长序列上混合上下文信息。
- [IV. 解码器框架](https://0809zheng.github.io/2025/01/22/gLM-4-decoder.html)：模型如何评估序列似然并生成新序列。
- [V. 多模态模型](https://0809zheng.github.io/2025/01/29/gLM-5-multimodal.html)：如何联合建模基因组序列、其他生物分子与人类自然语言。

真核基因组序列跨越 $10^2$ 至 $10^9\text{ bp}$ 量级，远端增强子与靶基因启动子可相距数十至数百 **kb**。**编码器框架（Encoder Architecture）**的设计主线是：在维持单碱基级分辨率与双向上下文可见性的同时，以更低复杂度的序列混合算子换取更长的有效感受野，并将 **DNA** 双链反向互补（**Reverse Complement, RC**）对称性等物理先验嵌入网络结构。

- **1. Transformer 框架（Transformer Architectures）**
  - [DNABERT：重叠 $k$-mer 分词的基因组双向编码器](#p-001)
  - [DNABERT-2：字节对编码与线性偏置注意力的多物种编码器](#p-002)
  - [Nucleotide Transformer：跨物种与群体多态性预训练的规模化编码器](#p-005)
  - [NTv3：卷积-Transformer 混合 U-Net 与功能-生成联合建模](#p-006)
  - [dnaGrinder：序列长度预热与双亲本变异去重语料的轻量编码器](#p-dnagrinder)
  - [AIDO.DNA：基于单核苷酸分词与依赖图谱的规模化多物种编码器](#p-007)
  - [DeepGene：基于泛基因组变异图序列化的双向编码器](#p-deepgene)
  - [GENA-LM：块稀疏注意力与循环记忆机制的长序列编码器](#p-008)
  - [ModernGENA：融合局部-全局交替注意力与无填充打包的高效基线](#p-009)
  - [TrinityDNA：多尺度沟槽卷积与滑动多窗口注意力的长序列编码器](#p-010)
- **2. 卷积网络框架（Convolutional Architectures）**
  - [GPN：基于指数循环扩张卷积的无监督基因组编码器](#p-012)
  - [ConvNova：非对称双分支门控扩张卷积的基因组编码器](#p-016)
  - [SPU-DNA：协同可塑性单元与时频双域多分辨率建模](#p-spu)
- **3. 状态空间框架（State-Space Architectures）**
  - [Caduceus：反向互补等变的双向选择性状态空间模型](#p-017)
  - [PlantCaduceus：面向被子植物基因组的跨物种等变状态空间模型](#p-018)
  - [PlantCAD2：基于结构化状态空间对偶的长上下文植物基因组编码器](#p-019)
  - [Wisteria：基于 BiMamba 与傅里叶注意力的多尺度编码器](#p-wisteria)
  - [eccDNAMamba：超长环状 DNA 的循环增强双向 Mamba-2 编码器](#p-eccdnamamba)
- **4. 循环与记忆网络框架（Recurrent & Memory Architectures）**
  - [Bio-xLSTM (DNA-xLSTM)：指数门控与矩阵记忆的基因组扩展长短期记忆网络](#p-020)
  - [CrossDNA：跨链动态交互与自蒸馏的双链循环记忆模型](#p-crossdna)
  - [Double-Take：局部注意力与双向快权重记忆混合的长上下文架构](#p-011)
  - [Mnemos-DNA：门控 Delta 规则线性注意力与 $k$-mer 寻址快权重记忆模型](#p-mnemos)

<a id="enc-transformer"></a>
# 1. Transformer 框架（Transformer Architectures）

**Transformer** 编码器通过自注意力在单层内直接建立任意两个位置间的交互通道，契合三维染色质折叠下远端调控元件跨越长距离发生物理接触的生物学特性（$B \in \mathbb{R}^{T \times T}$ 为相对位置偏置或稀疏掩码矩阵）：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + B\right)V.
$$

针对标准自注意力的 $\mathcal{O}(T^2)$ 复杂度与长序列注意力过平滑瓶颈，基因组 **Transformer** 编码器沿着三条路径演进：（1）通过 $k$-**mer**、**BPE** 分词或卷积 **U-Net** 下采样缩短进入注意力层的词元数 $T$；（2）引入块稀疏注意力、局部—全局交替滑窗或跨片段循环记忆以降至近线性复杂度；（3）采用旋转位置编码（**RoPE**）或线性距离偏置（**ALiBi**）赋予模型测试时的长度外推能力。

<a id="p-001"></a>
### ⚪ **DNABERT**：重叠 $k$-mer 分词的基因组双向编码器
- **paper**：[**DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome**](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)（*Bioinformatics* 2021）

**DNABERT** 将 **BERT** 架构直接迁移至基因组领域，通过重叠 $k$-**mer** 分词将局部碱基上下文融入单个词元，并配合连续跨度掩码在人类参考基因组上提取双向上下文表征。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-002-6808ae96.png)

**重叠 $k$-mer 分词与连续跨度掩码：** 模型采用步长为 $1$ 的滑动窗口将长度为 $L$ 的序列转化为 $T = L - k + 1$ 个重叠 $k$-**mer** 词元（$k \in \{3, 4, 5, 6\}$，词表含 $4^k$ 种排列及 $5$ 个控制符），与可学习绝对位置嵌入相加后输入双向 **Transformer**。由于重叠 $k$-**mer** 与左右相邻词元共享 $k-1$ 个碱基，预训练每次连续遮蔽 $k$ 个相邻词元以阻断局部信息泄漏，在掩码集 $\mathcal{M}$ 上最小化多分类交叉熵损失：

$$
\mathcal{L}_{\mathrm{MLM}} = -\sum_{i \in \mathcal{M}}\sum_{c=1}^{\vert\mathcal{V}\vert} y'_{i,c} \log y_{i,c}.
$$

下游微调利用 **[CLS]** 隐状态执行序列分类，并可通过聚合末层注意力权重定位功能基序（**DNABERT-viz**）；针对超长序列，**DNABERT-XL** 将序列分块独立编码后拼接送入顶层分类器。

<a id="p-002"></a>
### ⚪ **DNABERT-2**：字节对编码与线性偏置注意力的多物种编码器
- **paper**：[**DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome**](https://openreview.net/forum?id=oMLQB4EZE1)（*ICLR* 2024）

**DNABERT-2** 在双向 **Transformer** 中引入字节对编码（**BPE**）可变长分词与线性注意力距离偏置（**ALiBi**），消除重叠 $k$-**mer** 的长度膨胀与信息泄漏，在多物种基因组语料上实现高效表征学习。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-004-6808b649.png)

**BPE 可变长分词与线性偏置注意力（ALiBi）：** 模型迭代合并高频相邻片段构建容量为 $4{,}096$ 的无重叠 **BPE** 词表（平均将序列长度压缩约 $5$ 倍），可直接以 $15\%$ 比例独立随机遮蔽词元优化 **MLM** 损失。同时，编码器移除定长绝对位置嵌入，在注意力得分中直接减去与相对距离成正比的惩罚项（$m_h$ 为预设的头特异斜率），支持模型在推理时直接外推处理更长序列：

$$
\mathcal{L}_{\mathrm{MLM}}=-\sum_{i\in\mathcal{M}}\log p_\theta(x_i\mid x_{\setminus\mathcal{M}}), \qquad
e_{ij}^{(h)}=\frac{q_i\cdot k_j}{\sqrt{d_k}}-m_h\vert i-j\vert.
$$

<a id="p-005"></a>
### ⚪ **Nucleotide Transformer**：跨物种与群体多态性预训练的规模化编码器
- **paper**：[**Nucleotide Transformer: building and evaluating robust foundation models for human genomics**](https://www.nature.com/articles/s41592-024-02523-z)（*Nature Methods* 2025）

**Nucleotide Transformer（NT）** 采用非重叠 $6$-**mer** 分词与随机阅读框偏移增强，系统整合人类参考基因组、千人基因组群体变异及跨物种基因组语料进行十亿级参数规模化预训练。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-006-6809df29.png)

**群体变异注入、架构演进与单碱基解包：** 模型以步长为 $6$ 的非重叠窗口切分序列（遇未知碱基 **N** 或尾部残片回退至单字符），并随机偏移起始点以增强相位鲁棒性。针对千人基因组（**1000G**）分相数据，训练流动态替换个体的 **SNP/Indel** 变异，并移除 **MLM** 中的“$10\%$ 随机替换”以防干扰低频多态性似然估计。第二代 **NT-v2** 引入 **RoPE** 与无偏置 **SwiGLU** 将窗口扩展至 $12\text{ kb}$。下游针对单碱基分辨率任务直接将单个 $6$-**mer** 嵌入线性解包为 $6$ 个位置的分布；零样本变异评估则通过参考与突变序列的嵌入余弦距离或掩码重建损失差量化突变破坏性。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-007-6809df74.png)

<a id="p-006"></a>
### ⚪ **NTv3**：卷积-Transformer 混合 U-Net 与功能-生成联合建模
- **paper**：[**A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction**](https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1)（*bioRxiv* 2025）

**NTv3** 采用多尺度卷积—**Transformer** 混合 **U-Net** 架构，结合自适应层归一化（**AdaLN**）物种调制与掩码扩散建模，在 $1\text{ Mb}$ 窗口内统一单碱基表征学习、多物种功能轨道预测与条件序列生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-045-ntv3.png)

**多尺度 U-Net 骨架与功能联合后训练：** 模型以单核苷酸为输入，编码端经 $7$ 级残差一维卷积将长度压缩 $128$ 倍，瓶颈层由带 **RoPE/SwiGLU** 的 **Transformer** 捕获长程依赖，解码端经 $7$ 级上采样与跳跃连接恢复单碱基分辨率，并通过物种嵌入 $s$ 调制 $\mathrm{AdaLN}(x,s)=(1+\gamma(s))\odot\mathrm{LN}(x)+\beta(s)$。在完成 $1\text{ Mb}$ 长度课程 $\mathcal{L}_{\mathrm{MLM}}$ 预训练后，后训练阶段联合优化元件分割 Focal 损失及功能轨道的总覆盖度泊松损失与局部轮廓多项式损失（真实信号 $x_i$、预测 $y_i$）：

$$
\mathcal{L}_{\mathrm{Poisson}}=\sum_i y_i-\left(\sum_i x_i\right)\log\left(\sum_i y_i\right),\qquad
\mathcal{L}_{\mathrm{multinomial}}=-\sum_i x_i\log\left(\frac{y_i}{\sum_j y_j}\right).
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-047-ntv3-mddm.png)

**掩码扩散条件生成：** 基于双向主干，模型可进一步通过掩码扩散语言建模（**MDLM**，掩码调度 $\alpha_t=1-t$）微调为生成模型，并结合强度为 $\gamma>1$ 的无分类器引导（**CFG**）按目标活性 $y_{\mathrm{target}}$ 从头设计调控元件：

$$
\mathcal{L}_{\mathrm{MDLM}}=\int_0^1\frac{-\alpha'_t}{1-\alpha_t}\,\mathbb{E}_{\mathbf{x},\mathbf{z}_t}\!\left[\sum_{\ell=1}^{L}-\log\left\langle\mathbf{x}_\theta^\ell(\mathbf{z}_t),\mathbf{x}^\ell\right\rangle\right]\mathrm{d}t,\quad
p_\gamma(\mathbf{z}_s\mid\mathbf{z}_t)\propto p_\theta(\mathbf{z}_s\mid\mathbf{z}_t,y_{\mathrm{target}})^\gamma\,p_\theta(\mathbf{z}_s\mid\mathbf{z}_t,y_{\mathrm{bg}})^{1-\gamma}.
$$

<a id="p-dnagrinder"></a>
### ⚪ **dnaGrinder**：序列长度预热与双亲本变异去重语料的轻量编码器
- **paper**：[**dnaGrinder: A Lightweight and High-Capacity Genomic Foundation Model**](https://arxiv.org/abs/2409.15697)（*arXiv* 2024）

**dnaGrinder** 在集成 **FlashAttention-2**、**ALiBi** 与 **SwiGLU** 的轻量双向编码器中，引入流式 **BPE** 分词、序列长度预热（**SLW**）及融合双亲本变异的非重复区语料，支持从 $12\text{ kbp}$ 预训练窗口零样本外推至 $120\text{ kbp}$ 以上。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-097-dnagrinder.png)

**双亲本变异语料与序列长度预热（SLW）：** 预处理利用 **RepeatMasker** 滤除低信息重复区后截取 $12\text{ kbp}$ 片段，注入千人基因组个体的 **SNV/INDEL/SV** 构造父源与母源平行序列，并通过流式贪婪策略构建 $4{,}096$ 词表。由于分词后长度波动较大，**SLW** 将样本按词元数升序排列（简单重复序列被压缩得更短），在减少填充浪费的同时自然形成由简入繁的课程学习；配合无 **Dropout** 的 **ALiBi** 注意力：

$$
\mathrm{Attention}(q_i, K, V) = \mathrm{Softmax}\!\big(q_i K^\top + m \cdot [-(i-1), \ldots, -2, -1, 0]\big)V,
$$

使模型无需微调位置参数即可直接外推处理十倍于预训练长度的超长序列。

<a id="p-007"></a>
### ⚪ **AIDO.DNA**：基于单核苷酸分词与依赖图谱的规模化多物种编码器
- **paper**：[**Accurate and General DNA Representations Emerge from Genome Foundation Models at Scale**](https://openreview.net/forum?id=Kis8tVUeNi)（*NeurIPS 2024 Workshop*）

**AIDO.DNA** 在配备 **RoPE** 与 **SwiGLU** 的十亿参数稠密双向 **Transformer** 下直接采用单核苷酸分词，在不损失单碱基分辨率的前提下学习跨物种通用表征与核苷酸依赖图谱。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-014-699bb631.png)

**无监督核苷酸依赖图谱与条件扩散微调：** 模型以单字符 **A, T, C, G, N** 在跨物种切片上执行 $15\%$ 掩码预训练。冻结权重后，可通过逐位点计算机饱和突变构建二维依赖图谱：将位置 $i$ 突变为 $q$ 后，位置 $j$ 对 $i$ 的依赖强度定义为遍历所有突变 $q$ 与候选碱基 $k$ 时的最大对数优势比（$\widehat{\mathrm{odds}}=p/(1-p)$）绝对变化量：

$$
e_{i,j}=\max_{k,q\in\lbrace \mathrm{A,T,C,G}\rbrace }\left\vert\log_2\frac{\widehat{\mathrm{odds}}(n_j=k\mid n_1,\ldots,n_i=q,\ldots,n_L)}{\widehat{\mathrm{odds}}(n_j=k\mid n_1,\ldots,n_L)}\right\vert.
$$

下游支持先在无标签测试序列上持续 **MLM** 预训练以对齐分布偏移，或挂载轻量条件扩散头按目标表达量生成合成启动子。

<a id="p-deepgene"></a>
### ⚪ **DeepGene**：基于泛基因组变异图序列化的双向编码器
- **paper**：[**DeepGene: An Efficient Foundation Model for Genomics Based on Pan-Genome Graph Transformer**](https://doi.org/10.1109/TCBBIO.2025.3614354)（*IEEE TCBB* 2025）

**DeepGene** 引入人类泛基因组 **Minigraph** 变异图表示，通过有向无环图（**DAG**）去环、拓扑位置递归编码与子图序列化，在 **RoPE** 双向编码器上消除群体线性序列拼接的高冗余。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-096-minigraph.png)

**Minigraph 去环与拓扑位置 RoPE 编码：** 模型基于 **HPRC** 的 **Minigraph** 变异图（节点为非重叠序列，有向边连接不同单倍型路径），先用强连通分量算法去环转化为严格 **DAG**，再在节点内执行 **BPE** 分词。记节点 $i$ 的前驱集为 $P$、节点 $k$ 的词元数为 $\mathrm{len}(k)$，节点 $i$ 内第 $j$ 个词元的全局拓扑位置递归定义为：

$$
\mathrm{position}(i, j) = \begin{cases} 
\min_{k \in P} \big(\mathrm{position}(k, 0) + \mathrm{len}(k)\big), & j = 0, \\
\mathrm{position}(i, 0) + j, & j \neq 0,
\end{cases} \qquad
q_m^\top k_n = x_m^\top W_q R_{m-n}^d W_k x_n.
$$

词元按拓扑位置升序切分为子图样本并施加 **RoPE** 旋转，使注意力内积直接编码泛基因组图中的相对拓扑距离 $m - n$。

<a id="p-008"></a>
### ⚪ **GENA-LM**：块稀疏注意力与循环记忆机制的长序列编码器
- **paper**：[**GENA-LM: a family of open-source foundational DNA language models for long sequences**](https://academic.oup.com/nar/article/53/2/gkae1310/7954523)（*Nucleic Acids Research* 2025）

**GENA-LM** 在双向编码器中结合 $32\text{k}$ 词表 **BPE** 分词、**BigBird** 块稀疏注意力与循环记忆机制（**RMT**），将有效上下文感受野从数千碱基扩展至数十万碱基。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-009-6809f3a9.png)

**块稀疏注意力与跨片段循环记忆传递：** 预训练语料融合千人基因组与 **gnomAD** 单倍型变异的 **T2T** 参考基因组，主干采用结合局部滑窗、随机块与全局词元的 **BigBird** 线性稀疏注意力。为进一步突破单次窗口上限，模型引入 **RMT** 将超长序列切分为连续片段 $S^{(k)}$，令前一片段输出的记忆隐状态 $M^{(k-1)}$ 作为下一片段的前缀输入，通过 **BPTT** 实现跨片段递推（针对剪接位点等边界敏感任务，微调时对左、中、右区域独立分词再拼接以防词元跨界）：

$$
[H^{(k)},M^{(k)}]=\mathrm{GENA\text{-}LM}([M^{(k-1)},S^{(k)}]).
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-010-6809f3ce.png)

<a id="p-009"></a>
### ⚪ **ModernGENA**：融合局部-全局交替注意力与无填充打包的高效基线
- **paper**：[**Back to BERT in 2026: ModernGENA as a Strong, Efficient Baseline for DNA Foundation Models**](https://www.biorxiv.org/content/10.64898/2026.04.21.719816v1)（*ICLR 2026 Workshop*）

**ModernGENA** 在 **BPE** 双向编码器中系统引入 **ModernBERT** 的局部—全局交替注意力、独立 **RoPE** 与端到端无填充序列打包（**Unpadding**），构建了高吞吐量的基因组表征基线。

**TSS 邻近语料与无填充交替注意力：** 预训练围绕跨脊椎动物每个转录起始位点（**TSS**）截取 $[-16\text{ kbp}, +8\text{ kbp}]$ 窗口以富集顺式调控信号。主干采用每 $3$ 层局部滑窗注意力后插入 $1$ 层全局自注意力的交替模式（配置独立 **RoPE** 基频、**Pre-Norm** 与无偏置 **GeGLU**），并通过端到端无填充动态序列打包仅在有效词元上计算注意力与 **MLM** 损失，消除定长 `[PAD]` 填充带来的算力浪费。

<a id="p-010"></a>
### ⚪ **TrinityDNA**：多尺度沟槽卷积与滑动多窗口注意力的长序列编码器
- **paper**：[**TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling**](https://arxiv.org/abs/2507.19229)（*AAAI* 2026）

**TrinityDNA** 在单核苷酸双向编码器中集成多尺度沟槽卷积、滑动多窗口注意力与门控反向互补机制，配合从原核向真核过渡的两阶段演化训练，实现 $102.4\text{ kb}$ 长序列建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-016-699ab056.png)

**沟槽卷积、多窗口注意力与两阶段演化训练：** 主干包含三项核心算子：（1）**沟槽融合模块（GFM）**受 **DNA** 大小沟空间尺度启发，并行通过核宽 $k\in\{3,5,7\}$ 的一维卷积提取多尺度局部基序；（2）**滑动多窗口注意力（SMWA）**为不同头分配差异化窗口半径 $L_h\in\{128,512,2048,8192\}$，以线性复杂度同步捕获远近端上下文；（3）**门控反向互补（GRC）**令正反链共享编码器并在翻转对齐后经可学习门控逐位点融合：

$$
\mathrm{GrooveFusion}(S)=\sum_{k\in\lbrace 3,5,7\rbrace }\mathrm{GELU}(\mathrm{Conv}_k(S)).
$$

模型先在基因高密度的原核短序列上预训练，再迁移至包含真核生物的 $102.4\text{ kb}$ 跨物种长窗口，并在 $[0,80\%]$ 内动态采样 `<mask>` 实际替换率以缓解预训练—微调偏移。


<a id="enc-conv"></a>
# 2. 卷积网络框架（Convolutional Architectures）

**一维卷积网络（1D CNN）**天然具备**平移等变性（Translation Equivariance）**，与转录因子扫描识别局部基序的生物物理过程高度契合。为在不引入池化下采样（保持单碱基分辨率）的前提下捕获长程依赖，基因组卷积编码器普遍采用扩张率 $d_l$ 随深度指数递增（如 $d_l = 2^{l-1}$）的**扩张卷积（Dilated Convolution）**：

$$
h_t^{(l)} = \sum_{k=0}^{K-1} W_k^{(l)} h_{t - d_l \cdot k}^{(l-1)},
$$

从而在 $\mathcal{O}(L)$ 线性复杂度下呈指数级扩大感受野，并进一步结合**独立门控通路**或**时频双域变换（FFT / 小波）**过滤非编码背景噪声。

<a id="p-012"></a>
### ⚪ **GPN**：基于指数循环扩张卷积的无监督基因组编码器
- **paper**：[**DNA language models are powerful predictors of genome-wide variant effects**](https://www.pnas.org/doi/10.1073/pnas.2311219120)（*PNAS* 2023）

**GPN** 采用纯一维扩张卷积架构，结合功能区富集采样与重复序列损失降权策略，直接从近缘物种未比对参考基因组中无监督学习顺式调控语法与单碱基演化约束。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-018-6809d1d5.png)

**循环扩张卷积与零样本变异评分：** 模型输入单碱基独热编码，卷积扩张率按指数递增至上限后循环往复；预训练在十字花目近缘基因组上混合外显子/启动子富集窗口，并对软屏蔽重复区施加 $15\%$ **MLM** 损失降权。推理时取正反双链输出概率均值，通过遮蔽突变位点计算替代与参考等位基因的边际对数似然比；在 **GWAS** 精细定位中进一步按连锁不平衡系数 $r_{ij}^2$ 加权求和：

$$
\text{GPN Score}=\log\frac{P(\text{Allele}_{\mathrm{ALT}}\mid\text{Context})}{P(\text{Allele}_{\mathrm{REF}}\mid\text{Context})},\quad
\mathrm{GPN}\times\mathrm{LD}_i=-\sum_j\vert\mathrm{GPN}_j\vert\cdot r_{ij}^2.
$$

<a id="p-016"></a>
### ⚪ **ConvNova**：非对称双分支门控扩张卷积的基因组编码器
- **paper**：[**Revisiting Convolution Architecture in the Realm of DNA Foundation Models**](https://openreview.net/forum?id=B07dLVWLyD)（*ICLR* 2025）

**ConvNova** 在单碱基分辨率纯一维扩张卷积架构下引入非对称双分支门控机制（**GCB**），通过独立演化的门控通路实现内容依赖的动态特征筛选。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-019-6808a58f.png)

**非对称双分支门控扩张卷积块（GCB）：** 不同于在单一路径内临时分出门控的常规设计，**GCB** 维护参数独立的特征提取流 $A$ 与门控信号流 $B$：左分支以门控加权特征 $h\odot g$ 残差更新以保留调控基序并抑制低信息背景，右分支则通过残差累加 $g$ 独立维护跨层门控状态：

$$
h=\mathrm{GELU}\big(\mathrm{DilatedConv}_1(\mathrm{LN}(A))\big),\quad
g=\sigma\big(\mathrm{DilatedConv}_2(\mathrm{LN}(B))\big),\quad
A\leftarrow A+h\odot g,\quad B\leftarrow B+g.
$$

<a id="p-spu"></a>
### ⚪ **SPU-DNA**：协同可塑性单元与时频双域多分辨率建模
- **paper**：[**Adaptive DNA Sequence Modeling via Synergistic Plasticity Units**](https://openreview.net/forum?id=coNswqxIlk)（*ICML* 2026）

**SPU-DNA** 将局部可塑性层（**LPL**）、域可塑性层（**DPL**）与显著性可塑性层（**SPL**）组合为单一协同可塑性单元（**SPU**），以 $\mathcal{O}(L\log L)$ 复杂度同步捕获局部基序、长程依赖与核小体周期性信号。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-052-spu.png)

**局部—时频双域提取与显著性调制：** **LPL** 利用核宽 $\{1,3,5,7\}$ 的多尺度卷积与 $1\times1$ 动态权重逐位点聚合局部特征 $F_{\mathrm{lpl}}$。**DPL** 并行包含两条路径：时域路径生成带指数衰减窗的长卷积核 $p$ 并经 **FFT** 执行全局卷积 $\hat F_{\mathrm{lpl}}$；频域路径对 $F_{\mathrm{lpl}}$ 逐级做小波分解 $(P_i^l,P_i^h)=\mathrm{WT}(P_{i-1}^l)$，在子带上执行全局卷积后以可学习尺度 $\gamma_i$ 逆变换（$\mathrm{IWT}$）重构累加；最后由 **SPL** 计算通道门控 $S_c$ 与空间门控 $S_p$ 逐元素调制输出 $F_{\mathrm{out}}=F_{\mathrm{dpl}}\odot S_c\odot S_p$：

$$
\hat F_{\mathrm{lpl}}=\mathrm{CenterCrop}_L\!\left(\mathrm{iFFT}\big(\mathrm{FFT}(F_{\mathrm{lpl}})\odot\mathrm{FFT}(p)\big)\right),\quad
F_{\mathrm{dpl}}=\hat F_{\mathrm{lpl}}+\mathrm{IWT}(\gamma_0\tilde P_1^l,\hat P_1^h).
$$


<a id="enc-ssm"></a>
# 3. 状态空间框架（State-Space Architectures）

**选择性状态空间模型（SSMs，如 Mamba）**通过输入依赖的离散化状态转移矩阵 $\bar{A}_t, \bar{B}_t, C_t$ 执行线性递推，利用并行关联扫描或结构化状态空间对偶（**SSD**）在 $\mathcal{O}(T)$ 线性复杂度下过滤非编码冗余并压缩长程调控信号：

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t.
$$

为适配无方向偏好的双链 **DNA**，基因组状态空间编码器通过正反向权重共享扫描实现双向上下文聚合（**BiMamba**），并通过通道拆分与对偶置换构造严格满足 $\mathcal{F}(\mathrm{RC}(X)) = \mathrm{RC}(\mathcal{F}(X))$ 的**反向互补等变算子**。

<a id="p-017"></a>
### ⚪ **Caduceus**：反向互补等变的双向选择性状态空间模型
- **paper**：[**Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling**](https://proceedings.mlr.press/v235/schiff24a.html)（*ICML* 2024）

**Caduceus** 构建了参数共享的双向组件 **BiMamba** 与反向互补等变组件 **MambaDNA**，在单核苷酸分辨率下实现 $131\text{k}$ 窗口的长程双向等变建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-021-6809d72e.png)

**双向状态空间与反向互补等变算子：** **BiMamba** 共享输入输出投影权重分别沿正向与反转方向扫描序列并翻转相加；**MambaDNA** 进一步将通道维度 $D$ 的隐状态 $X$ 对半切分为两路并共享底层 **BiMamba** 模块 $M_\theta$，对第二路前后施加隐空间反向互补变换 $\mathrm{RC}(\cdot)$（长度轴反转加互补通道置换），严格保证 $\mathrm{MambaDNA}(\mathrm{RC}(X))=\mathrm{RC}(\mathrm{MambaDNA}(X))$：

$$
\mathrm{MambaDNA}(X)=\mathrm{concat}\!\left(M_\theta(X_{1:D/2}),\;\mathrm{RC}\!\left(M_\theta(\mathrm{RC}(X_{D/2+1:D}))\right)\right).
$$

模型提供端到端参数共享等变的 **Caduceus-PS**，以及训练期施加 **RC** 增强、推理期双链平均的 **Caduceus-Ph** 两种变体。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-020-6809d8cb.png)

<a id="p-018"></a>
### ⚪ **PlantCaduceus**：面向被子植物基因组的跨物种等变状态空间模型
- **paper**：[**Cross-species modeling of plant genomes at single-nucleotide resolution using a pretrained DNA language model**](https://www.pnas.org/doi/10.1073/pnas.2421738122)（*PNAS* 2025）

**PlantCaduceus** 在 **Caduceus** 双向等变架构下结合非编码区下采样与重复序列损失降权策略，克服植物基因组高比例转座子噪声，从多物种被子植物中学习单碱基演化约束。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-022-699ab674.png)

**功能区重采样与跨物种零样本迁移：** 模型在横跨禾本科与十字花目的参考基因组上主动下采样非编码区与转座子重复样本，聚焦编码区与近端调控区，并对落入软掩码重复区的 $15\%$ **MLM** 掩码位点施加损失降权。预训练后可直接冻结主干实现跨物种基因结构预测，或通过双向上下文条件概率比 $\log [P(\text{Allele}_{\mathrm{ALT}}\mid\text{Context}) / P(\text{Allele}_{\mathrm{REF}}\mid\text{Context})]$ 零样本评估突变效应。

<a id="p-019"></a>
### ⚪ **PlantCAD2**：基于结构化状态空间对偶的长上下文植物基因组编码器
- **paper**：[**PlantCAD2: A DNA foundation model for interpreting genomes across flowering plants**](https://www.cell.com/cell-genomics/fulltext/S2666-979X(26)00213-4)（*Cell Genomics* 2026）

**PlantCAD2** 将双向 **RC** 等变编码器的底层算子升级为 **Mamba2**，结合基因侧翼长窗口采样与重复区降权，在 $8{,}192\text{ bp}$ 窗口内高效建模跨物种被子植物基因组。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-023-699ab9f6.png)

**双向等变 Mamba2 与侧翼长窗口语料：** 模型围绕被子植物注释基因及上下游各 $5\text{ kb}$ 侧翼调控区提取长窗口，利用 **Mamba2** 的结构化状态空间对偶性（**SSD**）将线性状态递推转化为块状矩阵乘法以提升长序列并行效率，支持单碱基保守性评分与小片段缺失等结构变异的零样本似然评估。

<a id="p-wisteria"></a>
### ⚪ **Wisteria**：基于 BiMamba 与傅里叶注意力的多尺度编码器
- **paper**：[**Wisteria: A Unified Multi-Scale Feature Learning Framework for DNA Language Model**](https://arxiv.org/abs/2605.05913)（*Pattern Recognition* 2026）

**Wisteria** 构建自底向上的多尺度单碱基双向编码器：浅层通过门控扩张卷积—**BiMamba** 模块（**GCMB**）捕获局部基序，中深层通过 **BiMamba** 与门控 **MLP** 精炼长程上下文，顶层引入傅里叶位置注意力（**FoPE**）建模基因组频域周期性。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-098-wisteria.png)

**浅层 GCMB 模块与顶层傅里叶位置注意力（FoPE）：** 在浅层中，**BiMamba** 先提取正反向双向上下文均值 $H$，再经双分支扩张卷积生成局部特征 $h$ 与门控 $g$ 并由双门控 **MLP**（$\mathrm{SiLU}(U)\odot\sigma(G)$）更新：

$$
H = \frac{1}{2}\Big(\mathrm{Mamba}_{\mathrm{fwd}}(X) + \mathrm{Flip}\big(\mathrm{Mamba}_{\mathrm{bwd}}(\mathrm{Flip}(X))\big)\Big),\quad
Y = \mathrm{LN}\Big(\mathrm{MLP}\big(H + \mathrm{GeLU}(\mathrm{Conv}_A(H)) \odot \sigma(\mathrm{Conv}_B(H))\big)\Big).
$$

为捕获核小体定位相关的频域周期性（如 $10\sim 11\text{ bp}$ 振荡）并避免 **RoPE** 延拓时的频谱损伤，顶层自注意力采用含可学习谐波系数 $a_\omega$ 与截止频率 $\omega_l = 2\pi / N$ 的 **FoPE**：

$$
f(\omega_m) = \begin{cases} 
1, & \omega_m < \omega_l, \\
e^{i\omega_m n} + \sum_\omega a_\omega e^{i\omega n}, & \omega_m \ge \omega_l,
\end{cases}\qquad
\mathrm{Attn}(q, k) = \mathrm{Softmax}\!\left(\frac{\big(q f(\omega)\big)\big(k f(\omega)\big)^\top}{\sqrt{D}}\right).
$$

<a id="p-eccdnamamba"></a>
### ⚪ **eccDNAMamba**：超长环状 DNA 的循环增强双向 Mamba-2 编码器
- **paper**：[**eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis**](https://arxiv.org/abs/2506.18940)（*ICML 2025 Workshop GenBio*）

**eccDNAMamba** 结合 **BPE** 分词、首尾循环拓扑增强与双向 **Mamba-2** 编码器，通过连续跨度掩码语言建模实现长达 $200\text{ kbp}$ 的染色体外环状 **DNA（eccDNA）** 全长表征。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-103-eccdna.png)

**首尾循环拓扑增强与双向 Mamba-2 编码：** 针对环状分子在任意断点线性化会割裂首尾成环接合区依赖的问题，模型在 **BPE** 序列 $x_{1:L}$ 前端附加 `[CLS]`，并将起始端前 $s=64$ 个词元复制追加至尾部构造循环增强输入 $\tilde{x} = (\mathrm{[CLS]}, x_1, \ldots, x_L, x_1, \ldots, x_s)$。前向与后向 **Mamba-2** 分支分别扫描正反序列并经 **MLP** 融合（显式置零 `[PAD]` 状态以阻断填充噪声），配合连续跨度掩码（**SpanMLM**）预训练，支持无需截断的 $200\text{ kbp}$ 全长分类：

$$
\overrightarrow{h} = \overrightarrow{\mathrm{Mamba}}(\overrightarrow{\tilde{x}}) = [\overrightarrow{h}_{\mathrm{cls}}, \overrightarrow{h}_1, \ldots, \overrightarrow{h}_L], \quad
\overleftarrow{h} = \overleftarrow{\mathrm{Mamba}}(\overleftarrow{\tilde{x}}) = [\overleftarrow{h}_L, \ldots, \overleftarrow{h}_1, \overleftarrow{h}_{\mathrm{cls}}].
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-102-eccdnamamba.png)


<a id="enc-rnn"></a>
# 4. 循环与记忆网络框架（Recurrent & Memory Architectures）

**现代循环与快权重记忆网络（Extended RNNs & Fast-Weight Memory）**通过两项革新克服了传统 **LSTM** 的并行与容量瓶颈：（1）移除门控对上一时刻输出 $h_{t-1}$ 的非线性依赖，支持训练期分块矩阵并行与推理期 $\mathcal{O}(1)$ 状态递推；（2）将标量向量记忆升级为键值外积构成的**矩阵记忆（Matrix Memory）** $C_t \in \mathbb{R}^{d \times d}$ 或测试时训练（**TTT**）子网络：

$$
C_t = \alpha_t C_{t-1} + \beta_t v_t k_t^\top, \quad o_t = C_t q_t.
$$

在基因组编码器中，高容量循环记忆常与局部滑动窗口注意力结合，由局部注意力解析邻近基序语法，再通过跨链通信或双向变点差异写入将长程调控信号压缩进全局记忆状态。

<a id="p-020"></a>
### ⚪ **Bio-xLSTM (DNA-xLSTM)**：指数门控与矩阵记忆的基因组扩展长短期记忆网络
- **paper**：[**Bio-xLSTM: Generative Modeling, Representation and In-Context Learning of Biological and Chemical Sequences**](https://openreview.net/forum?id=IjbXZdugdj)（*ICLR* 2025）

**DNA-xLSTM** 在单核苷酸分辨率下引入指数门控与矩阵记忆原语（**sLSTM** 与 **mLSTM**），结合双向扫描与反向互补等变架构，统一支持基因组掩码表征学习与自回归生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-024-6819ca60.png)

**标量/矩阵扩展记忆与反向互补等变设计：** **sLSTM** 引入指数输入门 $i_t$ 与归一化状态 $n_t$ 动态修订标量记忆；**mLSTM** 则移除隐状态递归，通过键值外积更新矩阵记忆 $C_t \in \mathbb{R}^{d \times d}$ 以支持分块并行训练：

$$
\text{sLSTM: } c_t=f_t\odot c_{t-1}+i_t\odot z_t,\;\; h_t=o_t\odot(c_t\odot n_t^{-1}); \qquad
\text{mLSTM: } C_t=f_t C_{t-1}+i_t v_t k_t^\top,\;\; h_t=o_t\odot\frac{C_t q_t}{\max(\vert n_t^\top q_t\vert,1)}.
$$

为适配双链 **DNA**，模型共享参数 $w$ 扫描正反序列实现双向化，并在参数共享等变变体（**DNA-xLSTM-PS**）中将通道对半切分为 $[X_a, X_b]$，对第二路施加隐空间反向互补变换 $\mathcal{R}$：

$$
H=\mathrm{xLSTM}(X_{1:T};w)+\mathrm{xLSTM}(X_{T:1};w)_{T:1},\quad
[H_a,H_b]=\big[\mathrm{block}(X_a),\;\mathcal{R}\big(\mathrm{block}(\mathcal{R}(X_b))\big)\big].
$$

<a id="p-crossdna"></a>
### ⚪ **CrossDNA**：跨链动态交互与自蒸馏的双链循环记忆模型
- **paper**：[**Explicit Dynamic Cross-Strand Interactions for DNA Sequence Language Modeling**](https://www.nature.com/articles/s42256-026-01249-1)（*Nature Machine Intelligence* 2026）

**CrossDNA** 提出双分支动态跨链架构，结合闭环双线性循环记忆单元（**Comba**）、词元级跨链通信（**TokenBridge**）与教师自蒸馏，构建低方向敏感性的双链基因组表征。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-066-crossdna.png)

**Comba 记忆骨干、TokenBridge 通信与教师自蒸馏：** 正反双链切分为长 $\ell$ 的片段后按 $\sigma_t \in \{0,1\}$ 交替送入两学生分支，主干由局部滑窗注意力与引入双重负反馈抑制远距噪声的 **Comba** 矩阵状态单元交替构成。两分支对齐特征 $\tilde{H}_A, \tilde{H}_B$ 经 **TokenBridge** 门控交换双链信息得到 $F_A, F_B$；同时由 **EMA** 教师模型编码未切分连续序列提供平滑目标 $R_A^\star, R_B^\star$ 与 **MLM** 联合优化，以消除分段轮换引发的语义断裂：

$$
S_i = S_{i-1}\left(\alpha_i I - \tilde{\beta}_i k_i k_i^\top\right) + \beta_i v_i k_i^\top, \quad
o_i = S_i(q_i - d k_i), \quad
\mathcal{L} = \mathcal{L}_{\mathrm{MLM}} + \frac{w_{\mathrm{sem}}}{\ell}\sum_{u=1}^{\ell}\left(\|F_A(u) - R_A^\star(u)\|_2^2 + \|F_B(u) - R_B^\star(u)\|_2^2\right).
$$

<a id="p-011"></a>
### ⚪ **Double-Take**：局部注意力与双向快权重记忆混合的长上下文架构
- **paper**：[**Double-Take: Improving Capacity, Binding, and Selectivity in Long-Context within Bidirectional Memory**](https://arxiv.org/abs/2610.13073)（*arXiv* 2026）

**Double-Take（DT）** 在结合局部滑窗注意力的单核苷酸双向编码器中，引入多步正交化快权重更新、因果跨位置绑定与双向变点差异写入机制，实现百万碱基尺度的选择性记忆建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-084-double-take.png)

**正交化快权重更新与双向变点差异写入：** 每个记忆头参数化为 **SwiGLU** 子网络 $f_M(x)$，在第 $m$ 块内以 $\mathcal{L}_{\mathrm{mem}} = -\sum_i \langle f_M(k_i), v_i \rangle$ 为目标执行带 **Newton–Schulz** 正交化算子 $\mathrm{NS}(\cdot)$ 的多步梯度更新以消除跨块串扰，并在写入前对键流施加核宽为 $c$ 的因果卷积 $\tilde{k}_i = \sum_{r=0}^{c-1} \alpha_r^{(k)} k_{i-r}$ 建立跨位点绑定。为过滤均质背景并突显拓扑结构边界，模型并行处理正反序列，将左右上下文摘要之差 $y_i = \mu_i^L - \mu_i^R$ 作为写入目标（均质背景处相互抵消趋零，边界处保留高对比度写入）：

$$
M_m^{(s+1)} = M_m^{(s)} - \eta_s\,\mathrm{NS}\!\left(\nabla_M \mathcal{L}_{\mathrm{mem}}(M_m^{(s)}; K, V)\right), \qquad
\sum_i v_i \tilde{k}_i^\top = \sum_i \sum_{r=0}^{c-1} \alpha_r^{(k)} v_i k_{i-r}^\top, \qquad
y_i = \mu_i^L - \mu_i^R.
$$

<a id="p-mnemos"></a>
### ⚪ **Mnemos-DNA**：门控 Delta 规则线性注意力与 $k$-mer 寻址快权重记忆模型
- **paper**：[**Mnemos-DNA: Gated Delta-Rule Linear Attention with k-mer Fast-Weight Memory for Edge-Scale DNA Modeling**](https://github.com/B4K2/Mnemos-DNA)（*Preprint* 2026）

**Mnemos-DNA** 在边缘级单核苷酸架构中为每个门控 **Delta** 规则线性注意力（**KDA**）层并行挂载一个以精确 $8$-**mer** 内容寻址的快权重矩阵记忆表，以仅增加 $2.9\%$ 参数量的代价换取 $\mathcal{O}(N)$ 复杂度下跨 $64\text{ kbp}$ 的长程精确召回。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-105-mnemos.png)

**双记忆并行读取与 $k$-mer 哈希寻址：** 模型采用单核苷酸分词且不设位置编码，输入经短因果卷积与 $\ell_2$ 归一化生成 $q, k, v$ 后由两套并行记忆读取：（1）**全局共享 KDA 状态**维护单一矩阵 $S_t \in \mathbb{R}^{d_v \times d_k}$ 执行纠错式 **Delta** 更新；（2）**逐 $8$-mer 私有快权重表**额外维护 $4^8 = 65{,}536$ 个独立矩阵槽位 $W_a \in \mathbb{R}^{16 \times 16}$，由当前最近 $8$ 个碱基的精确序列哈希寻址至槽位 $a$ 执行局部 **Delta** 更新与门控读取（单槽位在正交键方向下最多可无干扰并存 $16$ 个关联映射，将长程召回代价从参数量转移至流式状态体积）：

$$
S_t = \mathrm{diag}(\alpha_t)S_{t-1} + \beta_t(v_t - S_{t-1}k_t)k_t^\top, \qquad
W_a \leftarrow W_a + g_w(v - W_a k)k^\top, \qquad
o_t = S_t q_t + g_r W_a q.
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-106-mnemos-detail.png)
