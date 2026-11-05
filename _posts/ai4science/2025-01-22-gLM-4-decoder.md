---
layout: post
title: '基因组基础模型(Genome Foundation Model) IV：解码器框架'
date: 2025-01-22
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm4-decoder-000-cover.png'
tags: AI4S
---

> Genome Foundation Model part IV: Decoder Architectures.

**基因组基础模型（Genome Foundation Model, gFM）**，又称**基因组语言模型（Genomic Language Model, gLM）**，将 **DNA** 序列映射为长度为 $T$ 的离散词元序列 $z_{1:T}$，通过在大规模无标注基因组语料上拟合边缘或条件概率分布 $p_\theta(z)$，学习具备跨任务迁移能力的通用基因组表征。

本文是**基因组基础模型**系列的第四篇。全系列按“**分词 → 预训练 → 编码器 → 解码器 → 多模态**”的脉络组织：
- [I. DNA 分词策略](https://0809zheng.github.io/2025/01/01/gLM-1-token.html)：如何将碱基序列切分为词元。
- [II. 预训练策略](https://0809zheng.github.io/2025/01/08/gLM-2-object.html)：如何构造自监督信号从无标注基因组中学习。
- [III. 编码器框架](https://0809zheng.github.io/2025/01/15/gLM-3-encoder.html)：模型如何在长序列上混合上下文信息。
- **IV. 解码器框架**：模型如何评估序列似然并生成新序列。
- [V. 多模态模型](https://0809zheng.github.io/2025/01/29/gLM-5-multimodal.html)：如何联合建模基因组序列、其他生物分子与人类自然语言。

**解码器框架（Decoder Architecture）**以自回归方式将序列联合概率分解为条件概率乘积 $p_\theta(x) = \prod_{t=1}^T p_\theta(x_t \mid x_{<t})$，或通过离散扩散建模数据分布的变分下界。与仅能计算局部伪似然的掩码编码器不同，解码器不仅能直接评估任意完整序列或复杂插入缺失突变（**Indel**）的对数似然差 $\log p_\theta(x^{\mathrm{alt}}) - \log p_\theta(x^{\mathrm{ref}})$，还支持从头生成调控元件与人工基因组，其深层隐状态同样可迁移至判别式任务。

- **1. Transformer 解码器框架（Transformer Decoders）**
  - [GenSLMs：密码子级原核预训练与潜空间扩散层次规划](#p-006)
  - [megaDNA：三级多尺度因果 Transformer 的噬菌体语言模型](#p-001)
  - [GENERator：功能基因座长上下文预训练的真核生成式模型](#p-002)
  - [GENERator-v2：分解式单核苷酸监督与基因组压缩预训练](#p-003)
  - [OmniReg-GPT：局部滑窗-全局混合注意力与多尺度顺式调控解码器](#p-omnireggpt)
  - [Genos：基于泛基因组与稀疏混合专家的百万碱基级模型](#p-004)
  - [Gene42：基于渐进式持续预训练与稠密注意力的长程模型](#p-gene42)
  - [AntigenLM：全基因组八节段保序预训练的流感演化预测模型](#p-005)
  - [DeepD：系统发育感知稀疏混合专家与上下文-锚点-主体染色体级生成模型](#p-deepd)
  - [Carbon：高信噪比注释语料与 CE-FNS 阶段切换的高效长上下文解码器](#p-carbon)
- **2. 隐式长卷积与混合框架（Convolutional & Hybrid Decoders）**
  - [HyenaDNA：基于隐式长卷积与数据门控的单核苷酸模型](#p-007)
  - [Evo：长卷积与注意力混合架构的全基因组生成基础模型](#p-008)
  - [Evo 2：多尺度卷积混合架构的跨生命域基因组基础模型](#p-009)
- **3. 状态空间混合框架（State-Space Hybrid Decoders）**
  - [HybriDNA：结构化状态空间对偶与因果注意力混合的长程模型](#p-010)
  - [SparseDNA：嵌入轻量线性共享-路由专家的状态空间模型](#p-sparsedna)
- **4. 其他框架（Other Architectures）**
  - [ENBED：字节级精度与亚二次注意力的非对称编码-解码器](#p-enbed)
  - [OpticalDNA：基于多模态文档渲染与 OCR 的基因组视觉建模](#p-opticaldna)
  - [D3LM：连续掩码率加权去噪的离散扩散基因组语言模型](#p-011)

<a id="dec-transformer"></a>
# 1. Transformer 解码器框架（Transformer Decoders）

**自回归 Transformer 解码器**通过下三角因果掩码 $M \in \{0, -\infty\}^{T \times T}$（当 $j > i$ 时 $M_{ij} = -\infty$）阻断未来信息回流，使位置 $i$ 仅聚合前缀上下文 $x_{\le i}$：

$$
\mathrm{CausalAttn}(Q, K, V) = \mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
$$

直接套用标准自回归 **Transformer** 面临三项矛盾：（1）单碱基长序列极易触发 $\mathcal{O}(T^2)$ 计算与 **KV Cache** 显存瓶颈；（2）真核基因组中大段低复杂度非编码背景会稀释功能元件的梯度信号；（3）采用 $k$-**mer** 分词虽能压缩序列长度，但扁平的 $4^k$ 分类损失忽略了近邻词元的碱基重叠相似性，且难以直接导出单碱基突变似然。为此，基因组 **Transformer** 解码器发展出多尺度层次化注意力、局部—全局混合注意力、稀疏混合专家（**MoE**）、基因中心语料压缩以及分解式单核苷酸监督（**FNS**）等改进方案。

<a id="p-006"></a>
### ⚪ **GenSLMs**：密码子级原核预训练与潜空间扩散层次规划
- **paper**：[**GenSLMs: Genome-scale language models reveal SARS-CoV-2 evolutionary dynamics**](https://doi.org/10.1177/10943420231201154)（*IJHPCA* 2023）

**GenSLMs** 采用密码子级分词在跨属原核生物基因上预训练 **GPT** 解码器，并结合潜空间连续扩散高层规划与奖励引导束搜索，在全基因组尺度建模并预测 **SARS-CoV-2** 变异株的演化轨迹。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-088-genslm.png)

**密码子级两阶段迁移与潜空间扩散规划：** 模型将每三个连续核苷酸编码为一个密码子词元（词表容量 $64$），先在原核单基因序列上预训练，再将窗口扩展至 $10{,}240$ 个密码子在完整 **SARS-CoV-2** 基因组上微调。为保持超长序列生成的全局连贯性，模型将每 $512$ 个密码子切分为一个“句子”：（1）用对比损失训练句子编码器提取潜表征；（2）在潜表征序列上训练去噪扩散模型恢复全局高层规划 $z_0$；（3）令自回归解码器同时以前置局部窗口 $r$ 与全局规划 $z_0$ 为条件建模 $p(x_t \mid r, z_0)$。在定向生成特定属性（如比对得分 $R(x) = \sum_{t=1}^L r_t(x_t)$）的候选毒株时，束搜索融合语言模型似然与阶段性奖励（$\alpha$ 为长度惩罚，$\mu \in [0, 1]$ 为混合系数）：

$$
S(x) = \frac{\mu}{L^\alpha}\sum_{t=1}^L \log p_\theta(x_t \mid x_{<t}, c) + (1-\mu)R(x), \quad
s_t(x_t) = \mu\, p_\theta(x_t \mid \mathcal{X}_{\mathrm{beams}}^i, c) + (1-\mu)\, r_t(x_t).
$$

<a id="p-001"></a>
### ⚪ **megaDNA**：三级多尺度因果 Transformer 的噬菌体语言模型
- **paper**：[**A long-context language model for deciphering and generating bacteriophage genomes**](https://doi.org/10.1038/s41467-024-53759-4)（*Nature Communications* 2024）

**megaDNA** 采用三级多尺度因果 **Transformer** 架构，在无注释完整噬菌体基因组上进行自回归预训练，实现高达 $96\text{ kbp}$ 的全基因组从头生成、零样本必需基因识别与跨尺度功能表征提取。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-087-megadna.png)

**三级多尺度解码器与跨层迁移：** 模型采用单核苷酸分词，引入 **MegaByte** 式三级层次化解码器以突破长序列自注意力瓶颈：（1）**局部层**在 $16\text{ bp}$ 窗口内捕获密码子与短调控基序；（2）**中间层**将有效感受野扩展至 $1{,}024\text{ bp}$ 以建模单个开放阅读框与顺式调控区；（3）**全局层**汇总中间层特征，将上下文扩展至完整输入序列（最高 $96\text{ kbp}$）。推理阶段既可通过温度采样从头生成包含启动子、核糖体结合位点与编码区的完整噬菌体基因组，也可通过突变前后的自回归损失增量零样本识别必需基因，或拼接三层激活向量输入线性探针完成有监督预测。

<a id="p-002"></a>
### ⚪ **GENERator**：功能基因座长上下文预训练的真核生成式模型
- **paper**：[**GENERator: A Long-Context Generative Genomic Foundation Model**](https://arxiv.org/abs/2502.07272)（*arXiv* 2025）

**GENERator** 在真核生物基因中心功能语料上结合随机相位偏移的 $6$-**mer** 分词与 **LLaMA** 因果解码器架构，在 $98\text{k}$ 碱基窗口内实现序列恢复、单碱基零样本变异效应预测及顺式调控元件设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-030-6891d386.png)

**随机相位 $6$-mer 分词与概率边缘化变异评估：** 模型仅提取注释基因边界内的完整功能基因座构建跨物种真核语料，以避免过拟合基因间区低复杂度重复背景。主干采用配备 **RoPE** 与 **GQA** 的 **LLaMA** 解码器，并在每个训练周期为序列在 $[0, 5]\text{ nt}$ 内随机采样起始截断偏移以遍历不同 $6$-**mer** 切分相位。在零样本单碱基变异评估中，模型将变异位点置于因果序列末端，聚合覆盖位点 $i$ 的词元集合 $\mathcal{T}_i$ 的预测概率（$j$ 为位点 $i$ 在词元 $t$ 内的相对偏移），边缘化投影回单碱基空间以计算参考与替代等位基因的对数似然比：

$$
p(s_i = X) = \sum_{t \in \mathcal{T}_i} p(t)\,\mathbb{I}(t_j = X), \quad
\mathrm{VEP}(R \to A) = \log \frac{p(s_i = R \mid S_{\setminus i})}{p(s_i = A \mid S_{\setminus i})}.
$$

在调控元件设计中，模型将训练集按活性分位数离散化为前缀提示词元微调生成，并配合微调的定量回归器筛选高活性候选序列。

<a id="p-003"></a>
### ⚪ **GENERator-v2**：分解式单核苷酸监督与基因组压缩预训练
- **paper**：[**Functional In-Context Learning in Genomic Language Models with Nucleotide-Level Supervision and Genome Compression**](https://doi.org/10.64898/2026.01.27.702015)（*bioRxiv* 2026）

**GENERator-v2** 在保持 **Transformer** 解码器与 $6$-**mer** 分词高吞吐量的基础上，提出分解式单核苷酸监督（**FNS**）与基因组压缩预训练（**GCP**），解决了非重叠 $k$-**mer** 分类损失忽略近邻碱基相似性以及真核功能元件稀疏导致跨基因互作欠拟合的问题。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-031-699abe35.png)

**分解式单核苷酸监督（FNS）与基因组压缩预训练（GCP）：** **FNS** 在不改变 $4^k$ 词表输出维度的前提下，将词元分布 $p(v)$ 边缘化投影为块内各相对位点 $j \in \{1, \ldots, k\}$ 的单碱基条件分布 $p^{(j)}(b)$，配合遍历全部偏移量 $\delta \in \{0, \ldots, k-1\}$ 的分词移位增强实现相位不变的单碱基监督；同时，**GCP** 剔除每条染色体 $C$ 上的基因间背景，将注释功能区间按原始坐标顺序以特殊分隔符 `<s>` 拼接为连续压缩流（屏蔽 `<s>` 位置的损失）：

$$
p^{(j)}(b) = \sum_{v \in \mathcal{V}:\, v^{(j)} = b} p(v), \quad
\mathcal{L}_{\mathrm{FNS}} = -\sum_{j=1}^k \log p^{(j)}\!\big(y^{(j)}\big), \quad
X_C = x_1 \,\|\, \text{<s>} \,\|\, x_2 \,\|\, \text{<s>} \,\|\, \cdots.
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-032-699abf04.png)

推理时，序列末位内容词元的隐状态编码正反链方向与密码子相位，而末端分隔符 `<s>` 的隐状态则自动坍缩局部相位差异，聚合出基因级分类学摘要表征。

<a id="p-omnireggpt"></a>
### ⚪ **OmniReg-GPT**：局部滑窗-全局混合注意力与多尺度顺式调控解码器
- **paper**：[**Omnireg-gpt: a high-efficiency foundation model for comprehensive genomic sequence understanding**](https://doi.org/10.1038/s41467-025-65066-7)（*Nature Communications* 2025）

**OmniReg-GPT**（$270\text{M}$ 参数）结合 **BPE** 分词与“$12$ 层局部滑窗注意力 + $2$ 层全局注意力”混合解码器架构，在单张 $32\text{ GB}$ 显存 **GPU** 上支持最长 $200\text{ kb}$ 序列预训练，统一赋能跨尺度顺式调控理解与细胞类型特异增强子从头生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-124-omnireg-gpt.png)

**局部—全局混合注意力与跨尺度调控迁移：** 模型在人类 **T2T v2** 参考基因组切分的 $20\text{ kb}$ 片段上优化因果下一词元预测损失（均配备 **FlashAttention** 与 **RoPE**）：（1）**局部块**（$12$ 层）将长度为 $L$ 的查询切分为大小为 $w$ 的非重叠窗口，并与前序相邻窗口拼接作为键值序列（把窗口维并入批次维 $\mathrm{bs} \to \mathrm{bs}\cdot L/w$ 并行计算），将复杂度从 $\mathcal{O}(\mathrm{bs}\cdot L^2 d)$ 降至线性 $\mathcal{O}(Lwd)$，同时沿隐维施加词元移位增强局部交互；（2）**全局块**（顶部 $2$ 层）执行全序列注意力以聚合跨窗口长程调控依赖：

$$
Q = h_l^{(i)} W_q, \quad
K_j = h_l^{(j)} W_k,\;\; V_j = h_l^{(j)} W_v \quad \big(j \in [\max(0, i-w+1),\, i]\big).
$$

预训练表征与生成接口支持三类跨尺度应用：提取 **TSS** 周围 $20\text{ kb}$ 均值嵌入回归基因表达或通过线性探针预测跨物种单细胞基因活性与 **scATAC-seq** 可及性（分类头权重可直接充当细胞类型嵌入）；利用 **RoPE** 外推直接编码 $2\text{ Mb}$ 序列并接入 **C.Origami** 解码器预测 **Hi-C** 接触矩阵；以及利用序列对数似然零样本拟合 **STARR-seq** 活性，并结合 **CODA** 评分神谕执行两轮渐进式前缀提示生成以设计高特异性合成增强子。

<a id="p-004"></a>
### ⚪ **Genos**：基于泛基因组与稀疏混合专家的百万碱基级模型
- **paper**：[**Genos: a human-centric genomic foundation model**](https://doi.org/10.1093/gigascience/giaf132)（*GigaScience* 2025）

**Genos** 基于人类泛基因组单倍型解析组装构建两阶段渐进式语料，并在集成超大基频 **RoPE** 与 **GQA** 的 **Transformer** 解码器中引入稀疏混合专家（**MoE**）架构，实现兼顾人类群体多样性与条件计算效率的百万碱基级（$1\text{ Mb}$）基因组建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-033-699bbadf.png)

**人类泛基因组两阶段语料与稀疏 MoE 解码器：** 模型采用单核苷酸分词，整合 **HPRC** 与 **HGSVC** 的高质量人类单倍型从头组装：（1）**第一阶段**在 $8\text{k}\sim 1\text{M}\text{ bp}$ 四尺度片段中剔除远端基因间区，按长度由短至长渐进训练；（2）**第二阶段**取消过滤并混合全尺度正反链样本，学习片段重复与转座元件等背景结构。架构上，注意力层结合基频达 $5\times 10^7$ 的 **RoPE** 与 **GQA**；前馈层升级为 **Top**-$2$ 路由的稀疏 $\mathrm{SwiGLU}$ **MoE**，自适应分配算力以区分简单重复区与复杂调控元件，并联合优化下一词元预测损失、负载均衡损失与路由器 **Z-loss** 惩罚以防止仅含 $4$ 类碱基词元引发的路由坍缩。

<a id="p-gene42"></a>
### ⚪ **Gene42**：基于渐进式持续预训练与稠密注意力的长程模型
- **paper**：[**Gene42: Long-Range Genomic Foundation Model With Dense Attention**](https://arxiv.org/abs/2503.16565)（*NeurIPS 2025 Workshop AI4D3*）

**Gene42** 采用不引入稀疏近似或卷积替代的标准 **LLaMA** 稠密自注意力解码器，结合单核苷酸分词、**RoPE** 基频缩放与多阶段渐进式持续预训练，将单碱基分辨率下的全注意力上下文窗口扩展至 $192\text{ kbp}$。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-092-gene42.png)

**渐进式长上下文持续预训练：** 模型在人类与 $14$ 种脊椎动物基因组上先以 $4{,}096\text{ bp}$ 短窗口和默认 **RoPE** 基频（$\theta = 10^4$）完成基础预训练；随后在持续预训练中将基频调大至 $\theta = 1.5 \times 10^7$ 以缓解远端衰减，并将窗口按 $4\text{k} \to 8\text{k} \to \cdots \to 192\text{k bp}$ 逐级翻倍拓宽。得益于稠密注意力对全局上下文的无损检索，模型在长程序列困惑度 $\mathrm{PPL}(X) = \exp\!\big\{-\frac{1}{t}\sum_{i=1}^t \log p_\theta(x_i \mid x_{<i})\big\}$ 评估上显著优于隐式长卷积基线。

<a id="p-005"></a>
### ⚪ **AntigenLM**：全基因组八节段保序预训练的流感演化预测模型
- **paper**：[**AntigenLM: Structure-Aware DNA Language Modeling for Influenza**](https://openreview.net/forum?id=AntigenLM)（*ICLR* 2026）

**AntigenLM** 在长上下文 **GPT-2** 解码器下提出两阶段功能单元编码策略，通过在无截断完整甲型流感基因组上执行保序自回归预训练，并结合多时间窗口时序提示微调，统一实现未来流行株抗原序列预测与病毒亚型判别。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-034-699bcad8.png)

**八节段保序拼接与时序块提示演化预测：** 预训练采用单核苷酸分词，将同一分离株的全部八个基因节段按由长至短的固定顺序（$\mathrm{PB2}\to\cdots\to\mathrm{NS}$）无间隔拼接为约 $13\text{k}$ 词元的单条序列，捕获跨节段共进化与代偿突变信号。在未来优势毒株预测中，模型引入哨兵词元将同一地理区域连续三个历史时间窗口与待预测未来窗口的血凝素（$\mathrm{HA}$）及神经氨酸酶（$\mathrm{NA}$）拼接为单条时序提示流（未来目标块 $\mathrm{block}^{(\star)}$ 省略前导 `<subtype>` 并从 `<HA>` 触发解码）：

$$
\underbrace{\mathrm{block}^{(1)}}_{\text{past}}\;\underbrace{\mathrm{block}^{(2)}}_{\text{past}}\;\underbrace{\mathrm{block}^{(3)}}_{\text{past}}\;\underbrace{\mathrm{block}^{(\star)}}_{\text{future}}, \quad
\mathrm{block}^{(i)} = \text{<subtype><HA>}\mathrm{HA}^{(i)}\text{<NA>}\mathrm{NA}^{(i)}\text{<sep>}\quad (i=1,2,3).
$$

<a id="p-deepd"></a>
### ⚪ **DeepD**：系统发育感知稀疏混合专家与上下文-锚点-主体染色体级生成模型
- **paper**：[**DeepD enables efficient genome modeling and controllable long-context sequence generation across domains of life**](https://doi.org/10.64898/2026.10.13.758170)（*bioRxiv* 2026）

**DeepD** 在单核苷酸分辨率与百万词元（$1\text{M}$）窗口下构建融合多尺度门控注意力与系统发育感知稀疏 **MoE** 的自回归模型，通过在跨生命域未筛选全基因组上预训练，统一支持零样本变异评估与染色体级可控生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-101-deepd.png)

**系统发育感知稀疏 MoE 与上下文–锚点–主体（CAB）接力生成：** 模型在完整保留基因间区与转座子的跨生命域全基因组（仅对软掩码重复碱基降权）上将窗口从 $8\text{K}$ 逐级扩展至 $1\text{M}$。解码块融合 $3/4$ 局部滑窗注意力头与 $1/4$ 全局注意力头，并由物种分类学嵌入引导稀疏专家路由（浅层专家分化为 **AT/GC** 组成传感器，中深层门控编码外显子边界与调控元件）。为克服长程续写中的误差累积与全局排布失控，**CAB** 接力策略将下游参考序列作为**上下文（Context）**、上游序列作为**锚点（Anchor）**，条件化生成中间定长（$8\text{ kb}$）的**主体（Body）**片段，并通过滑窗将新生成片段迭代转化为下一轮锚点，从而支持生成保持天然基因共线性的兆碱基级染色体区段。

<a id="p-carbon"></a>
### ⚪ **Carbon**：高信噪比注释语料与 CE-FNS 阶段切换的高效长上下文解码器
- **paper**：[**Carbon: Decoding the Language of Life**](https://doi.org/10.64898/2026.05.22.727119)（*bioRxiv* 2026）

**Carbon**（$3\text{B}/8\text{B}$）在标准 **Llama 3** 解码器架构上结合高信噪比注释感知语料、非重叠 $6$-**mer** 分词与“交叉熵（**CE**）转分解核苷酸监督（**FNS**）”两阶段目标调度，在维持最高 $786\text{ kbp}$ 有效上下文与单碱基分辨率的同时实现了数十倍的推理加速。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-119-carbon.png)

**信噪比引导语料清洗与 CE $\to$ FNS 阶段切换：** 为防止弱约束重复背景（占比 $1-\rho$、梯度方差 $\sigma_N^2$）稀释功能区信号（占比 $\rho$、对齐度 $\mu$），模型依据全基因组有效信噪比 $\mathrm{SNR}_{\mathrm{all}} \propto \rho\mu / \sqrt{\rho\sigma_S^2 + (1-\rho)\sigma_N^2}$ 滤除 $>100\text{ kbp}$ 长背景序列（保留率 $r \gg \epsilon$），构建高密度功能语料并按 $50\%$ 比例冠以 `<species type>` 与 `<gene type>` 控制标签。在 `<dna>...</dna>` 块内模型采用步长为 $6$ 的非重叠 $6$-**mer** 分词（将预填充注意力计算量降低 $36$ 倍）；为消除纯 **CE** 训练后期强制区分简并近似词元引发的梯度攀升与“损失阶梯（**Loss Staircase**）”现象，训练先在前 $100\text{B}$ 词元使用标准 **CE** 学习块内联合依赖，随后在剩余 $900\text{B}$ 词元切换至分解核苷酸监督（**FNS**）：

$$
p_t^{(j)}(b) = \sum_{v \in \mathcal{V}: v^{(j)}=b} p_t(v), \quad
\mathcal{L}(t) = \begin{cases}
-\log p_t(y_t), & t < T_{\mathrm{switch}}, \\[4pt]
-\dfrac{1}{6}\sum_{j=1}^{6} \log p_t^{(j)}\!\big(y_t^{(j)}\big), & t \ge T_{\mathrm{switch}}.
\end{cases}
$$

预训练后期通过缩放 **RoPE** 基频与 **YaRN** 外推将窗口扩展至 $131{,}072$ 词元（约 $786\text{ kbp}$），推理端直接利用 **FNS** 单碱基边缘分布执行零样本突变似然打分与逐碱基重组采样。


<a id="dec-conv"></a>
# 2. 隐式长卷积与混合框架（Convolutional & Hybrid Decoders）

**隐式长卷积框架（Implicit Long Convolution）**利用由小型网络隐式生成的全局因果卷积核 $h \in \mathbb{R}^L$ 替代自注意力矩阵，通过下三角 **Toeplitz** 矩阵乘法与逐元素数据门控交替操作，借助快速傅里叶变换（**FFT**）在 $\mathcal{O}(L \log_2 L)$ 时间内完成全局上下文聚合：

$$
y_t = (h * v)_t = \sum_{\tau=0}^{t} h_{t-\tau} v_\tau = \mathrm{iFFT}\big(\mathrm{FFT}(h) \odot \mathrm{FFT}(v)\big)_t.
$$

为克服纯隐式长卷积在精确跨位置信息检索上的容量瓶颈，新一代全基因组基础模型（如 **Evo** 与 **Evo 2**）演进为**多尺度卷积—注意力混合架构（StripedHyena）**：以占多数比例的多尺度输入依赖卷积层高效处理局部基序与长程背景，同时周期性交错插入少量 **RoPE** 自注意力层，兼顾百万碱基吞吐效率与高保真序列设计。

<a id="p-007"></a>
### ⚪ **HyenaDNA**：基于隐式长卷积与数据门控的单核苷酸模型
- **paper**：[**HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793246797c7-Abstract-Conference.html)（*NeurIPS* 2023）

**HyenaDNA** 以隐式长卷积与数据控制门控构成的 **Hyena** 算子完全替代稠密自注意力，结合序列长度渐进预热与软提示上下文学习，在 $\mathcal{O}(L\log_2 L)$ 复杂度下实现百万词元级单碱基自回归建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-037-680afb15.png)

**数据门控 Hyena 算子与软提示上下文学习：** 模型采用单核苷酸分词。$2$ 阶 **Hyena** 算子先经线性层与短卷积将输入映射为 $x_1, x_2, v \in \mathbb{R}^L$，随后交替施加对角数据门控 $D_{x_1}, D_{x_2}$ 与下三角 **Toeplitz** 因果长卷积 $T_h$（滤波器 $h_t = \gamma_\theta(t)$ 由带正弦激活与窗口调制的 **MLP** 隐式生成并经 **FFT** 计算）。预训练采用从 $L=64$ 逐级翻倍至百万碱基的长度预热；下游迁移支持冻结主干的**可调软提示（Soft Prompting）**，在任务序列 $x_p$ 前端拼接 $N$ 个可学习向量 $\theta \in \mathbb{R}^{N \times D}$ 并借用单碱基字符输出分类标签：

$$
(x_1, x_2, v) \mapsto H(x_1, x_2)v = D_{x_2} T_h D_{x_1} v, \qquad
x \leftarrow \mathrm{concat}[\mathrm{embed}(x_p),\, \theta] \in \mathbb{R}^{D \times (T+N)}.
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-036-680afb8f.png)

<a id="p-008"></a>
### ⚪ **Evo**：长卷积与注意力混合架构的全基因组生成基础模型
- **paper**：[**Sequence modeling and design from molecular to genome scale with Evo**](https://www.science.org/doi/10.1126/science.ado9336)（*Science* 2024）

**Evo** 采用将模态范式 **Hyena** 算子与 **RoPE** 自注意力交错堆叠的 **StripedHyena** 混合架构，在大规模原核与噬菌体全基因组语料（**OpenGenome**）上以单核苷酸分辨率训练，实现了跨 **DNA**、**RNA** 与蛋白质模态的零样本功能预测与多组件系统生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-038-680b267c.png)

**StripedHyena 混合架构与多组件协同生成：** 主干由数据控制 **Hyena** 层与等间隔插入的少量（约 $10\%$）**RoPE** 自注意力层交错构成。在 **Hyena** 层中，输入经线性投影与短卷积生成 $q, k, v$，随后通过模态范式状态空间隐式生成全局滤波器 $h = \mathrm{ImplicitFilter}(L)$，依次执行逐元素门控与 **FFT** 长卷积：

$$
v \leftarrow k \odot v, \quad v_t \leftarrow \mathrm{FFTConv}(h, v)_t, \quad y \leftarrow q \odot v.
$$

训练先在 $8\text{k}$ 窗口完成预训练，再通过 $16$ 倍 **RoPE** 线性位置插值扩展至 $131\text{k}$。依托原核操纵子共定位特性，模型直接利用对数似然比 $\log [p(x_{\mathrm{mut}}) / p(x_{\mathrm{wt}})]$ 零样本评估蛋白质、**ncRNA** 突变效应及必需基因，并支持协同生成 **CRISPR-Cas** 复合物与数百 **kb** 的人工原核基因组。

<a id="p-009"></a>
### ⚪ **Evo 2**：多尺度卷积混合架构的跨生命域基因组基础模型
- **paper**：[**Genome modelling and design across all domains of life with Evo 2**](https://doi.org/10.1038/s41586-026-10176-5)（*Nature* 2026）

**Evo 2** 在跨越全生命域的 **OpenGenome2** 语料上提出由短显式、中正则化与长隐式卷积及旋转注意力交错构成的 **StripedHyena 2** 多重混合架构，在单核苷酸分辨率与百万碱基（$1\text{ Mb}$）窗口下统一实现全谱系变异评估、稀疏自编码器可解释性解析与推理时引导的表观遗传设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-039-680b357e.png)

**StripedHyena 2 多重混合架构与引导式表观设计：** **StripedHyena 2** 将输入依赖卷积拆解为短显式卷积（**SE**）、中距离正则化卷积（**MR**）与长隐式卷积（**LI**），以 **SE-MR-LI** 组合块与 **RoPE** 自注意力层周期性交错堆叠；训练先在 $8\text{k}$ 短窗口优先采样真核功能元件学习核心语法，再经长上下文中期训练（**Midtraining**）逐级扩展至 $1\text{M}$ 词元。模型无需多序列比对即可通过自回归似然变化统一评估编码与非编码区的 **SNV** 及结构变异；同时在中间层激活上训练 **Batch-TopK** 稀疏自编码器（**SAE**），无监督解耦出对齐剪接边界、移码突变与转录因子基序的稀疏特征，并可结合 **Enformer / Borzoi** 按每 $128\text{ bp}$ 块执行束搜索引导采样以编程设计目标染色质可及性峰值。


<a id="dec-ssm"></a>
# 3. 状态空间混合框架（State-Space Hybrid Decoders）

**状态空间解码器（State-Space Decoders）**利用选择性状态空间模型（如 **Mamba2**）的线性递归形式，以 $\mathcal{O}(1)$ 的单步显存与计算开销维护历史上下文，消除了随长度线性膨胀的 **KV Cache** 瓶颈。基于结构化状态空间对偶性（**SSD**），其线性状态递推精确等价为半可分矩阵（**Semiseparable Matrix**）变换：

$$
h_t = a_t h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t \quad \Longleftrightarrow \quad y = M x, \quad M_{ij} = \mathbb{I}(j \le i)\, C_i^\top A_{j:i+1} B_j.
$$

针对固定维度递归隐状态在超长距离检索上的有损压缩以及单向掩码缺失右侧上下文的局限，基因组状态空间解码器通常交错混入少量自注意力层，并在判别微调时配合序列重放（回声嵌入）补全双向视野。

<a id="p-010"></a>
### ⚪ **HybriDNA**：结构化状态空间对偶与因果注意力混合的长程模型
- **paper**：[**HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model**](https://arxiv.org/abs/2502.10807)（*ICLR 2025 Workshop*）

**HybriDNA** 将 **Mamba2** 结构化状态空间对偶块与标准 **Transformer** 因果自注意力块按 $7:1$ 比例交错堆叠，结合回声嵌入（**Echo Embedding**）判别微调与提示生成微调，在 $131\text{ kb}$ 单碱基窗口下统一支持长程基因组理解与顺式调控元件设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-040-6891c3a3.png)

**7:1 交错混合架构与回声嵌入微调：** 模型采用单核苷酸分词且省略显式位置编码，每 $8$ 层由 $7$ 个 **Mamba2**（将状态转移简化为 $A_t = a_t I$，以 $\mathcal{O}(NL)$ 复杂度聚合全局上下文）与 $1$ 个因果注意力块交错构成。为弥补因果掩码无法感知右侧上下文的缺陷，分类微调将输入序列重复拼接为回声输入 $x_{\mathrm{echo}} = [x, x]$，仅提取能够完整关注前半段上下文的后半段末层隐状态均值池化得到 $h_\theta(x_{\mathrm{echo}})$；调控元件生成任务则在序列前端拼接活性等级提示词元 $z_{0:k-1}$ 最小化条件自回归损失：

$$
P(y \mid x) = \mathrm{Softmax}\big(h_\theta(x_{\mathrm{echo}})W + b\big), \quad
\mathcal{L}_{\mathrm{gen}}(\theta) = -\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid z_{0:k-1}, x_{<t}).
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-041-6891c3ef.png)

<a id="p-sparsedna"></a>
### ⚪ **SparseDNA**：嵌入轻量线性共享-路由专家的状态空间模型
- **paper**：[**SparseDNA: Efficient Genomic Sequence Modeling via Sparse Experts in State Spaces**](https://doi.org/10.1007/978-981-92-0369-7_5)（*DASFAA* 2026）

**SparseDNA** 将包含共享专家与路由专家的轻量级混合专家模块直接嵌入 **Mamba** 块的输入投影阶段，在保持状态空间模型 $\mathcal{O}(L)$ 线性复杂度的同时，为异质性基因组信号提供条件化特征特化能力。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-091-sparsedna.png)

**内嵌共享-路由线性专家投影：** 模型将每个专家参数化为单层线性投影 $E_k(x) = W_k x + b_k$，同时配置 $1$ 个常驻共享专家 $E_{\mathrm{shared}}$ 与 $N_e$ 个按 **Top**-$K$ 稀疏激活的路由专家以替代标准 **Mamba** 块的前置稠密投影，其输出沿通道维对半切分为一维卷积分支 $X_c$（随后进入选择性状态空间算子递推）与门控分支 $X_g$：

$$
\tilde{X} = \sum_{k=1}^{N_e} \mathrm{Softmax}(W_r X)_k E_k(X) + E_{\mathrm{shared}}(X), \qquad
[X_c, X_g] = \mathrm{Split}(\tilde{X}), \quad H_c = \mathrm{Conv1D}(X_c), \quad H_g = \sigma(X_g).
$$


<a id="dec-diffusion"></a>
# 4. 其他框架（Other Architectures）

<a id="p-enbed"></a>
### ⚪ **ENBED**：字节级精度与亚二次注意力的非对称编码-解码器
- **paper**：[**Understanding the natural language of DNA using encoder–decoder foundation models with byte-level precision**](https://doi.org/10.1093/bioadv/vbae117)（*Bioinformatics Advances* 2024）

**ENBED** 采用层数比为 $2:1$ 的非对称编码器—解码器架构，结合单核苷酸字节级分词与滑动窗口—全局块聚合混合注意力，在 $16{,}384\text{ bp}$ 窗口下统一支持序列分类、测序噪声识别与序列到序列的病毒突变生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-094-enbed.png)

**2:1 重编码器架构与亚二次混合注意力：** 模型将主要参数容量分配给双向编码器，并组合两类稀疏机制降低复杂度：（1）**滑动窗口注意力**限制每个词元仅关注半径为 $r$ 的局部邻域；（2）**块聚合全局注意力**将序列均分为 $k$ 个连续块并求和归一化构造 $k$ 个全局词元，令每个输入词元同时关注局部邻居与全部 $k$ 个全局词元，将总复杂度控制在 $\mathcal{O}\big(L(r + k)\big)$。模型在多物种语料上执行短跨度掩码预训练，判别任务直接提取编码器表征，病毒演化预测则由解码器自回归生成子代序列。

<a id="p-opticaldna"></a>
### ⚪ **OpticalDNA**：基于多模态文档渲染与 OCR 的基因组视觉建模
- **paper**：[**Rethinking Genomic Modeling Through Optical Character Recognition**](https://openreview.net/forum?id=nggzekChuU)（*ICML* 2026）

**OpticalDNA** 将一维长 **DNA** 序列光栅化为二维图像文档，利用高压缩比视觉词元避免逐碱基扫描大段低信息密度背景区域，实现区域级基因组阅读、定位与检索。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-conf-icml26-005-opticaldna.png)

**DNA 文档渲染与视觉压缩编解码：** 模型将长序列按行渲染为多页固定分辨率 **RGB** 图像文档 $D(S)=(\{\text{page}_p\}_{p=0}^{P-1},\{B^{(p)}\}_{p=0}^{P-1})$，建立从序列区间到二维像素边界框的双射映射，并围绕全文转录、感兴趣区域（**ROI**）框转录、涂白补全与子序列检索等六类 **OCR** 问答任务预训练。视觉编码器提取每页特征，经卷积 $16\times$ 下采样、投影器 $\Pi_\theta$ 与多页融合模块 $F_\theta$ 聚合为文档级视觉表示 $Z$，再由 **MoE** 自回归解码器 $G_\psi$ 在指令 $q$ 条件下生成目标序列（判别微调时直接丢弃解码器并在视觉编码器上池化分类）：

$$
Z = F_\theta\big(\Pi_\theta(V(\{\text{page}_p\}))\big) \in \mathbb{R}^{L\times d}, \quad
P_\psi(\hat Y\mid Z,q)=\prod_{t=1}^{T}P_\psi(\hat y_t\mid\hat y_{<t},Z,q).
$$

<a id="p-011"></a>
### ⚪ **D3LM**：连续掩码率加权去噪的离散扩散基因组语言模型
- **paper**：[**D3LM: A Discrete DNA Diffusion Language Model for Bidirectional DNA Understanding and Generation**](https://arxiv.org/abs/2603.01780)（*ICLR 2026 Workshop*）

**D3LM** 在双向 **Transformer**（复用 **NT-v2** 主干与非重叠 $6$-**mer** 分词）下引入离散掩码扩散范式，通过连续掩码率加权交叉熵目标，将双向表征学习与非顺序迭代去噪生成统一于单一模型。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-089-d3lm.png)

**连续掩码率扩散训练与逆向去掩码采样：** 前向过程从 $t \in [0, 1]$ 均匀采样掩码率，将干净序列 $x_0$ 各位置独立以概率 $t$ 替换为 `[M]` 得到 $x_t$；反向过程优化 $1/t$ 加权交叉熵变分上界（因训练覆盖 $t \to 0$，推理时可直接输入 $t=0$ 未加噪序列提取无偏双向表征）。从头生成时，模型从全掩码序列 $x_1$ 出发经 $T$ 步迭代恢复，每步对当前掩码集 $\mathcal{M}_t$ 预测温度缩放分布 $p_\theta^{\mathrm{temp}}(x_0^i \mid x_t) \propto p_\theta(x_0^i \mid x_t)^{1/\tau}$，并按置信度选取子集 $\mathcal{U}_t \subset \mathcal{M}_t$ 并行去掩码：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\!\left[\frac{1}{t}\sum_{i=1}^L \mathbb{I}[x_t^i = \mathrm{[M]}]\log p_\theta(x_0^i \mid x_t)\right], \quad
x_{t-1}^i = \begin{cases} \text{sample from } p_\theta^{\mathrm{temp}}(x_0^i \mid x_t), & i \in \mathcal{U}_t, \\ \mathrm{[M]}, & i \in \mathcal{M}_t \setminus \mathcal{U}_t, \\ x_t^i, & \text{otherwise}. \end{cases}
$$
