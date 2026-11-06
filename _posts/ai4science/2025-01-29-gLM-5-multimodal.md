---
layout: post
title: '基因组基础模型(Genome Foundation Model) V：多模态模型'
date: 2025-01-29
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm5-multimodal-000-cover.png'
tags: AI4S
---

> Genome Foundation Model part V: Multimodal Models.

**基因组基础模型（Genome Foundation Model, gFM）**，又称**基因组语言模型（Genomic Language Model, gLM）**，将 **DNA** 序列映射为长度为 $T$ 的离散词元序列 $z_{1:T}$，通过在大规模无标注基因组语料上拟合边缘概率或条件概率分布 $p_\theta(z)$，学习具备跨任务迁移能力的通用基因组表征。

本文是**基因组基础模型**系列的第五篇。全系列按“**分词 → 预训练 → 编码器 → 解码器 → 多模态**”的脉络组织：
- [I. DNA 分词策略](https://0809zheng.github.io/2025/01/01/gLM-1-token.html)：如何将碱基序列切分为词元。
- [II. 预训练策略](https://0809zheng.github.io/2025/01/08/gLM-2-object.html)：如何构造自监督信号从无标注基因组中学习。
- [III. 编码器框架](https://0809zheng.github.io/2025/01/15/gLM-3-encoder.html)：模型如何在长序列上混合上下文信息。
- [IV. 解码器框架](https://0809zheng.github.io/2025/01/22/gLM-4-decoder.html)：模型如何评估序列似然并生成新序列。
- **V. 多模态模型**：如何联合建模基因组序列、其他生物分子与人类自然语言。

基因组序列在生物体内的功能实现天然处于多模态语境之中：顺式调控元件与编码序列遵循中心法则转录为 **RNA** 并翻译为蛋白质，而实验测定与文献积累又为序列赋予了丰富的连续数值属性与自然语言功能注释。**基因组多模态模型**对这些模态的数据进行同时建模。

- **1. 统一词表建模（Unified Vocabulary Modeling）**
  - [DNAGPT：离散序列与连续数值的因果多任务建模](#p-004)
  - [LLaMA-Gene：扩展多模态 BPE 词表与 Alpaca 指令微调的基因对话模型](#p-llamagene)
  - [Omni-DNA：解码器的跨模态词表扩展与多任务指令统一](#p-omnidna)
  - [OmniNA：融合核酸序列与文本注释的自回归问答](#p-omnina)
  - [OmniGene-4：生物-语言-视觉混合专家模型与路由器级可解释性分解](#p-omnigene4)
  - [GenNA：跨模态 BPE 词表与自然语言注释引导的条件核苷酸生成](#p-genna)
  - [Carbon-X：混合 $6$-mer 词表中期训练与可验证奖励强化学习的 DNA-语言模型](#p-carbonx)
- **2. 编码器-LLM 桥接（DNA Encoder–LLM Bridge）**
  - [GeneChat：基于线性适配器桥接基因组编码器与大语言模型的问答框架](#p-genechat)
  - [ChatNT：英语问题感知感知器重采样与多任务基因组对话代理](#p-chatnt)
  - [BioReason：融合基因组编码器与大语言模型的强化学习多步生物推理](#p-bioreason)
- **3. 中心法则对齐（Central Dogma Alignment）**
  - [CD-GPT：共享多分子 BPE 词表与中心法则配对的跨模态自回归预训练](#p-cdgpt)
  - [ProDMM：非编码调控区-编码序列-蛋白质协同演化的统一编码-解码模型](#p-prodmm)
  - [NatureLM：多科学域实体词表扩展与中心法则交错序列联合预训练](#p-naturelm)
  - [Life-Code：逆转录/逆翻译核苷酸统一的中心法则模型](#p-lifecode)
  - [LucaOne：统一字符词表与四层级半监督注释的核酸-蛋白质编码器](#p-lucaone)
  - [C3P：启动子与冻结蛋白质表征的跨模态对比对齐](#p-c3p)
  - [MIMIC：分轨求和编码器-解码器与多模态中心法则联合生成模型](#p-mimic)

<a id="mm-unified"></a>
# 1. 统一词表建模（Unified Vocabulary Modeling）

**统一词表建模**将生物序列直接视为一种特殊的“外语”，通过扩充预训练大语言模型（或从头训练解码器）的词表，将离散 **DNA** 词元、连续生物物理数值与自然语言指令置于同一嵌入空间。给定输入序列 $x_{\mathrm{DNA}}$ 与自然语言提示 $\mathrm{Prompt}$，模型在单一自回归解码器内优化跨模态条件生成损失：

$$
\mathcal{L}_{\mathrm{Cross\text{-}CLM}} = -\sum_{t=1}^{T} \log p_\theta\big(y_t \mid y_{<t}, x_{\mathrm{DNA}}, \mathrm{Prompt}\big).
$$

该路线的优势在于架构简洁，单一主干即可同时处理序列分类、功能描述生成与多序列互作判别；核心挑战则在于如何缓解新词元并入引发的概率分布偏移，以及如何防止生物语料持续预训练覆盖原有的自然语言推理能力。

<a id="p-004"></a>
### ⚪ **DNAGPT**：离散序列与连续数值的因果多任务建模
- **paper**：[**DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks**](https://arxiv.org/abs/2307.05628)（*arXiv* 2023）

**DNAGPT** 在自回归 **Transformer** 解码器中引入双模态输入输出接口与结构化指令词表，将离散 **DNA** 序列与连续数值属性统一至同一因果生成框架，联合优化下一词元预测、序列翻转判别与 **GC** 含量回归。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-069-dnagpt.png)

**非重叠分词与结构化控制词表：** 序列采用非重叠 **$k$-mer** 分词。词表共包含六类符号：**序列词元**表示 **DNA** 片段；**数值词元**承载连续标量属性；**分类词元**标识二元真假；**指令词元**指定物种来源或输出模态（分类或数值回归）；**连接词元**中 $+$ 表示多模态特征聚合，$=$ 界定条件输入与目标输出的映射关系；另设特殊符号作为**预留词元**。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-027-680a1e1b.png)

**双流嵌入与联合预训练目标：** 主干网络为 **GPT** 解码器。离散词元通过常规嵌入层查表映射，连续数值则通过由卷积与残差全连接层构成的数值嵌入层投影至同一隐空间；两路嵌入拼接后输入主干，输出端按模态分流至线性分类头或标量回归头。预训练通过指令模板将三项自监督任务拼接在同一上下文窗口内：（1）**下一词元预测（NTP）**优化离散序列似然；（2）**序列顺序预测**随机反转 **DNA** 片段并判别序列是否翻转；（3）**GC 含量回归**随机截取动态长度子片段，经回归头最小化 **GC** 比例的均方误差损失。

<a id="p-llamagene"></a>
### ⚪ **LLaMA-Gene**：扩展多模态 BPE 词表与 Alpaca 指令微调的基因对话模型
- **paper**：[**LLaMA-Gene: A General-purpose Gene Task Large Language Model Based on Instruction Fine-tuning**](https://github.com/maris205/llama-gene)（*arXiv* 2024）

**LLaMA-Gene** 将 **DNA** 与蛋白质的 **BPE** 子词词表直接并入预训练 **LLaMA** 大语言模型，通过混合生物学文献的参数高效持续预训练与统一的 **Alpaca** 格式指令微调，无需引入特殊模态边界标记即可将多种生物序列分析任务统一为生成式对话问答。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-113-llamagen.png)

**两步多模态词表扩展与防遗忘持续预训练：** 模型在原 **LLaMA** 自然语言词表（$32{,}000$ 词元）基础上分两步扩充：先后在多模式生物 **DNA** 语料与 **UniProt** 蛋白质语料上独立训练各含 $30{,}000$ 词元的 **BPE** 词表，去重合并后构建总容量约 $91{,}000$ 的统一词表。为在注入生物序列语法的同时避免自然语言推理能力发生灾难性遗忘，模型按 $1:1:1$ 的数据量混合 **DNA** 片段、蛋白质序列与生物学英文研究论文（**S2ORC**），采用 **LoRA** 适配注意力与前馈投影层，同时全量放开词嵌入层与语言模型头完成自回归持续预训练。

**多任务 Alpaca 指令模板重构：** 在指令微调阶段，模型复用单一因果语言建模头，将五类异构生物任务统一映射为三元组 $(\mathrm{Instruction}, \mathrm{Input}, \mathrm{Output})$ 的 **Alpaca** 自然语言提示模板，最小化条件自回归损失：

$$
\mathcal{L}_{\mathrm{IFT}} = -\sum_{t=1}^{T} \log p_\theta\big(y_t \mid y_{<t}, \mathrm{Instruction}, \mathrm{Input}\big).
$$

各类任务的转换规则包括：（1）**序列分类**（如启动子检测）将候选类别名嵌入指令，要求直接输出目标类别文本；（2）**词元级结构预测**（如二级结构）截取目标词元及其前序 $30\text{ bp}$ 滑窗作为输入，按频次多数投票将单碱基/残基结构标签对齐至末端 **BPE** 词元；（3）**多序列关联判别**（如中心法则转录翻译一致性或蛋白互作）以 `\n###\n` 拼接两条序列并生成二元判别词；（4）**基因功能预测**直接以序列为输入生成 **GO（Gene Ontology）** 功能描述文本；（5）**连续回归任务**（如基因表达水平）按中位数阈值离散化为 **high / low** 二元文本分类。

<a id="p-omnidna"></a>
### ⚪ **Omni-DNA**：解码器的跨模态词表扩展与多任务指令统一
- **paper**：[**Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning**](https://arxiv.org/abs/2502.03499)（*arXiv* 2025）

**Omni-DNA** 基于 **OLMo** 自回归 **Transformer** 解码器架构，通过“去重多物种语料自回归预训练 + 跨模态多任务指令微调”，将多项表观遗传分类、**DNA** 功能文本生成（**DNA2Func**）及连续变量离散化映射（**DNA2Image**）统一于单一生成式模型中。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-093-omnidna.png)

**跨模态词表扩展与分布偏移：** 预训练阶段采用 **BPE** 分词完成下一词元预测训练。在有监督微调阶段，模型将不同模态任务整合为统一的指令—响应格式 $$\tilde{x} = (x, \mathrm{Prompt}_k)$$，并将原基因组词表 $V_o$ 扩展为并入任务新词元 $V_z$（如自然语言词汇、类别标签或量化视觉词元）的联合词表。给定前缀隐状态 $h_{t-1}$ 与词元嵌入 $e$，目标词元条件概率为：

$$
p_\theta(y_t \mid y_{<t}, x) = \frac{\exp(h_{t-1}^\top e_{y_t})}{\sum_{m \in V_o} \exp(h_{t-1}^\top e_m) + \sum_{n \in V_z} \exp(h_{t-1}^\top e_n)}.
$$

**关键标签词元重复、NEFTune 与连续响应离散化：** 为缓解新词元引入导致的原词元概率偏移与遗忘，微调过程结合三项机制：（1）**关键标签词元重复**：将分类任务响应序列中的类别标签词元重复 $\alpha$ 次以强化监督信号，并在联合微调多项任务时激发协同增益；（2）**NEFTune**：向词元嵌入注入强度为 $\beta$ 的均匀随机噪声以抑制过拟合；（3）**连续响应离散化**：利用预训练 **VQ-VAE** 将图像或高维连续信号压缩为离散潜词元序列并入 $V_z$，统一转化为自回归生成任务。

<a id="p-omnina"></a>
### ⚪ **OmniNA**：融合核酸序列与文本注释的自回归问答
- **paper**：[**A foundation model for nucleotide sequences**](https://doi.org/10.1093/nar/gkag083)（*Nucleic Acids Research* 2026）

**OmniNA** 基于 **LLaMA** 自回归解码器架构，在预训练阶段将跨物种核酸序列（**DNA** 与 **RNA**）及其对应的自然语言生物学注释联合建模，并通过统一的问答（**QA**）指令微调范式，支持纯序列输入的基因组判别与突变效应评估。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-095-omnina.png)

**序列—注释双模态语料与联合 BPE 分词：** 模型利用 **SentencePiece BPE** 在跨物种核酸序列与标准化文本注释（包含分类学层级、基因符号与功能描述）上联合训练子词词表。每条预训练样本按固定模板组织为“指令 + 核酸序列 + 文本注释”的有序词元流，通过自回归下一词元预测联合学习序列语法与注释语义。

**无注释推理的 QA 微调与饱和突变分析：** 下游适配阶段将各项任务统一重构为“固定指令 + 核酸序列 + 任务问题 + 期望回答”的四元组格式，微调时仅对“期望回答”词元计算自回归损失；部署推理时完全脱离先验文本注释，仅输入纯核酸序列与问题即可生成预测回答。在饱和突变分析中，模型通过计算野生型序列 $s_{\mathrm{WT}}$ 与突变型序列 $s_{\mathrm{mut}}$ 在对应微调任务正类目标上的对数似然相对变化量，量化单碱基突变的破坏性：

$$
\mathrm{Effect}(s_{\mathrm{WT}}, s_{\mathrm{mut}}) = \left\lvert \frac{\mathrm{LL}(s_{\mathrm{WT}}) - \mathrm{LL}(s_{\mathrm{mut}})}{\mathrm{LL}(s_{\mathrm{WT}})} \right\rvert, \quad
\mathrm{LL}(s) = \sum_{j} \log \Pr(x_j = s_j \mid s).
$$

<a id="p-omnigene4"></a>
### ⚪ **OmniGene-4**：生物-语言-视觉混合专家模型与路由器级可解释性分解
- **paper**：[**OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability and Modality-Invariant Transfer**](https://doi.org/10.64898/2026.05.12.724542)（*bioRxiv* 2026）

**OmniGene-4** 基于稀疏混合专家（**MoE**）架构扩展生物序列与结构字母表词表，通过包含路由器投影层的 **QLoRA** 持续预训练（**CPT**）与双头指令微调（**SFT**），首次在离散路由器层面量化了表征分化与输出对齐的阶段贡献，并进一步扩展至融合分子与医学图像的八模态通用模型（**OmniGene-4-MM**）。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-114-omnigene4.png)

**带前缀多模态词表扩展与双头联合训练：** 模型以 **Gemma-4-26B-A4B**（每层 $128$ 个专家、**Top**-$8$ 路由）为骨干，注入带显式模态前缀的 $28{,}028$ 个生物词元（涵盖 **DNA/Protein BPE**、$20$ 类 **Foldseek 3Di** 三维结构字母与 $8$ 类 **DSSP** 二级结构标签，新词元嵌入由子词片段均值初始化）。训练分两阶段推进：（1）**CPT 阶段**在按 $3:1$ 混合生物序列（文本+氨基酸+**DSSP**+**3Di**）与通用文本（**OpenWebText**）的语料上，对注意力、前馈层及路由器投影层施加 **4-bit QLoRA** 更新；（2）**SFT 阶段**采用对提示词屏蔽损失的 **Alpaca** 模板微调，并在末层隐状态上并行挂载 **3Di** 与 **DSSP** 逐残基分类头，按 $$0.5\mathcal{L}_{\mathrm{gen}} + 0.5\mathcal{L}_{\mathrm{cls}}$$ 联合优化对话生成与结构预测。

**路由器级阶段分解：** 为剖析多模态路由机制，研究在全部 $30$ 层路由器上挂载前向钩子，统计任务 $t$ 在第 $\ell$ 层激活专家 $e$ 的路由分布 $p_{t,\ell} \in \Delta^{128}$，定义专家特化度 $s_{t,e}^{(\ell)}$ 与跨任务平均 **Jensen–Shannon** 散度 $J_\ell$：

$$
s_{t,e}^{(\ell)} = \log\!\left(\frac{p_{t,e}^{(\ell)}}{\bar{p}_e^{(\ell)}}\right), \quad
J_\ell = \binom{8}{2}^{-1} \sum_{t \neq t'} \mathrm{JS}\!\left(p_{t,\ell} \,\|\, p_{t',\ell}\right).
$$

跨检查点分解表明：**CPT** 贡献了 $96\%$ 的跨任务路由分化（集中在网络中层 $\mathrm{L11}\text{--}\mathrm{L22}$ 重塑模态表征，使生物词元收敛至专用专家而自然语言扩展至正交专家集以避免灾难性遗忘），**SFT** 仅贡献 $4\%$（集中在末两层对齐输出格式）；而在远程蛋白质同源判别内部，正负样本对间的路由散度极低（$<0.04$），证实序列级判别发生在被激活专家的内部计算而非路由门控处。

<a id="p-genna"></a>
### ⚪ **GenNA**：跨模态 BPE 词表与自然语言注释引导的条件核苷酸生成
- **paper**：[**GenNA: Conditional generation of nucleotide sequences guided by natural-language annotations**](https://doi.org/10.64898/2026.04.22.720063)（*bioRxiv* 2026）

**GenNA**（$3.6\text{B}$ 参数，基于配备 **GQA**、**RoPE** 与 **SwiGLU** 的 **Qwen3** 自回归解码器）在跨越 $2{,}221$ 个真核物种、约 $4{,}160$ 亿字符的多模态基因中心语料上从头预训练，通过统一的跨模态 **BPE** 词表将自然语言功能注释、物种上下文及 **XML** 结构标签与核苷酸序列联合建模，统一实现零样本变异与系统发育感知以及自然语言引导的从头序列设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-129-genna.png)

**跨模态 BPE 分词与动态提示掩码预训练：** 预训练语料聚焦 **NCBI RefSeq** 注释基因区（随机向两侧延伸 $0\text{--}100\text{ bp}$），将每条样本按统一前缀模板 `[Molecule Type], [Species Name], [Gene Symbol], [Functional Annotation]<seq>` 衔接含 `<gene>`、`<cds>`、`<tRNA>` 等 **XML** 边界标签的核苷酸序列，并打包至 $4{,}096$ 词元（约 $20\text{ kb}$）。专用跨模态 **BPE** 分词器（容量 $6{,}000$）将高频英文功能短语、结构标签与局部序列基序映射至共享离散词元空间，并在训练后期以 $0.5$ 概率随机移除物种或基因标识符，迫使模型依赖功能语义与统计上下文优化因果语言建模损失与条件困惑度：

$$
\mathcal{L}_{\mathrm{CLM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(y_t \mid C, y_{<t}), \quad
\mathrm{PPL}(S \mid C) = \exp\!\big(\mathcal{L}_{\mathrm{CLM}}(S \mid C)\big).
$$

**零样本困惑度扰动评估与可控序列生成：** 在零样本评估中，模型直接利用全序列条件困惑度差值 $\Delta\mathrm{PPL}$ 量化突变破坏性（同义 $<$ 错义 $<$ 无义 $<$ 移码）、序列—功能语义失配及跨物种标签替换矩阵 $S_i \mid C_j$ 的系统发育距离：

$$
\begin{aligned}
&\Delta\mathrm{PPL}_{\mathrm{mut}} = \mathrm{PPL}(S_{\mathrm{mut}} \mid C) - \mathrm{PPL}(S_{\mathrm{wt}} \mid C), \\
&\Delta\mathrm{PPL}_{i,j} = \mathrm{PPL}(S_i \mid C_j) - \mathrm{PPL}(S_i \mid C_i).
\end{aligned}
$$

在条件生成阶段，模型结合温度缩放 $\tau$、核采样（**Top**-$p$）与 $n$-gram 重复惩罚，支持从物种特异背景、结构化非编码 **RNA**（如自发符合摆动配对规则的 **tRNA** 与 **rRNA**）或组蛋白家族提示词出发可控生成目标序列。

<a id="p-carbonx"></a>
### ⚪ **Carbon-X**：混合 $6$-mer 词表中期训练与可验证奖励强化学习的 DNA-语言模型
- **paper**：[**Carbon-X: When Natural Language Meets DNA**](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release)（*preprint* 2026）

**Carbon-X** 在 **Qwen3-14B** 自回归解码器中扩展定长 $6$-**mer** 基因组词表，通过 $300\text{B}$ 词元的跨模态中期训练、确定性验证器校验的有监督微调以及可验证奖励强化学习（**RLVR**），在单一模型内统一了长窗口变异致病性解释、序列问答与 **DNA** 生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-118-carbonx.png)

**混合 $6$-mer 分词器与防遗忘中期训练：** 为避免文本 **BPE** 分词器将 **DNA** 切分为不规则碎片，模型采用混合分词器：自然语言沿用 **Qwen3 BPE**，`<dna>...</dna>` 边界内的基因组序列则采用步长为 $6$ 的非重叠 $6$-**mer** 分词（扩充 $4^6=4{,}096$ 个六碱基词元及边界与残缺片段控制符，相比原文本分词器将序列长度压缩约 $3$ 倍），共享同一输入输出嵌入矩阵。针对高学习率加速 **DNA** 收敛但引发严重英文遗忘的梯度冲突，中期训练在 $300\text{B}$ 混合语料（整合 **OpenGenome2** 与 **GENERator** 原始基因组、文献与网页挖掘的 **DNA**—文本配对数据、**RefSeq** 合成分析样本及通用英文指令回放）上，通过扩展模型容量（$14\text{B}$）并配合低峰值余弦衰减学习率，在不损失自然语言推理能力的前提下注入序列语法。

**确定性序列验证器与两阶段后训练（Grounded SFT + RLVR）：** 为解决自蒸馏或通用大模型蒸馏时频繁出现的基因误认与数据库引用幻觉，后训练引入基于输入序列事实核查的确定性验证器（**Deterministic Verifiers**）：
1. **事实校验 SFT**：利用领域专家规则编写的确定性验证器直接针对输入 **DNA** 序列核查候选回答中的基因坐标、突变位点与剪接基序声明，过滤无依据断言以构建高保真监督响应集；
2. **可验证奖励强化学习（RLVR）**：从 **SFT** 检查点出发，在编码区与复杂非 **SNV** 变异效应预测、长窗口（$8{,}192\text{ bp}$）单碱基变异评估（**FrontierVEP**）、序列恢复与位置定位等任务上，以任务专用确定性验证器提供的二元/标量奖励直接优化生成策略，使模型既能直接输出可解释的分子致病机制（如识别破坏剪接受体 **AG** 二核苷酸的突变），又可作为专业子代理辅助外部通用大模型解析复杂序列。


<a id="mm-bridge"></a>
# 2. 编码器-LLM 桥接（DNA Encoder–LLM Bridge）

受视觉语言模型（**VLM**）启发，**编码器—LLM 桥接**架构直接复用成熟的预训练 **DNA** 基础模型作为序列感知前端，通过轻量投影层或交叉注意力重采样器，将连续的基因组上下文表征注入自回归大语言模型：

$$
X_{\mathrm{LLM}} = \mathrm{Concat}\!\Big(\mathrm{Adapter}\big(f_{\mathrm{DNA}}(S_{\mathrm{DNA}})\big),\; E_{\mathrm{LLM}}(Q_{\mathrm{Text}})\Big).
$$

该范式完整保留了专用基因组编码器的单碱基或 $k$-**mer** 局部敏感性，同时直接调用现代 **LLM** 的指令跟随与思维链（**Chain-of-Thought, CoT**）多步逻辑推理能力。

<a id="p-genechat"></a>
### ⚪ **GeneChat**：基于线性适配器桥接基因组编码器与大语言模型的问答框架
- **paper**：[**GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction**](https://github.com/Shashi-Sekar/GeneChat)（*ICML 2025 Workshop FM4LS*）

**GeneChat** 采用多模态桥接架构，通过非重叠窗口池化与线性适配器将基因组编码器（**DNABERT-2**）与指令微调大语言模型（**Vicuna-13B**）对接，直接从最长 $160\text{ kb}$ 的原始基因核苷酸序列生成自由文本形式的功能描述。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-117-genechat.png)

**重叠分块编码与线性适配器投影：** 给定来自 **NCBI** 数据库的基因核苷酸序列 $x_g$，针对超出编码器单次窗口上限的长序列，模型先按 $512\text{ nt}$ 窗口、相邻块保留 $10\text{ nt}$ 重叠的方式切分并送入 **DNABERT-2** 提取上下文嵌入 $\mathbb{R}^{l \times 768}$。随后沿序列维在大小为 $k$ 的非重叠窗口上执行平均池化，将特征压缩为 $h(x_g) \in \mathbb{R}^{(l/k) \times 256}$，再通过可学习的线性适配器 $W \in \mathbb{R}^{256 \times 5120}$ 投影对齐至 **Vicuna-13B** 的词嵌入空间：

$$
h_g = h(x_g) \cdot W \in \mathbb{R}^{(l/k) \times 5120}.
$$

**占位符注入与端到端微调：** 模型将文本输入构造为对话模板 `Human: <Gene> GeneHere </Gene> Prompt Assistant:`，把除 `GeneHere` 外的辅助提示词元映射为嵌入序列 $h_{\mathrm{aux}}$，并在 `GeneHere` 占位符处直接替换为对齐后的基因序列嵌入 $h_g$。训练过程仅对目标回答序列 $x_a = (x_a^{(1)}, \dots, x_a^{(l)})$ 计算因果自回归负对数似然损失，同时全参数微调 **DNABERT-2** 编码器、从头训练线性适配器，并对 **Vicuna-13B** 的查询与值投影矩阵施加 **LoRA**（$\mathrm{rank}=8$）参数高效微调：

$$
p(x_a \mid x_g, x_{\mathrm{aux}}) = \prod_{i=1}^{l} p_\theta\big(x_a^{(i)} \mid x_g, x_{\mathrm{aux}}, x_a^{<i}\big).
$$


<a id="p-chatnt"></a>
### ⚪ **ChatNT**：英语问题感知感知器重采样与多任务基因组对话代理
- **paper**：[**A multimodal conversational agent for DNA, RNA and protein tasks**](https://doi.org/10.1038/s42256-025-01047-1)（*Nature Machine Intelligence* 2025）

**ChatNT** 将预训练 **Nucleotide Transformer（NT-v2, 500M）** 编码器与冻结的 **Vicuna-7B** 英文解码器桥接，通过问题条件化的感知器重采样器，将横跨 **DNA**、**RNA** 与蛋白质（输入对应编码序列 **CDS**）的分类与定量回归任务统一为自然语言生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-116-chatnt.png)

**英语问题感知感知器重采样与跨模态解冻训练：** 给定输入序列 $s$（采用 $6$-**mer** 分词，最长填充至 $2{,}048$ 词元即约 $12\text{ kb}$），**NT-v2** 编码器提取词元级隐状态并经线性层对齐至语言模型维度。为避免固定潜在查询在多任务下产生信息瓶颈，投影模块引入包含 $K=64$ 个可学习潜在查询 $Q_{\mathrm{lat}}$ 的英语问题感知感知器重采样器（**English-aware Perceiver Resampler**）：先令 $Q_{\mathrm{lat}}$ 与英文问题词元嵌入 $E_{\mathrm{question}}$ 执行交叉注意力以注入任务上下文，再与 **DNA** 序列嵌入 $H_{\mathrm{DNA}}$ 执行门控交叉注意力，将变长序列压缩为 $K$ 个任务相关的软提示向量，替换用户指令中的 **DNA** 位置标签。

$$
\begin{aligned}
&\tilde{Q} = \mathrm{CrossAttn}(Q_{\mathrm{lat}}, E_{\mathrm{question}}), \\
&Z_{\mathrm{DNA}} = \mathrm{CrossAttn}(\tilde{Q}, H_{\mathrm{DNA}}) \in \mathbb{R}^{K \times d_{\mathrm{llm}}}.
\end{aligned}
$$

不同于常规多模态模型冻结编码器的做法，**ChatNT** 保持 **Vicuna-7B** 解码器完全冻结以保留对话能力，同时将统一交叉熵损失的梯度反向传播至重采样投影器与 **NT-v2** 编码器（对定量回归任务，直接将连续数值拆解为十进制数字字符序列进行自回归预测）。

**基于困惑度的概率校准与跨模态梯度归因：** 为克服生成式对话模型无法直接输出分类置信度的局限，模型在二分类任务中引入事后**困惑度分类器（Perplexity-based Classifier）**：给定问题 $x$ 与序列 $s$，分别计算模型在 $N$ 个预设正类回答模板 $y_i^{\mathrm{pos}}$ 与负类回答模板 $y_i^{\mathrm{neg}}$ 上的平均困惑度以导出分类对数几率，并通过在训练集上拟合 **Platt** 逻辑回归模型校准预测概率：

$$
\mathrm{pp}_\theta^{\mathrm{pos}}(x, s) = \frac{1}{N}\sum_{i=0}^{N-1} \exp\!\left(\sum_j p_\theta\big((y_i^{\mathrm{pos}})_j \mid x, s, y_i^{\mathrm{pos}}\big) \log p_\theta\big((y_i^{\mathrm{pos}})_j \mid x, s, y_i^{\mathrm{pos}}\big)\right).
$$

此外，通过将英文解码器输出词元（如 **yes / no**）的梯度经重采样器一路反向传播至 **DNA** 编码器输入端计算 $\vert\nabla_{\mathrm{emb}} \odot \mathrm{Input}\vert$ 显著性图谱，无需任务专用头即可从自然语言回答直接定位剪接供体 **GT**、受体 **AG** 二核苷酸及启动子 **TATA-box** 等保守基序。


<a id="p-bioreason"></a>
### ⚪ **BioReason**：融合基因组编码器与大语言模型的强化学习多步生物推理
- **paper**：[**BIOREASON: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model**](https://arxiv.org/abs/2505.23579)（*NeurIPS* 2025）

**BioReason** 将冻结的 **DNA** 基础模型与自回归 **LLM** 在嵌入层直接桥接，通过有监督微调与群组相对策略优化（**GRPO**）强化学习，激励模型从成对基因组变异序列出发生成分步分子机制推理链并预测下游表型效应。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-115-bioreason.png)

**冻结 DNA 编码器与跨模态序列拼接：** 框架接收一条或多条 **DNA** 序列（如配对的参考与突变序列 $S_{\mathrm{DNA},k}$）及包含通路或基因上下文的文本查询 $Q_{\mathrm{TEXT}}$。冻结权重的 **DNA** 基础模型（如 **Evo 2**、**Nucleotide Transformer** 或有监督模型 **Enformer**）首先提取逐词元上下文表征 $$E_{\mathrm{DNA},k} \in \mathbb{R}^{L'_k \times d_{\mathrm{dna}}}$$，经可学习线性层 $$\mathrm{Proj}: \mathbb{R}^{d_{\mathrm{dna}}} \to \mathbb{R}^{d_{\mathrm{llm}}}$$ 投影并沿序列维堆叠为特征块 $E'_{\mathrm{DNA}}$；随后与边界控制符嵌入及 **LLM** 文本嵌入 $E_{Q_{\mathrm{text}}}$ 拼接为统一多模态输入，统一施加 **RoPE** 位置编码后送入 **Qwen3** 主干：

$$
\begin{aligned}
&X_{\mathrm{LLM}} = \big(e_{\text{<dna_start>}},\, E'_{\mathrm{DNA}},\, e_{\text{<dna_end>}},\, E_{Q_{\mathrm{text}}}\big), \\
&\mathcal{L}_{\mathrm{SFT}} = -\sum_{i} \log P(y_i \mid y_{<i}, X_{\mathrm{LLM}}; \theta_{\mathrm{LLM}}),
\end{aligned}
$$

其中 **SFT** 阶段采用 **LoRA** 微调 **LLM**，并通过掩码仅对 `<think>...</think>` 推理过程及最终回答词元计算自回归损失。

**通路推理数据集与 Dr. GRPO 强化学习对齐：** 基于整合 **KEGG** 分子互作网络（符号化表示为 `GENE1+GENE2 -> GENE3 -| GENE4`）与 **ClinVar/OMIM** 临床变异构建的多步因果推理数据集，模型进一步采用无偏变体 **Dr. GRPO** 优化推理生成策略。对每个输入 $X_{\mathrm{LLM}}$ 从旧策略采样 $G$ 条包含思维链与最终结论的候选输出 $\{o_1, \dots, o_G\}$，利用包含答案正确性（$+2.0$）、精炼度（$\le 10$ 词，$+0.5$）与 `` 标签格式闭合性（$+0.5$）的确定性复合奖励计算无标准差缩放的组内相对优势 $A_i = r_i - \mathrm{mean}(\{r_1, \dots, r_G\})$，最大化带 **KL** 正则的截断代理目标：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\!\left(\frac{\pi_\theta(o_i \mid X_{\mathrm{LLM}})}{\pi_{\theta_{\mathrm{old}}}(o_i \mid X_{\mathrm{LLM}})}A_i,\, \mathrm{clip}\!\left(\frac{\pi_\theta(o_i \mid X_{\mathrm{LLM}})}{\pi_{\theta_{\mathrm{old}}}(o_i \mid X_{\mathrm{LLM}})}, 1-\epsilon, 1+\epsilon\right)A_i\right) - \beta D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})\right)\right].
$$



<a id="mm-centraldogma"></a>
# 3. 中心法则对齐（Central Dogma Alignment）

除了与人类自然语言对齐，基因组序列的另一大天然监督源是分子生物学**中心法则（DNA $\to$ RNA $\to$ Protein）**本身。由于蛋白质序列受更强的三维结构与功能纯化选择约束，蛋白质语言模型（**pLM**）往往能学到比纯 **DNA** 模型更清晰的功能流形。通过跨分子配对生成、多层级功能注释联合监督或双塔 **InfoNCE** 对比对齐，可将蛋白质与转录组层的先验知识反向注入基因组表征中。

<a id="p-cdgpt"></a>
### ⚪ **CD-GPT**：共享多分子 BPE 词表与中心法则配对的跨模态自回归预训练
- **paper**：[**CD-GPT As a Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma**](https://doi.org/10.1101/2024.06.24.600337)（*bioRxiv* 2025）

**CD-GPT** 构建统一的跨分子子词词表，并在旋转位置编码自回归 **Transformer** 解码器上通过三阶段课程预训练显式对齐中心法则的转录与翻译映射，统一支持基因组元件判别、跨分子互作预测与逆向翻译生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-112-cdgpt.png)

**跨分子混合 BPE 词表与三阶段中心法则预训练：** 模型按 **DNA**、**RNA** 与蛋白质长度比例以 $5:3:1$ 混合多物种序列，通过 **SentencePiece BPE** 构建容量为 $64{,}000$（词元长 $1\sim 16$）的共享多分子词表。预训练分为三个递进阶段：
1. **单分子预训练（Stage 1）**：在独立的 **DNA**、**RNA** 与蛋白质单序列上优化下一词元预测损失 $$\mathcal{L}_{\mathrm{ntp}} = -\sum \log p_\theta(u_i \mid u_{<i})$$，并以 $5\%$ 的概率随机回退至单核苷酸或单氨基酸字符级分词以保留残基级细粒度特征；
2. **中心法则配对预训练（Stage 2）**：基于 **GenBank** 注释提取具有真实转录与翻译关系的配对序列，通过模态与过程控制符拼接为单条因果指令流（如 `<DNA>...<splicing><mRNA><translate><:><mRNA>...<protein>...`），在自回归建模中拉近同源三元组在共享隐空间中的余弦距离；
3. **蛋白质结构多任务预训练（Stage 3）**：在 **PDB** 结构数据上联合优化下一词元预测、二级结构预测与残基接触图预测损失 $$\mathcal{L}_{\mathrm{total}} = \alpha \mathcal{L}_{\mathrm{ntp}} + \beta \mathcal{L}_{\mathrm{ss}} + \gamma \mathcal{L}_{\mathrm{contact}}$$。

**多范式输出头适配与跨分子生成：** 下游迁移根据任务类型挂载四类输出头：（1）**序列级分类**（如启动子、剪接位点及直接拼接双分子序列的 **RNA**—蛋白质互作预测）提取末位词元隐状态接入线性分类层；（2）**词元级预测**在因果主干顶部附加单层双向自注意力以融合右侧上下文；（3）**残基对预测**对注意力矩阵施加对称化与平均乘积校正（**APC**）后输入双层 **MLP**；（4）**条件生成任务**通过提示标签触发自回归解码，支持指定蛋白家族的从头蛋白质设计，以及给定 `<protein>...<reverse translate><:>` 提示直接生成符合天然密码子偏好与高翻译起始率的编码 **mRNA** 序列。

<a id="p-prodmm"></a>
### ⚪ **ProDMM**：非编码调控区-编码序列-蛋白质协同演化的统一编码-解码模型
- **paper**：[**Harnessing A Unified Multi-modal Sequence Modeling to unveil Protein-DNA Interdependency**](https://doi.org/10.1101/2025.02.26.640480)（*bioRxiv* 2025）

**ProDMM** 在 $1.29\times 10^8$ 组原核四元组（`<pre-NCDS, CDS, Protein, post-NCDS>`）上构建统一词表，结合双向掩码编码器（**ProDMM-Encoder**）与条件自回归解码器（**ProDMM-Seq2Seq**），联合捕获密码子偏好、顺式调控语法与蛋白质稳定性之间的跨模态依赖。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-120-prodmm.png)

**三模态粒度感知词表与四格式 MLM 编码器预训练：** 通用分词器构建包含 $97$ 个词元（$20$ 种大写单字母氨基酸、$64$ 种三联体密码子、$4$ 种小写单碱基 **a/t/c/g** 及 $9$ 个特殊控制符）的共享词表，对蛋白质与非编码序列（**NCDS**）按单字符分词，对 **CDS** 按三联体密码子分词。**ProDMM-Encoder** 采用配备 **RoPE**、**Post-LayerNorm** 与嵌入缩放的 $48$ 层双向 **Transformer**，将每条四元组等概率随机转化为四类序列格式之一（纯 **Protein**、纯 **CDS**、`{<cls>, pre-NCDS, <sep>, CDS, <sep>, post-NCDS, <sep>}` 或 `{<cls>, pre-NCDS, <sep>, Protein, <sep>, post-NCDS, <sep>}`），以 $15\%$ 掩码率优化掩码语言建模损失。

**伪困惑度零样本表型排序与跨模态条件生成：** 下游评估与设计分两路展开：（1）**零样本功能与通路产量预测**利用冻结编码器计算单序列或复合序列（如启动子—蛋白拼接序列 $S = \{q_{1:L_1}, c_{1:L_2}\}$ 或多基因代谢通路 `<NCDS-Protein>×4`）的伪困惑度与突变对数几率比（**Log-Odds Ratio**）：

$$
\begin{aligned}
&\mathrm{PP}_{\mathrm{cond}}(S) = \exp\!\left(-\frac{1}{L_1}\sum_{i=1}^{L_1} \log P(w_i \mid S_{\setminus i})\right), \\
&\mathrm{ME}(\mathcal{M}) = \sum_{i=1}^{k} \log \frac{P(S[p_i] = mt_i \mid S)}{P(S[p_i] = wt_i \mid S)}.
\end{aligned}
$$

（2）**ProDMM-Seq2Seq 条件生成**在冻结编码器顶部通过交叉注意力挂载 $12$ 层因果解码器，并扩充 `<aas2cds>`、`<cds2pre-ncds>` 与 `<cds2post-ncds>` 三个生成提示符，联合训练三项条件自回归任务：带遗传密码表掩码约束的蛋白质逆向翻译（$\text{Protein} \to \text{CDS}$）、**CDS** 引导的上下游调控元件定制生成（$\text{CDS} \to \text{pre-/post-NCDS}$）以及反向的启动子特异 **CDS** 协同设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-121-prodmm-app.png)

<a id="p-naturelm"></a>
### ⚪ **NatureLM**：多科学域实体词表扩展与中心法则交错序列联合预训练
- **paper**：[**Nature Language Model: Deciphering the Language of Nature for Scientific Discovery**](https://arxiv.org/abs/2502.07527)（*arXiv* 2025）

**NatureLM**（涵盖 $1\text{B}$、**Llama-3-8B** 稠密架构及 **Mixtral-8x7B** 稀疏混合专家架构）将 **DNA**、**RNA**、蛋白质、小分子与晶体材料全部序列化为“自然语言”，在 $143.8\text{B}$ 词元（其中 $90\%$ 为科学序列与跨域配对数据）上通过两阶段持续预训练与指令微调，统一支持基因组元件判别、向导 **RNA（gRNA）** 设计及靶点引导的跨分子生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-122-naturellm.png)

**多域边界标签词表与三类跨域交错语料：** 模型在原大语言模型词表基础上扩充专用科学词元（**DNA/RNA** 采用单核苷酸分词，词表各 $16$ 个；蛋白质采用单氨基酸分词，词表 $26$ 个；小分子采用正则化 **SMILES** 分词，词表 $1{,}401$ 个；晶体材料展平为元素重复加空间群 `<sgN>`，词表 $396$ 个），并以 `<dna>...</dna>`、`<rna>...</rna>`、`<protein>...</protein>` 与 `<mol>...</mol>` 显式界定模态边界。为打通模态壁垒，预训练构建了 $12.7\text{B}$ 词元的三类跨域序列：（1）**文献交错序列**：利用生物医学命名实体识别（**BERN2**）定位科学文献中的分子与蛋白质提及，直接在正文中内嵌其 **SMILES** 或 **FASTA** 序列；（2）**平行文本—实体对**：对齐 **NCBI**、**UniProt** 与 **PubChem** 中的序列与自然语言功能描述；（3）**中心法则替换序列**：从 **RefSeq** 提取基因组序列，将其中的编码外显子区间直接替换为对应的翻译蛋白质异构体序列，使翻译后的氨基酸序列与上下游侧翼 **DNA** 序列直接共现。

**两阶段防遗忘持续预训练与 DPO 强化后训练：** 为防止随机初始化的新增科学词元破坏预训练 **LLM** 的原有语言表征，持续预训练分两阶段推进：**Stage 1** 冻结全部主干网络，仅更新新增科学词元嵌入以实现平稳对齐；**Stage 2** 解冻全部参数联合优化自回归似然。后训练阶段首先在 $5.1\times 10^6$ 条单轮指令—响应对上仅针对响应区间计算有监督微调（**SFT**）损失，进而引入直接偏好优化（**DPO**）利用成对的正负属性改写样本 $(y_w, y_l)$ 强化多目标分子与序列优化能力：

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right].
$$

在基因组与核酸任务上，模型不仅能通过归一化 **Yes/No** 词元概率完成启动子与转录因子结合位点识别，还能直接根据输入靶标 `<dna>` 自动定位 **NGG PAM** 位点并反向互补生成高活性 **CRISPR crRNA**（$\text{DNA}\to\text{RNA}$），或给定靶标 `<protein>` 从头设计高亲和力结合 **RNA** 诱饵（$\text{Protein}\to\text{RNA}$）。


<a id="p-lifecode"></a>
### ⚪ **Life-Code**：逆转录/逆翻译核苷酸统一的中心法则模型
- **paper**：[**Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification**](https://arxiv.org/abs/2502.07299)（*arXiv* 2025）

**Life-Code**（$350\text{M}/500\text{M}$ 参数）通过逆转录与逆翻译将 **DNA**、**RNA** 与蛋白质全部重映射至统一的核苷酸序列空间，结合双任务预训练的密码子级分词器、双链等变的 **Gated DeltaNet**—自注意力混合编码器以及来自蛋白质语言模型的知识蒸馏，在单一模型内实现了跨三组学长序列的高效表征、零样本蛋白质适应度预测与中心法则关联判别。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-126-lifecode.png)

**三组学核苷酸统一分词与 $11:1$ 双链混合编码器：** 数据管线将 **RNA** 逆转录回基因组 **DNA**，并按高频标准密码子表 $$g:\Sigma\to\{\mathrm{A,T,C,G}\}^3$$ 将长度为 $L$ 的氨基酸序列逆翻译为长度为 $3L$ 的 **cDNA** 序列（短 **CDS**—氨基酸对通过分隔符 `<S>` 打包拼接）。**Life-Code 分词器**先经单层 **Gated DeltaNet（GDN）**、核宽为 $3$ 的一维卷积与步长为 $3$ 的展开操作将连续三联体 $x_{3j+1:3j+3}$ 压缩为密码子级嵌入 $e_j$，并通过对称反卷积解分词器重建掩码碱基（$$\mathcal{L}_{\mathrm{MLM}}$$）与双层 **MLP** 翻译器预测对应氨基酸（$$\mathcal{L}_{\mathrm{Trans}}$$）完成预训练。随后的 $24$ 层**双向混合编码器**将特征沿通道维对半拆分为正反双链 $(E^{(+)}, E^{(-)})=\mathrm{Split}(E)$，共享参数并行处理正向与反转序列后拼接输出 $H=[f_\theta(E^{(+)}), f_\theta(\mathrm{Reverse}(E^{(-)}))]$；主干按 $11:1$ 的比例交错堆叠线性复杂度的 **GDN** 块与标准多头自注意力块，在保持高吞吐量的同时兼顾长程上下文建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-127-lifecode-preprocess.png)

**三任务对数和联合预训练与蛋白质知识蒸馏：** 固定分词器的解码与翻译头后，编码器在掩码 **DNA** 重建（$$\mathcal{L}_{\mathrm{MLM}}$$）与 **CDS**—氨基酸翻译（$$\mathcal{L}_{\mathrm{Trans}}$$）基础上，引入以冻结 **ESM2-650M** 为教师的词元级知识蒸馏（$$\mathcal{L}_{\mathrm{KD}}$$）：利用双层 **MLP** 蛋白质解码器 $D(\cdot)$ 将编码器输出映射至教师特征维度，对齐教师分布 $$P_{\theta'}(h_j^{\mathrm{PLM}})$$，并采用对数和自适应多任务加权策略联合优化三项损失（防止单一任务主导梯度）：

$$
\begin{aligned}
&\mathcal{L}_{\mathrm{KD}} = -\sum_{j=1}^{L} P_{\theta'}\!\left(h_j^{\mathrm{PLM}}\right)^\top \log P_\theta\!\left(D\big(h_j^{\mathrm{LifeCode}}\big)\right), \\
&\mathcal{L}_{\mathrm{total}} = \log(\mathcal{L}_{\mathrm{MLM}}+\epsilon) + \log(\mathcal{L}_{\mathrm{Trans}}+\epsilon) + \log(\mathcal{L}_{\mathrm{KD}}+\epsilon).
\end{aligned}
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-128-lifecode-pretrain.png)

<a id="p-lucaone"></a>
### ⚪ **LucaOne**：统一字符词表与四层级半监督注释的核酸-蛋白质编码器
- **paper**：[**Generalized biological foundation model with unified nucleic acid and protein language**](https://doi.org/10.1038/s42256-025-01044-4)（*Nature Machine Intelligence* 2025）

**LucaOne**（$1.8\text{B}$ 参数，$20$ 层 **Pre-LN** 与 **RoPE** 双向 **Transformer** 编码器）在跨越 $169{,}861$ 个物种的核酸（**DNA/RNA**）与蛋白质序列上构建统一字符级词表，通过自监督掩码重建与八项跨四个尺度的半监督注释任务联合预训练，在无需显式配对语料的条件下涌现出中心法则转录翻译对齐与跨分子互作表征能力。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-125-lucaone.png)

**去歧义统一词表与四层级十任务联合预训练：** 为避免核苷酸与氨基酸的单字母符号重叠，分词器构建容量为 $39$ 的统一字符词表（将核酸 **A, T/U, C, G, N** 映射为 **1–5**，保留大写字母供标准与稀有氨基酸使用），并叠加二元分子类型嵌入（核酸置 $0$、蛋白质置 $1$，最大窗口 $1{,}280$）。预训练在未配对的 **RefSeq** 核酸（含正反双链）与 **UniProt / ColabFoldDB / PDB** 蛋白质数据上联合优化四个层级的十项任务：（1）**词元级**随机遮蔽 $15\%$ 位点执行核酸与蛋白质掩码恢复；（2）**跨度级**逐位点分类核酸 $8$ 类基因组功能区以及蛋白质的活性/结合位点、同源超家族与结构域；（3）**序列级**预测核酸与蛋白质的分类学目级归属及 **UniProt** 功能关键词多标签分类；（4）**结构级**回归蛋白质残基链内归一化的 $C_\alpha$ 三维空间坐标。

在下游中心法则 **DNA**—蛋白质配对判别、非编码 **RNA**—蛋白质互作（**ncRPI**）等异构双序列任务中，直接冻结主干提取两路表征，经最大值池化或值级注意力池化（$\mathrm{Pool}$）拼接后输入单层全连接网络，即可在少样本设定下显著超越独立预训练的 **DNABERT-2 + ESM2-3B** 组合。


<a id="p-c3p"></a>
### ⚪ **C3P**：启动子与冻结蛋白质表征的跨模态对比对齐
- **paper**：[**C3P: Contrastive promoter-protein pretraining yields representations capturing bacterial gene regulation**](https://arxiv.org/abs/2605.25242)（*arXiv* 2026）

**C3P** 借鉴 **CLIP/LiT** 跨模态对比范式，以冻结的预训练蛋白质语言模型（**ESM-2**）为监督信号，在共享隐空间对齐细菌启动子与其下游编码蛋白，学习捕获共调控规律的启动子表征。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-110-c3p.png)

**操纵子感知配对与冻结蛋白质双塔：** 模型从跨物种细菌基因组中提取启动子—蛋白质正样本对：截取编码序列（**CDS**）起始密码子前至少 $100\text{ bp}$ 的非编码区作为启动子；若同链相邻基因间距不超过 $20\text{ bp}$，则视为同一操纵子（**Operon**）并将内部所有蛋白质与首个基因的启动子配对。架构包含非对称双塔：**启动子编码器**对序列执行重叠 $3$-**mer** 分词，经带 **RoPE** 的双向 **Transformer** 提取 **CLS** 表征；**蛋白质编码器**直接复用完全冻结的 **ESM-2** 末层均值池化表征，在降低开销的同时防止表征坍缩。

**对称 InfoNCE 对齐与零样本共调控检索：** 两路输出经线性投影与 $L_2$ 归一化得到 $z_i^{\mathrm{prom}}$ 与 $z_i^{\mathrm{prot}}$，通过对称 **InfoNCE** 损失拉近配对样本并推远批次内错配负样本（$\tau$ 为可学习温度）：

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left( \log \frac{\exp(z_i^{\mathrm{prom}} \cdot z_i^{\mathrm{prot}} / \tau)}{\sum_{j=1}^{N} \exp(z_i^{\mathrm{prom}} \cdot z_j^{\mathrm{prot}} / \tau)} + \log \frac{\exp(z_i^{\mathrm{prot}} \cdot z_i^{\mathrm{prom}} / \tau)}{\sum_{j=1}^{N} \exp(z_i^{\mathrm{prot}} \cdot z_j^{\mathrm{prom}} / \tau)} \right).
$$

由于功能相似的蛋白质往往具有协同表达模式，向蛋白质空间对齐会驱动受相同转录因子调控的启动子自发聚簇；下游无需实验数据，仅在启动子嵌入空间检索余弦最近邻即可实现零样本共调控基因识别。

<a id="p-mimic"></a>
### ⚪ **MIMIC**：分轨求和编码器-解码器与多模态中心法则联合生成模型
- **paper**：[**MIMIC: A Generative Multimodal Foundation Model for Biomolecules**](https://arxiv.org/abs/2604.24506)（*arXiv* 2026）

**MIMIC**（$1\text{B}$ 参数）在跨物种多模态对齐数据集 **LORE**（涵盖 $1.3\times 10^7$ 条转录本、$1.55\times 10^7$ 条蛋白质及 $4\text{B}$ 生物医学与实验上下文文本词元）上提出分轨（**Split-Track**）编码器—解码器架构，支持以任意观测模态子集为条件重建或生成基因组、转录组与蛋白质组的缺失分子状态。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-123-mimic.png)

**分轨逐位置求和与非对称编码器—解码器架构：** 为在不引爆序列长度的前提下保持共定位信号对齐，模型将输入组织为三个并行区域并施加组内归零的局部旋转位置编码：
1. **核酸求和轨（Nucleic Acid Track）**：将共享基因组坐标的未剪接序列、剪接位点与外显子掩码、进化保守性（**phyloP**）、染色质可及性（**ATAC-seq**）、启动子活性（**CAGE**）及化学探测反应性（**RASP2**，连续值按百分位离散化为 $100$ 个分箱词元）分别嵌入后逐位置相加；
2. **蛋白质求和轨（Protein Track）**：将氨基酸序列、对齐的编码三联体密码子、**DSSP** 二级结构、**ESM3 VQ-VAE** 离散三级主链词元、溶剂可及表面积（**SASA**）及 $2.8\text{ \AA}$ 邻域聚合的 **MaSIF** 分子表面指纹逐残基相加；
3. **独立语义轨与寄存器词元（Register Tokens）**：细胞系/实验条件自由文本、功能描述及 $5$ 个可学习全局寄存器词元（通过 $0\text{--}10\%$ 随机词元丢弃自监督强制压缩全局状态）直接拼接在生物轨之后。主干采用非对称窗口（$20$ 层编码器窗口 $10{,}000$ 词元，$12$ 层交叉注意力解码器窗口 $1{,}000$ 词元，自注意力掩码按 $50\%$ 双向、$25\%$ 因果与 $25\%$ 反因果混合），训练中结合约 $25$ 条启发式路径采样（$P_k = (I_{\mathrm{req}}, I_{\mathrm{opt}}, T_{\mathrm{req}}, T_{\mathrm{opt}}, w_k)$）、解码端多目标打包（**Target Packing**）与从 $1\text{k}$ 至 $10\text{k}$ 的分阶段分桶课程学习。

**多模态条件推理与受约束逆向分子设计：** 统一生成接口支持三类核心跨模态范式：（1）**生成式演化约束变异效应预测（phyloP VEP）**：分别预测野生型与突变序列的保守性轨迹 $\hat{p}_i^{\mathrm{wt}}, \hat{p}_i^{\mathrm{mut}}$，以变异周围 $W=30\text{ nt}$ 窗口的平均绝对扰动捕获弱保守区的致病调控破坏：

$$
\Delta\hat{p}_i = \hat{p}_i^{\mathrm{mut}} - \hat{p}_i^{\mathrm{wt}}, \quad
s_{\Delta p}(W) = \frac{1}{\lvert W\rvert}\sum_{i\in W} \lvert\Delta\hat{p}_i\rvert;
$$

（2）**转录本级剪接预测与致病剪接修复**：联合输入转录起始/终止位点（**TSS/TES**）实现异构体特异剪接预测；在固定 $\beta$-珠蛋白（**HBB**）内含子致病突变位点的前提下，以野生型剪接模式与 **phyloP** 轨迹为条件重设计侧翼 $30/50\text{ nt}$ 窗口，从而抑制隐蔽外显子掺入；（3）**上下文感知 RNA 折叠与靶标蛋白设计**：以体内/体外及探针类型（**icSHAPE / DMS-seq**）的自然语言描述为条件预测 **RASP2** 反应性，转化为伪能量惩罚 $\Delta G_{\mathrm{SHAPE}}(i) = m\ln(r_i+1)+b$ 引导 **ViennaRNA** 二级结构折叠，或同时以主链几何与 **MaSIF** 表面化学特征为条件迭代生成并自校验高亲和力蛋白结合剂。