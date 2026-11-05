---
layout: post
title: '基因组基础模型(Genome Foundation Model) V：多模态模型'
date: 2025-01-29
author: 郑之杰
cover: 'https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm5-multimodal-000-cover.png'
tags: AI4S
---

> Genome Foundation Model part V: Multimodal Models.

**基因组基础模型（Genome Foundation Model, gFM）**，又称**基因组语言模型（Genomic Language Model, gLM）**，将 **DNA** 序列映射为长度为 $T$ 的离散词元序列 $z_{1:T}$，通过在大规模无标注基因组语料上拟合边缘或条件概率分布 $p_\theta(z)$，学习具备跨任务迁移能力的通用基因组表征。

本文是**基因组基础模型**系列的第五篇。全系列按“**分词 → 预训练 → 编码器 → 解码器 → 多模态**”的脉络组织：
- [I. DNA 分词策略](https://0809zheng.github.io/2025/01/01/gLM-1-token.html)：如何将碱基序列切分为词元。
- [II. 预训练策略](https://0809zheng.github.io/2025/01/08/gLM-2-object.html)：如何构造自监督信号从无标注基因组中学习。
- [III. 编码器框架](https://0809zheng.github.io/2025/01/15/gLM-3-encoder.html)：模型如何在长序列上混合上下文信息。
- [IV. 解码器框架](https://0809zheng.github.io/2025/01/22/gLM-4-decoder.html)：模型如何评估序列似然并生成新序列。
- **V. 多模态模型**：如何联合建模基因组序列、其他生物分子与人类自然语言。

基因组序列在生物体内的功能实现天然处于多模态语境之中：顺式调控元件与编码序列遵循中心法则转录为 **RNA** 并翻译为蛋白质，而实验测定与文献积累又为序列赋予了丰富的连续数值属性与自然语言功能注释。**基因组多模态模型**对这些异构模态数据进行联合建模。

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

**统一词表建模**通过扩充大语言模型（或从头训练解码器）的词表，将离散 **DNA** 词元、连续生物物理数值与自然语言指令置于同一嵌入空间。给定输入序列 $x_{\mathrm{DNA}}$ 与提示 $\mathrm{Prompt}$，模型在单一自回归解码器内优化跨模态条件生成损失：

$$
\mathcal{L}_{\mathrm{Cross\text{-}CLM}} = -\sum_{t=1}^{T} \log p_\theta\big(y_t \mid y_{<t}, x_{\mathrm{DNA}}, \mathrm{Prompt}\big).
$$

该路线以单一主干统一处理序列分类、功能描述生成与互作判别；核心挑战在于缓解新词元并入引发的分布偏移，并防止生物语料持续预训练覆盖原有的自然语言推理能力。

<a id="p-004"></a>
### ⚪ **DNAGPT**：离散序列与连续数值的因果多任务建模
- **paper**：[**DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks**](https://arxiv.org/abs/2307.05628)（*arXiv* 2023）

**DNAGPT** 在自回归 **GPT** 解码器中引入双模态输入输出接口与结构化指令词表，将离散 **DNA** 序列与连续数值属性统一至同一因果生成框架。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-069-dnagpt.png)

**非重叠分词与结构化控制词表：** 序列采用非重叠 **$k$-mer** 分词，词表涵盖六类符号：表示 **DNA** 片段的**序列词元**、承载连续标量的**数值词元**、标识二元真假的**分类词元**、指定物种来源与输出模态的**指令词元**、用于特征聚合（`+`）与界定输入输出映射（`=`）的**连接词元**，以及特殊**预留词元**。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-027-680a1e1b.png)

**双流嵌入与联合预训练目标：** 离散词元经查表映射，连续数值经卷积与残差全连接层投影至同一隐空间；两路嵌入拼接后输入解码器，输出端按模态分流至线性分类头或标量回归头。预训练在同一上下文窗口内拼接三项自监督任务：（1）**下一词元预测（NTP）**优化离散序列似然；（2）**序列顺序预测**判别随机片段是否发生反向翻转；（3）**GC 含量回归**最小化动态长度子片段 **GC** 比例的均方误差。

<a id="p-llamagene"></a>
### ⚪ **LLaMA-Gene**：扩展多模态 BPE 词表与 Alpaca 指令微调的基因对话模型
- **paper**：[**LLaMA-Gene: A General-purpose Gene Task Large Language Model Based on Instruction Fine-tuning**](https://github.com/maris205/llama-gene)（*arXiv* 2024）

**LLaMA-Gene** 将 **DNA** 与蛋白质的 **BPE** 子词词表直接并入预训练 **LLaMA**，通过混合生物学文献的参数高效持续预训练与 **Alpaca** 指令微调，将多种生物序列分析任务统一为生成式对话问答。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-113-llamagen.png)

**多模态词表扩展与防遗忘持续预训练：** 模型在原 **LLaMA** 词表（$32{,}000$ 词元）基础上，合并不重叠训练的多物种 **DNA** 与 **UniProt** 蛋白质 **BPE** 词表（各 $30{,}000$ 词元），去重构建约 $91{,}000$ 词元的统一词表。预训练按 $1:1:1$ 混合 **DNA**、蛋白质与生物医学英文论文（**S2ORC**）以防止灾难性遗忘，采用 **LoRA** 更新注意力与前馈层，并全量放开词嵌入层与语言模型头。

**多任务 Alpaca 指令模板重构：** 指令微调阶段复用单一因果语言建模头，将五类任务统一映射为 $(\mathrm{Instruction}, \mathrm{Input}, \mathrm{Output})$ 模板并优化条件自回归损失 $\mathcal{L}_{\mathrm{IFT}} = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, \mathrm{Instruction}, \mathrm{Input})$：（1）**序列分类**要求直接生成指令中给定的候选类别名；（2）**词元级结构预测**截取目标词元及前序 $30\text{ bp}$ 滑窗，按多数投票将单残基标签对齐至末端 **BPE** 词元；（3）**多序列关联判别**以 `\n###\n` 拼接双序列并输出二元判别词；（4）**功能预测**直接生成 **GO** 描述文本；（5）**连续回归**按中位数离散化为 **high / low** 二元文本分类。

<a id="p-omnidna"></a>
### ⚪ **Omni-DNA**：解码器的跨模态词表扩展与多任务指令统一
- **paper**：[**Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning**](https://arxiv.org/abs/2502.03499)（*arXiv* 2025）

**Omni-DNA** 基于 **OLMo** 自回归解码器，通过多物种 **BPE** 预训练与跨模态词表扩展微调，将表观遗传分类、**DNA** 功能文本生成（**DNA2Func**）及连续变量离散化映射（**DNA2Image**）统一于单一生成模型。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-093-omnidna.png)

**跨模态词表扩展与穩定微调机制：** 微调阶段将不同任务重构为 $\tilde{x} = (x, \mathrm{Prompt}_k)$，并将原基因组词表 $V_o$ 扩展为并入任务新词元 $V_z$（自然语言词汇、类别标签或量化视觉词元）的联合词表：

$$
p_\theta(y_t \mid y_{<t}, x) = \frac{\exp(h_{t-1}^\top e_{y_t})}{\sum_{m \in V_o} \exp(h_{t-1}^\top e_m) + \sum_{n \in V_z} \exp(h_{t-1}^\top e_n)}.
$$

为缓解新词元引入导致的分布偏移与过拟合，微调结合三项机制：（1）**关键标签词元重复**：将分类响应中的标签词元重复 $\alpha$ 次以强化监督信号；（2）**NEFTune**：向词元嵌入注入强度为 $\beta$ 的均匀随机噪声；（3）**连续响应离散化**：利用预训练 **VQ-VAE** 将高维连续信号或图像压缩为离散词元并入 $V_z$ 进行自回归生成。

<a id="p-omnina"></a>
### ⚪ **OmniNA**：融合核酸序列与文本注释的自回归问答
- **paper**：[**A foundation model for nucleotide sequences**](https://doi.org/10.1093/nar/gkag083)（*Nucleic Acids Research* 2026）

**OmniNA** 基于 **LLaMA** 解码器，在预训练阶段联合建模跨物种核酸序列（**DNA/RNA**）与自然语言注释，并通过问答（**QA**）指令微调支持纯序列输入的基因组判别与突变效应评估。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-095-omnina.png)

**联合 BPE 预训练与无注释 QA 微调：** 模型利用 **SentencePiece BPE** 在跨物种核酸序列与标准化注释（分类学层级、基因符号与功能描述）上联合训练词表，按“指令 + 核酸序列 + 文本注释”模板执行自回归预训练。下游微调重构为“固定指令 + 核酸序列 + 任务问题 + 期望回答”四元组（仅对回答词元计损），推理时无需任何先验文本注释即可直接从纯序列生成预测。在饱和突变分析中，模型通过野生型 $s_{\mathrm{WT}}$ 与突变型 $s_{\mathrm{mut}}$ 在目标正类上的对数似然相对变化量化突变效应：

$$
\mathrm{Effect}(s_{\mathrm{WT}}, s_{\mathrm{mut}}) = \left\lvert \frac{\mathrm{LL}(s_{\mathrm{WT}}) - \mathrm{LL}(s_{\mathrm{mut}})}{\mathrm{LL}(s_{\mathrm{WT}})} \right\rvert, \qquad
\mathrm{LL}(s) = \sum_{j} \log \Pr(x_j = s_j \mid s).
$$

<a id="p-omnigene4"></a>
### ⚪ **OmniGene-4**：生物-语言-视觉混合专家模型与路由器级可解释性分解
- **paper**：[**OmniGene-4: A Unified Bio-Language MoE Model with Router-Level Interpretability and Modality-Invariant Transfer**](https://doi.org/10.64898/2026.05.12.724542)（*bioRxiv* 2026）

**OmniGene-4** 基于稀疏混合专家（**MoE**）架构（**Gemma-4-26B-A4B**，每层 $128$ 专家、**Top**-$8$ 路由）扩展生物序列与结构词表，通过包含路由器投影层的 **QLoRA** 持续预训练（**CPT**）与双头指令微调（**SFT**），首次在离散路由器层面量化了跨模态表征分化与输出对齐的阶段贡献。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-114-omnigene4.png)

**带前缀多模态词表与双头训练：** 模型注入 $28{,}028$ 个带模态前缀的生物词元（涵盖 **DNA/Protein BPE**、$20$ 类 **Foldseek 3Di** 三维结构字母与 $8$ 类 **DSSP** 二级结构标签）。**CPT 阶段**按 $3:1$ 混合生物序列与通用文本，对注意力、前馈层及路由器投影层施加 **4-bit QLoRA** 更新；**SFT 阶段**在末层隐状态上并行挂载 **3Di/DSSP** 逐残基分类头，按 $0.5\mathcal{L}_{\mathrm{gen}} + 0.5\mathcal{L}_{\mathrm{cls}}$ 联合优化对话生成与结构预测。

**路由器级阶段分解：** 统计任务 $t$ 在第 $\ell$ 层激活专家 $e$ 的路由分布 $p_{t,\ell} \in \Delta^{128}$，定义专家特化度 $s_{t,e}^{(\ell)}$ 与跨任务平均 **Jensen–Shannon** 散度 $J_\ell$：

$$
s_{t,e}^{(\ell)} = \log\!\left(\frac{p_{t,e}^{(\ell)}}{\bar{p}_e^{(\ell)}}\right), \qquad
J_\ell = \binom{8}{2}^{-1} \sum_{t \neq t'} \mathrm{JS}\!\left(p_{t,\ell} \,\|\, p_{t',\ell}\right).
$$

跨检查点分解表明：**CPT** 贡献了 $96\%$ 的跨任务路由分化（集中在中层 $\mathrm{L11}\text{--}\mathrm{L22}$ 将生物与语言词元分流至正交专家集以防遗忘），**SFT** 仅在末两层贡献 $4\%$ 以对齐输出格式；而同任务正负样本对间的路由散度极低（$<0.04$），证实细粒度判别发生在被激活专家内部而非路由门控处。

<a id="p-genna"></a>
### ⚪ **GenNA**：跨模态 BPE 词表与自然语言注释引导的条件核苷酸生成
- **paper**：[**GenNA: Conditional generation of nucleotide sequences guided by natural-language annotations**](https://doi.org/10.64898/2026.04.22.720063)（*bioRxiv* 2026）

**GenNA**（$3.6\text{B}$ 参数，基于配备 **GQA**、**RoPE** 与 **SwiGLU** 的 **Qwen3** 自回归解码器）在跨越 $2{,}221$ 个真核物种、约 $4{,}160$ 亿字符的多模态基因中心语料上从头预训练，通过统一的跨模态 **BPE** 词表将自然语言功能注释、物种上下文及 **XML** 结构标签与核苷酸序列联合建模，统一实现零样本变异与系统发育感知以及自然语言引导的从头序列设计。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-129-genna.png)

**跨模态 BPE 分词、动态提示掩码与困惑度扰动评估：** 预训练语料聚焦 **NCBI RefSeq** 注释基因区（随机向两侧延伸 $0\text{--}100\text{ bp}$），将每条样本按统一前缀模板 `[Molecule Type], [Species Name], [Gene Symbol], [Functional Annotation]<seq>` 衔接含 `<gene>`、`<cds>`、`<tRNA>` 等 **XML** 边界标签的核苷酸序列，并打包至 $4{,}096$ 词元（约 $20\text{ kb}$）；专用跨模态 **BPE** 分词器（容量 $6{,}000$）将高频英文功能短语、结构标签与局部序列基序映射至共享离散词元空间，并在训练后期以 $0.5$ 概率随机掩码物种或基因标识符，迫使模型依赖功能语义与统计上下文而非机械记忆标签。在零样本评估中，模型直接利用全序列条件困惑度差值 $\Delta\mathrm{PPL}$ 量化突变破坏性（同义 $<$ 错义 $<$ 无义 $<$ 移码）、序列—功能语义失配及跨物种标签替换矩阵 $S_i \mid C_j$ 的系统发育距离；在条件生成阶段，则结合温度缩放 $\tau$、核采样（**Top**-$p$）与 $n$-gram 重复惩罚，从物种特异背景、结构化非编码 **RNA**（如自发符合摆动配对规则的 **tRNA** 与 **rRNA**）或组蛋白家族提示词出发可控生成目标序列：

$$
\mathrm{PPL}(S \mid C) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(y_t \mid C, y_{<t})\right), \qquad
\Delta\mathrm{PPL}_{\mathrm{mut}} = \mathrm{PPL}(S_{\mathrm{mut}} \mid C) - \mathrm{PPL}(S_{\mathrm{wt}} \mid C), \qquad
\Delta\mathrm{PPL}_{i,j} = \mathrm{PPL}(S_i \mid C_j) - \mathrm{PPL}(S_i \mid C_i).
$$


<a id="p-carbonx"></a>
### ⚪ **Carbon-X**：混合 $6$-mer 词表中期训练与可验证奖励强化学习的 DNA-语言模型
- **paper**：[**Carbon-X: When Natural Language Meets DNA**](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release)（*preprint* 2026）

**Carbon-X** 在 **Qwen3-14B** 解码器中扩展定长 $6$-**mer** 基因组词表，通过 $300\text{B}$ 词元的跨模态中期训练、确定性验证器校验的有监督微调以及可验证奖励强化学习（**RLVR**），在单一词元流内统一了长窗口变异致病性解释、序列问答与 **DNA** 生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-118-carbonx.png)

**混合 $6$-mer 分词与确定性验证后训练：** 模型对自然语言沿用 **Qwen3 BPE**，对 `<dna>...</dna>` 内的序列采用步长为 $6$ 的非重叠 $6$-**mer** 分词（扩充 $4^6=4{,}096$ 个词元，将序列长度较文本分词器压缩约 $3$ 倍）。中期训练混合原始基因组、文献挖掘配对数据、**RefSeq** 合成样本及英文指令回放，配合低峰值余弦衰减学习率防遗忘。为消除基因坐标与突变位点幻觉，后训练引入基于专家规则直接核查输入 **DNA** 序列的**确定性验证器（Deterministic Verifiers）**：先过滤无依据断言以构建高保真 **SFT** 训练集，再在长窗口（$8{,}192\text{ bp}$）变异评估（**FrontierVEP**）、序列恢复与定位任务上，以验证器提供的二元/标量奖励执行 **RLVR** 策略优化，使模型能够输出可验证的分子致病机制（如识别破坏剪接受体 **AG** 的突变）。


<a id="mm-bridge"></a>
# 2. 编码器-LLM 桥接（DNA Encoder–LLM Bridge）

受视觉语言模型（**VLM**）启发，**编码器—LLM 桥接**架构复用预训练 **DNA** 基础模型作为序列感知前端，通过轻量投影层或交叉注意力重采样器，将连续基因组表征注入自回归大语言模型：

$$
X_{\mathrm{LLM}} = \mathrm{Concat}\!\Big(\mathrm{Adapter}\big(f_{\mathrm{DNA}}(S_{\mathrm{DNA}})\big),\; E_{\mathrm{LLM}}(Q_{\mathrm{Text}})\Big).
$$

该范式完整保留了专用基因组编码器的局部单碱基敏感性，同时直接调用 **LLM** 的指令跟随与思维链（**CoT**）多步逻辑推理能力。

<a id="p-genechat"></a>
### ⚪ **GeneChat**：基于线性适配器桥接基因组编码器与大语言模型的问答框架
- **paper**：[**GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction**](https://github.com/Shashi-Sekar/GeneChat)（*ICML 2025 Workshop FM4LS*）

**GeneChat** 通过非重叠窗口池化与线性适配器将 **DNABERT-2** 编码器与 **Vicuna-13B** 对接，直接从最长 $160\text{ kb}$ 的原始基因核苷酸序列生成自由文本功能描述。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-117-genechat.png)

**分块池化投影与占位符微调：** 超长基因序列 $x_g$ 按 $512\text{ nt}$ 窗口（重叠 $10\text{ nt}$）切分并经 **DNABERT-2** 提取特征，沿序列维经窗口大小为 $k$ 的平均池化压缩为 $h(x_g) \in \mathbb{R}^{(l/k) \times 256}$，再通过线性适配器 $W \in \mathbb{R}^{256 \times 5120}$ 对齐至 **Vicuna-13B** 词嵌入空间并替换对话模板中的 `GeneHere` 占位符：

$$
h_g = h(x_g) \cdot W \in \mathbb{R}^{(l/k) \times 5120}, \qquad
p(x_a \mid x_g, x_{\mathrm{aux}}) = \prod_{i=1}^{l} p_\theta\big(x_a^{(i)} \mid x_g, x_{\mathrm{aux}}, x_a^{<i}\big).
$$

训练仅对回答序列 $x_a$ 计算自回归负对数似然，同时全参微调 **DNABERT-2**、训练线性适配器，并对 **Vicuna-13B** 施加 **LoRA**（$\mathrm{rank}=8$）微调。

<a id="p-chatnt"></a>
### ⚪ **ChatNT**：英语问题感知感知器重采样与多任务基因组对话代理
- **paper**：[**A multimodal conversational agent for DNA, RNA and protein tasks**](https://doi.org/10.1038/s42256-025-01047-1)（*Nature Machine Intelligence* 2025）

**ChatNT** 将 **Nucleotide Transformer（NT-v2, 500M）** 编码器与冻结的 **Vicuna-7B** 解码器桥接，通过问题条件化的感知器重采样器，将横跨 **DNA**、**RNA** 与蛋白质（输入编码序列 **CDS**）的分类与回归任务统一为自然语言生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-116-chatnt.png)

**英语问题感知感知器重采样与困惑度校准：** 给定 $6$-**mer** 分词序列 $s$（最长约 $12\text{ kb}$），为避免固定查询在多任务下产生信息瓶颈，包含 $K=64$ 个可学习查询 $Q_{\mathrm{lat}}$ 的重采样器先与英文问题嵌入 $E_{\mathrm{question}}$ 交叉注意力交互，再从 **DNA** 隐状态 $H_{\mathrm{DNA}}$ 中提取 $K$ 个任务特异的软提示向量（训练时冻结 **Vicuna-7B**，联合更新重采样器与 **NT-v2**）：

$$
\tilde{Q} = \mathrm{CrossAttn}(Q_{\mathrm{lat}}, E_{\mathrm{question}}), \qquad
Z_{\mathrm{DNA}} = \mathrm{CrossAttn}(\tilde{Q}, H_{\mathrm{DNA}}) \in \mathbb{R}^{K \times d_{\mathrm{llm}}}.
$$

为在二分类任务中导出校准概率，模型引入**事后困惑度分类器**：分别计算输入在 $N$ 个正/负回答模板 $y_i^{\mathrm{pos/neg}}$ 上的平均困惑度并拟合 **Platt** 逻辑回归：

$$
\mathrm{pp}_\theta^{\mathrm{pos}}(x, s) = \frac{1}{N}\sum_{i=0}^{N-1} \exp\!\left(\sum_j p_\theta\big((y_i^{\mathrm{pos}})_j \mid x, s, y_i^{\mathrm{pos}}\big) \log p_\theta\big((y_i^{\mathrm{pos}})_j \mid x, s, y_i^{\mathrm{pos}}\big)\right).
$$

此外，将英文输出词元的梯度经重采样器反向传播至 **DNA** 输入端计算 $\vert\nabla_{\mathrm{emb}} \odot \mathrm{Input}\vert$，可直接从自然语言回答定位剪接位点与 **TATA-box** 等保守基序。

<a id="p-bioreason"></a>
### ⚪ **BioReason**：融合基因组编码器与大语言模型的强化学习多步生物推理
- **paper**：[**BIOREASON: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model**](https://arxiv.org/abs/2505.23579)（*NeurIPS* 2025）

**BioReason** 在嵌入层直接桥接冻结的 **DNA** 基础模型与自回归 **LLM**，通过有监督微调与群组相对策略优化（**GRPO**），激励模型从成对基因组变异序列出发生成分步分子机制推理链并预测下游表型。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-115-bioreason.png)

**跨模态特征拼接与 Dr. GRPO 强化学习对齐：** 冻结的 **DNA** 编码器（**Evo 2** / **NT** / **Enformer**）提取参考与突变序列的逐词元表征，经线性投影堆叠为 $E'_{\mathrm{DNA}}$ 后与边界控制符及文本查询嵌入 $E_{Q_{\mathrm{text}}}$ 拼接，送入 **LoRA** 微调的 **Qwen3** 主干：

$$
X_{\mathrm{LLM}} = \big(e_{\text{<dna\_start>}},\, E'_{\mathrm{DNA}},\, e_{\text{<dna\_end>}},\, E_{Q_{\mathrm{text}}}\big), \qquad
\mathcal{L}_{\mathrm{SFT}} = -\sum_{i} \log P(y_i \mid y_{<i}, X_{\mathrm{LLM}}; \theta_{\mathrm{LLM}}).
$$

基于整合 **KEGG** 符号化互作网络与 **ClinVar/OMIM** 临床变异构建的多步推理数据集，模型进一步采用无偏变体 **Dr. GRPO** 优化策略：对每个输入采样 $G$ 条候选输出 $\{o_1, \dots, o_G\}$，结合答案正确性（$+2.0$）、精炼度（$+0.5$）与 `<think>` 标签闭合性（$+0.5$）计算无标准差缩放的组内相对优势 $A_i = r_i - \mathrm{mean}(\{r_1, \dots, r_G\})$，最大化截断代理目标：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\!\left(\frac{\pi_\theta(o_i \mid X_{\mathrm{LLM}})}{\pi_{\theta_{\mathrm{old}}}(o_i \mid X_{\mathrm{LLM}})}A_i,\, \mathrm{clip}\!\left(\frac{\pi_\theta(o_i \mid X_{\mathrm{LLM}})}{\pi_{\theta_{\mathrm{old}}}(o_i \mid X_{\mathrm{LLM}})}, 1-\epsilon, 1+\epsilon\right)A_i\right) - \beta D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})\right)\right].
$$


<a id="mm-centraldogma"></a>
# 3. 中心法则对齐（Central Dogma Alignment）

除自然语言外，分子生物学**中心法则（DNA $\to$ RNA $\to$ Protein）**本身构成了另一大天然跨模态监督源。由于蛋白质序列受更强的三维结构与功能纯化选择约束，通过跨分子配对生成、多层级注释联合监督或双塔 **InfoNCE** 对比对齐，可将蛋白质与转录组层的先验知识反向注入基因组表征中。

<a id="p-cdgpt"></a>
### ⚪ **CD-GPT**：共享多分子 BPE 词表与中心法则配对的跨模态自回归预训练
- **paper**：[**CD-GPT As a Biological Foundation Model Bridging the Gap between Molecular Sequences Through Central Dogma**](https://doi.org/10.1101/2024.06.24.600337)（*bioRxiv* 2025）

**CD-GPT** 构建统一的跨分子子词词表，在 **RoPE** 自回归解码器上通过三阶段课程预训练显式对齐中心法则的转录与翻译映射，统一支持基因组元件判别、跨分子互作预测与逆向翻译生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-112-cdgpt.png)

**混合 BPE 词表与三阶段课程预训练：** 模型按 $5:3:1$ 混合 **DNA**、**RNA** 与蛋白质序列，通过 **SentencePiece BPE** 构建 $64{,}000$ 词元的共享词表。预训练分三阶段递进：（1）**单分子预训练**在独立序列上优化下一词元预测损失 $\mathcal{L}_{\mathrm{ntp}}$，并以 $5\%$ 概率随机回退至单字符分词以保留残基级特征；（2）**中心法则配对预训练**将同源转录翻译序列通过控制符拼接为单条因果流（`<DNA>...<splicing><mRNA><translate><:><mRNA>...<protein>...`）进行联合自回归建模；（3）**结构多任务预训练**在 **PDB** 数据上联合优化 $\mathcal{L}_{\mathrm{total}} = \alpha \mathcal{L}_{\mathrm{ntp}} + \beta \mathcal{L}_{\mathrm{ss}} + \gamma \mathcal{L}_{\mathrm{contact}}$。下游通过挂载线性层、单层双向自注意力或对称化注意力 **MLP** 头适配序列/词元/残基对预测，并支持通过 `<reverse translate>` 提示从蛋白质逆向生成高表达 **mRNA**。

<a id="p-prodmm"></a>
### ⚪ **ProDMM**：非编码调控区-编码序列-蛋白质协同演化的统一编码-解码模型
- **paper**：[**Harnessing A Unified Multi-modal Sequence Modeling to unveil Protein-DNA Interdependency**](https://doi.org/10.1101/2025.02.26.640480)（*bioRxiv* 2025）

**ProDMM** 在 $1.29\times 10^8$ 组原核四元组（`<pre-NCDS, CDS, Protein, post-NCDS>`）上构建统一词表，结合双向掩码编码器（**ProDMM-Encoder**）与条件自回归解码器（**ProDMM-Seq2Seq**），联合捕获密码子偏好、顺式调控语法与蛋白质稳定性之间的协同演化依赖。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-120-prodmm.png)

**三模态粒度感知词表与伪困惑度/条件生成：** 分词器构建含 $97$ 个词元的共享词表（对蛋白质与非编码区 **NCDS** 按单字符分词，对编码区 **CDS** 按 $64$ 种三联体密码子分词）。$48$ 层双向编码器将每条四元组等概率随机转化为纯 **Protein**、纯 **CDS** 或拼接上下游调控区的复合格式（`<cls>, pre-NCDS, <sep>, CDS/Protein, <sep>, post-NCDS, <sep>`）优化 **MLM** 损失。下游分两路应用：（1）**零样本表型与代谢通路产量排序**利用冻结编码器计算单序列或复合序列（如 `<NCDS-Protein>×4`）的伪困惑度与突变对数几率比：

$$
\mathrm{PP}_{\mathrm{cond}}(S) = \exp\!\left(-\frac{1}{L_1}\sum_{i=1}^{L_1} \log P(w_i \mid S_{\setminus i})\right), \qquad
\mathrm{ME}(\mathcal{M}) = \sum_{i=1}^{k} \log \frac{P(S[p_i] = mt_i \mid S)}{P(S[p_i] = wt_i \mid S)};
$$

（2）**ProDMM-Seq2Seq 条件生成**在冻结编码器顶部通过交叉注意力挂载 $12$ 层因果解码器，利用 `<aas2cds>`、`<cds2pre-ncds>` 与 `<cds2post-ncds>` 提示符联合支持带密码子表掩码约束的逆向翻译及 **CDS**—启动子双向定制生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-121-prodmm-app.png)

<a id="p-naturelm"></a>
### ⚪ **NatureLM**：多科学域实体词表扩展与中心法则交错序列联合预训练
- **paper**：[**Nature Language Model: Deciphering the Language of Nature for Scientific Discovery**](https://arxiv.org/abs/2502.07527)（*arXiv* 2025）

**NatureLM**（$1\text{B}$、**Llama-3-8B** 与 **Mixtral-8x7B**）将 **DNA**、**RNA**、蛋白质、小分子与晶体材料统一序列化，在 $143.8\text{B}$ 词元上通过两阶段持续预训练与强化后训练，支持基因组元件判别、向导 **RNA（gRNA）** 设计及跨分子生成。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-122-naturelm.png)

**多域边界标签词表与两阶段持续预训练：** 模型扩充单碱基 **DNA/RNA**、单氨基酸蛋白质、正则化 **SMILES** 及晶体空间群词元，以 `<dna>`、`<rna>`、`<protein>`、`<mol>` 界定模态边界，并构建三类跨域语料：（1）利用 **BERN2** 实体识别在文献正文中直接内嵌分子 **SMILES** 与蛋白质 **FASTA** 的交错序列；（2）数据库序列—文本平行对；（3）将 **RefSeq** 基因组中的编码外显子直接替换为对应翻译蛋白质序列的中心法则共现序列。持续预训练先冻结主干仅更新新增科学词元嵌入，再全参解冻联合训练；后训练在 $5.1\times 10^6$ 条指令对 **SFT** 的基础上引入直接偏好优化（**DPO**）：

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right],
$$

在核酸任务上支持根据靶标 `<dna>` 自动定位 **NGG PAM** 并反向互补生成 **CRISPR crRNA**（$\text{DNA}\to\text{RNA}$），以及给定靶标 `<protein>` 从头设计高亲和力结合 **RNA**（$\text{Protein}\to\text{RNA}$）。

<a id="p-lifecode"></a>
### ⚪ **Life-Code**：逆转录/逆翻译核苷酸统一的中心法则模型
- **paper**：[**Life-Code: Central Dogma Modeling with Multi-Omics Sequence Unification**](https://arxiv.org/abs/2502.07299)（*arXiv* 2025）

**Life-Code**（$350\text{M}/500\text{M}$ 参数）通过逆转录与逆翻译将 **DNA**、**RNA** 与蛋白质全部重映射至统一的核苷酸序列空间，结合双任务预训练的密码子级分词器、双链等变的 **Gated DeltaNet**—自注意力混合编码器以及来自蛋白质语言模型的知识蒸馏，在单一模型内实现了跨三组学长序列的高效表征、零样本蛋白质适应度预测与中心法则关联判别。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-126-lifecode.png)

**三组学核苷酸统一分词与 $11:1$ 双链混合编码器：** 数据管线将 **RNA** 逆转录回基因组 **DNA**，并按高频标准密码子表 $g:\Sigma\to\{\mathrm{A,T,C,G}\}^3$ 将长度为 $L$ 的氨基酸序列逆翻译为长度为 $3L$ 的 **cDNA** 序列（短 **CDS**—氨基酸对通过分隔符 `<S>` 打包拼接）。**Life-Code 分词器**先经单层 **Gated DeltaNet（GDN）**、核宽为 $3$ 的一维卷积与步长为 $3$ 的展开操作将连续三联体 $x_{3j+1:3j+3}$ 压缩为密码子级嵌入 $e_j$，并通过对称反卷积解分词器重建掩码碱基（$\mathcal{L}_{\mathrm{MLM}}$）与双层 **MLP** 翻译器预测对应氨基酸（$\mathcal{L}_{\mathrm{Trans}}$）完成预训练。随后的 $24$ 层**双向混合编码器**将特征沿通道维对半拆分为正反双链 $(E^{(+)}, E^{(-)})=\mathrm{Split}(E)$，共享参数并行处理正向与反转序列后拼接输出 $H=[f_\theta(E^{(+)}), f_\theta(\mathrm{Reverse}(E^{(-)}))]$；主干按 $11:1$ 的比例交错堆叠线性复杂度的 **GDN** 块与标准多头自注意力块，在保持高吞吐量的同时兼顾长程上下文建模。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-127-lifecode-preprocess.png)

**三任务对数和联合预训练与蛋白质知识蒸馏：** 固定分词器的解码与翻译头后，编码器在掩码 **DNA** 重建（$\mathcal{L}_{\mathrm{MLM}}$）与 **CDS**—氨基酸翻译（$\mathcal{L}_{\mathrm{Trans}}$）基础上，引入以冻结 **ESM2-650M** 为教师的词元级知识蒸馏（$\mathcal{L}_{\mathrm{KD}}$）：利用双层 **MLP** 蛋白质解码器 $D(\cdot)$ 将编码器输出映射至教师特征维度，对齐教师分布 $P_{\theta'}(h_j^{\mathrm{PLM}})$，并采用对数和（Log-Sum）自适应多任务加权策略联合优化三项损失（防止单一任务主导梯度）：

$$
\mathcal{L}_{\mathrm{KD}} = -\sum_{j=1}^{L} P_{\theta'}\!\left(h_j^{\mathrm{PLM}}\right)^\top \log P_\theta\!\left(D\big(h_j^{\mathrm{LifeCode}}\big)\right), \qquad
\mathcal{L}_{\mathrm{total}} = \log(\mathcal{L}_{\mathrm{MLM}}+\epsilon) + \log(\mathcal{L}_{\mathrm{Trans}}+\epsilon) + \log(\mathcal{L}_{\mathrm{KD}}+\epsilon).
$$

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-128-lifecode-pretrain.png)

<a id="p-lucaone"></a>
### ⚪ **LucaOne**：统一字符词表与四层级半监督注释的核酸-蛋白质编码器
- **paper**：[**Generalized biological foundation model with unified nucleic acid and protein language**](https://doi.org/10.1038/s42256-025-01044-4)（*Nature Machine Intelligence* 2025）

**LucaOne**（$1.8\text{B}$ 参数，$20$ 层 **Pre-LN** 与 **RoPE** 双向 **Transformer** 编码器）在跨越 $169{,}861$ 个物种的核酸（**DNA/RNA**）与蛋白质序列上构建统一字符级词表，通过自监督掩码重建与八项跨四个尺度的半监督注释任务联合预训练，在无需显式配对语料的条件下涌现出中心法则转录翻译对齐与跨分子互作表征能力。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-125-lucaone.png)

**去歧义统一词表与四层级十任务联合预训练：** 为避免核苷酸与氨基酸的单字母符号重叠，分词器构建容量为 $39$ 的统一字符词表（将核酸 `A, T/U, C, G, N` 映射为 `'1'`–`'5'`，保留大写字母供标准与稀有氨基酸使用），并叠加二元分子类型嵌入（核酸置 $0$、蛋白质置 $1$，最大窗口 $1{,}280$）。预训练在未配对的 **RefSeq** 核酸（含正反双链）与 **UniProt / ColabFoldDB / PDB** 蛋白质数据上联合优化四个层级的十项任务（未知注释标签统一置 $-100$ 屏蔽损失）：（1）**词元级**随机遮蔽 $15\%$ 位点执行核酸与蛋白质掩码恢复（$\mathcal{L}_{\mathrm{Gene\text{-}Mask}}, \mathcal{L}_{\mathrm{Prot\text{-}Mask}}$）；（2）**跨度级**逐位点分类核酸 $8$ 类基因组功能区（`CDS`、内含子、调控区及各类非编码 **RNA**）以及蛋白质的活性/结合位点（$946$ 类）、同源超家族（$3{,}442$ 类）与结构域（$13{,}717$ 类）；（3）**序列级**预测核酸与蛋白质的分类学目级归属（Order-level Taxonomy）及 **UniProt** 功能关键词多标签分类（$\mathcal{L}_{\mathrm{Prot\text{-}Keyword}}$）；（4）**结构级**回归蛋白质残基链内归一化的 $C_\alpha$ 三维空间坐标（$\mathcal{L}_{\mathrm{Prot\text{-}Structure}}$）。为平衡任务复杂度，模型对核心与辅助任务分配两级损失权重（$w_{\mathrm{main}}=1.0,\ w_{\mathrm{aux}}=0.2$）：

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{Gene\text{-}Mask}} + \mathcal{L}_{\mathrm{Prot\text{-}Mask}} + \mathcal{L}_{\mathrm{Prot\text{-}Keyword}} + \mathcal{L}_{\mathrm{Prot\text{-}Structure}} + 0.2 \sum_{k=1}^{6} \mathcal{L}_{\mathrm{Span/Tax}}^{(k)}, \qquad
\mathbf{h}_{\mathrm{pair}} = \mathrm{Concat}\!\big(\mathrm{Pool}(f_\theta(x_{\mathrm{NA}})),\; \mathrm{Pool}(f_\theta(x_{\mathrm{Prot}}))\big).
$$

在下游中心法则 **DNA**—蛋白质配对判别、非编码 **RNA**—蛋白质互作（**ncRPI**）等异构双序列任务中，直接冻结主干提取两路表征，经最大值池化或值级注意力池化（$\mathrm{Pool}$）拼接后输入单层全连接网络，即可在少样本设定下显著超越独立预训练的 **DNABERT-2 + ESM2-3B** 组合。


<a id="p-c3p"></a>
### ⚪ **C3P**：启动子与冻结蛋白质表征的跨模态对比对齐
- **paper**：[**C3P: Contrastive promoter-protein pretraining yields representations capturing bacterial gene regulation**](https://arxiv.org/abs/2605.25242)（*arXiv* 2026）

**C3P** 借鉴 **CLIP/LiT** 跨模态对比范式，以冻结的预训练蛋白质语言模型（**ESM-2**）为语义锚点，在共享隐空间对齐细菌启动子与其下游编码蛋白，克服非编码序列演化噪声并学习捕获共调控规律的启动子表征。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-110-c3p.png)

**操纵子感知双塔与对称 InfoNCE 对齐：** 模型截取细菌起始密码子前至少 $100\text{ bp}$ 的非编码区作为启动子，并将同链间距 $\le 20\text{ bp}$ 的相邻基因归为同一操纵子（**Operon**）共享首基因启动子以构建正样本对。**启动子编码器**对重叠 $3$-**mer** 词元应用带 **RoPE** 的双向 **Transformer** 提取 `<CLS>` 表征，**蛋白质编码器**直接复用冻结的 **ESM-2** 均值池化表征；两路输出经线性投影与 $L_2$ 归一化后优化对称 **InfoNCE** 损失：

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left( \log \frac{\exp(z_i^{\mathrm{prom}} \cdot z_i^{\mathrm{prot}} / \tau)}{\sum_{j=1}^{N} \exp(z_i^{\mathrm{prom}} \cdot z_j^{\mathrm{prot}} / \tau)} + \log \frac{\exp(z_i^{\mathrm{prot}} \cdot z_i^{\mathrm{prom}} / \tau)}{\sum_{j=1}^{N} \exp(z_i^{\mathrm{prot}} \cdot z_j^{\mathrm{prom}} / \tau)} \right).
$$

向蛋白质功能流形对齐驱动受相同转录因子调控的启动子自发聚簇，下游仅靠启动子嵌入的余弦最近邻检索即可实现零样本共调控基因识别。

<a id="p-mimic"></a>
### ⚪ **MIMIC**：分轨求和编码器-解码器与多模态中心法则联合生成模型
- **paper**：[**MIMIC: A Generative Multimodal Foundation Model for Biomolecules**](https://arxiv.org/abs/2604.24506)（*arXiv* 2026）

**MIMIC**（$1\text{B}$ 参数）在跨物种多模态对齐数据集 **LORE** 上提出分轨（**Split-Track**）编码器—解码器架构，支持以任意观测模态子集为条件重建或逆向设计基因组、转录组与蛋白质组的缺失分子状态。

![](https://pub-c304ca0128b34bff97119b39961bc4f0.r2.dev/ai4s-glm-123-mimic.png)

**分轨逐位置求和与非对称编码器—解码器：** 为在不引爆序列长度的前提下保持共定位信号对齐，模型将输入组织为三个并行区域并施加组内归零的局部 **RoPE**：（1）**核酸求和轨**将共享基因组坐标的未剪接序列、剪接掩码、进化保守性（**phyloP**）、染色质可及性（**ATAC-seq/CAGE**）及化学探测反应性（**RASP2**，连续值按百分位离散化为 $100$ 个分箱词元）逐位置嵌入相加；（2）**蛋白质求和轨**将氨基酸、对齐密码子、**DSSP** 二级结构、**ESM3 VQ-VAE** 三级结构词元、溶剂可及表面积及 **MaSIF** 分子表面指纹逐残基相加；（3）**独立语义轨与寄存器词元**将实验条件文本及 $5$ 个可学习全局寄存器词元拼接在生物轨之后。主干采用非对称窗口（$20$ 层编码器 $10{,}000$ 词元，$12$ 层交叉注意力解码器 $1{,}000$ 词元），结合约 $25$ 条启发式路径采样与解码端多目标打包完成分阶段课程预训练。

**多模态条件推理与受约束逆向设计：** 统一生成接口支持三类跨模态范式：（1）**生成式演化约束变异评估（phyloP VEP）**利用野生型与突变序列预测保守性轨迹在 $W=30\text{ nt}$ 窗口内的平均绝对扰动捕获弱保守区致病突变：

$$
\Delta\hat{p}_i = \hat{p}_i^{\mathrm{mut}} - \hat{p}_i^{\mathrm{wt}}, \qquad
s_{\Delta p}(W) = \frac{1}{\lvert W\rvert}\sum_{i\in W} \lvert\Delta\hat{p}_i\rvert;
$$

（2）**剪接预测与致病剪接修复**通过联合输入转录起始/终止位点（**TSS/TES**）实现异构体特异剪接预测，并在固定 **HBB** 内含子致病突变的前提下，以野生型剪接与 **phyloP** 轨迹为条件重设计侧翼窗口以抑制隐蔽外显子掺入；（3）**上下文感知结构建模与蛋白设计**以实验条件文本为条件预测 **RASP2** 反应性引导 **RNA** 二级结构折叠，或同时以主链几何与 **MaSIF** 表面化学特征为条件迭代生成高亲和力蛋白结合剂。
