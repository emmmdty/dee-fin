# 主线重定位提案 v2：通用事件图谱的可验证自动化构建与应用（2026-07-10）

> 状态：**提案，待作者确认**。确认后合并进 `THESIS_DESIGN.md`。
> v1（实体中心，推荐 FinDKG 的 12 类实体本体）已作废——作者明确课题核心是**事件/事理图谱**，
> 不是以实体为中心的知识图谱。
>
> 作者定调（2026-07-10）：核心 = **事件/事理图谱的通用自动化构建 + 应用**；金融语料仅作方法论的
> **一个验证域**；三章与数据形成闭环；**不限中文、可换数据集**；**要有算法创新**；
> 尽量结合 **LLM / agent** 技术，或证明我们的方法**能赋能 LLM**。

---

## 1. 先摆实测证据（决定语料与本体）

### 1.1 事件图谱的命脉是「文档内多事件」，Astock 结构上不具备

事理关系（顺承 / 因果 / 条件 / 上下位）发生在一篇叙事内部。本地实测：

| 语料 | 事件 / 文档 | 文档内 ≥2 事件的比例 | 事件类型数 | 多数类占比 |
|---|---|---|---|---|
| **Astock 新闻摘要** | **0.61** | **1.2%** | 13 | **60.1%**（企业收购） |
| DuEE-Fin 公告全文 | 1.58 | 35.4% | 13 | — |
| **MAVEN-ERE** | **23.34** | **100%** | **168** | **3.8%** |

98.8% 的 Astock 文档只有 ≤1 个事件——**没有链可抽**。

事件抽象粒度消融（同一时间线上，给定前一抽象事件预测下一个，测试对 1791）：

| 抽象粒度 | 节点数 | 后继预测 top-1 |
|---|---|---|
| 事件实例 | 14874 | 0.0% |
| 事件类型（13 类） | 13 | 61.6% ← **多数类基线就是 60.1%** |
| (类型, object 实体) | 6515 | 38.2% |
| (类型, object 类型) | 46 | 38.0% |
| (主体类型, 类型, 客体类型)（事理图谱的 (主,谓,宾) 泛化） | 46 | 38.0% |
| (类型, 论元角色签名) | 84 | 38.7% |

**Astock 上的"后继事件预测"几乎全是在猜多数类，不是事件演化信号。**

> **决定**：Astock / CMIN-CN 退出事件图谱构建，仅保留下游涨跌标签。
> 「新闻摘要语料无法支撑事件图谱构建」作为**负面结果**写入论文（支撑语料选择）。

### 1.2 附带证据：实体中心那条路为什么也走不通（v1 的遗产，保留为消融）

| | quads/实体 | object 只出现一次 | top-10 object 覆盖 | 熵 / 最大熵 | recurrency mrr_tfilt |
|---|---|---|---|---|---|
| Astock entity 图 | 1.21 | 79.2% | 2.6% | **0.978** | **0.0393** |
| ICEWS14（5% 抽样，同重复率） | 2.25 | — | — | — | **0.1903** |
| ICEWS14（全量） | 12.73 | 36.2% | 21.8% | 0.743 | 0.3599 |
| FinDKG | 10.56 | — | 12.2% | 0.845 | — |

同等重复率下 ICEWS 是我们的 5 倍：不是稀疏，是 object 分布接近**最大熵**（预测目标≈随机 ID）。
这条实验保留为「为什么必须事件中心」的实证支撑。

---

## 2. 本体选型（决策依据）

| 本体 | 事件类型 | 论元角色 | 论元实体类型 | 文档 | 事件/文档 |
|---|---|---|---|---|---|
| ACE 2005 | 33 | 36 | 7 | — | — |
| WikiEvents | 50 | 59 | — | — | — |
| DocEE | 59 | 356 | — | — | — |
| RAMS | 139 | 65 | — | — | — |
| **MAVEN / MAVEN-ERE** | **168**（从 598 个 FrameNet frame 归纳） | — | — | **4480** | **23.3** |
| **MAVEN-ARG** | 162 | 612（143 名） | **7** | **4480（同一批）** | — |
| CAMEO（ICEWS/GDELT） | 200+ | — | actor 角色体系 | — | — |
| GLEN | 3000+ | — | — | — | — |
| DuEE-Fin / ChFinAnn | 13 / 5 | ~50 / 35 | — | — | 1.58 |

**事件关系体系：MAVEN-ERE 的四族与事理图谱的四类关系一一对应**（本地实测 train 2913 篇）：

| 事理图谱（CCF 2022 术语） | MAVEN-ERE | 边数 |
|---|---|---|
| 顺承 | temporal `BEFORE` | 683,581 |
| 因果 | causal `CAUSE` | 6,797 |
| 条件 | causal `PRECONDITION` | 29,519 |
| 上下位 | `subevent` | 9,193 |
| （共指，非事理关系） | coreference | — |

合计 837,954 条边，**12.33 边/事件**。

> **选型**：事件类型 = **MAVEN 168 类**（通用主场）+ DuEE-Fin 13 类（金融验证域）；
> 论元实体类型 = **MAVEN-ARG 的 7 个粗类**；事件关系 = **MAVEN-ERE 四族**，写作时对齐事理图谱四关系。
> 理由：MAVEN / MAVEN-ARG / MAVEN-ERE 是**同一批 4480 篇文档**，ch1（检测+论元）与 ch2（关系）
> 的 gold 全齐 —— 通用域上存在**真正的端到端 gold 闭环**，ch1 无需外挂。

---

## 3. 竞品格局（已联网核实，2026-07-10）

| 工作 | 它做了什么 | 它没有的 |
|---|---|---|
| **EventRAG**（ACL 2025 long, ZJU/UCL/蚂蚁） | 事件中心 RAG：抽事件 → 合并语义等价节点 → 补全欠连接关系 → EKG → agent 迭代检索推理。UltraDomain / MultiHopRAG，胜过 NaiveRAG / GraphRAG / LightRAG | **无不确定性量化、无覆盖保证、无弃权、无构建误差分析** |
| **DAO**（arXiv 2406.12197） | 多智能体辩论 + Diverse-RAG + **自适应 conformal 作单阶段拒答过滤**；ACE05 / CASIE 事件抽取；tuning-free | **不建图、单阶段、不跨阶段、未陈述覆盖保证、不处理漂移** |
| **KARMA**（NeurIPS 2025） | 9 个 agent 做 KG 富集，LLM 校验正确率 83.1%，冲突边 −18.6% | 实体 KG；无形式化保证 |
| **CGEP / SeDGPL**（2409.17480） | 提出「事件因果图上的后继事件预测」新任务 | 无风控、无构建侧保证 |
| **NEEG / SGNN**（IJCAI'18）+ MCNC | 从新闻抽叙事事件链 → 事件演化图 → 后继事件预测 | 前 LLM 时代；无保证 |
| **MDPI Electronics 14(19):3827** | 文档级未来事件预测 = EKG + LLM 时序推理 | **自陈"构建重度依赖 LLM，缺乏误差传播的定量分析，无人工校验/过滤"** |
| **arXiv 2605.05476**（Unified Benchmark, KGC + GNN） | text → 自动构建 KG → 下游 GNN，**经验量化**误差传播 | 静态图、无保证 |
| **arXiv 2407.11638** | 指出 GDELT 结构化事件仅约 50% 准确，抽取质量影响预测有效性 | 只是观察 |
| MAVEN-ERE SOTA | temporal 60.7 / causal 37.4 / subevent 32.9 / coref MUC 86.1 | — |

**共同缝隙**：LLM 驱动的事件图谱构建是**多阶段 pipeline，误差逐级累积**；这一点被反复承认
（MDPI 自陈、2605.05476 只能经验量化、2407.11638 只是观察），但**没有人给出形式化的、
有限样本的跨阶段风险界**。这就是本课题的算法位置。

---

## 4. 三章结构与三个算法创新点

**主线**：*通用事件图谱的可验证自动化构建与应用*。
**贯穿**：证据接地 / 可验证性；同一验证器内核的三重身份——**推理门控 · 训练奖励 · 风险控制器**。

### ch1 · 事件抽取（图谱的节点）
- 方法：schema 接地的生成式事件抽取（触发词 + 论元），角色接地输出契约、输出可评测。
- 主表：**MAVEN / MAVEN-ARG**（当前 SOTA：PAIE 论元 40.5% F1，GPT-4 约 30.3% —— 有空间）。
- 验证域：ChFinAnn 86.0 / DuEE-Fin 78.0（SARGE 收编重做，不再外挂）。
- 口径：借 Text2KGBench 的 **ontology conformance / hallucination** 指标（对应 SchemaOK、InvType/InvRole）。

### ch2 · 事件图谱的自动化构建（图谱的边）★ 算法创新点 A1、A2 的构建侧
- 方法：LLM 事件关系抽取（coref / 顺承 / 因果 / 条件 / 上下位）+ grounding 验证器 + 一致性仲裁。
- **★A1｜事理结构约束作为可验证奖励（GRPO-RLVR）**
  事件图谱天然带**可自动验证的结构约束**：时序传递闭包、因果无环、共指等价类、子事件层级。
  把它们做成 verifier，构成复合可验证奖励（format + grounding + **structure** + gold F1），用 GRPO 训练。
  *新颖性*：DAO 是 tuning-free；VeriGate / Faithful GRPO 是通用的 verifier-gated GRPO；
  **把事理结构约束当 RLVR 奖励，检索未见先例**。这是「算法味」最足、最像 LLM 后训练的一点。
- **★A2（构建侧）｜CRC 边准入**：以 FNR ≤ α_e 的有限样本保证保留 ≥1−α_e 的真实边并提升精度。
- 主表：**MAVEN-ERE**（对标 temporal 60.7 / causal 37.4 / subevent 32.9 / coref 86.1）。
- 验证域：CCKS-2021 中文金融因果。
- agent 味：proposer / grounding-verifier / consistency-arbiter 三 agent + 可审计证据链（对标 KARMA，但我们有保证）。

### ch3 · 事件图谱的应用（★ A2 的组合保证、A3）— **主任务已定：CGEP**

**主任务 = CGEP（Causality Graph Event Prediction）**，基座 = **SeDGPL**（EMNLP 2024 Findings, 2409.17480）。

选它的理由（四条约束全中）：
1. **闭环**：CGEP-MAVEN 由 **MAVEN-ERE 构建** —— 正是 ch2 的主表。ch1/ch2/ch3 落在同一批 4480 篇文档上。
   本地实测：MAVEN-ERE 每篇因果子图平均 11.7 节点 / 12.8 边，与论文报告的 8.4 / 12.9 同量级，**可自行重建**
   （其 repo 仅 2 commits、无 license、数据未完整释出）。
2. **算法味**：SeDGPL 是真模型（图线性化 + 三路 PLM 编码 + 两级门控融合 + 对比学习 + prompt 预测），
   不是脚本流水线 —— **画得出模型架构图**。
3. **改进点是"小修补"且踩在它的痛处**（见下 M1–M3），不需要另起炉灶。
4. **绕开主竞争点**：不在 MRR 上硬拼。gold 图上追平即可，主张落在
   *constructed 图上的鲁棒性* 与 *selective accuracy@coverage / AURC / 组合覆盖*。

**要打的数字**（SeDGPL 原文，CGEP-MAVEN / CGEP-ESC）：

| 模型 | MAVEN MRR | Hit@1 | Hit@3 | Hit@10 | ESC MRR |
|---|---|---|---|---|---|
| SimKG | 9.3 | 4.5 | 9.2 | 18.0 | 14.9 |
| MCPredictor | 18.1 | 13.0 | 18.4 | 27.3 | 9.7 |
| CSProm-KG | 22.3 | 18.1 | 23.2 | 31.0 | 14.2 |
| BART-base | 24.7 | 19.5 | 24.5 | 34.8 | 16.0 |
| **SeDGPL** | **27.9** | **21.9** | **28.9** | **40.8** | **19.6** |

（其原文亦测了 Llama3-7B / GPT-3.5-turbo，均不敌 SeDGPL —— 这是 A3「赋能 LLM」的现成对照。）

#### 三个"小修补"，各挂在 SeDGPL 的一个模块上

- **M1｜风险感知的图线性化（Risk-aware DsGL）** ★ 直接消解它自陈的 limitation
  SeDGPL 原文 limitation：*"受 PLM 输入长度限制，线性化时不得不丢弃部分三元组，可能损失有益信息。"*
  它按到 anchor 的最短路距离截断（启发式）。我们把 **CRC 边准入分数**引入截断准则：
  在 token 预算下选边，使 **FNR ≤ α_e** 有限样本成立。把一个工程妥协变成有保证的算法组件。
- **M2｜结构感知的因果编码（Structure-aware EeCE）**
  它融合 graph / contextual / schema 三路表示。我们加第四路 **结构一致性特征**
  （该候选是否引入因果环、是否违反时序传递闭包、共指等价类归属），作为门控信号；
  同一验证器同时充当 **A1 的 GRPO 奖励**。
- **M3｜选择性对比预测（Selective ScEP → CS-CRP head）**
  它的 ScEP 确定性地输出单个事件。我们换成 **conformal 预测集 + 弃权**：
  nonconformity 取对比得分或忠实度；构建期 α_e 与应用期 α_p 组合，并按可达性修正。

- **★A3｜赋能 LLM**：把验证过的事件因果图 + CS-CRP 作为结构先验喂给 LLM predictor，
  论证「可验证的图谱构建 + 形式化保证 → LLM 从落后于专用模型变成带保证的选择性预测器」。
  无需另开 RAG 章节（EventRAG 那条线降级为相关工作 / future work）。
- **杀手实验**：**gold ECG vs 我们构建的 ECG** 上的 CGEP 性能差 = 误差传播曲线；CS-CRP 给它一个界。
  这正是 MDPI 3827 自陈缺失、2605.05476 只能经验量化的东西。
- 验证域：金融事件图（DuEE-Fin / ChFinAnn 构建）+ Astock/CMIN 涨跌标签的选择性选股。
- 备选/附表：MCNC（经典脚本事件预测）；CGEP-ESC 作小样本附表。

### 数据闭环（通用域，gold 全齐）

```
MAVEN 4480 篇文档
   ├─ ch1  事件检测(MAVEN 168类) + 论元抽取(MAVEN-ARG, 7 实体类型/612 角色)   [gold]
   ├─ ch2  事件关系(MAVEN-ERE: coref/顺承/因果/条件/上下位) → 事件图谱        [gold]
   └─ ch3  后继事件预测(CGEP/MCNC) + EKG-RAG(MultiHopRAG/UltraDomain)        [gold]
                       ↑ CS-CRP 把 ch2 的 FNR 保证与 ch3 的覆盖保证组合起来

金融验证域：ChFinAnn / DuEE-Fin 公告全文 → 事件图谱 → 事件预测 → 选择性选股(Astock/CMIN 标签)
```

---

## 4.5 模型架构图（论文主图，非流程图）

同一验证器内核在图中出现 **三次**，对应它的三重身份。这是全文的视觉主张。

```
                            ┌──────────── 共享验证器内核 ────────────┐
                            │  时序传递闭包 · 因果无环 · 共指等价类     │
                            │  · 子事件层级 · 证据接地                │
                            └───┬──────────┬──────────┬─────────────┘
                     ①训练奖励  │  ②准入控制 │  ③风险控制
   ┌────────┐                  │          │          │
   │  文本   │                  │          │          │
   └───┬────┘                  │          │          │
       │                       │          │          │
  ┌────▼─────────┐             │          │          │
  │  事件抽取器   │  触发词+论元 │          │          │
  │ (LLM + LoRA) │             │          │          │
  └────┬─────────┘             │          │          │
       │ 事件节点               │          │          │
  ┌────▼─────────┐             │          │          │
  │ 关系提案器    │◄────GRPO────┘          │          │
  │ (LLM 生成式)  │   可验证奖励 (A1)       │          │
  └────┬─────────┘                        │          │
       │ 候选边 + 置信度                    │          │
  ┌────▼─────────┐                        │          │
  │ CRC 边准入    │◄───────────────────────┘          │
  │  FNR ≤ α_e   │      有限样本召回保证 (A2 构建侧)    │
  └────┬─────────┘                                   │
       │ 构建出的事件因果图 Ĝ                           │
  ┌────▼──────────────────────────────────────┐      │
  │            改造版 SeDGPL                    │      │
  │  ┌──────────────────────────────────────┐ │      │
  │  │ M1 风险感知图线性化 (Risk-aware DsGL)  │ │      │
  │  │   按准入分数×距离在 token 预算下选边     │ │      │
  │  └──────────────┬───────────────────────┘ │      │
  │  ┌──────────────▼───────────────────────┐ │      │
  │  │ M2 结构感知因果编码 (EeCE + 第四路)     │ │      │
  │  │   graph ⊕ context ⊕ schema ⊕ 结构一致性 │ │      │
  │  │            └── 两级门控融合 ──┘          │ │      │
  │  └──────────────┬───────────────────────┘ │      │
  │  ┌──────────────▼───────────────────────┐ │      │
  │  │ ScEP 对比预测 → 候选事件打分            │ │      │
  │  └──────────────┬───────────────────────┘ │      │
  └─────────────────┼──────────────────────────┘      │
                    │                                 │
       ┌────────────▼─────────────┐                   │
       │ M3  CS-CRP conformal head │◄──────────────────┘
       │  α_e ⊗ α_p + 可达性修正    │   漂移鲁棒覆盖 (A2 应用侧)
       └────────────┬─────────────┘
                    │
        ┌───────────▼────────────┐
        │  预测集合  /  弃权       │   组合覆盖 ≥ 1−α（漂移下仍成立）
        └────────────────────────┘
```

## 5. 与现有代码资产的映射（改造量）

| 资产 | 现状 | 本次去向 |
|---|---|---|
| `relations/` 验证器 + 一致性求解器 + CRC 边准入 | 已实现，253 测试绿 | **直接复用**，A1 的 verifier 就是它 |
| `relations/rl/`（GRPO-RLVR，复合奖励） | 已实现，T1 SFT adapter 就绪 | **A1 的载体**；奖励里加结构约束分量 + 类别重加权 |
| `core/calibration/`（split/aci/weighted/crc）+ `propagation.py`（CS-CRP） | 已实现 | **A2 的载体**；补加权 CRC（非可交换）与 e-process 上界 |
| `relations/pairs.py`（pair harness） | 已实现 | MAVEN-ERE 主表口径 |
| `forecasting/`（re_gcn / recurrency / hybrid / path_rl） | 已实现 | **降级**：ICEWS/TKG 只作参考，主任务改 CGEP / MCNC |
| `forecasting/llm/`（rules / retrieve） | 部分实现 | **升级为 A3 的 EKG-RAG 检索器** |
| `agents/`（proposer / verifier / arbiter） | 已实现 | agent 味的载体，对标 KARMA |
| SARGE（`external/sarge`） | subtree 外挂 | **收编**：ch1 重做，主表换 MAVEN-ARG，金融作验证域 |
| Astock entity 图 / `quad_mode: entity` | 本轮刚建成 | **降级为负面结果与消融**（§1.1、§1.2） |

---

## 6. 待定与风险

- **ch3 主任务的最终取舍**：CGEP（新任务、图上推理、conformal 干净）/ MCNC（经典、可比）/
  EKG-RAG（直接对打 EventRAG、体现赋能 LLM）。当前倾向：**CGEP + MCNC 作主表，EKG-RAG 作赋能 LLM 的独立一节**。
- **ch1 通用性风险**：SARGE 目前只在中文金融公告上验证；MAVEN-ARG 上从零起步，SOTA 仅 40.5% F1，
  但也意味着空间；需要 GPU 预算。
- **A1 新颖性**需在终稿前复核 VeriGate（2605.30451）与 Faithful GRPO（2604.08476）的具体做法。
- **A3 与 EventRAG 的可比性**：需拿到 UltraDomain / MultiHopRAG 并复现其 baseline。
- **金融闭环的下游信号弱**（方向性 ≈0.48），按 future work 如实呈现。

## 7. 参考

- Wang et al., *MAVEN-ERE*, EMNLP 2022, arXiv 2211.07342.｜*MAVEN-ARG*, arXiv 2311.09105.｜*MAVEN*, arXiv 2004.13590.
- Yang et al., *EventRAG: Enhancing LLM Generation with Event Knowledge Graphs*, ACL 2025 long (2025.acl-long.830).
- *Debate as Optimization (DAO): Adaptive Conformal Prediction and Diverse Retrieval for Event Extraction*, arXiv 2406.12197.
- Zhan et al., *What Would Happen Next? Predicting Consequences from An Event Causality Graph (CGEP/SeDGPL)*, arXiv 2409.17480.
- Li et al., *Constructing Narrative Event Evolutionary Graph for Script Event Prediction (NEEG/SGNN)*, IJCAI 2018, arXiv 1805.05081.
- *KARMA: Multi-Agent LLMs for Automated KG Enrichment*, NeurIPS 2025, arXiv 2502.06472.
- Guan et al., *What is Event Knowledge Graph: A Survey*, TKDE 2023.
- *A Unified Benchmark for Evaluating KG Construction Methods and GNNs*, arXiv 2605.05476.
- *A Comprehensive Evaluation of LLMs on Temporal Event Forecasting*, arXiv 2407.11638.
- *Text2KGBench*, ISWC 2023, arXiv 2308.02357.｜*SCTc-TE*, arXiv 2312.01052.｜*MIRAI*, NeurIPS 2025, arXiv 2407.01231.
- CCF「事理图谱」术语发布，2022.
