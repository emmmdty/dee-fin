# Fin-EKG 实验协议（EXPERIMENTS）

> **实验设计的单一权威**：评测协议 + 每章 baseline 矩阵 + 消融矩阵 + 报数规范。
> Phase A–I **照此跑**；主表/消融/协议声明**遵循已发表论文写法**。设计定义见
> [`SPEC.md`](SPEC.md)（§1 四章 + §5 新颖性）；实时状态见 [`TODO.md`](TODO.md)；
> 阶段验收见 [`phases/`](phases/README.md)。**本文件与 SPEC 冲突时以 SPEC 为准。**
> 更新于 2026-07-23（据一轮联网 baseline/竞品/评测协议核实建立）。

## 0. 总原则（每章都要满足才算"可信服"）

1. **主表 + 消融表 + 多种子**，缺一不可：
   - **主表**：本章方法 vs baseline（同数据、同划分、同指标）。
   - **消融表**：本章每个"环节"各做一次 `±`，证明它单独有效（对应一个可信度维度）。
   - **多种子**：seeds 13/17/42，报 `mean±std`（放 Phase H，主结果先单种子）。
2. **baseline 光谱**（避免"只跟老方法比"被审稿人打）：每章至少覆盖
   **① dataset 原文/经典 baseline（参照点）② 近 1–2 年代表方法（≥1 个方法族不同者）
   ③ 通用 LLM（zero/few-shot，作下界或上界参照）**。
3. **报数如实**：数字降就说降；受控实验不冒充真实图结果；不换指标掩盖负结果（`CLAUDE.md` 硬约束）。
4. **协议声明**：论文里显式写清训练/选模/报数用的 split（见 §1），像 SeDGPL 那样说明"为何这样切"。

## 1. 评测协议：test 无 gold 怎么办（有先例，非自娱自乐）

MAVEN 四件套官方 **test 标签隐藏**，走 CodaLab。这是该领域**常态**，处理方式有三档，v4 每章对号入座：

| 档 | 做法 | 先例 | v4 落到哪章 |
|---|---|---|---|
| **A. 官方 test（CodaLab 提交）** | 预测 `results.jsonl`→zip→提交竞赛拿官方分；开发/消融用 valid | MAVEN 检测 / MAVEN-ERE 均有永久竞赛 | **Ch1 检测、Ch2 关系** 主表首选 |
| **B. dev/valid 当 test** | 无官方 test 时，用 development set 当 test 报数并**显式声明** | **SeDGPL 本人**："因 MAVEN-ERE 未发布 test，用 dev 当 test" | **Ch4 后继预测**（CGEP 派生任务无官方 test） |
| **C. train 调参 / valid 报数** | 只发 train/valid 时，train 选模、valid 报最终 | MAVEN-FACT（只发 train/valid） | **Ch3 事实性**；Ch1 论元(MAVEN-Arg) |

**硬规矩**：
- **官方 test 一次性**：CodaLab 提交次数有限，**严禁**拿 test 反复调参。所有超参/早停/模型选择只用 valid。
- **Ch4 可比性红线**：SeDGPL 的 MAVEN 版重建数据**未公开**（论文承诺 review 后发，实际只发 ESC 的 `.npy`），
  其公开 MRR **27.9 不可比**。Ch4 主表**必以我们自跑的 SeDGPL 为基线**（当前单折 MRR 0.1836 / strict 0.1265），
  引用 27.9 须标注"原论文数据构建，非同数据可比"。
- **ESC 必 topic 交叉验证**（EventStoryLine topic），文档级切分泄漏（实测 topic-CV 0.0599 vs doc-split 0.1802）。
- **词表 transductive 要主动交代**：`<a_i>` token 清单覆盖 train+test，与 SeDGPL `to_add.json` 一致，
  只 token 清单跨切分、无标签/图/梯度泄漏——论文里显式说明，不留把柄。

## 2. 每章 baseline 矩阵（新老搭配 · 已核实真实）

> 角色标注：**⛳经典/原文**（参照点）｜**🆕近期代表**（2024–2026）｜**🤖通用 LLM**（zero/few-shot）｜**★我们**。
> "靶"= 该数据集当前公开最强，是本章要打平或超过的目标。**具体选哪几个进主表，按 Phase 当时可复现性再定；本表是候选池。**

### Ch1 身份 —— 事件检测 / 共指 / 论元

| 方法 | 年 | 方法族 | 角色 | 参考数字（MAVEN 系） |
|---|---:|---|---|---|
| DMBERT / MOGANED / BiLSTM+CRF | 2020 | 判别式序列标注 | ⛳ | 检测 F1 ~66–68 |
| CLEVE / 对比预训练 | 2021 | 事件对比预训练 | ⛳ | — |
| APEX-Prompt / 类型特定 prompt | 2022 | prompt tuning | 🆕 | 监督 +≈4 F1 |
| TextEE（重评测基准） | 2023 | 统一评测框架 | 🆕 | 校准各法可比性 |
| Context-Aware Encoder + LoRA | 2024/25 | PLM+LoRA 长尾 | 🆕 | Macro-F1 长尾↑ |
| DiCoRe（发散-收敛推理） | 2025 | LLM zero-shot ED | 🤖 | zero-shot 参照 |
| **规范事件节点（难例判别+校准）** | — | 判别式+不确定性 | ★ | 检测 F1 ~68 靶 |

共指（cross-doc）候选：MAVEN-ERE RoBERTa（⛳，MUC 靶 **86.1**）｜HGCN-ECR 超图卷积（🆕2024）｜
X-AMR 线性共指（🆕LREC-COLING 2024）｜反事实数据增强（🆕2024）｜Synergetic+LLM（🤖2024）。
论元候选：MAVEN-Arg 官方 baseline（⛳2024）｜多答案 QA 式论元抽取（🆕）。

### Ch2 结构 —— 事件关系抽取（本章关键路径）

| 方法 | 年 | 方法族 | 角色 | 参考 F1（MAVEN-ERE）|
|---|---:|---|---|---|
| RoBERTa-base + pair 分类头 | 2022 | 判别式成对分类 | ⛳（官方 strong baseline，**也是 Phase A 复现架构**） | causal/subevent 偏低 |
| ProtoEM | 2023 | 原型增强匹配 | 🆕 | 多关系联合 |
| TacoERE | 2024 | 聚类感知压缩 | 🆕 | 长文档关系 |
| **graph propagation（富事件结构）** | 2024 | 联合+图传播 | 🆕（**当前公开 SOTA**） | **temporal 60.7 / causal 37.4 / subevent 32.9 / MUC 86.1** |
| MAQInstruct | 2025 | 指令式统一 ERE | 🆕 | 生成式统一 |
| LLMERE（带 rationale, O(n)） | 2025 | LLM 生成 + rationale | 🤖/🆕 | 降 O(n²)→O(n) |
| Llama3 / GPT-4（few-shot） | 2024 | 通用 LLM | 🤖 | 下界参照 |
| **判别式 supervised + 一致解码 + CRC 准入** | — | 判别式+全局解码+风控 | ★ | **causal 靶 30–37 / subevent 靶 ~30** |

> Phase A 现状对照：**生成式 SFT+GRPO 探针 causal 召回 0.4%（3/810）/ subevent 0%（0/139）**——
> 文献已证"文档内事件多时生成长度受限、覆盖不全"是生成式通病，**判别式成对分类召回一致更高**，故换判别式打底。

### Ch3 事实 —— 事件事实性检测 + 图净化

| 方法 | 年 | 方法族 | 角色 | 参考（MAVEN-FACT macro-F1）|
|---|---:|---|---|---|
| DLGRN（有向标注图递归网络） | 2021 | 图递归网络 | ⛳ | 早期依存结构法 |
| MAVEN-FACT 官方 fine-tuned（含 RoBERTa/结构感知） | 2024 | PLM + 论元/关系 | ⛳（**dataset 原文 SOTA**） | **best 47.6** |
| MAVEN-FACT 官方 GPT-4 | 2024 | 通用 LLM | 🤖 | **42.8**（劣于 fine-tuned）|
| ModaFact | 2025 | 情态+事实联合 | 🆕 | 跨语言 bonus 参照 |
| **结构感知检测 + 事实性净化** | — | PLM+结构 → 净化算子 | ★ | **macro-F1 靶 ≥47.6** |

> **novelty 落点写法**：明说检测"打底=复现"（用结构非新，MAVEN-FACT 已证）；delta 只主张两点——
> **① gold 输入 vs 预测图(有错)输入的掉点**（原文全用 gold）② **净化后下游 MRR 增益**（没人做过闭环）。

### Ch4 传播 —— 后继事件预测（headline）

| 方法 | 年 | 方法族 | 角色 | 参考（CGEP-MAVEN，SeDGPL 自建数据）|
|---|---:|---|---|---|
| MCPredictor | — | 脚本事件预测 | ⛳ | script chain 参照 |
| CSProm-KG / SimKG | 2023 | KG embedding 补全 | ⛳/🆕 | 图补全式 |
| BART-base | 2020 | 生成式 seq2seq | ⛳ | SeDGPL 最强 baseline **24.7 MRR** |
| Llama3-8B / GPT-3.5-turbo | 2024 | 通用 LLM | 🤖 | 下界参照 |
| **SeDGPL（DsGL+EeCE+ScEP）** | 2024 | 图 prompt learning | ⛳（**基座，自跑基线**）| 论文 27.9（不可比）/ **自跑 0.1836** |
| **下游门控闭环修复 + CS-CRP 误差预算** | — | 闭环控制+conformal | ★ | **repaired > predicted（如实）** |

## 3. 每章消融矩阵（每个环节 = 一个可信维度的因果证据）

| 章 | 消融项（`±`） | 证明什么 | 观测指标 | 现状 |
|---|---|---|---|---|
| Ch1 | ± 难例判别（同类型近义触发词负采样） | 身份不误合并 | **相似事件误合并率** | ⬜ 待建 |
| Ch1 | ± 不确定性感知聚类 | 置信可下游消费 | `node_confidence` **ECE** | ⬜ |
| Ch2 | 判别式 vs 生成式 SFT+GRPO | 判别式解召回 | causal/subevent **P/R/F1**（对照 0.4%） | ⬜ Phase A |
| Ch2 | ± 类不平衡处理（加权CE/focal/负采样） | 稀疏关系可学 | causal recall | ⬜ |
| Ch2 | ± 全局一致解码（闭包/破环/对称） | 结构自洽 | **violation / cycle 率** | 🟡 求解器有、trace 无 |
| Ch2 | ± CRC 边准入 | 关键边不漏 | **分层 FNR** + 准入集大小 | 🟡 原语有、真实图未接 |
| Ch3 | gold 输入 vs 预测图输入 | 预测图鲁棒性（delta①） | macro-F1 **掉点** | ⬜ Phase D |
| Ch3 | ± 结构（论元+关系）特征 | 结构对检测的作用 | macro-F1 | ⬜ |
| Ch3 | ± 事实性净化 | 净化换下游增益（delta②） | 下游 **MRR** 前后 | ⬜ |
| Ch4 | gold / predicted / repaired 三图 | 误差传播 + 修复有效 | **MRR/Hits** 三图对比 | 🟡 受控扫描有、真实图待A/B |
| Ch4 | ± 下游门控（只在MRR↑才接受编辑） | 治 self-refine 掉点 | vs 无门控 self-refine | ⬜ 控制器待建 |
| Ch4 | M1 / M2 / M3 逐个 | 各机制增量 | ΔMRR / risk-coverage | ✅ M1/M2/M3a 已跑 |
| Ch4 | naive vs 预算法 vs 条件回收 | reachability 预算价值 | 覆盖 + 集大小 | ✅ 受控扫描已跑 |

> 已跑结论（如实、含负结果）：**M1（BFS 选边）+0.005 噪声级；M2（reach_anchor）−0.0015 持平=负结果；
> M3a 同覆盖集缩 43–68%（有价值）；M3b 受控扫描预算法守覆盖**。真实三图闭环待 Phase A/B 解堵。

## 4. 报数规范（主表/消融统一口径）

- **指标**：Ch1 检测 micro-F1、共指 MUC/B³/CEAFe/CoNLL、论元 F1、`node_confidence` ECE；
  Ch2 per-relation P/R/F1 + doc-macro-F1、violation/cycle 率、分层 FNR；
  Ch3 5 类 macro-F1 + evidence span F1；Ch4 **MRR/Hit@k**（同报乐观[SeDGPL 口径]与 `mrr_strict`）+ risk-coverage。
- **多种子**：seeds 13/17/42，主结果先单种子跑通，`mean±std` 放 Phase H。
- **协议声明模板**（每章方法部分必写）：数据集 + split 来源 + 报数用 test/valid（引用本文件 §1 对应档）
  + 是否 CodaLab + 词表/切分泄漏澄清。
- **产物落盘**：`runs/<域>/<方法>_<配置>.json`，含配置、指标、n、种子；写入 `TODO.md`。

## 5. 防审稿：与最近竞品的显式区分（详见 SPEC §5）

投稿相关工作**必逐条区分**下列已核实竞品（否则"具体组合"的窄 delta 立不住）：

| 竞品 | 编号 | 它做了什么 | v4 的区分点 |
|---|---|---|---|
| PASC | 2605.18812 | pipeline 联合覆盖，统一 nonconformity | **不碰** reachability/异质保证/drift/条件回收（四点全无）|
| SCRC | 2512.12844 | 单模型 selective + CRC | 不碰 pipeline/跨阶段；**只撞名 → CS-CRP 须改名** |
| C-RAG | 2402.03181 | RAG 生成风险 conformal 上界 | 生成风险非"构建边准入 recall + 下游 reachability" |
| CASCADE | 2605.20468 | 两阶段临床区间，上游不确定性传播 | 医疗回归区间，非事件图/reachability |
| **DeepRefine** | **2605.10488** | **下游导向 KB 精化 + 无 gold GBD 奖励 RL + downstream gains** | **通用 KB 非事件因果图；RL 无覆盖保证；无 reachability/三图误差分解**（★最近威胁，2026-05）|
| MedCEG | 2512.13510 | 结构/因果图作 RLVR 奖励 | 结构作奖励**非新** → RL-reward 仅作消融、**不写"首次"** |

**口径**：一律"据我们所知"+显式区分先例，不写全球首创；headline claim 收窄为
**"事件因果图上、带 reachability 与 conformal 误差预算的下游门控修复"**（区别于 DeepRefine 的通用 KB RL 精化）。
