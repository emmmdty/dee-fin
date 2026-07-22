# 证据与风险约束的事件图谱构建及其误差传播 · 硕士论文三章设计（供专家评审）

> **文档性质**：学位论文实验方案设计稿，供专家评审。日期 2026-07-20（v3 修订 2026-07-21）。
> 作者专业 = 控制工程（模式识别 / 深度学习方向）。下一节点 = 2027-04 论文初稿 + 终辩。
> **战略方向**：**前沿方法进 MAVEN 闭环**（不切纯跨文档，保 CGEP/SeDGPL/conformal 资产，见 §5.4）。
> **论文工作题**：*证据与风险约束的事件图谱构建及其误差传播研究*。三章递进：
> **Ch1 证据约束的规范事件节点构建（含不确定性建模）→ Ch2 风险受控的全局一致多关系事件图构建与
> 可追溯修复 → Ch3 事件图构建误差传播与可靠后继事件预测。**

## v3 修订记录（据专家评审 + 作者「综合评判、专家不等于正确」指示）

| 专家意见 | 处置 | 说明 |
|---|---|---|
| ① MAVEN-ERE 官方 test 标注未就位 | **采纳（已亲验坐实）** | 本地 test 857 篇 events/关系**全 0**；改用 train+valid + 内部固定测试划分协议（§3.1）。既有 CGEP 用 `train+valid`、未碰 test，实验干净，仅 v2 描述有误 |
| ② Ch1 创新不足 | **采纳，按硕士级校准** | Ch1 加真机制（相似事件难例判别 + 不确定性感知规范化 + 簇级证据/置信聚合），标定降为辅助（§4-Ch1）。不扩成第二篇独立论文 |
| ③ Ch2 塞太多贡献 | **采纳** | 收敛为单一命题 + 三层，打底器不算贡献（§4-Ch2） |
| ④ CRC 表述过强 | **采纳，不矫枉过正** | 改为「交换性 + 固定后处理下的**边际期望** FNR」，分层报告；保留其真保证、不摆成「什么都不保证」（§5.5） |
| ⑤ Ch3 被 CFEP 挤压 | **采纳转向，改进攻框架** | CFEP（2026.findings-acl.258）已核=**TKG 时间戳预测 + 纯预测阶段、不碰构建**；Ch3 转「构建误差传播」= 亮出本项目独有的跨阶段闭环资产，非防御避让（§4-Ch3、§7） |
| 软化 v2 若干过强表述 | **采纳** | 「三段式公认」→「主流组织方式」；「coref 本质不是边」→设计选择；「端到端 distribution-free guarantee」→「误差传播分析 + 有前提保证」；一律不写「首次」 |
| （专家未提）金融层 | **保留** | 课题标题要求金融验证，是学位硬约束（§3.3） |
| （专家未点）Ch3 真实误差传播依赖 Ch2 | **补入** | Ch2 不成 → Ch3 退受控扰动版（`cross_stage.py` 已实现），依赖写入 §4-Ch3/§8/§10 |

---

## 1. 研究问题、总体目标、方法论

### 1.1 领域与要解决的问题
课题处在**事件图谱（事理图谱）自动化构建及应用**领域。据 2024–2026 文献，痛点不在「能不能抽」，
而在**构建出的图与图上的预测「多可用、多可靠、可否追溯」**：
1. **可用性缺口**：文档级检测长尾难（MAVEN 检测最佳 F1 ~62）；causal/subevent 是**稀疏但关键**的边，
   主流抽取器召回低（本项目当前抽取器 causal 召回 0.4%）；**成对独立分类保证不了全图对称/传递/无环**。
2. **可靠性缺口**：抽取丢边（假阴）无法量化；推理错误无法控制；两阶段误差**级联传播**（上游丢一条
   边 → 下游正确答案不可达）。
3. **可追溯缺口**：生成式输出**缺显式证据 / provenance**（综述列为硕士适配度「很高」的空白）。

### 1.2 总体目标
交付**能在真实语料上跑通的「事件图谱构建 → 事件预测」系统**，做到：
- **可用**：Ch1 节点、Ch2 边各自超过 baseline，构建的图足以支撑 Ch3 在**真实图**（非金标图/受控模拟）
  上做后继事件预测；
- **可靠**：给节点置信、边召回、预测覆盖三处配统计控制，并**分析其端到端误差传播**（不作无前提的
  端到端 distribution-free 保证）；
- **可追溯**：节点挂 evidence provenance，边可溯源与可修复（repair trace）；
- **整体更优**：三章同一语料、同一套图往下流，**「修好上游 → 下游变好」可实测**。

### 1.3 方法论
- **工程方法论**：冻结跨阶段契约 + 插件式 registry + CPU/GPU 惰性分层（已落地）。
- **建模方法论**：对齐 2024–2026 前沿——**pairwise 局部分类 → global graph modeling / 约束优化 / 修复**；
  **抽取 → 抽取 + 验证/修复**；节点侧引入 **evidence-aware + uncertainty-aware** 表示。判别式仅作
  「打底边打分器」，**贡献上移到全局一致性 + 修复 + 风险控制层**（§5.2）。
- **实验方法论**：复现锚定 + 逐字节一致 A/B；无发布数据的公开数字标「不可比」；**官方隐藏 test 走
  CodaLab、主实验用内部固定划分**（§3.1）。
- **报告方法论**：结果如实（数字降就说降；负结果如实入消融，不换指标绕过）。
- **可靠性的方法学身份**：conformal / 风险控制是「让构建与预测可信」的**支撑组件**（对齐综述「验证-
  修复」新兴方向），**不是主菜、不作 Ch3 唯一新颖性**；主菜是构建与预测本身先做扎实、超过 baseline。

> **控制工程学科归属**：模式识别、深度学习本属控制工程范畴。可靠性反馈校准与自适应控制有真实概念
> 对应，绪论点一句作学科锚点即可；**不作主线、不引入控制器设计**（作者明确此意）。

---

## 2. 领域大趋势与本方案定位（据综述，证明对齐学界）

事件图谱构建的三段式（事件抽取→关系抽取→下游推理）是**一种主流组织方式**（EKG 综述），本论文
Ch1（节点）/ Ch2（边）/ Ch3（预测）沿此组织。**2024–2026 方法论转向**（已独立核验承重论文）：

| 主线 | 状态 | 代表工作 | 对本方案的意义 |
|---|---|---|---|
| 长文档事件图生成 | 主流 | Set-Aligning(NAACL24)、CALLMSAE(NAACL25)、Global Zero-shot(EMNLP25) | Ch2 借「生成后约束优化」结构 |
| 跨文档事件对齐/聚合 | 主流 | CDEE、SECURE(ACL24)、MEET(EMNLP25) | Ch1 **关键泛化实验**基线 |
| 多关系联合 + 全局一致性 | 主流 | GraphERE、LogicERE(AAAI25)、GDLLM | Ch2 主干基线 |
| 事件中心 KG 构建/融合 | 有代表作 | EventRAG、ChronoGrapher | 背景 |
| 验证/修复/补全/增量 | **新兴** | PLAF(NN26)、CALLMSAE、Global Zero-shot | **本方案可靠性主线归宿** |

**三点转向恰好印证本项目已有方向**（一致性求解器 + CRC 准入 + conformal）踩在前沿缺口上：① pairwise
→ global（独立分类保证不了全图一致）；② 抽取 → 抽取+修复（无代表作把 LLM 当裸生成器放行）；③
provenance/证据是公认开放缺口。**定位**：规范三段式 + 前沿方法论 + MAVEN 闭环；LLM 蒸馏留作 Ch2
第二实验、跨文档留作 Ch1 泛化。**对齐学界、不做独立 demo、不破闭环。**

---

## 3. 数据集与数据闭环

### 3.1 主干语料：MAVEN / MAVEN-ERE（贯穿三章）★test 口径已修正
**同一批文档贯穿三章。** 本地实测标注可用性（`data/raw/maven_ere/`）：

| split | 文档 | events | causal | temporal | subevent | 用途 |
|---|---:|---:|---:|---:|---:|---|
| train | 2913 | 67984 | 36316 | 792445 | 9193 | 训练 |
| valid | 710 | 16301 | 9698 | 188928 | 2826 | 开发 + 内部测试划分来源 |
| **test** | 857 | **0** | **0** | **0** | **0** | **官方隐藏标注 → 仅 CodaLab 端到端提交** |

**⚠️ 实验协议（据专家意见修正）**：官方 test **无任何 gold**（连事件都没有）。故：**train 训练、valid 开发，
从 train+valid 派生的实例上做固定内部测试划分报主表；官方 test 结果（如报）经 CodaLab 提交。** 论文
**不得**写「官方测试标注已就位」。既有 CGEP 构建用 `--split train+valid`（`scripts/build_cgep.py`）、
**从未碰官方 test**，现有 Ch3 结果在此点上干净、可复现。

```
MAVEN 检测标注 ──Ch1──▶ 证据感知规范事件节点   （events + mention 分组=coref；train+valid）
MAVEN-ERE 关系 ──Ch2──▶ 全局一致的事件图边     （temporal/causal/subevent；train+valid）
CGEP（重建）  ──Ch3──▶ 后继事件预测 + 误差传播  （2994 文档 / 8.82 节点·ECG / 13.21 边·ECG）
```
MAVEN 构建→CGEP 预测经核实是**真实研究线**（CGEP 2409.17480、ERGO 等），非自造。

### 3.2 辅助/泛化/协议语料
- **ECB++（MEET）/ CLES（中文跨文档）**：**Ch1 关键泛化实验**——检验 canonical event 表示能否从文档内
  聚合泛化到跨文档聚合。基线 SECURE/MEET/DIE-EC。**验证通用性，不进主干闭环**。
- **OmniTemp（Global Zero-shot）**：⚠️ **仅 30 文档 / 470 mention / 3483 关系、只有时序** → **不作主数据**，
  仅作全图一致性评测协议参考。
- **ESC**：Ch3 第二数据集，**必须 topic 交叉验证**（本项目已证 SeDGPL 公开 19.6 依赖文档级切分泄漏：
  topic-CV 0.0599 vs doc-split 0.1802，此发现本身是协议贡献）。

### 3.3 金融应用验证层（专家未提，但为课题标题硬约束，保留）
**闭环以 MAVEN 主干为准；金融作应用验证层**，兑现课题标题「金融领域数据验证」：
- **CCKS-FinCausal**（中文金融因果，本地 `data/processed/ccks_fin_causal/` 带 train/val/test）→ 验证 Ch2
  关系抽取/一致性管线可迁移到金融文本。
- **SARGE**（金融事件抽取器，external/sarge subtree）→ 金融侧 Ch1 节点来源。
- 定位 = **聚焦金融案例研究/一节**；**不拿整章赌金融稀疏因果图**（Astock、entity-mode 已证死路，冻结）。

---

## 4. 三章设计（逐章：问题 / 方法 / 贡献 / 数据 / 预期 / 评测）

### Ch1 — 证据约束的规范事件节点构建与不确定性建模（★据专家意见加真机制）

- **核心研究问题（重定位）**：从文档多个事件提及构建**去重、可溯源、身份统一**的 canonical event
  nodes——**多个 mention 是否属于同一真实事件？相似但不同的事件如何避免误合并？节点置信度如何由
  mention 级不确定性聚合？**
- **方法主干（含真机制，非仅检测+标定）**：事件检测（PLM 编码 + 类型分类）→ **evidence-conditioned
  mention-pair 表示 + 相似事件难例判别（hard-negative）→ 不确定性感知规范化聚类 → 簇级证据/置信聚合**，
  输出每节点 `{event_type, canonical_trigger, canonical_arguments, mention_cluster, evidence_spans,
  node_confidence, provenance}`。**标定（temperature/conformal）降为辅助组件，不作主创新。**
- **核心贡献**：**evidence-aware & uncertainty-aware event canonicalization**——把「事件共指」提升为「统一
  事件节点构建」，显式处理相似事件误合并/漏合并、证据冲突、簇级置信聚合。命中综述「provenance」空白。
- **数据**：MAVEN 检测 + MAVEN-ERE coref（主，train+valid）；**ECB++/CLES 跨文档泛化（关键实验）**。
- **基线**：MAVEN 检测+coref 基线；泛化对比 SECURE/MEET/DIE-EC/Okay。
- **评测**：检测 micro-F1；coref CoNLL/MUC/B³/CEAF_e；**相似事件判别准确率**；evidence attribution P/R；
  标定 ECE/reliability（辅助）。
- **预期效果**：检测 F1 ~60+、coref MUC ~86 可比区间；相似事件误合并率显著下降；节点置信可用于下游。
  **达不到怎么办**：evidence 对齐评测难 → 优先可自动计算指标 + 小规模人工核验；退为「canonical event
  table + 难例判别」仍成章。

### Ch2 — 风险受控的全局一致多关系事件图构建与可追溯修复（★收敛为单一命题）

- **核心研究问题**：在 Ch1 节点上构建 temporal/causal/subevent 图，**同时控制关键边漏检风险并减少结构
  冲突**。当前抽取器 **causal 召回 0.4%（3/810）、subevent 0%**（探针实测），且成对独立分类保证不了
  全图对称/传递/无环。
- **统一命题（一句话）**：**在控制稀疏关键关系漏检风险的同时，生成满足跨关系全局约束、且修复过程可
  追溯的事件图。** 只保留三层：

  ```
  ① 候选边打分器（判别式成对分类，打底、非贡献）
        ↓ 每类关系基础置信度
  ② 全局一致解码 + 可追溯修复（时序传递闭包 / 因果无环 / coref 对称；记录 repair trace）  ← 核心
        ↓
  ③ 风险受控边准入（校准集上控制关键边 false-negative risk）                              ← 核心
  ```
  层 ② 复用既有 `consistency/`（identity/greedy），层 ③ 复用 `admission.py::edge_admission("crc")`。
  **GAT / LLM 蒸馏 / RAG / 多智能体不进主方法**（至多作第二实验或消融）。
- **核心贡献**：层 ②+③ ——全局一致性 + provenance 修复 + 风险受控准入（下游漏因果边=断可达性，给了
  准入的原理动机）。
- **新颖性差异（逐条，见 §7）**：LogicERE（逻辑约束）/ Global Zero-shot（时序约束）**都无分布无关风险
  控制、非下游驱动**；PLAF/SchemaEGC 是图补全非端到端。
- **数据**：MAVEN-ERE 三类关系（主，train+valid）；OmniTemp 仅作一致性评测参考。
- **基线**：MAVEN-ERE 官方基线、GraphERE、LogicERE、Global Zero-shot、无约束 pairwise、只约束无 CRC、
  只 CRC 无修复。
- **评测**：三类 relation P/R/F1、doc-level macro-F1；**边际/文档级/关系类条件 false-negative risk**、
  准入边集大小；cycle rate、传递性违反、跨关系矛盾率；repair precision/recall、repair gain；evidence/
  provenance coverage；**CGEP graph 可重建率**。
- **预期效果**：causal F1 从 0.4% 召回 → 文献可比（~30–37）；violation rate 显著下降；准入后关键边风险
  受控；ECG 可重建率提升 → **解堵 Ch3 真实闭环**。**达不到怎么办**：见 §10（召回 <10% 维持受控模拟；
  修复收益小则退 consistency-aware reranking）。**Ch2 是最值得作主要投稿论文的章。**

### Ch3 — 事件图构建误差传播与可靠后继事件预测（★据 CFEP 转向，进攻框架）

- **核心研究问题（转向）**：真实构建图中的**节点/边错误如何传播影响后继事件预测**？Ch2 的修复是否真正
  改善下游？——这是本项目**独有资产**（跨阶段构建→预测闭环），也是 CFEP **明确不碰**的空间。
- **方法主干**：SeDGPL（DsGL+EeCE+ScEP，已自跑复现：CGEP-MAVEN MRR 0.1836 乐观 / 0.1265 strict）。
  **三种输入图对比 + 受控扰动**：① gold graph、② raw predicted graph（Ch2 抽取）、③ repaired graph
  （Ch2 修复后）；受控扰动 = 删/增因果边、并/拆节点、扰乱时序（`succession/cross_stage.py` +
  `induce_reachability` **已实现一大半**）。
- **核心贡献（转向后）**：**construction-error-aware event graph prediction**——量化哪类 node/edge error 最
  破坏 reachability、Ch2 修复的下游增益、节点→边→预测的不确定性传播、evidence chain 是否仍可达可解释。
  **conformal 覆盖/选择性预测/risk-coverage 曲线保留为可靠性模块，不作主要新颖性。**
- **已有可靠性证据（保留，降为组件）**：M3a 覆盖保证选择性预测（同覆盖下集大小 SeDGPL vs frequency：
  90% 243 vs 425 −43%、70% 99 vs 313 −68%）；M1/M2 精度机制噪声级、如实入消融。
- **★依赖（专家未点，补入）**：② predicted / ③ repaired graph 研究**依赖 Ch2 成功**（causal 召回修到可用）。
  **Ch2 不成 → Ch3 退回受控扰动版**（已实现），仍能回答「构建误差如何影响下游」这一核心问题。
- **数据**：CGEP-MAVEN（主）+ ESC topic-CV（副）。
- **与 CFEP 区分（§7）**：CFEP = TKG 时间戳未来预测 + 纯预测阶段 conformal；本项目 = CGEP 后继预测 +
  **构建误差传播**（gold/predicted/repaired 对比 + 跨阶段），任务、图类型、研究对象均不同。

---

## 5. 关键设计决策与证据

### 5.1 章节划分（节点 / 边 / 预测）
coref 用于**节点身份归并**（放 Ch1 节点章），temporal/causal/subevent 作为**图的边**（放 Ch2）——
这是**设计选择**（非「coref 本质不是边」的本体论断言）。边界清晰、各章有实质、直接映射标注。

### 5.2 Ch2 主干：判别式打底，贡献上移（据综述）
综述有据地指出「只在事件对上独立分类保证不了全图一致」「只换 backbone 不够」。故判别式降为「打底边
打分器」（对齐 GraphERE/LogicERE），**贡献上移到一致性+修复+风险控制**。当前 0.4% 召回本身是一次
生成式 LLM 微调（SFT+GRPO LoRA）的产物 → 不重赌生成式；LLM 蒸馏/MAQInstruct 选择重构留作第二实验。

### 5.3 可靠性作支撑组件，不作主菜
Ch1 标定 + Ch2 风控准入 + Ch3 覆盖是贯穿三章的可靠性主线（对齐「验证-修复」），但**每章须先在主任务
超过 baseline**；可靠性是「整体更优」的严谨化，不替代有效性、不作 Ch3 唯一新颖性。

### 5.4 战略方向：前沿方法进 MAVEN 闭环（不照搬跨文档方案甲）
纯跨文档（ECB++/CLES）三章三套语料、且综述方案甲明确排除下游预测 → **打断 MAVEN→CGEP 闭环**，与
课题「构建→预测闭环」的本冲突；OmniTemp 太小无法撑 Ch2。故：**采纳前沿方法论、保 MAVEN 闭环、跨
文档降为 Ch1 泛化。** 代价=投稿前沿性略低于方案甲，收益=闭环完整 + 复用 CGEP/SeDGPL/conformal。
（作者已拍板；专家复核同意不切纯跨文档。）

### 5.5 CRC 保证的严谨表述（★据专家意见修正）
Conformal Risk Control（Angelopoulos 等）的标准保证是：**交换性假设 + 单调损失下，控制新样本上的
边际期望风险 E[L] ≤ α**。它**不**直接等价于「每篇文档 FNR≤α」「每类边 FNR≤α」「每个 event pair 单独
保证」「漂移后自动成立」「修复器改动后自动保持」。故本文：
1. 表述改为「**在交换性与固定后处理协议下，控制 causal/subevent 预测集的边际期望 false-negative
   risk**」；
2. **分层报告** marginal FNR / 文档级 FNR 分布 / 关系类条件 FNR / 准入边集大小 / precision–recall–risk
   权衡，**不只报「达到 1−α recall」**；
3. **修复顺序问题**：若修复在准入后改动边，则把「准入+修复」作为**单一固定后处理映射**、在其输出上
   校准（保交换性），或另设修复后再校准——论文显式讨论此设计与其对保证的影响。

---

## 6. 三章综合与「整体更优」（保证语言已软化）

1. **真闭环（最硬兑现）**：Ch2 修好 → Ch3 在**真实构建图**上跑（当前仅金标图/受控模拟），「修好上游→
   下游变好」可实测；Ch2 不成则以**受控扰动**量化误差传播（已实现）。
2. **端到端误差传播分析 + 有前提的预算**：节点漏（α_d）+ 边漏（α_e）+ 预测漏（α_p）经 union bound 组合，
   **在各阶段交换性 + 固定后处理假设下**给端到端界；reachability 单列 + 条件回收（`propagation.py` 已测）。
   **明确标注前提，不作无条件 distribution-free 保证。**
3. **证据贯穿**：Ch1 节点 evidence provenance → Ch2 边 repair trace → Ch3 预测 evidence_chain，全链可追溯。

---

## 7. 新颖性与差异性（防雷同，逐条区分）

**口径（硬约束）**：一律「据我们所知」、**不写「首次」**；无发布数据的公开数字标「原论文数据构建、
非同数据可比」；**不写「官方 test 标注就位」**。

- **Ch1**：检测/coref 主干为复现，不主张新颖；贡献在 **uncertainty-aware evidence-constrained
  canonicalization + 相似事件判别**。跨文档泛化与 SECURE/MEET/DIE-EC/Okay 逐条比。
- **Ch2**：判别式打底不主张新颖；贡献在 **风险受控 + provenance 修复 + 下游驱动**的一致图构建。逐条
  区分：LogicERE（逻辑约束、**无风险控制、非下游驱动**）、Global Zero-shot（**只时序、无 causal/subevent、
  无风险控制**）、GraphERE（关系预测、**无全图修复与保证**）、PLAF/SchemaEGC（图补全、**非端到端**）、
  CALLMSAE（LLM 生成+refinement、**无统计风险保证**）。
- **Ch3**：SeDGPL 为**基座**（自实现复现作基线），非竞品。★**与 CFEP（2026.findings-acl.258）区分**：
  CFEP = TKG 时间戳未来预测 + 纯预测阶段 conformal、**不碰构建**；本项目 = CGEP 后继预测 + **构建误差
  传播**（gold/predicted/repaired 对比 + 跨阶段风险）。conformal 组件另逐条区分 PASC(2605.18812)/
  SCRC(2512.12844,⚠️撞名待改)/Two-stage CRC Ranked Retrieval(2404.17769,**投稿前必读全文**)。RL-奖励
  线被 MedCEG(2512.13510) 抢占 → 降为消融、不主张。
- **代码防抄袭**：SARGE 以 subtree 隔离且来源声明清晰；SeDGPL 按论文自实现（有自测锁），非搬运。

---

## 8. 分阶段执行计划（至 2027-04 初稿）

> 顺序原则：先修瓶颈（Ch2）用金标节点解耦，再建上游（Ch1），再连真闭环。GPU 已获标准授权（本地
> 全绿后自行起，逐卡核查 + 如实报）。校验：`uv run pytest` / `ruff` / smoke。

| 阶段 | 时段 | 内容 | 交付物 / 验收 |
|---|---|---|---|
| **P0** | 即刻（1–2 天） | 提交现工作区（M1/M2/M3 落库）；本 v3 稿评审定稿 | git 干净；专家意见回收 |
| **A. Ch2 打底** | 2026-08 | 注册判别式关系抽取器（`supervised`），金标节点上训练评测 | **causal F1 ~30–37 / subevent ~30**，对比 0.4% |
| **B. Ch2 一致性+修复+风控** | 2026-08 末 | 全局一致解码 + 可追溯修复（violation/cycle/矛盾率）+ CRC 风控准入（分层报告 FNR） | violation 降；关键边风险受控；ECG 可重建率升 |
| **C. Ch1 真机制** | 2026-09 | 检测 + 难例判别 + 不确定性规范化 + 簇级证据/置信聚合；MAVEN-ERE 评测 | 检测 F1 ~60+、coref MUC ~86、相似事件误合并率降 |
| **C2. Ch1 跨文档泛化** | 2026-09 末 | ECB++/CLES 泛化实验（关键，验通用性） | 对比 SECURE/MEET/DIE-EC |
| **D. 真闭环 + 误差传播** | 2026-10 | Ch1→Ch2 建真实图 → Ch3 三图对比（gold/predicted/repaired）+ 受控扰动 | 误差传播曲线；repair 下游增益；「整体更优」实测 |
| **E. 端到端预算（有前提）** | 2026-11 | 三段预算 + reachability + 条件回收，标注假设 | 端到端界 + 分层 FNR；naive vs 预算法对照 |
| **F. 金融应用层** | 2026-12 | CCKS-FinCausal + SARGE 验证金融可迁移 | 金融构建→预测案例；标题承诺兑现 |
| **G. 稳健化** | 2027-01~02 | 多种子 13/17/42；消融补齐；投稿前 pipeline-CP 新颖性扫（读 2404.17769） | 主表 mean±std；新颖性 MEDIUM→HIGH |
| **H. 写作** | 2027-03~04 | 初稿 + 终辩 | 初稿；答辩 |

**关键路径 = A→B→C→D**（Ch2 修复解堵真闭环）。**D 的真实图部分依赖 A/B 成功**，否则退受控扰动。

---

## 9. 当前进度快照（如实）

- ✅ **已完成**：Ch3 SeDGPL 自跑基线；M3a 覆盖保证选择性预测（−43%/−68% 集收缩）；M3b 受控跨阶段
  扫描（`cross_stage.py`，Ch3 转向的现成件）；CRC 准入 + 一致性求解器 + conformal 原语（已测）；ESC
  泄漏发现；CGEP 重建（train+valid，未碰官方 test）；本地 330 passed / 14 skipped、ruff 0 error。
  M1/M2 噪声级、入消融。
- 🔴 **未完成/待建**：Ch2 判别式打底 + 一致性/修复/风控（**首要**，解堵真闭环）；Ch1 真机制（难例判别+
  不确定性规范化）；跨文档泛化；真实图误差传播；金融层；多种子。
- ⚠️ **风险点**：Ch2 causal 召回 0.4%（重构目标）；GPU card 2/3 常被占（优先 card 1）；Ch3 真实误差
  传播依赖 Ch2。

---

## 10. 风险与止损条件

| 风险 | 触发 | 应对（如实，不虚报） |
|---|---|---|
| Ch2 判别式打底仍低召回 | causal 召回 < 10% | 排查（类不平衡/文档级候选/编码）；仍不行则维持受控模拟版，贡献收缩到一致性/修复/风控 |
| Ch2 修复收益小 | repair gain 微弱 | 退 consistency-aware reranking / constrained decoding（综述同款替代） |
| Ch1 evidence 评测难设计 | 无现成金标 | 优先可自动计算指标 + 小规模人工核验；退「canonical table + 难例判别」仍成章 |
| Ch3 真实图不可得 | Ch2 未达标 | 退**受控扰动版**（已实现），仍答「构建误差如何影响下游」 |
| CRC 分层 FNR 不达标 | 某类/某文档超 α | 如实报边际达标、分层分布；讨论交换性/后处理假设边界，不掩盖 |
| 新颖性被先例占 | 2404.17769 等 | 加限定/改主打，不写「首次」 |
| GPU 长期不可用 | card 全被占 | 冻结训练类，转本地（新颖性扫、消融设计、金融数据准备） |
| 金融层稀疏 | 因果图连通性过低 | 降级为「构建可迁移性」验证，MAVEN 主干不受影响 |

---

## 11. 参考文献（评审核实用）

**构建方法论**：Set-Aligning(NAACL24) ｜ CALLMSAE(NAACL25, 2406.18449) ｜ Beyond Pairwise/OmniTemp
(EMNLP25, 2502.11114) ｜ MEET(EMNLP25, 2025.emnlp-main.972) ｜ CDEE/CLES(Findings ACL24) ｜
SECURE(ACL24) ｜ GraphERE(2403.12523) ｜ LogicERE(AAAI25) ｜ GDLLM(Findings EMNLP25) ｜
MAQInstruct(SAC25, 2502.03954) ｜ PLAF(NN26)/SchemaEGC ｜ 图传播联合 ERE(IPM24) ｜ EKG 综述(2112.15280)
**任务/数据**：SeDGPL/CGEP(Findings EMNLP24, 2409.17480) ｜ MAVEN-ERE(2211.07342, THU-KEG，**官方
test 隐藏、CodaLab**) ｜ MAVEN 检测(EMNLP20) ｜ ECB+/GVC/CLES ｜ CCKS-FinCausal(本地) ｜ FinKario(2508.00961)
**可靠性/conformal**：Conformal Risk Control(Angelopoulos 等, 2208.02814) ｜ **CFEP: Conformal Event
Prediction with Temporal Knowledge Graph(Findings ACL26, 2026.findings-acl.258)** ｜ PASC(2605.18812) ｜
SCRC(2512.12844) ｜ Two-stage CRC Ranked Retrieval(2404.17769) ｜ MedCEG(2512.13510)

---

## 12. 请评审重点关注

1. **Ch1 真机制**（难例判别 + 不确定性规范化 + 簇级证据聚合）是否足以支撑独立研究章、硕士级是否适度。
2. **Ch2 收敛**（判别式打底 + 一致性/修复/风控三层）是否清晰、与 LogicERE/Global Zero-shot 是否充分区分。
3. **Ch3 转向构建误差传播** 是否成立、与 CFEP 区分是否干净；受控扰动作为 Ch2 未达标时的退路是否可接受。
4. **CRC 分层表述**（§5.5）与端到端「有前提的预算」（§6）是否严谨。
5. **金融应用层** 规模是否足够兑现课题标题，还是需提升为独立章节。
6. **时间线**（至 2027-04）与关键路径 A→B→C→D（D 依赖 Ch2）是否现实。
