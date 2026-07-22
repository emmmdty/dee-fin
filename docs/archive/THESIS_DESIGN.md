# Fin-EKG 学位论文统一设计稿（三章 + 实验路线）

> **用途**：学位论文三章（事件抽取 → 图谱构建 → 风险监测）的**单一权威设计稿**。
> 每章给出：解决什么问题 / 用什么方法 / 在他人工作上改了什么或原创了什么 / 效果如何 / 与 2024–2026 对标。
> **与其他文档关系**：本文是顶层叙事；细节见 `RESEARCH_MAP.md`(代码↔章节映射)、`CLOSED_LOOP.md`(闭环)、
> `BENCHMARK_SURVEY.md`(baseline)、`RISK_CONTROL_DESIGN.md`+`RL_DESIGN.md`(方法形式化)、`paper/OUTLINE.md`(论文)、
> `GPU_RUNBOOK.md`(实验执行)、`chapter1/`(SARGE 第一章结果)。日期：2026-06-29。

---

## 0. 主线与总策略

- **贯穿主线（论文身份）= 证据接地 / 可验证性**：节点有 `evidence`、边有 `evidence`、预测有 `evidence_chain`。
- **同一验证器内核**（grounding / consistency / faithfulness）以**三重身份**串起三章：**推理门控 · 训练奖励 · 风险控制器**。
- **总策略**：不在 SOTA 主场用弱方法硬拼原始准确率；而是**强 backbone 做有竞争力底座 + 叠加我们的验证器/风控/RL 层**——加 SOTA 没有的保证（漂移鲁棒覆盖 / 忠实度弃权 / 一致性传递），并在特定可比指标上超越。
- **诚实定位**：4B+LoRA 全面超 TKG/ERE SOTA 高风险；可达且经得起答辩的贡献 = 强底座追平/逼近 + 我们的层在校准/一致性/带保证选择性准确率上超越 + 形式化保证。

---

## 1. 第一章 · 中文金融文档级事件抽取（SARGE）

- **问题**：从长篇中文金融公告抽取结构化事件记录（多事件、长距离论元、记录绑定），产出"带证据的事件记录 = 图谱节点"。
- **方法**：SARGE = Schema-Aware Role-Grounded **事件表生成**（Qwen3-4B + LoRA SFT）+ 角色接地输出契约（role-safe contract）+ 可评测输出。Surface Memory / Slot Plan / SACD / LRD 作为**系统组件/诊断**呈现，**不**包装成核心增益。
- **改进 / 原创**：相对 PLM 判别式结构预测（Doc2EDAG/GIT/EPAL/SEELE）——生成式 + schema 接地 + 输出可评测 + 强多事件 F1；利用"金融公告关键论元有稳定表面值"的任务特性。
- **效果**：ChFinAnn **86.0**（>SEELE 85.1，多事件 81.8 领先）/ DuEE-Fin **78.0**（<SEELE 80.8，但多事件强）；SchemaOK 100%、ParseFail/InvType/InvRole 全 0；SFT 为主要增益。
- **诚实边界（保留）**：record binding 未解决，ExactRec Δ_bind ≈ 29.9 / 34.7——作为"可测量现象 + 诊断 + 未来工作"。
- **本章定位（2026-06-29 定）**：**巩固现状 + 诚实边界**，算法原创由 ch2/ch3 承载；与 SARGE CCKS-2026 稿口径一致。
- **2024–26 对标**：DocFEE(Nature Sci Data 2025)、CFinDEE、LLM-DEE 综述(arXiv 2512.19537)、DDEE(2408.05566)；文献证"跨事件实体关联利用不足"=record binding 仍是公开难题（佐证诚实边界）。

## 2. 第二章 · 事件关系抽取 + 图谱构建（relations / edges）

- **问题**：抽取事件关系（coref / temporal / causal / subevent），构建**证据接地、全局一致**的事件图。LLM 已知缺陷（文献明载）：**编造事件、违反传递性、漏长距离、密集事件失效** → recall≈0、micro F1≈7。
- **方法**：LLM 生成式 ERE（Qwen3-4B+LoRA）+ grounding verifier（边接证据/弃权）+ **consistency arbiter**（贪心破环 + 时序闭合的约束推理）+ **GRPO-RLVR**（复合可验证奖励 = format + grounding + consistency + gold F1）+ **CRC 边准入**（FNR≤α 召回保证）。
- **改进 / 原创**：
  1. **一致性 / 传递闭包作为"可验证奖励 + 约束解码"**——直接治 LLM 的传递性/编造缺陷；区别于 MAQInstruct(纯指令/F1)、AutoGraph-R1(检索效用奖励)、EventRL(纯 F1)、Joint Constrained Learning(可微约束但非 LLM-RLVR)。
  2. **CRC 边准入 = 构建阶段召回保证**（CRC 用于图谱构建：FNR≤α 保留 ≥1−α 真实边并提精度）——构建阶段做风险控制，少见。
- **效果（现状 + 优化路线）**：GRPO coref 0.771(vs SFT 0.265)、temporal precision 0.565；生成式设定下绝对 F1 低于监督 RoBERTa(总 51.8)。
  **必做修正**：① 增加**与 baseline 同设定的 pair 分类评测**（否则不可比）；② 强 encoder backbone(RoBERTa/LLM-embedding) pair 分类底座逼近 51.8；③ 叠加一致性/传递验证器 + CRC + GRPO → 在 consistency / 传递违例率 / 精度子指标超越，并带可验证性。
- **2024–26 对标**：MAVEN-ERE SOTA 60.7/37.4/32.9(coref MUC 86.1)；MAQInstruct(2502.03954)、LLMERE(COLING2025)、GREP(ACL2025 findings)、SERE(2605.03701)、WISTERIA(2603.23319)。

## 3. 第三章 · 时序推理 + 风险监测（forecasting / reasoning）——论文与专利之锚

- **问题**：在自建金融事件图上**预测未来事件**(extrapolation)，并给出**可靠、对市场/制度漂移鲁棒**的预测；金融场景中无风险控制的点预测是危险的。
- **方法（守住的集成 delta）**：强 backbone(`re_gcn`: RGCN+GRU+ConvTransE) + recurrency/copy + hybrid 融合(wc≈0.3) + path-RL re-ranker(faithfulness 塑形) + **验证器即风险控制器**：
  - **C1** 流式漂移自适应 conformal（split/ACI/weighted/CRC；ACI 状态不裁剪、仅阈值映射极端值 → 保 telescoping 覆盖）。
  - **C2** CRC 边准入（FNR≤α）。
  - **C3** 跨阶段不确定性传播（边置信度经侧信道掩码路径动作空间）。
  - **C4** 模型一致的反事实忠实度（同名注册反事实模型消融重打分；推理弃权 + 训练奖励）。
  - **选择性输出**：accuracy@coverage / risk-coverage / AURC + 下游选择性选股。
- **★ 本轮新增形式化机制：CS-CRP（见 §4）**。
- **效果（现状 + 目标）**：hybrid 0.411 filtered MRR(≈RE-GCN 0.42)；re_gcn_full 预期 ~0.44+。C1 已实证：split drift_gap 0.29 vs **ACI 0.20**、覆盖 0.899≈0.9。目标：backbone 逼近 0.44–0.47 + 提供 SOTA 没有的漂移鲁棒覆盖 / 弃权 / CS-CRP 组合保证。
- **2024–26 对标**：DynaGen(2512.12669)、RLGNet、TRCL、CognTKE(2412.16557)、DiMNet、HGCT、RECIPE-TKG(2505.17794)；风控侧 NCPNET/non-exch CP for temporal GNN(2507.02151)、graph-MTS CP(2605.04957)、CAP(MLR2026)、SCRC(2512.12844)、General-Risk CRC/SCoRE(2603.24704)；金融侧 MDPI Electronics(事件KG+LLM 未来事件预测，**无风控**)、FinKario(2508.00961)、FinRipple(2505.23826)。

---

## 4. ★ ch3 新增形式化机制：CS-CRP（漂移鲁棒的跨阶段 conformal 风险传播）

- **动机**：现专利 C3 是**启发式掩码**，无形式化保证。把"构建→推理"的不确定性传播升级为**端到端组合保证**，作为本章原创算法点。
- **机制**：把构建阶段 CRC 边准入的**召回保证(FNR≤α_e)** 与推理阶段**漂移自适应覆盖保证(1−α_p)** 组合为单一选择性预测器；显式建模"**准入会削减下游候选支撑**"（被丢边可使 gold 不可达）的**覆盖↔可达性权衡**：把总风险预算 α 在 α_e(构建) + α_p(推理) 间分配，并据可达性损失修正推理阈值，使**组合覆盖 ≥ 1−α 在漂移下仍成立**。
- **与最相近竞品的 delta（写 Related Work 必须点明）**：
  - **PASC(2605.18812, 2026-05)** 做通用 NLP 多阶段 pipeline 联合覆盖，但**假设可交换、同质覆盖型 nonconformity、不处理"上游削减下游支撑"**；
  - **CASCADE(2605.20468)** 是分类→回归的不确定度缩放；
  - 二者均非"**召回保证⊗漂移覆盖保证**"的异质组合，也不在时序事件图；CAP(CP+RL 单阶段弃权)、SCoRE/SCRC(通用 e-value/两阶段) 同样不覆盖。
- **实现**（registry 式，复用现有内核）：
  - 复用 `core/calibration/`(split/aci/weighted/crc) + `relations/admission.py`(CRC) + `forecasting/data/event_graph.py:event_graph_confidence` + `forecasting/rl/env.py:TemporalPathEnv`。
  - 新增 `core/calibration/propagation.py`（组合器插件）+ `scripts/evaluate_cross_stage_risk.py` + configs。
  - 新指标：`composed_coverage`、`composed_drift_gap`。
- **专利**：CS-CRP 超出现交底书 C3 → **公开前补一条权利要求**（跨阶段组合覆盖保证），先递交占优先权日。
- **新颖性备选**（若评审压力大）：以"忠实度作 nonconformity 的漂移自适应选择性预测"替代。

---

## 5. 三章闭环

```
DuEE-Fin / 新闻 → ch1 SARGE 抽取(节点) → ch2 关系+验证器(边) → ch3 时序推理+风控(预测/选择性输出)
            └────────── 同一中文金融数据流；验证器三身份贯穿(grounding/consistency/faithfulness) ──────────┘
```
公开 benchmark 负责"对外可比 SOTA 对标"，自建中文图负责"闭环 + 金融落地"，互补（解决 `BENCHMARK_SURVEY.md §C` 的割裂张力）。实测量级：677 事件/13 类 → 20683 边 → recurrency filtered MRR 0.422（`CLOSED_LOOP.md`）。

---

## 6. 与 2026 关键竞品差异化表

| 竞品 | 它做了什么 | 我们的 delta |
|---|---|---|
| PASC 2605.18812 | NLP 多阶段 pipeline 联合覆盖（可交换、同质） | 漂移非可交换 + 召回⊗覆盖异质组合 + 上游削减下游支撑 + 时序事件图（CS-CRP） |
| CAP (MLR2026) | CP+RL 单阶段 per-instance 弃权 | 跨阶段(C2+C1) + 忠实度作奖励(非仅弃权) + 漂移流式 + 金融图 |
| NCPNET 2507.02151 | temporal GNN 端到端 CP | 不另造 CP；集成进多智能体金融系统 + 跨阶段耦合 + 漂移鲁棒 |
| SCoRE/SCRC 2603.24704/2512.12844 | 通用选择性 CRC / e-value | 图结构反事实忠实度作选择信号 + 跨阶段 + 金融时序 |
| MDPI Electronics 2025 | 事件KG+LLM 未来事件预测 | 形式化风险保证 + 漂移鲁棒 + 验证器统一（其无风控；终稿前核 PDF） |
| Self-Exploring LM 2509.00975 | TKG RL + 可解释 | + conformal 风控 + 忠实度奖励 + 金融域 |
| MAQInstruct 2502.03954 | 指令式统一 ERE | 一致性/传递闭包可验证奖励 + CRC 边准入 |
| DynaGen/RLGNet/TRCL 24-26 | 纯准确率 TKG SOTA | 追平底座 + 加它们没有的覆盖/弃权/组合保证 |

---

## 7. 实验路线 TODO（分阶段；多种子统一放最后）

> 复用现有脚本/配置；新增项标 ★。GPU 任务沿用 `GPU_RUNBOOK.md`(T1 SFT→…→T9)卡分配与坑。
> 口径铁律：TKG 报 raw + time-aware filtered(filtered≈raw)；ICEWS14 用 TiRGN/TLogic 365 天切分**勿重切**；GRPO **先 SFT 热启**再 RL(fresh LoRA reward=0)。

- **Phase 0 迁移与统一(CPU,本轮重心)**：SARGE git subtree → `external/sarge/`；uv workspace/依赖；保冻结契约；闭环溯源补档 `data/raw/event_graph_zh/`+PROVENANCE；pytest+ruff 全绿；文档落地(本文 + RESEARCH_MAP/OUTLINE/交底书+CS-CRP)。
- **Phase 1 Ch2 强化**：★同设定 pair 分类 harness；★强 encoder backbone 底座；一致性/传递验证器 + GRPO-RLVR；CRC 边准入；CCKS-2021 中文因果。
- **Phase 2 Ch3 强 backbone(GPU)**：re_gcn_full → ~0.42+；hybrid 调权 → 0.42–0.44；path-RL re-ranker；GRPO-RLVR 全量；扩 ICEWS18/05-15/FinDKG。
- **Phase 3 风控头条 + 新机制**：C1 漂移全量(ICEWS18/05-15,`evaluate_calibration.py --cache-ranks`)；C2/C3/C4；★CS-CRP(`propagation.py`+`evaluate_cross_stage_risk.py`，对比 naive/ Bonferroni/CS-CRP)；选择性输出 risk-coverage/AURC。
- **Phase 4 闭环加密 + 下游(条件性 stretch)**：SARGE 主模型对 CMIN/Astock 新闻做密图；密图上 learned 关系 + path-RL + 风控；下游选择性选股(弱则降级 future work)。
- **Phase 5 消融(单种子)**：ch2 −grounding/−consistency/−传递/−edge-admission/SFT-vs-GRPO；ch3 −shaping/−faithfulness/−warmstart/random-walk；split vs aci/weighted/crc；CS-CRP vs naive vs Bonferroni。
- **Phase 6 多种子 + 统计(★最后)**：seeds 13/17/42 跑所有头条(ch2 GRPO / ch3 re_gcn_full·hybrid·path-RL / C1 漂移 / CS-CRP / 下游)，报 mean±std。

每阶段验收标准见 plan 文件 `~/.claude/plans/sarge-24-26-todo-structured-tiger.md §五`。

---

## 8. 诚实风险 / 待核

- **CS-CRP 新颖性**：PASC(2605.18812) 已占"通用 pipeline 联合覆盖"；delta 真实但需明确引用 PASC 写清差异；专利先补再公开。
- **MDPI 同胞**："无风控"据搜索摘要，终稿前补读 PDF 核实(`mdpi.com/2079-9292/14/19/3827`)。
- **ch3 准确率**：全面超 SOTA 高风险；诚实可达 = 强底座 + 可比指标超越 + 保证。
- **ch2 设定**：生成式 vs 监督 pair 分类不可苹果对苹果，必须补同设定评测。
- **下游选股**：当前方向性偏弱(≈0.48)，按 future work 如实呈现。
