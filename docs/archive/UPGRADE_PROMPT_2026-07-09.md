# 方案升级提示词（2026-07-09 复核后）

> 用途：把下面代码块整段粘贴给一个新的 claude code session，启动本轮方案升级。
> 依据：本轮对 `docs/THESIS_DESIGN.md`、上一版 U1–U5 方案、`src/finekg` 与 `runs/` 全部产物、
> 远端 `gpu-4090:/data/TJK/Fin-EKG/runs` 训练日志的一手复核，以及 2026 年新竞品的联网核实。
> 完整诊断见 `~/.claude/plans/claude-code-ticklish-tome.md`。

```
你在 /home/tjk/myProjects/masterProjects/Fin-EKG 这个硕士学位论文课题上工作。三章主线是
事件抽取(SARGE) → 事件关系与图谱构建 → 时序推理与风险监测，贯穿主线是"证据接地/可验证性"，
同一验证器内核以三重身份（推理门控 / 训练奖励 / 风险控制器）串起三章。
顶层设计见 docs/THESIS_DESIGN.md，上一版升级方案 U1–U5 见 ~/.claude/plans/cheerful-watching-prism.md。

先读这两份文档，再读下面这份 2026-07-09 的复核结论——它推翻了若干此前记录的乐观结论，以它为准。

## 已核实的真实现状

远端 gpu-4090:/data/TJK/Fin-EKG/runs 上，4 张 4090 当前全空，最后一次训练是 7 月 3 日的关系 SFT
（adapter 已就绪于 runs/relation_extractor_lora）。ICEWS14 canonical 切分（n_test=7371）上
re_gcn 实测 tfilt MRR 0.384（RE-GCN 原文 0.42），re_gcn_full.yaml 因为用了非 canonical 的
0.15/0.15 切分只有 0.362，hybrid 权重搜索最好 0.4095。

runs/astock_graph_standard.json 是 15623 节点 / 0 条边，metadata 显示
edges_raw=332283 且 edges_dropped_ungrounded=332283——新闻节点缺 argument_evidence 回填，
heuristic 边全部被 require_evidence 丢弃。且闭环 quads 是 (股票, HAS_EVENT, 事件类型, 日期)，
object 空间只有 13 个事件类型，所以 runs/astock_news_recurrency_metrics.json 里 mrr_tfilt=0.699、
hits@10=0.987 是"13 选 1"的平凡数字，不是图推理证据。连带闭环 CS-CRP 的 reachable 全为 true，
跨阶段耦合从未被真正演示。

ch2 pair 口径实测（runs/pair_eval_phase2.json）：coref F1 0.611、temporal F1 0.048（recall 0.025）、
causal F1 0.004、subevent 0.0；窗口上限 0.803 说明瓶颈是生成式全窗枚举。GRPO 相对 SFT 让
causal 从 0.024 跌到 0.004、subevent 塌成 0——复合奖励压制稀有类。pair_encoder 判别底座至今零实现。

## 新竞品（已联网核实，影响创新点定位）

- SKER (Pattern Recognition 2026, doi S0031320326007995)：时序加权 GNN 检索历史证据 + LLM re-rank +
  双标识符约束输出，无需微调即超 SOTA。这就是我们 U1（llm_graph + gated_fusion）的骨架。
- CEHis / Selective TKG Reasoning (arXiv 2404.01695)：TKG 弃权 + 基于历史准确率的置信度估计 +
  risk-coverage。占了 ch3"选择性输出"的位置。
- 两者软肋一致：无有限样本保证、不处理漂移、不跨阶段。
- Non-Exchangeable CRC (ICLR 2024, arXiv 2310.01262) 与 Anytime-Valid CRC (arXiv 2602.04364)
  是现成工具，用来补掉 core/calibration/propagation.py:18-23 那个"构建侧 CRC 假设可交换、
  漂移下漏风"的死穴。

## 已拍板的三条决策

1. 闭环图重建为实体级，对齐 FinDKG：(主体实体, 事件类型作关系, 客体实体, 日期)。
   事件间关系边（ch2 learned 边）叠加为第二层，供 CRC 准入削减下游路径可达性。
2. GPU 修底座与 CPU 重建闭环并行推进。
3. ch3 创新点收缩到形式化保证：CS-CRP 是唯一头条；SKER 与 CEHis 复现为必打 baseline；
   U1 的 gated_fusion 降级为工程组合与消融的一支，不再单列为创新点。

## 你要做的事（按此顺序）

P0-A（CPU，闭环地基）
- scripts/sarge_to_event_nodes.py：仿照已有 _SUBJECT_CUES/_DATE_CUES 新增 _OBJECT_CUES
  （被收购方、标的公司、质押公司、交易对手、增持方…），从论元角色派生 object 实体；
  并补 --source-docs 证据回填（DuEE-Fin 那条链已实现过，见 docs/midterm/project_issues.md 的 P0 条目），
  使 argument_evidence 非空、heuristic 边不再被 require_evidence 全丢。
- src/finekg/forecasting/data/event_graph.py::event_graph_to_dataset 加 quad_mode: "type"|"entity"
  开关，默认 type 保持现有测试不破，entity 模式产出 (subject, event_type, object, t)。
- 更新 configs/forecasting/{frequency,recurrency}_astock_news.yaml，新增 re_gcn_astock_news.yaml。
- 重跑 scripts/export_cross_stage_ranks.py，确认 reachable 不再恒 true。
- 闭环 MRR 会从 0.699 大幅下降。这是正确的，不要试图保住旧数字，如实报告。

P0-B（GPU，立刻排队占卡）
- re_gcn 复现到 ≥0.42（canonical 切分）。诊断清单：官方在测试时把 valid 并入历史快照；
  ICEWS14 官方 history_len=3（我们用 5/7）；epochs=10 偏少（官方 30+ 带 early stop）；
  static graph / self-loop / relation evolution 是否开启。禁止再用 0.15/0.15 切分报数。
- hybrid 重扫权重 → ≥0.43。
- T2 GRPO 从 T1 adapter 热启，奖励加类别重加权，防止 causal/subevent 被压制。
- pair_encoder：新增 relations/extractor/pair_encoder.py + scripts/train_pair_classifier.py，
  MAVEN 用 roberta-large、CCKS 用 chinese-roberta-wwm-ext，输出 RelationEdge(confidence)
  直喂 CRC 与仲裁。验收 MAVEN pair micro F1 ≥46 且 temporal recall ≥0.35。

P1
- core/calibration/propagation.py：构建侧 Clopper-Pearson 上界并列/替换为加权 CRC（非可交换）与
  anytime-valid e-process 上界；新增 allocate_budget_adaptive（双 ACI 耦合、在线预算再分配），
  修 cs_crp_cond 局部窗口 drift_gap 0.9 的塌陷。compare_cross_stage_methods 走 registry 式扩展，
  旧四方法行为不变。docs/RISK_CONTROL_DESIGN.md 补主定理与两段证明（固定量版 + 漂移版）。
- 复现 SKER 与 CEHis 作为 baseline，在 risk-coverage / AURC / 覆盖偏差上必须赢 CEHis。
- docs/THESIS_DESIGN.md §6 差异化表补 SKER / CEHis / CPR(2605.08077) 三行，软化 PASC 表述。
- ch2 两阶段判别后端换成 pair_encoder；subevent 降级为 honest limitation；CCKS 中文因果成主表。

P2
- U5：agents/composition.py（EndToEndPipeline + transcript_provenance 导出）、通用
  leave-one-agent-out 消融驱动、单体 vs 委员会同 token 预算对照。
- 多种子 13/17/42 统一放最后。

## 纪律

- 包与函数名不得出现 ch1/ch2/ch3 章节标记；新组件走 registry 注册；lazy import；
  GPU 组件必须配 CPU 缓存回放。
- uv run pytest 当前 231 测试全绿，只增不改；uv run ruff check src tests scripts 必须过。
- 口径铁律：TKG 同时报 raw 与 time-aware filtered；ICEWS14 用 TiRGN 365 天切分，勿重切；
  GRPO 必须从 SFT 热启（fresh LoRA reward=0）。
- 远端 GPU 坑：tmux 不在非交互 ssh 的 PATH 里（用 nohup + 绝对 uv 路径）；起任务前先
  nvidia-smi 查空闲卡并原子检查，卡会被其他用户抢占。
- 专利申请与论文写作动作不在本次范围；闭环优先使用公开数据集。
- 报告结果时如实呈现，数字下降就说下降，不要粉饰。

先给我一份实施计划，不要直接改代码。
```
