# Fin-EKG TODO / 状态（实时）

> 实时更新「在做/已完成/待办」。每个待办带**预期效果**与**达不到预期怎么办**（止损/人工判断/改计划）。
> 设计见 [`SPEC.md`](SPEC.md)，坑见 [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md)。更新于 **2026-07-13**。

## 🚧 在做（v4 四章重设 · 2026-07-21 批准）

- **主线升 v4 = 4 章「可信事件图谱构建」**（身份/结构/事实/传播，headline=Ch4 下游门控闭环修复；详见
  [`SPEC.md`](SPEC.md) §1 + 批准计划稿）。**CS-CRP / M1–M3 线降为 Ch4 可靠性模块**、证据全保留、不作头条。
- **关键路径 Stage A（即刻起）= Ch2 判别式监督关系抽取器**：`relations/extractor/` 新增
  `@register("supervised")`，金标节点上训练评测，**解 causal 召回 0.4% 瓶颈**（目标 ~30–37 / subevent ~30）。
  这是 Ch3/Ch4 真实图的前提。走 TDD：本地 CPU 结构/契约测试 + 服务器 GPU 训练。
- **数据（2026-07-21 核验）**：MAVEN-Arg ✅ 下载核验、MAVEN-FACT train ✅——三套 doc-id 与 MAVEN-ERE **完全一致**
  （同 4480 文档坐实，`data/raw/DATA_PROVENANCE.md`）；**MAVEN-FACT `valid.jsonl` 待作者手动下载**（Google Drive 断连，
  file id `13hg3fR9Lbof9ffqBCNifVVhLH1TyMjzh`）；ESC/CCKS 已在库。
- **阶段化执行手册就绪**：[`docs/phases/`](phases/)（README + PHASE_A…H **单会话自包含契约**，附如何在新 Claude Code/Codex
  会话跑）。关键路径 **A→B→C→D→E**。新会话跑某 phase：`/clear` 后只读对应 `PHASE_X.md` 执行。
- **文档同步待办**：CLAUDE.md ↔ AGENTS.md 主线描述同步（多为 no-op，指向 SPEC）；归档 v3 设计稿 `docs/superpowers/specs/2026-07-20-*`。

## ✅ 已完成

- **M2 结构感知编码（CPU 实现 + TDD + GPU A/B）已完成 → 噪声级、入消融**（2026-07-17）：EeCE 第四路 =
  每事件 token 的 `reach_anchor` bit（有向可达 anchor＝上游证据），zero-init `nn.Embedding(2,768)` + 门控残差
  `h3=h2+g·struct`（`succession/structure.py` + `model.py`，默认关、byte-identical）。**信号收敛靠证据**：真实
  CGEP-MAVEN valid CPU 预筛——结构解释真实 SeDGPL 难度 5 折 CV R² **≤5%**、reach_anchor 是唯一带出样本信号的
  per-token 特征（加度/proximity 出样本反降）→ 4 维收 1 bit（文献 lineage：HDSE/CAPE/Graphormer 距离结构编码）。
  **GPU A/B（单折 10ep, card 1 ~2.3h, `runs/cgep/maven_sedgpl_m2_structure.json`）**：ON MRR **0.1852 乐观 /
  0.1290 strict**、hits@1/3/10 = 0.120/0.189/0.316 vs OFF（=rerun 0.1867/0.1281）→ **ΔMRR −0.0015 乐观 /
  +0.0009 strict、hits@10 +0.010**，噪声级、无 MRR 增益（同 M1 画像）。formal 价值仍靠 CS-CRP、不依赖 M2。
  ⚠️**初版 bug**：插值门 `g·h2+(1−g)·struct` + 默认 `nn.Embedding`（N(0,1) 范数~28 碾压 h2~8）→ init 扰动事件
  表示 185%、MRR 腰斩 0.088；根因（诊断 `diag_m2.py`：`||h3−h2||/||h2||`=1.85→0）后修为 no-op 起步。本地 330
  passed / 14 skipped，ruff 0 error。
- **M3a CGEP 选择性 conformal 头（CPU 实现 + 测试 + 端到端验证）已完成**（2026-07-12）：`succession/selective.py`
  把预测器候选分数经 `run_cross_stage` 桥成 conformal 预测集，`selective_report` 扫 α 出 risk-coverage 曲线。
  **CPU frequency 实证（真 MAVEN，n_cal=n_test=954，split）**：覆盖 0.996/0.972/0.931/0.818/0.698 @
  α=0.01/.05/.1/.2/.3 **每档达标 1−α**，集大小 441→313 随 α 收缩。5 个 CPU 测试；**预测器无关**→SeDGPL 换
  预测器即得主表曲线。gold ECG `reachable` 全 True；跨阶段 reachability = M3b。CLI `scripts/evaluate_cgep_selective.py`。
- **M3a 主表 risk-coverage 曲线（SeDGPL·GPU）已完成**（2026-07-12，card 1 ~2.24h，`runs/cgep/maven_sedgpl_selective.json`；
  frequency 基线 `maven_frequency_selective.json`）：n_cal=n_test=954。**覆盖保证成立**（aci 每档 ≥1−α：
  0.990/0.951/0.900/0.801/0.703；split 因排名离散在 α=.05/.1 微欠 0.949/0.886）。**头条对比 = 同覆盖下集大小
  SeDGPL vs frequency**：95%覆盖 338vs432、**90% 243vs425(−43%)**、80% 141vs372(−62%)、**70% 99vs313(−68%)**。
  强 ranker 价值 = 覆盖保证下的集收缩，量化且不依赖 MRR → **ch3 头条 formal 贡献有了主表证据**。
- **M3b 受控 cross-stage 扫描（CPU 实现 + 测试）+ ★Ch2 抽取器探针**（2026-07-13）：**探针发现**——训好的 MAVEN
  抽取器（SFT+GRPO LoRA phase2）在 CGEP 要的关系上近乎无能：**causal 召回 0.4%（3/810）、subevent 0%（0/139）**
  （50 文档，`runs/relations/sample50_metrics.json`），coreference 尚可(0.42)。→ 真实构建版 ECG 退化（reachability
  损失 ~1.0），**M3b 真实扫描被 Ch2 堵死**。故落为**受控扫描**：`succession/cross_stage.py`（induce_reachability +
  cross_stage_sweep 复用 `compare_cross_stage_methods`）+ `scripts/evaluate_cgep_cross_stage.py`（frequency CPU /
  SeDGPL ranks-file）。4 CPU 测试。**CPU frequency 实证**（α_total=0.2/target 0.8）：naive 随损失崩（0.818→0.573），
  **cs_crp_cond 最稳**（每档比 naive 高 5-7pp）。
  **★真 SeDGPL 排名版已跑**（`runs/cgep/maven_sedgpl_cross_stage.json`，card 2 ~2.4h，n=954，α_total=0.2）：
  看 coverage+set_size 全貌——**naive「confidently wrong」**（cov 0.80→0.56、集恒 ~140，忽略上游剪枝）；
  **reachability 预算(cs_crp) 守覆盖到预留档**（loss≤0.1 ≥0.80，代价是恒大集 ~270）；**★conditional(cs_cond)=
  同覆盖下自适应更紧集**（loss=0 时 152 vs cs_crp 272 = −44%，随 loss 增到 ~270）——正是原语许诺的"tighter sets
  under same budget"。诚实 caveat：真实离散排名+无漂移下 cs_cond 在 loss=0.15/0.20 覆盖微欠（0.757/0.712，aci
  欠调+CP 保守）；"holds exactly at target"是合成漂移结果(test_propagation)。selective2 曲线复现 M3a（可复现验证）。
- **M1 风险感知线性化（CPU 实现 + 测试）已完成**（2026-07-12）：`succession/linearize.select_nearest_edges`
  按**全图 BFS 距离**选边替换 SeDGPL 存储序切片；registry `edge_selectors`（`sedgpl`/`distance`）+
  `evaluate_cgep.py --edge-selector/--max-edges`，默认 `sedgpl` 关闭保基线逐字节一致。**实测 22.83%
  实例（2309/10116）触发预算**（触发时平均丢 17 条 / 最多 137）→ 发力面实在，非近似 no-op。
- **M1 A/B GPU 评测已完成**（2026-07-12，CGEP-MAVEN 单折 10ep，同 seed 209、仅 selector 不同，card 0/1 并行 ~2.4h）：
  **distance = MRR 0.1889 乐观 / 0.1304 strict、hits@1/3/10 = 0.124/0.189/0.317**；**匹配基线 sedgpl = MRR
  0.1867 / 0.1281、hits 0.125/0.182/0.306**（n=1908）。**Δ = MRR +0.0021 乐观 / +0.0023 strict、hits@10 +0.011、
  hits@1 −0.001**。判读：**持平/略升，非退回**；但 MRR 增量 ≈ 单折噪声（同配置 rerun 0.1867 vs Phase2 0.1836
  差 0.003），hits@3/@10 增量（+0.007/+0.011）超同配置噪声、更可能真。**要多种子(待办#6)才能确认稳定性**。
  产物 `runs/cgep/maven_sedgpl_{m1_distance,rerun}.json`（已回传本地）。
- **Phase 2** CGEP-MAVEN 自跑 SeDGPL 基线**已完成**（2026-07-11）：**MRR 0.1836（乐观）/ 0.1265（strict）**，hits@1/3/10 = 0.120 / 0.182 / 0.308（单折，`runs/cgep/maven_sedgpl.json`）= ch3 主表「自跑 SeDGPL」基线行。
- **Phase 0** CGEP-MAVEN 重建（`succession/data/cgep.py`；2994 文档 / 8.82 节点·ECG / 13.21 边·ECG，验收 PASS）。
- **Phase 1** SeDGPL 复现锚 + ★发现 **ESC 19.6 依赖切分泄漏**（topic-CV 0.0599 vs doc-split 0.1802）。
- **新颖性复核 A1 + CS-CRP**：结论见 SPEC §5（RL-奖励非新→降级；CS-CRP 具体组合有窄 delta→ch3 头条；撞名 SCRC）。
- **专利归档**、**文档体系重构**（本轮：README/CLAUDE/AGENTS/SPEC/TODO/ENGINEERING_NOTES + 旧文档归档）。
- 通用原语：`core/calibration`（split/aci/weighted/crc/propagation）+ `relations/admission`（CRC 边准入）已实现并测试。
- 本地 **323 passed / 7 skipped**，ruff 0 error（M1 前实测 308/6；文档旧记「313/6」是 M1 前已存在的漂移，非本轮引入）。

## 📋 待办（含预期与止损）

| # | 任务 | 预期效果 | 达不到预期怎么办 |
|---|---|---|---|
| 1 | Phase 2 完成后读基线数、入主表 | 得到可比基线 MRR（topic 协议 + strict/乐观双 tie-break） | 异常→按「在做」止损排查；确认后如实记录，**不换指标绕过** |
| 2 | **M1** 风险感知线性化（距离选边）✅CPU+测试+GPU A/B | MRR 持平/提升 + 预算旋钮 → **达标**（+0.002 乐观/持平略升、hits@10 +0.011、旋钮就位） | 达标非退回；MRR 增量 noise-level，多种子(#6)确认，不硬 tune |
| 3 | **M2** 结构感知编码（EeCE 第四路=reach_anchor bit）✅CPU+TDD+GPU A/B | 结构信号带增益 | **无增益达标**：ΔMRR ~噪声（−0.0015 乐观/+0.0009 strict）→ 已如实入消融（CPU 证据 R²≤5% 佐证），不硬 tune |
| 4 | **M3a** 选择性头 ✅（CPU+测试+端到端实证）／**M3b** 跨阶段待办 | risk-coverage 曲线 + 覆盖保证 → **M3a 达标**（CPU 覆盖每档达标）；SeDGPL 主表曲线待 GPU | formal 贡献成立、不依赖 MRR → 头条卖点稳 |
| 5 | **M3b 受控 reachability 扫描**（真实构建 ECG 被 Ch2 堵，抽取器 causal 0.4%）✅CPU+测试／SeDGPL 排名版 GPU 中 | naive 覆盖崩 vs CS-CRP(conditional) 稳在 1−α_total | 真实构建=future work；抽取器 0.4% 作诚实数据点报出 |
| 6 | 多种子 13/17/42（★最后） | mean±std 稳健 | — |
| 7 | CS-CRP 改名（撞名 SCRC 2512.12844） | 消除 reviewer 混淆 | **需人工判断**命名，涉 docs/代码多处统一改 |
| 8 | 投稿前穷尽 pipeline-CP 新颖性扫 | 把 CS-CRP 未占从 MEDIUM 提到 HIGH | 若发现先例 → 加限定或改主打 |

## 📋 v4 待办（新脊柱 · 详见计划稿执行路线 A→I）

> 上表 #1–8 属 CS-CRP 时代,证据保留、机制降为 Ch4 可靠性模块。下为 v4 四章新增待办,关键路径 = A→B→C→D→E。

| 阶段 | 任务 | 预期效果 | 达不到怎么办 |
|---|---|---|---|
| **A** | Ch2 判别式 `supervised` 抽取器(金标节点) | causal F1 ~30–37 / subevent ~30（对比 0.4%） | 排查类不平衡/文档级候选/编码;仍不行维持受控模拟,贡献收缩到一致性/修复/风控 |
| **B** | Ch2 一致性+CRC(已建)+ **repair trace**(新) | violation 降;关键边风险受控;ECG 可重建率升 | 修复收益小→退 consistency-aware reranking |
| **C** | Ch1 检测复用+难例判别+不确定性规范化+论元(MAVEN-Arg) | 检测 F1 ~60+;coref MUC ~86;误合并率降 | evidence 评测难→优先可自动指标+小规模人工核验 |
| **D** | Ch3 事实性检测(MAVEN-FACT)+预测图鲁棒性+图净化 | macro-F1 ≥47.6;净化改善下游 | 净化收益小→退「检测+鲁棒性分析」仍成章 |
| **E** | Ch4 下游门控闭环控制器 + 三类真实扰动 + 三图对比 | **repaired>predicted**;误差传播曲线;下游增益 | 闭环收益微弱→退一致性重排+误差传播分析,不硬撑 |
| **F–I** | 端到端预算 / 金融层(CCKS+SARGE) / 多种子 13·17·42 / 写作 | 见计划稿 | — |

## 🚦 终止 / 人工判断 / 改计划条件

- **自跑 SeDGPL 基线远低于 random 且查不出 bug** → 停下，人工判断数据/口径（可能 MAVEN 重建口径需复议）。
- **M1/M2/M3 全部无 MRR 增益** → 收缩到只报 **CS-CRP 选择性/覆盖保证**（formal contribution，不依赖精度提升）。
- **CS-CRP 具体组合被发现有先例** → 改主打或加限定；不得对外写「首次」（见 SPEC §5）。
- **GPU 长期不可用 / 被抢** → 冻结训练类待办，转本地可做项（新颖性扫、消融设计、文档）。
