> ⚠️ **主线已更新（2026-07-11）**：以 `docs/HANDOFF_2026-07-11.md` 为准。本文的**方法论内核仍有效**
> —— ACI / 加权 CP / CRC / CS-CRP 是算法创新点 **A2**（跨阶段风险传播）的基础，代码在
> `src/finekg/core/calibration/`。但 CS-CRP 从「ch3 独立头条」改为 **SeDGPL 的 M3 head**
> （conformal 预测集 + 弃权，见 HANDOFF §3），金融验证域口径已过时。

# 验证器即风险控制器：分布漂移下的可验证金融事件图谱推理

> Verifier-as-Risk-Controller: Distribution-shift-robust, risk-controlled reasoning
> over financial event graphs.
>
> 本文档是"风险控制"增强部分的权威设计稿，与 `docs/RL_DESIGN.md`（验证器即奖励）
> 并列。代码不含章节标记；章节↔代码映射见 `docs/RESEARCH_MAP.md`，架构机制见
> `docs/ARCHITECTURE.md`。

## 1. 定位与动机

已有系统把验证器（grounding / consistency / faithfulness）用作**推理时门控**与
**训练时奖励**（[[RL_DESIGN]]）。本设计补上第三重身份：

> **门控（gate，已有）· 奖励（reward，已有）· 风险控制器（risk controller，本设计）。**

动机是一个真实算法缺口：`core/calibration` 原仅有 split-conformal，其有效性依赖
**可交换性**。但时序金融预测里可交换性被违反——市场状态/制度漂移，模型 nonconformity
随时间增大——故静态 split-conformal 的实际覆盖率会偏离 1−α。我们迁移成熟统计方法
（ACI、加权/非可交换 CP、Conformal Risk Control）把验证器升级为**有有限样本保证的
风险控制器**，并贯穿"图谱构建（关系）"与"时序推理（预测）"两章，跨阶段传播不确定性。

**创新口径（自圆其说，不 claim 全球首创）**：conformal-on-temporal-graph 已是 2024–25
活跃方向（non-exchangeable CP for temporal GNN, arXiv:2507.02151；NCPNET；KDD'24
Conformalized Link Prediction）。本课题的 delta 不在"把 conformal 用于图"，而在把
**verifier-as-reward 训练 × 漂移鲁棒风险控制推理 × 跨阶段不确定性传播 × 选股下游效用**
统一进金融事件图谱这一集成系统。

## 2. C1：时序自适应 conformal（registry 化）

`core/calibration/` 包，`Registry("conformal_calibrator")`，四个实现，nonconformity
约定为 gold rank（越小越合规，阈值 q → 取 top-⌈q⌉ 为覆盖集）：

| 名称 | 方法 | 关键式 / 不变量 |
| --- | --- | --- |
| `split` | 静态 split-conformal（基线） | q = 第 ⌈(n+1)(1−α)⌉ 小校准分数；漂移下覆盖率失真 |
| `aci` | Adaptive Conformal Inference (Gibbs–Candès 2021) | `α_{t+1}=α_t+γ(α−err_t)`；**状态 α_t 不裁剪**，仅 threshold() 把极端映射 ±inf；保 telescoping 长期覆盖 |
| `weighted` | 近因加权 / 非可交换 CP | 阈值=实现 nonconformity 流的近因加权 (1−α) 分位（半衰期 halflife）；按当前 regime 重分位 |
| `crc` | Conformal Risk Control (Angelopoulos 2022) | 选最小 k 使 `(n/(n+1))·miss_rate(k)+1/(n+1) ≤ α`；比裸分位多有限样本修正 |

**流式接入**（`forecasting/pipeline.py`）：test 查询按时间序处理；每条查询经
`board.context["conformal_rank"]=calibrator.threshold()` 注入 `CalibratorAgent`（agent
只读阈值、在 copy 上标注，**不看 gold**，blackboard 不可变）；服务后揭示 gold，
`calibrator.observe(miscovered, score=gold_rank)` 更新自适应状态。`split` 的 observe 为
no-op，行为与旧版**完全一致**（向后兼容，原 conformal 测试不破）。

**诊断**（`core/calibration/metrics.py`，复用 `core/eval/faithfulness` 的 risk-coverage/
AURC，不重复造）：`rolling_coverage`（滚动覆盖率随时间）、`drift_coverage_gap`（覆盖率
与 1−α 的最坏偏差——头号指标）、`set_size_efficiency`、`accuracy_at_coverage`。

**头号实验**：在 ICEWS18/05-15（更长跨度、更强漂移）上，`split` 的 `coverage_drift_gap`
应明显大于 `aci`/`weighted`。脚本 `scripts/evaluate_calibration.py` 一键对比四者。

## 3. C2：CRC 风险可控的图构建（把保证带进关系章）

`relations/admission.py`，`Registry("edge_admission")`：`crc`（保证）/ `none`（消融）。
把图构建变成有保证的决策——以**单调损失 FNR=1−召回**为控制目标（Angelopoulos CRC 的
FNR 示例）：在校准文档上读每条 gold 边的模型分（`confidence` 或验证器 `faithfulness`，
未提议记 0），选**最紧**（最高）阈值 τ 使 CRC 界下 FNR≤α。推理时 `apply()` 保留 score≥τ
的边（标 `admitted`、产出不可变子图），在保证"≥1−α 真实关系被保留"的同时丢弃低置信边
以提精度。`admission_report` 复用 `relation_prf` 报 precision/recall/FNR。

## 4. C3：跨阶段不确定性传播（两章耦合）

C2 产出的边置信度经 `event_graph_confidence(graph)`（`quad.ref()→置信度`）传入路径游走：
`TemporalPathEnv(edge_confidence=…, min_confidence=…)` 把低于阈值的 fact **从动作空间掩码**
——策略只在关系阶段 admitted 的边上推理。置信度走 side-map 携带，**不改 frozen
`TemporalQuad`**；无 map 时 env 行为完全不变（benchmark TKG 不受影响）。这条耦合把
"风险可控构建"与"时序推理"接成一个整体（`forecasting/forecaster/path_rl.py` 透传）。

## 5. C4：学习式反事实忠实度

`forecasting/rl/rewards.py` 的忠实度 bonus 原用 frequency 代理消融。`build_path_reward_scorer(
…, proxy=…)` 让反事实模型按 forecaster registry 名选择：`frequency` 走精确快路；任意其它
forecaster（如训练中的 `temporal_gnn`）走**通用 predict-based 消融**——消融轨迹经过的 quads、
用该模型重打分、取 `intervention_faithfulness` 的得分降幅，使 bonus 与模型一致。trainer 经
`faithfulness_proxy` 配置选择。

## 6. 下游：事件驱动选股（选择性预测）

`forecasting/downstream/trading.py`：把事件图预测的 top-1 事件类型映射为涨/跌信号
（`directional_signal`，置信度=top-1 自归一化 margin 或路径 faithfulness），**只在最自信的
覆盖比例上下注**，报 `accuracy@coverage`（复用 C1 的 `accuracy_at_coverage`）与 `selective_aurc`。
风险控制论点预测 accuracy 随覆盖率下降而升。数据=CMIN-CN/Astock 规整成 canonical
`stock<TAB>date<TAB>label` TSV。脚本 `scripts/evaluate_downstream_trading.py`。

## 7. 实验矩阵

### forecasting（漂移鲁棒覆盖）
| 系统 | 配置 | 指标 |
| --- | --- | --- |
| split（基线）/ **aci（主）** / weighted / crc | `configs/forecasting/conformal_*.yaml` | conformal_coverage、**coverage_drift_gap**、set_size、MRR/Hits |
数据：ICEWS18 / ICEWS05-15（强漂移）、FinDKG、ICEWS14。

### relations（风险可控构建）
| 系统 | 配置 | 指标 |
| --- | --- | --- |
| **CRC 准入（主）** / −准入（消融） | `configs/relations/crc_edge_admission.yaml` / `ablation_no_edge_admission.yaml` | precision/recall/**FNR≤α**、`relation_prf` |
数据：MAVEN-ERE、CCKS-2021。

### downstream（选股效用）
| 系统 | 配置 | 指标 |
| --- | --- | --- |
| 选择性交易（faithfulness/confidence 门控） | `configs/downstream/trading_selective.yaml` | accuracy、**accuracy@coverage**、selective_aurc |
数据：CMIN-CN、Astock + 自建中文事件图。

## 8. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| ACI 在短序列覆盖波动 | 报长期覆盖 + 滚动窗口；γ∈[0.01,0.1] 调速 |
| 支撑漂移超出校准池 | ACI 状态不裁剪→饱和为 cover-all（仍保覆盖）；weighted 重分位跟踪 |
| CRC 单调性假设 | 损失取 FNR / set-size（单调）；非单调（如 FDR）留待 non-monotone 扩展 |
| 数据 split 合规 | 沿用 [[fin-ekg-datasets-status]]：ICEWS14 勿重切；新数据接 `load_tkg_tsv` / canonical TSV |

## 9. 代码映射

| 模块 | 内容 |
| --- | --- |
| `core/calibration/` | `functional`(split 核)、`base`(Registry+Calibrator)、`split/aci/weighted/crc`、`metrics`(漂移诊断) |
| `forecasting/pipeline.py` | MultiAgent 流式自适应（calibrator 注入 + observe + 漂移指标） |
| `forecasting/agents/calibrator.py` | per-query 阈值经 context 注入；复制标注不变 |
| `relations/admission.py` | `Registry("edge_admission")`：CRC 边准入（FNR≤α）+ 报告 |
| `forecasting/rl/env.py` · `forecaster/path_rl.py` | C3：置信度掩码（admitted-edge-only 游走） |
| `forecasting/data/event_graph.py` | `event_graph_confidence`：C2→C3 置信度桥 |
| `forecasting/rl/rewards.py` · `rl/trainer.py` | C4：`build_path_reward_scorer(proxy=…)` 反事实模型可选 |
| `forecasting/downstream/trading.py` | 选择性选股评测 |
| `scripts/evaluate_calibration.py` · `evaluate_downstream_trading.py` | 头号实验 + 下游 |
| `configs/forecasting/conformal_*.yaml` · `configs/relations/*admission*.yaml` · `configs/downstream/trading_selective.yaml` | 主方法 + 消融面板 |
