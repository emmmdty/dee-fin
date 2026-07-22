# Fin-EKG 项目总纲（SPEC）

> **单一权威的开发驱动文档。** 讲清「做什么、怎么组织、当前机制、定位约束、实验设计」。
> 动态内容分流：**当前在做/待办/止损条件** 见 [`TODO.md`](TODO.md)；**工程坑** 见
> [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md)；**服务器运维** 见 [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md)；
> **数据与切分** 见 [`DATASETS.md`](DATASETS.md)。历史设计/交接稿已归档 `docs/archive/`（勿作依据）。

## 1. 主线与任务（v4 · 2026-07-21 重设）

课题主线 = **可信事件图谱的自动化构建与下游可靠应用**。学位论文 **4 章脊柱 = 可信度四维**
（身份 → 结构 → 事实 → 传播/下游），代码按功能域组织、**名字不含 `ch1/ch2/ch3`**（章节↔代码映射见 §3）。
> 全文完整设计（问题 / 贡献 / 防审稿创新点 / 路线）见批准计划稿
> `~/.claude/plans/agent-agent-llm-4-4090-1-2-llm-2026-07-wiggly-platypus.md`。**取代旧 3 章脊柱。**

| 章 | 可信维度 | 任务 | 主数据（公开 gold） | 代码域 |
|---|---|---|---|---|
| Ch1 | 身份可信 | 证据+不确定性的规范事件节点（带论元） | MAVEN 检测 + MAVEN-Arg + ERE-coref | `finekg.core`(schema/eval) + 新建规范化 |
| Ch2 | 结构可信 | 风险受控全局一致多关系边 + 可追溯修复 | MAVEN-ERE | `finekg.relations` |
| Ch3 | 事实可信 | **构建图上**事实性检测 + 事实性驱动图净化 | MAVEN-FACT | 新建 factuality + `core/calibration` |
| **Ch4** | **传播可信/可用（headline）** | **下游门控闭环修复 + 构建误差传播 + 可靠后继预测** | CGEP-MAVEN / ESC，基座 SeDGPL | `finekg.succession` + `agents` |

**全篇统一创新（headline = Ch4）** = **面向下游的构建误差预算 + 下游验证的闭环修复**：每阶段量化校准
不确定性 → union bound + 可达性合成端到端误差预算（`core/calibration/propagation.py`）→ 修复控制器
**仅在下游后继预测改善时接受编辑**。**CS-CRP / conformal（§4.3）由头条降级为 Ch4 的可靠性模块。**

贯穿属性 = **evidence-grounded / verifiable**：节点挂 evidence provenance、边带 repair trace、预测带
`evidence_chain`。验证器跨阶段「货币」三身份：**门控 · 奖励 · 风险控制器（带有限样本保证）**。

> ⚠️ 旧「实体中心中文金融事件图谱 + TKG 外推」（re_gcn/path_rl/hybrid）**已从主干移除**（见 git tag
> `frozen-tkg-line`，消融时从 tag 取）；Astock 与 entity-mode 是已证死路。**金融作应用验证层**
> （CCKS-FinCausal + SARGE），非整章。
> ⚠️ 数据看**公开可下载性**：MAVEN 四件套（检测/Arg/ERE/FACT）全 THU-KEG 公开 gold、同 4480 文档、
> test 隐藏走 CodaLab；**MAVEN-Arg + MAVEN-FACT 需下载**（本地暂只有 MAVEN-ERE + 重建 CGEP + ESC + CCKS）。

## 2. 架构不变量（升级=加实现，不返工）

三条机制保证「把某阶段从 baseline 升到 full method 只是**加一个实现**」：

1. **冻结的跨阶段契约**（`finekg.core.schema`）：唯一跨阶段类型
   `EventNode → RelationEdge / EventGraph → TemporalQuad / ForecastQuery → Prediction`。
   规则：**只加可选字段，绝不复用/改义既有字段**。`EventNode` schema 零新增字段（扩展走
   `metadata`）；`CgepNode` 是 succession 自己的类型、可加字段。
2. **插件式 registry**（`finekg.core.registry`）：可换组件各自注册，config 按名选择。加方法 =
   `@registry.register("name")` + 写实现，pipeline/config schema/调用方零改动。
3. **CPU/GPU 惰性分层**：`core` + 启发式 baseline 无 torch，可在纯 CPU 机导入整包；神经代码
   （LLM/PyG/SeDGPL）**lazy import torch**，只有实例化才需 `llm`/`gnn` extra。故本地
   `uv run pytest` / `finekg-smoke` 全绿，GPU 路径在服务器跑。GPU 组件必须配 CPU 缓存回放。

**不可违反的纪律**：包/函数名不得含 `ch1/2/3`；新组件走 registry + lazy import；报告结果如实
（数字降就说降；观察失败如 ssh/工具，不得伪装成被观察对象的结论）。不可改的测试锁见
`tests/core/test_propagation.py`。

## 3. 代码地图（chapters ↔ code）

```
src/finekg/
├── core/          冻结契约: schema, io(+SARGE adapter), graph, registry, config,
│                  calibration/(split·aci·weighted·crc·propagation), eval/(faithfulness·指标)
├── succession/    ★Ch4 CGEP 后继事件预测(SeDGPL 基座 + selective/structure/cross_stage)
│   ├── data/cgep.py     从 MAVEN-ERE 重建 CGEP 实例(ECG 抽取/anchor 选取/候选采样)
│   ├── data/esc.py      官方 ESCSubWoRe.npy 白名单 Unpickler + topic 交叉验证切分
│   ├── linearize.py     DsGL 图线性化 + EventVocabulary(<a_i> token)
│   ├── metrics.py       MRR/Hit@k, 乐观(SeDGPL)+strict 两套 tie-break
│   ├── predictor.py     SuccessorPredictor ABC + registry + random/frequency + UnscorableInstance
│   ├── model.py         EeCE 两级门控 + ScEP 对比头(torch 守卫, CPU 可导入)
│   ├── encode.py        批编码(保「第 i 个事件 token ↔ 第 i 个句子」不变量)
│   └── sedgpl.py        SeDGPLPredictor: linearize→encode→model, 纳入统一 evaluate
├── relations/     Ch2 关系抽取+图构建: data/·extractor/(heuristic·llm)·grounding/·
│                  consistency/·admission.py(CRC 边准入)·rl/(GRPO-RLVR)·agents/·pipeline
├── agents/        多智能体基底: Agent/Blackboard/Stage/Orchestrator/Verifier(阶段无关)
└── rl/            RL 基底(阶段无关): 组合奖励·组相对优势·势塑形·课程
scripts/           功能命名 CLI: build_cgep·evaluate_cgep·profile_cgep_step / evaluate_* / train_*
configs/relations/   YAML 实验配置
tests/{core,succession,relations,agents,rl,scripts}/   单测 + CPU 冒烟
```

## 4. 当前机制（Ch3 CGEP）

### 4.1 基座 SeDGPL（自跑基线）
监督式 prompt learning，三组件：**DsGL**（距离敏感图线性化，按存储顺序取前 20 条边，最短路距离
只用于排序幸存边）+ **EeCE**（事件富集因果编码，两级门控）+ **ScEP**（语义对比事件预测头）。
CGEP 只消费 (事件类型, 触发词, 所在句子)，**不用论元**。词表 **transductive**（覆盖 train+test 的
`<a_i>` token 清单，与 SeDGPL 的 `to_add.json` 一致，只 token 清单跨切分、无标签/图/梯度泄漏）。

### 4.2 我们的三个机制（加在已完成的自跑 SeDGPL 基线之上；状态见 TODO.md）
**M1 已接入 `succession/` 预测器（CPU 实现 + 测试，GPU 评测待跑）；M2/M3 仍留位。** 代码：
- **M1 风险感知线性化**（✅ CPU + 测试；GPU 待跑）：`linearize.select_nearest_edges` 按**全图 BFS
  距离**保留离 query 最近的 budget 条边，替换 SeDGPL「按存储序取前 20」的任意切片。registry
  `edge_selectors`（`sedgpl` 默认 / `distance`）+ `evaluate_cgep.py --edge-selector` flag，**默认关闭**
  保基线逐字节一致（测试锁全绿）。**发力面实测 = 22.83% 实例触发预算**（触发时平均丢 17 条）。
  定位：gold ECG 无 confidence → 主表用**距离（structure-aware）**；admission 打分留给构建版 ECG / CS-CRP。
- **M2 结构感知编码**（✅ CPU + 测试 + GPU A/B；**噪声级、入消融**）：EeCE 加第四路 = 每事件 token 的
  **`reach_anchor` bit**（是否经有向因果边可达 anchor＝是否为预测的上游证据），经 **zero-init `nn.Embedding(2,768)`
  + 门控残差** `h3=h2+g·struct` 融入（`succession/structure.py` + `model.py`）。默认关、baseline 逐字节一致。
  **信号收敛依据**（真实数据 CPU 预筛）：结构解释真实 SeDGPL 难度的 5 折 CV R² **≤5%**、`reach_anchor` 是**唯一**带
  出样本信号的 per-token 特征（加度/proximity 出样本 R² 反降）→ 从 4 维收为 1 bit。**GPU A/B（单折 10ep）**：
  ON MRR 0.1852/0.1290 vs OFF 0.1867/0.1281 = **持平**（ΔMRR −0.0015 乐观 / +0.0009 strict、hits@10 +0.010）→
  与 M1 同类，噪声级、如实入消融附录。⚠️初版「插值门 + 默认 `nn.Embedding`」曾 MRR 腰斩（0.088）：默认 N(0,1)
  embedding 范数 ~28 碾压 `h2` ~8，init 扰动事件表示 185%、lr=1e-6 救不回；根因修复＝no-op 起步（诊断见 ENGINEERING_NOTES）。
- **M3 = CS-CRP 选择性头**（✅ M3a CPU 实现 + 测试 + 端到端验证；M3b 待办）：`succession/selective.py`
  把预测器候选分数经 `core/calibration` 桥成 conformal 预测集，产 **risk-coverage 曲线 + 覆盖保证**（gold
  ECG `reachable` 全 True → 退化为推理侧选择性预测器）。跨阶段 reachability 预算（§4.3 头号差异点）= M3b，
  随构建版 ECG 杀手实验落地。机制见 §4.3。

### 4.3 CS-CRP（跨阶段漂移鲁棒 conformal 风险传播）★Ch3 主打机制
代码 `core/calibration/propagation.py`。把两个**异质**保证在单一预算 α_total=α_e+α_p 下组合成
端到端选择性预测器：
1. **构建阶段**：CRC 边准入，**召回**保证 FNR≤α_e（`relations/admission.py`，Angelopoulos CRC）。
2. **推理阶段**：**漂移自适应覆盖**保证 miss≤α_p（`core/calibration` 的 ACI/weighted 流式校准器）。
3. **关键**：边准入**移除候选** → 丢金标边使答案**不可达**（推理校准器看不到的 miss）。为此单列
   reachability 预算，union bound：P(miss) ≤ P(unreachable)+P(reason miss|reachable) ≤ α_e+α_p。
4. **条件回收**（`allocate_budget_conditional`）：用 held-out 准入结果证不可达率上界 u
   （Clopper-Pearson，CRC 界收紧），推理侧跑修正水平 **α_p'=(α_total−u)/(1−u)**，收紧预测集。
5. 推理侧**非可交换**（漂移自适应），区别于可交换 pipeline-CP。

**实现状态**：通用原语已实现并测试（`core/calibration` 的 aci/weighted/crc/propagation + `relations/admission`
的 CRC 边准入；`propagation.py::compare_cross_stage_methods` 是其头号实验）。**M3a（推理侧选择性头）已接入
CGEP**（`succession/selective.py`：候选分数→gold 排名→`run_cross_stage`，产 risk-coverage 曲线 + 覆盖保证）。
**SeDGPL 主表曲线已跑**（GPU，954/954）：覆盖保证成立（aci 每档 ≥1−α），**同覆盖下集大小 SeDGPL≪frequency**
（90%覆盖 243 vs 425/−43%、70% 99 vs 313/−68%）——强 ranker 价值=覆盖保证下的集收缩、不依赖 MRR。
**M3b（跨阶段 reachability）：真实构建版 ECG 被 Ch2 堵死**——训好的 MAVEN 抽取器（SFT+GRPO LoRA）causal 召回
**0.4%（3/810）、subevent 0%（0/139）**，构建版 ECG 退化（reachability 损失 ~1.0，非可扫描范围）。故 M3b 落为
**受控 reachability 扫描**（`succession/cross_stage.py` + `scripts/evaluate_cgep_cross_stage.py`）：真 SeDGPL 推理排名 +
受控 reachability 损失。**真 SeDGPL 排名实证**（954，α_total=0.2）：naive 覆盖崩(0.80→0.56)、集恒~140（忽略剪枝）；
cs_crp 守覆盖到预留档(loss≤0.1)、集恒~270；**cs_cond 同覆盖下自适应更紧集**(loss=0: 152 vs 272，−44%)——原语的
"tighter sets"兑现。离散排名+无漂移下 cs_cond 覆盖在 loss=0.15/0.20 微欠 target。抽取器 0.4% 作诚实数据点；
强抽取器诱导真实损失 = future work。**这是继 M1(MRR 噪声级) 后第二个经验墙 → 价值靠方法讲干净、非端到端数字。**

### 4.4 验证器即奖励（RL-reward，**降级为机制之一/消融**）
`finekg/rl` + `relations/rl`（GRPO-RLVR：format+grounding+consistency+F1）。path RL（旧 TKG 线）已随
主干移除（tag `frozen-tkg-line`）。**定位收缩**：新颖性复核表明「结构作 RLVR 奖励」是红海（见 §5），故 RL-reward
不作头条卖点，仅作机制/消融。历史全设计见归档 `docs/archive/RL_DESIGN.md`。

## 5. 新颖性定位约束（硬约束，投稿安全）

复核证据见归档 `docs/archive/NOVELTY_A1_2026-07-11.md` / `NOVELTY_CSCRP_2026-07-11.md`。

> **v4 重定位（2026-07-21，权威 = 批准计划稿「创新点与新颖性（防审稿）」节）**：全篇创新是**组合式 +
> 系统集成 + 窄 delta**，headline = **Ch4 下游门控闭环修复 + 构建误差预算**（非单个颠覆算法）。逐章防审稿
> = 最近邻 / 精确 delta / 质疑→反驳：Ch1 下游可消费校准置信 + 难例判别；Ch2 风险目标是**下游可达性损失**
> 而非泛化 FNR（**改名避 SCRC 2512.12844**）；Ch3 novelty = **预测图鲁棒性 + 净化下游**（不主张「用结构检测」新，
> MAVEN-FACT 已证）；Ch4 = **下游目标门控接受**（治 self-refine 掉点）+ 三图误差传播（区分 CFEP/self-healing KG）。
> 权威缺口引用用**事件领域文献**（EKG 综述 2112.15280 / EE 综述 2512.19537 / MAVEN-FACT），不靠通用 LLM-KG 综述。
> 下列 A1/CS-CRP 结论仍成立（作 Ch4 可靠性模块的护栏）：

- **A1（RLVR 奖励）**：❌ **不得写「首次把结构/事理当 RLVR 奖励」**。MedCEG(2512.13510) 直接先例
  （gold 推理图路径作可验证奖励），另 Structure-R1/GraphThinker/K2V。VeriGate(2605.30451)/
  Faithful GRPO(2604.08476) 已排除不抢先。注：2409.17480=SeDGPL 本体（监督、无 RL），是 base 非竞品。
- **CS-CRP**：一般命题「cross-stage/selective conformal（甚至 under drift）」也非新（C-RAG 2402.03181/
  PASC 2605.18812/SCRC 2512.12844/CASCADE 2605.20468）；但**具体组合**（召回⊗漂移自适应覆盖 +
  上游剪枝致不可达的 reachability 预算 + 条件回收）**未见先例**——**比 RL 线干净**，作 Ch3 头条。
  相关工作须逐条区分上述四篇；**reachability 预算**是最硬差异点。
- ⚠️ **CS-CRP 与 SCRC(2512.12844) 撞名**，建议改名（突出 reachability-budgeted/recall-coverage 组合），
  待定；改名涉及 docs/代码多处，须统一。
- 置信度：结构作 RLVR 奖励非新=HIGH；CS-CRP 具体组合未占=MEDIUM（投稿前须做一次穷尽 pipeline-CP 扫）。
- 口径：不 claim 全球首创；一律「据我们所知」+显式区分先例。**专利已归档、不再安排专利写作。**

## 6. 数据与切分（要点，详见 DATASETS.md / ENGINEERING_NOTES.md）

- **CGEP-MAVEN 重建口径**（`succession/data/cgep.py` 权威）：ECG = 文档内 **causal(CAUSE+PRECONDITION)
  +subevent** 无向连通分量、**节点≥4**；时序 BEFORE **不进拓扑**（只作 M2 结构特征）；查询边 =
  **出度 0 且入度 1**；候选 512、同 split 均匀采样。验收：2994 文档 / 8.82 节点·ECG / 13.21 边·ECG。
- **ESC 必须 topic 交叉验证**（EventStoryLine topic）；**文档级切分会泄漏**（同 topic 同一故事）。
  实测：SeDGPL 公开的 **ESC 19.6 依赖切分泄漏**（topic-CV 0.0599 vs doc-split 0.1802≈复现 0.196）。
- **MAVEN 版数据未发布**（SeDGPL 只发 ESCSubWoRe.npy）→ 论文 CGEP-MAVEN 27.9 **不可比**；主表**必须以
  我们自跑的 SeDGPL 为基线**，公开数字标「原论文数据构建，非同数据可比」。
- ICEWS14 已验证=正版 TiRGN，**勿重切**；ICEWS 必须用 timestamps 计数切分。

## 7. 实验设计

- **主表基线**：自跑 SeDGPL 在 CGEP-MAVEN（Phase 2，单折 10ep ≈ 2.5h）。M1/M2/M3 都挂它上。
- **协议**：ESC 报 **topic 交叉验证**（`--split-mode document` 只作「论文数字来源解释」，绝不当协议）；
  tie-break 同报 `mrr`（乐观/SeDGPL）与 `mrr_strict`。
- **杀手实验（受控版）**：真实构建版 ECG 被 Ch2 堵死（抽取器 causal 召回 0.4%）→ 落为**受控 reachability 扫描**
  （真 SeDGPL 排名 + 受控损失，`cross_stage.py`）：naive 覆盖崩 vs CS-CRP(conditional) 稳。真实构建=future work。
- **旧 TKG 线**：re_gcn/hybrid/path_rl **已移出主干**（git tag `frozen-tkg-line`），不在当前测试/CI。
- **多种子最后**：seeds 13/17/42，报 mean±std。

## 8. 复现（本地 CPU 冒烟；GPU 训练在服务器）

```bash
uv sync --extra dev && uv run pytest && uv run ruff check src tests scripts   # 本地：契约/评测/冒烟全绿
uv run python scripts/build_cgep.py --split train+valid --report-stats         # CGEP 重建验收
# 服务器 CGEP 训练(screen/nohup + 绝对 uv 路径, 见 GPU_RUNBOOK/AGENTS.md):
#   CUDA_VISIBLE_DEVICES=<空卡> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   uv run --extra llm python -u scripts/evaluate_cgep.py --dataset maven --predictor sedgpl \
#     --model-path <roberta-base> --epochs 10 --output runs/cgep/maven_sedgpl.json
```
