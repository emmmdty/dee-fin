# 三章闭环呈现：数据闭环 · 逻辑闭环 · Baseline 对比 · 超越预期

> 目的（作者 2026-06-22 要求）：先把实验跑出结果、能与 baseline 同口径比较、让三章
> **数据闭环 + 论文逻辑闭环**，不求当下超越 baseline，但要有**闭环呈现 + 超越预期**。
> 数字为实测（同口径 time-aware filtered，已实证 filtered≈raw）；公开 benchmark 数见
> `docs/BENCHMARK_SURVEY.md`。

## 一、数据闭环：同一份中文金融数据贯穿三章

```
DuEE-Fin 金融公告
  │  ch1  SARGE 文档级事件抽取（Qwen3-4B+LoRA，schema 接地）
  ▼
事件记录 = 图谱节点   ──实测── 677 事件 / 13 类金融事件 / 204 带时间锚 / 677 带主体 / 证据接地
  │  ch2  关系抽取 + 验证器（grounding/consistency）
  ▼
事件关系图 = 边       ──实测── 20683 边（coreference 145 + temporal 20538）/ 497 条带证据关系边
  │  ch3  时序推理（recurrency→re_gcn/path-RL）+ 风险控制（conformal 覆盖/弃权）
  ▼
下一事件预测          ──实测── n_test 4251 / 155 实体 / 116 时间步 / recurrency MRR(filtered) 0.422（frequency 0.388）
```

**闭环成立**：同一份中文金融公告，经 SARGE 抽取成节点、关系抽取成边、时序推理预测下一事件——
extraction → relations → reasoning 三阶段在**同一数据流**上闭合。脚本一键复现见 §六。

## 二、逻辑闭环：验证器为贯穿主线

三章不是三个孤立任务，而是被**同一个验证器内核**（grounding / consistency / faithfulness）串起来：

| 阶段 | 验证器的作用 | 体现 |
|---|---|---|
| ch1 节点 | grounding：每个论元接地到原文 span | argument_evidence（char offset 回填） |
| ch2 边 | grounding + consistency：边接证据、破环/时序闭合 | 497 条带证据边；greedy 一致性求解 |
| ch3 推理 | faithfulness + 风险控制：反事实消融验证证据链、conformal 覆盖、低忠实度弃权 | 覆盖集 / accuracy@coverage |

**组织原则 = 证据接地/可验证性**：节点有 evidence、边有 evidence、预测有 evidence_chain。验证器三重身份
（推理门控 · 训练奖励 · 风险控制器）贯穿全栈——这是论文逻辑闭环，也是区别于纯准确率方法的护城河。

## 三、Baseline 对比（同口径，公开 benchmark 负责严谨对标）

> 自建中文图负责"闭环演示"（§一），公开 benchmark 负责"对外可比的 SOTA 对标"。两者互补。

**ch1 事件抽取（test，Legacy-FS F1）**
| | Doc2EDAG | GIT | EPAL | SEELE | **SARGE(本文)** |
|---|---|---|---|---|---|
| ChFinAnn | 78.8 | 80.3 | 83.4 | 85.1 | **86.0** |
| DuEE-Fin | 63.4 | 67.8 | 76.4 | 80.8 | **78.0** |
→ ch1 **已具竞争力**（ChFinAnn 超 SEELE；DuEE-Fin 接近、多事件 F1 强）。

**ch2 事件关系抽取（MAVEN-ERE，F1）**
| | ERGO | Joint RoBERTa | MAQInstruct | **本文 GRPO-RLVR** |
|---|---|---|---|---|
| coref(CoNLL) | 90.4 | 92.1 | — | **77**（生成式，设定不同） |
| temporal | 50.4 | 56.0 | 53.8 | precision 56.5 |
→ ch2 当前**低于监督基线**，且生成式 vs pair 分类**设定不同**；优化见 §四。

**ch3 时序推理（ICEWS14，**extrapolation/forecasting**，time-aware filtered MRR；实测 filtered≈raw）**
> 口径铁律：我们做的是**预测未来**(extrapolation)，**不是**补全已知时刻(interpolation/completion)——后者 MRR 可达 0.6-0.7(如 SiSe 0.72)但任务不同、不可混为对标。

| 本文方法 | MRR | | 对标 baseline | 年份 | MRR |
|---|---|---|---|---|---|
| frequency(旧基线) | 0.105 | | RE-GCN(经典参照) | 2021 | 0.420 |
| temporal_gnn(简化) | 0.286 | | TiRGN(经典参照) | 2022 | 0.440 |
| **recurrency(新,CPU)** | **0.356** | | **CENET/RPC/HGLS** | 2023-24 | ~0.42-0.46 |
| **path-RL(本文)** | **0.360** | | **RLGNet/DL-CompGCN** | 2024 | ~0.44-0.46 |
| **re_gcn(强backbone,训练中)** | 待出 | | **近年 SOTA(对标靶)** | 2024-26 | **~0.44-0.47** |

→ 新增 **recurrency**（多信号 copy，CPU 36s）**0.356 追平 path-RL、超过 tgnn**；当前距**近年 SOTA ~0.44-0.47** 约 0.08-0.11。**对标必须用近年(2024-26)发表方法**，RE-GCN/TiRGN 仅作经典参照。追赶路线见 §四。

## 四、超越 baseline 的预期（路线 + 量化目标）

核心策略：**采强 backbone 做有竞争力底座 + 叠加我们的层（验证器/风控/RL）**——不在纯准确率主场用弱方法硬拼。

- **ch3（已动手）**：实现 RE-GCN 级 backbone `re_gcn`（2 层 RGCN + **GRU 跨快照演化** + **ConvTransE**，已写完待 GPU 训练）。现有 temporal_gnn 缺这两件故弱（0.286）。
  - **预期**：`re_gcn` 忠实实现 → filtered MRR **≈0.42 追平 RE-GCN**；再叠加 path-RL re-ranker + faithfulness 塑形 → **预期 0.42→0.43-0.45 小幅超越**，且额外提供 SOTA 没有的**漂移鲁棒覆盖 + 弃权保证**（C1）。
- **ch2**：增加与 baseline **同设定的 pair 分类**评测；用我们的**一致性/传递闭包验证器**治 LLM 传递性缺陷 + CRC 边准入 + GRPO 微调。
  - **预期**：pair 分类底座逼近 RoBERTa(总 51.8)，叠加验证器在 **consistency/精度**子指标上超越，并带可验证性。
- **ch1**：已竞争力（86.0/78.0），保持。
- **跨章 delta（真正贡献）**：在"竞争性准确率"之上，提供**验证器三重身份 + 漂移鲁棒风险控制 + 跨阶段不确定性传播**——这些是 RE-GCN/TiRGN/RoBERTa **都没有**的，且可在**带保证的选择性准确率 / 校准 / 一致性**等可比指标上**超越**。
  - **✅ 已验证（T6, 2026-06-22, ICEWS14 + GNN reasoner）**：静态 split 的覆盖漂移失真 **0.29** vs 自适应 **aci 0.20 / weighted 0.23**；aci 覆盖率 **0.899≈目标 0.9**（split 欠覆盖 0.858）。**验证器即风险控制器的漂移鲁棒性成立**——这是纯准确率 SOTA 没有的能力，构成我们可"超越"的可比维度。结果 `runs/conformal_gnn_icews14.json`。

## 五、当前闭环的诚实边界 + 强化计划

- 闭环目前**偏薄**：自建图上 ch2 用启发式（非 learned GRPO）、ch3 用 frequency（非 path-RL）、规模小（155 实体）、ch3 的 hits@10≈1.0 是小实体空间产物。
- DuEE-Fin 公告独立 → 跨文档关系稀疏（temporal 多为闭包扩展）。
- **强化计划**：① 用 SARGE 主模型 + 在 CMIN/Astock **新闻**上跑推理做密图（公司随时间多事件=真时序，T9）；② 在密图上跑 learned 关系抽取(ch2) + path-RL(ch3) + 风控，使闭环既密又强；③ 公开 benchmark 持续负责 SOTA 对标。

## 六、一键复现（CPU）

```bash
# 三章闭环各阶段指标（中文图）
uv run python - <<'PY'
from collections import Counter
from finekg.core.io import load_event_graph
from finekg.forecasting.data import event_graph_to_dataset
from finekg.forecasting.pipeline import ForecastingPipeline, ForecastingPipelineConfig
g = load_event_graph('data/processed/event_graph_zh/event_graph.json')
nodes=list(g.nodes.values()); edges=list(g.edges.values())
print('ch1 事件', len(nodes), '类型', len({n.event_type for n in nodes}))
print('ch2 边', len(edges))
ds=event_graph_to_dataset(g)
r=ForecastingPipeline(ForecastingPipelineConfig(forecaster='frequency',val_ratio=0.15,test_ratio=0.15)).run(ds)
print('ch3 n_test', r.n_test, 'MRR(filtered)', round(r.metrics['mrr_tfilt'],3))
PY
```
公开 benchmark 对标：`scripts/train_forecaster.py --config configs/forecasting/re_gcn.yaml --path data/processed/icews14/icews14.tsv`（服务器，报 raw+`_tfilt`）。
