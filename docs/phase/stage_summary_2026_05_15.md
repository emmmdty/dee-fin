# R3 / P5b 阶段总结（交付审阅版）

> 日期：2026-05-15  
> 范围：仅 DuEE-Fin dev500 的 dev-diagnostic 证据；无 hidden-test、final-test、跨数据集、SOTA 主张  
> 目的：把 R3 v5、role-conflict splitter、P5b PlannerGate 集成、CARVE 分配机制诊断的全部本地证据整理成可审阅的形态

---

## 1. 已完成工作摘要

### 1.1 R3 RecordPlanner v5（agglomerative APCC + temperature scaling）

**问题**：v2/v2.1/v3/v4 均未通过 count_mae 验收门。v4 多事件 dev B³ precision 0.358 表明 connected-components 聚类在传递闭包下放大单条假阳性边。

**实现**：
- `carve/p3_planner.py:predict_clusters_agglomerative` — scipy average linkage 替代 connected-components，单条假阳性边不再级联合并
- `carve/p3_planner_only_runner.py:_calibrate_coref_temperature` — dev-NLL 网格搜索拟合标量温度 T（0.05–5.00 步长 0.05）
- `--count-head-mode coref_v5` 路径

**验收结果**（`docs/measurements/r3_planner_only_duee_fin_seed42_v5.md`）：
- 11/14 通过，相对 v4（10/13）提升 1 项
- multi_event_dev count_mae **0.755**（首次通过 0.787 阈值）
- all_dev count_mae 0.376（仍未通过 0.328 阈值，差 0.048）
- multi B³ precision 0.379（仍未通过新增的 #14 检查 0.55）

**2×2 消融**（agglomerative × temperature）：
| Cell | multi count_mae | multi B³ precision |
|---|---:|---:|
| v4 baseline | 0.8173 | 0.3582 |
| temperature only | 0.8077 | 0.3545 |
| clustering only | 0.8077 | 0.3692 |
| **v5 (both)** | **0.7548** | **0.3790** |

→ 两项贡献独立可加，组合在 multi count_mae 上**超加性**（单独各 -0.0096，联合 -0.0625）。

### 1.2 Role-conflict splitter（gold-mention 上限）

**问题**：v5 仍有 3/14 失败，B³ precision 0.379 远低于 0.55。v5 plan 原本包含 splitter 但因 `MentionSpan` 缺 `role_id` 字段被推迟。

**实现**：
- `_compute_role_tags_per_mention` — 用 gold record 反查每个 mention 的角色集合（仅 gold-mention 路径可用）
- `_apply_role_conflict_splitter` — 对每个聚类：若任一角色出现 >1 distinct surface value 则按值分裂；未打角色标签的 mentions 按 affinity 分配到最近 sub-cluster（calibrated logits/T）
- `--apply-role-splitter` 标志

**v5 + affinity-aware splitter 验收结果**（同测量文档 §"Role-Conflict Splitter"）：

| 指标 | v5 alone | **v5 + splitter** | 阈值 |
|---|---:|---:|---:|
| multi_event_dev count_mae | 0.7548 ✅ | **0.2981** ✅ | ≤ 0.787 |
| **all_dev count_mae** | 0.3760 ❌ | **0.3060** ✅ ★ | ≤ 0.328 |
| multi B³ precision | 0.3790 ❌ | 0.4431 ❌ | ≥ 0.55 |
| multi B³ F1 | 0.4920 ❌ | 0.5254 ❌ | ≥ 0.65 |
| 总分 | 11/14 | **12/14** | 14 |

★ = all_dev count_mae 门历史上从未通过过；v5+splitter 是首次通过。

**关键发现**：
- Splitter 主导 count_mae（v4+splitter 与 v5+splitter 在 count_mae 上差 ≤ 0.015）
- Affinity-aware vs naive 在 B³ precision 上 +0.029
- 最优 τ 跌到 0.10，splitter 偏好"先大合并再激进分裂"模式
- B³ precision 0.443 仍未达 0.55 — 表示这是 representation-quality bound，不是 aggregation-mechanism bound

### 1.3 P5b + R3 PlannerGate 集成（Path A）

**问题**：原始 P5b 在 DuEE-Fin dev500 上 CARVE 路径 FP=39,098（baseline FP=8,599，比例 1:4.5）。

**FP 来源诊断**（`docs/diagnostics/p5b_fp_breakdown.md`）：
- **H2 验证**：2,775 个超额记录（90% of pred records）；几乎每个多事件 dev doc 都至少有一个 event type 被超估计数
- H3 否决：hallucination rate 0.1%

**实现**：
- `carve/p5b_runner.py:PlannerGate.predict` — 对 coref/coref_v5 count head 模式返回 `(present, None)`，调用方退化到词法 count fallback（Path A 语义）
- `--planner-checkpoint` / `--planner-encoder-path` / `--planner-feature-mode` / `--planner-presence-threshold`
- `--alloc-checkpoint` — 跳过 allocation 训练，加载已训权重做推理（用于 threshold sweep）

**结果**（`docs/measurements/p5b_duee_fin_dev500_r3typegate.md`）：

| Route | 原始 P5b | + R3 v4 PlannerGate | Δ |
|---|---:|---:|---:|
| baseline F1 | 0.0373 | **0.1109** | +3.0× |
| baseline FP | 8,599 | **1,244** | -86% |
| carve F1 | 0.0109 | **0.0637** | +5.8× |
| carve FP | 39,098 | **1,106** | -97% |

→ R3 v4 TypeGate（presence_threshold=0.27，从 checkpoint 元数据继承 Youden 优化值）拒绝约 85% 的 (doc, event_type) 对，把超额记录从 2,775 砍到 154。

### 1.4 CARVE 分配机制失败的结构性诊断

**Threshold sweep**（`docs/diagnostics/p5b_threshold_sweep_2026_05_15.md`，5 个阈值 0.05/0.10/0.15/0.20/0.27）：
- carve F1 在所有阈值下均低于 baseline F1
- carve TP **结构性地卡在 103**（5 个阈值下完全不变）
- baseline TP 随阈值降低从 192 上升到 199

**根因**（`carve/p5b_runner.py:_predict_route`）：
```python
null_score = row[-1]
best_index = int(torch.argmax(row[:-1]).item()) if record_count else 0
if row[best_index] <= null_score:
    continue  # ← Sinkhorn 的 NULL-column dominance 丢弃 baseline 会保留的候选
```

→ 这是架构属性，不是参数。降阈值无效。

**Allocation-level FP 分解**（`docs/diagnostics/p5b_carve_alloc_breakdown.md`，after PlannerGate）：
| 类别 | FP 占比 |
|---|---:|
| Noise records（未匹配 gold 的 pred 记录） | 79.5% |
| Misallocation（匹配但角色错） | 20.5% |
| Share-gate excess | 0.0% |

→ Share-gate 完全没问题。记录数过估仍是 CARVE 路径上最主要的 FP 源（即使 PlannerGate 已修正大部分 TypeGate 错误）。

### 1.5 工具产出

- `scripts/carve/p5b_fp_diagnostic.py` — per-event TP/FP/FN + 记录数 Δ + hallucination + 角色分布
- `scripts/carve/p5b_carve_alloc_diagnostic.py` — 记录级 FP 分类（noise / misallocation / share excess / candidate miss / pathological gold）
- `carve/p3_planner_only_runner.py:--eval-only` — 跳过训练，加载已训 R3 checkpoint
- `carve/p3_planner_only_runner.py:--override-cluster-method,--override-temperature` — 推理时切换聚类算法和温度（消融用）
- `carve/p3_planner_only_runner.py:--apply-role-splitter` — gold-mention 路径的角色冲突分裂器
- `carve/p5b_runner.py:--alloc-checkpoint` — 跳过 allocation 训练，仅推理

---

## 2. 存在的问题（按严重性）

### 2.1 阻塞性 / 影响 venue 框架

**P-1. CARVE 分配机制在 DuEE-Fin 上**结构性**弱于 baseline**
- carve F1 0.064 vs baseline F1 0.111（差距 -0.047）
- 5 个阈值扫描均失败，TP 卡在 103
- Sinkhorn NULL-column dominance 是架构属性，非超参
- **影响**：venue_decision 已锁定 COLING/SCI Q2 框架，方法论叙述需从"CARVE 分配机制"转向"R3 planner as precondition"
- **是否可救**：可能，但需重构 allocation 训练（recall-weighted loss / 取消 NULL drop / 替换 Sinkhorn 为别的 allocation）—— 工作量超出当前 phase

**P-2. R3 v5 仍有 2/14 检查未通过（v5+splitter 版本）**
- multi B³ precision 0.443 vs 0.55（缺 0.107）
- multi B³ F1 0.525 vs 0.65（缺 0.125）
- pair_auc 0.84（排序好）+ B³ precision 0.44（分区差）的并存表明 representation-quality bound
- 现行 affinity-aware 已是较好的后处理，进一步推动需要 fine-grained 表示学习
- **影响**：v5+splitter 12/14 可以作为论文主表，B³ 失败作为 known limitation 讨论

### 2.2 影响完整性 / 可重复性

**P-3. R3 v5 + splitter 是 gold-mention 上限，realistic 路径未实现**
- 当前 splitter 依赖 gold-record 反查每个 mention 的角色集合
- 真实 pipeline 需要训练 P3 MentionCRF + 角色 tagging
- **影响**：论文应明确把 v5+splitter 定位为 *gold-mention upper bound*，把 v5-without-splitter 作为 *realistic-pipeline contribution*

**P-4. `misallocated_rate_eligible` 未接入 P5b runner**
- `docs/measurements/p5b_decision_table_template.md` 的主验收指标缺失
- 当前只能用 record F1 替代，无法严格按 Strong/Weak/No-support 三档分类
- **影响**：DuEE-Fin 行的判定结论是基于次要指标的"No support"，不是基于主验收门的结论

**P-5. ChFinAnn / DocFEE 数据集未跑**
- 全 P5b 决策表需要 3 个数据集行
- R3 v5 跨数据集鲁棒性未验证
- **影响**：方法论"generalizes across datasets"的主张目前无证据

**P-6. P3 MentionCRF 未训练**
- v4 / v5 当前用 gold-mention 抽取作为 oracle 上限
- 没有 trained mention CRF 意味着 R3 v5 的 acceptance 路径目前不存在
- **影响**：必须在论文中明确标注"gold-mention extraction ablation only"

### 2.3 工程类问题

**P-7. P5b allocation 训练 loss 第 1 epoch 就 plateau 在 0.965**
- 说明模型基本没学到东西，可能是训练数据/loss 设计问题
- **影响**：可解释 carve TP 卡在 103 的部分原因（allocation 没有判别力）

**P-8. P5b candidate generation 召回 0.529**
- 53% 的 gold args 在候选池里
- 47% 的 recall 缺失源头是 candidate generation（不是 allocation）
- **影响**：即使 allocation 完美，P5b recall 也被候选池 capped 在 ~53%

---

## 3. 论文方向建议（基于当前证据）

### 3.1 可辩护的技术 claim

1. **R3 v5（agglomerative + temperature）** 是首个在 DuEE-Fin multi_event_dev 通过 count_mae 验收门的神经计数头
2. **v5 + role-conflict splitter** 是首个在 BOTH 群体通过 count_mae 门的方案（gold-mention 上限，all_dev count_mae 历史上首次通过）
3. **R3 PlannerGate → P5b 集成** 把 CARVE 路径 FP 砍掉 97%，验证了 H2 假设（记录数过估是 P5b FP 爆炸的主要原因）
4. **2×2 消融**显示 agglomerative 和 temperature 各自贡献独立，组合超加性

### 3.2 不可辩护的 claim

- CARVE Sinkhorn allocation + share gate 机制在 DuEE-Fin 上的有效性
- 跨数据集 robustness
- SOTA 或近 SOTA

### 3.3 venue 推荐

详见 `docs/phase/venue_decision_2026_05_15.md`：**COLING / SCI Q2 框架**，方法论叙述从"CARVE 分配机制"转向"R3 planner as precondition + APCC 计数与精化"。

如果要冲 EMNLP Findings：必须先解决 P-1（allocation 机制重做）或 P-5（跨数据集）。两条都是 paper-scope 之外的大工作量。

---

## 4. 复现指南（核心命令）

### R3 v5 训练（gpu-4090）
```bash
scripts/carve/run_r3_planner_only.py \
  --dataset DuEE-Fin-dev500 \
  --data-root data/processed/DuEE-Fin-dev500 \
  --schema data/processed/DuEE-Fin-dev500/schema.json \
  --model-path .../chinese-roberta-wwm-ext_safetensors \
  --run-dir runs/carve/r3_planner_only_duee_fin_seed42_v5_gold \
  --seed 42 --max-epochs 50 --batch-size 64 \
  --encoder-feature-mode evidence_lexical \
  --count-head-mode coref_v5 \
  --mention-source gold --max-mentions 64 \
  --coref-threshold-grid 0.10,0.15,...,0.70
```
约 13 分钟，单 GPU。

### R3 v5 + splitter 评估（eval-only）
```bash
scripts/carve/run_r3_planner_only.py \
  ... \  # 同上
  --eval-only runs/.../v5_gold/checkpoints/r3_planner.pt \
  --override-cluster-method coref_v5 \
  --override-temperature 3.25 \
  --apply-role-splitter
```
约 6 分钟。

### P5b + R3 PlannerGate（Path A）
```bash
scripts/carve/run_p5b_diagnostic.py \
  --dataset DuEE-Fin-dev500 \
  --data-root data/processed/DuEE-Fin-dev500 \
  --schema data/processed/DuEE-Fin-dev500/schema.json \
  --model-path .../chinese-roberta-wwm-ext_safetensors \
  --run-dir runs/carve/p5b_duee_fin_dev500_seed42_r3typegate \
  --planner-checkpoint runs/.../v4_gold/checkpoints/r3_planner.pt \
  --planner-encoder-path .../chinese-roberta-wwm-ext_safetensors \
  --planner-feature-mode evidence_lexical
```
约 30 分钟（约 9 epoch 早停）。

### P5b FP 诊断
```bash
scripts/carve/p5b_fp_diagnostic.py \
  --run-dir runs/.../p5b_duee_fin_dev500_seed42_r3typegate \
  --dev-jsonl data/processed/DuEE-Fin-dev500/dev.jsonl
scripts/carve/p5b_carve_alloc_diagnostic.py \
  --run-dir runs/.../p5b_duee_fin_dev500_seed42_r3typegate \
  --dev-jsonl data/processed/DuEE-Fin-dev500/dev.jsonl
```
本地、不需 GPU，< 1 分钟。

---

## 5. 本次阶段交付的文档列表

| 类型 | 路径 | 内容 |
|---|---|---|
| 测量 | `docs/measurements/r3_planner_only_duee_fin_seed42_v5.md` | R3 v5 主结果 + 2×2 消融 + splitter 结果 |
| 测量 | `docs/measurements/p5b_duee_fin_dev500_r3typegate.md` | P5b + R3 v4 PlannerGate 结果 |
| 诊断 | `docs/diagnostics/p5b_fp_breakdown.md` | 原始 P5b FP 分解（H2 验证） |
| 诊断 | `docs/diagnostics/p5b_fp_breakdown_r3typegate.md` | PlannerGate 后 P5b FP 分解 |
| 诊断 | `docs/diagnostics/p5b_carve_alloc_breakdown.md` | Carve 分配级 FP 分类 |
| 诊断 | `docs/diagnostics/p5b_threshold_sweep_2026_05_15.md` | PlannerGate 阈值扫描 5 个 cell |
| 计划 | `docs/phase/r3_v5_plan.md` | v5 验收合同（14 项） |
| 计划 | `docs/phase/venue_decision_2026_05_15.md` | venue 决策与方法论框架 |
| 计划 | `docs/phase/stage_summary_2026_05_15.md` | 本文档 |
| 代码 | `carve/p3_planner.py` | `predict_clusters_agglomerative` 新增 |
| 代码 | `carve/p3_planner_only_runner.py` | coref_v5 模式 + splitter + eval-only |
| 代码 | `carve/p5b_runner.py` | PlannerGate + alloc-checkpoint + Path A graceful degradation |
| 工具 | `scripts/carve/p5b_fp_diagnostic.py` | FP 来源分解 |
| 工具 | `scripts/carve/p5b_carve_alloc_diagnostic.py` | 分配级 FP 分类 |

---

## 6. 待审专家的具体问题

1. **B³ precision 0.443（multi）能否作为合格结果**接受？还是必须达到原 v5 plan 设定的 0.55？如果必须达到，建议路径是什么（fine-grained 表示学习 vs 更激进的分裂器 vs 调整 acceptance 门）？

2. **splitter 作为 gold-mention upper bound** 的论文定位是否合理？是否需要先训练 P3 MentionCRF + 角色 tagging 给一个 realistic 数据点？

3. **CARVE Sinkhorn allocation 机制失败**：在仅有 DuEE-Fin 一个数据集的负面证据上，能否做"机制不适用"的结论？还是必须跑 ChFinAnn / DocFEE 才能下此判断？

4. **venue 框架的方法论叙述**：从"CARVE 分配机制"转向"R3 planner as precondition"这一定位转换，对审稿人是否站得住脚？还是更宜直接简化到 ECPD-CRV 类的旧框架？

5. **`misallocated_rate_eligible` 接入 P5b runner** 的优先级：先接入再做剩余决策，还是接受当前仅有 record F1 的次要判定？
