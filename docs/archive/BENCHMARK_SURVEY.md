# Baseline 调研 & 优化路线（2026-06-20）

> 目的：按作者要求，先调研二、三两章要对比的**近年方案 + 真实表现**，诚实定位我们当前
> 的差距，再据此用 **infra/算法创新**优化，目标在指标上**超过/逼近**这些方案。数字为同口径
> 整理（time-aware filtered for TKG；F1 for ERE），近似值，写论文前需逐一回源核对。

## A. ch3 时序预测 baseline（ICEWS14，time-aware **filtered** MRR）

| 方法 | 年份 | MRR | 核心技术 |
|---|---|---|---|
| RE-GCN | 2021 | ~0.42 | 演化式 GCN + 循环编码 |
| xERTE | 2021 | ~0.41 | 可解释子图注意力 |
| TITer | 2021 | ~0.41 | **路径强化学习**（与我们 path-RL 同族） |
| CEN | 2022 | ~0.42 | 长度感知 CNN 演化 |
| TiRGN | 2022 | ~0.44 | 局部+全局历史循环 |
| CENET | 2023 | ~0.39 | 历史/非历史对比学习 |
| DL-CompGCN | 2024 | **0.455** | 时间感知 GCN + 双解码器 |
| TaReT / MTS-RE-GCN / ACDm / DynaGen / RECIPE-TKG | 2024-26 | ~0.45-0.50 | 拓扑动态 / 扩散 / 多任务 / LLM |

**我们当前**：path-RL **0.360 RAW**、temporal_gnn 0.286 RAW、frequency 0.105 RAW。

⚠️ **口径/设定（已实测澄清，2026-06-20）**：
1. **time-aware filtered ≈ raw**（实测 frequency on icews14：raw mrr 0.1174 → tfilt 0.1177，**+0.2%**）——因为多数 (s,r,t) 只有 1 个客体、可过滤的"同时刻其它真值"极少。**所以 filtered 不会显著抬升我们**：path-RL ≈ **0.36 同口径**。已实现 `mrr_tfilt`（`core/eval/forecasting.py` + `dataset.time_aware_filter_sets`），同时报 raw+filtered。
   - 注：SOTA 的 **RE-GCN 42.0 / TiRGN 44.0 也是单步 time-aware filtered**（RE-GCN 官方 repo Lee-zix/RE-GCN）。故**真实差距 ≈ 0.06–0.08**，真实但**可追**。
2. **path-based 可达性天花板**：path-RL/TITer 只能预测路径可达实体，gold 不可达即 rank=∞，结构性压低 MRR。
3. 我们 temporal_gnn 是**单快照+DistMult 简化版**（0.286 raw），缺 RE-GCN 的 **GRU 跨快照演化 + ConvTransE** → 这就是弱的根因。**已实现 `re_gcn` 强 backbone**（`model/re_gcn.py` + `forecaster/re_gcn.py`，2层RGCN+GRU演化+ConvTransE），待 GPU 训练验证是否达 ~0.42。

**关键结论**：差距 ~0.06 真实但**可达**——我们实现的 RE-GCN 本身就值 0.42。路线 = 用 `re_gcn` 追平 RE-GCN(~0.42) → 叠加 path-RL re-ranker/验证器/风控层取 delta + 加 SOTA 没有的保证。新近强基线还有 RLGNet、History-Repeats(IJCAI'24)、CountTRuCoLa 需纳入对比。

## B. ch2 事件关系抽取 baseline（MAVEN-ERE，F1，**pair 分类**设定）

| 方法 | 年份 | Temporal | Causal | Subevent | Coref | 总 |
|---|---|---|---|---|---|---|
| ERGO | 2022 | 50.4 | 22.5 | 18.9 | 90.4 | 45.6 |
| **Joint RoBERTa**（官方基线） | 2022 | **56.0** | 31.5 | 27.5 | **92.1** | 51.8 |
| MAQInstruct（指令式 LLM） | 2025 | 53.8 | 32.5 | 25.2 | — | — |
| 近期最佳 | 2025 | 60.7 | 37.4 | 32.9 | — | — |

**我们当前**：micro F1 **7**、coref CoNLL 77、temporal precision 56.5（GRPO）。

⚠️ **关键设定问题**：baseline 是**监督式 pair 分类**（给事件对判关系，F1 高）；我们是**生成式抽取**（从文档生成关系 JSON，LLM 易漏/幻觉/不满足传递性 → recall≈0、micro F1≈7）。**两者设定不同、不可苹果对苹果**；且生成式在这个 benchmark 上结构性吃亏（文献亦证 LLM 在 MAVEN-ERE 显著低于监督基线）。

## C. 数据集闭环 vs baseline 支撑（作者关切）

| 用途 | 数据集 | 有 baseline? | 闭环? |
|---|---|---|---|
| ch3 预测 公开评测 | ICEWS14/18/05-15、FinDKG | ✅ 丰富 | ❌（预建 TKG，非我们抽取链） |
| ch2 关系 公开评测 | MAVEN-ERE（英）、CCKS-2021（中金融因果） | ✅ | ❌ |
| **闭环演示** | 自建中文金融事件图（SARGE→节点→关系→推理） | ❌ 无外部 baseline | ✅ 但**很薄**（0 真实关系边、规模小） |

**张力（必须解决）**：**有 baseline 的公开 benchmark 不是闭环；唯一的闭环（自建中文图）没有 baseline 且单薄。** 两者目前是割裂的。

## D. 诚实结论

**当前方法 as-is，在两章公开 benchmark 上都明显低于 SOTA。**我们之前的"GRPO≫SFT""path-RL≫frequency"都是**赢自家弱基线**，不是赢 SOTA。用 4B+LoRA 的生成式/path-RL 直接在原始准确率上压过专用监督/GNN SOTA，**很难**。

## E. 能真正"超过/逼近 baseline"的优化路线

**通用策略：不要用弱方法在 SOTA 的主场硬拼准确率；而是"采用强 backbone 做有竞争力的底座 + 叠加我们的新贡献（验证器/风控/RL）"，让新贡献既加 SOTA 没有的保证、又在特定指标上带来提升。**

**ch3（infra + 算法）**：
1. infra：实现 **time-aware filtered MRR**（立即把我们的数字抬到可比口径）。
2. backbone：在 registry 里复现一个**强演化式 GNN**（RE-GCN/TiRGN 级）作为 `forecasters` 新实现 → 有竞争力底座（目标 ~0.42-0.44 filtered）。
3. 叠加：path-RL 作 re-ranker、验证器/风控作覆盖与弃权层 → 在强底座上**加保证、并改善校准/忠实度**。
4. 目标：filtered MRR 逼近/追平 ~0.44，同时给出 SOTA 没有的漂移鲁棒覆盖（C1）。

**ch2（infra + 算法）**：
1. 设定：增加与 baseline **同设定的 pair 分类**评测（公平对比），或明确声明设定差异。
2. backbone：强编码器（RoBERTa/LLM-embedding）pair 分类底座 → 逼近 RoBERTa 56。
3. 叠加：我们的**一致性/传递闭包验证器**正好治 LLM 的传递性缺陷 + CRC 边准入 + GRPO 微调 → 在底座上**提一致性/精度**。
4. 目标：总 F1 逼近/超过 RoBERTa 基线，并带可验证性/风险保证。

**闭环**：把自建中文图做密（T9：SARGE 对 CMIN/Astock 新闻跑推理，公司随时间多事件=真时序），并在其上用 frequency/GNN/path-RL 三方对比作**内部 baseline**；公开 benchmark 负责"对外可比的 SOTA 对标"，自建图负责"闭环+金融落地"。

## F. 现实判断（不画大饼）

在原始准确率上**全面超过** TKG/ERE SOTA，对 4B+LoRA 是高风险目标。**可达且诚实的贡献** = 强底座（追平/逼近 SOTA）+ 我们的新层（加 SOTA 没有的：漂移鲁棒覆盖、忠实度弃权、一致性保证），并在**特定可比指标**（如一致性/校准/带保证的选择性准确率，或某子类型 F1）上**超过**。这既满足"工作贡献"，又经得起答辩。
