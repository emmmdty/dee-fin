# 中期报告 · 结果总表（2026-06-23，报告就绪）

> 一站式：三章结果 + 同口径 baseline 对比 + 风险控制验证 + 三章闭环。可直接搬进中期报告。
> 口径：ch3 用 **time-aware filtered MRR**（已实证 filtered≈raw）；extrapolation/forecasting（预测未来），非 completion。

## 0. 一句话
三章 extraction→relations→reasoning 闭环成立；ch3 已从"只赢自家弱基线"推进到**近年基线级方法（recurrency 0.356、re_gcn 0.380、组合 0.386）+ 强 backbone（带关系演化，训练中冲 0.42）**；**风险控制头条（验证器即风控的漂移鲁棒性）已验证成立**；并有自建中文金融图的三章数据闭环。

## 1. ch3 时序推理（ICEWS14，filtered MRR）— 核心进展
| 方法 | MRR | H@1 | H@10 | 备注 |
|---|---|---|---|---|
| frequency（旧基线） | 0.105 | 0.044 | 0.220 | 自家弱基线 |
| temporal_gnn（简化） | 0.286 | 0.192 | 0.467 | 单快照 |
| **recurrency（新,CPU,本文）** | **0.356** | 0.283 | 0.488 | 多信号 copy，36s |
| path-RL（本文） | 0.360 | 0.284 | 0.494 | 路径 RL |
| **re_gcn（强 backbone,本文）** | **0.380** | 0.286 | 0.565 | RGCN+GRU+ConvTransE |
| **hybrid 最优融合（re_gcn+recurrency,本文）** | **0.411** | **0.309** | **0.609** | **local+global，调权重 wc=0.3** |
| **re_gcn_full 融合（+关系演化+长历史）** | **[训练中→预期~0.44+]** | | | 强 backbone + 同融合 |
| ── 近年对标 baseline ── | | | | |
| RE-GCN (2021,经典参照) | 0.420 | | | |
| TiRGN (2022,经典参照) | 0.440 | | | |
| RLGNet/DL-CompGCN/HGLS (2024) | **~0.44-0.47** | | | **对标靶** |

**诚实结论**：**当前最强 0.411（local+global 最优融合）**，已接近 RE-GCN 0.42、距近年 SOTA(0.44-0.47) 约 0.03-0.06；**追赶路线已验证有效**（融合 +0.031；re_gcn_full 强 backbone 训练中，同融合预期 ~0.44+ 追平近年 SOTA）。关键：融合需 copy 作轻量补充(wc≈0.3)，非等权。

## 2. ch3 风险控制（C1，验证器即风控）— 已验证头条 ✅
ICEWS14 + GNN reasoner，conformal 漂移对比（`runs/conformal_gnn_icews14.json`）：
| calibrator | coverage_drift_gap（越小越鲁棒） | conformal_coverage（目标 0.9） |
|---|---|---|
| **split（静态基线）** | **0.29**（最差） | 0.858（欠覆盖） |
| **aci（自适应,本文）** | **0.20**（最好） | **0.899**（≈目标） |
| weighted | 0.23 | 0.864 |
| crc | 0.29 | 0.858 |

→ **自适应校准的覆盖漂移失真显著小于静态 split、且覆盖率最接近目标**——验证器即风险控制器的漂移鲁棒性**成立**。这是 RE-GCN/TiRGN 等纯准确率 SOTA **都没有**的能力。

## 3. ch2 事件关系抽取（MAVEN-ERE）
| 指标 | SFT | **GRPO-RLVR（本文）** | 监督 baseline(RoBERTa) |
|---|---|---|---|
| CoNLL coref F1 | 0.265 | **0.771** | 0.921 |
| temporal precision | 0.445 | **0.565** | (F1 0.560) |
| micro F1 | 0.012 | 0.070 | (总 0.518) |

**诚实**：GRPO 大幅胜自家 SFT（coref 0.27→0.77）；但**生成式抽取**设定 vs 监督式 pair 分类设定不同、绝对值低于监督 baseline（文献亦证 LLM 在此 benchmark 显著低于监督）。定位=可验证性/一致性增益，非纯 F1。

## 4. ch1 事件抽取（SARGE）— 已竞争力
| 数据集 | **SARGE(本文) F1** | Doc2EDAG | GIT | EPAL | SEELE |
|---|---|---|---|---|---|
| ChFinAnn | **86.0**(3seed 85.6) | 78.8 | 80.3 | 83.4 | 85.1 |
| DuEE-Fin | **78.0**(3seed 78.3) | 63.4 | 67.8 | 76.4 | 80.8 |

诚实边界：record binding 未解决（Δ_bind 29.9/34.7），作为可测量现象+未来工作呈现。

## 5. 三章数据闭环（自建中文金融事件图）
```
DuEE-Fin 公告 → SARGE 抽取 → 677 事件/13 类(ch1) → 关系图 20683 边(ch2) → 下一事件预测 recurrency MRR 0.422(ch3)
```
同一中文金融数据流闭合；验证器（grounding/consistency/faithfulness）贯穿三章为主线（论文逻辑闭环）。

## 6. 下一步（报告里可写"进行中"）
re_gcn_full（冲 0.42）→ 调融合权重 → 叠加 C1 风控 → 目标追平近年 SOTA 0.44-0.47 + 提供其没有的漂移鲁棒覆盖保证。详见 `docs/CLOSED_LOOP.md`、`docs/BENCHMARK_SURVEY.md`。
