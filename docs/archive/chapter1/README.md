# 第一章 · 事件抽取（SARGE）结果摘要

> 本目录把学位论文**第一章**（中文金融文档级事件抽取，方法 = SARGE）的权威结果、证据与图整合进 Fin-EKG，作为三章衔接与论文/学位写作时第一章数字的**单一可信来源**。
> 第一章实现仓库在 `~/myProjects/masterProjects/DEE/SARGE`（代码解耦，本仓只通过冻结契约 `core.io.event_nodes_from_sarge` 消费其 canonical 预测）。
> 数字来源：`SARGE/paper/exp/seed13_summary.md` + `SARGE/docs/exp_result.md`（已拷小证据到 `evidence/`）。

## 1. 方法定位

**SARGE = Schema-Aware Role-Grounded Extractor**：把固定槽位**事件表**当作中心预测对象，用 schema 接地的事件表生成 + 角色接地输出契约 + 4B LLM 指令微调，做中文金融文档级事件抽取（DEE）。
- 基座：**Qwen3-4B-Instruct-2507**；LoRA/QLoRA 4-bit NF4（rank 16、α=32、dropout 0.05、q/k/v/o），lr 2e-4，2 epochs，max len 4096。
- 主结果后端：HF-4bit + LoRA，k=1 greedy。口径 = **Legacy-FS**（固定 schema 角色槽 micro-F1，对齐 EPAL/SEELE 公开基线表）。

## 2. 主结果（test，seed 13）

| 数据集 | P | R | **F1** | F1(单事件) | F1(多事件) | 3-seed mean±std |
|---|---:|---:|---:|---:|---:|---:|
| **ChFinAnn-Doc2EDAG** | 84.4 | 87.7 | **86.0** | 89.9 | 81.8 | 85.57 ± 0.39 |
| **DuEE-Fin-dev500** | 76.6 | 79.3 | **78.0** | 79.3 | 77.5 | 78.32 ± 0.38 |

- **ChFinAnn 超过所列基线**（Doc2EDAG 78.8 / GIT 80.3 / ReDEE 81.9 / ProCNet 80.8 / EPAL 83.4；接近并略超 SEELE 85.1）；多事件 F1 81.8 明显领先。
- **DuEE-Fin 诚实定位**：F1 78.0 **低于 SEELE 80.8**，但多事件 F1 77.5 强、且高于 EPAL/ProCNet 的多事件项。
- 输出质量：**SchemaOK 100%，ParseFail / InvType / InvRole 全 0**（事件表生成契约稳定）。

## 3. 关键消融与诚实边界（写论文时务必保留）

- **SFT 是主要增益来源**：去掉 SFT，base 模型暴跌至 **ChFinAnn 24.8 / DuEE-Fin 3.3（HF）**。
- **Surface Memory / Slot Plan 无稳定正向证据**：HF-4bit 单变量去除几乎无下降（DuEE-Fin 78.0→78.1 / 77.6）；vLLM 0.70 显存下的召回坍塌是**后端故障形态**，非模块贡献。→ 论文**不**把这两个模块包装成核心创新。
- **record binding 未解决（核心诚实边界）**：ExactRec 远低于 role-F1 —— **ChFinAnn 56.1±2.0 / DuEE-Fin 43.3±0.5**，绑定差 **Δ_bind ≈ 29.9 / 34.7**。论文把同质记录绑定作为「可测量现象 + 诊断 + 未来工作」呈现，不声称解决。

## 4. 在三章里的角色（衔接点）

第一章产出**带证据的事件记录 = 图谱节点**，经冻结契约喂给第二、三章：

```
SARGE canonical 预测(事件表) ──core.io.event_nodes_from_sarge──▶ EventNode(jsonl)
   ──scripts/build_event_graph.py──▶ 中文金融事件图(event_graph_zh) ──▶ ch2 关系/ch3 推理
```

- 当前中文事件图的溯源与重建建议见 **`PROVENANCE.md`**（重要：现图源头存在一致性弱点，建议用主模型重建）。
- 第一章权威英文叙事源 = `SARGE/paper/ccks_2026/main_en_final.tex`（学位论文第一章据此回译，勿另起炉灶）。

## 5. 本目录内容

- `evidence/`：`asset_registry.json` / `baseline_constants.json` / `ablation_evidence.json`（论文复现表的权威常量源）+ `tables/`（主表 01/02、消融 03/04、事件级 F1 06/07、seed 稳定性 12/16）。
- `figures/`：`fig1_binding_gap`（绑定差）、`fig2_sarge_pipeline`（方法管线）、`fig3_record_buckets`（记录桶）的 PDF/SVG 矢量图，供三章统一排版。
- `PROVENANCE.md`：中文事件图的数据溯源、复现风险与重建方案。
