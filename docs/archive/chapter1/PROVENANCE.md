# 中文金融事件图（event_graph_zh）数据溯源与重建记录

> 目的：修复"第一章 → 二三章"衔接的**可复现性**与**模型一致性**。记录现图来源、复现风险、以及 2026-06-20 的重建尝试与结论。

## 1. 现状（主工件，2026-06-20 起沿用）

- 图：`data/processed/event_graph_zh/event_graph.json` = **677 节点 / 429 文档 / 20683 边**（coreference 145 + temporal 20538；metadata `edges_raw=498` 只数关系抽取候选边，时序闭包边另计）。
- 中间产物：`data/raw/event_graph_zh/event_nodes.jsonl`（677 条 EventNode，已回填 argument evidence；trigger 仍空——SARGE canonical 不导出 trigger span）。
- ch1 统计/图（`docs/archive/midterm/`）即基于此图。**这是 midterm 用过、结构可用（有时序/共指边）的工件，保留为主。**

## 2. 溯源核查发现（2026-06-20 实测）

- 429 文档 = **DuEE-Fin-dev500 的 dev split**（500 docs 中 429 有可转换事件）；与主模型全量 test（1171 docs）**零文档重叠**。
- 来源批次 = `SARGE/runs/` 下 **20260522 的 `sarge_deepseek_*` 预测**（DuEE-Fin-dev500 dev，limit500）。**弱点**：这批是 **DeepSeek-API 探针 run，非 SARGE 主模型（本地 Qwen3-4B SFT，seed13）**；且无单个 canonical pred 精确对上 677 事件（最接近 537）→ 确切上游未干净留档。

## 3. 风险

1. **可复现性**：产出现图的上游预测未在本仓 bit 级留档；`event_nodes.jsonl` 是事实上的已留档中间产物，但上游不可一键复现。（**2026-06-29 部分缓解**：**主模型** seed13 源 pred 已归档进 data/（见 §4），主模型路径现可一键重建；**dev 主图**的 20260522 DeepSeek 探针源仍如 §2 所述无法 bit 级精确留档。）
2. **三章一致性**：现图由 DeepSeek-API 抽取，而第一章头条由**本地 Qwen3-4B SFT（SARGE 主模型）**报告 → "我们的 SARGE 模型抽事件→建图→推理"叙事不闭合。

## 4. 2026-06-20 重建尝试与结论（重要）

用 **SARGE 主模型 seed13 在全量 DuEE-Fin test（1171 docs）的预测**重建（源 `record_binding_dueefin_test_seed13_surface_tau085_20260525/.../test.canonical.pred.jsonl` + 源文档 `SARGE/data/DuEE-Fin-dev500/test.jsonl`，CPU）：

- 结果：**1554 节点 / 1171 docs / argument evidence 有效**（char offset 真实，如 联络互动@276-280）—— 节点侧更大、更干净、与主模型一致。✅
- **但 0 边**（1120 候选关系边全被 grounding 丢弃，且时序闭包边=0）。对比旧 dev 图 20538 条时序边，主模型 test 图的时序结构**没有生成**。
- 结论：**已回退**，主工件仍用旧 dev 图（有可用的时序/共指结构）。主模型重建产物存为 side 工件待调试：
  - `data/raw/event_graph_zh/event_nodes_mainmodel_seed13_1554.jsonl`
  - `data/processed/event_graph_zh_mainmodel_seed13/`（1554 节点 / 0 边）
  - `data/raw/event_graph_zh/sarge_mainmodel_seed13_dueefin_test.canonical.pred.jsonl`（**2026-06-29 归档**：SARGE 主模型 Qwen3-4B seed13 在 DuEE-Fin test 1171 docs 的 canonical 源预测，520K；SARGE 迁入 `external/sarge` 后从其 `runs/`（gitignore、不进 subtree）固化进本仓 data/，消除"主模型源未留档"风险。一键重建节点：`uv run python scripts/sarge_to_event_nodes.py --pred data/raw/event_graph_zh/sarge_mainmodel_seed13_dueefin_test.canonical.pred.jsonl --source-docs data/raw/event_graph_zh/dueefin_test_source_docs.jsonl --output data/raw/event_graph_zh/event_nodes_mainmodel_seed13_1554.jsonl`。源文档 `dueefin_test_source_docs.jsonl`（DuEE-Fin test 1171 docs，2.2M，同 2026-06-29 一并归档）用于回填 argument evidence 的 char offset。）

**待办（follow-up，非关键路径）**：调试为何主模型 test 图时序闭包边=0（疑似 P1 闭包改造后时序边须过 grounding 而被丢、或时序闭包按公司分组而 1171 独立公告少重复）。修好后用主模型图替换为主工件，可一并增强下游选股 T8 的公司覆盖。⚠️ 中文事件图的 0/稀疏边是**已知根本限制**（DuEE-Fin 公告独立、跨文档无共指），真正做密靠 **T9**（在 CMIN/Astock 新闻上跑 SARGE，公司随时间重复=真时序）。

## 5. 旁注：ch3 头条不依赖此图

ch3 论文头条实验（path-RL、C1 漂移 conformal）跑在公开 TKG benchmark（ICEWS14/18、FinDKG）上，**不依赖**中文自建图。中文图用于：ch1 节点统计 + fig9 事件中心 KG 示例 + 下游选股 T8（需公司名 join）。故上述 0 边问题不影响论文主体。
