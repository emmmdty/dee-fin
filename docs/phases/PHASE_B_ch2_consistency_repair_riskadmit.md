# Phase B — Ch2 全局一致解码 + 可追溯修复 + 风险受控准入

> 单会话自包含契约。硬约束/环境/校验命令见自动载入的 `CLAUDE.md`；设计见 `docs/SPEC.md` §4.2/§4.3/§5.5。

## Goal（完成目标）
在 Phase A 的边打分之上：① **全局一致解码**（时序传递闭包 / 因果无环 / coref 对称）；② 产**结构化 repair trace**
（改了哪条边、触发哪条违规、前后差异）；③ **可达性驱动的风险受控边准入**（分层报告 FNR，**改名避 SCRC**）。
目标：violation/cycle 率显著↓、关键边风险受控、**ECG 可重建率↑**（解堵 Ch4 真实图）。

## 依赖 / 产物
- 前置：**Phase A**（提供每类关系基础置信度）。
- 产出：repair trace 数据结构 + 一致图；`runs/relations/consistency_repair_*.json`。

## Context（复用 / 新建）
- **复用**：`relations/consistency/__init__.py`（`identity`/`greedy`、`_break_cycles`、`_temporal_closure` 已测）、
  `core/graph.py`（`transitive_closure_pairs`、`consistency_report`）、`relations/admission.py`
  （`edge_admission.register("crc")`、`CRCEdgeAdmission` 已测）、`relations/grounding/`（provenance）。
- **新建**：给一致性求解器加 `RepairTrace`（结构化编辑记录）；准入+修复的**校准顺序**处理（§5.5：把"准入+修复"作单一
  固定后处理映射后再校准，或修复后再校准）。

## 执行内容（Steps · TDD）
1. 测试：`RepairTrace` 记录每次编辑（dropped/added edge、触发的 violation、before/after 计数）。
2. 实现：扩展 greedy 求解器发出 trace；不改既有默认行为（保测试锁）。
3. 接 CRC 准入到 A 的分数；**分层报告** marginal / 文档级 / 关系类条件 FNR + 准入边集大小（§5.5）。
4. 度量 violation/cycle 率修复前后 + **ECG 可重建率**（喂 Ch4）。

## Constraints
- 遵守 `CLAUDE.md` 硬约束；`tests/core/test_propagation.py` 是测试锁，不可改语义。
- CRC 表述按 §5.5：交换性+固定后处理下的**边际期望 FNR**，分层报告，不写"每篇/每类都保证"。

## 验收标准（Done when）
- [ ] `RepairTrace` 产出 + 测试；校验命令全绿。
- [ ] 分层 FNR + 准入边集大小报出；violation/cycle 率修复后显著↓。
- [ ] ECG 可重建率较 A 原始图↑（数字如实）；结果落 `runs/` + `docs/TODO.md`。

## GPU
轻（主要 CPU）。

## 达不到怎么办（止损）
repair gain 微弱 → 退 consistency-aware reranking / constrained decoding（综述同款替代），仍成章。
