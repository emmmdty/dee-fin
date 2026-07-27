# Phase G — 金融应用验证层

> 骨架级契约。硬约束见 `CLAUDE.md`；设计见 `docs/SPEC.md` §1（金融作应用验证层，非整章）。

## Goal（完成目标）
用 **CCKS-FinCausal + SARGE** 验证"构建→预测"管线**可迁移到金融文本**，兑现课题标题的金融验证。
**聚焦案例研究/一节；不拿整章赌金融稀疏因果图**（Astock、entity-mode 已证死路，冻结）。

## 依赖 / 产物
- 前置：**Phase A·B**（关系抽取/一致性管线）。
- 数据：`data/processed/ccks_fin_causal/`（train/val/test 已在库）；`external/sarge`（subtree，金融事件抽取器）。
- 产出：金融构建→预测案例 + `runs/financial_*.json`。

## Context（复用）
- `scripts/sarge_to_event_nodes.py`（SARGE→事件节点适配）；`core/io.py::event_nodes_from_sarge`；Phase A/B 关系管线。

## 执行内容（Steps · 骨架）
1. SARGE 金融事件抽取 → 事件节点（应用层节点来源，不替代 v4 Ch1）。
2. CCKS-FinCausal 关系抽取/一致性迁移（用 Phase A/B 管线）。
3. 金融构建→预测小案例；报可迁移性。

## 验收标准（Done when）
- [ ] 金融构建→预测案例 + 可迁移性结论（如实）；校验命令全绿；结果落 `runs/` + `docs/TODO.md`。

## 达不到怎么办
金融因果图连通性过低 → 降级为"构建可迁移性"验证，MAVEN 主干不受影响。
