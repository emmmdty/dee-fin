# Phase F — 端到端误差预算（有前提）

> 骨架级契约。硬约束见 `CLAUDE.md`；设计见 `docs/SPEC.md` §4.3/§5.5/§6。

## Goal（完成目标）
把三段误差——节点漏 α_d（Phase C 置信）+ 边漏 α_e（Phase B 准入）+ 预测漏 α_p（Phase E 覆盖）——经 **union bound**
组合，**reachability 单列 + 条件回收**，给端到端界 + 分层 FNR；**naive vs 预算法对照**。**显式标注交换性/固定后处理
前提，不作无条件 distribution-free 保证。**

## 依赖 / 产物
- 前置：**Phase B·C·D·E**（节点预算来自 C）。
- 产出：端到端预算实验 + `runs/cgep/end2end_budget_*.json`。

## Context（复用）
- `core/calibration/propagation.py`（union bound + 条件回收 `allocate_budget_conditional`，已测——测试锁 `tests/core/test_propagation.py`）；
  `succession/selective.py`（选择性预测）；`succession/cross_stage.py`。

## 执行内容（Steps · 骨架）
1. 串三段预算（α_d + α_e + α_p），reachability 单列。
2. 条件回收（held-out 准入证不可达率上界 → 收紧预测集）。
3. naive union vs 预算法对照曲线；分层 FNR。

## 验收标准（Done when）
- [ ] 端到端界 + 分层 FNR 报出；naive vs 预算法对照；**前提（交换性/固定后处理）显式写明**。
- [ ] 校验命令全绿；结果落 `runs/` + `docs/TODO.md`。

## GPU
轻（复用已训模型的排名文件）。
