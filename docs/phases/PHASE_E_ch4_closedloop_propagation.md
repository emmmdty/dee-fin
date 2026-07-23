# Phase E — Ch4 下游门控闭环修复 + 构建误差传播（headline）

> 单会话自包含契约。硬约束/校验命令见自动载入的 `CLAUDE.md`；设计见 `docs/SPEC.md` §1（Ch4）/§4。
> **这是全篇 headline + 求职(agent)/控制学科展示章。**

## Goal（完成目标）
① **agentic 下游门控闭环修复**：观测（Phase B 一致性 + Phase D 事实性 + 证据）→ 控制器选**最小代价编辑**（增/删/重标边、
并/拆节点）→ 风险闸门 → **仅当 SeDGPL 后继预测目标改善才接受**（治 self-refine 内在掉点）→ 迭代至收敛，带 repair trace。
② **gold / predicted / repaired 三图误差传播**：量化哪类 node/edge/factuality error 最破坏 reachability、修复的下游增益。

## 依赖 / 产物
- 前置：**Phase A·B**（真实预测输入）、**Phase C**（构图/约束修复）、**Phase D**（事实性信号）。A/B 未达标 → 退**受控扰动版**。
- 产出：闭环控制器 + 三图对比 + `runs/cgep/ch4_closedloop_*.json`、误差传播曲线。

## Context（复用 / 新建）
- **复用（大部分已建）**：`succession/`（`sedgpl.py`/`model.py`/`encode.py`/`linearize.py`/`selective.py`/`structure.py`/
  `predictor.py`/`metrics.py`）；`succession/cross_stage.py`（`induce_reachability:38`、`cross_stage_sweep:64`）；
  `agents/protocol.py`（`Blackboard`/`Orchestrator`/`agent_roles`——目前**单遍前馈**）+ `relations/agents/`；
  `core/calibration/propagation.py`；`scripts/evaluate_cgep*.py`、`build_cgep.py`。
- **新建**：闭环控制器（回灌边 + 终止判据 + **下游目标门控接受**）；`cross_stage.py` 补 **3 类真实扰动生成器**
  （删/增因果边、并/拆节点、扰乱时序——现仅 reachability 掩码）。

## 执行内容（Steps · TDD）
1. **下游目标探针**：SeDGPL MRR delta 作接受信号（编辑后重评估）。
2. **控制器回路**：observe→act→gate→iterate（复用 agents 脚手架，从单遍改回路）；repair trace。
3. **三类真实扰动生成器**补齐（`cross_stage.py`）。
4. **三图评测**：gold / predicted（A/B）/ repaired 的 MRR/Hits + 误差传播曲线（哪类错误最伤可达性）。

## Constraints
- 遵守 `CLAUDE.md` 硬约束；`tests/core/test_propagation.py` 测试锁不可改语义。
- 与 CFEP（TKG 纯预测）、self-healing KG（规则非下游验证）、**DeepRefine（2605.10488，下游导向 KB 精化 +
  无金标 RL）** 显式区分（`docs/SPEC.md` §5）——我们是事件因果图 + reachability + conformal 覆盖保证 + 三图分解。
- **门控信号来源必须先定死**（`docs/SPEC.md` §4.5）：无标签在线代理 or 离线诊断定位，别默认用金标 MRR（防 oracle）。

## 验收标准（Done when）
- [ ] 闭环控制器（下游门控接受 + 终止判据 + repair trace）实现 + 测试；校验命令全绿。
- [ ] **repaired MRR > predicted MRR**（如实，可能小）；误差传播曲线产出。
- [ ] 结果落 `runs/cgep/` + `docs/TODO.md`。

## GPU
重（SeDGPL 训练/推理；闭环 LLM 可走 API 不占训练）。选卡前 `nvidia-smi`，优先 card 1。

## 达不到怎么办（止损）
真实图不可得（A/B 未达标）→ 退**受控扰动版**（`cross_stage.py` 已实现 reachability 扫描，仍缺三类
真实扰动），回答“构建误差如何影响下游”；
闭环增益微弱 → 退一致性重排 + 误差传播分析，不硬撑"闭环有效"。
