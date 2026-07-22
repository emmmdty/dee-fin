# Phase C — Ch1 证据感知规范事件节点（含论元与不确定性）

> 单会话自包含契约。硬约束/校验命令见自动载入的 `CLAUDE.md`；设计见 `docs/SPEC.md` §1（Ch1）。

## Goal（完成目标）
构建**去重、可溯源、身份统一**的 canonical event nodes：事件检测 → **相似事件难例判别（hard-negative）** →
**不确定性感知规范化聚类** → **簇级证据/置信聚合** → 挂论元（MAVEN-Arg）。产出的 `node_confidence` 是**下游可消费的
校准置信**（喂 Ch4 误差预算）——这是与普通事件共指的关键差异。SARGE 只作 Phase G 金融迁移，
不等同于本章已完成。

## 依赖 / 产物
- 前置：P0（`data/raw/maven_arg/` 已下、`maven_ere` coref、MAVEN 检测均在库）。
- 产出：MAVEN-Arg 加载器 + 节点规范化模块 + `runs/ch1_nodes_*.json`。

## Context（复用 / 新建）
- **复用**：`core/schema.py`（`EventNode:58`、`EvidenceSpan:38`、`EvidenceLink`）、`core/eval/coreference.py`
  （MUC/B³/CEAFe/CoNLL 已建）、`core/calibration/`（不确定性）、`core/io.py`。
- **新建**：`relations/data/maven_arg.py` 加载器（仿 `relations/data/maven_ere.py`）；节点规范化模块
  （难例判别 pair 分类 + 不确定性感知聚类 + 簇级聚合 + `node_confidence` 校准）。

## 执行内容（Steps · TDD）
1. MAVEN-Arg 加载器（events + argument + entities，doc-id 与 ere 对齐）。
2. 检测评测（micro-F1）；相似事件**难例对**判别器（同类型近义触发词负采样）。
3. 不确定性感知规范化聚类 → canonical node；簇级 evidence/置信聚合；`node_confidence` 校准（ECE/reliability）。
4. 输出每节点 `{event_type, canonical_trigger, canonical_arguments, mention_cluster, evidence_spans,
   node_confidence, provenance}`；新增语义统一放入 `EventNode.metadata`，不改冻结 schema。

## Constraints
- 遵守 `CLAUDE.md` 硬约束（`EventNode` schema 冻结，扩展走 `metadata`）。
- coref 主干是复现、不主张新颖；创新在难例判别 + 下游可消费校准置信 + 证据冲突消解。

## 验收标准（Done when）
- [ ] 检测 micro-F1 ~60+；coref MUC ~86 可比区间；**相似事件误合并率显著↓**；`node_confidence` 校准（ECE 报出）。
- [ ] 校验命令全绿；结果落 `runs/` + `docs/TODO.md`（如实）。

## GPU
轻（检测可复用/小模型 fine-tune）。

## 达不到怎么办（止损）
evidence 对齐评测难设计 → 优先可自动计算指标 + 小规模人工核验；退"canonical table + 难例判别"仍成章。
