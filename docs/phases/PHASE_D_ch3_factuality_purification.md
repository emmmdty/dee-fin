# Phase D — Ch3 构建图上的事件事实性检测与图净化

> 单会话自包含契约。硬约束/校验命令见自动载入的 `CLAUDE.md`；设计见 `docs/SPEC.md` §1（Ch3）。
> **novelty 落点**：不主张"用结构检测事实性"新（MAVEN-FACT 已证）；新在**预测图鲁棒性 + 净化下游**。

## Goal（完成目标）
① 结构感知事实性检测（利用论元+关系）**打底**，MAVEN-FACT 5 类 macro-F1 **打平/超 47.6**；② 量化在**预测/构建图
（有错）** 上的鲁棒性掉点（MAVEN-FACT 原文全用 gold 输入）；③ **事实性驱动图净化**（剔除/降权非事实事件及其边）
并接下游后继预测（Phase E）。

## 依赖 / 产物
- 前置：P0（`data/raw/maven_fact/train.jsonl` 已下；**`valid.jsonl` 待作者手动下载**，见 `data/raw/DATA_PROVENANCE.md`）。
  评测阶段需 valid；开发可先用 train。可选：Phase B 的预测图（做鲁棒性对比）。
- 产出：MAVEN-FACT 加载器 + 事实性检测器 + 净化算子 + `runs/ch3_factuality_*.json`。

## Context（复用 / 新建）
- **数据**：MAVEN-FACT 每 doc 含 `events[].mention[].factuality`（CT+/CT−/PS+/PS−/Uu）+ `evidence_word/evidence_offset`
  + 论元 + causal/subevent/temporal 关系（同 MAVEN-ERE 文档）。
- **复用**：`core/schema.py`（`EvidenceSpan` 存证据词）、`core/calibration/`（不确定性）。
- **新建**：`relations/data/maven_fact.py` 加载器；事实性检测器（RoBERTa/DMBERT 级 fine-tune，输入可含 gold 或预测的
  论元+关系）；净化算子（按事实性剔/降权节点与边）；**鲁棒性协议**（gold-input vs predicted-input 掉点）。

## 执行内容（Steps · TDD）
1. MAVEN-FACT 加载器（5 类标签 + evidence span）。
2. 检测器 fine-tune；报 5 类 macro-F1 vs 47.6、evidence span F1。
3. **鲁棒性**：gold 事件/论元/关系 输入 vs Phase A/B 预测输入，量化掉点。
4. **净化**：非事实事件降权/剔除 → 图质量变化；接 Phase E 度量对后继预测的增益。

## Constraints
- 遵守 `CLAUDE.md` 硬约束；**明说打底检测为复现**，novelty 在鲁棒性 + 净化下游；投稿前做新颖性扫。

## 验收标准（Done when）
- [ ] 5 类 macro-F1 ≥ 47.6（打平/超，如实）；evidence span F1 报出。
- [ ] **gold-input vs predicted-input 掉点量化**；净化前后图质量 + 下游（接 E）增益报出。
- [ ] 校验命令全绿；结果落 `runs/` + `docs/TODO.md`。

## GPU
轻（小模型 fine-tune）。

## 数据门槛
评测需 MAVEN-FACT `valid.jsonl`（作者手动下载后再跑评测行）；train 已在库可先开发+train 内切验证。

## 达不到怎么办（止损）
净化收益小 → 退"事实性检测 + 预测图鲁棒性分析"仍成独立章。
