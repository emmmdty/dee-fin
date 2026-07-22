# Phase C2 — Ch1 跨文档泛化（ECB+/CLES）

> 骨架级契约（细节待 Phase C 完成后按真实节点模块补全）。硬约束见 `CLAUDE.md`；设计见 `docs/SPEC.md` §1（Ch1）。

## Goal（完成目标）
检验 Phase C 的 canonical event 表示能否从**文档内聚合泛化到跨文档聚合**（ECB+/CLES），对比 SECURE/MEET/DIE-EC。
**验证通用性，不进 MAVEN 主干闭环。**

## 依赖 / 产物
- 前置：**Phase C**（节点规范化模块）。
- **数据门槛**：ECB+/CLES **尚未在库**。若为云盘/需授权下载 → **交作者手动下载**（见 `data/raw/DATA_PROVENANCE.md` 惯例）。
- 产出：跨文档 coref 评测 + `runs/ch1_crossdoc_*.json`。

## 执行内容（Steps · 骨架）
1. 取 ECB+/CLES；写加载器（对齐 mention/cluster 口径）。
2. 把 Phase C 节点模块适配到跨文档设定（缺文档内语境的难例）。
3. 跨文档 coref 评测（CoNLL）对比 SECURE/MEET/DIE-EC。

## 验收标准（Done when）
- [ ] 跨文档 coref 结果 vs 三基线报出（如实）；校验命令全绿；结果落 `runs/` + `docs/TODO.md`。

## GPU
轻。

## 达不到怎么办
泛化差 → 如实报为"文档内强、跨文档需额外机制"的负结果，仍是有价值的通用性边界发现。
