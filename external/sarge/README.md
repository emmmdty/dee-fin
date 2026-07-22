# SARGE

**S**chema-**A**ware **R**ole-**G**rounded **E**xtractor — 面向中文金融文档级事件抽取（Document-Level Event Extraction, DEE）的 schema-grounded event-table generation 方法。

## 概述

SARGE 将中文金融公告、闭集事件模式、角色约束的原文取值选择和可评测事件记录统一到一个生成式抽取目标中。项目论文目标不是技术报告或 SOTA/leaderboard 声明，而是提出一种用于解决部分中文金融 DEE 问题的方法：让大语言模型在模式约束下完成文档理解、角色取值和事件表生成。

当前论文叙事应强调方法相对既有 PLM/判别式结构预测路线的差异：schema adherence、role grounding、输出可评测性，以及金融公告中关键论元具有稳定表面值这一任务特性。Surface Memory、Slot Plan、SACD、LRD 等组件应按系统实现或诊断辅助呈现，不应被写成已经稳定贡献主指标增益的核心创新；record binding / homogeneous-record assembly 仍是需要明确呈现的边界。

## Pipeline

```
Document
   → Surface Memory Builder (CSG)
   → Slot Planner (LEEP, schema-aware)
   → GETM Generator (Qwen3-4B + LoRA SFT, greedy, role-safe contract, optional SACD)
   → Record Disambiguation (rule planner)
   → Canonical Export → Evaluator (3 tracks)
```

## 项目结构

```
SARGE/
├── src/sarge/         # Python package
├── scripts/           # CLI 入口
├── tests/             # pytest
├── docs/              # 文档（含 reproducibility.md）
├── paper/             # 论文草稿与可复现图表资产
├── data/              # 复制到本项目的数据快照（git 忽略）
├── models/            # 复制到本项目的模型快照（git 忽略）
└── evaluator/         # 复制到本项目的评测器快照
```

## 论文草稿

CCKS 2026 论文主稿以 [`paper/ccks_2026/main_en.tex`](paper/ccks_2026/main_en.tex) 为权威叙事源，中文同步稿为 [`paper/ccks_2026/main_zh.tex`](paper/ccks_2026/main_zh.tex)。写作时避免技术报告式模块堆叠，不写“首次”“SOTA”“全面超过”或“解决记录绑定”等强 claim。

## 快速开始

详见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 评测器

3 个 track，来自本项目内的 `evaluator/`：

- `legacy_doc2edag` — Doc2EDAG/ProcNet-style micro-F1，用于 fixed-slot 兼容参考比较
- `unified_strict` — 全局二分图严格匹配，内部诊断
- `docfee_official` — DocFEE 官方评测器，DocFEE 基线对比

## 许可

MIT
