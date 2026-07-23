# Phase H — 稳健化（多种子 + 消融 + 投稿前新颖性扫）

> 骨架级契约。硬约束见 `CLAUDE.md`；新颖性约束见 `docs/SPEC.md` §5。**多种子放最后。**

## Goal（完成目标）
① **多种子 13/17/42** 跑各章主表，报 mean±std；② 消融补齐（M1/M2 噪声级如实入消融，各章 delta 消融）；
③ **投稿前穷尽 pipeline-CP 新颖性扫**（读 2404.17769、CFEP 2026.findings-acl.258、**PASC 2605.18812 /
SCRC 2512.12844 / DeepRefine 2605.10488 / SRE 2506.06910** 全文；竞品区分见 `EXPERIMENTS.md` §5），把窄 delta 从 MEDIUM 提到 HIGH 或加限定；
④ **Ch2 方法改名定稿**（避 SCRC 2512.12844），docs/代码统一。

## 依赖 / 产物
- 前置：**Phase A–F**（各章主结果就位）。
- 产出：主表 mean±std + 消融表 + 新颖性扫记录 + 改名 PR。

## 执行内容（Steps · 骨架）
1. 各章主实验多种子重跑（GPU），聚合 mean±std。
2. 消融补齐；负结果如实入消融，不换指标绕过。
3. 新颖性扫：逐条核 CRC/CS-CRP/事实性/闭环最近邻；文档化结论。
4. Ch2 方法统一改名（突出 reachability-coupled / recall-coverage 组合）。

## 验收标准（Done when）
- [ ] 各章主表 mean±std；消融全；新颖性扫文档化（含是否发现先例的处置）；改名 docs/代码一致。
- [ ] 校验命令全绿；结果落 `runs/` + `docs/TODO.md`。

## GPU
重（多种子重跑）。GPU 长期不可用 → 冻结训练类，先做新颖性扫/消融设计/改名（本地可做）。
