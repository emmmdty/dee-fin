# 归档文档（历史留档，勿作依据）

这些是**已被取代的旧文档**。当前权威见仓库现役文档：

- **`docs/SPEC.md`** — 开发驱动总纲（主线/架构/机制/新颖性/实验）。
- **`docs/TODO.md`** — 实时状态与待办。
- **`README.md` · `AGENTS.md` · `CLAUDE.md`**（根） — 方法论 / agent 运维 / 索引。
- 现役参考：`docs/GPU_RUNBOOK.md` · `docs/DATASETS.md` · `docs/ENGINEERING_NOTES.md`。

**2026-07-11 文档体系重构**：把设计/交接/新颖性/专利等旧文档统一归档，活内容已 distill 进
`docs/SPEC.md` 等现役文档；旧「实体中心中文金融 + TKG 外推」主线整体作废。

| 文档 | 为何归档 |
|---|---|
| `ARCHITECTURE.md` `RESEARCH_MAP.md` `RISK_CONTROL_DESIGN.md` `RL_DESIGN.md` | 架构/代码映射/CS-CRP/verifier-as-reward 设计；活内容并入 `docs/SPEC.md`，金融口径过时。 |
| `HANDOFF_2026-07-10.md` `HANDOFF_2026-07-11.md` | 冷启动交接稿；状态/计划并入 `docs/SPEC.md` + `docs/TODO.md`。 |
| `NOVELTY_A1_2026-07-11.md` `NOVELTY_CSCRP_2026-07-11.md` | 新颖性复核证据表；结论并入 `docs/SPEC.md §5`。 |
| `OUTLINE.md` | 旧金融论文 outline（2026-06-29，方案 A）；主线已改 CGEP。 |
| `SETUP.md` | 环境安装，已并入根 `README.md` 快速开始。 |
| `midterm/` | 中期报告全套（图/数据/脚本/docx/pptx，旧金融三章）。中期答辩已过。 |
| `patent/交底书.md` | 旧金融风控专利交底书。作者定：专利归档、不再安排专利写作（2026-07-11）。 |
| `THESIS_DESIGN.md` `THESIS_REDESIGN_2026-07-10.md` `UPGRADE_PROMPT_2026-07-09.md` | 旧顶层主线/提案/升级提示，均被推翻。 |
| `CLOSED_LOOP.md` `BENCHMARK_SURVEY.md` `MIDTERM_HANDOFF.md` `SERVER_W3-4_RUNBOOK.md` | 旧闭环/竞品/中期交接/运维稿，均过时。 |
| `chapter1/` | 第一章 SARGE 中文金融事件抽取的证据/图/表（结果非设计）；事件中心主线下不属现役，2026-07-11 归档。 |
| `projects/` | 中期答辩 PPT 工程（gitignore 的二进制，已移入 archive）。 |

归档时间：2026-07-11（含此前批次）。
