# 归档索引（历史留痕 · 正文已移出仓库）

> 这里只留**索引**：归档了什么、为何归档、去哪找。**正文一律不在仓内**，避免旧口径污染检索面
> 与 agent 上下文。当前权威见 [`SPEC.md`](SPEC.md)（总纲）与 [`TODO.md`](TODO.md)（实时状态）。

**正文位置**：仓库外 `../ekg-backup-20260727/`（与本仓同级，不入 git）。
git 历史中最后一次含正文的提交是 `3390363`，可用 `git show 3390363:docs/archive/<文件>` 取回。

| 备份子目录 | 内容 |
|---|---|
| `docs_archive/` | 下表全部 `.md` + `specs/`（三份历史设计稿） |
| `docs_archive_large/` | `midterm/`（中期报告 38M）· `patent/`（旧专利交底书）· `chapter1/`（SARGE 图表证据）· `projects/`（答辩 PPT 工程 79M） |
| `docs_orphans/` | `事件图谱构建研究进展.md`（v3「两章」时代综述，与 v4 四章冲突）· `OVERVIEW.html`（课题总览，内容与 SPEC 重叠） |

## 归档清单与理由

| 文档 | 为何归档 |
|---|---|
| `ARCHITECTURE.md` `RESEARCH_MAP.md` `RISK_CONTROL_DESIGN.md` `RL_DESIGN.md` | 架构 / 代码映射 / CS-CRP / verifier-as-reward 设计；活内容已并入 `SPEC.md`，金融口径过时。 |
| `HANDOFF_2026-07-10.md` `HANDOFF_2026-07-11.md` | 冷启动交接稿；状态与计划并入 `SPEC.md` + `TODO.md`。 |
| `NOVELTY_A1_2026-07-11.md` `NOVELTY_CSCRP_2026-07-11.md` | 新颖性复核证据表；结论并入 `SPEC.md §5`。**投稿前重扫新颖性时需从备份取回。** |
| `OUTLINE.md` `THESIS_DESIGN.md` `THESIS_REDESIGN_2026-07-10.md` `UPGRADE_PROMPT_2026-07-09.md` | 旧顶层主线 / outline / 提案 / 升级提示，均已被推翻。 |
| `CLOSED_LOOP.md` `BENCHMARK_SURVEY.md` `MIDTERM_HANDOFF.md` `SERVER_W3-4_RUNBOOK.md` | 旧闭环 / 竞品 / 中期交接 / 运维稿，均过时。 |
| `SETUP.md` | 环境安装，已并入根 `README.md`。 |
| `SARGE_RESULTS_SNAPSHOT.md` | SARGE 主结果数字快照 + 源仓与取回路径；2026-07-27 从主干移除时留档。 |
| `PHASE_G_financial_layer.md` | 旧 Phase G（金融应用验证层）契约。四章无一依赖，题目本无「金融」→ 整体移除。 |
| `specs/` | 三份历史设计稿（2026-06-16 评测口径 · 2026-07-17 结构感知编码 · 2026-07-20 四章重设）。 |

## 时间线

- **2026-07-11**：文档体系重构，设计 / 交接 / 新颖性 / 专利类旧稿统一归档；旧「实体中心中文金融 + TKG
  外推」主线整体作废（代码在 tag `frozen-tkg-line`）。
- **2026-07-27**：SARGE 与 Phase G 金融应用层移出主干；项目改名 `Fin-EKG` → `ekg`；归档正文与两份
  孤儿文档一并移出仓库，仓内只留本索引页。
