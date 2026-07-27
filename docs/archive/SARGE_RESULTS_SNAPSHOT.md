# SARGE 结果快照（归档，勿作 v4 依据）

> 2026-07-27 从主干移除 SARGE 时留下的**数字快照**。SARGE 不是 v4 任何一章的组件：
> Ch1 用 MAVEN 检测/Arg/ERE-coref、Ch2 用 MAVEN-ERE、Ch3 用 MAVEN-FACT、Ch4 用 CGEP-MAVEN/ESC。
> 保留此文件只为两件事：中期报告已交付内容的可追溯，以及这组**唯一达到公开第一梯队**的可比数字。

## 它现在在哪

- **活的源仓**：`/home/tjk/myProjects/masterProjects/DEE/SARGE`（独立项目，仍在推进；
  2026-07-22 最新提交为 JCST 方向的 hyperedge 可行性实验）。论文稿、评测器、模型快照都在那里。
- **服务器**：`/data/TJK/DEE/SARGE/`（与 Fin-EKG 的 `/data/TJK/Fin-EKG` 无关）。
- **本仓历史**：移除前的 vendored 副本与适配代码见 `git show 080f93a:external/sarge/`、
  `git show 080f93a:scripts/sarge_to_event_nodes.py`、`git show 080f93a:src/finekg/core/io.py`
  （`event_nodes_from_sarge`）。要用时从历史取回即可，无需重写。

## 主结果（口径 `legacy_doc2edag` / Legacy-FS，seed 13，HF-4bin + LoRA, k=1 greedy）

| 数据集 | Doc2EDAG | GIT | EPAL | SEELE | **SARGE** |
|---|---:|---:|---:|---:|---:|
| ChFinAnn | 78.8 | 80.3 | 83.4 | 85.1 | **86.0** |
| DuEE-Fin (dev500) | — | — | — | 80.8 | **78.0** |

- 多事件子集 DuEE-Fin F1 **77.5**（相对单事件回落幅度小于多数基线）。
- 三随机种子标准差：ChFinAnn **±0.39**、DuEE-Fin **±0.38**。
- 消融要点（如实）：SFT 是主要增益来源（no-SFT 仅 0.2482）；**Surface Memory 与 Slot Plan
  无稳定正向证据**，不得写成核心创新；record binding 仍是需明确呈现的边界。
- 完整实验表见源仓 `docs/exp_result.md`。

## 与学位论文的关系

- 中期报告《基于约束引导的事件知识图谱构建与时序推理方法研究》（2026-06-26）的
  §1.1.1 与 §2.2 是 SARGE 内容；v4 已把 Ch1 换成 MAVEN 规范事件节点，**SARGE 不再等于 Ch1**。
- 终稿若需交代中期连续性，用本文件的数字叙述即可，**不需要在主干保留任何 SARGE 代码**。
- 已移除的 Phase G（金融应用验证层）原设计见本目录 `PHASE_G_financial_layer.md`。
  其立论「兑现课题标题的金融验证」不成立——注册题目中并无「金融」。
