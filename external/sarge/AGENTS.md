# SARGE Codex Instructions

> **SARGE** = Schema-Aware Role-Grounded Extractor. 中文金融文档级事件抽取系统，CCKS 2026 主投。

## Repository Rules

- 代码修改只进 `src/sarge/`；历史快照以 Git 历史追溯，不在主代码树里保留单独拷贝。
- `data/`、`models/`、`evaluator/` 是复制到本项目内的实体目录；不使用软链接，不再依赖外部数据/模型/评测器路径。
- 不在 `src/sarge/` 中出现阶段命名（`v1` / `v2` / `R3` / `R7` / `Phase_A/B/C` / `S4` 等）；用功能性名字。

## Paper Claim Boundaries

- 论文不是技术报告；正文叙事不要按 pipeline、脚本、JSON、parser、export 等实现清单展开。
- 不强调 SOTA、leaderboard、全面超过传统方法，或严格可控地战胜 SEELE / EPAL / ProCNet / Doc2EDAG / ReDEE 等已有系统。
- 不声称 SARGE 解决 record binding / homogeneous-record assembly；记录级组装仍应作为边界、诊断或未来工作呈现。
- 论文目标是提出一个用于解决中文金融 DEE 中部分关键问题的新方法：在闭集事件模式下，将文档证据、角色语义、原文表面值选择和可评测事件记录统一为 schema-grounded event-table generation。
- 方法贡献不必包装为“具体技术模块全新”；可以从任务重述、金融公告表面值规律、schema adherence、role grounding、输出可评测性、与既有 PLM/判别式结构预测路线的差异等角度体现。
- 英文稿是主叙事源，中文稿同步翻译；两者都要保持论文式问题-方法-证据逻辑，而不是工程实现流水账。

## Environments

- 本地 Python：`/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`
- 服务器 Python：`/data/TJK/envs/sarge_vllm_full/bin/python`
- 本地项目根：`/home/tjk/myProjects/masterProjects/DEE/SARGE/`
- 服务器项目根：`/data/TJK/DEE/SARGE/`

## Model Artifacts

- 服务器 Chinese-RoBERTa-wwm-ext：`/data/TJK/DEE/SARGE/models/chinese-roberta-wwm-ext_safetensors`
- 服务器 Qwen3-4B-Instruct-2507：`/data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507`
- 服务器 Lawformer（fallback）：`/data/TJK/DEE/SARGE/models/thunlp_Lawformer_safetensors`
- 加载时设 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` 与 `local_files_only=True`、`use_safetensors=True`

## Execution Boundaries

- 本地默认开发；不在本地跑 GPU 训练 / 推理
- 服务器 GPU 远程命令前先报告 exact command + cwd + expected outputs
- 不未经授权启动 long-running job
- additive sync（`cp -a` / `rsync -av`），不用 `rsync --delete`
- 多用户共享 gpu-4090，礼让其他用户作业。
- **kill 进程前必须检查**：若进程属于 `TJK` 用户之外的其他用户，**禁止 kill**（多人共享服务器）

## Single-Seed-First GPU Strategy

- 长训实验先单种子（seed 13）验证 hard gate
- W7 单种子达标 → W11 才启动 seed 17 / 19 补齐 mean±std
- W8 hard gate FAIL → 进入 plan §8 fallback，不消耗 GPU 在多种子上
- **GPU 分配**：gpu-4090 共 4 块 GPU（均为 4090）；小显存模型可在单块 GPU 上加载；**优先选择空闲 GPU**（兼顾时间效益与资源公平）

## Documentation Rules

- 新文档必须记录本地 + 服务器双路径与 Python 环境
- 实验产物写入 `runs/{run_name}/`，包含 `summary.json` + `checkpoints/` + `predictions/` + `eval/`
- 文档保持简洁、正确；不写过期内容

## Phase Gates

- 每周阶段对应 plan W1-W12 的 acceptance gate
- 未通过 gate 不进入下一周；进入 fallback 路径

## Local Validation Commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m pytest tests/ -v
```
