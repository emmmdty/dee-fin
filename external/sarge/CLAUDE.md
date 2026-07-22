@AGENTS.md

# 本地路径速查

- 项目根：`/home/tjk/myProjects/masterProjects/DEE/SARGE/`
- 数据（复制，固定快照）：`/home/tjk/myProjects/masterProjects/DEE/SARGE/data/`
- 评测器（复制，固定快照）：`/home/tjk/myProjects/masterProjects/DEE/SARGE/evaluator/`
- 本地 Python：`/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`

# 服务器路径速查

- 项目根：`/data/TJK/DEE/SARGE/`
- 数据目录：`/data/TJK/DEE/SARGE/data/`
- 模型目录：`/data/TJK/DEE/SARGE/models/`
- Qwen 模型：`/data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507`
- 服务器 Python：`/data/TJK/envs/sarge_vllm_full/bin/python`

# Plan 与进度

- Plan 文件：`/home/tjk/.claude/plans/`
- 阶段：W1-W12（CCKS 2026 主投，2026-08 截稿前）
- 论文标题（主）："**SARGE: Schema-Grounded Event-Table Generation for Chinese Financial Document-Level Event Extraction**"
- 论文口径：英文稿是主叙事源，中文稿同步翻译；不要写成技术报告，不强调 SOTA/leaderboard，不声称解决 record binding；重点写成一个用于解决部分中文金融 DEE 问题的新方法。
- 贡献表达：不强行强调具体技术模块全新，可从任务表述、schema adherence、role grounding、输出可评测性、金融公告表面值规律，以及与既有 PLM/判别式结构预测路线的差异来体现。

# 多人共享与 GPU 规则

- **Kill 进程**：服务器多人共享，kill 进程前必须检查进程属主；仅允许 kill `TJK` 用户的进程，其他用户进程禁止 kill
- **GPU 资源**：gpu-4090 共 4 块 GPU（均为 4090），小显存可在单块 GPU 上加载；**优先选择空闲 GPU**，兼顾时间效益与资源公平
