# Fin-EKG · coding agent 上下文

> **`CLAUDE.md` 与 `AGENTS.md` 内容保持一致**（Claude Code 读 `CLAUDE.md`，Codex 读 `AGENTS.md`）。
> **改一份必须同步另一份。** 设计总纲 → `docs/SPEC.md`｜实时状态 → `docs/TODO.md`｜工程坑 →
> `docs/ENGINEERING_NOTES.md`｜服务器运维 → `docs/GPU_RUNBOOK.md`。

## 校验命令（改代码后必跑）

```bash
uv run pytest          # 全绿（只增不改；旧 TKG 线已移出主干，见 tag frozen-tkg-line）
uv run ruff check src tests scripts   # 0 error，≤100 列
uv run finekg-smoke    # CPU 端到端冒烟
```

## 本地环境

- 工作区 `/home/tjk/myProjects/masterProjects/Fin-EKG`；WSL2 上 Ubuntu，zsh；Python 用 `uv`。
- **本地是 git 仓库**（分支 `main`）。提交/推送**仅在用户明确要求时**。

## GPU 服务器

- SSH `gpu-4090`（cpolar 隧道，间歇性掉线）。**ssh 失败 ≠ 远端进程死亡**：三态判活（ALIVE / GONE /
  ssh 失败），只有成功 ssh 读到进程 GONE 才算结束。
- 远端根 `/data/TJK/Fin-EKG`；远端 uv `/home/TJK/.local/bin/uv`；远端 Python `.venv/bin/python`。
- **远端不是 git 仓库**：同步用 `scp`/`rsync` 指定文件 + `sha256` 双端核；**禁 `rsync --delete`**
  （会删 `runs/`、`nvmlshim/` 等 remote-only 产物）。
- 非交互 ssh 里 `python`/`jq`/`rg`/`tmux` 可能不在 PATH；用绝对路径 / `bash -lc` / `sed`·`grep`·`find`。

## GPU 运行约束

- **card 3 故障**需 NVML shim；card 0/2 常被别人占，**优先 card 1**，选卡前 `nvidia-smi`（原子核卡，不隔启动复用）。
- **未经用户明确要求，不起长 GPU 训练/推理**；`uv run` 约 1 分钟才占显存。
- 长训练用 `screen -dmS` / `nohup` + `python -u`，输出重定向 `logs/`。

## 硬约束（最易违反）

- 包/函数名**不得含 `ch1/ch2/ch3`**；新组件走 registry + lazy import；GPU 组件配 CPU 缓存回放。
- **`EventNode` schema 零新增字段**（扩展用 `metadata`）；`tests/core/test_propagation.py` 是测试锁。
- 报告结果**如实**（数字降就说降；ssh/工具失败不得伪装成结论）。**专利 / 论文写作不在计划范围。**
