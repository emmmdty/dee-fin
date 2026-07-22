# Fin-EKG v4 · GPU 服务器运行与同步手册

> 适用于 `gpu-4090:/data/TJK/Fin-EKG`。本手册只覆盖 v4 主线与仍在主干的 Ch4 可靠性模块。
> 旧 temporal GNN、RE-GCN、Path-RL、hybrid 命令已失效；需要复现时从 tag `frozen-tkg-line`
> 单独建立工作区，不得在当前主干照抄旧命令。

## 1. 当前状态

- P0 主数据已就位；当前关键路径是 Phase A 判别式关系抽取器。
- `src/finekg/relations/extractor/supervised.py` 尚未实现，因此**当前没有可启动的 Phase A 长训练命令**。
- 现有 `scripts/train_relation_extractor.py` 是旧生成式 Qwen LoRA 基线，不是 Phase A 的判别式模型；
  不得把重跑该脚本写成 v4 Phase A 进展。
- 已有 GPU 证据：SeDGPL 基线、M1/M2 A/B、M3a risk-coverage 和受控 cross-stage 排名。
  准确数字以 `docs/TODO.md` 和 `runs/cgep/*.json` 为准。

## 2. 环境

- SSH：`ssh gpu-4090`；远端根：`/data/TJK/Fin-EKG`。
- uv：`/home/TJK/.local/bin/uv`；项目 Python：`/data/TJK/Fin-EKG/.venv/bin/python`。
- 非交互 SSH 的 `python`、`uv`、`jq`、`rg`、`tmux` 可能不在 PATH；用绝对路径或 `bash -lc`。
- card 3 故障，需 NVML shim；card 0/2 常被占，优先 card 1，但**每次启动前必须重新运行
  `nvidia-smi` 原子核卡**。
- `uv run` 可能约 1 分钟后才占显存。SSH 失败不等于任务死亡；只有成功 SSH 并确认进程 GONE 才算结束。

## 3. 三端同步协议

GitHub 与 WSL 以同一 Git commit 为代码权威；4090 不是 Git 仓库，通过 tracked-file 清单同步。

1. WSL 工作区先验证并提交、推送 GitHub。
2. 只同步 `git ls-files` 返回的 tracked files，保留 `data/`、`runs/`、`.venv/`、`nvmlshim/` 等
   remote-only 内容。
3. **禁止 `rsync --delete`**；禁止把仓库根、`data/` 或 `runs/` 当删除目标。
4. 同步后基于同一 tracked-file SHA-256 清单在 WSL 和 4090 校验；GitHub 用 commit SHA 校验。

推荐同步形状：

```bash
git ls-files -z | rsync -azR --from0 --files-from=- ./ gpu-4090:/data/TJK/Fin-EKG/
```

若只同步本次提交，可把 `git diff-tree --no-commit-id --name-only -r HEAD` 的结果写入临时 file list，
再使用 `rsync -azR --files-from=<list>`。无论哪种方式都要先确认清单不含 `data/raw`、`data/processed`
和 `runs`（唯一例外是已跟踪的 `data/raw/DATA_PROVENANCE.md`）。

## 4. 启动授权门槛

任何新的 GPU 长任务启动前，先向用户给出：

- 完整命令；
- 远端工作目录；
- 物理 GPU 编号；
- 预计时长；
- 日志与模型/指标产物路径。

未经明确授权不启动训练或大模型推理。授权后使用 `screen -dmS` 或 `nohup` + `python -u`，日志写入
`logs/`；不得以前台 SSH 会话承载长任务。

## 5. v4 GPU 路线

| Phase | GPU | 当前可运行性 | 远端产物 |
|---|---|---|---|
| A 判别式关系抽取 | 重 | 待实现 `supervised.py` 与训练 CLI 后才能启动 | `runs/relations/supervised_*.json` + checkpoint |
| B 一致性/修复/风控 | 轻 | 依赖 A 的缓存预测；CPU 路径优先 | `runs/relations/*repair*.json` |
| C 规范节点 | 轻 | 依赖 MAVEN-Arg loader/模型实现 | `runs/nodes/*.json` |
| D 事实性/净化 | 轻 | MAVEN-FACT 数据已就位，代码未实现 | `runs/factuality/*.json` |
| E 闭环/三图传播 | 重 | SeDGPL 可复用；真实闭环依赖 A/B/C/D | `runs/cgep/*closedloop*.json` |
| H 多种子 | 重 | 只在 A–F 主结果稳定后执行 | 各主表 seed 13/17/42 |

Phase A 实现完成后，具体命令必须从实际 CLI `--help` 和配置生成，不能预先复制旧 LoRA/GRPO 命令。

## 6. 仍可复现的 Ch4 基线

本地 CPU 可先跑数据和 frequency/random 路径；SeDGPL 才需要 GPU：

```bash
cd /data/TJK/Fin-EKG
CUDA_VISIBLE_DEVICES=<空卡> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/TJK/.local/bin/uv run --extra llm python -u scripts/evaluate_cgep.py \
  --dataset maven --predictor sedgpl --model-path <roberta-base-path> --epochs 10 \
  --output runs/cgep/maven_sedgpl.json
```

该命令仅用于复现既有 Ch4 基线，不代表 Phase E 闭环完成。启动前仍需执行 §4 的授权流程。

## 7. 监控与结束判定

- 进程：使用不会匹配探针自身的模式，例如 `pgrep -af '[e]valuate_cgep'`。
- 显存：`nvidia-smi`；只有进程与显存同时符合预期，才判 ALIVE。
- 日志：优先 `tail` 任务自己的日志，不用 SSH 连通性推断训练状态。
- 结束：成功 SSH 后确认进程 GONE，再检查退出日志和产物完整性；只有文件存在不代表训练成功。
- 回传：指标 JSON、必要日志和 manifest 定向回传；checkpoint 只在明确需要时传输。
