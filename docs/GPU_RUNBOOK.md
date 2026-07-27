# EKG v4 · GPU 服务器运行手册

> 适用于 `gpu-4090:/data/TJK/ekg`。只覆盖 v4 主线与仍在主干的 Ch4 可靠性模块。旧 temporal GNN、
> RE-GCN、Path-RL、hybrid 命令已失效；复现需从 tag `frozen-tkg-line` 单独建工作区，**不得在当前
> 主干照抄旧命令**。代码/数据同步的权威是 [`PIPELINE.md`](PIPELINE.md)（代码走 git、产物走 scp）。

## 0. ⛔ 头号红线：服务器上不要跑 `uv run` / `uv sync`

**服务器 venv 装的是 `llm+serve+rl+gnn+dev` 全套 extras**（torch 2.6.0+cu124 / transformers 4.53.3 /
vllm 0.8.5 / trl 0.18.2 / peft 0.15.2 / ray / xformers）。`uv run` 与 `uv sync` 会把环境**对齐到你
给出的 extras 集合，多出来的一律卸载**。实测（2026-07-27 `--dry-run`）：

| 命令 | 后果 |
|---|---|
| `uv sync` / `uv run …`（不带 extra） | **卸 165 个包** —— torch、transformers、vllm、trl、peft、bitsandbytes 全没 |
| `uv sync --extra llm` / `uv run --extra llm …` | **仍卸 109 个** —— vllm、trl、ray、xformers、torchvision、torch-geometric |
| 全套 `--extra llm --extra serve --extra rl --extra gnn --extra dev` | 只卸 `patchelf`/`setuptools` 构建残留（安全，但没必要） |

恢复要重下数 GB。**因此服务器上一律直接用项目 Python**：

```bash
/data/TJK/ekg/.venv/bin/python -u scripts/<x>.py ...        # ✅ 标准姿势
/data/TJK/ekg/.venv/bin/pytest                              # ✅
# 非要用 uv 就必须加 --no-sync：
/home/TJK/.local/bin/uv run --no-sync python -u scripts/<x>.py ...
```

`uv pip install -e . --no-deps`（只装本项目、不碰依赖）是安全的，改名/换路径后用它重建 editable 安装。

## 1. 当前状态（2026-07-27）

- **Phase A 已达标**：判别式 `supervised` 抽取器 causal F1 .250 / subevent .213 / temporal .338，
  召回 0.4%→67.5%，`hallucinated=0`。交付 checkpoint = `runs/relations/supervised_maven`。
- **关键路径 = Phase B 真实图闭环**：代码 CPU 全绿，只差「GPU 产 dump → scp 回本地 → 离线分析 →
  回填」。逐步执行照 [`phases/PHASE_B_HANDOFF.md`](phases/PHASE_B_HANDOFF.md)。
- 已有 GPU 证据：SeDGPL 基线、风险感知选边 / 结构编码 A/B、选择性预测 risk-coverage、受控 cross-stage。
  准确数字以 [`TODO.md`](TODO.md) 与 `runs/` 为准。

## 2. 环境

- SSH `ssh gpu-4090`（cpolar 隧道，间歇掉线）；远端根 `/data/TJK/ekg`。
- 项目 Python `/data/TJK/ekg/.venv/bin/python`；uv `/home/TJK/.local/bin/uv`（见 §0 限制）。
- 非交互 SSH 里 `python`/`uv`/`jq`/`rg`/`tmux` 可能不在 PATH；用绝对路径或 `bash -lc`。
- **card 3 故障**需 NVML shim（`nvmlshim/`，remote-only）；card 0/2 常被别人占，优先 card 1，
  但**每次启动前重新 `nvidia-smi` 原子核卡**（检查结果不隔启动复用）。
- 判活：`uv run`/首次加载约 1 分钟才占显存；**SSH 失败 ≠ 任务死亡**，三态 ALIVE / GONE / ssh 失败，
  只有成功 SSH 读到进程 GONE 才算结束。
- 远端仓库外备份 `/data/TJK/ekg-backup-20260727/`（历史 docs 残留等，不进 git）。

## 3. 同步协议

**服务器是 git 仓库**（`main` 跟踪 `origin/main`），日常同步就一条：

```bash
ssh gpu-4090 'bash -lc "cd /data/TJK/ekg && git fetch origin && git reset --hard origin/main"'
```

- 数据/产物/大文件**不在 git**，走 `scp` + 两端 `sha256sum` 核对。
- **禁 `rsync --delete` 与 `git clean -fdx`**（会删 `runs/`、`nvmlshim/`、`data/raw/` 等 remote-only 产物）。
- 目录改名或换路径后：venv 里 console script 的 shebang 与 editable `.pth` 都写死绝对路径，
  **必须修**（做法见 [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md) 环境/工具链节）。

完整三端闭环见 [`PIPELINE.md`](PIPELINE.md)。

## 4. 启动纪律

**GPU 使用无限制，有空就可以去用**（作者授权），无需逐次点头。仍须：

- 本地三件套先全绿；选卡前 `nvidia-smi` 原子核卡；**不挤占他人正在跑的卡**
  （判据：`memory.used ≤ 2500` 且 `utilization ≤ 20`，跳过 card 3）。
- 长任务用 `nohup` / `screen -dmS` + `python -u`，日志重定向 `logs/`，**不得用前台 SSH 承载长任务**。
- 报数如实：升降都报；ssh/工具失败不得伪装成被观察对象的结论。

## 5. v4 GPU 路线

| Phase | GPU | 当前可运行性 | 远端产物 |
|---|---|---|---|
| A 判别式关系抽取 | 重 | ✅ 已达标，checkpoint 在位 | `runs/relations/supervised_maven` + `pair_eval_FINAL.json` |
| B 一致性/修复/风控 | 轻 | ✅ 只差产 dump（见 §6）；修复/CRC 全在本地 CPU | `runs/relations/supervised_dump.jsonl` |
| C 规范节点 | 轻 | ⬜ 依赖 MAVEN-Arg loader/模型实现 | `runs/nodes/*.json` |
| D 事实性/净化 | 轻 | ⬜ MAVEN-FACT 数据就位，代码未实现 | `runs/factuality/*.json` |
| E 闭环/三图传播 | 重 | 🟡 SeDGPL 可复用；真实闭环依赖 A/B/C/D | `runs/cgep/*closedloop*.json` |
| H 多种子 | 重 | ⬜ 只在 A–F 主结果稳定后执行 | 各主表 seed 13/17/42 |

新 phase 的命令必须从实际 CLI `--help` 与配置生成，不能照抄旧命令。

## 6. 当前队首命令：Phase B 真实图 dump

卡空闲时（`<card>` 优先 1、跳 3）：

```bash
cd /data/TJK/ekg
CUDA_VISIBLE_DEVICES=<card> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u scripts/evaluate_relations.py \
  --config configs/relations/supervised_dump.yaml \
  --dump-predictions runs/relations/supervised_dump.jsonl \
  --output runs/relations/supervised_dump_metrics.json \
  > logs/phaseB_dump.log 2>&1
```

卡忙时可用服务器端待机脚本（等空卡**自动开跑**，288×5min≈24h 窗口）——
**当前状态：已修好并验证可用，但按作者指示处于停止态**（`status` 末行 `STOPPED-BY-USER`）。
它会在无人值守时启动 GPU 任务，**只在你确实想要无人值守跑的时候才起**：

```bash
ssh gpu-4090 'bash -lc "cd /data/TJK/ekg && nohup bash runs/phaseB_dump_wait.sh >/dev/null 2>&1 &"'
ssh gpu-4090 'tail -3 /data/TJK/ekg/runs/relations/phaseB_dump.status'   # 轮询进度
```

`status` 末行读法：`DONE rc=0 dump_lines=NNN` = 成功；`DONE rc≠0` = 读 `logs/phaseB_dump.log`；
`TIMEOUT` / `STOPPED-BY-USER` / 进程已死 = 没抢到卡，按需重起。

⚠️ **停它要按三态判活**：脚本卡在 `sleep 300`，SIGTERM 被 bash 挂起到 sleep 返回才生效，kill 后仍会
再写 1–2 条 `WAIT`。判死的决定性依据是 **`status` 文件是否还在按 5min 增长**，不是单次 `pgrep`。

## 7. 仍可复现的 Ch4 基线

```bash
cd /data/TJK/ekg
CUDA_VISIBLE_DEVICES=<空卡> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u scripts/evaluate_cgep.py \
  --dataset maven --predictor sedgpl --model-path <roberta-base-path> --epochs 10 \
  --output runs/cgep/maven_sedgpl.json
```

单折 10ep ≈ 2.5h。仅复现既有 Ch4 基线，**不代表 Phase E 闭环完成**。

## 8. 监控与结束判定

- 进程：用不会匹配探针自身的模式，如 `pgrep -af '[e]valuate_relations'`。
- 显存：`nvidia-smi`；**进程与显存同时符合预期**才判 ALIVE。
- 日志：`tail` 任务自己的日志，不用 SSH 连通性推断训练状态。
- 结束：成功 SSH 确认进程 GONE → 查退出码与产物完整性；**文件存在不等于训练成功**。
- 回传：指标 JSON、必要日志、manifest 定向 scp；checkpoint 只在明确需要时传。
