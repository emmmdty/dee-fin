# PHASE B · 交接（真实 predicted 图闭环最后一公里）

> **给新会话**：`/clear` 后只读本文件 + 自动载入的 `CLAUDE.md` + [`../SPEC.md`](../SPEC.md) 即可执行。
> Phase B 的**代码（W1–W4）已完成、CPU 全绿、已提交推送**；**只剩「真实图 dump → 离线分析 → 回填 → 提交」**。
> 原始 phase 契约见 [`PHASE_B_ch2_consistency_repair_riskadmit.md`](PHASE_B_ch2_consistency_repair_riskadmit.md)。

## Goal（要的结果）

用 **Phase A 判别式抽取器的真实 predicted 图**，跑出并**如实回填**下面这组三档轨迹
（`raw → repaired → repaired+admitted`）：violation/cycle 前后、分层 FNR、准入集大小、ECG 可重建率 R1/R2。
这是**首次用真实 predicted ECG 做闭环**，解堵 Ch4（Phase E）headline。

## 现状（2026-07-27 更新）

- **代码已交付**：`771d5c3`（W1–W4）+ `501e798`（docs 回填），后经 `3390363`（改名 ekg + 移除
  SARGE/Phase G）与 `2e6703b`（一致性收口 + 环境修复）。服务器已对齐 `origin/main`。
- **两端校验基线**（计数不同不是回归）：本地无 torch = `241 passed / 12 skipped`；
  服务器有 torch = `252 passed / 1 skipped`。ruff 0、`ekg-smoke` OK。
- **合成 dump 已验证**（CPU，注入因果环）：`causal_cycle 1→0`、`dropped=1`；**R1 持平 1.0、R2 f1 0→1.0**。
- **环境已修复（2026-07-27）**——此前服务器**根本跑不了任何脚本**：目录改名后 editable 安装仍指向
  `/data/TJK/Fin-EKG/src`（`import ekg` 与 `import finekg` 双双失败）、56 个 console script shebang 全废；
  待机脚本自己也 `cd` 到旧路径 → 每次 exit 2，**从 2026-07-25 起就没真正等过卡**。均已修好并验证
  （`import ekg` OK、torch 2.6.0+cu124 完好、checkpoint 与 710 篇 gold 在位、`--help` 通）。
- **真实图 dump 仍未出，且当前没有在等的 GPU 任务**：2026-07-27 全天 4 卡被他人占满
  （15–18GB @ 87–100%），未硬塞。待机脚本已修好并验证可用，但**按作者指示已主动停掉**
  （`status` 末行 `STOPPED-BY-USER 2026-07-27 12:32`），服务器上无我们的进程。
  接手时**先 `nvidia-smi` 看有没有空卡**：有空卡直接走 Step 1；没有再决定要不要起待机
  （`ssh gpu-4090 'bash -lc "cd /data/TJK/ekg && nohup bash runs/phaseB_dump_wait.sh >/dev/null 2>&1 &"'`，
  288×5min≈24h 窗口，会在抢到空卡时**自动开跑**——别在不想无人值守跑 GPU 时起它）。
  ⚠️ kill 待机脚本时注意：它卡在 `sleep 300`，SIGTERM 会被 bash 挂起到 sleep 返回才生效，
  判死要看 `status` 文件**是否还在按 5min 增长**，别只看一次 `pgrep`。

## 依赖 · 产物

- **前置产物**（服务器 `/data/TJK/ekg/`，均已在位）：checkpoint `runs/relations/supervised_maven`、
  gold valid `data/processed/maven_ere/valid.jsonl`（本地同路径也在，710 篇）。
- **产出**：`runs/relations/supervised_dump.jsonl`（原始边 dump）→ `runs/relations/consistency_repair_supervised.json`（离线分析）。
- **数据/产物走 scp + sha256 双端核，不进 git**（`CLAUDE.md`/`PIPELINE.md`）。

## Steps（照这个顺序执行）

### 0) 先查服务器待机脚本状态

```bash
ssh gpu-4090 'bash -lc "tail -5 /data/TJK/ekg/runs/relations/phaseB_dump.status; \
  echo --alive--; pgrep -af phaseB_dump_wait.sh | grep -v bash.-lc; \
  echo --smi--; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"'
```

按末行分三种情况：

- **`DONE rc=0 dump_lines=NNN`** → 跳到 **Step 2**（dump 已生成在服务器）。
- **`DONE rc=<非0>`** → dump 跑挂了，读 `ssh gpu-4090 'tail -40 /data/TJK/ekg/logs/phaseB_dump.log'` 定位，修好后回到 Step 1。
- **`TIMEOUT ...` 或进程已死 / 末行还是 `WAIT`** → 待机没抢到卡。看 `--smi--`：**有空卡**（某卡 `used≤2500 且 util≤20`，跳过 card3）就走 **Step 1** 手动跑；**仍全占**就重起待机脚本
  `ssh gpu-4090 'bash -lc "cd /data/TJK/ekg && nohup bash runs/phaseB_dump_wait.sh >/dev/null 2>&1 &"'` 然后按 §GPU 纪律等卡、别硬塞。

### 1) 产 dump（服务器 GPU，有空卡时；`<card>` 选 1 优先，跳 3）

```bash
ssh gpu-4090 'bash -lc "cd /data/TJK/ekg && CUDA_VISIBLE_DEVICES=<card> \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u scripts/evaluate_relations.py \
  --config configs/relations/supervised_dump.yaml \
  --dump-predictions runs/relations/supervised_dump.jsonl \
  --output runs/relations/supervised_dump_metrics.json"'
```

⛔ **必须 `.venv/bin/python`，不要 `uv run`** —— 服务器装的是全套 extras，`uv run` 会按默认 extras
把 torch/vllm/trl 卸掉（连 `--extra llm` 都卸 109 个包）。详见 [`../GPU_RUNBOOK.md`](../GPU_RUNBOOK.md) §0。

`supervised_dump.yaml` = supervised 抽取器 + checkpoint + **`consistency: identity` + 无准入** → 产**原始 scored+grounded 边**（修复/准入全走离线，checkpoint 不下本地）。

### 2) scp dump 回本地 + 核对

```bash
scp gpu-4090:/data/TJK/ekg/runs/relations/supervised_dump.jsonl runs/relations/
scp gpu-4090:/data/TJK/ekg/runs/relations/supervised_dump_metrics.json runs/relations/
# sha256 双端核：
ssh gpu-4090 'sha256sum /data/TJK/ekg/runs/relations/supervised_dump.jsonl' ; \
  sha256sum runs/relations/supervised_dump.jsonl
```

### 3) 离线分析（本地 CPU，无需 GPU/torch）

```bash
uv run python scripts/consistency_repair_report.py \
  --dump runs/relations/supervised_dump.jsonl \
  --gold data/processed/maven_ere/valid.jsonl \
  --alpha 0.2 --cal-ratio 0.3
# → 写 runs/relations/consistency_repair_supervised.json（含 consistency/reconstruction 三档、
#   admission 分层 FNR、tau、repair_trace_totals/sample、reachable_flags）
```

`analyze()` 已复刻在线 `evaluate_relations` 的 **repair∘admit 固定映射**：cal 分片上用**修复后**图读 `gold_edge_scores` 校准 CRC，全部指标只报**held-out test 分片**（apples-to-apples，非缩样假象）。可扫 `--alpha 0.1/0.2`。

### 4) 回填 + 提交

把 `consistency_repair_supervised.json` 的真实三档数字，**替换** [`../TODO.md`](../TODO.md) 与
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) 里「Phase B 实施」段落现有的 **PENDING / 合成 dump** 占位，
填成真实 `raw→repaired→repaired+admitted` 轨迹表（violation/cycle、分层 FNR、准入集大小、R1/R2）。
`git add` 代码/文档、`scp` 产物**不进 git**；commit + `git push origin main`（作者已授权本任务提交）。

## 怎么读结果（如实，PHASE_B 止损口径）

- **R1（query 边可达率，召回义）= CS-CRP 桥**：CRC 准入受 α_edge 约束，**R1 可能持平/略降**，属正常，别当负结果掩盖。
- **R2（query 边保真，precision 义）= 修复增益该出现的地方**：破矛盾环/去矛盾边后 R2 的 F1 可升。
- **主张修复有效 = violation/cycle↓ + R2↑**；若 R2 增益也微弱 → 按 PHASE_B 止损**退 consistency-aware reranking / constrained decoding，仍成章，不换指标**。
- **分层 FNR 表述**（SPEC §5.5）：交换性 + 固定后处理下的**边际期望 FNR**；只报边际/分族/doc-macro 三层让差距说话，**不写「每篇/每类都保证」**；风险目标=下游可达性损失，命名避与 SCRC 2512.12844 撞。

## Constraints / GPU 纪律

- **原子核卡、卡空闲才跑、不挤占他人训练**；`nvidia-smi` 选卡；**card3 故障需 NVML shim（跳过）、优先 card1**。
- 长跑用 `nohup`/`screen -dmS` + `python -u`、输出重定向 `logs/`；非交互 ssh 用绝对 `uv` 路径 / `bash -lc`。
- **禁 `git clean -fdx` / `rsync --delete`**（会删 `runs/`）；代码走 git，数据/产物走 scp。
- 报数**如实**（升降都报；ssh/工具失败不得伪装成结论）。**专利/论文写作不在计划范围**。

## Done when（验收）

- [ ] `runs/relations/supervised_dump.jsonl` 生成（710 篇量级）、sha256 双端一致。
- [ ] `runs/relations/consistency_repair_supervised.json` 产出，三档轨迹齐（consistency/admission/reconstruction + tau + repair_trace）。
- [ ] `docs/TODO.md` + `docs/EXPERIMENTS.md`「Phase B 实施」段真实数字**替换** PENDING（升降都如实）。
- [ ] 本地 `uv run pytest && uv run ruff check src tests scripts && uv run ekg-smoke` 仍全绿（只增不改）。
- [ ] commit + push `origin/main`；dump/产物未进 git。

## 交付物地图（W1–W4 已落地，供排查）

| 文件 | 角色 |
|---|---|
| `src/ekg/relations/consistency/__init__.py` | W1 `RepairEdit`/`RepairTrace`/`solve_with_trace`（`solve()` 逐字节不变=测试锁） |
| `src/ekg/relations/admission.py` | W2 `stratified_admission_report`（边际/分族/doc-macro FNR + 准入集大小） |
| `src/ekg/succession/reconstruction.py` | W3 `reconstruction_report`/`ecg_reachable_flags`（R1 可达=CS-CRP 桥, R2 保真） |
| `scripts/consistency_repair_report.py` | W4 `analyze()` 离线编排 + CLI |
| `configs/relations/supervised_dump.yaml` | W4 原始边 dump producer（supervised+identity+无准入） |
| `tests/{relations/test_consistency_repair,relations/test_admission,succession/test_reconstruction,scripts/test_consistency_repair_report}.py` | W1–W4 测试 |
| 服务器 `runs/phaseB_dump_wait.sh` + `runs/relations/phaseB_dump.status` | 待机脚本 + 进度（非 git，scp 来的） |
