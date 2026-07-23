# Fin-EKG · 三端协作流水线（PIPELINE）

> **本地开发 → GitHub 中转 → 服务器 GPU 执行 → 回传结果** 的标准闭环。一句话铁律：
> **代码走 git，数据 / 产物 / 大文件走 scp。** 设计以 [`SPEC.md`](SPEC.md) 为准；服务器内运维细节见
> [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md)；实时状态见 [`TODO.md`](TODO.md)。

## 1. 三端角色

| 端 | 位置 | 职责 | git 角色 |
|---|---|---|---|
| **本地** | WSL `/home/tjk/myProjects/masterProjects/Fin-EKG` | 编辑代码 + CPU 校验（pytest / ruff / smoke） | git 仓库，**唯一编辑端** |
| **GitHub** | `github.com/emmmdty/dee-fin`（**PUBLIC**，分支 `main`） | 代码唯一中转中心（single source of truth） | remote `origin` |
| **服务器** | `gpu-4090:/data/TJK/Fin-EKG` | GPU 训练 / 推理执行 | git 仓库，**只拉取不编辑** |

## 2. 什么走 git，什么不走（铁律）

- ✅ **走 git（三端必一致）**：`src/` `tests/` `scripts/` `configs/` `docs/` `pyproject.toml`
  `uv.lock` `external/sarge/` `data/fixtures/` `data/raw/DATA_PROVENANCE.md`
- ❌ **不走 git（`.gitignore` 排除，各端本地 / scp）**：
  - `data/raw/*` 数据 → 服务器已就位或单独 scp
  - `runs/` 实验产物、`logs/` 训练日志 → 服务器生成，回传走 scp
  - `models/` `outputs/` `*.ckpt` `*.safetensors` 大文件；`.venv/`（各端 `uv sync` 重建）
  - `nvmlshim/` → 服务器 remote-only（card 3 NVML shim）
- ⚠️ **服务器 `git pull` 拿不到数据**（数据不在 git）：首次 / 更新数据用 scp（见 §4 step 3）。

## 3. 远端首次 git 化（一次性，**待 ssh 恢复后执行**）

远端 `/data/TJK/Fin-EKG` 当前**不是 git 仓库**、且有 remote-only 产物（`runs/` `nvmlshim/`）。
用**原地 `git init`**（产物留原位，不迁移、不删除）：

```bash
ssh gpu-4090
cd /data/TJK/Fin-EKG
git init && git branch -M main
git remote add origin https://github.com/emmmdty/dee-fin.git   # PUBLIC，免 token
git fetch origin
git status            # ← 先看：确认 runs/ data/ nvmlshim/ 均在 ignored/untracked（不会被动）
git reset --hard origin/main   # tracked 代码 = GitHub；untracked 产物全部保留原位
git log --oneline -1           # 应 == 本地 / GitHub 的 HEAD
```

- `reset --hard` **只重置 tracked 代码**；`.gitignore` 覆盖的 `runs/ data/ models/ outputs/ nvmlshim/` 不受影响。
- **永不 `git clean -fdx`**（会删 ignored 的 `runs/ data/ nvmlshim/`）。
- reset 前务必 `git status` 核对：服务器约定「只执行不编辑」，正常不应有「未同步到 GitHub 的 tracked 改动」被覆盖。
- git 可能不在非交互 ssh PATH → 用绝对路径或 `bash -lc`。
- 仓库若转 private：远端 remote 换 `https://<PAT>@github.com/emmmdty/dee-fin.git` 或配 deploy key。

## 4. 标准迭代闭环

1. **本地开发 → 校验全绿**
   ```bash
   uv run pytest && uv run ruff check src tests scripts && uv run finekg-smoke
   ```
2. **本地 → GitHub**
   ```bash
   git add -A && git commit -m "<msg>" && git push origin main
   ```
3. **服务器拉取**（ssh 恢复后）
   ```bash
   ssh gpu-4090
   cd /data/TJK/Fin-EKG && git fetch origin && git reset --hard origin/main
   ```
   - 数据首次 / 更新（不在 git）：本地
     `scp -r data/raw/<x> gpu-4090:/data/TJK/Fin-EKG/data/raw/`，两端 `sha256sum` 核对。
4. **服务器 GPU 执行**（选卡前 `nvidia-smi`，优先 card 1；长跑用 `screen -dmS` / `nohup` + `python -u`）
   ```bash
   cd /data/TJK/Fin-EKG
   CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     /home/TJK/.local/bin/uv run --extra llm python -u scripts/<x>.py ... \
     > logs/<x>.log 2>&1 &
   ```
5. **出问题 → 回传本地**（日志 + 结果，在本地执行）
   ```bash
   scp gpu-4090:/data/TJK/Fin-EKG/logs/<x>.log logs/
   scp gpu-4090:/data/TJK/Fin-EKG/runs/<x>.json runs/
   ```
   两端 `sha256sum` 核对；产物落 `runs/`，结论写入 `TODO.md`（如实：降就说降）。
6. **本地修改 → 回到 step 1**（改代码、push、服务器 pull、再实验）。

## 5. 服务器判活（ssh 间歇掉线）

cpolar 隧道间歇性掉线（`Connection reset by peer` / `ConnectTimeout`）。**ssh 失败 ≠ 远端进程死亡**：
三态判活 **ALIVE / GONE / ssh 失败**，只有成功 ssh 读到进程 GONE 才算结束。ssh 失败时**不得**把「连不上」
当成「任务结束」或「任务失败」的结论（CLAUDE.md 硬约束「如实报告」）。
