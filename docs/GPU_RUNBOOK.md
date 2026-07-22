# GPU 任务交接手册（Fin-EKG · 全量 GPU 运行）

> 给新会话/新窗口冷启动接手用。本地非 GPU 工作 + 全部 GPU 路线小样本探测**已完成**；本手册按依赖顺序列出**全量 GPU 任务**的可执行命令、卡分配、产物、验收与坑。
> 关联文档：开发驱动总纲 `docs/SPEC.md`；agent 运维 `AGENTS.md`；工程坑 `docs/ENGINEERING_NOTES.md`。

## 0. 现状一句话
C1–C4 方法论升级 + 数据 + 服务器环境**全部就绪**，本地&服务器全套件**绿（仅 2 个 torch-gated skip；服务器含额外神经测试，数略多于本地）**。4 个 GPU 管线已用小样本验证可跑通（见 §3）。剩下的就是**按 §4 顺序起全量训练/评测**。

## 1. 环境与访问
- **服务器**：`ssh gpu-4090`（cpolar 隧道）。项目根：`/data/TJK/Fin-EKG`。
- **uv**：`/home/TJK/.local/bin/uv`。⚠️ **非交互 ssh 的 PATH 没有 uv**——所有远程命令用 **`bash -lc "..."`** 包裹（登录 shell 才加载 uv PATH）。
- **GPU**：4× RTX 4090（24GB）。**用户要求最多 1–2 卡，勿占满**。执行前 `nvidia-smi` 看空闲卡，**显式 `CUDA_VISIBLE_DEVICES=<id>`**（曾见 CUDA init UserWarning，显式设卡可避免）。
- **依赖（已装，无需 sync）**：torch 2.6.0+cu124 · transformers 4.53.3 · peft 0.15.2 · trl 0.18.2 · vllm 0.8.5 · accelerate 1.5.2 · bitsandbytes 0.49.2 · torch-geometric 2.8.0。
- **Qwen3-4B 基座（离线，完整 7.6G）**：`/data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507`。GRPO/SFT 用 `--base-model`/`--model` 指向它（配置里的 HF id 离线不可解析）。离线跑加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
- **代码/数据同步**（如本地有改动）：`rsync -az --exclude '/data/' --exclude '/runs/' --exclude '/.venv/' ... ./ gpu-4090:/data/TJK/Fin-EKG/`。⚠️ **`--exclude` 必须锚定 `/data/`**（写 `data/` 会误删 `src/finekg/forecasting/data/`，踩过这个坑）。

## 2. 数据与代码现状（服务器已就位）
- `data/processed/`：icews14 / **icews18(468558)** / **icews05-15(461329)** / findkg(144062) / maven_ere / ccks_fin_causal / astock / cmin_cn / **event_graph_zh（677 节点）**。
- movements（下游选股）：`data/processed/cmin_cn/movements.tsv`（270366）、`data/processed/astock/movements.tsv`（14760）。
- 事件图：`data/processed/event_graph_zh/event_graph.json`（677 节点/478 主体/117 个主体≥2 事件；20683 边，raw=498，dropped_ungrounded=1）。

## 3. 路线探测结论（已 de-risk，产物在 `runs/smoke_*`）
| 管线 | 小样本结果 | 全量前置（必读） |
|---|---|---|
| temporal_gnn | ✅ 跑通 `runs/smoke_tgnn` | 无前置，直接放大 |
| path-RL | ✅ 跑通 `runs/smoke_pathrl`（MRR 0.418） | 无前置，直接放大 |
| conformal+GNN（C1） | ✅ 链路通；smoke 因 GNN 近随机+OOV 三 calibrator 饱和 | **需全量 ICEWS18 + 训练好的 GNN**（排全实体空间）才会分化 |
| GRPO-RLVR | ✅ 4/4 步通 `runs/smoke_grpo`；**reward 全 0** | **必须先 SFT 热启**（fresh LoRA 不解析 JSON→0 reward）；全量上 vLLM rollout |

## 4. TODO：全量 GPU 任务（按依赖顺序）
> 约定：均在 `ssh gpu-4090` 后 `cd /data/TJK/Fin-EKG`，命令置于 `bash -lc "..."`。`QWEN=/data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507`。

### T1 — 关系 SFT（GRPO 的前置；单卡）
```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/train_relation_extractor.py \
  --train data/processed/maven_ere/train.jsonl \
  --model "$QWEN" --window-events 24 --epochs 3 \
  --output runs/relation_extractor_lora
```
- 产物：`runs/relation_extractor_lora/`（LoRA adapter）。**路径要与 `configs/relations/grpo_rlvr.yaml` 的 `sft_adapter_path` 一致**（默认就是它）。
- 验收：`print_trainable_parameters` 有输出、loss 下降、adapter 落盘。
- 注意：`--window-events` 必须 = GRPO 的 `curriculum.window_events`（24）。maven_ere train.jsonl 101MB，windowing 较慢但可接受。

### T2 — 关系 GRPO-RLVR（依赖 T1；推荐 2 卡 vLLM）
**双卡（推荐，vLLM rollout 提速）**：终端 A 起 vLLM serve，终端 B 训练。
```bash
# 终端A（serve，1 卡）
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 uv run --extra serve --extra rl trl vllm-serve --model "$QWEN" --port 8000
# 终端B（train，另 1 卡）
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --extra llm --extra rl \
  python scripts/train_relation_grpo.py --config configs/relations/grpo_rlvr.yaml \
  --base-model "$QWEN" --output runs/relation_grpo
```
**单卡兜底**：改 `configs/relations/grpo_rlvr.yaml` 的 `rollout.backend: vllm_colocate`（或 `hf`），`grpo.num_generations: 4`、`max_completion_length: 512`，单卡 `CUDA_VISIBLE_DEVICES=1` 只跑终端 B。
- 产物：`runs/relation_grpo/phase{0,1,2}/`（课程各阶段 adapter）+ `reward_curve.json` + `summary.json`。
- 验收：**四分量奖励（format/grounding/consistency/task_f1）同向上升、reward_std>0**（smoke 里全 0 是因 fresh LoRA；T1 之后应非 0）。盯 `reward_curve.json` 防 reward hacking（某分量塌、总分涨）。
- 评测：把 `summary.json` 的 `adapter_path` 填入 `configs/relations/llm_grounded_consistent.yaml` 的 `adapter_path`，再跑 `scripts/evaluate_relations.py`（可选叠加 `crc_edge_admission.yaml` 的 CRC 准入）。

### T3 — 关系评测（依赖 T2；CPU 即可，LLM 抽取需 1 卡）
```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate_relations.py \
  --config configs/relations/llm_grounded_consistent.yaml --path data/processed/maven_ere/valid.jsonl
# CRC 风险可控图（C2）：
uv run python scripts/evaluate_relations.py --config configs/relations/crc_edge_admission.yaml \
  --path data/processed/maven_ere/valid.jsonl   # 看 admission.fnr ≤ alpha
```

### T4 — 预测：temporal_gnn 全量（单卡）
```bash
for DS in icews14 icews18 findkg; do
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_forecaster.py \
    --config configs/forecasting/temporal_gnn.yaml \
    --path data/processed/$DS/$DS.tsv --output runs/tgnn_$DS
done
```
- 产物：`runs/tgnn_<ds>/temporal_gnn.pt` + `metrics.json`（MRR/Hits）。

### T5 — 预测：path-RL 全量（单卡）
```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_path_rl.py \
  --config configs/forecasting/path_rl.yaml \
  --path data/processed/icews14/icews14.tsv --output runs/path_rl_icews14
# C4 反事实忠实度：在 path_rl.yaml 的 path_rl_train 增 faithfulness_proxy: temporal_gnn
```

### T6 — C1 头号漂移实验（依赖 T4 的 GNN 思路；C1 核心结果）
做一个 **GNN-reasoner 全量 ICEWS18** 的 conformal 配置（参照 `configs/forecasting/conformal_gnn_smoke.yaml`，把 data.path 改 `icews18/icews18.tsv`、`epochs` 调大、`coverage_window: 200`），然后：
```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate_calibration.py \
  --config configs/forecasting/conformal_gnn_icews18.yaml --calibrators split aci weighted crc
```
- 期望：**split 的 `coverage_drift_gap` 明显大于 aci/weighted**（漂移下 split 覆盖率失真、自适应稳住 1−α）。
- ⚠️ 算力坑：`evaluate_calibration` **每个 calibrator 重跑一遍 pipeline（含重训 GNN）**；全量 ICEWS18 测试流很大。建议先用时间窗子集或较小 `test_ratio` 出图，或后续加"先缓存 GNN 排名、再跑各 calibrator"的优化（目前未实现）。

### T7 — 多种子 + 消融（论文表）
- seeds 13/17/42 重跑 T2/T5，报 mean±std。
- 消融配置已就绪：`configs/relations/ablation_grpo_no_{grounding,consistency}.yaml`、`configs/forecasting/ablation_path_no_{shaping,faithfulness,warmstart}.yaml`、`configs/relations/ablation_no_edge_admission.yaml`、`conformal_{split,weighted,crc}.yaml`。

### T8 — 下游事件驱动选股（CPU 为主）
```bash
uv run python scripts/evaluate_downstream_trading.py --config configs/downstream/trading_selective.yaml
```
- ⚠️ 前置：`trading_selective.yaml` 的 `graph_path` 指向 677 节点图、`movements_path` 指向 cmin/astock movements；但完整 join 仍受限于事件图公司覆盖（见 T9）。先确认 movements 里的公司名与事件图 subject 有交集。

### T9 —（可选 GPU）把事件图做密：SARGE 跑 CMIN/Astock 新闻
当前 `event_graph_zh` 由 SARGE 在 DuEE-Fin-dev500 上的**现成**预测建成（677 节点，参数 evidence 已从原文回填，trigger span 仍缺失）。要得到"公司随时间多事件"的真预测图：在 **SARGE 仓库**（`~/myProjects/masterProjects/DEE/SARGE`，服务器 `/data/TJK/DEE/SARGE`）对 CMIN/Astock 新闻跑 SARGE 推理（GPU），再 `scripts/sarge_to_event_nodes.py --source-docs ...` → `scripts/build_event_graph.py`。属较大独立工作。

## 5. 验证/验收总清单
- [ ] T1 adapter 落盘且 loss 收敛。
- [ ] T2 四分量 reward 同向上升、std>0、无 reward hacking；3 阶段 adapter + curve 落盘。
- [ ] T3 relation F1 ≥ SFT 基线；CRC 准入 fnr ≤ alpha。
- [ ] T4/T5 MRR/Hits 超 frequency 基线（`evaluate_forecasting.py --config frequency_baseline.yaml`）。
- [ ] T6 split drift_gap > aci/weighted drift_gap（C1 核心论点）。
- [ ] 跑前后 `uv run pytest`（应全绿、仅 2 个 torch-gated skip）确认未回归。

## 6. 已知坑 & 硬规则
- **远程 in-place 编辑被 hook 拦**：勿在服务器 `sed -i` 改 tracked 文件；本地改→rsync。临时配置写 `/tmp` 或本地新建后同步。
- **rsync 排除要锚定**：`--exclude '/data/'`（非 `data/`）。
- **hf rollout 慢**（smoke 16s/步）→ 全量用 vLLM。
- **GRPO 冷启动 reward=0** 是没 SFT，不是 bug。
- **C1 smoke 三 calibrator 同值** 是召回封顶（小数据/弱模型），全量+GNN 才分化。
- **建图勿用 `--from-sarge`**：该分支走 `core.io.event_nodes_from_sarge`，它读的键是 `evidence`（转换脚本写的是 `argument_evidence`）且不透传 `metadata`。用了它会把刚回填的证据清空，边被 `require_evidence` 全丢（Astock 曾因此 15623 节点 / 0 条边）。用 `build_event_graph.py --nodes`。
- **ICEWS14 口径**：`val_ratio/test_ratio: 0.15` 是**非 canonical** 重切（n_test=14068），不可与 RE-GCN 0.42 比。用 `val_timestamps: 30 / test_timestamps: 31`（= 304/30/31 天 = 74845/8514/7371，n_test=7371），配置见 `configs/forecasting/re_gcn_canonical.yaml`。
- **卡会被别的用户抢**：起任务前 `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i <id>` 原子检查，`free < 8000 MiB` 就别起。
- **硬规则**（改代码时）：命名无章节标记；registry 式扩展；CPU/GPU 分层（torch lazy import）；blackboard 复制不可变；**schema 零新增字段**（复用 `Prediction.{coverage_set,abstained,faithfulness}`、`RelationEdge.{confidence,faithfulness,admitted}`）。

## 7. 速查
| 用途 | 路径 |
|---|---|
| 项目根（服务器） | `/data/TJK/Fin-EKG` |
| Qwen3-4B 基座 | `/data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507` |
| roberta-base（CGEP / SeDGPL 复现锚） | `/data/TJK/Fin-EKG/models/roberta-base`（safetensors，sha256 `5bde1d28…`） |
| CGEP-ESC 官方数据 | `/data/TJK/Fin-EKG/data/raw/sedgpl_esc/ESCSubWoRe.npy`（sha256 `8ec791fb…`；**外部 pickle，须用白名单 Unpickler 加载**，只放行 `numpy.dtype` / `numpy.ndarray` / `numpy.core.multiarray._reconstruct`） |
| CGEP-MAVEN 重建 | `scripts/build_cgep.py --split train+valid --report-stats`（纯 CPU，约 20 s） |
| CGEP 基线评测 | `scripts/evaluate_cgep.py --dataset {esc,maven} --predictor {random,frequency}`（纯 CPU） |
| CGEP 单步计时 | `scripts/profile_cgep_step.py --model-path /data/TJK/Fin-EKG/models/roberta-base` |

**CGEP 训练成本（2026-07-10 在 card 1 实测，非估算）**：batch_size=1、seq_len=200、roberta-base ×3。
ESC 均值形状（10 句/实例）**113.6 ms/实例**，峰值 **7.09 GiB**；ESC p90（20 句）142.8 ms；
MAVEN 形状（12 句、512 候选）116.1 ms/实例。
→ ESC 5 折 × 10 epoch ≈ **1.5 h**；MAVEN 10 epoch ≈ **2.45 h**。句子数翻倍只涨 26 %，
因为每个实例的所有句子是**一次 batched 前向**，不是逐句串行。显存充裕，可考虑真批处理再提速。
| SFT / GRPO / 评测 | `scripts/train_relation_extractor.py` · `train_relation_grpo.py`（`--base-model`）· `evaluate_relations.py` |
| 预测训练 | `scripts/train_forecaster.py` · `train_path_rl.py` |
| C1 漂移实验 | `scripts/evaluate_calibration.py` + `configs/forecasting/conformal_*.yaml` |
| 下游选股 | `scripts/evaluate_downstream_trading.py` + `configs/downstream/trading_selective.yaml` |
| movement 标签生成 | `scripts/build_movement_labels.py --dataset {cmin_cn,astock}` |
| 事件图构建 | `scripts/sarge_to_event_nodes.py --source-docs …` → `scripts/build_event_graph.py --nodes …`（**勿用 `--from-sarge`**，见 §6） |
| ICEWS18/05-15 下载 | `scripts/download_icews_extra.py` |
