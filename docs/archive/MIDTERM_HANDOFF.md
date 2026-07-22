# 中期答辩交接文档（Fin-EKG）

> 给新窗口冷启动接手用，避免上下文污染。深层 GPU 运维见 `docs/GPU_RUNBOOK.md`，方法论见 `docs/RL_DESIGN.md` / `docs/RISK_CONTROL_DESIGN.md`。
> 章节：**ch1=事件抽取(SARGE)** · **ch2=关系抽取(GRPO-RLVR)** · **ch3=时序推理(forecasting + 风控)**。

## 0. 现状一句话（截至 2026-06-17）
ch2 全链路跑通并验证（T1 SFT → T2 GRPO-RLVR 三阶段 → T3 评测，GRPO 在核心指标上明显优于 SFT）；**ch3 forecasting 也完成**（icews14 上 path-RL > temporal_gnn > frequency）。extraction→relations→**reasoning** 三阶段弧线均有真结果。**P1 评测口径修复 + P2 ch1 事件统计已完成**（含服务器刷新、闭包分析、新图 3 张）。PPT 图（ch1/ch2/ch3 共 **11 张**）在 `docs/midterm/figures/`。

## 1. 已完成结果（可直接进 PPT）

### ch2 关系抽取：GRPO-RLVR vs SFT（maven_ere valid，710 docs）
> 口径：这是单次 valid 评测结果，不声明统计显著性；主推 coreference / temporal precision / micro 指标，causal 和 subevent 不作为中期正向卖点。

| 指标 | SFT | GRPO-RLVR |
|---|---|---|
| micro F1 | 0.012 | **0.070**（5.7×） |
| micro Precision | 0.332 | **0.647** |
| 正确关系数 tp | 797 | **4710** |
| coreference F1 | 0.060 | **0.611** |
| **CoNLL coref F1** | 0.265 | **0.771** |
| **temporal precision (头条)** | 0.445 | **0.565** |

- 图：`docs/midterm/figures/`
  - `fig2_coref_conll.png` ← **首选头条**（coref CoNLL 0.27→0.77，最干净）
  - `fig1_relation_f1.png`（分类型 F1）、`fig5_micro_prf.png`（P/R/F1）
  - `fig8_temporal_precision.png` ← **temporal 头条**（precision 0.445→0.565，预测排序准确率）
  - `fig4_phase_components.png`（四分量×三阶段，verifiability + 无 reward hacking）
  - `fig3_reward_curve.png`（训练 reward 曲线，三阶段课程）
  - `table_sft_vs_grpo.md`（完整表）、`table_ch2_temporal.md`（temporal 口径说明）
  - 重新生成：`uv run --with matplotlib --with numpy python docs/midterm/make_figures.py`
- ⚠️ **temporal 只报 precision**：MAVEN-ERE temporal gold=传递闭包(n_gold≈110k, 87%)，稀疏 LLM 抽取撞稠密闭包 → recall/F1 不可比。闭包对齐评分（`relation_prf(temporal_closure=True)`）已实现+测试，但**在本模型预测上为 no-op**（预测的 4905 条 BEFORE 边不成链，闭包 +0 边）。未来更稠密/传递结构抽取受益。头条仍主推 coreference(0.61)/CoNLL(0.77)，辅推 temporal precision(0.57 高准确）。
- 产物：新口径评测 `docs/midterm/data/eval_{sft,phase2}_v2.json` + 预测 dump `pred_{sft,phase2}.jsonl`（离线 CPU 重算用 ）。SFT/GRPO adapter 在服务器 `runs/relation_extractor_lora`、`runs/relation_grpo/phase{0,1,2}`。

### ch3 时序推理：path-RL vs baselines（ICEWS14，n_test=14068）
| 方法 | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| frequency baseline | 0.105 | 0.044 | 0.110 | 0.220 |
| temporal_gnn | 0.286 | 0.192 | 0.325 | 0.467 |
| **path-RL（本文 ch3 方法）** | **0.360** | **0.284** | **0.423** | **0.494** |

- **path-RL 全面胜出**（MRR 比 GNN +26%、比 frequency 3.4×），且训练曲线干净上升（mean_reward 0.25→0.35、hit_rate 0.20→0.28，30 epochs）。ch3（faithfulness 塑形 RL）= ch2（verifier-as-reward）的同构延伸。
- 图：`fig6_forecasting_compare.png`（三方 MRR/Hits 对比，头条）、`fig7_pathrl_training.png`（训练曲线）；表 `table_forecasting.md`。
- 产物：`runs/{path_rl_icews14,tgnn_icews14}/*.json`、`runs/freq_icews14.json`。

## 2. 中期优先推进项（按性价比排序）

- ✅ **ch3 forecasting 已完成**（path-RL 胜出，见 §1）。三阶段弧线已立。
- ✅ **P1 修 ch2 评测口径（已完成，代码+测试+实证）** — `relation_prf`（`src/finekg/core/eval/relation.py`）新增 `temporal_closure` 选项 + `core/graph.py:close_pairs` 原语；`evaluate_relations.py` 同时输出 raw/closed + `--dump-predictions`；`scripts/recompute_relation_metrics.py` 吃 dump 离线 CPU 重算。**实证结论**：闭包对齐评分在合成 gold 上正确（R 0.153→1.00），但对两个 LLM 真实预测均为 **no-op**（4905 条 BEFORE 不成链 +0 边）→ temporal 改报 **precision**（SFT 0.445→GRPO 0.565，+27%），recall/F1 标注"不可比（稠密闭包 gold vs 稀疏抽取）"。图 `fig8_temporal_precision.png` + 表 `table_ch2_temporal.md`。闭包自环过滤已有回归测试；本地/远端全套件绿。
- ✅ **P2 ch1 SARGE 抽取统计/图（已完成）** — `docs/midterm/make_ch1_figures.py` → `figures/fig_ch1_{event_types,arg_richness}.png` + `table_ch1_event_extraction.md`；`docs/midterm/make_event_graph_example.py` 先构建 `NetworkX MultiDiGraph`，再生成 `figures/fig9_event_graph_example.{png,graphml,json,cypher}`（新奥股份事件中心 KG：事件节点 + 实体/时间/数值节点 + 角色边 + BEFORE/COREF 事件边，可导入 Neo4j）。数字：**677 事件 / 429 文档 / 13 类金融事件 / 5.02 参数·事件 / time 覆盖 30% / subject 100%（478 主体）**。中文图谱已从 SARGE 参数文本回填 evidence，当前闭包后一致性图 **20683 边**（metadata：edges_raw=498 原始抽取候选，dropped_ungrounded=1）。注意：SARGE prediction 仍不导出 trigger span，因此 trigger evidence 仍为空；中文边是启发式关系图，不应包装成 learned relation extractor 的最终闭环。
- **P3 ch3 风控novelty（C1，verifier-as-risk-controller）** — `scripts/evaluate_calibration.py` + `configs/forecasting/conformal_*.yaml`，做 split vs aci/weighted 的覆盖漂移对比（算力坑见 runbook §T6，建议先小子集）。这是 ch3 超越“纯预测”的卖点。
- **P4 加固（较慢，中期非必须）** — path-RL/GRPO 多 seed(13/17/42) + 消融（`configs/.../ablation_*.yaml`）报 mean±std；更多数据集（icews18/05-15 已在服务器）；下游选股（`scripts/evaluate_downstream_trading.py`）。

## 3. 环境 & 关键坑（务必先读）
- **服务器**：`ssh gpu-4090`，项目根 `/data/TJK/Fin-EKG`，uv 绝对路径 `/home/TJK/.local/bin/uv`，远程命令包 `bash -lc "..."`。`QWEN=/data/TJK/DEE/SARGE/models/Qwen/Qwen3-4B-Instruct-2507`，离线加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
- **🔴 card3 是坏卡**，且 vLLM/NCCL 启动时枚举全部物理卡 → 不带垫片**任何 vLLM/GRPO 进程都崩**。**所有命令前缀加**：
  `CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/TJK/Fin-EKG/nvmlshim:$LD_LIBRARY_PATH`
  垫片产物已在服务器 `nvmlshim/`，源 `scripts/nvml_hide_faulted_gpu.c`；若 `.venv` 重装需重建（命令见 §4 末 + 记忆 `fin-ekg-gpu-ops-gotchas`）。
- **卡分配**：card0 常有**另一用户(LiAo)的任务**——别动；用 card1/card2（CVD=1/2）。`nvidia-smi` 看空闲。
- **无 tmux**（非交互 PATH）→ 用 `nohup ... > log 2>&1 < /dev/null &`。GRPO/eval 加 `python -u`（否则 stdout 块缓冲、看不到 reward）。
- **杀 vLLM**：先 `pkill -9 -f '[v]llm-serve'`（括号防自杀），再 `nvidia-smi --query-compute-apps=pid,process_name` 找**自己的** Fin-EKG 孤儿 engine-core 按 PID 杀（别误杀 LiAo 的）。
- **cpolar 隧道会抖**（ssh exit 255）；nohup 进程不受影响，监视脚本要容忍 ssh 失败。

## 4. 复现/续跑命令（均已验证）
前缀统一：`cd /data/TJK/Fin-EKG && CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=/data/TJK/Fin-EKG/nvmlshim:$LD_LIBRARY_PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1`

```bash
# ── ch2 T2 GRPO（双卡：serve card1 + train card2）──
# serve：
CUDA_VISIBLE_DEVICES=1 <前缀其余> nohup $UV run --extra serve --extra rl trl vllm-serve \
  --model $QWEN --port 8000 --max_model_len 4096 > runs/vllm_serve.log 2>&1 < /dev/null &
# train（等 serve 端口8000 LISTEN 后）：--resume-from-phase N 可续跑
CUDA_VISIBLE_DEVICES=2 <前缀其余> nohup $UV run --extra llm --extra rl python -u \
  scripts/train_relation_grpo.py --config configs/relations/grpo_rlvr.yaml \
  --base-model $QWEN --output runs/relation_grpo > runs/relation_grpo.log 2>&1 < /dev/null &
# ⚠️ 杀 train 必须连 serve 一起重启（NCCL 1:1 配对）；脚本已含 close_communicator 支持多 phase

# ── ch2 T3 评测（单卡；--model/--adapter 是我加的覆盖参数）──
CUDA_VISIBLE_DEVICES=2 <前缀其余> nohup $UV run --extra llm python -u \
  scripts/evaluate_relations.py --config configs/relations/llm_grounded_consistent.yaml \
  --model $QWEN --adapter runs/relation_grpo/phase2 \
  --path data/processed/maven_ere/valid.jsonl --output runs/eval_phase2.json > runs/eval_grpo.log 2>&1 < /dev/null &

# ── ch3 forecasting（icews14；tgnn card1 / path-RL card2 / frequency CPU）──
CUDA_VISIBLE_DEVICES=1 <前缀其余> nohup $UV run --extra gnn python -u \
  scripts/train_forecaster.py --config configs/forecasting/temporal_gnn.yaml \
  --path data/processed/icews14/icews14.tsv --output runs/tgnn_icews14 > runs/tgnn_icews14.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2 <前缀其余> nohup $UV run --extra gnn --extra rl python -u \
  scripts/train_path_rl.py --config configs/forecasting/path_rl.yaml \
  --path data/processed/icews14/icews14.tsv --output runs/path_rl_icews14 > runs/path_rl_icews14.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES= <前缀其余> nohup $UV run --extra gnn python -u \
  scripts/evaluate_forecasting.py --config configs/forecasting/frequency_baseline.yaml \
  --path data/processed/icews14/icews14.tsv --output runs/freq_icews14.json > runs/freq_icews14.log 2>&1 < /dev/null &

# ── nvml 垫片重建（仅 .venv 重装后需要）──
mkdir -p nvmlshim && cp /lib/x86_64-linux-gnu/libnvidia-ml.so.1 nvmlshim/libnvidia-ml-real.so.1
$UV pip install patchelf && .venv/bin/patchelf --set-soname libnvidia-ml-real.so.1 nvmlshim/libnvidia-ml-real.so.1
gcc -O2 -fPIC -shared -o nvmlshim/libnvidia-ml.so.1 scripts/nvml_hide_faulted_gpu.c \
  -Lnvmlshim -Wl,--no-as-needed -l:libnvidia-ml-real.so.1 -Wl,-rpath,$PWD/nvmlshim -ldl
```
（`$UV=/home/TJK/.local/bin/uv`，`$QWEN` 见 §3）

## 5. 我这次改动的代码（已 rsync 到服务器）
- `scripts/train_relation_extractor.py`：加 gradient checkpointing（修 24GB OOM）。
- `scripts/train_relation_grpo.py`：加 `close_communicator()`（修 phase 间 NCCL 死锁）+ `--resume-from-phase N`。
- `scripts/evaluate_relations.py`：加 `--model`/`--adapter` 覆盖 + 进度打印 + `--dump-predictions` + `relation_prf_temporal_closed` 输出。
- `scripts/recompute_relation_metrics.py`（新）：吃预测 dump 在本地 CPU 离线重算 raw/closed/coref（改口径不必再上 GPU）。
- `src/finekg/core/graph.py`：新增 `close_pairs` 原语（pairs 级传递闭包）。
- `src/finekg/core/eval/relation.py`：`relation_prf` 加 `temporal_closure` 参数（默认 False，向后兼容）。
- `tests/core/test_graph.py`：`close_pairs` 链/空/菱形 3 用例。
- `tests/core/test_metrics.py`：闭包修复 recall / 非 strict 不变 / causal 不变 / 默认向后兼容 4 用例。
- `tests/rl/test_configs.py`：同步期望清单（含 `grpo_rlvr_{colocate,hf}.yaml`，修既有红）。
- `docs/midterm/make_figures.py`：加 `fig_temporal_precision` + `write_temporal_table`（精度主导的 ch2 temporal 展示）。
- `docs/midterm/make_ch1_figures.py`（新）：ch1 事件抽取统计图/表。
- `docs/midterm/make_event_graph_example.py`：构建 NetworkX `MultiDiGraph` 事件中心 KG，并导出 PNG/GraphML/JSON/Neo4j Cypher。
- `scripts/nvml_hide_faulted_gpu.c`：NVML 垫片源（绕坏 card3）。
- `configs/relations/grpo_rlvr_{colocate,hf}.yaml`：备选后端（都不如垫片+主配置 vllm_server，仅留档）。
