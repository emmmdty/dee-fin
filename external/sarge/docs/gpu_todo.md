# GPU 待办任务清单

> 最后更新：2026-05-23 13:09 UTC+8
> 目的：记录当前服务器 GPU 任务、已完成实验资产和下一步可执行队列。本文只描述状态和命令入口，不把 running 任务写入主结果表。
> 服务器快照来源：`gpu-4090:/data/TJK/DEE/SARGE/` 只读查询。未启动、停止或 kill 任何任务。

---

## 当前服务器状态

| 项 | 值 |
|---|---|
| 本地项目根 | `/home/tjk/myProjects/masterProjects/DEE/SARGE/` |
| 服务器项目根 | `/data/TJK/DEE/SARGE/` |
| 本地 Python | `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python` |
| 服务器 Python | `/data/TJK/envs/sarge_vllm_full/bin/python` |
| 服务器分支 / HEAD | `main` / `1849d7a55d9701148b4f4b83509c947d11c93e6d` |
| 同步原则 | additive sync only；不用 `rsync --delete` |
| GPU 规则 | 优先空闲 GPU；kill 前必须确认 owner 是 `TJK`；禁止 kill 其他用户任务 |

只读刷新命令：

```bash
ssh gpu-4090 'cd /data/TJK/DEE/SARGE && git rev-parse HEAD && git status --short'
ssh gpu-4090 'ps -eo user,pid,etime,cmd | grep -E "SARGE|sarge_|infer_checkpoint|train_sft|postprocess_lrd_eval" | grep -v grep || true'
ssh gpu-4090 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

GPU 快照：

| GPU | 当前占用 | 状态判断 | 调度含义 |
|---:|---|---|---|
| 0 | `2653MB`, util `0%`, non-TJK `Zhhy` eval process | 他人任务占用 | 不 kill；避免默认抢占 |
| 1 | `40MB`, util `0%` | 空闲 | 可作为后续短任务候选；启动前仍需重新确认 |
| 2 | `3949MB`, util `15%`, TJK 非 SARGE CCKS 推理 | 忙 | 不叠加 SARGE 新任务，避免干扰同用户任务 |
| 3 | `40MB`, util `0%` | 空闲 | 可作为后续短任务候选；启动前仍需重新确认 |

注意：2026-05-23 08:45 UTC+8 的 SARGE 进程查询无输出，说明当前没有 SARGE 训练/推理/eval 任务在跑。后续启动 GPU 任务前仍需重新给出 exact command、cwd 和 expected outputs。

---

## 活动任务

| 任务 | GPU | 状态 | Log | 备注 |
|---|---:|---|---|---|
| SARGE | - | none | - | 2026-05-23 08:45 UTC+8 只读查询未见 SARGE 任务 |
| TJK 非 SARGE CCKS 推理 | 2 | running | `/data/TJK/competition/ccks2026-exp5/` | PID `3979305`；同卡共享，SARGE 不应再叠加 GPU2 |
| non-TJK eval job | 0 | running | n/a | owner `Zhhy` PID `3976622`；共享服务器任务，禁止 kill |

活动任务只进入状态表。只有存在完整 `eval/eval_{legacy_doc2edag,unified_strict,docfee_official}.json` 后，才允许进入结果表或 `paper/exp/data/asset_registry.json` 的 completed/main 条目。

---

## 已完成主资产

| 数据集 | Split | Seed | 资产 | Legacy-FS F1 | 处理 |
|---|---|---:|---|---:|---|
| ChFinAnn-Doc2EDAG | test | 13 | HF-4bin + LoRA, k=1 greedy | 0.8603 | 当前主结果 |
| ChFinAnn-Doc2EDAG | test | 17 | HF-4bin + LoRA, k=1 greedy | 0.8536 | 小型 JSON 快照已拉取；seed-extension diagnostic |
| ChFinAnn-Doc2EDAG | test | 42 | HF-4bin + LoRA, k=1 greedy | 0.8533 | 小型 JSON 快照已拉取；seed-extension diagnostic；ChFinAnn seeds 13/17/42 mean±std 已生成 |
| DuEE-Fin-dev500 | test | 13 | HF-4bin + LoRA, k=1 greedy, no-LRD | 0.7796 | 当前主结果 |
| DuEE-Fin-dev500 | test | 17 | HF-4bin + LoRA, k=1 greedy, no-LRD | 0.7872 | 小型 JSON 快照已拉取；registry/表已更新 |
| DuEE-Fin-dev500 | test | 42 | HF-4bin + LoRA, k=1 greedy, no-LRD | 0.7828 | 小型 JSON 快照已拉取；registry/表已更新 |
| ChFinAnn-Doc2EDAG | train | 17 | HF-4bin LoRA ep2 adapter | n/a | 训练 JSON 快照已拉取；registry/表已更新；HF test eval 已同步 |
| ChFinAnn-Doc2EDAG | train | 42 | HF-4bin LoRA ep2 adapter | n/a | 训练 JSON 快照已拉取；seed42 HF test eval 已同步 |

权威索引：`paper/exp/data/asset_registry.json`。小型证据快照：`paper/exp/data/run_snapshots/`。主表汇总：`paper/exp/seed13_summary.md` 和 `docs/exp_result.md`。

非 GPU P0 完成项：

| 完成时间 | 任务 | 本地产物 |
|---|---|---|
| 2026-05-22 09:30 UTC+8 | 拉取 DuEE-Fin seed17/42 HF test 小型 JSON 快照，并将 running 资产替换为 completed diagnostic seed-extension 资产 | `paper/exp/data/run_snapshots/dueefin_test_seed17_hf4bin_k1_no_lrd/`, `paper/exp/data/run_snapshots/dueefin_test_seed42_hf4bin_k1_no_lrd/`, `paper/exp/data/asset_registry.json` |
| 2026-05-22 09:30 UTC+8 | 生成 DuEE-Fin seed13/17/42 mean±std 草稿表 | `paper/exp/tables/12_dueefin_seed_stability.md` |
| 2026-05-22 09:30 UTC+8 | 拉取 ChFinAnn seed17 training 小型 JSON 快照，并将 seed17 train 标为 completed、seed42 train 标为 status-only training asset | `paper/exp/data/run_snapshots/chfinann_train_seed17/`, `paper/exp/data/asset_registry.json`, `paper/exp/seed13_summary.md` |
| 2026-05-22 09:40 UTC+8 | 拉取 DuEE-Fin seed13/17/42 vLLM backend 小型 JSON 快照，补 registry、自动表和 backend seed-stability 表 | `paper/exp/data/run_snapshots/dueefin_test_seed13_vllm_bf16_k1/`, `paper/exp/data/run_snapshots/dueefin_test_seed17_vllm_bf16_k1/`, `paper/exp/data/run_snapshots/dueefin_test_seed42_vllm_bf16_k1/`, `paper/exp/tables/13_dueefin_backend_seed_stability.md` |
| 2026-05-23 01:00 UTC+8 | 拉取 ChFinAnn seed17 HF test、ChFinAnn seed42 training、DuEE-Fin vLLM 模块快筛、vLLM mechanism probes、HF no_surface_memory 小型 JSON 快照 | `paper/exp/data/run_snapshots/`, `paper/exp/data/asset_registry.json`, `paper/exp/tables/14_dueefin_prompt_module_ablation.md`, `paper/exp/tables/15_dueefin_vllm_mechanism_probes.md` |
| 2026-05-23 08:45 UTC+8 | 拉取并归档 ChFinAnn seed42 HF test 与 DuEE-Fin HF no_slot_plan 小型 JSON 快照，替换 status-only 资产并生成 ChFinAnn seed stability 表 | `paper/exp/data/run_snapshots/chfinann_test_seed42_hf4bin_k1/`, `paper/exp/data/run_snapshots/dueefin_test_seed13_hf4bin_ablation_no_slot_plan/`, `paper/exp/tables/16_chfinann_seed_stability.md` |
| 2026-05-23 13:09 UTC+8 | 拉取 ChFinAnn vLLM BF16 gmem=0.80 full/no_surface_memory/no_slot_plan 消融小型 JSON 快照，并补充模块归因风险分析 | `paper/exp/data/run_snapshots/chfinann_test_seed13_vllm_ablation_*_mem080/`, `paper/exp/tables/17_chfinann_vllm_module_ablation.md`, `docs/sft_module_risk_analysis_20260523.md` |

---

## 已完成消融和诊断资产

| 数据集 | Split | Seed | 类型 | 资产 | Legacy-FS F1 | 结论 |
|---|---|---:|---|---|---:|---|
| ChFinAnn-Doc2EDAG | test | 13 | backend | vLLM BF16 + LoRA, k=1 | 0.8547 | 低于 HF-4bin 主结果 |
| ChFinAnn-Doc2EDAG | test | 13 | module-fast-screen | vLLM BF16 full, gmem=0.80 | 0.8547 | ChFinAnn 消融 control；只作诊断，不替代 HF 主路径 |
| ChFinAnn-Doc2EDAG | test | 13 | module-fast-screen | vLLM no_surface_memory, gmem=0.80 | 0.8538 | 精度升、召回降，净 F1 基本持平；Surface Memory 无稳定正向证据 |
| ChFinAnn-Doc2EDAG | test | 13 | module-fast-screen | vLLM no_slot_plan, gmem=0.80 | 0.8567 | 去掉 Slot Plan 略高，提示 Slot Plan 可能抑制 multi-event recall |
| ChFinAnn-Doc2EDAG | test | 13 | decoding | vLLM BF16 + LoRA, k=4 T=0.7 | 0.8421 | sampling 未优于 greedy |
| ChFinAnn-Doc2EDAG | test | 13 | SFT | vLLM BF16 no-SFT | 0.2482 | SFT 是必要增益源 |
| ChFinAnn-Doc2EDAG | test | 17 | backend | vLLM BF16 + LoRA, k=1 | 0.8473 | seed17 backend 诊断已完成；低于 seed13 HF 主路径 |
| DuEE-Fin-dev500 | test | 13 | backend | vLLM BF16 + LoRA, k=1 | 0.7502 | fresh rerun；比 HF seed13 低 2.94pp |
| DuEE-Fin-dev500 | test | 17 | backend | vLLM BF16 + LoRA, k=1 | 0.7470 | backend 诊断 |
| DuEE-Fin-dev500 | test | 42 | backend | vLLM BF16 + LoRA, k=1 | 0.7583 | backend 诊断 |
| DuEE-Fin-dev500 | test | 13 | decoding | vLLM BF16 + LoRA, k=4 T=0.7 | 0.7313 | sampling 未优于 k=1 |
| DuEE-Fin-dev500 | test | 13 | SFT | HF-4bin no-SFT | 0.0330 | 无 SFT 基线极低 |
| DuEE-Fin-dev500 | test | 13 | SFT | vLLM BF16 no-SFT | 0.1129 | 无 SFT 基线极低 |
| DuEE-Fin-dev500 | test | 13 | LRD | safe-anchor tau=0.90 | 0.7800 | 增益很小；诊断/附录，不进主方法 |
| DuEE-Fin-dev500 | test | 13 | module | HF-4bit no_surface_memory | 0.7812 | HF 主后端确认：去掉 Surface Memory 未造成可见下降 |
| DuEE-Fin-dev500 | test | 13 | module | HF-4bit no_slot_plan | 0.7758 | HF 主后端确认：相对 full 仅小幅下降约 0.38pp；结合 ChFinAnn vLLM，Slot Plan 只能算弱且不稳定证据 |
| DuEE-Fin-dev500 | test | 13 | module-fast-screen | vLLM no_surface_memory | 0.0208 | vLLM 0.70 显存配置下召回坍塌；不能单独作为模块结论 |
| DuEE-Fin-dev500 | test | 13 | module-fast-screen | vLLM no_slot_plan | 0.0164 | vLLM 0.70 显存配置下召回坍塌；不能单独作为模块结论 |
| DuEE-Fin-dev500 | test | 13 | module-fast-screen | vLLM no_surface_or_slot | 0.7549 | 与 full vLLM 接近，提示 backend/prompt 交互 |
| DuEE-Fin-dev500 | test | 13 | coarse lower bound | vLLM schema_only / direct_json | 0.6555 / 0.0000 | 非单变量，适合诊断/附录 |
| DuEE-Fin-dev500 | dev | 17 | invalid LRD | all k=4 parsed candidates | 0.3354 | 输入契约误用，禁止作为模型/LRD 性能 |

LRD 重要边界：主评测只能使用与 no-LRD/MRS 可比的 selected candidates。不要把 `parsed_candidates.*.jsonl` 里的全部 k=4 候选直接喂给 LRD 主评测；这样会造成 FP 爆炸，seed17 dev 的 `0.3354` 就是该误用的反例。

---

## 后续队列：按综合优先级排序

排序依据：先保证主结果和多种子稳定性，再补高论文价值、低 GPU 成本的模块消融；同一优先级内优先使用空闲 GPU3，避免干扰 GPU1/2 的长任务。估计时间来自当前服务器日志：DuEE-Fin HF test 约 `2.8h/run`，ChFinAnn HF test 约 `16h/run`，vLLM DuEE-Fin full test 约 `4-5min/run`，vLLM ChFinAnn full test 约 `15-20min/run`。

| 优先级 | 触发条件 | 任务 | 估计 GPU 时间 / 显存 | 论文价值 | GPU 调度 |
|---|---|---|---|---|---|
| P0 | 论文主线需要收缩 | 不再为挽救 Slot Plan 临时补训练；把主方法收缩到 schema-grounded SFT + role-safe JSON contract | 0 | 高：降低审稿风险 | 文档/论文修改优先 |
| P1 | 论文仍坚持写模块贡献 | 先在 dev 设计 predicted-plan 任务和 oracle-plan 上界，再决定是否花 GPU；不能直接用 test gold plan | 需要新实验设计 | 高但风险大：会改变方法主线 | 需单独立项 |
| P2 | 论文只需要附录模块表 | 已有 DuEE-Fin HF 单变量和 ChFinAnn vLLM 快筛；不建议补多种子消融，除非正文必须写模块贡献 | DuEE-Fin 每行每种子约 `2.8h`；ChFinAnn 每行每种子约 `16h` | 中：附录即可 | 只补最终要写进正文的行 |
| P3 | P1/P2 结果需要论文 lower bound | HF 或 vLLM `schema_only` / `direct_json` 粗粒度下界 | HF 昂贵，vLLM 低成本 | 中：不是严格单变量，适合作附录/诊断 | 优先 vLLM；HF 仅在论文需要时跑 |
| P4 | 用户单独授权 | 仅做 LRD 可比候选诊断 | 低到中 | 低到中：目前增益极小 | 只用 selected/fair candidate contract |
| P5 | 用户单独授权 | 继续扩展 backend/decoding ablation | vLLM 低，HF 中 | 低：已有足够 backend/sampling 证据 | 非当前优先事项 |

---

## 消融设计边界

核心消融要满足单变量原则：

| Profile | 模块变化 | 是否严格单变量 | 用途 |
|---|---|---|---|
| `full` | schema + role-safe instruction + surface candidates + slot plan | baseline | 现有主结果可作为 full control；若需要同代码版本 manifest，再补跑 |
| `no_surface_memory` | 只去掉 `[Surface Candidates]`，保留 slot plan | 是 | 评估 Surface Memory 贡献 |
| `no_slot_plan` | 只去掉 `[Event Slot Plan]`，保留 surface candidates | 是 | 评估 Slot Plan 贡献 |
| `no_surface_or_slot` | 同时去掉 surface candidates 和 slot plan | 否，组合去除 | 评估两个辅助模块的联合下界和交互 |
| `schema_only` | 去掉 role-safe 闭包、surface candidates、slot plan | 否，粗粒度 | 附录/诊断下界，不作为单变量主消融 |
| `direct_json` | 去掉 schema 和所有辅助 grounding | 否，粗粒度 | 最弱 direct extraction 下界 |

固定项：同一 checkpoint、同一 split、同一 `k=1` greedy、同一 seed、同一 evaluator 三轨、同一 no-LRD 主路径、同一 `slot_train_limit=50`。不要用 test 结果调参；如果要选择 profile 子集，优先在 dev/vLLM 筛选后冻结，再上 test。

方法边界：当前 SFT 训练没有消费 Slot Plan，训练脚本构造 SFT 样本时 `slot_plan=None`。推理时 Slot Plan 是 train-prior 弱提示，不是 learned planner。主论文不应把 Slot Plan 写成有效贡献；如需保留，只能放入附录诊断或未来工作。

---

## 消融命令模板

远程启动前仍需按项目规则再次给出 exact command + cwd + expected outputs。以下仅是 todo 模板。

HF-4bit + LoRA profile 模板：

```bash
cd /data/TJK/DEE/SARGE
export CUDA_VISIBLE_DEVICES=<gpu>
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export SARGE_ABLATION_PROFILE=<profile>
/data/TJK/envs/sarge_vllm_full/bin/python -u scripts/infer_checkpoint.py \
  --ckpt runs/sarge_sft_DuEE_Fin_dev500_s13_ep2_gpu0/artifacts/model/adapter \
  --model models/Qwen/Qwen3-4B-Instruct-2507 \
  --dataset DuEE-Fin-dev500 \
  --split test \
  --seed 13 \
  --k 1 \
  --slot-train-limit 50 \
  --source-commit <commit_with_ablation_profile_support> \
  --out runs/sarge_ablation_DuEE-Fin-dev500_test_seed13_<profile>_hf4bit_k1_<timestamp>
```

vLLM merged BF16 profile 筛选模板：

```bash
cd /data/TJK/DEE/SARGE
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export SARGE_ABLATION_PROFILE=<profile>
/data/TJK/envs/sarge_vllm_full/bin/python -u scripts/infer_checkpoint_vllm.py \
  --merged runs/merged_models/qwen3_4b_dueefin_ep2_s13 \
  --dataset DuEE-Fin-dev500 \
  --split test \
  --seed 13 \
  --k 1 \
  --slot-train-limit 50 \
  --source-commit <commit_with_ablation_profile_support> \
  --out runs/sarge_ablation_DuEE-Fin-dev500_test_seed13_<profile>_vllm_bf16_k1_<timestamp>
```

ChFinAnn 对应替换：

```bash
# HF seed17 adapter
--ckpt runs/sarge_sft_ChFinAnn_Doc2EDAG_s17_ep2_gpu1/artifacts/model/adapter
--dataset ChFinAnn-Doc2EDAG

# vLLM seed13 / seed17 merged model
--merged runs/merged_models/qwen3_4b_chfinann_ep2_s13
--merged runs/merged_models/qwen3_4b_chfinann_ep2_s17
```

---

## 常用后处理命令

三轨评测不需要 GPU：

```bash
cd /data/TJK/DEE/SARGE
/data/TJK/envs/sarge_vllm_full/bin/python -B scripts/eval_three_tracks.py \
  --run-root runs/<run_name>/<inner_run_name> \
  --dataset <DuEE-Fin-dev500|ChFinAnn-Doc2EDAG> \
  --split test
```

拉取小型证据快照时只同步 JSON，不拉 checkpoint、prediction JSONL、raw output 或 parsed candidates：

```bash
rsync -av \
  --include='*/' \
  --include='*.json' \
  --exclude='*' \
  gpu-4090:/data/TJK/DEE/SARGE/runs/<run_name>/ \
  paper/exp/data/run_snapshots/<asset_id>/
```

本地重建实验汇总：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B paper/exp/scripts/build_seed13_summary.py
```
