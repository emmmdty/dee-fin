# W3–4 服务器 Runbook：TRL↔vLLM 版本冒烟 + easy 桶小规模 GRPO

> ⚠️ **已过时（2026-06-20 标注）**：本文档是 W3–4 里程碑（easy 桶小规模 GRPO 冒烟）的旧
> runbook，其 Qwen 路径、卡分配、坏卡处理均已被取代。**全量 GPU 运维一律以
> `docs/GPU_RUNBOOK.md` + `docs/MIDTERM_HANDOFF.md` 为准**（含 card3 坏卡 nvmlshim 垫片、
> card0 他人占用、nohup 替代 tmux、SFT gradient checkpointing 等关键坑，本文档没有）。仅作历史留档。

> 里程碑 W3–4（`docs/RL_DESIGN.md` §7 / plan §七）。**前提：1–2× RTX 4090、CUDA
> 12.4 的服务器。** 本地（无 GPU）只到 W1–2，本阶段全部在服务器上跑。
>
> 退出标准（两条都要满足才算 W3–4 完成）：
> 1. TRL↔vLLM 结合跑通 ——「LoRA 权重每步同步进 rollout worker」被验证；
> 2. easy 桶（事件数 ≤6）小规模 GRPO 收敛 —— **四分量奖励同向上升**（无某一项塌成
>    0 而 total 还在涨 = reward hacking），且产出 adapter 的 **F1 ≥ SFT 基线**。

照下面 0→5 的顺序敲。每步给了「为什么」和「过/不过」判据。

---

## 0. 环境（一次性）

```bash
cd ~/myProjects/masterProjects/Fin-EKG
uv venv --python 3.10
# torch+cu124 来自 pyproject.toml 里 explicit 的 pytorch-cu124 索引
uv pip install -e ".[dev,llm,gnn,serve,rl]"
```

**版本预检（W3–4 第一风险 = TRL↔vLLM 版本耦合，先卡死再说）：**

```bash
uv run python - <<'PY'
import torch, transformers, peft, trl, vllm
print("torch        ", torch.__version__,        "（应 2.6.0+cu124）")
print("transformers ", transformers.__version__, "（应 4.51–4.53）")
print("peft         ", peft.__version__)
print("trl          ", trl.__version__,          "（应 0.17–0.18.x，<0.19）")
print("vllm         ", vllm.__version__,         "（应 0.8.5）")
print("cuda         ", torch.cuda.is_available(), torch.cuda.device_count(), "卡")
PY
```

> trl 落在 `[0.17,0.19)`、vllm==0.8.5、torch==2.6.0 这三条任意一条不符就**别往下走**，
> 先对 `uv.lock`。`trainer.py` 会按已装 TRL 的签名过滤 `GRPOConfig` 字段，patch 级
> 升级不会因改名旋钮崩入口（被丢弃的旋钮会打印出来），但跨 minor 的 vllm 不在保护范围。

---

## 1. 数据

MAVEN-ERE 要手动下载（官方 jsonl，test 标签隐藏）：

```bash
uv run python scripts/download_datasets.py --dataset maven_ere   # 打印下载地址与落盘路径
# 按提示把 train.jsonl / valid.jsonl / test.jsonl 放到 data/raw/maven_ere/
# 然后运行 uv run python scripts/preprocess_datasets.py
```

冒烟用的小切片（前 ~50 篇即可）：

```bash
uv run python scripts/preprocess_datasets.py --maven-smoke-n 50
```

> 注意 grounding 奖励要原文：loader 的 `doc_text` 取自记录的 `text`/`sentences`/`tokens`。
> 若你的 jsonl 这三个字段全空，`R_ground` 会恒为 0 —— 预检见第 3 步的奖励面板。

---

## 2. SFT 依赖（只 easy 桶 / 主训练需要；冒烟不需要）

`grpo_rlvr_easy.yaml` 从 `runs/relation_extractor_lora` 热启。若还没跑第一章 SFT：

```bash
uv run --extra llm python scripts/train_relation_extractor.py \
    --config configs/relations/llm_grounded_consistent.yaml \
    --output runs/relation_extractor_lora
```

> 冒烟配置（`grpo_rlvr_smoke*.yaml`，`sft_adapter_path: null`）是 fresh LoRA、自含，
> 可以在 SFT 还没好时先跑，专门验证结合链路。

---

## 3. 链路冒烟（hf 后端，最快，不碰 vLLM）

先用「永远能跑」的 hf 后端确认奖励/训练循环本身没问题，单卡：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra llm --extra rl \
    python scripts/train_relation_grpo.py \
    --config configs/relations/grpo_rlvr_smoke.yaml \
    --output runs/grpo_smoke_hf
```

**过：** 跑完 `runs/grpo_smoke_hf/reward_means_phase0.json` 里 `format/grounding/`
`consistency/task_f1/total` 五个键都在，且不是全 0（尤其 grounding≠0 证明 doc_text 接上了）。

---

## 4. ⭐ TRL↔vLLM 结合冒烟（W3–4 的核心关卡）

hf 后端**不**测 vLLM 耦合。这一步用 `vllm_colocate`（vLLM 与训练共享单卡，无需第二张卡、
无需 `trl vllm-serve` 进程）真正叩「rollout 走 vLLM + 每步把 LoRA 权重广播给 rollout worker」：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra llm --extra serve --extra rl \
    python scripts/train_relation_grpo.py \
    --config configs/relations/grpo_rlvr_smoke_vllm.yaml \
    --output runs/grpo_smoke_vllm
```

**权重同步验证（过/不过判据，三选一即可，建议都看）：**
1. **不报错即首关**：colocate 模式下 TRL 每步把训练侧权重灌进同进程 vLLM 引擎；版本不配
   会在第一步权重加载/广播处直接抛错。能稳定跑过 50 步 = 同步通道没断。
2. **生成在变**：`logging_steps=10`，看 TRL 日志里 `reward`/`completions` 随步变化、
   reward 不是一条平的常数线 —— 平线常意味着 rollout 没吃到更新后的权重。
3. **行为可复现回主线**：把本步存下的 adapter 当 `adapter_path` 塞进
   `configs/relations/llm_grounded_consistent.yaml`，跑 `evaluate_relations.py`，
   指标应明显偏离 base（证明训出来的 adapter 是真的、且能被现有 pipeline 加载）。

> 显存紧（colocate + 4B + grad-ckpt）：配置已把 `max_completion_length` 降到 384、
> `vllm_gpu_memory_utilization=0.3`、`G=4`。仍 OOM 就把 completion 降到 256 或 G 降到 2。
>
> 有两张卡时也可改用 `vllm_server` 后端验证（更贴近主训练，见第 5 步的启动方式）。

---

## 5. easy 桶小规模 GRPO（W3–4 的训练落点）

事件数 ≤6 的窗口、~150 步、四分量满奖励、SFT 热启。双卡 vLLM-server 模式：

```bash
# 终端 A：rollout 引擎独占 GPU0
CUDA_VISIBLE_DEVICES=0 uv run --extra serve --extra rl \
    trl vllm-serve --model /data/TJK/models/Qwen/Qwen3-4B-Instruct-2507

# 终端 B：训练独占 GPU1，连上 A 的 8000 端口
CUDA_VISIBLE_DEVICES=1 uv run --extra llm --extra rl \
    python scripts/train_relation_grpo.py \
    --config configs/relations/grpo_rlvr_easy.yaml \
    --output runs/relation_grpo_easy
```

只有单卡时改 colocate（无需终端 A）：

```bash
# 把 configs/relations/grpo_rlvr_easy.yaml 的 rollout.backend 改成 vllm_colocate，
# num_generations 降到 4、max_completion_length 降到 512，然后：
CUDA_VISIBLE_DEVICES=0 uv run --extra llm --extra serve --extra rl \
    python scripts/train_relation_grpo.py \
    --config configs/relations/grpo_rlvr_easy.yaml \
    --output runs/relation_grpo_easy
```

**收敛验证：**

```bash
cat runs/relation_grpo_easy/reward_means_phase0.json   # 四分量 + total 的 phase 内均值
cat runs/relation_grpo_easy/reward_curve.json          # 窗口化逐分量曲线 —— 看"四分量同向上升"
cat runs/relation_grpo_easy/summary.json               # 最终 adapter_path
```

> 判据读 `reward_curve.json`：每个点是连续 64 条补全的逐分量均值。四分量应同向
> 上升；某一分量塌向 0 而 total 还在涨 = reward hacking，停下来查样例。

**F1 ≥ SFT 验证**（同一评测脚本，零改动）：

```bash
# 1) SFT 基线：adapter_path = runs/relation_extractor_lora
# 2) GRPO 后：adapter_path = runs/relation_grpo_easy/phase0
# 两次都跑：
uv run --extra llm python scripts/evaluate_relations.py \
    --config configs/relations/llm_grounded_consistent.yaml
```

---

## 退出清单（W3–4 完成 = 全勾）

- [ ] 版本预检：trl∈[0.17,0.19) · vllm 0.8.5 · torch 2.6.0 全对
- [ ] hf 链路冒烟跑通，grounding 奖励 ≠ 0
- [ ] **vLLM 结合冒烟跑过 50 步不报错，且 reward 随步变化（权重同步成立）**
- [ ] easy 桶 GRPO 跑完，`reward_means_phase0.json` 四分量同向上升、无单项塌 0
- [ ] GRPO 后 adapter 的 F1 **≥** SFT 基线
- [ ] 全程逐分量奖励曲线无 reward hacking 迹象（grounding/consistency 不在 total 上涨时反降）

全勾后进入 W5–7：`grpo_rlvr.yaml` 全课程 6/12/24 ×3 seeds(13/17/42) + CCKS 混训。

---

## 排错（对应 plan §八 风险表）

| 现象 | 处置 |
| --- | --- |
| `trl vllm-serve` / colocate 启动即报版本/符号错 | 回退到 `rollout.backend: hf`，先把训练循环验通；再对 `uv.lock` 修 trl/vllm，**绝不为 trl 升 vllm/torch** |
| 入口打印 `dropping knobs unknown to this TRL version: [...]` | 正常 —— 是签名过滤在保护你；确认被丢的不是关键旋钮即可 |
| OOM | G 8→4→2、completion 768→512→256、colocate `gpu_memory_utilization` 调低；最后退 `hf` 后端 |
| grounding 恒 0 | jsonl 的 `text`/`sentences`/`tokens` 为空，doc_text 取不到原文；补原文字段后重跑 |
| reward 一条平线、F1 不动 | rollout 没吃到更新权重 —— 优先排版本耦合；用 colocate 复现确认 |
| GRPO 不稳/震荡 | 已开 SFT 热启 + lr 1e-5 + KL β0.01 + grad-clip 0.5 + 组内 std 下限；再不稳缩小 easy 桶步数、加大 batch |
