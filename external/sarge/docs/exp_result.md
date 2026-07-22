# SARGE 实验结果记录

> 最后更新：2026-05-23 13:09 UTC+8
> 证据来源：服务器 `/data/TJK/DEE/SARGE/runs/` 的已完成 run；小型 JSON 快照已收拢到 `paper/exp/data/run_snapshots/`，索引见 `paper/exp/data/asset_registry.json`。

---

## 1. 当前主结果

主表口径为 `legacy_doc2edag` / Legacy-FS。`unified_strict`、`docfee_official` 和 ExactRec 只作为诊断指标，不与主表基线混算。

| Dataset | Split | Seed | 主设置 | Asset | Legacy-FS F1 | P | R | F1(S) | F1(M) | Unified F1 | DocFEE F1 | ExactRec |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ChFinAnn-Doc2EDAG | test | 13 | HF-4bin + LoRA, k=1 greedy | `chfinann_test_seed13_hf4bin_k1` | 0.8603 | 0.8437 | 0.8775 | 0.8995 | 0.8182 | 0.8742 | 0.8653 | 0.5842 |
| DuEE-Fin-dev500 | test | 13 | HF-4bin + LoRA, k=1 greedy, no-LRD | `dueefin_test_seed13_hf4bin_k1_no_lrd` | 0.7796 | 0.7664 | 0.7933 | 0.7927 | 0.7751 | 0.7888 | 0.7771 | 0.4285 |

ChFinAnn 主结果已从原 vLLM BF16 行切换到 HF-4bin cross-check 行，因为 full-test HF-4bin Legacy-FS F1 为 `0.8603`，高于 vLLM BF16 的 `0.8547`，且与 DuEE-Fin 主路径保持一致。

---

## 2. Test 消融与诊断

| Dataset | Split | Seed | Factor | Setting | Legacy-FS F1 | P | R | F1(S) | F1(M) | 结论 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| ChFinAnn-Doc2EDAG | test | 13 | backend | HF-4bin + LoRA, k=1 | 0.8603 | 0.8437 | 0.8775 | 0.8995 | 0.8182 | 当前 ChFinAnn 主路径 |
| ChFinAnn-Doc2EDAG | test | 13 | backend | vLLM BF16 + LoRA, k=1 | 0.8547 | 0.8407 | 0.8691 | 0.8978 | 0.8082 | 后端交叉验证，略低于 HF |
| ChFinAnn-Doc2EDAG | test | 13 | module-fast-screen | vLLM full, gmem=0.80 | 0.8547 | 0.8426 | 0.8671 | 0.8994 | 0.8064 | ChFinAnn vLLM 消融 control；只作模块诊断，不替代 HF 主路径 |
| ChFinAnn-Doc2EDAG | test | 13 | module-fast-screen | vLLM no surface memory, gmem=0.80 | 0.8538 | 0.8483 | 0.8595 | 0.9035 | 0.7999 | 精度升、召回降，净 F1 基本持平；Surface Memory 无稳定正向证据 |
| ChFinAnn-Doc2EDAG | test | 13 | module-fast-screen | vLLM no slot plan, gmem=0.80 | 0.8567 | 0.8403 | 0.8739 | 0.8963 | 0.8139 | 召回和 multi-event F1 更高；Slot Plan 不支持作为稳定正向模块 |
| ChFinAnn-Doc2EDAG | test | 13 | no-SFT | vLLM BF16 base, k=1 | 0.2482 | 0.6119 | 0.1556 | 0.3893 | 0.0675 | SFT 是主要增益来源 |
| ChFinAnn-Doc2EDAG | test | 13 | decoding | vLLM BF16 + LoRA, k=4 T=0.7 | 0.8421 | 0.7954 | 0.8945 | 0.8752 | 0.8071 | sampling 未优于 k=1 |
| DuEE-Fin-dev500 | test | 13 | backend | HF-4bin + LoRA, k=1 | 0.7796 | 0.7664 | 0.7933 | 0.7927 | 0.7751 | 当前 DuEE-Fin 主路径 |
| DuEE-Fin-dev500 | test | 13 | module | HF-4bit no surface memory | 0.7812 | 0.7653 | 0.7978 | 0.7975 | 0.7767 | HF 单变量确认：去掉 Surface Memory 未造成可见下降 |
| DuEE-Fin-dev500 | test | 13 | module | HF-4bit no slot plan | 0.7758 | 0.7628 | 0.7892 | 0.7937 | 0.7703 | HF 单变量确认：相对 full 小幅下降约 0.38pp，Slot Plan 有弱正向证据 |
| DuEE-Fin-dev500 | test | 13 | backend | vLLM BF16 + LoRA, k=1 | 0.7502 | 0.7480 | 0.7524 | 0.7857 | 0.7323 | fresh rerun；比 HF 主路径低 2.94pp |
| DuEE-Fin-dev500 | test | 17 | backend | vLLM BF16 + LoRA, k=1 | 0.7470 | 0.7515 | 0.7426 | 0.7875 | 0.7225 | backend seed-extension 诊断 |
| DuEE-Fin-dev500 | test | 42 | backend | vLLM BF16 + LoRA, k=1 | 0.7583 | 0.7579 | 0.7588 | 0.7941 | 0.7378 | backend seed-extension 诊断 |
| DuEE-Fin-dev500 | test | 13 | module-fast-screen | vLLM no surface memory | 0.0208 | 0.4520 | 0.0106 | 0.0358 | 0.0092 | vLLM 0.70 显存配置下召回坍塌；只作快筛/故障形态证据 |
| DuEE-Fin-dev500 | test | 13 | module-fast-screen | vLLM no slot plan | 0.0164 | 0.4565 | 0.0084 | 0.0289 | 0.0069 | vLLM 0.70 显存配置下召回坍塌；只作快筛/故障形态证据 |
| DuEE-Fin-dev500 | test | 13 | module-fast-screen | vLLM no surface or slot | 0.7549 | 0.7509 | 0.7589 | 0.7831 | 0.7419 | 与 full vLLM 接近，提示 profile 与 vLLM 输出控制存在交互 |
| DuEE-Fin-dev500 | test | 13 | coarse-lower-bound | vLLM schema only | 0.6555 | 0.7527 | 0.5805 | 0.7091 | 0.6122 | 非单变量；附录/诊断下界 |
| DuEE-Fin-dev500 | test | 13 | coarse-lower-bound | vLLM direct json | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 最弱直接抽取失败 |
| DuEE-Fin-dev500 | test | 13 | no-SFT | HF-4bin base, k=1 | 0.0330 | 0.4479 | 0.0171 | 0.0368 | 0.0304 | HF no-SFT 极低，确认 SFT 必要 |
| DuEE-Fin-dev500 | test | 13 | no-SFT | vLLM BF16 base, k=1 | 0.1129 | 0.3722 | 0.0665 | 0.1378 | 0.0936 | SFT 是主要增益来源 |
| DuEE-Fin-dev500 | test | 13 | decoding | vLLM BF16 + LoRA, k=4 T=0.7 | 0.7313 | 0.6922 | 0.7751 | 0.7583 | 0.7249 | sampling 未优于 k=1 |
| DuEE-Fin-dev500 | test | 13 | LRD | safe-anchor tau=0.90 | 0.7800 | 0.7671 | 0.7933 | 0.7937 | 0.7751 | 增益很小；LRD 保持诊断/附录 |

---

## 3. Dev 与多种子诊断

这些结果用于种子稳定性和错误形态分析，不进入 test 主表。

| Dataset | Split | Seed | Setting | Legacy-FS F1 | Unified F1 | DocFEE F1 | ExactRec | Status |
|---|---|---:|---|---:|---:|---:|---:|---|
| ChFinAnn-Doc2EDAG | test | 17 | HF-4bin + LoRA, k=1 | 0.8536 | 0.8705 | 0.8627 | 0.5532 | seed-extension diagnostic |
| ChFinAnn-Doc2EDAG | test | 42 | HF-4bin + LoRA, k=1 | 0.8533 | 0.8701 | 0.8613 | 0.5461 | seed-extension diagnostic |
| ChFinAnn-Doc2EDAG | test | mean±std | HF seeds 13/17/42, k=1 | 0.8557±0.0039 | 0.8716±0.0023 | 0.8631±0.0020 | 0.5612±0.0203 | stability diagnostic |
| DuEE-Fin-dev500 | test | 17 | HF-4bin + LoRA, k=1, no-LRD | 0.7872 | 0.7937 | 0.7822 | 0.4314 | seed-extension diagnostic |
| DuEE-Fin-dev500 | test | 42 | HF-4bin + LoRA, k=1, no-LRD | 0.7828 | 0.7921 | 0.7809 | 0.4382 | seed-extension diagnostic |
| DuEE-Fin-dev500 | test | mean±std | HF seeds 13/17/42, no-LRD | 0.7832±0.0038 | 0.7915±0.0025 | 0.7801±0.0026 | 0.4327±0.0050 | stability diagnostic |
| DuEE-Fin-dev500 | dev | 13 | HF-4bin + LoRA, k=1 | 0.7666 | 0.7723 | 0.7675 | 0.4175 | valid diagnostic |
| DuEE-Fin-dev500 | dev | 17 | train_sft k=4 + MRS selected, no-LRD | 0.7794 | 0.7849 | 0.7812 | 0.4187 | valid diagnostic |
| DuEE-Fin-dev500 | dev | 42 | train_sft k=4 + MRS selected, no-LRD | 0.7669 | 0.7736 | 0.7688 | 0.4186 | valid diagnostic |
| DuEE-Fin-dev500 | dev | 17 | LRD over all k=4 parsed candidates | 0.3354 | 0.3458 | 0.3458 | 0.1809 | invalid |

Seed17 dev LRD `0.3354` 是输入契约误用诊断：`postprocess_lrd_eval.py` 吃入 `parsed_candidates.dev.jsonl` 的全部 k=4 候选，`events_in=2692`、`events_out=2444`，而 fair no-LRD/MRS 预测只有 `673` 个事件。该结果不得作为 seed17 模型或 LRD 性能报告。

---

## 4. 训练资产

| Dataset | Seed | Epochs | Train docs | Train events | Train time | Run root | Status |
|---|---:|---:|---:|---:|---:|---|---|
| DuEE-Fin-dev500 | 13 | 2 | 6515 | 8824 | 9970.7s | `runs/sarge_sft_DuEE_Fin_dev500_s13_ep2_gpu0/` | completed |
| DuEE-Fin-dev500 | 17 | 2 | 6515 | 8824 | 9840.3s | `runs/sarge_sft_DuEE_Fin_dev500_s17_ep2_gpu0/` | completed |
| DuEE-Fin-dev500 | 42 | 2 | 6515 | 8824 | 10025.2s | `runs/sarge_sft_DuEE_Fin_dev500_s42_ep2_gpu1/` | completed |
| ChFinAnn-Doc2EDAG | 13 | 2 | 25632 | 38088 | 30310.2s | `runs/sarge_sft_ChFinAnn_Doc2EDAG_s13_ep2_gpu1/` | completed |
| ChFinAnn-Doc2EDAG | 17 | 2 | 25632 | 38088 | 30152.7s | `runs/sarge_sft_ChFinAnn_Doc2EDAG_s17_ep2_gpu1/` | completed; test eval synced |
| ChFinAnn-Doc2EDAG | 42 | 2 | 25632 | 38088 | - | `runs/sarge_sft_ChFinAnn_Doc2EDAG_s42_ep2_gpu1/` | completed; test eval synced |

---

## 5. 运行中任务

| Task | GPU | Log | Status |
|---|---:|---|---|
| none | - | - | 2026-05-23 08:45 UTC+8 只读查询未见 SARGE 训练/推理/eval 进程 |

新增完成项已拉取小型 JSON 快照并写入 registry、自动表和本结果记录；仍不把 seed-extension 或单 seed module ablation 提升为主表行。

---

## 6. SFT 与模块归因边界

LoRA SFT 本身不是算法创新；LoRA 是参数高效适配手段。当前可写入论文主方法的设计点是 schema-grounded event-table instruction tuning：训练样本把每篇金融公告、数据集 schema 和 role-safe JSON 输出合同对齐，只对 assistant answer 部分计算 loss。

当前实现没有真正训练 Slot Plan。`scripts/train_sft.py` 在构造 SFT 样本时使用 `surface_candidates=memory.candidates`，但 `slot_plan=None`。推理时的 Slot Plan 来自 train split 统计先验，而不是 learned planner。因此 Slot Plan 存在训练-推理不一致，不应作为主贡献。

当前主路径未发现 test/dev gold 泄露到推理 prompt 的证据：predict 文档以 `gold=None` 加载，prompt builder 与 Qwen/vLLM backend 都拒绝 gold-visible 字段，Slot Plan fit 只使用 train docs。evaluation 阶段按 predicted doc_ids 过滤 gold 是 subset evaluator 对齐，不是模型输入。

详细分析见 `docs/sft_module_risk_analysis_20260523.md`。

---

## 7. API 诊断

DeepSeek API 诊断是 CPU/API-only，不使用 GPU，不进入论文主表。汇总文件为 `paper/exp/data/api_diagnostics/deepseek_api_diagnostics_20260522.json`，报告为 `docs/deepseek_api_diagnostics_20260522.md`。

关键结果：修复响应预算后，DuEE-Fin dev500 full prompt 达到 flash Legacy-FS F1 `0.4529`、pro `0.4348`；`schema_only` 和 `surface_only` 没有追上本地 SARGE checkpoint。临时 value normalization probe 将 flash/pro full prompt 提升到 `0.5046` / `0.4849`，说明主要问题是输出值表面形式与评测规范不匹配，而不是 DeepSeek API 连接失败。
