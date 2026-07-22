# DuEE-Fin Test Backend Sensitivity Report

> Date: 2026-05-22  
> Local root: `/home/tjk/myProjects/masterProjects/DEE/SARGE/`  
> Server root: `/data/TJK/DEE/SARGE/`  
> Local Python: `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`  
> Server Python: `/data/TJK/envs/sarge_vllm_full/bin/python`  
> Dataset and split: `DuEE-Fin-dev500 / test`, 1,171 documents  
> Primary metric: `legacy_doc2edag` / Legacy-FS F1. `unified_strict` is auxiliary.

## Abstract

The DuEE-Fin test results show a reproducible sensitivity to the generation backend. The current HF Transformers path uses a 4-bit NF4 base model with a LoRA adapter loaded at inference time, while the vLLM path uses a pre-merged BF16 checkpoint. On the completed seed-13 full-test evidence, HF-4bit + LoRA obtains Legacy-FS F1 `0.7796`, whereas the current vLLM BF16 merged validation obtains `0.7502`. Across the available seed-13/17/42 full-test rows, HF averages `0.7832` Legacy-FS F1 and vLLM averages `0.7518`, a gap of about `3.1` points. The same evidence also shows a large throughput difference: HF sequential generation takes about `10,394s` on average for 1,171 documents, while vLLM prebatched generation takes about `262s`, roughly a `40x` generation-time speedup under the current scripts. The most plausible explanation is not an evaluator discrepancy, but a combined backend-path effect: runtime LoRA + 4-bit NF4 versus merged BF16 weights, different engine kernels and batching, different EOS/pad resolution, and different JSON stopping/repair behavior. Therefore, HF-4bit + LoRA should remain the main reported DuEE-Fin test path, and vLLM should be reported as a fast diagnostic or backend-cross-check path unless future controlled runs close the metric gap.

## 1. Problem Discovery

The discrepancy was first visible in the experiment tables: the seed-13 DuEE-Fin HF-4bit + LoRA test run was recorded as the main path with Legacy-FS F1 `0.7796`, while an earlier vLLM BF16 merged backend cross-check was lower. During this audit, a fresh vLLM full-test validation was run on source commit `ff0761e88e456ea001429be86d892375bec36349`; it reproduced the newer vLLM seed-13 value exactly at Legacy-FS F1 `0.7502`. This revises the current seed-13 backend gap to `2.94` points relative to the historical HF seed-13 full test, rather than the older `4.42`-point table snapshot.

The no-SFT runs must not be interpreted as backend comparisons. The HF no-SFT row uses `--no-adapter` and collapses to Legacy-FS F1 `0.0330`; the vLLM no-SFT row also remains low at `0.1129`. These rows confirm that supervised fine-tuning is the main source of extraction ability, but they do not explain the residual gap between HF-4bit + LoRA and vLLM BF16 merged after fine-tuning.

## 2. Full-Test Evidence

The following table uses the same DuEE-Fin test split and the same evaluator family. For vLLM rows, the listed seed is the training/merged-model seed inferred from the model artifact path; the vLLM command itself records generation seed `13` unless explicitly overridden. HF `time` and vLLM `prebatch` both exclude model loading because both scripts construct the backend before starting the reported generation timer.

| Path | Model seed | Backend path | Legacy F1 | P | R | Multi F1 | Single F1 | Unified F1 | Accepted events | Empty pred | Parse status error/schema | Generation time |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HF | 13 | 4-bit NF4 base + LoRA adapter | 0.7796 | 0.7664 | 0.7933 | 0.7751 | 0.7927 | 0.7888 | 1557 | 176 | 2 / 1 | 10813s |
| HF | 17 | 4-bit NF4 base + LoRA adapter | 0.7872 | 0.7840 | 0.7904 | 0.7780 | 0.8017 | 0.7937 | 1508 | 183 | 0 / 2 | 10097s |
| HF | 42 | 4-bit NF4 base + LoRA adapter | 0.7828 | 0.7743 | 0.7915 | 0.7749 | 0.8004 | 0.7921 | 1534 | 181 | 0 / 0 | 10271s |
| vLLM | 13 | merged BF16 checkpoint | 0.7502 | 0.7480 | 0.7524 | 0.7323 | 0.7857 | 0.7596 | 1481 | 182 | 16 / 4 | 255s + 2s |
| vLLM | 17 | merged BF16 checkpoint | 0.7470 | 0.7515 | 0.7426 | 0.7225 | 0.7875 | 0.7544 | 1466 | 180 | 23 / 2 | 268s + 2s |
| vLLM | 42 | merged BF16 checkpoint | 0.7583 | 0.7579 | 0.7588 | 0.7378 | 0.7941 | 0.7675 | 1485 | 179 | 15 / 7 | 263s + 2s |

The metric gap is not a one-off seed-13 artifact. HF is consistently in the `0.7796-0.7872` range, while vLLM is consistently in the `0.7470-0.7583` range. The gap is also not caused by a catastrophic schema failure: all rows have evaluator-side `schema_valid_rate = 1.0`. Instead, vLLM has more generation-side parse/status problems, fewer accepted events, and lower recall on the main fixed-slot metric.

## 3. Supplemental Validation

Two bounded probes were run on GPU 3 under source commit `ff0761e88e456ea001429be86d892375bec36349`, followed by a full vLLM validation. The generated server summary is stored at `runs/analysis_dueefin_backend_report_20260522/summary/backend_report_evidence.json`.

| Probe | Docs | Backend path | Legacy F1 | P | R | Events | Parse error | Schema count | Timing |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `hf_seed13_limit128` | 128 | 4-bit NF4 base + LoRA adapter | 0.7812 | 0.7807 | 0.7816 | 169 | 1 | 0 | 1128s |
| `vllm_seed13_limit128` | 128 | merged BF16 checkpoint | 0.6612 | 0.7411 | 0.5968 | 141 | 13 | 4 | 74s + 0s |
| `vllm_seed13_full_validation` | 1171 | merged BF16 checkpoint | 0.7502 | 0.7480 | 0.7524 | 1481 | 16 | 3 | 255s + 2s |

The 128-document probe is useful for failure-shape analysis but should not be used as a final estimate of vLLM performance. On the first 128 documents, vLLM concentrated `13` parse errors and produced only `141` accepted events, causing an unusually low recall. The full run diluted this subset effect and returned to the stable full-test value of `0.7502`. The probe nevertheless identifies the same mechanism seen in the full-test diagnostics: vLLM is much faster, but its output stream more often reaches malformed or schema-invalid states under the current prompt, prefix, and stopping setup.

## 4. Attribution of Metric Differences

The evaluator is not the primary source of the discrepancy. The compared rows use canonical JSONL predictions, filtered gold from the same DuEE-Fin test split, and the same `eval_three_tracks.py` path. The Legacy-FS evaluator counts fixed schema-role slots under the Doc2EDAG-style greedy matching policy, and the unified-strict metric is reported only as an auxiliary diagnostic. The direction of the gap is consistent across both Legacy-FS and unified-strict, which argues against an evaluator-only explanation.

The strongest attribution is the backend artifact path. HF inference loads the base Qwen3-4B model with `quantization = 4-bit NF4`, `double_quantization = true`, and a PEFT LoRA adapter at runtime. The vLLM path consumes a self-contained merged BF16 checkpoint. Those are not bit-identical inference paths. Even with greedy decoding, small logit-level differences can change boundary decisions in event type selection, role value copying, and JSON closure. This matters for financial event extraction because most scored units are exact normalized surface strings rather than tolerant semantic matches.

The second contributor is generation control. In the HF path, the manifest records resolved `eos_token_id = 151645` and `pad_token_id = 151643`; in the vLLM path, the resolved generation config leaves EOS and pad IDs as `null`. Both paths use the same response prefix and balanced JSON stopping logic, but the observed output forms differ. The HF seed-13 full run generated array-continuation style outputs that were repaired by the prefix reconstruction logic in almost all rows, while the vLLM run produced JSON-object style outputs directly. This does not by itself make one output invalid, but it shows that the two engines are not traversing the same serialized output regime.

The third contributor is batched decoding. vLLM prebuilds all prompts and calls `backend.preload_prompts`, then the pipeline loop consumes cached outputs. This changes the execution path from sequential single-prompt generation to large-batch scheduling. The speed gain is the intended effect, but the batch path also uses different kernels, KV-cache layout, scheduler behavior, and stopping aggregation. The model remains deterministic in configuration, but deterministic does not mean cross-backend bit identity.

The fourth contributor is recall loss from missing or invalid events. On seed 13 full test, HF accepts `1557` events and obtains `5976` true-positive fixed slots. vLLM accepts `1481` events and obtains `5668` true-positive fixed slots. vLLM also has slightly more false positives (`1910` versus `1822`) and more false negatives (`1865` versus `1557`). Thus the metric gap is not only a precision tradeoff; the dominant seed-13 effect is lower recall with additional parse/status losses.

## 4.1 Follow-up Mechanism Probes

After the first backend report, a bounded vLLM `limit128` mechanism suite was run to test whether the surprising module-ablation behavior was caused by the SARGE modules themselves or by backend/prompt execution. The key result is that the same nominal `full` profile changed from Legacy-FS F1 `0.0024` at `gpu_memory_utilization=0.70` to `0.6555` at `gpu_memory_utilization=0.80`. This is too large to attribute to a paper module; it is backend execution sensitivity under the current vLLM prompt/stopping setup.

| Profile | Variant | gmem | Docs | Legacy-FS F1 | Unified F1 | DocFEE F1 | ExactRec |
|---|---|---:|---:|---:|---:|---:|---:|
| full | base | 0.70 | 128 | 0.0024 | 0.0024 | 0.0024 | 0.0000 |
| no_surface_memory | base | 0.70 | 128 | 0.0025 | 0.0047 | 0.0047 | 0.0000 |
| no_surface_or_slot | base | 0.70 | 128 | 0.0785 | 0.0781 | 0.0759 | 0.0308 |
| full | base | 0.80 | 128 | 0.6555 | 0.6515 | 0.6434 | 0.2866 |
| no_surface_memory | base | 0.80 | 128 | 0.6748 | 0.6684 | 0.6579 | 0.2684 |
| no_slot_plan | base | 0.80 | 128 | 0.6306 | 0.6253 | 0.6146 | 0.2330 |
| no_surface_or_slot | base | 0.80 | 128 | 0.6584 | 0.6529 | 0.6449 | 0.2572 |
| no_surface_memory | SACD strict | 0.80 | 128 | 0.5790 | 0.5814 | 0.5728 | 0.2093 |
| no_slot_plan | SACD strict | 0.80 | 128 | 0.5780 | 0.5783 | 0.5703 | 0.2385 |

This resolves the apparent conflict with the full-test vLLM fast-screen ablation. The full-test `no_surface_memory` and `no_slot_plan` rows at `gmem=0.70` collapsed to Legacy-FS F1 `0.0208` and `0.0164`, but the `limit128` `gmem=0.80` suite shows much healthier behavior for the same conceptual profiles. Therefore, these vLLM rows are valuable as fault-shape diagnostics and fast screening, but they should not be used as final module-attribution evidence unless the backend configuration is frozen and shown to reproduce the HF path. The HF-4bit confirmation is the safer paper route: the completed HF `no_surface_memory` row is `0.7812`, essentially tied with the HF full row `0.7796`, while HF `no_slot_plan` is `0.7758`. This gives weak positive evidence for Slot Plan, but the effect is small and should be reported as single-seed module evidence rather than a final causal claim.

## 5. Inference-Time Analysis

The timing difference is structural. The HF script calls the candidate generator with `backend.generate_one` per document and reports per-document GETM progress. The full-test logs show `time=10813s`, `10097s`, and `10271s` for the three HF seeds, equivalent to about `8.6-9.2s/doc`. The 128-document current-commit HF probe also reports `1128s`, or `8.8s/doc`, confirming that the historical full-run timing is representative.

The vLLM script precomputes all prompts, runs one batched `preload_prompts` call, and then spends only about `2s` in the downstream pipeline loop. The full-test vLLM logs and the new validation show `255-268s` for 1,171 prompts, or about `0.22s/doc` for generation after backend construction. Relative to the HF mean of about `10,394s`, this is roughly a `40x` generation-time speedup. The 128-document vLLM probe is slower per document (`74s`, about `0.58s/doc`) because fixed scheduling overhead is less amortized on a small batch; the full-test run is the more relevant throughput measurement.

This means vLLM is operationally valuable for fast diagnostic sweeps, multi-seed sanity checks, and prompt/backend sensitivity searches. It should not automatically replace the HF path for the main DuEE-Fin paper result because the speed improvement currently comes with a stable metric drop.

## 6. Conclusion

The DuEE-Fin backend difference is real and reproducible under the current artifacts. It should be described as backend-path sensitivity, not as a pure evaluator issue or a generic claim that one framework is intrinsically better. HF-4bit + LoRA is slower but currently more accurate on DuEE-Fin test. vLLM BF16 merged is about an order of magnitude faster in wall-clock generation and about `40x` faster under the script-level generation timers, but it consistently loses about `3` Legacy-FS points across the available seeds.

For paper reporting, the main DuEE-Fin result should remain the HF-4bit + LoRA, k=1, no-LRD line. vLLM results should be included as backend cross-checks or diagnostic ablations with their own label. Future work should only promote vLLM to the main path if a controlled backend-alignment run demonstrates that merged BF16, EOS/pad handling, prompt serialization, and JSON stopping reproduce the HF metric level without relying on test-set tuning.

## Evidence Index

Existing full-test HF logs:

- `logs/sarge_infer_DuEE-Fin-dev500_test_seed13_safe_anchor_source_f56a0d3_20260520T003122Z.log`
- `logs/sarge_infer_DuEE-Fin-dev500_test_seed17_4bitNF4_k1_20260521T2141Z.log`
- `logs/sarge_infer_DuEE-Fin-dev500_test_seed42_4bitNF4_k1_20260521T2221Z.log`

Existing and new vLLM full-test evidence:

- `logs/sarge_vllm_DuEE-Fin-dev500_test_seed13_bf16_k1_rerun_20260521T230125Z.log`
- `logs/sarge_vllm_DuEE-Fin-dev500_test_seed17_bf16_k1_20260521T223822Z.log`
- `logs/sarge_vllm_DuEE-Fin-dev500_test_seed42_bf16_k1_20260521T224849Z.log`
- `runs/analysis_dueefin_backend_report_20260522/vllm_bf16_merged_seed13_full/sarge_infer_DuEE-Fin-dev500_test_20260522T002503Z/`

Supplemental probe summary:

- `runs/analysis_dueefin_backend_report_20260522/summary/backend_report_evidence.json`
- `runs/analysis_dueefin_backend_report_20260522/hf4bit_lora_seed13_limit128/sarge_infer_DuEE-Fin-dev500_test_20260521T235902Z/`
- `runs/analysis_dueefin_backend_report_20260522/vllm_bf16_merged_seed13_limit128/sarge_infer_DuEE-Fin-dev500_test_20260522T001941Z/`
