# DeepSeek API Diagnostics 20260522

Generated from `paper/exp/data/api_diagnostics/deepseek_api_diagnostics_20260522.json`.

## Scope

- Dataset/split: `DuEE-Fin-dev500 / dev`.
- Local root: `/home/tjk/myProjects/masterProjects/DEE/SARGE/`.
- Server root: `/data/TJK/DEE/SARGE/`.
- Runner: `src/sarge/experiments/deepseek_api_eval.py`.
- These are API/CPU diagnostics; `uses_gpu=false`, `concurrency=100`, and they are not paper main-table evidence.
- Secrets are not archived: `.env` is ignored and manifests record `api_key_recorded=false`.

## Main Finding

The low first DeepSeek scores were not caused by the API connection itself. The first `limit500` full-prompt runs produced many parse failures because the response budget was too small. After rerunning with a larger response budget, flash reached Legacy-FS F1 `0.4529` and pro reached `0.4348` on dev500. Temporary value-normalization probes lifted the numbers to `0.5046` and `0.4849`, so the dominant residual gap is value-surface mismatch and extraction format mismatch, not GPU or transport failure.

## Full Dev500 Runs

| Model | Prompt mode | Docs | Legacy-FS F1 | P | R | Accepted events | Parse errors | API errors | API seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash` | `role_safe_surface_memory` | 500 | 0.4529 | 0.5113 | 0.4066 | 528 | 1 | 0 | 90.0 |
| `deepseek-v4-pro` | `role_safe_surface_memory` | 500 | 0.4348 | 0.5249 | 0.3711 | 483 | 7 | 0 | 196.3 |
| `deepseek-v4-flash` | `role_safe_surface_only` | 500 | 0.4479 | 0.4972 | 0.4075 | 537 | 0 | 0 | 78.6 |
| `deepseek-v4-pro` | `role_safe_surface_only` | 500 | 0.4373 | 0.5109 | 0.3823 | 518 | 9 | 0 | 203.0 |
| `deepseek-v4-flash` | `schema_only` | 500 | 0.4330 | 0.4972 | 0.3835 | 509 | 1 | 0 | 73.7 |
| `deepseek-v4-pro` | `schema_only` | 500 | 0.4158 | 0.4820 | 0.3656 | 503 | 2 | 0 | 181.2 |

## Earlier Truncation Runs

| Model | Prompt mode | Docs | Legacy-FS F1 | Accepted events | Parse errors | API seconds | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| `deepseek-v4-flash` | `role_safe_surface_memory` | 500 | 0.2580 | 217 | 222 | 55.6 | response-budget/truncation diagnostic, superseded by the 4096-token rerun |
| `deepseek-v4-pro` | `role_safe_surface_memory` | 500 | 0.1255 | 89 | 325 | 105.4 | response-budget/truncation diagnostic, superseded by the 4096-token rerun |

## Limit50 Prompt Sweep

| Model | Prompt mode | Docs | Legacy-FS F1 | P | R | Accepted events | Parse errors |
|---|---|---:|---:|---:|---:|---:|---:|
| `deepseek-v4-flash` | `direct_json` | 50 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| `deepseek-v4-pro` | `direct_json` | 50 | 0.0000 | 0.0000 | 0.0000 | 0 | 4 |
| `deepseek-v4-flash` | `role_safe` | 50 | 0.4235 | 0.4760 | 0.3814 | 53 | 0 |
| `deepseek-v4-pro` | `role_safe` | 50 | 0.4222 | 0.4777 | 0.3782 | 53 | 0 |
| `deepseek-v4-flash` | `role_safe_slot_plan_only` | 50 | 0.4196 | 0.4615 | 0.3846 | 53 | 0 |
| `deepseek-v4-pro` | `role_safe_slot_plan_only` | 50 | 0.4314 | 0.4859 | 0.3878 | 51 | 0 |
| `deepseek-v4-flash` | `role_safe_surface_memory` | 50 | 0.3826 | 0.4974 | 0.3109 | 45 | 1 |
| `deepseek-v4-pro` | `role_safe_surface_memory` | 50 | 0.3694 | 0.5472 | 0.2788 | 37 | 1 |
| `deepseek-v4-flash` | `role_safe_surface_only` | 50 | 0.4428 | 0.5105 | 0.3910 | 53 | 0 |
| `deepseek-v4-pro` | `role_safe_surface_only` | 50 | 0.3707 | 0.4660 | 0.3077 | 48 | 2 |
| `deepseek-v4-flash` | `schema_only` | 50 | 0.4521 | 0.5187 | 0.4006 | 53 | 0 |
| `deepseek-v4-pro` | `schema_only` | 50 | 0.4377 | 0.4920 | 0.3942 | 52 | 1 |

## Temporary Value Normalization Probe

| Model | Prompt mode | Legacy-FS F1 | P | R | Note |
|---|---|---:|---:|---:|---|
| `deepseek-v4-flash` | `role_safe_surface_memory` | 0.5046 | 0.5696 | 0.4530 | temporary diagnostic, not production evaluator |
| `deepseek-v4-flash` | `role_safe_surface_only` | 0.4999 | 0.5550 | 0.4548 | temporary diagnostic, not production evaluator |
| `deepseek-v4-pro` | `role_safe_surface_memory` | 0.4849 | 0.5854 | 0.4138 | temporary diagnostic, not production evaluator |
| `deepseek-v4-pro` | `role_safe_surface_only` | 0.4901 | 0.5726 | 0.4284 | temporary diagnostic, not production evaluator |

## Interpretation Boundary

DeepSeek API runs are useful for checking whether the SARGE prompt modules can be attached to an external OpenAI-compatible model without GPU. They should not be compared directly with the fine-tuned HF/vLLM SARGE rows as model-performance evidence because the API models are not the trained SARGE checkpoint and their output surface conventions differ.
