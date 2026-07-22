# SARGE Handoff

> Last updated: 2026-05-23 08:45 UTC+8
> Project: CCKS 2026 main submission candidate
> Current status: seed13 test evidence consolidated; DuEE-Fin seed17/42, ChFinAnn seed17/42, DuEE-Fin HF no_slot_plan diagnostics, and ChFinAnn vLLM gmem=0.80 module fast-screen diagnostics are synced. A 2026-05-23 08:45 UTC+8 read-only process query found no active SARGE jobs on `gpu-4090`.

---

## 1. Paths And Environments

| Resource | Local | Server |
|---|---|---|
| Project root | `/home/tjk/myProjects/masterProjects/DEE/SARGE/` | `/data/TJK/DEE/SARGE/` |
| Python | `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python` | `/data/TJK/envs/sarge_vllm_full/bin/python` |
| Data | `data/` | `data/` |
| Models | `models/` | `models/` |
| Qwen3-4B | `models/Qwen/Qwen3-4B-Instruct-2507` | `models/Qwen/Qwen3-4B-Instruct-2507` |
| Evaluator | `evaluator/` | `evaluator/` |
| Runs | small JSON snapshots under `paper/exp/data/run_snapshots/` | `/data/TJK/DEE/SARGE/runs/` |

Offline runtime flags:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
```

GPU jobs run only on `gpu-4090`. Prefer an idle GPU, set `CUDA_VISIBLE_DEVICES`, and never kill non-`TJK` processes.

---

## 2. Repository State

The latest read-only server refresh before this consolidation saw `main` at:

```text
1849d7a55d9701148b4f4b83509c947d11c93e6d
```

Earlier consolidated artifacts also reference `ff0761e88e456ea001429be86d892375bec36349`; keep both commits in the registry because runs were produced before and after the ablation-profile support commit.

This handoff intentionally keeps runtime logs and server `runs/` as server artifacts. Git tracks only source, docs, paper assets, and small JSON evidence snapshots needed for reproducible tables.

---

## 3. Authoritative Evidence Layer

| Artifact | Path |
|---|---|
| Experiment registry | `paper/exp/data/asset_registry.json` |
| Small run snapshots | `paper/exp/data/run_snapshots/` |
| Generated experiment summary | `paper/exp/seed13_summary.md` |
| Markdown tables | `paper/exp/tables/` |
| Current result narrative | `docs/exp_result.md` |
| ACL-family draft | `paper/emnlp_aacl_draft/main.tex` |
| Draft build script | `paper/emnlp_aacl_draft/scripts/build_assets.py` |

Main paper numbers must come from real non-mock runs with complete eval JSON and traceable manifests. Running jobs stay in status tables until their eval files exist.

---

## 4. Current Main Results

The main comparison metric is Legacy-FS / `legacy_doc2edag`. Unified-Strict, DocFEE, and ExactRec are diagnostics and must not be mixed into the main baseline table.

| Dataset | Split | Seed | Main setting | Legacy-FS F1 | Unified F1 | DocFEE F1 | ExactRec |
|---|---|---:|---|---:|---:|---:|---:|
| ChFinAnn-Doc2EDAG | test | 13 | HF-4bin + LoRA, k=1 greedy | 0.8603 | 0.8742 | 0.8653 | 0.5842 |
| DuEE-Fin-dev500 | test | 13 | HF-4bin + LoRA, k=1 greedy, no-LRD | 0.7796 | 0.7888 | 0.7771 | 0.4285 |

ChFinAnn was promoted from the older vLLM BF16 line to the HF-4bin line after full-test backend cross-check. DuEE-Fin remains HF-4bin no-LRD as the main path.

---

## 5. Important Findings

- Greedy `k=1` is the default production path; `k=4` sampling did not improve either dataset in completed test ablations.
- LoRA SFT is the dominant gain source. No-SFT baselines are very low, especially DuEE-Fin HF no-SFT F1 `0.0330`.
- vLLM BF16 is useful as a diagnostic backend, but it underperforms HF-4bin on both completed full-test backend checks.
- DuEE-Fin HF seed17/42 test diagnostics are complete and synced: Legacy-FS F1 `0.7872` and `0.7828`; HF seeds 13/17/42 mean±std is `0.7832±0.0038`.
- DuEE-Fin vLLM BF16 seed13/17/42 backend diagnostics are complete and synced: Legacy-FS F1 `0.7502`, `0.7470`, and `0.7583`; backend rows remain diagnostic only.
- ChFinAnn HF seed17/42 test diagnostics are complete and synced: Legacy-FS F1 `0.8536` and `0.8533`; ChFinAnn HF seeds 13/17/42 mean±std is `0.8557±0.0039`.
- DuEE-Fin vLLM prompt-module fast screen is contradictory under `gpu_memory_utilization=0.70`: `no_surface_memory` and `no_slot_plan` collapse to `0.0208` / `0.0164`, while `no_surface_or_slot` stays at `0.7549`. Treat this as backend/prompt interaction evidence, not final module proof.
- DuEE-Fin HF main-backend `no_surface_memory` completed at Legacy-FS F1 `0.7812`, effectively matching the HF full row `0.7796`; HF `no_slot_plan` completed at `0.7758`, giving only weak positive evidence for Slot Plan and no vLLM-style collapse.
- ChFinAnn vLLM BF16 gmem=0.80 module fast-screen completed: full `0.8547`, `no_surface_memory` `0.8538`, and `no_slot_plan` `0.8567`. This weakens any claim that Surface Memory or Slot Plan are stable positive modules; `no_slot_plan` improves recall and multi-event F1 in this diagnostic row.
- SFT training uses surface candidates but sets `slot_plan=None`; inference Slot Plan is a train-prior prompt, not a learned planner. Do not present Slot Plan as a trained contribution unless a new predicted-plan training/evaluation design is added.
- DeepSeek API diagnostics are CPU/API-only, not GPU jobs. The 4096-token rerun reached dev500 Legacy-FS F1 `0.4529` for flash and `0.4348` for pro; these runs are diagnostic only and are archived under `paper/exp/data/api_diagnostics/`.
- Safe-anchor LRD on DuEE-Fin test changes Legacy-FS F1 only from `0.7796` to `0.7800`; keep it diagnostic/appendix unless a later fair candidate-contract run shows a real gain.
- Seed17 dev LRD F1 `0.3354` is invalid as a performance number because it used all k=4 parsed candidates instead of MRS-selected/fair k=1-compatible candidates.

---

## 6. Active Remote Jobs

| Task | GPU | Status | Log |
|---|---:|---|---|
| none | - | no active SARGE training/inference/eval process in 2026-05-23 08:45 UTC+8 read-only query | - |

For future completed runs, first inspect log tail and output tree, then pull only JSON summaries/manifests/eval/diagnostics into `paper/exp/data/run_snapshots/`. Do not pull checkpoints, full prediction JSONL, raw outputs, or parsed candidates into Git.

---

## 7. Key Commands

Local validation:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m pytest tests/ -v
```

Server eval for completed inference:

```bash
cd /data/TJK/DEE/SARGE
/data/TJK/envs/sarge_vllm_full/bin/python -B scripts/eval_three_tracks.py \
  --run-root runs/<run_name>/<inner_run_name> \
  --dataset <DuEE-Fin-dev500|ChFinAnn-Doc2EDAG> \
  --split test
```

Rebuild local experiment summary:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B paper/exp/scripts/build_seed13_summary.py
```

Build ACL-family draft assets:

```bash
cd /home/tjk/myProjects/masterProjects/DEE/SARGE/paper/emnlp_aacl_draft
./build.sh
```

---

## 8. Next Work

1. Do not add a Slot Plan training plan merely to rescue the module. Current paper route should emphasize schema-grounded event-table generation as a method for part of Chinese financial DEE, not a technical-report pipeline or SOTA claim.
2. If the paper insists on Slot Plan as a contribution, design a separate predicted-plan experiment with no-plan / train-prior / predicted-plan / oracle-plan upper bound; dev/test oracle plans must not enter the main table.
3. Regenerate `paper/exp/seed13_summary.md` after any new completed eval snapshot.
4. Keep LRD fair-candidate policy explicit: no main LRD result from all k=4 parsed candidate pools.
5. Keep paper claims centered on schema adherence, role grounding, evaluator-compatible records, and the method's boundary; do not claim that SARGE solves homogeneous-record binding.
