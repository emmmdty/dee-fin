# P2 DuEE-Fin acceptance measurement

> Date: 2026-05-14
> Run dir: `runs/carve/p2_duee_fin_seed42_acceptance`
> Boundary: non-smoke acceptance run only. Not unified-strict, not hidden-test, not final-test, not main-table.

## Run configuration

| Field | Value |
|---|---|
| Dataset | DuEE-Fin-dev500 |
| Train documents | 1892 (multi_event_subset) |
| Dev documents | 146 (multi_event_subset) |
| Encoder | `models/chinese-roberta-wwm-ext_safetensors` (offline, safetensors, local-only) |
| Seed | 42 |
| Max epochs | 10 |
| Encoder LR | 2e-5 |
| Heads LR | 5e-4 |
| Grad accumulation | 8 |
| Warmup ratio | 0.1 |
| Lambda ground MI | 0.5 |
| Elapsed | 1184.7s (19.7 min) |

## Acceptance verdict

| Criterion | Threshold | Final value | Status |
|---|---|---|---|
| train `evidence_bce` | ≤ 0.30 | 0.113240 | PASS |
| train `pointer_mi` | ≤ 1.5 | 0.338068 | PASS |
| dev `unalignable_rate` | ≤ 0.30 | 0.013639 | PASS |

All three frozen P2 acceptance criteria pass. The phase gate that R3/P3 acceptance depends on (`docs/phase/p3_mention_planner_plan.md` line 44: "all P2 acceptance checks remain passing") is closed.

## Train loss trajectory

| Epoch | loss | evidence_bce | pointer_mi |
|---:|---:|---:|---:|
| 1 | 2.337891 | 1.009687 | 2.656298 |
| 2 | 0.896845 | 0.355330 | 1.083024 |
| 3 | 0.638102 | 0.283497 | 0.709306 |
| 4 | 0.515641 | 0.244093 | 0.543049 |
| 5 | 0.441922 | 0.208956 | 0.465775 |
| 6 | 0.385692 | 0.168191 | 0.435094 |
| 7 | 0.349710 | 0.146577 | 0.406203 |
| 8 | 0.322383 | 0.130766 | 0.383138 |
| 9 | 0.301472 | 0.122174 | 0.358599 |
| 10 | 0.282273 | 0.113240 | 0.338068 |

Crossings:
- `train evidence_bce <= 0.30` crossed at epoch 3 (0.2835).
- `train pointer_mi <= 1.5` crossed at epoch 2 (1.0830).

## Dev evidence metrics

| Field | Value |
|---|---|
| `aligned_args` | 1591 |
| `unalignable_args` | 22 |
| `unalignable_rate` | 0.013639 |
| `evidence_bce` | 0.114657 |
| `pointer_mi` | 0.591746 |

Total dev gold arguments after alignment: 1613. 98.6% are alignable inside their gold sentence; the 22 unalignable args are excluded from `L_ground^MI` per the frozen P2 Decision L4.

## Hardware context

Run launched by `/tmp/p2_auto_launch.sh` polling launcher on `gpu-4090`. The launcher waited until another user's job released GPU 0 (DJB yolov11, taking 17-18 GB) and then auto-started P2 once free memory exceeded 18 GB. Launcher logs at `/tmp/p2_launcher.log`. P2 stdout at `/tmp/p2_acceptance.log`.

## Boundaries

This measurement is the P2 phase acceptance closure. It does not assert:

- R3 v2 acceptance (separate phase; presence head passes, count head still fails — see `docs/phase/p3_replan.md`).
- P5b dev diagnostic improvement against `runs/carve/p5b_duee_fin_dev500_seed42` rule-path evidence.
- Any unified-strict, hidden-test, or final-test result.
- Any paper main-table or SOTA claim.

The P2 checkpoint at `runs/carve/p2_duee_fin_seed42_acceptance/checkpoints/p2.pt` is a phase artifact and is not promoted to a published encoder.
