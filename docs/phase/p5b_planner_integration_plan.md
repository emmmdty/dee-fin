# P5b Planner Integration Plan

Status: implemented locally on 2026-05-14 (in the same session that closed the R3 v2 implementation). Server smoke against a real R3 checkpoint is gated by R3 v2 non-smoke completion. P5b acceptance (dev diagnostic improvement) remains gated by R3 v2 acceptance, which is not yet closed.

The original "design only" stub is preserved below for context; the **Implementation Notes** section at the end records what changed and what is still pending.

## Purpose

Today's P5b runner determines event-type presence with `carve/p5b_runner.py::_type_gate` (a string match between normalized event type and document title+text) and estimates record count with `_estimate_record_count` (a clipped trigger-word count). These are rule baselines that the lexical_trigger baseline in `carve/p3_planner_only_runner.py` already mirrors. If an R3 trained planner is accepted, P5b should be able to load that checkpoint and replace the rule path with the trained two-stage planner.

This document records the integration interface so that, once R3 v2 acceptance closes, the P5b change is a localized refactor and not a redesign.

## Preconditions

This plan does not execute until all of the following are met:

- R3 v2 non-smoke acceptance on `gpu-4090` is closed and recorded in `docs/measurements/`.
- P2 non-smoke acceptance is closed and recorded.
- A new phase is opened with explicit acceptance criteria for P5b under the trained planner. The current `docs/measurements/p5b_duee_fin_dev500_seed42.md` evidence is bound to the rule-based runner and cannot be reused.

If any of the above is unmet, this stub stays informational only.

## Interface

The runner reuses the checkpoint format already produced by `carve/p3_planner_only_runner.py::run_r3_planner_only`:

```python
torch.save({
    "planner": planner.state_dict(),
    "planner_metadata": {
        "event_types": list[str],
        "hidden_size": int,
        "k_clip": int,
        "presence_pos_weight": float,
        "presence_threshold_multi_event_dev": float,
        "presence_threshold_all_dev": float,
        "acceptance_population": list[str],
        "train_population": str,
        "encoder_feature_mode": str,
        "max_sentences": int,
    },
}, ...)
```

Required CLI additions on `carve/p5b_runner.py::build_arg_parser`:

- `--planner-checkpoint`: path to the `r3_planner.pt` artifact.
- `--planner-encoder-path`: path to the same `RobertaSentenceEncoder` safetensors used during R3 training.
- `--planner-feature-mode`: `{global_only, evidence, evidence_lexical}`. Must match the mode used to train the checkpoint.
- `--planner-presence-threshold`: float; defaults to `presence_threshold_all_dev` from metadata.

Behavioral changes inside `_predict_route`:

- `_type_gate(document, event_type)` is replaced with a call into the trained `RecordPlanner.type_gate.predict_present`. The wrapper encodes the document with `RobertaSentenceEncoder`, builds `evidence_vec` and `lexical_hit` according to `--planner-feature-mode`, and reuses `_type_gate` only when building `lexical_hit`. The lexical detector is kept as a feature, not as a gate.
- `_estimate_record_count` is replaced with `RecordPlanner.count_planner.predict_count`. The 1..3 clip is dropped because the trained planner already produces zero-truncated Poisson counts up to `k_clip`.
- When `--planner-checkpoint` is not provided, the runner falls back to the current rule-based behavior. This preserves the existing `docs/measurements/p5b_duee_fin_dev500_seed42.md` baseline.

## Non-Goals

- No P5b acceptance claim from this stub.
- No update to `docs/measurements/p5b_duee_fin_dev500_seed42.md`.
- No change to the AllocationDiagnosticModel or other P5b components.
- No automatic re-training inside P5b. The R3 planner is loaded read-only.

## Risks

- The R3 checkpoint encodes documents through `RobertaSentenceEncoder`. P5b currently does not load this encoder. Loading it doubles P5b runtime in the integration path. Document the new runtime in `docs/measurements/` only after the new phase runs.
- If `--planner-feature-mode` mismatches the training mode, the planner produces meaningless logits. The runner must compare metadata and refuse to load on mismatch.
- The presence threshold is calibrated on R3 train+dev. Applying it to a different P5b dev split may be miscalibrated. The integration phase must add a recalibration step or accept the calibrated value with a recorded caveat.

## Validation (when this stub is promoted to an active phase)

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B \
  -m unittest tests.carve.test_p5b_runner -v
```

New tests required at integration time:

- planner checkpoint loads without modifying P5b output when `--planner-checkpoint` is absent.
- planner checkpoint with mismatched `--planner-feature-mode` raises.
- when the checkpoint is loaded, `_type_gate` outputs are derived from `RecordPlanner.type_gate.predict_present`, not from the string match.

A server smoke (toy data, 2 epochs) is required before any DuEE-Fin dev rerun, and the new dev diagnostic JSON must be written to a fresh path (for example `runs/carve/p5b_duee_fin_dev500_planner_v1_seed42/`) so it does not overwrite the existing rule-based evidence.

## Implementation Notes (2026-05-14)

The integration is now implemented in `carve/p5b_runner.py`:

- New class `PlannerGate` wraps `RecordPlanner` plus optional `RobertaSentenceEncoder`. It loads the checkpoint produced by `carve.p3_planner_only_runner`, refuses to load when `--planner-feature-mode` does not match the checkpoint metadata, and exposes a single `predict(document, event_type) -> (present, count)` method that respects the configured feature mode.
- `build_arg_parser` adds `--planner-checkpoint`, `--planner-encoder-path`, `--planner-feature-mode {global_only, evidence, evidence_lexical}`, `--planner-presence-threshold`.
- `run_p5b` constructs `PlannerGate` only when `--planner-checkpoint` is set; otherwise behavior is unchanged.
- `_predict_route` accepts an optional `planner_gate=None`. When provided, the neural planner replaces both `_type_gate` and `_estimate_record_count` for that iteration; rule-based behavior is unaffected when the flag is absent.

New tests in `tests/carve/test_p5b_runner.py`:

- `test_planner_gate_loads_with_valid_global_only_checkpoint`
- `test_planner_gate_feature_mode_mismatch_raises`
- `test_planner_gate_predict_uses_trained_biases`
- `test_smoke_run_with_planner_gate_overrides_type_gate`

All four pass under `__toy__` mode without GPU. All 43 `tests/carve` and 39 `tests/evaluator` tests remain green.

### What this implementation does not do

- It does not assert any P5b behavior improvement. A server smoke against the actual `runs/carve/r3_planner_only_duee_fin_seed42_v2/checkpoints/r3_planner.pt` is the next step once R3 v2 non-smoke finishes. Even if that smoke runs, P5b acceptance still requires:
  - R3 v2 non-smoke acceptance (currently in progress, not asserted by this commit).
  - A separate P5b phase doc with explicit acceptance criteria.
  - A fresh dev diagnostic write path so the existing `runs/carve/p5b_duee_fin_dev500_seed42*` artifacts are not overwritten.
- It does not change the rule-based fallback. The runner is bit-for-bit identical to v1 when `--planner-checkpoint` is empty.
- It does not load multiple planners or auto-select feature modes. Both must match the checkpoint exactly.
- It does not recalibrate the presence threshold. The threshold from checkpoint metadata is used as-is, with `--planner-presence-threshold` as the override hatch.

### Open questions for the follow-up phase

- Should presence threshold be recalibrated on the P5b dev split, or is the R3 train+dev calibration acceptable? Document the choice in the P5b acceptance phase doc.
- For `evidence_lexical` mode, `_type_gate` is used both as a P5b baseline and as a feature into the planner. The P5b dev diagnostic must report the lexical baseline alongside the neural-with-lexical path so the marginal contribution of the neural part is auditable.
- Should P5b use a different lexical detector than `_type_gate` (e.g., per-sentence event-type keyword detection)? Out of scope for this stub; record it as a method risk in the follow-up phase doc.
