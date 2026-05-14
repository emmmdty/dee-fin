# R3 v3 Phase: Sentence-Level CountPlanner

Date: 2026-05-14
Status: Plan locked, awaiting server execution.

## Context

R3 v2.1 (commit `89e0341`, `runs/carve/r3_planner_only_duee_fin_seed42_v2_1/`) closed the TypeGate question (4/4 acceptance checks pass; AUC ≈ 0.98–0.99, F1 ≈ 0.78–0.83 dominating the strongest baseline by 15–20 absolute F1 points) and isolated the CountPlanner as a structural failure:

- `multi_event_dev/count_mae_positive = 0.8702` vs `predict_one = 0.8365`
- `all_dev/count_mae_positive = 0.4040` vs `predict_one = 0.348`
- `training/count_loss_trend = 1.34×` (weighted Poisson NLL reshapes the curve below the 2× gate)

The v2.1 measurement (`docs/measurements/r3_planner_only_duee_fin_seed42_v2_1.md:84-87`) and the v2.1 replan note (`docs/phase/p3_replan.md:243-258`) converge on the same conclusion: a document-level pooled feature set `[global_repr; type_emb; evidence_vec; lexical_hit]` collapses per-occurrence evidence into a single attention pool and cannot distinguish "doc contains 1 record of type X" from "doc contains 3 records of type X". The external audit (`docs/diagnostics/current_stage_external_audit_20260514.md:248-254`) further requires that any v3 acceptance contract use baseline-relative gates rather than absolute thresholds.

v3 replaces the document-level CountPlanner with a sentence-level scorer that predicts a per-sentence binary "this sentence contains a record of type X" label, aggregating sigmoid scores into an expected count `n_t = Σ_s σ(logit_st) · mask_s`. TypeGate (and its 4 acceptance checks) is untouched.

## Methodology

### 1. Sentence-level supervision

DuEE-Fin records carry no explicit sentence pointer. v3 derives `y_st ∈ {0, 1}` heuristically, mirroring the alignment style of `carve/p3_runner.py:537-553` (`_gold_values_for_sentence`):

```text
for each (doc, sentence s, event_type t):
    y_st = 1 if there exists a gold record r with
              normalize(r.event_type) == normalize(t)
              and some argument value v in r with normalize(v) ⊂ normalize(s.text)
              and the per-record argument hit count ≥ MIN_ARG_HITS
           else 0
```

`MIN_ARG_HITS = 1` by default. The heuristic over-counts in two known ways: (a) a single record argument can match multiple sentences if the surface form repeats, inflating `Σ_s y_st` above the true `gold_n_t`; (b) records whose argument values appear only in the title or as pronouns may yield `Σ_s y_st = 0` for a positive (doc, type) pair, falsely zeroing the supervision. Both are quantified and reported as run prerequisites in §3.

### 2. SentenceLevelCountPlanner

New module in `carve/p3_planner.py`. `CountPlanner` and `truncated_poisson_nll` are retained byte-for-byte so v2.1 remains a callable baseline.

```python
class SentenceLevelCountPlanner(nn.Module):
    def __init__(self, hidden_size: int, num_event_types: int):
        self.type_embedding = nn.Embedding(num_event_types, hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # [sent_h; type_emb; sent_h * type_emb]
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def sentence_logits(self, sentence_repr, type_id) -> [B, S]: ...
    def predict_expected_count(self, sentence_repr, sentence_mask, type_id) -> [B]: ...
```

`predict_expected_count` returns `round(Σ_s σ(logit) · mask).clamp(min=1).to(long)` — clamp to 1 mirrors v2's `truncated_poisson_argmax` convention that the count branch always returns ≥1 when invoked.

`RecordPlanner.__init__` takes `count_head_mode: str = "document"`. When `"sentence"`, it instantiates `SentenceLevelCountPlanner` instead of `CountPlanner`. A new `predict_n_t_with_sentences(...)` codepath dispatches the count branch through the sentence head when present.

### 3. Loss

```python
def sentence_count_loss(sent_logits, sent_labels, sent_mask, pos_weight) -> Tensor:
    raw = F.binary_cross_entropy_with_logits(
        sent_logits, sent_labels.float(), pos_weight=pos_weight, reduction="none",
    )
    mask = sent_mask.float()
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)
```

`pos_weight = min(negatives / positives, --sentence-bce-pos-weight-cap)` where the counts are taken over all `(doc, sentence, type)` triples in the **train cache** that have `sentence_mask == True`. Cap defaults to 20.

The v3 total loss is `λ_presence · presence_loss + λ_count · sentence_count_loss`. The Poisson NLL term is **removed** from the v3 count branch; in `document` mode (default), the v2.1 Poisson path is unchanged.

### 4. Cache and runner changes

`PlannerFeatureCache` gains `sentence_record_label: Tensor[N_docs, S_max, num_event_types] (uint8)`. `_build_feature_cache` populates it in the same pass that already computes `gold_n_t` from `cache.documents` (no extra encoder pass).

Two noise diagnostics are computed per cache and written to `summary.json`:

- `gold_record_sentence_recall = P(Σ_s y_st ≥ 1 | gold_n_t > 0)`
- `mean_sentence_label_count_over_gold = E[Σ_s y_st / gold_n_t | gold_n_t > 0]`

### 5. CLI surface

```
--count-head-mode {document, sentence}    default=document
--sentence-label-min-hits int             default=1
--sentence-bce-pos-weight-cap float       default=20.0
```

Defaults preserve v2.1 byte-for-byte. v3 acceptance runs use `--count-head-mode sentence`.

### 6. Checkpoint metadata + P5b dispatch

The R3 checkpoint adds `planner_metadata.count_head_mode`. `carve/p5b_runner.PlannerGate` reads this field and dispatches: `document` mode keeps the v2.1 path; `sentence` mode instantiates a `SentenceLevelCountPlanner` inside `RecordPlanner` and calls `predict_expected_count`. Old checkpoints without the field default to `document`, preserving backward compatibility.

## Acceptance Criteria (frozen before any v3 run)

`accepted=true` iff **all** of the following 10 checks pass on `runs/carve/r3_planner_only_duee_fin_seed42_v3/summary.json`:

| # | Check | Rule |
|---|---|---|
| 1 | `multi_event_dev/type_gate_auc` | abs ≥ 0.80 **AND** ≥ best_baseline + 0.05 (v2.1 standard, regression guard) |
| 2 | `multi_event_dev/type_gate_f1_youden` | abs ≥ 0.55 **AND** ≥ best_baseline + 0.05 |
| 3 | `all_dev/type_gate_auc` | abs ≥ 0.80 **AND** ≥ best_baseline + 0.05 |
| 4 | `all_dev/type_gate_f1_youden` | abs ≥ 0.55 **AND** ≥ best_baseline + 0.05 |
| 5 | `multi_event_dev/count_mae_positive` | **baseline-relative-only**: value ≤ `predict_one − 0.05` (target ≤ ~0.7865) |
| 6 | `all_dev/count_mae_positive` | **baseline-relative-only**: value ≤ `predict_one − 0.02` (target ≤ ~0.328) |
| 7 | `training/presence_loss_trend` | first/last ≥ 2.0 over ≥10 epochs (v2.1 criterion, unchanged) |
| 8 | `training/sentence_count_loss_trend` | first/last ≥ 1.5 over ≥10 epochs |
| 9 | `multi_event_dev/sentence_score_auc` | ≥ 0.75 (sentence head sanity, global AUC over masked triples) |
| 10 | `all_dev/sentence_score_auc` | ≥ 0.75 |

Check 5/6 thresholds are computed **at run time** from `baselines[population]["predict_one"]["count_mae_positive"]` minus the margin, then `_acceptance_checks` reports both `absolute_threshold` and `baseline_relative_threshold` set to that value — collapsing into a single baseline-relative gate as recommended in `docs/diagnostics/current_stage_external_audit_20260514.md:209-215`.

Check 8 is rewritten from v2.1's `count_loss_trend` because BCE-with-logits + high `pos_weight` does not decay as fast as zero-truncated Poisson NLL; the 1.5× threshold reflects the actual convergence shape we expect for a per-sentence binary head and is documented here so it cannot be retro-fitted to make a borderline run pass.

### Path A fallback (user-confirmed 2026-05-14)

If checks 5 or 6 fail on the v3 long run, **do not** spin v3.1 / v3.2 retraining loops. The phase is instead closed as **TypeGate-only acceptance**:

- Checks 1–4 pass → record R3 TypeGate as accepted methodology.
- Count head: documented as structural bottleneck. P5b uses the existing lexical `_estimate_record_count` fallback (`carve/p5b_runner.py:621-628`).
- Checks 5/6/8/9/10 are reported but do not block the phase; the measurement file states explicitly that the count branch did not clear baseline.

This Path A invocation must be recorded as the final paragraph in both `docs/measurements/r3_planner_only_duee_fin_seed42_v3.md` and this phase doc's Acceptance Result section.

## Validation Commands

### Local (mandatory before any server work)

```bash
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/carve -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/evaluator -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest tests.data_split.test_split_utils -v
PYTHONDONTWRITEBYTECODE=1 /home/tjk/miniconda3/envs/feg-dev-py310/bin/python -B -m unittest discover -s tests/baseline/procnet -v
```

All four must exit 0.

### Server rsync (additive only)

Working directory: `/home/tjk/myProjects/masterProjects/DEE/dee-fin` on the local box.

```bash
rsync -avz --exclude='.git' --exclude='runs/' --exclude='__pycache__/' --exclude='.claude/' \
  /home/tjk/myProjects/masterProjects/DEE/dee-fin/ \
  gpu-4090:/data/TJK/DEE/dee-fin/
```

No `--delete`. Per `AGENTS.md:34`, sync is strictly additive.

### Server smoke (gpu-4090)

Working directory: `/data/TJK/DEE/dee-fin`.

```bash
ssh gpu-4090 "cd /data/TJK/DEE/dee-fin && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/TJK/.conda/envs/tjk-feg/bin/python -u -m scripts.carve.run_r3_planner_only \
    --dataset DuEE-Fin-dev500 \
    --data-root data/processed/DuEE-Fin-dev500 \
    --schema data/processed/DuEE-Fin-dev500/schema.json \
    --model-path models/chinese-roberta-wwm-ext_safetensors \
    --run-dir runs/carve/r3_planner_only_duee_fin_seed42_v3_smoke \
    --seed 42 --max-epochs 2 --batch-size 8 --smoke \
    --encoder-feature-mode evidence_lexical \
    --count-head-mode sentence"
```

Expected outputs:
- `summary.json` containing `count_head_mode: sentence`, 10-key `acceptance_checks`, the two noise diagnostics
- `diagnostics/r3_planner_{train_history,metrics,baselines}.json`
- `checkpoints/r3_planner.pt` (payload metadata includes `count_head_mode`)
- `cache/{all_train,multi_event_dev,all_dev}.pt`

Exit 0; elapsed comparable to the v2 smoke (~5–30 s). Smoke gate before promoting to the long run:

1. Exit 0.
2. `gold_record_sentence_recall ≥ 0.6` on the train cache (otherwise the heuristic alignment is too lossy and we stop to investigate).
3. TypeGate metrics shape consistent with v2.1 (AUC > 0.9 even on 16-doc smoke is unreasonable; AUC ≥ 0.5 is enough to confirm the head is wired correctly).

### Server long run (gpu-4090) — authorized by user 2026-05-14

```bash
ssh gpu-4090 "cd /data/TJK/DEE/dee-fin && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/TJK/.conda/envs/tjk-feg/bin/python -u -m scripts.carve.run_r3_planner_only \
    --dataset DuEE-Fin-dev500 \
    --data-root data/processed/DuEE-Fin-dev500 \
    --schema data/processed/DuEE-Fin-dev500/schema.json \
    --model-path models/chinese-roberta-wwm-ext_safetensors \
    --run-dir runs/carve/r3_planner_only_duee_fin_seed42_v3 \
    --seed 42 --max-epochs 50 --batch-size 64 \
    --encoder-feature-mode evidence_lexical \
    --count-head-mode sentence"
```

Reference elapsed: v2.1 with identical configuration ran 668 s (`docs/measurements/r3_planner_only_duee_fin_seed42_v2_1.md:31`). v3 should be within ±25%.

### P5b integration smoke (after the long run completes)

```bash
ssh gpu-4090 "cd /data/TJK/DEE/dee-fin && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/TJK/.conda/envs/tjk-feg/bin/python -u -m scripts.carve.run_p5b_diagnostic \
    --dataset DuEE-Fin-dev500 \
    --data-root data/processed/DuEE-Fin-dev500 \
    --schema data/processed/DuEE-Fin-dev500/schema.json \
    --run-dir runs/carve/p5b_duee_fin_dev500_planner_v3_smoke \
    --seed 42 --max-epochs 2 --routes baseline,carve --smoke \
    --planner-checkpoint runs/carve/r3_planner_only_duee_fin_seed42_v3/checkpoints/r3_planner.pt \
    --planner-encoder-path models/chinese-roberta-wwm-ext_safetensors \
    --planner-feature-mode evidence_lexical"
```

This verifies `PlannerGate` dispatches the sentence-mode checkpoint without raising. It is **not** a P5b acceptance claim; per the audit, P5b acceptance requires its own phase doc.

## Scope Boundaries

- No change to `./data` or `./baseline` (`AGENTS.md:5-6`).
- No change to the `TypeGate` class, `CountPlanner` class, `truncated_poisson_nll`, or v2.1 acceptance checks 1–4 (regression guard).
- No hidden / test / final runs. No paper main-table content.
- No three-dataset rollout (ChFinAnn, DocFEE remain out of scope).
- `--count-head-mode document` remains the default, so any caller that does not opt in keeps v2.1 behavior.
- No P5b acceptance promotion. The Step 8 smoke only verifies the dispatch path.

## Acceptance Result (2026-05-14)

Run: `runs/carve/r3_planner_only_duee_fin_seed42_v3` (50 epochs, batch 64, `evidence_lexical`, `count-head-mode sentence`, elapsed 588.4 s).

**8/10 PASS.** The two count_mae_positive gates (checks #5, #6) fail by a wide margin — v3 count MAE (5.61 / 4.66) is much worse than v2.1 (0.87 / 0.40). The sentence-level binary classifier trains correctly (sentence_score_auc 0.882 / 0.888, count_loss decreased 5.14×), but the expected-count aggregation compounds the ~4.5× over-counting inherent in the heuristic sentence labels.

### Path A triggered

Per the frozen Path A clause (§Path A fallback):

- **R3 TypeGate is accepted.** Checks 1–4 (TypeGate AUC ≥ baseline + 0.05, F1 ≥ baseline + 0.05 on both populations) pass by large margins and are stable across v2, v2.1, and v3.
- **Count head is out of scope.** Neither document-level Poisson NLL (v2/v2.1) nor sentence-level BCE aggregation (v3) clears the baseline-relative count gate. R3 does not claim a trained count estimator. The finding is documented as a structural limitation of DuEE-Fin's annotation: records lack sentence pointers, and heuristic alignment introduces ~4.5× over-counting noise that any neural aggregator amplifies.
- **P5b count fallback.** P5b continues to use `_estimate_record_count` (lexical trigger string count, capped at 3) as its count estimator. This is a deterministic fallback, not a neural count head.

Full result table and analysis in `docs/measurements/r3_planner_only_duee_fin_seed42_v3.md`.
