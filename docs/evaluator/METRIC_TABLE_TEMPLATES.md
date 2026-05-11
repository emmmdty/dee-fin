# Metric Table Templates

This document provides paper/report Markdown templates for the evaluation
protocol. Example rows use placeholders and are not reported results.

## 1. Historical Native Replay Table

Intended use:

- Historical baseline reproduction.
- Native ProcNet / Doc2EDAG-style event-table replay.
- Paper-reported historical scores when native artifacts are unavailable.

Columns:

- Model
- Dataset
- Split
- Evaluator
- Input Artifact
- P
- R
- F1
- Source
- Notes

Template:

| Model | Dataset | Split | Evaluator | Input Artifact | P | R | F1 | Source | Notes |
|---|---|---|---|---|---:|---:|---:|---|---|
| ProcNet s42 | ChFinAnn-Doc2EDAG | test | `legacy-doc2edag --input-format native-event-table` | `<procnet_s42_native_table.json>` | `<P>` | `<R>` | `<F1>` | native replay | Native event-table replay; not canonical JSONL. |
| Doc2EDAG | ChFinAnn-Doc2EDAG | test | reported paper/native metric | `<paper reported>` | `<P>` | `<R>` | `<F1>` | paper | Include only if no replay artifact is available. |
| This method | `<dataset>` | `<split>` | `legacy-doc2edag --input-format native-event-table` | `<native_artifact>` | `<P>` | `<R>` | `<F1>` | native replay | Include only if a comparable native artifact exists. |

Comparison notes:

- Compare rows only when they use the same historical/native protocol or are
  explicitly labeled as paper-reported historical scores.
- Do not compare these F1 values as `unified-strict` canonical role-value
  scores.
- Do not place exported canonical JSONL scores in this table unless they are in
  a separately labeled column.

## 2. Exported Canonical Evaluation Table

Intended use:

- Evaluate exported canonical JSONL predictions under both fixed-slot and
  canonical role-value metrics.
- Diagnose export-level compatibility and score drift.

Columns:

- Model
- Dataset
- Split
- legacy-doc2edag canonical-jsonl F1
- unified-strict F1
- Input prediction file
- Notes

Template:

| Model | Dataset | Split | legacy-doc2edag canonical-jsonl F1 | unified-strict F1 | Input prediction file | Notes |
|---|---|---|---:|---:|---|---|
| ProcNet exported s42 | ChFinAnn-Doc2EDAG | test | `<F1>` | `<F1>` | `<procnet_s42_predictions.jsonl>` | Exported canonical JSONL; not bit-exact native replay. |
| This method | DuEE-Fin-dev500 | test | `<F1>` | `<F1>` | `<method_predictions.jsonl>` | Fixed-slot and role-value columns answer different questions. |
| Qwen exported | `<dataset>` | `<split>` | `<F1>` | `<F1>` | `<qwen_predictions.jsonl>` | Include only if canonical predictions are available. |

Comparison notes:

- The `legacy-doc2edag canonical-jsonl F1` column is fixed-slot evaluation on
  exported canonical JSONL.
- The `unified-strict F1` column is canonical role-value evaluation.
- Do not describe the exported legacy column as ProcNet native training-report
  replay.

## 3. Unified Strict Main Table

Intended use:

- Main cross-dataset scientific comparison.
- Canonical role-value scoring across ChFinAnn, DuEE-Fin, and DocFEE.

Columns:

- Model
- ChFinAnn unified-strict F1
- DuEE-Fin unified-strict F1
- DocFEE unified-strict F1
- Average or macro summary if justified
- Notes

Template:

| Model | ChFinAnn unified-strict F1 | DuEE-Fin unified-strict F1 | DocFEE unified-strict F1 | Average or macro summary if justified | Notes |
|---|---:|---:|---:|---:|---|
| This method | `<F1>` | `<F1>` | `<F1>` | `<macro F1>` | Main canonical role-value comparison. |
| ProcNet exported | `<F1>` | `<F1>` | `n/a` | `<macro F1 or n/a>` | Use exported canonical predictions only. |
| Qwen exported | `<F1>` | `<F1>` | `<F1>` | `<macro F1>` | Include only deterministic canonical outputs. |

Comparison notes:

- This table should contain `unified-strict` scores only.
- Include an average or macro summary only when the datasets, splits, and model
  coverage justify it.
- Do not include native replay or DocFEE official-style F1 as main-table metric
  columns.

## 4. DocFEE Official-style Table

Intended use:

- DocFEE compatibility reporting.
- Side-by-side view of DocFEE official-style and canonical `unified-strict`
  scores.

Columns:

- Model
- DocFEE official-style P/R/F1
- DocFEE unified-strict P/R/F1
- Notes

Template:

| Model | DocFEE official-style P/R/F1 | DocFEE unified-strict P/R/F1 | Notes |
|---|---|---|---|
| DocFEE baseline | `<P>/<R>/<F1>` | `<P>/<R>/<F1>` | Official-style compatibility and canonical role-value scores are separate. |
| This method | `<P>/<R>/<F1>` | `<P>/<R>/<F1>` | Report both if canonical predictions are available. |
| Qwen exported | `<P>/<R>/<F1>` | `<P>/<R>/<F1>` | Include only if DocFEE predictions are in the required format. |

Comparison notes:

- The official-style column is for DocFEE compatibility.
- The `unified-strict` column is the canonical role-value metric.
- Do not use DocFEE official-style scores as ChFinAnn or DuEE-Fin metrics.

## 5. Export Drift Table

Intended use:

- Explain why ProcNet native training-report scores can differ from exported
  canonical JSONL scores.
- Separate native replay, exported fixed-slot, and canonical role-value layers.

Columns:

- Seed
- Native replay F1
- Exported legacy F1
- Unified-strict F1
- Native-to-exported gap
- Exported-to-unified gap
- Reason

Template:

| Seed | Native replay F1 | Exported legacy F1 | Unified-strict F1 | Native-to-exported gap | Exported-to-unified gap | Reason |
|---:|---:|---:|---:|---:|---:|---|
| 42 | `<F1>` | `<F1>` | `<F1>` | `<gap>` | `<gap>` | Native event-table replay and exported canonical JSONL are different input layers. |
| 43 | `<F1>` | `<F1>` | `<F1>` | `<gap>` | `<gap>` | Canonical export may change fixed-slot compatibility. |
| 44 | `<F1>` | `<F1>` | `<F1>` | `<gap>` | `<gap>` | `unified-strict` preserves canonical role-value units. |

Comparison notes:

- This table is diagnostic, not a leaderboard.
- Do not claim that a gap is an evaluator bug without source-level evidence.
- Use the reason column to document whether the gap comes from native/export
  representation, fixed-slot collapse, multi-value roles, or missing artifacts.

## 6. Multi-value Role Audit Table

Intended use:

- Justify why the paper reports `unified-strict`.
- Show where fixed-slot units and canonical role-value units differ.

Columns:

- Dataset
- Split
- Documents
- Event records
- Fixed-slot units
- Canonical role-value units
- Extra role-value units
- Multi-value role docs
- Top affected roles

Template:

| Dataset | Split | Documents | Event records | Fixed-slot units | Canonical role-value units | Extra role-value units | Multi-value role docs | Top affected roles |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ChFinAnn-Doc2EDAG | test | `<docs>` | `<records>` | `<units>` | `<units>` | `<extra>` | `<docs>` | `<role list>` |
| DuEE-Fin-dev500 | test | `<docs>` | `<records>` | `<units>` | `<units>` | `<extra>` | `<docs>` | `<role list>` |
| DocFEE-dev1000 | test | `<docs>` | `<records>` | `not_applicable` | `<units>` | `<extra>` | `<docs>` | `<role list>` |

Comparison notes:

- This is an audit table, not a model performance table.
- Fixed-slot units count at most one value per role slot.
- Canonical role-value units preserve multiple values for the same role.
- Use this table to explain why fixed-slot scores and `unified-strict` scores
  should be reported separately.
