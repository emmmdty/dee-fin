# Evaluator Package

This package provides reproducible evaluators for Chinese financial document-level event extraction. It is intentionally limited to strict surface-form scoring. Offset F1, semantic similarity, embedding matching, and LLM judging are not part of the main metrics.

## Tracks

### `legacy-doc2edag`

Use this track for historical compatibility with native ProcNet / Doc2EDAG-style fixed-slot results.

It supports two explicit input modes:

- `--input-format canonical-jsonl` reads exported canonical prediction and gold files, converts records to fixed slots, and evaluates them with the native fixed-slot scorer. This is the backward-compatible default. Because canonical export may already have changed native event-table structure, this mode is not guaranteed bit-exact with ProcNet internal training metrics.
- `--input-format native-event-table` reads saved Doc2EDAG/ProcNet-style event-table matrices directly. Use this for native replay when event-table artifacts were saved before canonical export.

The implementation is a native fixed-slot reimplementation of the local Doc2EDAG/ProcNet metric logic:

- records are grouped by document and event type;
- each event type uses a fixed ordered role list from the schema;
- each role slot contributes at most one argument;
- missing or empty role slots are represented as `None`;
- repeated or multi-value roles are collapsed to one fixed slot with last normalized non-empty value winning;
- predicted records are sorted by non-empty fixed-slot count;
- each prediction greedily matches the same-event gold record with maximum slot equality, including `None == None`;
- no gold record can be matched more than once;
- matched records contribute TP/FP/FN by fixed role slot;
- unmatched predictions contribute FP fixed slots;
- unmatched gold records contribute FN fixed slots.

This track does not use canonical multi-value role-value unit counting. Pass `--schema` for historical comparison so the role order matches the dataset's fixed event table.

Native event-table JSON uses one value per fixed role slot:

```json
{
  "format": "procnet_native_event_table_v1",
  "dataset": "toy",
  "seed": 44,
  "split": "test",
  "event_types": ["质押", "股份回购"],
  "event_type_fields": {
    "质押": ["质押方", "质押物", "质押数量", "质权方"],
    "股份回购": ["回购方", "回购股份数量", "每股交易价格", "交易金额"]
  },
  "documents": [
    {
      "document_id": "doc1",
      "gold": [
        [["华夏控股", "股份", "1000万股", null]],
        [["某上市公司", "500万股", "6.12元", "3060万元"]]
      ],
      "pred": [
        [["华夏控股", "股份", "900万股", null]],
        [["某上市公司", "500万股", "6.12元", null]]
      ]
    }
  ]
}
```

The indexing semantics are:

```text
documents[i].gold[event_type_index][record_index][role_index]
documents[i].pred[event_type_index][record_index][role_index]
```

`event_types[event_type_index]` selects the event type, and `event_type_fields[event_type]` defines role-slot order. Native event-table mode bypasses canonical role-value loaders and does not count multiple role values because each matrix slot has at most one value.

### `docfee-official`

Use this track for DocFEE official-style comparability. It mirrors the metric logic in `baseline/DocFEE/baseline/utils.py` without using unsafe `eval()`.

The implementation converts each record to an `event_type -> list[role:value]` representation, then greedily removes the most similar predicted record for each gold record. It reports overall and per-event P/R/F1 plus basic schema and format diagnostics.

DocFEE processed records in this project use Chinese event and role names. The copied `data/processed/DocFEE-dev1000/schema.json` is the raw English schema, so this track defaults to the Chinese schema embedded from the local DocFEE baseline unless `--schema` is explicitly provided.

### `unified-strict`

Use this track for cross-dataset scientific comparison across ChFinAnn, DuEE-Fin, and DocFEE. It converts supported dataset formats into canonical records:

```json
{
  "document_id": "doc-id",
  "events": [
    {
      "event_type": "EventType",
      "record_id": "optional",
      "arguments": {
        "RoleName": ["value"]
      }
    }
  ]
}
```

Matching is constrained by document id and event type. Within each group, the evaluator maximizes strict canonical role-value true positives with deterministic exact DP for small groups, optional SciPy linear assignment for larger groups when SciPy is already installed, and a documented deterministic greedy fallback otherwise.

`unified-strict` remains the canonical role-value metric. It is not a native Doc2EDAG/ProcNet event-table replay metric.

## Normalization

All tracks use strict surface normalization only:

- Unicode NFKC;
- strip leading/trailing whitespace;
- collapse repeated whitespace;
- ignore empty values;
- `legacy-doc2edag` collapses multi-value roles to one fixed slot;
- `unified-strict` treats multi-value roles as unordered role-value sets.

The evaluator does not perform alias expansion, semantic matching, embedding similarity, LLM judging, company-code mapping, date reasoning, amount conversion, event-type guessing, role guessing, or gold-based repair.

## Metrics

All tracks report raw TP/FP/FN with safe P/R/F1:

```text
precision = TP / (TP + FP), else 0
recall    = TP / (TP + FN), else 0
f1        = 2PR / (P + R), else 0
```

Reports include:

- `dataset`;
- `metric_family`;
- `overall`;
- `per_event`;
- `subset_metrics` for single-event and multi-event gold documents;
- `diagnostics`;
- `input_paths`;
- `schema_path`;
- `normalization_policy`;
- `matching_policy`.

`legacy-doc2edag` reports fixed-slot unit diagnostics and its slot collapse policy. `unified-strict` also reports record exact-match diagnostics as auxiliary information.

## CLI Examples

```bash
python -m evaluator --help
python -m evaluator inspect-gold --dataset DuEE-Fin-dev500 --gold data/processed/DuEE-Fin-dev500/dev.jsonl --schema data/processed/DuEE-Fin-dev500/schema.json
python -m evaluator legacy-doc2edag --input-format canonical-jsonl --dataset ChFinAnn-Doc2EDAG --gold data/processed/ChFinAnn-Doc2EDAG/test.json --pred predictions.jsonl --schema data/processed/ChFinAnn-Doc2EDAG/schema.json --out report.json
python -m evaluator legacy-doc2edag --input-format native-event-table --native-table native_event_tables.json --out report.json
python -m evaluator docfee-official --dataset DocFEE-dev1000 --gold data/processed/DocFEE-dev1000/test.jsonl --pred predictions.jsonl --out report.json
python -m evaluator unified-strict --dataset DuEE-Fin-dev500 --gold data/processed/DuEE-Fin-dev500/test.jsonl --pred predictions.jsonl --schema data/processed/DuEE-Fin-dev500/schema.json --out report.json
```

If `--out` is omitted, the JSON report is printed to stdout.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/evaluator
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/evaluator
```

If your environment uses a specific Python interpreter, replace `python` with that interpreter.

## Limitations

- The Doc2EDAG track is a native fixed-slot metric compatible with ProcNet / Doc2EDAG-style event-table scoring, but it is not a byte-for-byte runner for upstream pickle/model internals.
- The DocFEE track is official-style and does not run model-output retry logic or content-substring checks from HAC annotation scripts.
- Invalid prediction schema items are diagnosed but not repaired.
- Offset-based span metrics are intentionally excluded from main scoring.
- Large unified-strict record groups may use deterministic greedy fallback if SciPy is unavailable and the group exceeds the exact DP limit.
