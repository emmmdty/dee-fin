from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_split import split_utils as u


DATASET_NAME = "DuEE-Fin"
OUTPUT_NAME = "DuEE-Fin-dev500"
SPLIT_VERSION = "public_raw_dev_as_test_train_dev500_seed42_v1"
SEED = 42
DEV_SIZE = 500


def _write_readme(path: Path) -> None:
    content = """# DuEE-Fin-dev500

## Source Files

- `data/raw/DuEE-Fin/duee_fin_train.json`
- `data/raw/DuEE-Fin/duee_fin_dev.json`
- `data/raw/DuEE-Fin/duee_fin_test.json`
- `data/raw/DuEE-Fin/duee_fin_event_schema.json`

## Split Rule

- Official raw dev is used as the labeled test split.
- Raw train is split deterministically into `dev.jsonl` with 500 documents and
  `train.jsonl` with the remaining documents.
- Official raw test has no public gold and is preserved as
  `blind_test_unlabeled.jsonl`.

The dev split uses a seed-42 deterministic stratified hash split over
document-level event signatures. This is a reproducible public dev500 split, not
a claim to recover hidden prior-paper dev IDs.

## Deduplication

No deduplication was applied. Duplicate diagnostics in `split_manifest.json` are
audit-only.

## Offset Policy

Offsets are not required as canonical gold for role-value extraction.

## Schema Policy

The raw JSONL event schema is normalized into a UTF-8 JSON array at
`schema.json`.

## Reproduction Command

```bash
python scripts/data_split/prepare_all_splits.py --project-root .
```

## Output Files

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `blind_test_unlabeled.jsonl`
- `schema.json`
- `split_manifest.json`
- `README.md`

## Cautions

The public raw test split is unlabeled. Use `test.jsonl` for labeled local
evaluation because it comes from the official raw dev file.
"""
    path.write_text(content, encoding="utf-8")


def prepare(project_root: Path) -> dict[str, Any]:
    raw_dir = project_root / "data/raw/DuEE-Fin"
    output_dir = project_root / "data/processed" / OUTPUT_NAME
    u.ensure_dir(output_dir)

    raw_train_path = raw_dir / "duee_fin_train.json"
    raw_dev_path = raw_dir / "duee_fin_dev.json"
    raw_test_path = raw_dir / "duee_fin_test.json"
    raw_schema_path = raw_dir / "duee_fin_event_schema.json"

    raw_train, train_format, train_warnings = u.load_records(raw_train_path)
    raw_dev, dev_format, dev_warnings = u.load_records(raw_dev_path)
    raw_test, test_format, test_warnings = u.load_records(raw_test_path)
    schema_records, schema_format, schema_warnings = u.load_records(raw_schema_path)

    warnings: list[str] = []
    for name, fmt in {
        "duee_fin_train.json": train_format,
        "duee_fin_dev.json": dev_format,
        "duee_fin_test.json": test_format,
        "duee_fin_event_schema.json": schema_format,
    }.items():
        if fmt != "jsonl":
            warnings.append(f"{name} loaded as {fmt}; expected JSONL based on local raw inspection")
    warnings.extend(train_warnings + dev_warnings + test_warnings + schema_warnings)

    dev_indices, split_diagnostics = u.stratified_hash_dev_indices(raw_train, DATASET_NAME, DEV_SIZE, seed=SEED)
    processed_dev = u.select_records_preserving_order(raw_train, dev_indices)
    processed_train = u.exclude_records(raw_train, dev_indices)
    processed_test = raw_dev
    blind_test = raw_test

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"
    blind_path = output_dir / "blind_test_unlabeled.jsonl"
    schema_path = output_dir / "schema.json"
    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "split_manifest.json"

    u.write_jsonl_records(train_path, processed_train)
    u.write_jsonl_records(dev_path, processed_dev)
    u.write_jsonl_records(test_path, processed_test)
    u.write_jsonl_records(blind_path, blind_test)
    u.write_json(schema_path, [record.obj for record in schema_records])
    _write_readme(readme_path)

    imbalance = u.distribution_imbalance_diagnostics(raw_train, processed_dev, DATASET_NAME)
    warnings.extend(imbalance["warnings"])

    split_records_map = {
        "train": processed_train,
        "dev": processed_dev,
        "test": processed_test,
        "blind_test_unlabeled": blind_test,
    }
    notes = [
        "no deduplication was applied",
        "duplicate diagnostics are audit-only",
        "offsets are not required for canonical role-value extraction",
        "ChFinAnn uses the official Doc2EDAG split unchanged",
        "DuEE-Fin uses raw dev as test and deterministic train-dev500 split",
        "DocFEE uses official test and deterministic train-dev1000 split",
        "this split does not claim to reproduce hidden prior-paper dev ids",
    ]
    manifest = u.build_split_manifest(
        dataset_name=DATASET_NAME,
        split_version=SPLIT_VERSION,
        creation_script_path=Path(__file__),
        project_root=project_root,
        seed=SEED,
        split_algorithm="deterministic_stratified_hash_by_event_signature_largest_remainder",
        raw_input_paths=[raw_train_path, raw_dev_path, raw_test_path, raw_schema_path],
        output_paths={
            "train": train_path,
            "dev": dev_path,
            "test": test_path,
            "blind_test_unlabeled": blind_path,
            "schema": schema_path,
            "README": readme_path,
        },
        split_records_map=split_records_map,
        notes=notes,
        warnings=warnings,
        extra={
            "split_selection_diagnostics": split_diagnostics,
            "split_quality_diagnostics": imbalance,
        },
    )
    u.write_json(manifest_path, manifest)
    u.print_dataset_summary(OUTPUT_NAME, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DuEE-Fin deterministic dev500 split.")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    prepare(u.resolve_project_root(args.project_root))


if __name__ == "__main__":
    main()
