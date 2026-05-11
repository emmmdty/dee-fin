from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_split import split_utils as u


DATASET_NAME = "DocFEE"
OUTPUT_NAME = "DocFEE-dev1000"
SPLIT_VERSION = "official_test_train_dev1000_seed42_v1"
SEED = 42
DEV_SIZE = 1000


def _duplicate_audit_note(duplicate_diagnostics: dict[str, Any]) -> str:
    cross_counts = duplicate_diagnostics["cross_split_exact_text_duplicate_count"]
    audit_entries = duplicate_diagnostics["cross_split_exact_text_duplicate_json_hash_audit"]
    if not any(cross_counts.values()):
        return "No cross-split exact text duplicates were found.\n"

    same_raw_count = sum(1 for entry in audit_entries if entry["same_raw_json_line_hash"])
    same_canonical_count = sum(1 for entry in audit_entries if entry["same_canonical_json_hash"])
    count_text = ", ".join(f"`{pair}`={count}" for pair, count in sorted(cross_counts.items()))
    return (
        f"Cross-split exact text duplicate counts: {count_text}. Whole-record JSON hash audit: "
        f"`same_raw_json_line_hash`={same_raw_count}, `same_canonical_json_hash`={same_canonical_count}. "
        "For the current split, the duplicate texts are same-content/different-record cases, not copied identical JSON rows.\n"
    )


def _write_readme(path: Path, duplicate_diagnostics: dict[str, Any]) -> None:
    duplicate_note = _duplicate_audit_note(duplicate_diagnostics)
    content = f"""# DocFEE-dev1000

## Source Files

- `data/raw/DocFEE/train.jsonl`
- `data/raw/DocFEE/test.jsonl`
- `data/raw/DocFEE/schema.json`

## Split Rule

- Official raw test remains the labeled test split.
- Raw train is split deterministically into `dev.jsonl` with 1000 documents and
  `train.jsonl` with the remaining documents.

The dev split uses a seed-42 deterministic stratified hash split over
document-level event signatures.

## Deduplication

No deduplication was applied. Duplicate diagnostics in `split_manifest.json` are
audit-only.

## Duplicate Audit Note

{duplicate_note}

## Offset Policy

Offsets are not required as canonical gold for role-value extraction.

## Schema Policy

The raw `schema.json` file is copied into the processed dataset.

## Reproduction Command

```bash
python scripts/data_split/prepare_all_splits.py --project-root .
```

## Output Files

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `schema.json`
- `split_manifest.json`
- `README.md`

## Cautions

The split is deterministic and documented, but no deduplication or filtering is
performed. Duplicate diagnostics are for audit only.
"""
    path.write_text(content, encoding="utf-8")


def prepare(project_root: Path) -> dict[str, Any]:
    raw_dir = project_root / "data/raw/DocFEE"
    output_dir = project_root / "data/processed" / OUTPUT_NAME
    u.ensure_dir(output_dir)

    raw_train_path = raw_dir / "train.jsonl"
    raw_test_path = raw_dir / "test.jsonl"
    raw_schema_path = raw_dir / "schema.json"

    raw_train, train_format, train_warnings = u.load_records(raw_train_path)
    raw_test, test_format, test_warnings = u.load_records(raw_test_path)

    warnings: list[str] = []
    if train_format != "jsonl":
        warnings.append(f"train.jsonl loaded as {train_format}; expected JSONL")
    if test_format != "jsonl":
        warnings.append(f"test.jsonl loaded as {test_format}; expected JSONL")
    warnings.extend(train_warnings + test_warnings)

    dev_indices, split_diagnostics = u.stratified_hash_dev_indices(raw_train, DATASET_NAME, DEV_SIZE, seed=SEED)
    processed_dev = u.select_records_preserving_order(raw_train, dev_indices)
    processed_train = u.exclude_records(raw_train, dev_indices)
    processed_test = raw_test

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"
    schema_path = output_dir / "schema.json"
    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "split_manifest.json"
    split_records_map = {
        "train": processed_train,
        "dev": processed_dev,
        "test": processed_test,
    }
    duplicate_diagnostics = u.duplicate_diagnostics(split_records_map, DATASET_NAME)

    u.write_jsonl_records(train_path, processed_train)
    u.write_jsonl_records(dev_path, processed_dev)
    u.write_jsonl_records(test_path, processed_test)
    shutil.copyfile(raw_schema_path, schema_path)
    _write_readme(readme_path, duplicate_diagnostics)

    imbalance = u.distribution_imbalance_diagnostics(raw_train, processed_dev, DATASET_NAME)
    warnings.extend(imbalance["warnings"])

    notes = [
        "no deduplication was applied",
        "duplicate diagnostics are audit-only",
        "offsets are not required for canonical role-value extraction",
        "ChFinAnn uses the official Doc2EDAG split unchanged",
        "DuEE-Fin uses raw dev as test and deterministic train-dev500 split",
        "DocFEE uses official test and deterministic train-dev1000 split",
    ]
    manifest = u.build_split_manifest(
        dataset_name=DATASET_NAME,
        split_version=SPLIT_VERSION,
        creation_script_path=Path(__file__),
        project_root=project_root,
        seed=SEED,
        split_algorithm="deterministic_stratified_hash_by_event_signature_largest_remainder",
        raw_input_paths=[raw_train_path, raw_test_path, raw_schema_path],
        output_paths={
            "train": train_path,
            "dev": dev_path,
            "test": test_path,
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
    parser = argparse.ArgumentParser(description="Prepare DocFEE deterministic dev1000 split.")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    prepare(u.resolve_project_root(args.project_root))


if __name__ == "__main__":
    main()
