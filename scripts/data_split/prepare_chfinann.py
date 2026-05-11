from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_split import split_utils as u


DATASET_NAME = "ChFinAnn"
OUTPUT_NAME = "ChFinAnn-Doc2EDAG"
SPLIT_VERSION = "doc2edag_official_split_schema_validated_v1"


def _parse_candidate_schema(candidate: Any) -> tuple[dict[str, list[str]], list[str]]:
    warnings: list[str] = []
    schema: dict[str, list[str]] = {}
    if not isinstance(candidate, list):
        return schema, ["candidate schema is not a JSON list"]
    for index, item in enumerate(candidate):
        if not isinstance(item, dict):
            warnings.append(f"candidate schema item {index} is not an object")
            continue
        event_type = item.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            warnings.append(f"candidate schema item {index} has missing/non-string event_type")
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, list):
            warnings.append(f"candidate schema item {index} ({event_type}) has non-list arguments")
            arguments = []
        roles: list[str] = []
        for role_index, role_item in enumerate(arguments):
            if isinstance(role_item, str) and role_item:
                roles.append(role_item)
            elif isinstance(role_item, dict) and isinstance(role_item.get("role"), str):
                roles.append(role_item["role"])
            else:
                warnings.append(
                    f"candidate schema item {index} ({event_type}) has malformed role at arguments[{role_index}]"
                )
        schema[event_type] = roles
    return schema, warnings


def _extract_empirical_schema(records_by_split: dict[str, list[u.LoadedRecord]]) -> dict[str, Any]:
    event_type_counts = Counter()
    role_counts = Counter()
    role_counts_by_event_type: dict[str, Counter[str]] = defaultdict(Counter)
    malformed: list[str] = []

    for split, records in records_by_split.items():
        for record in records:
            if not (isinstance(record.obj, list) and len(record.obj) >= 2 and isinstance(record.obj[1], dict)):
                malformed.append(f"{split}[{record.index}] is not Doc2EDAG [doc_id, payload]")
                continue
            for event_index, event in enumerate(u.extract_event_records(record.obj, DATASET_NAME)):
                event_type = u.event_type_from_record(event, DATASET_NAME)
                if not event_type:
                    malformed.append(f"{split}[{record.index}].event[{event_index}] has no event type")
                    continue
                event_type_counts[event_type] += 1
                roles = u.role_names_from_event(event, DATASET_NAME)
                if not roles:
                    malformed.append(f"{split}[{record.index}].event[{event_index}] ({event_type}) has no roles")
                for role in roles:
                    role_counts[role] += 1
                    role_counts_by_event_type[event_type][role] += 1

    return {
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "global_role_counts": dict(sorted(role_counts.items())),
        "role_counts_by_event_type": {
            event_type: dict(sorted(counter.items()))
            for event_type, counter in sorted(role_counts_by_event_type.items())
        },
        "malformed_or_suspicious_fields": malformed,
    }


def _build_processed_schema(candidate_schema: dict[str, list[str]], empirical: dict[str, Any]) -> list[dict[str, Any]]:
    role_counts_by_type: dict[str, dict[str, int]] = empirical["role_counts_by_event_type"]
    event_types = [event_type for event_type in candidate_schema if event_type in role_counts_by_type]
    event_types.extend(sorted(event_type for event_type in role_counts_by_type if event_type not in candidate_schema))

    processed_schema: list[dict[str, Any]] = []
    for event_type in event_types:
        empirical_roles = set(role_counts_by_type.get(event_type, {}))
        candidate_roles = [role for role in candidate_schema.get(event_type, []) if role in empirical_roles]
        extra_roles = sorted(empirical_roles - set(candidate_roles))
        processed_schema.append({"event_type": event_type, "arguments": candidate_roles + extra_roles})
    return processed_schema


def _schema_validation_report(
    candidate_schema: dict[str, list[str]],
    candidate_warnings: list[str],
    empirical: dict[str, Any],
) -> dict[str, Any]:
    empirical_event_types = set(empirical["event_type_counts"])
    candidate_event_types = set(candidate_schema)
    candidate_roles_global = set().union(*(set(roles) for roles in candidate_schema.values())) if candidate_schema else set()
    empirical_roles_global = set(empirical["global_role_counts"])

    role_comparison: dict[str, Any] = {}
    for event_type in sorted(candidate_event_types | empirical_event_types):
        candidate_roles = set(candidate_schema.get(event_type, []))
        empirical_roles = set(empirical["role_counts_by_event_type"].get(event_type, {}))
        role_comparison[event_type] = {
            "roles_in_candidate_but_not_data": sorted(candidate_roles - empirical_roles),
            "roles_in_data_but_not_candidate": sorted(empirical_roles - candidate_roles),
        }

    differences = {
        "event_types_in_candidate_but_not_data": sorted(candidate_event_types - empirical_event_types),
        "event_types_in_data_but_not_candidate": sorted(empirical_event_types - candidate_event_types),
        "roles_in_candidate_but_not_data": sorted(candidate_roles_global - empirical_roles_global),
        "roles_in_data_but_not_candidate": sorted(empirical_roles_global - candidate_roles_global),
        "roles_by_event_type": role_comparison,
    }

    status = "pass"
    if (
        candidate_warnings
        or empirical["malformed_or_suspicious_fields"]
        or differences["event_types_in_candidate_but_not_data"]
        or differences["event_types_in_data_but_not_candidate"]
        or differences["roles_in_candidate_but_not_data"]
        or differences["roles_in_data_but_not_candidate"]
        or any(
            value["roles_in_candidate_but_not_data"] or value["roles_in_data_but_not_candidate"]
            for value in role_comparison.values()
        )
    ):
        status = "warning"

    return {
        "status": status,
        "candidate_schema_is_treated_as_official": False,
        "candidate_schema_warnings": candidate_warnings,
        "empirical_schema": empirical,
        "comparison": differences,
    }


def _write_readme(path: Path) -> None:
    content = """# ChFinAnn-Doc2EDAG

## Source Files

- `data/raw/ChFinAnn/train.json`
- `data/raw/ChFinAnn/dev.json`
- `data/raw/ChFinAnn/test.json`
- `data/raw/ChFinAnn/schema.json` (candidate schema only)

## Split Rule

This processed dataset preserves the existing Doc2EDAG-style official split. The
raw `train.json`, `dev.json`, and `test.json` files are copied unchanged.

## Deduplication

No deduplication was applied. Duplicate diagnostics in `split_manifest.json` are
audit-only.

## Offset Policy

Existing Doc2EDAG offset/drange fields are preserved exactly. They are not
recomputed and are not required as canonical role-value gold.

## Schema Policy

The raw schema is loaded only as a candidate schema. The processed
`schema.json` is generated from empirical train/dev/test event records, and
`schema_validation_report.json` compares the empirical schema with the candidate.

## Reproduction Command

```bash
python scripts/data_split/prepare_all_splits.py --project-root .
```

## Output Files

- `train.json`
- `dev.json`
- `test.json`
- `schema.json`
- `schema_validation_report.json`
- `split_manifest.json`
- `README.md`

## Cautions

Do not treat the raw candidate schema as independently verified official
documentation. Use the processed schema and validation report for reproducible
local experiments.
"""
    path.write_text(content, encoding="utf-8")


def prepare(project_root: Path) -> dict[str, Any]:
    raw_dir = project_root / "data/raw/ChFinAnn"
    output_dir = project_root / "data/processed" / OUTPUT_NAME
    u.ensure_dir(output_dir)

    split_paths = {
        "train": raw_dir / "train.json",
        "dev": raw_dir / "dev.json",
        "test": raw_dir / "test.json",
    }
    output_paths = {split: output_dir / f"{split}.json" for split in split_paths}
    for split, source_path in split_paths.items():
        shutil.copyfile(source_path, output_paths[split])

    records_by_split: dict[str, list[u.LoadedRecord]] = {}
    load_warnings: list[str] = []
    for split, source_path in split_paths.items():
        records, fmt, warnings = u.load_records(source_path)
        if fmt != "json_array":
            load_warnings.append(f"{source_path} loaded as {fmt}, expected json_array")
        load_warnings.extend(warnings)
        records_by_split[split] = records

    candidate_schema_path = raw_dir / "schema.json"
    candidate_schema, candidate_warnings = _parse_candidate_schema(u.read_json_file(candidate_schema_path))
    empirical = _extract_empirical_schema(records_by_split)
    processed_schema = _build_processed_schema(candidate_schema, empirical)
    schema_report = _schema_validation_report(candidate_schema, candidate_warnings, empirical)

    schema_path = output_dir / "schema.json"
    report_path = output_dir / "schema_validation_report.json"
    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "split_manifest.json"
    u.write_json(schema_path, processed_schema)
    u.write_json(report_path, schema_report)
    _write_readme(readme_path)

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
        seed=None,
        split_algorithm="copy_official_doc2edag_split_no_dedup_schema_empirical_validation",
        raw_input_paths=[*split_paths.values(), candidate_schema_path],
        output_paths={
            **output_paths,
            "schema": schema_path,
            "schema_validation_report": report_path,
            "README": readme_path,
        },
        split_records_map=records_by_split,
        notes=notes,
        warnings=load_warnings + candidate_warnings + empirical["malformed_or_suspicious_fields"],
        extra={"schema_validation": schema_report},
    )
    u.write_json(manifest_path, manifest)
    u.print_dataset_summary(OUTPUT_NAME, manifest)
    print(f"ChFinAnn schema validation status: {schema_report['status']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ChFinAnn Doc2EDAG split and schema audit.")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    prepare(u.resolve_project_root(args.project_root))


if __name__ == "__main__":
    main()
