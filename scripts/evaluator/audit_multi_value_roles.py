#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("ChFinAnn-Doc2EDAG", "DuEE-Fin-dev500", "DocFEE-dev1000")
SPLITS = ("train", "dev", "test")
SPACE_RE = re.compile(r"\s+")
OUTPUT_JSON = "multi_value_role_audit_results.json"


@dataclass
class EventRecord:
    document_id: str
    event_type: str
    arguments: dict[str, list[str]]
    record_id: str | None = None


@dataclass
class SplitAudit:
    dataset: str
    split: str
    document_count: int = 0
    event_record_count: int = 0
    canonical_raw_role_value_unit_count: int = 0
    canonical_unique_role_value_unit_count: int = 0
    fixed_slot_non_empty_unit_count: int | str = "not_applicable"
    fixed_slot_status: str = "not_applicable"
    fixed_slot_note: str = ""
    multi_value_role_occurrence_count: int = 0
    documents_with_multi_value_role: int = 0
    records_with_multi_value_role: int = 0
    multi_value_extra_units_over_fixed_slot_representation: int = 0
    top_event_type_role_pairs_by_extra_units: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "document_count": self.document_count,
            "event_record_count": self.event_record_count,
            "canonical_raw_role_value_unit_count": self.canonical_raw_role_value_unit_count,
            "canonical_unique_role_value_unit_count": self.canonical_unique_role_value_unit_count,
            "fixed_slot_non_empty_unit_count": self.fixed_slot_non_empty_unit_count,
            "fixed_slot_status": self.fixed_slot_status,
            "fixed_slot_note": self.fixed_slot_note,
            "multi_value_role_occurrence_count": self.multi_value_role_occurrence_count,
            "documents_with_multi_value_role": self.documents_with_multi_value_role,
            "records_with_multi_value_role": self.records_with_multi_value_role,
            "multi_value_extra_units_over_fixed_slot_representation": (
                self.multi_value_extra_units_over_fixed_slot_representation
            ),
            "top_event_type_role_pairs_by_extra_units": self.top_event_type_role_pairs_by_extra_units,
            "examples": self.examples,
        }


def normalize_value(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    normalized = unicodedata.normalize("NFKC", value)
    normalized = SPACE_RE.sub(" ", normalized.strip())
    return normalized or None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_schema_roles(dataset: str, dataset_dir: Path) -> tuple[dict[str, list[str]], str]:
    schema_path = dataset_dir / "schema.json"
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)

    if dataset == "ChFinAnn-Doc2EDAG":
        roles = {entry["event_type"]: list(entry.get("arguments", [])) for entry in schema}
        return roles, "ordered_schema_arguments"

    if dataset == "DuEE-Fin-dev500":
        roles = {
            entry["event_type"]: [role_entry["role"] for role_entry in entry.get("role_list", [])]
            for entry in schema
        }
        return roles, "ordered_schema_role_list"

    return {}, "not_applicable"


def load_chfinann_records(path: Path) -> tuple[int, list[EventRecord]]:
    rows = json.load(path.open(encoding="utf-8"))
    records: list[EventRecord] = []
    for doc_id, payload in rows:
        for recguid, event_type, event_dict in payload.get("recguid_eventname_eventdict_list", []):
            args: dict[str, list[str]] = {}
            for role, value in event_dict.items():
                values = value if isinstance(value, list) else [value]
                args[role] = [item for item in values if item is not None]
            records.append(EventRecord(str(doc_id), str(event_type), args, str(recguid)))
    return len(rows), records


def load_duee_records(path: Path) -> tuple[int, list[EventRecord]]:
    document_count = 0
    records: list[EventRecord] = []
    for obj in read_jsonl(path):
        document_count += 1
        document_id = str(obj.get("id") or obj.get("doc_id") or document_count)
        for record_index, event in enumerate(obj.get("event_list") or []):
            event_type = str(event.get("event_type", ""))
            args: dict[str, list[str]] = defaultdict(list)
            for argument in event.get("arguments") or []:
                role = argument.get("role")
                if role is None:
                    continue
                args[str(role)].append(argument.get("argument"))
            records.append(EventRecord(document_id, event_type, dict(args), str(record_index)))
    return document_count, records


def load_docfee_records(path: Path) -> tuple[int, list[EventRecord]]:
    document_count = 0
    records: list[EventRecord] = []
    for obj in read_jsonl(path):
        document_count += 1
        document_id = str(obj.get("doc_id") or obj.get("id") or document_count)
        for record_index, event in enumerate(obj.get("events") or []):
            event_type = str(event.get("event_type", ""))
            args: dict[str, list[str]] = {}
            for role, value in event.items():
                if role in {"event_type", "event_id", "trigger"}:
                    continue
                values = value if isinstance(value, list) else [value]
                args[str(role)] = values
            record_id = event.get("event_id")
            records.append(EventRecord(document_id, event_type, args, str(record_id or record_index)))
    return document_count, records


def load_split(dataset: str, dataset_dir: Path, split: str) -> tuple[int, list[EventRecord]]:
    json_path = dataset_dir / f"{split}.json"
    jsonl_path = dataset_dir / f"{split}.jsonl"
    if dataset == "ChFinAnn-Doc2EDAG":
        return load_chfinann_records(json_path)
    if dataset == "DuEE-Fin-dev500":
        return load_duee_records(jsonl_path)
    if dataset == "DocFEE-dev1000":
        return load_docfee_records(jsonl_path)
    raise ValueError(f"Unsupported dataset: {dataset}")


def audit_split(
    dataset: str,
    split: str,
    document_count: int,
    records: list[EventRecord],
    schema_roles: dict[str, list[str]],
    schema_status: str,
    example_limit: int,
    top_limit: int,
) -> SplitAudit:
    audit = SplitAudit(dataset=dataset, split=split)
    audit.document_count = document_count
    audit.event_record_count = len(records)

    docs_with_multi: set[str] = set()
    records_with_multi: set[tuple[str, str | None, int]] = set()
    extra_by_pair: Counter[tuple[str, str]] = Counter()

    fixed_slot_applicable = schema_status != "not_applicable"
    if fixed_slot_applicable:
        audit.fixed_slot_non_empty_unit_count = 0
        audit.fixed_slot_status = schema_status
        audit.fixed_slot_note = "ordered schema roles available; one non-empty role slot counted at most once"
    else:
        audit.fixed_slot_non_empty_unit_count = "not_applicable"
        audit.fixed_slot_status = "not_applicable"
        audit.fixed_slot_note = (
            "DocFEE schema is a JSON object properties schema, not a native ordered fixed-slot role schema"
        )

    for record_index, record in enumerate(records):
        record_has_multi = False
        schema_role_set = set(schema_roles.get(record.event_type, []))
        for role, raw_values in record.arguments.items():
            normalized_values: list[str] = []
            for raw_value in raw_values:
                normalized = normalize_value(raw_value)
                if normalized is None:
                    continue
                audit.canonical_raw_role_value_unit_count += 1
                normalized_values.append(normalized)

            unique_values = sorted(set(normalized_values))
            unique_count = len(unique_values)
            audit.canonical_unique_role_value_unit_count += unique_count

            if fixed_slot_applicable:
                if unique_count > 0 and (not schema_role_set or role in schema_role_set):
                    assert isinstance(audit.fixed_slot_non_empty_unit_count, int)
                    audit.fixed_slot_non_empty_unit_count += 1

            if unique_count > 1:
                extra_units = unique_count - 1
                audit.multi_value_role_occurrence_count += 1
                audit.multi_value_extra_units_over_fixed_slot_representation += extra_units
                docs_with_multi.add(record.document_id)
                records_with_multi.add((record.document_id, record.record_id, record_index))
                extra_by_pair[(record.event_type, role)] += extra_units
                record_has_multi = True
                if len(audit.examples) < example_limit:
                    audit.examples.append(
                        {
                            "document_id": record.document_id,
                            "record_id": record.record_id,
                            "event_type": record.event_type,
                            "role": role,
                            "values": unique_values,
                            "extra_units": extra_units,
                        }
                    )

        if record_has_multi:
            continue

    audit.documents_with_multi_value_role = len(docs_with_multi)
    audit.records_with_multi_value_role = len(records_with_multi)
    audit.top_event_type_role_pairs_by_extra_units = [
        {"event_type": event_type, "role": role, "extra_units": count}
        for (event_type, role), count in extra_by_pair.most_common(top_limit)
    ]
    return audit


def run_audit(project_root: Path, example_limit: int = 5, top_limit: int = 10) -> dict[str, Any]:
    processed_root = project_root / "data" / "processed"
    generated_at_note = "deterministic local audit; no data files modified"
    results: list[dict[str, Any]] = []

    for dataset in DATASETS:
        dataset_dir = processed_root / dataset
        schema_roles, schema_status = load_schema_roles(dataset, dataset_dir)
        for split in SPLITS:
            if not (dataset_dir / f"{split}.json").exists() and not (dataset_dir / f"{split}.jsonl").exists():
                continue
            document_count, records = load_split(dataset, dataset_dir, split)
            split_audit = audit_split(
                dataset=dataset,
                split=split,
                document_count=document_count,
                records=records,
                schema_roles=schema_roles,
                schema_status=schema_status,
                example_limit=example_limit,
                top_limit=top_limit,
            )
            results.append(split_audit.to_dict())

    return {
        "audit_name": "multi_value_role_audit",
        "normalization": ["Unicode NFKC", "strip whitespace", "collapse whitespace"],
        "generated_at_note": generated_at_note,
        "datasets": list(DATASETS),
        "splits": list(SPLITS),
        "results": results,
    }


def print_summary(report: dict[str, Any], output_path: Path | None = None) -> None:
    print("# Multi-value role audit")
    print("normalization: Unicode NFKC; strip whitespace; collapse whitespace")
    print()
    header = (
        "dataset",
        "split",
        "docs",
        "records",
        "raw_units",
        "unique_units",
        "fixed_slots",
        "multi_occ",
        "multi_docs",
        "multi_records",
        "extra_units",
    )
    print("\t".join(header))
    for row in report["results"]:
        print(
            "\t".join(
                str(row[key])
                for key in (
                    "dataset",
                    "split",
                    "document_count",
                    "event_record_count",
                    "canonical_raw_role_value_unit_count",
                    "canonical_unique_role_value_unit_count",
                    "fixed_slot_non_empty_unit_count",
                    "multi_value_role_occurrence_count",
                    "documents_with_multi_value_role",
                    "records_with_multi_value_role",
                    "multi_value_extra_units_over_fixed_slot_representation",
                )
            )
        )
    print()
    if output_path is None:
        print(f"not writing JSON; pass --write-json to update docs/evaluator/{OUTPUT_JSON}")
    else:
        print(f"wrote: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root containing data/processed")
    parser.add_argument("--example-limit", type=int, default=5, help="Examples to keep per dataset split")
    parser.add_argument("--top-limit", type=int, default=10, help="Top event_type/role pairs per dataset split")
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write docs/evaluator/multi_value_role_audit_results.json",
    )
    parser.add_argument(
        "--no-write-json",
        action="store_true",
        help="Deprecated no-op; printing without writing is now the default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    report = run_audit(project_root, example_limit=args.example_limit, top_limit=args.top_limit)
    output_path = None
    if args.write_json:
        output_dir = project_root / "docs" / "evaluator"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / OUTPUT_JSON
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
