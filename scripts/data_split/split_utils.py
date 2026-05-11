from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMMON_LIST_KEYS = ("data", "examples", "records", "documents", "items")
DOC_ID_KEYS = ("id", "doc_id", "document_id", "guid")


@dataclass(frozen=True)
class LoadedRecord:
    index: int
    obj: Any
    raw_text: str | None = None


@dataclass(frozen=True)
class DocUid:
    uid: str
    source: str


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def load_records(path: Path) -> tuple[list[LoadedRecord], str, list[str]]:
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()

    if not stripped:
        return [], "empty", [f"{path} is empty"]

    if stripped[0] in "[{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, list):
            return [LoadedRecord(i, item, None) for i, item in enumerate(obj)], "json_array", warnings
        if isinstance(obj, dict):
            for key in COMMON_LIST_KEYS:
                value = obj.get(key)
                if isinstance(value, list):
                    return (
                        [LoadedRecord(i, item, None) for i, item in enumerate(value)],
                        f"json_object_list_key:{key}",
                        warnings,
                    )
            warnings.append(f"{path} is a JSON object without a common list key; treating object as one record")
            return [LoadedRecord(0, obj, None)], "json_object_single_record", warnings

    records: list[LoadedRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for physical_line_number, line in enumerate(f, 1):
            raw_line = line.rstrip("\n")
            if raw_line.endswith("\r"):
                raw_line = raw_line[:-1]
            if not raw_line.strip():
                continue
            try:
                records.append(LoadedRecord(len(records), json.loads(raw_line), raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at physical line {physical_line_number}: {exc}") from exc
    return records, "jsonl", warnings


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_records(path: Path, records: list[LoadedRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            if record.raw_text is not None:
                f.write(record.raw_text.rstrip("\r\n"))
            else:
                f.write(json.dumps(record.obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_first_string(obj: Any, keys: tuple[str, ...]) -> tuple[str | None, str | None]:
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, (str, int)) and str(value) != "":
                return str(value), key
    if isinstance(obj, list) and obj and isinstance(obj[0], (str, int)):
        return str(obj[0]), "list[0]"
    return None, None


def document_text(obj: Any, dataset: str) -> str:
    if dataset == "ChFinAnn" and isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], dict):
        sentences = obj[1].get("sentences")
        if isinstance(sentences, list):
            return "\n".join(str(item) for item in sentences)
    if isinstance(obj, dict):
        title = obj.get("title")
        text = obj.get("text")
        content = obj.get("content")
        document = obj.get("document")
        sentences = obj.get("sentences")
        if title is not None and text is not None:
            return f"{title}\n{text}"
        if isinstance(text, str):
            return text
        if isinstance(content, str):
            return content
        if isinstance(document, str):
            return document
        if isinstance(sentences, list):
            return "\n".join(str(item) for item in sentences)
    return ""


def derive_doc_uid(obj: Any, dataset: str, index: int) -> DocUid:
    if dataset == "ChFinAnn" and isinstance(obj, list) and obj and str(obj[0]) != "":
        return DocUid(str(obj[0]), "list[0]")
    doc_id, key = find_first_string(obj, DOC_ID_KEYS)
    if doc_id is not None and key is not None:
        return DocUid(doc_id, key)
    text = document_text(obj, dataset)
    if text:
        return DocUid(f"sha256:{sha256_text(text)}", "text_hash")
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return DocUid(f"line_index_text_hash:{index}:{sha256_text(payload)}", "line_index_text_hash")


def extract_event_records(obj: Any, dataset: str) -> list[Any]:
    if dataset == "ChFinAnn" and isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], dict):
        records = obj[1].get("recguid_eventname_eventdict_list")
        return records if isinstance(records, list) else []
    if isinstance(obj, dict):
        for key in ("event_list", "events"):
            records = obj.get(key)
            if isinstance(records, list):
                return records
        records = obj.get("recguid_eventname_eventdict_list")
        if isinstance(records, list):
            return records
    return []


def event_type_from_record(event: Any, dataset: str) -> str | None:
    if dataset == "ChFinAnn" and isinstance(event, (list, tuple)) and len(event) >= 2:
        value = event[1]
        return str(value) if value not in (None, "") else None
    if isinstance(event, dict):
        value = event.get("event_type") or event.get("type") or event.get("event_name")
        return str(value) if value not in (None, "") else None
    return None


def event_dict_from_record(event: Any, dataset: str) -> dict[str, Any]:
    if dataset == "ChFinAnn" and isinstance(event, (list, tuple)) and len(event) >= 3 and isinstance(event[2], dict):
        return event[2]
    if isinstance(event, dict):
        return event
    return {}


def event_types(obj: Any, dataset: str) -> list[str]:
    types: list[str] = []
    for event in extract_event_records(obj, dataset):
        event_type = event_type_from_record(event, dataset)
        if event_type:
            types.append(event_type)
    return types


def event_signature(obj: Any, dataset: str) -> str:
    unique_types = sorted(set(event_types(obj, dataset)))
    return "|".join(unique_types) if unique_types else "NO_EVENT"


def role_names_from_event(event: Any, dataset: str) -> list[str]:
    event_dict = event_dict_from_record(event, dataset)
    roles: list[str] = []
    if dataset == "DuEE-Fin" and isinstance(event_dict.get("arguments"), list):
        for argument in event_dict["arguments"]:
            if not isinstance(argument, dict):
                continue
            role = argument.get("role")
            value = argument.get("argument")
            if role not in (None, "") and value not in (None, ""):
                roles.append(str(role))
        return roles
    if dataset == "DocFEE":
        skip = {"event_id", "event_type", "trigger", "trigger_start_index", "type", "class"}
    elif dataset == "ChFinAnn":
        skip = set()
    else:
        skip = {"event_id", "event_type", "trigger", "trigger_start_index", "type", "class", "arguments"}
    for key, value in event_dict.items():
        if key in skip:
            continue
        if value not in (None, ""):
            roles.append(str(key))
    return roles


def event_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    return "6+"


def fallback_stratum_key(obj: Any, dataset: str, global_event_counts: Counter[str]) -> str:
    types = sorted(set(event_types(obj, dataset)))
    count = len(extract_event_records(obj, dataset))
    if not types:
        primary = "NO_EVENT"
    else:
        primary = sorted(types, key=lambda item: (-global_event_counts[item], item))[0]
    return f"fallback:{event_count_bucket(count)}:{primary}"


def stratified_hash_dev_indices(
    records: list[LoadedRecord],
    dataset: str,
    dev_size: int,
    seed: int = 42,
    min_signature_stratum_size: int = 5,
) -> tuple[set[int], dict[str, Any]]:
    if dev_size < 0 or dev_size > len(records):
        raise ValueError(f"dev_size must be between 0 and {len(records)}, got {dev_size}")

    signature_counts = Counter(event_signature(record.obj, dataset) for record in records)
    global_event_counts = Counter()
    for record in records:
        global_event_counts.update(event_types(record.obj, dataset))

    strata: dict[str, list[LoadedRecord]] = defaultdict(list)
    signature_to_final_key: dict[str, str] = {}
    for record in records:
        signature = event_signature(record.obj, dataset)
        if signature_counts[signature] >= min_signature_stratum_size:
            key = signature
        else:
            key = fallback_stratum_key(record.obj, dataset, global_event_counts)
        signature_to_final_key[signature] = key
        strata[key].append(record)

    total = len(records)
    quotas: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for key, items in strata.items():
        numerator = len(items) * dev_size
        quotas[key] = numerator // total if total else 0
        remainders.append((numerator % total if total else 0, key))

    remaining = dev_size - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            remaining -= 1

    if remaining:
        for key in sorted(strata):
            if remaining <= 0:
                break
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remaining -= 1

    selected: set[int] = set()
    diagnostics: dict[str, Any] = {
        "algorithm": "deterministic_stratified_hash_largest_remainder",
        "seed": seed,
        "dev_size": dev_size,
        "total_records": total,
        "min_signature_stratum_size": min_signature_stratum_size,
        "signature_count": len(signature_counts),
        "fallback_signature_count": sum(1 for sig, count in signature_counts.items() if count < min_signature_stratum_size),
        "signature_to_final_key_for_fallbacks": {
            sig: key
            for sig, key in sorted(signature_to_final_key.items())
            if signature_counts[sig] < min_signature_stratum_size
        },
        "strata": {},
    }

    for key, items in sorted(strata.items()):
        quota = quotas[key]
        ranked = sorted(
            items,
            key=lambda record: hashlib.sha256(
                f"{seed}:{derive_doc_uid(record.obj, dataset, record.index).uid}".encode("utf-8")
            ).hexdigest(),
        )
        chosen = ranked[:quota]
        selected.update(record.index for record in chosen)
        diagnostics["strata"][key] = {"count": len(items), "selected": quota}

    if len(selected) != dev_size:
        raise RuntimeError(f"Selected {len(selected)} records, expected {dev_size}")
    return selected, diagnostics


def select_records_preserving_order(records: list[LoadedRecord], selected_indices: set[int]) -> list[LoadedRecord]:
    return [record for record in records if record.index in selected_indices]


def exclude_records(records: list[LoadedRecord], excluded_indices: set[int]) -> list[LoadedRecord]:
    return [record for record in records if record.index not in excluded_indices]


def numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
        }
    ordered = sorted(values)
    p90_index = math.ceil(0.9 * len(ordered)) - 1
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
        "median": statistics.median(values),
        "p90": ordered[max(0, min(p90_index, len(ordered) - 1))],
    }


def split_statistics(records: list[LoadedRecord], dataset: str) -> dict[str, Any]:
    event_counts: list[int] = []
    length_counts: list[int] = []
    event_distribution: Counter[str] = Counter()
    role_distribution: Counter[str] = Counter()
    signatures: Counter[str] = Counter()

    for record in records:
        events = extract_event_records(record.obj, dataset)
        event_counts.append(len(events))
        length_counts.append(len(document_text(record.obj, dataset)))
        signatures[event_signature(record.obj, dataset)] += 1
        for event in events:
            event_type = event_type_from_record(event, dataset)
            if event_type:
                event_distribution[event_type] += 1
            role_distribution.update(role_names_from_event(event, dataset))

    return {
        "document_count": len(records),
        "event_record_count": sum(event_counts),
        "event_type_distribution": dict(sorted(event_distribution.items())),
        "role_distribution": dict(sorted(role_distribution.items())),
        "per_document_event_count_distribution": {
            **numeric_summary(event_counts),
            "single_event_documents": sum(1 for count in event_counts if count == 1),
            "multi_event_documents": sum(1 for count in event_counts if count > 1),
            "zero_event_documents": sum(1 for count in event_counts if count == 0),
        },
        "document_length_statistics": numeric_summary(length_counts),
        "event_signature_distribution_summary": {
            "unique_signature_count": len(signatures),
            "top_signatures": dict(signatures.most_common(50)),
        },
    }


def uid_source_counts(records: list[LoadedRecord], dataset: str) -> dict[str, int]:
    counts = Counter(derive_doc_uid(record.obj, dataset, record.index).source for record in records)
    return dict(sorted(counts.items()))


def split_ids(records: list[LoadedRecord], dataset: str) -> list[str]:
    return [derive_doc_uid(record.obj, dataset, record.index).uid for record in records]


def _event_types_for_audit(record: LoadedRecord, dataset: str) -> list[str]:
    return sorted(set(event_types(record.obj, dataset)))


def _json_hash_audit_entry(split: str, record: LoadedRecord, dataset: str) -> dict[str, Any]:
    doc_uid = derive_doc_uid(record.obj, dataset, record.index)
    return {
        "split": split,
        "record_index": record.index,
        "doc_uid": doc_uid.uid,
        "doc_uid_source": doc_uid.source,
        "event_count": len(extract_event_records(record.obj, dataset)),
        "event_types": _event_types_for_audit(record, dataset),
        "raw_json_line_hash": sha256_text(record.raw_text) if record.raw_text is not None else None,
        "canonical_json_hash": canonical_json_hash(record.obj),
    }


def duplicate_diagnostics(splits: dict[str, list[LoadedRecord]], dataset: str) -> dict[str, Any]:
    split_hashes: dict[str, Counter[str]] = {}
    split_records_by_text_hash: dict[str, dict[str, list[LoadedRecord]]] = {}
    examples: dict[str, list[str]] = {}
    for split, records in splits.items():
        hashes = Counter(sha256_text(document_text(record.obj, dataset)) for record in records)
        split_hashes[split] = hashes
        records_by_hash: dict[str, list[LoadedRecord]] = defaultdict(list)
        for record in records:
            records_by_hash[sha256_text(document_text(record.obj, dataset))].append(record)
        split_records_by_text_hash[split] = records_by_hash
        examples[split] = [hash_value for hash_value, count in hashes.items() if count > 1][:10]

    within = {
        split: sum(count - 1 for count in hashes.values() if count > 1)
        for split, hashes in split_hashes.items()
    }
    cross: dict[str, int] = {}
    cross_examples: dict[str, list[str]] = {}
    cross_json_hash_audit: list[dict[str, Any]] = []
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(split_hashes[left]) & set(split_hashes[right])
            key = f"{left}__{right}"
            cross[key] = sum(min(split_hashes[left][hash_value], split_hashes[right][hash_value]) for hash_value in shared)
            cross_examples[key] = sorted(shared)[:10]
            for hash_value in sorted(shared):
                for left_record in split_records_by_text_hash[left][hash_value]:
                    for right_record in split_records_by_text_hash[right][hash_value]:
                        left_entry = _json_hash_audit_entry(left, left_record, dataset)
                        right_entry = _json_hash_audit_entry(right, right_record, dataset)
                        cross_json_hash_audit.append(
                            {
                                "split_pair": key,
                                "text_hash": hash_value,
                                "left": left_entry,
                                "right": right_entry,
                                "same_raw_json_line_hash": (
                                    left_entry["raw_json_line_hash"] is not None
                                    and left_entry["raw_json_line_hash"] == right_entry["raw_json_line_hash"]
                                ),
                                "same_canonical_json_hash": (
                                    left_entry["canonical_json_hash"] == right_entry["canonical_json_hash"]
                                ),
                            }
                        )

    return {
        "within_split_exact_text_duplicate_count": within,
        "within_split_duplicate_hash_examples": examples,
        "cross_split_exact_text_duplicate_count": cross,
        "cross_split_duplicate_hash_examples": cross_examples,
        "cross_split_exact_text_duplicate_json_hash_audit": cross_json_hash_audit,
        "deduplication_applied": False,
    }


def distribution_imbalance_diagnostics(
    source_records: list[LoadedRecord],
    dev_records: list[LoadedRecord],
    dataset: str,
    warning_threshold: float = 0.05,
) -> dict[str, Any]:
    source_counts = Counter()
    dev_counts = Counter()
    for record in source_records:
        source_counts.update(event_types(record.obj, dataset))
    for record in dev_records:
        dev_counts.update(event_types(record.obj, dataset))

    source_total = sum(source_counts.values())
    dev_total = sum(dev_counts.values())
    deltas: dict[str, dict[str, float | int]] = {}
    warnings: list[str] = []
    for event_type in sorted(set(source_counts) | set(dev_counts)):
        source_prop = source_counts[event_type] / source_total if source_total else 0.0
        dev_prop = dev_counts[event_type] / dev_total if dev_total else 0.0
        delta = dev_prop - source_prop
        deltas[event_type] = {
            "source_count": source_counts[event_type],
            "dev_count": dev_counts[event_type],
            "source_proportion": round(source_prop, 6),
            "dev_proportion": round(dev_prop, 6),
            "absolute_delta": round(abs(delta), 6),
        }
        if abs(delta) > warning_threshold and source_counts[event_type] >= 20:
            warnings.append(
                f"{event_type}: dev proportion differs from raw train by {abs(delta):.3f} "
                f"(threshold {warning_threshold:.3f})"
            )
    return {
        "warning_threshold": warning_threshold,
        "event_type_proportion_deltas": deltas,
        "warnings": warnings,
    }


def output_file_info(paths: dict[str, Path], project_root: Path, split_records_map: dict[str, list[LoadedRecord]]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        info[name] = {
            "path": relative_path(path, project_root),
            "sha256": sha256_file(path) if path.exists() else None,
            "count": len(split_records_map[name]) if name in split_records_map else None,
        }
    return info


def build_split_manifest(
    *,
    dataset_name: str,
    split_version: str,
    creation_script_path: Path,
    project_root: Path,
    seed: int | None,
    split_algorithm: str,
    raw_input_paths: list[Path],
    output_paths: dict[str, Path],
    split_records_map: dict[str, list[LoadedRecord]],
    notes: list[str],
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats_by_split = {split: split_statistics(records, dataset_name) for split, records in split_records_map.items()}
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "split_version": split_version,
        "creation_script_path": relative_path(creation_script_path, project_root),
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_algorithm": split_algorithm,
        "raw_input_files": [
            {"path": relative_path(path, project_root), "sha256": sha256_file(path)}
            for path in raw_input_paths
        ],
        "output_files": output_file_info(output_paths, project_root, split_records_map),
        "document_counts_per_split": {
            split: values["document_count"] for split, values in stats_by_split.items()
        },
        "event_record_counts_per_split": {
            split: values["event_record_count"] for split, values in stats_by_split.items()
        },
        "event_type_distribution_per_split": {
            split: values["event_type_distribution"] for split, values in stats_by_split.items()
        },
        "role_distribution_per_split": {
            split: values["role_distribution"] for split, values in stats_by_split.items()
        },
        "per_document_event_count_distribution": {
            split: values["per_document_event_count_distribution"] for split, values in stats_by_split.items()
        },
        "document_length_statistics": {
            split: values["document_length_statistics"] for split, values in stats_by_split.items()
        },
        "event_signature_distribution_summary": {
            split: values["event_signature_distribution_summary"] for split, values in stats_by_split.items()
        },
        "duplicate_diagnostics": duplicate_diagnostics(split_records_map, dataset_name),
        "uid_source_counts": {
            split: uid_source_counts(records, dataset_name) for split, records in split_records_map.items()
        },
        "split_ids": {
            split: split_ids(records, dataset_name) for split, records in split_records_map.items()
        },
        "notes": notes,
        "warnings": warnings or [],
    }
    if extra:
        manifest.update(extra)
    return manifest


def print_dataset_summary(dataset: str, manifest: dict[str, Any]) -> None:
    print(f"\n== {dataset} ==")
    print("documents:")
    for split, count in manifest["document_counts_per_split"].items():
        events = manifest["event_record_counts_per_split"].get(split, 0)
        event_dist = manifest["per_document_event_count_distribution"][split]
        print(
            f"  {split}: docs={count}, events={events}, "
            f"single={event_dist['single_event_documents']}, multi={event_dist['multi_event_documents']}"
        )
    print("top event types:")
    for split, dist in manifest["event_type_distribution_per_split"].items():
        top = sorted(dist.items(), key=lambda item: (-item[1], item[0]))[:5]
        print(f"  {split}: {top}")
    if manifest.get("warnings"):
        print("warnings:")
        for warning in manifest["warnings"][:10]:
            print(f"  - {warning}")
