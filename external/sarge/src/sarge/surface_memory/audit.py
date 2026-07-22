from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from sarge.surface_memory.types import SurfaceMemory
from sarge.surface_memory.weak_alignment import WeakAlignmentRecord
from sarge.data.jsonl import write_jsonl


def compute_audit_summary(
    memories: Iterable[SurfaceMemory],
    alignments: Iterable[WeakAlignmentRecord],
    *,
    recall_ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, object]:
    memory_list = list(memories)
    alignment_list = list(alignments)
    candidate_count_per_doc = {memory.doc_id: len(memory.candidates) for memory in memory_list}
    total = len(alignment_list)
    located = [record for record in alignment_list if record.status == "located"]
    unlocated = [record for record in alignment_list if record.status == "unlocated"]
    ambiguous = [record for record in alignment_list if record.ambiguous]

    return {
        "document_count": len(memory_list),
        "candidate_count_per_doc": candidate_count_per_doc,
        "candidate_count_total": sum(candidate_count_per_doc.values()),
        "gold_argument_count": total,
        "gold_argument_located_count": len(located),
        "gold_argument_unlocated_count": len(unlocated),
        "ambiguous_match_count": len(ambiguous),
        "gold_argument_located_rate": _rate(len(located), total),
        "unlocated_argument_rate": _rate(len(unlocated), total),
        "ambiguous_match_rate": _rate(len(ambiguous), total),
        "candidate_recall_at_k": _candidate_recall_at_k(memory_list, alignment_list, recall_ks),
        "per_role_located_rate": _grouped_located_rate(alignment_list, group_name="role"),
        "per_event_type_located_rate": _grouped_located_rate(alignment_list, group_name="event_type"),
    }


def write_audit_outputs(
    out_dir: str | Path,
    memories: Iterable[SurfaceMemory],
    alignments: Iterable[WeakAlignmentRecord],
    *,
    dataset: str,
    split: str,
    mode: str,
    gold_visible: bool,
    allow_gold_audit: bool,
    warning: str | None = None,
) -> dict[str, object]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_list = list(memories)
    alignment_list = list(alignments)
    summary = compute_audit_summary(memory_list, alignment_list)
    summary.update(
        {
            "dataset": dataset,
            "split": split,
            "mode": mode,
            "gold_visible": gold_visible,
            "allow_gold_audit": allow_gold_audit,
            "warning": warning,
        }
    )
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if gold_visible:
        write_jsonl(output_dir / "weak_alignment.jsonl", [record.to_dict() for record in alignment_list])
        write_jsonl(
            output_dir / "examples_unlocated.jsonl",
            [record.to_dict() for record in alignment_list if record.status == "unlocated"],
        )
        write_jsonl(
            output_dir / "examples_ambiguous.jsonl",
            [record.to_dict() for record in alignment_list if record.ambiguous],
        )
    return summary


def _candidate_recall_at_k(
    memories: list[SurfaceMemory],
    alignments: list[WeakAlignmentRecord],
    recall_ks: tuple[int, ...],
) -> dict[str, float | None]:
    if not alignments:
        return {str(k): None for k in recall_ks}
    ids_by_doc = {memory.doc_id: [candidate.candidate_id for candidate in memory.candidates] for memory in memories}
    recall: dict[str, float | None] = {}
    for k in recall_ks:
        hit_count = 0
        for record in alignments:
            top_k = set(ids_by_doc.get(record.doc_id, [])[:k])
            if top_k.intersection(record.candidate_ids):
                hit_count += 1
        recall[str(k)] = _rate(hit_count, len(alignments))
    return recall


def _grouped_located_rate(records: list[WeakAlignmentRecord], *, group_name: str) -> dict[str, float | None]:
    totals: defaultdict[str, int] = defaultdict(int)
    located: defaultdict[str, int] = defaultdict(int)
    for record in records:
        key = getattr(record, group_name)
        totals[key] += 1
        if record.status == "located":
            located[key] += 1
    return {key: _rate(located[key], total) for key, total in sorted(totals.items())}


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
