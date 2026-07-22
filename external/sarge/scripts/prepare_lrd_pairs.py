"""Generate pairwise training labels for LRD from SARGE candidate output.

Reads the intermediate parsed-candidates jsonl produced by the GETM stage
and stages the raw dataset gold via ``stage_dataset`` to obtain canonical
``{doc_id, content, events}`` rows.  Runs greedy record-to-record matching
between predicted candidates and gold records to label pairs: pairs whose
two predictions point at the same gold record are positives, all others
negatives.

Output: a jsonl file where each row is a single document with fields
``doc_id``, ``text`` (doc body, needed by preencode_lrd), ``records``
(predicted event dicts), and ``pairs``.

Example:
    python scripts/prepare_lrd_pairs.py \\
        --candidates runs/.../intermediate/getm/parsed_candidates.train.jsonl \\
        --dataset DuEE-Fin-dev500 \\
        --split train \\
        --out runs/lrd/dueefin_train_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sarge.data.schema import load_schema  # noqa: E402
from sarge.data.staging import stage_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="parsed GETM candidates jsonl")
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. DuEE-Fin-dev500")
    parser.add_argument("--split", default="train", help="dataset split to load gold from")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"),
                        help="processed dataset root containing <dataset>/<split>.{jsonl|json}")
    parser.add_argument("--out", required=True, help="output jsonl for LRD training pairs")
    parser.add_argument("--max-docs", type=int, default=None, help="cap on documents")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    load_schema(args.dataset, data_root=data_root)  # sanity-check schema

    staging = Path(tempfile.mkdtemp(prefix="sarge_lrd_pairs_"))
    try:
        stage_dataset(
            dataset=args.dataset,
            processed_root=data_root,
            output_root=staging,
            splits=(args.split,),
            limit=args.max_docs,
        )
        gold_by_doc = _load_canonical_gold(
            staging / args.dataset / f"{args.split}.jsonl", args.max_docs
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    candidates_by_doc = _load_candidates(args.candidates, args.max_docs)

    total_pairs = 0
    total_pos = 0
    written = 0
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for doc_id, info in gold_by_doc.items():
            gold_records = info["records"]
            doc_text = info["text"]
            candidate_rows = candidates_by_doc.get(doc_id)
            if not candidate_rows:
                continue
            pred_records = _flatten_candidates(candidate_rows)
            if len(pred_records) < 2:
                continue

            pairs, pos_count = _build_pairs(pred_records, gold_records)
            if len(pairs) < 1:
                continue

            row = {
                "doc_id": doc_id,
                "text": doc_text,
                "records": pred_records,
                "pairs": pairs,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            total_pairs += len(pairs)
            total_pos += pos_count

    print(f"wrote {written} docs, {total_pairs} pairs (pos={total_pos}, neg={total_pairs - total_pos})")
    return 0


def _load_canonical_gold(path: Path, limit: int | None) -> dict[str, dict]:
    """Read staged canonical jsonl ({doc_id, content, events}) and return
    ``{doc_id: {"records": [...], "text": "..."}}`` in the shape expected by
    ``_build_pairs``."""
    result: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id") or ""
            if not doc_id:
                continue
            text = str(row.get("content") or "")
            records: list[dict] = []
            for event in row.get("events") or []:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("event_type") or "")
                if not event_type:
                    continue
                normalized: dict[str, list[str]] = {}
                for role, values in (event.get("arguments") or {}).items():
                    for value in (values or []):
                        text_v = value.get("text") if isinstance(value, dict) else str(value)
                        if text_v:
                            normalized.setdefault(str(role), []).append(str(text_v))
                records.append({"event_type": event_type, "arguments": normalized})
            if records:
                result[doc_id] = {"records": records, "text": text}
    return result


def _flatten_candidates(rows: list[dict]) -> list[dict]:
    """Each row may contain multiple candidate events (k variants)."""
    records: list[dict] = []
    for row in rows:
        events = row.get("events") or row.get("predictions") or []
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    records.append(event)
    return records


def _build_pairs(
    pred_records: list[dict],
    gold_records: list[dict],
) -> tuple[list[dict], int]:
    """Hungarian-align predictions to gold, then label pairwise matches."""
    n = len(pred_records)
    # Simple greedy matching (unified_strict style: event_type match,
    # role-value overlap score).
    assignments = _match_predictions(pred_records, gold_records)

    pairs: list[dict] = []
    pos_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            same = assignments.get(i) is not None and assignments.get(j) is not None and assignments[i] == assignments[j]
            label = 1 if same else 0
            pairs.append({
                "i": i, "j": j,
                "label": label,
                "event_type_i": pred_records[i].get("event_type", ""),
                "event_type_j": pred_records[j].get("event_type", ""),
            })
            if label == 1:
                pos_count += 1
    return pairs, pos_count


def _match_predictions(pred_records: list[dict], gold_records: list[dict]) -> dict[int, int]:
    """Soft record-to-record matching — each pred independently maps to its
    best gold (event_type constrained).  Multiple predictions may share the
    same gold, which is the signal that they are duplicates."""
    assignments: dict[int, int] = {}

    def _score(pred: dict, gold: dict) -> int:
        if pred.get("event_type") != gold.get("event_type"):
            return 0
        pred_args = _flat_args(pred)
        gold_args = _flat_args(gold)
        return len(pred_args & gold_args)

    for pi, pred in enumerate(pred_records):
        best_g = -1
        best_s = 0
        for gi, gold in enumerate(gold_records):
            s = _score(pred, gold)
            if s > best_s:
                best_s = s
                best_g = gi
        if best_g >= 0 and best_s > 0:
            assignments[pi] = best_g
    return assignments


def _flat_args(record: dict) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    args = record.get("arguments") or {}
    if isinstance(args, list):
        # Gold format: [{"role": ..., "argument": ...}]
        for arg in args:
            if isinstance(arg, dict):
                role = str(arg.get("role", "")).strip()
                val = str(arg.get("argument") or arg.get("text") or "").strip()
                if role and val:
                    result.add((role, val))
    elif isinstance(args, dict):
        # Candidate/pred format: {role: [{"text": ...}]}
        for role, values in args.items():
            for value in (values or []):
                text = str(value if isinstance(value, str) else value.get("text", "")).strip()
                if text:
                    result.add((role, text))
    return result


def _load_candidates(path: str, limit: int | None) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id") or row.get("document_id") or ""
            if not doc_id:
                continue
            result.setdefault(doc_id, []).append(row)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
