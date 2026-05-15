#!/usr/bin/env python
"""P5b CARVE allocation failure diagnostic.

Decomposes carve-route FPs at the record level to identify the dominant failure
mode of the Sinkhorn + share gate allocation mechanism.

Categories:
  1. Noise records       — pred records that match no gold record (0 role-value overlap)
  2. Misallocation FPs   — pred record matches a gold record on some roles, but has wrong values
  3. Share-gate excess   — same value appears in >1 sibling pred records beyond what gold supports
  4. Candidate miss      — gold values present in doc text but absent from all pred records
  5. Pathological gold   — gold values not present in doc text (annotation noise)

Usage:
    python scripts/carve/p5b_carve_alloc_diagnostic.py \\
        --run-dir runs/carve/p5b_duee_fin_dev500_seed42_r3typegate \\
        --dev-jsonl data/processed/DuEE-Fin-dev500/dev.jsonl \\
        --dataset DuEE-Fin-dev500 \\
        --out docs/diagnostics/p5b_carve_alloc_breakdown.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carve.datasets import load_duee_documents


def _load_canonical_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _doc_full_text(document) -> str:
    parts = []
    if getattr(document, "title", None):
        parts.append(document.title)
    if getattr(document, "text", None):
        parts.append(document.text)
    return "".join(parts)


def _record_args_flat(record: dict) -> set[tuple[str, str]]:
    """Flatten record.arguments into (role, value) pairs."""
    pairs = set()
    for role, values in record.get("arguments", {}).items():
        for v in (values if isinstance(values, list) else [values]):
            pairs.add((role, str(v)))
    return pairs


def _count_record_args(record: dict) -> int:
    return sum(len(v) if isinstance(v, list) else 1 for v in record.get("arguments", {}).values())


def _greedy_match_records(
    pred_recs: list[dict],
    gold_recs: list[dict],
) -> tuple[dict[int, int], dict[int, int]]:
    """Greedy max-overlap matching of pred records to gold records.

    Returns:
        pred_to_gold: pred_index -> gold_index (only for matched pairs)
        gold_to_pred: gold_index -> pred_index
    """
    pred_to_gold: dict[int, int] = {}
    gold_to_pred: dict[int, int] = {}
    used_pred: set[int] = set()
    used_gold: set[int] = set()

    candidates: list[tuple[int, int, int]] = []  # (overlap, pred_idx, gold_idx)
    for pi, p in enumerate(pred_recs):
        p_pairs = _record_args_flat(p)
        if not p_pairs:
            continue
        for gi, g in enumerate(gold_recs):
            g_pairs = _record_args_flat(g)
            overlap = len(p_pairs & g_pairs)
            if overlap > 0:
                candidates.append((overlap, pi, gi))

    candidates.sort(key=lambda x: -x[0])
    for overlap, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        pred_to_gold[pi] = gi
        gold_to_pred[gi] = pi

    return pred_to_gold, gold_to_pred


def _analyze_run(
    gold_rows: list[dict],
    pred_rows: list[dict],
    doc_texts: dict[str, str],
) -> dict:
    gold_by_doc: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pred_by_doc: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in gold_rows:
        doc_id = row["document_id"]
        for ev in row.get("events", []):
            gold_by_doc[doc_id][ev["event_type"]].append(ev)
    for row in pred_rows:
        doc_id = row["document_id"]
        for ev in row.get("events", []):
            pred_by_doc[doc_id][ev["event_type"]].append(ev)

    # FP attribution counters
    noise_record_count = 0
    noise_fp_args = 0
    misallocation_fp_args = 0  # matched record, wrong (role, value)
    tp_args = 0
    share_gate_total_dup_args = 0  # all duplicated values (value appears >1 in sibling preds)
    share_gate_excess_dup_args = 0  # duplications beyond what gold supports

    # FN attribution
    candidate_miss_or_record_drop = 0  # gold value present in doc text, missing from pred
    pathological_gold = 0  # gold value not in doc text

    # Per-event-type breakdown
    by_event_type: dict[str, dict] = defaultdict(lambda: {
        "noise_records": 0, "noise_fp_args": 0, "misallocation_fp_args": 0,
        "tp_args": 0, "share_gate_excess": 0, "candidate_miss": 0,
        "pred_records": 0, "gold_records": 0, "matched_records": 0,
    })

    all_doc_ids = set(gold_by_doc) | set(pred_by_doc)
    for doc_id in all_doc_ids:
        text = doc_texts.get(doc_id, "")
        gold_types = gold_by_doc.get(doc_id, {})
        pred_types = pred_by_doc.get(doc_id, {})
        all_types = set(gold_types) | set(pred_types)

        for et in all_types:
            gold_recs = gold_types.get(et, [])
            pred_recs = pred_types.get(et, [])
            by_event_type[et]["pred_records"] += len(pred_recs)
            by_event_type[et]["gold_records"] += len(gold_recs)

            # Greedy match
            pred_to_gold, gold_to_pred = _greedy_match_records(pred_recs, gold_recs)
            by_event_type[et]["matched_records"] += len(pred_to_gold)

            # Per-pred-record attribution
            for pi, p in enumerate(pred_recs):
                n_args = _count_record_args(p)
                if pi not in pred_to_gold:
                    # Unmatched: pure noise record
                    noise_record_count += 1
                    noise_fp_args += n_args
                    by_event_type[et]["noise_records"] += 1
                    by_event_type[et]["noise_fp_args"] += n_args
                else:
                    gi = pred_to_gold[pi]
                    g = gold_recs[gi]
                    g_pairs = _record_args_flat(g)
                    p_pairs = _record_args_flat(p)
                    tp = len(p_pairs & g_pairs)
                    fp = len(p_pairs - g_pairs)
                    tp_args += tp
                    misallocation_fp_args += fp
                    by_event_type[et]["tp_args"] += tp
                    by_event_type[et]["misallocation_fp_args"] += fp

            # Share-gate duplication detection (per role across all sibling pred records)
            for role in {role for r in pred_recs for role in r.get("arguments", {})}:
                pred_values_for_role = []
                for r in pred_recs:
                    pred_values_for_role.extend(r.get("arguments", {}).get(role, []))
                gold_values_for_role = []
                for r in gold_recs:
                    gold_values_for_role.extend(r.get("arguments", {}).get(role, []))
                pred_counter = Counter(pred_values_for_role)
                gold_counter = Counter(gold_values_for_role)
                for value, pcount in pred_counter.items():
                    if pcount > 1:
                        share_gate_total_dup_args += pcount
                        gcount = gold_counter.get(value, 0)
                        excess = max(pcount - max(gcount, 1), 0)
                        share_gate_excess_dup_args += excess
                        by_event_type[et]["share_gate_excess"] += excess

            # Candidate miss: gold values not in any pred record
            for gi, g in enumerate(gold_recs):
                if gi in gold_to_pred:
                    matched_pred = pred_recs[gold_to_pred[gi]]
                    matched_p_pairs = _record_args_flat(matched_pred)
                else:
                    matched_p_pairs = set()
                g_pairs = _record_args_flat(g)
                # Aggregate all pred values across event_type (for "covered anywhere" check)
                all_pred_pairs_this_et = set()
                for r in pred_recs:
                    all_pred_pairs_this_et |= _record_args_flat(r)
                for role, value in g_pairs:
                    if value in str(text):
                        if (role, value) not in all_pred_pairs_this_et:
                            candidate_miss_or_record_drop += 1
                            by_event_type[et]["candidate_miss"] += 1
                    else:
                        pathological_gold += 1

    total_fp_args = noise_fp_args + misallocation_fp_args
    total_pred_args = total_fp_args + tp_args

    # Estimate "share gate excess" share of misallocation+noise FPs
    # (note: this isn't a strict partition — share excess can overlap with noise/misallocation,
    # because a record that's entirely duplicated values is both noise and share-gate)

    return {
        "summary": {
            "tp_args": tp_args,
            "total_fp_args": total_fp_args,
            "noise_record_count": noise_record_count,
            "noise_fp_args": noise_fp_args,
            "misallocation_fp_args": misallocation_fp_args,
            "share_gate_total_dup_args": share_gate_total_dup_args,
            "share_gate_excess_dup_args": share_gate_excess_dup_args,
            "candidate_miss_or_record_drop": candidate_miss_or_record_drop,
            "pathological_gold": pathological_gold,
            "total_pred_args": total_pred_args,
        },
        "ratios": {
            "noise_fp_share": round(noise_fp_args / max(total_fp_args, 1), 4),
            "misallocation_fp_share": round(misallocation_fp_args / max(total_fp_args, 1), 4),
            "share_gate_excess_share_of_fp": round(share_gate_excess_dup_args / max(total_fp_args, 1), 4),
            "precision_args": round(tp_args / max(total_pred_args, 1), 4),
        },
        "by_event_type": {
            et: dict(m) for et, m in sorted(
                by_event_type.items(),
                key=lambda x: -(x[1]["noise_fp_args"] + x[1]["misallocation_fp_args"]),
            )
        },
    }


def _write_markdown(path: Path, result: dict, run_dir: str) -> None:
    s = result["summary"]
    r = result["ratios"]

    # Diagnose dominant failure mode
    if r["noise_fp_share"] > 0.5:
        verdict = (
            "**Dominant failure: noise records (record-count over-prediction).** "
            "Sinkhorn is allocating to N record columns where most should be empty. "
            "The lexical record_count fallback overestimates."
        )
    elif r["share_gate_excess_share_of_fp"] > 0.3:
        verdict = (
            "**Dominant failure: share gate over-firing.** "
            "Values are being duplicated across sibling records beyond what gold supports. "
            "The share_threshold=0.50 default is too permissive; try 0.70+."
        )
    elif r["misallocation_fp_share"] > 0.5:
        verdict = (
            "**Dominant failure: within-record misallocation.** "
            "Records that match gold partially also include wrong role-value pairs. "
            "Indicates Sinkhorn permutation noise or candidate generation precision issue."
        )
    else:
        verdict = (
            "**Mixed failure mode.** No single category dominates. "
            "Allocation mechanism likely needs structural changes, not single-knob tuning."
        )

    lines = [
        "# P5b CARVE Allocation Failure Diagnostic",
        "",
        f"Run: `{run_dir}`",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Summary",
        "",
        f"- TP arguments: **{s['tp_args']:,}**  ",
        f"- Total FP arguments: **{s['total_fp_args']:,}**  ",
        f"- Argument-level precision: **{r['precision_args']:.4f}**",
        "",
        "### FP breakdown",
        "",
        "| Category | Args | Share of FP |",
        "|---|---:|---:|",
        f"| Noise records (unmatched pred records, all args FP) | {s['noise_fp_args']:,} | {r['noise_fp_share']:.1%} |",
        f"| Misallocation (matched record, wrong (role,value)) | {s['misallocation_fp_args']:,} | {r['misallocation_fp_share']:.1%} |",
        f"| Share-gate excess (overlapping subset of above) | {s['share_gate_excess_dup_args']:,} | {r['share_gate_excess_share_of_fp']:.1%} |",
        "",
        f"Noise record count: **{s['noise_record_count']:,}**",
        "",
        "Note: noise records and misallocation FPs partition the total FP space. ",
        "Share-gate excess is an *additional* attribution that may overlap with either category — ",
        "any duplicated value in an unmatched record is counted in both noise_fp and share_gate_excess.",
        "",
        "### FN breakdown",
        "",
        "| Category | Count |",
        "|---|---:|",
        f"| Gold value in doc text, missing from pred | {s['candidate_miss_or_record_drop']:,} |",
        f"| Gold value not in doc text (pathological) | {s['pathological_gold']:,} |",
        "",
        "## Per-event-type breakdown (sorted by total FP args)",
        "",
        "| Event Type | Pred recs | Gold recs | Matched | TP args | Noise FP | Misalloc FP | Share excess | Cand. miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for et, m in result["by_event_type"].items():
        lines.append(
            f"| {et} | {m['pred_records']} | {m['gold_records']} | {m['matched_records']} | "
            f"{m['tp_args']} | {m['noise_fp_args']} | {m['misallocation_fp_args']} | "
            f"{m['share_gate_excess']} | {m['candidate_miss']} |"
        )

    lines += [
        "",
        "## Recommended Next Step",
        "",
    ]
    if r["noise_fp_share"] > 0.5:
        lines.append(
            "- Improve the record-count estimator before re-running P5b. The lexical fallback "
            "(`_estimate_record_count`) is currently the bottleneck. Options:"
        )
        lines.append("  - Use R3 v5's coref count (requires inference-time mention extraction)")
        lines.append("  - Cap record_count at min(lexical_estimate, k_clip=16) — likely already done")
        lines.append("  - Run a lightweight regression on document features to predict record_count")
    elif r["share_gate_excess_share_of_fp"] > 0.3:
        lines.append(
            "- Sweep `--share-threshold` ∈ {0.60, 0.70, 0.80, 0.90} on the existing P5b+PlannerGate "
            "checkpoint. Inference-only re-run, ~5 min × 4 settings."
        )
    elif r["misallocation_fp_share"] > 0.5:
        lines.append(
            "- The Sinkhorn allocation is producing wrong assignments. Investigate:"
        )
        lines.append("  - Allocation training data quality (per-record gold supervision)")
        lines.append("  - Sinkhorn iteration count / temperature")
        lines.append("  - Candidate feature representation")
    else:
        lines.append(
            "- No single fix is indicated. The CARVE allocation mechanism likely needs structural "
            "redesign rather than parameter tuning. Consider venue framing pivot to R3 v5 standalone."
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P5b carve allocation failure diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--out", default="docs/diagnostics/p5b_carve_alloc_breakdown.md")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    gold_path = run_dir / "canonical" / "dev.gold.jsonl"
    pred_path = run_dir / "canonical" / "dev.carve.pred.jsonl"
    for p in (gold_path, pred_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    print("Loading canonical JSONL files...", flush=True)
    gold_rows = _load_canonical_jsonl(gold_path)
    pred_rows = _load_canonical_jsonl(pred_path)

    print("Loading dev documents for text lookup...", flush=True)
    dev_docs = load_duee_documents(Path(args.dev_jsonl), dataset=args.dataset)
    doc_texts = {doc.document_id: _doc_full_text(doc) for doc in dev_docs}

    print("Analyzing carve allocation failure modes...", flush=True)
    result = _analyze_run(gold_rows, pred_rows, doc_texts)

    out_path = Path(args.out)
    _write_markdown(out_path, result, args.run_dir)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written JSON: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
