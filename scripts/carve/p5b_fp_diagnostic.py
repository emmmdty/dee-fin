#!/usr/bin/env python
"""P5b false-positive decomposition diagnostic.

Reads a completed P5b run directory and decomposes carve-route FPs by:
  - event_type (which event types contribute most FPs)
  - record count delta (n_pred vs n_gold per doc/type) — tests H2
  - hallucination rate (predicted value absent from document text) — tests H3
  - role distribution of FP arguments

Usage:
    python scripts/carve/p5b_fp_diagnostic.py \\
        --run-dir runs/carve/p5b_duee_fin_dev500_seed42 \\
        --dev-jsonl data/processed/DuEE-Fin-dev500/dev.jsonl \\
        --dataset DuEE-Fin-dev500 \\
        --out docs/diagnostics/p5b_fp_breakdown.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carve.datasets import load_duee_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _value_in_text(value: str, text: str) -> bool:
    return value in text


# ---------------------------------------------------------------------------
# Per-event TP/FP/FN from existing eval JSON (fast path)
# ---------------------------------------------------------------------------

def _load_per_event_eval(eval_path: Path) -> dict[str, dict]:
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    per_event = report.get("per_event", {})
    result: dict[str, dict] = {}
    for event_name, metrics in per_event.items():
        result[event_name] = {
            "tp": int(metrics.get("tp", 0)),
            "fp": int(metrics.get("fp", 0)),
            "fn": int(metrics.get("fn", 0)),
            "precision": round(float(metrics.get("precision", 0.0)), 4),
            "recall": round(float(metrics.get("recall", 0.0)), 4),
            "f1": round(float(metrics.get("f1", 0.0)), 4),
        }
    return result


# ---------------------------------------------------------------------------
# Record count delta analysis (H2)
# ---------------------------------------------------------------------------

def _count_delta_analysis(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> dict:
    gold_by_doc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pred_by_doc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in gold_rows:
        doc_id = row["document_id"]
        for ev in row.get("events", []):
            gold_by_doc[doc_id][ev["event_type"]] += 1
    for row in pred_rows:
        doc_id = row["document_id"]
        for ev in row.get("events", []):
            pred_by_doc[doc_id][ev["event_type"]] += 1

    overpredict_docs = 0
    exact_docs = 0
    underpredict_docs = 0
    total_excess_records = 0
    by_event_type: dict[str, dict] = defaultdict(lambda: {"overpredict": 0, "exact": 0, "underpredict": 0, "excess": 0})

    all_doc_ids = set(gold_by_doc) | set(pred_by_doc)
    for doc_id in all_doc_ids:
        gold_et = gold_by_doc[doc_id]
        pred_et = pred_by_doc[doc_id]
        all_types = set(gold_et) | set(pred_et)
        doc_has_over = False
        for et in all_types:
            g = gold_et.get(et, 0)
            p = pred_et.get(et, 0)
            delta = p - g
            if delta > 0:
                by_event_type[et]["overpredict"] += 1
                by_event_type[et]["excess"] += delta
                total_excess_records += delta
                doc_has_over = True
            elif delta == 0:
                by_event_type[et]["exact"] += 1
            else:
                by_event_type[et]["underpredict"] += 1
        if doc_has_over:
            overpredict_docs += 1

    return {
        "total_excess_records": total_excess_records,
        "overpredict_doc_count": overpredict_docs,
        "by_event_type": {k: dict(v) for k, v in
                          sorted(by_event_type.items(), key=lambda x: -x[1]["excess"])},
    }


# ---------------------------------------------------------------------------
# Hallucination analysis (H3 partial)
# ---------------------------------------------------------------------------

def _hallucination_analysis(
    pred_rows: list[dict],
    doc_texts: dict[str, str],
) -> dict:
    total_args = 0
    hallucinated = 0
    by_role: dict[str, dict] = defaultdict(lambda: {"total": 0, "hallucinated": 0})

    for row in pred_rows:
        doc_id = row["document_id"]
        text = doc_texts.get(doc_id, "")
        for ev in row.get("events", []):
            for role, values in ev.get("arguments", {}).items():
                for value in (values if isinstance(values, list) else [values]):
                    total_args += 1
                    by_role[role]["total"] += 1
                    if not _value_in_text(str(value), text):
                        hallucinated += 1
                        by_role[role]["hallucinated"] += 1

    by_role_sorted = {}
    for role, counts in sorted(by_role.items(), key=lambda x: -x[1]["hallucinated"]):
        total_r = counts["total"]
        hall_r = counts["hallucinated"]
        by_role_sorted[role] = {
            "total": total_r,
            "hallucinated": hall_r,
            "hallucination_rate": round(hall_r / max(total_r, 1), 4),
        }

    return {
        "total_args": total_args,
        "hallucinated": hallucinated,
        "hallucination_rate": round(hallucinated / max(total_args, 1), 4),
        "by_role": by_role_sorted,
    }


# ---------------------------------------------------------------------------
# Role FP analysis: pred args not matched to any gold record
# Uses canonical matching: for each doc, match pred records to gold records by
# event_type, find unmatched pred records, count their role arguments as FP.
# ---------------------------------------------------------------------------

def _role_fp_analysis(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> dict:
    gold_by_doc: dict[str, list[dict]] = defaultdict(list)
    pred_by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in gold_rows:
        gold_by_doc[row["document_id"]].extend(row.get("events", []))
    for row in pred_rows:
        pred_by_doc[row["document_id"]].extend(row.get("events", []))

    fp_by_role: dict[str, int] = defaultdict(int)
    fp_by_event_type: dict[str, int] = defaultdict(int)
    total_fp_args = 0

    for doc_id in pred_by_doc:
        gold_evs = gold_by_doc.get(doc_id, [])
        pred_evs = pred_by_doc[doc_id]

        gold_by_et: dict[str, list] = defaultdict(list)
        pred_by_et: dict[str, list] = defaultdict(list)
        for ev in gold_evs:
            gold_by_et[ev["event_type"]].append(ev)
        for ev in pred_evs:
            pred_by_et[ev["event_type"]].append(ev)

        for et, pred_group in pred_by_et.items():
            gold_group = gold_by_et.get(et, [])
            # Excess pred records (beyond gold count) are pure FP records
            n_excess = max(len(pred_group) - len(gold_group), 0)
            # Sort pred records by argument density (most args first) as proxy
            # for which records are "excess" — use the last n_excess as excess
            sorted_pred = sorted(pred_group, key=lambda r: -sum(len(v) for v in r.get("arguments", {}).values()))
            for pred_rec in sorted_pred[-n_excess:] if n_excess else []:
                for role, values in pred_rec.get("arguments", {}).items():
                    count = len(values) if isinstance(values, list) else 1
                    fp_by_role[role] += count
                    fp_by_event_type[et] += count
                    total_fp_args += count

    return {
        "total_fp_args_from_excess_records": total_fp_args,
        "by_role": dict(sorted(fp_by_role.items(), key=lambda x: -x[1])),
        "by_event_type": dict(sorted(fp_by_event_type.items(), key=lambda x: -x[1])),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# P5b FP Decomposition Diagnostic",
        "",
        f"Run: `{result['run_dir']}`  ",
        f"Route: carve  ",
        f"Total CARVE FP: **{result['total_fp']:,}** vs baseline FP: **{result['baseline_fp']:,}**",
        "",
        "---",
        "",
        "## 1. Per-Event TP/FP/FN (from unified_strict eval)",
        "",
        "| Event Type | TP | FP | FN | Prec | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for et, m in sorted(result["per_event_eval"].items(), key=lambda x: -x[1]["fp"]):
        lines.append(f"| {et} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")

    lines += [
        "",
        "## 2. Record Count Delta (H2: over-prediction of record count)",
        "",
        f"Total excess records (n_pred - n_gold, summed over doc/type pairs where n_pred > n_gold): "
        f"**{result['count_delta']['total_excess_records']}**  ",
        f"Documents with at least one over-predicted event type: "
        f"**{result['count_delta']['overpredict_doc_count']}**",
        "",
        "| Event Type | Over-pred cases | Exact cases | Under-pred cases | Excess records |",
        "|---|---:|---:|---:|---:|",
    ]
    for et, m in result["count_delta"]["by_event_type"].items():
        lines.append(f"| {et} | {m['overpredict']} | {m['exact']} | {m['underpredict']} | {m['excess']} |")

    lines += [
        "",
        "## 3. Hallucination Rate (H3 partial: value absent from document text)",
        "",
        f"Total predicted arguments: **{result['hallucination']['total_args']:,}**  ",
        f"Hallucinated (value not in doc text): **{result['hallucination']['hallucinated']:,}** "
        f"({result['hallucination']['hallucination_rate']:.1%})",
        "",
        "| Role | Total | Hallucinated | Rate |",
        "|---|---:|---:|---:|",
    ]
    for role, m in result["hallucination"]["by_role"].items():
        lines.append(f"| {role} | {m['total']} | {m['hallucinated']} | {m['hallucination_rate']:.1%} |")

    lines += [
        "",
        "## 4. Role Distribution of FP Arguments (from excess records)",
        "",
        f"Total FP arguments from excess (over-predicted) records: "
        f"**{result['role_fp']['total_fp_args_from_excess_records']:,}**",
        "",
        "### By role",
        "",
        "| Role | FP args |",
        "|---|---:|",
    ]
    for role, count in result["role_fp"]["by_role"].items():
        lines.append(f"| {role} | {count} |")

    lines += [
        "",
        "### By event type",
        "",
        "| Event Type | FP args from excess records |",
        "|---|---:|",
    ]
    for et, count in result["role_fp"]["by_event_type"].items():
        lines.append(f"| {et} | {count} |")

    lines += [
        "",
        "## 5. Hypothesis Assessment",
        "",
        "| Hypothesis | Indicator | Verdict |",
        "|---|---|---|",
    ]
    excess = result["count_delta"]["total_excess_records"]
    hall_rate = result["hallucination"]["hallucination_rate"]
    total_pred = result["total_pred_records"]
    excess_frac = excess / max(total_pred, 1)

    h2_verdict = "**Confirmed**" if excess_frac > 0.2 else ("Partial" if excess_frac > 0.05 else "Not primary")
    h3_verdict = "**Confirmed**" if hall_rate > 0.1 else ("Partial" if hall_rate > 0.03 else "Not primary")
    lines.append(f"| H2: record count over-prediction | {excess} excess records ({excess_frac:.1%} of pred records) | {h2_verdict} |")
    lines.append(f"| H3: candidate over-generation | hallucination rate {hall_rate:.1%} | {h3_verdict} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P5b FP decomposition diagnostic")
    parser.add_argument("--run-dir", required=True, help="P5b run directory")
    parser.add_argument("--dev-jsonl", required=True, help="Dev documents JSONL path")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--out", default="docs/diagnostics/p5b_fp_breakdown.md")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)

    gold_path = run_dir / "canonical" / "dev.gold.jsonl"
    carve_pred_path = run_dir / "canonical" / "dev.carve.pred.jsonl"
    baseline_pred_path = run_dir / "canonical" / "dev.baseline.pred.jsonl"
    carve_eval_path = run_dir / "eval" / "dev.carve.unified_strict.json"
    baseline_eval_path = run_dir / "eval" / "dev.baseline.unified_strict.json"

    for p in (gold_path, carve_pred_path, carve_eval_path):
        if not p.exists():
            print(f"ERROR: required file missing: {p}", file=sys.stderr)
            return 1

    print("Loading canonical JSONL files...", flush=True)
    gold_rows = _load_canonical_jsonl(gold_path)
    pred_rows = _load_canonical_jsonl(carve_pred_path)

    baseline_fp = 0
    if baseline_eval_path.exists():
        b_report = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
        baseline_fp = int(b_report.get("overall", {}).get("fp", 0))

    carve_report = json.loads(carve_eval_path.read_text(encoding="utf-8"))
    total_fp = int(carve_report.get("overall", {}).get("fp", 0))
    total_pred_records = sum(len(row.get("events", [])) for row in pred_rows)

    print("Loading dev documents for text lookup...", flush=True)
    dev_docs = load_duee_documents(Path(args.dev_jsonl), dataset=args.dataset)
    doc_texts = {doc.document_id: _doc_full_text(doc) for doc in dev_docs}

    print("Computing per-event eval breakdown...", flush=True)
    per_event_eval = _load_per_event_eval(carve_eval_path)

    print("Computing record count delta (H2)...", flush=True)
    count_delta = _count_delta_analysis(gold_rows, pred_rows)

    print("Computing hallucination rates (H3)...", flush=True)
    hallucination = _hallucination_analysis(pred_rows, doc_texts)

    print("Computing role FP distribution...", flush=True)
    role_fp = _role_fp_analysis(gold_rows, pred_rows)

    result = {
        "run_dir": args.run_dir,
        "total_fp": total_fp,
        "baseline_fp": baseline_fp,
        "total_pred_records": total_pred_records,
        "per_event_eval": per_event_eval,
        "count_delta": count_delta,
        "hallucination": hallucination,
        "role_fp": role_fp,
    }

    out_path = Path(args.out)
    _write_markdown(out_path, result)

    json_out = out_path.with_suffix(".json")
    json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written JSON: {json_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
