#!/usr/bin/env python
"""Offline Phase-B analysis: repair + risk-controlled admission over an edge dump.

Consumes a raw predicted-edge dump (produced on GPU by `evaluate_relations.py
--dump-predictions` with `consistency: identity` and no admission) plus the gold
valid split, and reports — on a single held-out document set, apples-to-apples —
the raw -> repaired -> repaired+admitted trajectory:

- **consistency** before/after: `consistency_report` cycle/closure diagnostics;
- **admission**: stratified FNR (marginal / per-family / doc-macro) + admitted
  set size, under a CRC bound calibrated on a held-out split;
- **reconstructability**: R1 query-edge reachability (the CS-CRP bridge) and R2
  query-edge fidelity, plus the per-query `reachable` flags Phase E consumes.

The checkpoint never comes local: the GPU produces the dump, this runs on CPU.

    uv run python scripts/consistency_repair_report.py \\
        --dump runs/relations/supervised_dump.jsonl \\
        --gold data/maven_ere/valid.jsonl --alpha 0.2 --cal-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finekg.core.eval.consistency import consistency_report
from finekg.core.eval.relation import PRF
from finekg.core.io import read_jsonl
from finekg.core.schema import EventGraph, RelationEdge, RelationType
from finekg.relations.admission import (
    edge_admission,
    gold_edge_scores,
    stratified_admission_report,
)
from finekg.relations.consistency import RepairTrace, consistency_solvers
from finekg.relations.data import load_maven_ere
from finekg.relations.data.maven_ere import RelationDocument
from finekg.succession.reconstruction import ecg_reachable_flags, reconstruction_report

STAGES = ("raw", "repaired", "repaired_admitted")


def _edges_from_dump(record: dict) -> list[RelationEdge]:
    """Rebuild the predicted edges of one document from its dump record.

    Field-by-field so a malformed dump fails fast rather than silently dropping
    edges; the dump writer always emits all six keys.
    """
    return [
        RelationEdge(
            head_id=e["head_id"],
            tail_id=e["tail_id"],
            relation_type=RelationType(e["relation_type"]),
            subtype=e["subtype"],
            directed=e["directed"],
            confidence=e["confidence"],
        )
        for e in record["edges"]
    ]


def _graph(doc: RelationDocument, edges: list[RelationEdge]) -> EventGraph:
    return EventGraph(nodes={n.event_id: n for n in doc.nodes}, edges=list(edges))


def _sum_consistency(graphs: list[EventGraph]) -> dict[str, float]:
    """Corpus-summed `consistency_report` (cycle counts add; matches the online
    evaluator's `consistency_sum`)."""
    totals: dict[str, float] = {}
    for graph in graphs:
        for key, value in consistency_report(graph).items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


def _aggregate_reconstruction(pairs: list[tuple[RelationDocument, EventGraph]]) -> dict:
    """Corpus R1 reachability rate + micro-aggregated R2 query-edge PRF."""
    reachable = queries = ecgs = 0
    tp = n_pred = n_gold = 0
    for doc, graph in pairs:
        report = reconstruction_report(doc, graph)
        reachable += report["r1_reachable"]
        queries += report["n_gold_queries"]
        ecgs += report["n_gold_ecgs"]
        prf = report["r2_query_prf"]
        tp, n_pred, n_gold = tp + prf["tp"], n_pred + prf["n_pred"], n_gold + prf["n_gold"]
    agg = PRF.from_counts(tp, n_pred, n_gold)
    return {
        "n_gold_ecgs": ecgs,
        "n_gold_queries": queries,
        "r1_reachable": reachable,
        "r1_reachability_rate": reachable / queries if queries else 0.0,
        "r2_query_prf": {
            "precision": float(agg["precision"]),
            "recall": float(agg["recall"]),
            "f1": float(agg["f1"]),
            "tp": int(agg["tp"]),
            "n_pred": int(agg["n_pred"]),
            "n_gold": int(agg["n_gold"]),
        },
    }


def _trace_sample(trace: RepairTrace | None) -> dict | None:
    """A compact, JSON-safe snapshot of one non-trivial RepairTrace (evidence)."""
    if trace is None:
        return None
    return {
        "counts": trace.counts,
        "before": trace.before,
        "after": trace.after,
        "edits": [
            {
                "action": edit.action,
                "violation": edit.violation,
                "edge_key": list(edit.edge_key),
                "confidence": edit.confidence,
            }
            for edit in trace.edits[:20]
        ],
    }


def analyze(
    dump: list[dict],
    gold_docs: dict[str, RelationDocument],
    *,
    alpha: float = 0.2,
    cal_ratio: float = 0.3,
) -> dict:
    """Repair + risk-controlled admission over a predicted-edge dump.

    Rebuilds each document's raw predicted graph from the dump, repairs it with
    the greedy consistency solver (emitting a `RepairTrace`), then admits edges
    under a CRC FNR bound whose ``tau`` is calibrated on a held-out ``cal_ratio``
    split of the *repaired* graphs — replicating the fixed repair∘admit map the
    online evaluator calibrates against. Consistency, stratified FNR and ECG
    reconstructability are all reported on the held-out remainder over the
    identical document set, so the raw -> repaired -> admitted trajectory is
    apples-to-apples (never a shrinking-population artifact).
    """
    records = [
        (gold_docs[r["doc_id"]], _edges_from_dump(r))
        for r in dump
        if r["doc_id"] in gold_docs
    ]
    solver = consistency_solvers.create("greedy")

    # Split for calibration; keep the test split non-empty (min guard) so the
    # reported trajectory always covers real documents.
    n_cal = min(len(records) - 1, max(1, int(len(records) * cal_ratio))) if len(records) > 1 else 0
    cal, test = records[:n_cal], records[n_cal:]

    cal_scores: list[float] = []
    for doc, edges in cal:
        repaired, _ = solver.solve_with_trace(_graph(doc, edges))
        cal_scores.extend(gold_edge_scores(repaired.edges, doc.gold_edges))
    admitter = edge_admission.create("crc", alpha=alpha).fit(cal_scores)

    staged: dict[str, list[tuple[RelationDocument, EventGraph]]] = {s: [] for s in STAGES}
    admitted_pairs: list[tuple[list[RelationEdge], list[RelationEdge]]] = []
    reachable_flags: list[bool] = []
    dropped = added = 0
    sample: RepairTrace | None = None
    for doc, edges in test:
        raw = _graph(doc, edges)
        repaired, trace = solver.solve_with_trace(raw)
        admitted = admitter.apply(repaired)
        dropped += trace.counts["dropped"]
        added += trace.counts["added"]
        if sample is None and trace.edits:
            sample = trace
        staged["raw"].append((doc, raw))
        staged["repaired"].append((doc, repaired))
        staged["repaired_admitted"].append((doc, admitted))
        admitted_pairs.append((list(admitted.edges), doc.gold_edges))
        reachable_flags.extend(ecg_reachable_flags(doc, admitted))

    return {
        "n_docs": len(test),
        "n_cal": len(cal),
        "alpha": alpha,
        "tau": float(admitter.threshold()),
        "consistency": {s: _sum_consistency([g for _, g in staged[s]]) for s in STAGES},
        "reconstruction": {s: _aggregate_reconstruction(staged[s]) for s in STAGES},
        "admission": stratified_admission_report(admitted_pairs),
        "repair_trace_totals": {"dropped": dropped, "added": added},
        "repair_trace_sample": _trace_sample(sample),
        # The per-query booleans Phase E's CS-CRP (`run_cross_stage`) consumes.
        "reachable_flags": reachable_flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path, help="per-doc predicted edges (jsonl)")
    parser.add_argument("--gold", required=True, type=Path, help="MAVEN-ERE gold jsonl (valid)")
    parser.add_argument("--alpha", type=float, default=0.2, help="CRC gold-FNR bound")
    parser.add_argument("--cal-ratio", type=float, default=0.3, help="held-out calibration split")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/relations/consistency_repair_supervised.json"),
    )
    args = parser.parse_args()

    dump = list(read_jsonl(args.dump))
    gold_docs = {doc.doc_id: doc for doc in load_maven_ere(args.gold)}
    result = analyze(dump, gold_docs, alpha=args.alpha, cal_ratio=args.cal_ratio)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"[repair] wrote analysis over {result['n_docs']} test docs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
