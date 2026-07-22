#!/usr/bin/env python
"""Re-score relation predictions offline (CPU) from a dumped prediction file.

This decouples the expensive GPU extraction pass from metric iteration: run
`evaluate_relations.py --dump-predictions PRED.jsonl` once (on the GPU box),
pull `PRED.jsonl` here, and recompute relation P/R/F1 (raw and temporal-closed)
plus coreference CoNLL locally — no model, no GPU. Gold is reloaded from the
original dataset, so changing the scoring rule never requires re-running the LLM.

    uv run python scripts/recompute_relation_metrics.py \
        --predictions runs/pred_phase2.jsonl \
        --gold-path data/processed/maven_ere/valid.jsonl \
        --output runs/eval_phase2_recomputed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finekg.core.eval import conll_coref_f1, relation_prf
from finekg.core.graph import coreference_clusters
from finekg.core.schema import EventGraph, RelationEdge, RelationType
from finekg.relations.data import load_ccks_causal, load_maven_ere

LOADERS = {"maven_ere": load_maven_ere, "ccks_causal": load_ccks_causal}


def _edge(d: dict) -> RelationEdge:
    return RelationEdge(
        head_id=d["head_id"],
        tail_id=d["tail_id"],
        relation_type=RelationType(d["relation_type"]),
        subtype=d.get("subtype", ""),
        directed=d.get("directed", True),
    )


def _clusters(nodes, edges) -> list[set[str]]:
    graph = EventGraph(nodes={n.event_id: n for n in nodes}, edges=edges)
    return coreference_clusters(graph, min_size=2)  # CoNLL convention: drop singletons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", required=True, type=Path, help="JSONL from --dump-predictions"
    )
    parser.add_argument(
        "--gold-path", required=True, type=Path, help="original dataset path (gold source)"
    )
    parser.add_argument("--loader", default="maven_ere", choices=sorted(LOADERS))
    parser.add_argument("--output", type=Path, help="write metrics JSON here")
    args = parser.parse_args()

    pred_by_doc: dict[str, list[RelationEdge]] = {}
    with args.predictions.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pred_by_doc[rec["doc_id"]] = [_edge(e) for e in rec["edges"]]

    gold_docs = {doc.doc_id: doc for doc in LOADERS[args.loader](args.gold_path)}

    pred_edges: list[RelationEdge] = []
    gold_edges: list[RelationEdge] = []
    pred_clusters: list[set[str]] = []
    gold_clusters: list[set[str]] = []
    missing = 0
    for doc_id, pedges in pred_by_doc.items():
        doc = gold_docs.get(doc_id)
        if doc is None:
            missing += 1
            continue
        pred_edges.extend(pedges)
        gold_edges.extend(doc.gold_edges)
        pred_clusters.extend(_clusters(doc.nodes, pedges))
        gold_clusters.extend(_clusters(doc.nodes, doc.gold_edges))

    metrics = {
        "n_docs": len(pred_by_doc) - missing,
        "relation_prf": {k: dict(v) for k, v in relation_prf(pred_edges, gold_edges).items()},
        "relation_prf_temporal_closed": {
            k: dict(v)
            for k, v in relation_prf(pred_edges, gold_edges, temporal_closure=True).items()
        },
        "coref_conll": conll_coref_f1(pred_clusters, gold_clusters),
        "predictions": str(args.predictions),
        "gold_path": str(args.gold_path),
    }
    if missing:
        metrics["missing_docs"] = missing
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
