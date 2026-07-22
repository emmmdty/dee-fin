#!/usr/bin/env python
"""Evaluate the relation stage on a dataset described by a YAML config.

Builds an event graph per document with the configured pipeline, then reports
relation P/R/F1, event-coreference CoNLL (MUC/B3/CEAFe) and global-consistency
diagnostics aggregated over the corpus. Runs on CPU with the heuristic baseline;
the LLM extractor additionally needs the `llm` extra and a GPU.

    uv run python scripts/evaluate_relations.py --config configs/relations/heuristic_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finekg.core.config import load_config
from finekg.core.eval import conll_coref_f1, consistency_report, relation_prf
from finekg.core.graph import coreference_clusters
from finekg.core.schema import EventGraph
from finekg.relations import RelationPipeline, RelationPipelineConfig
from finekg.relations.admission import admission_report, edge_admission, gold_edge_scores
from finekg.relations.data import load_ccks_causal, load_maven_ere
from finekg.relations.extractor.base import ExtractionContext

LOADERS = {"maven_ere": load_maven_ere, "ccks_causal": load_ccks_causal}


def _calibrate_admission(pipeline, cal_docs, spec):
    """Fit a CRC edge-admission threshold on the calibration documents.

    Returns an `EdgeAdmission` whose `apply` enforces the gold-FNR bound on the
    test graphs, or None when admission is disabled (the default).
    """
    method = (spec or {}).get("method", "none")
    if not spec or method == "none":
        return None
    score_field = spec.get("score_field", "confidence")
    cal_scores: list[float] = []
    for doc in cal_docs:
        context = ExtractionContext(doc_text={doc.doc_id: doc.doc_text}) if doc.doc_text else None
        graph = pipeline.build_graph(doc.nodes, context)
        cal_scores.extend(gold_edge_scores(graph.edges, doc.gold_edges, score_field))
    admitter = edge_admission.create(
        method, alpha=float(spec.get("alpha", 0.1)), score_field=score_field
    )
    return admitter.fit(cal_scores)


def _gold_clusters(doc) -> list[set[str]]:
    gold_graph = EventGraph(
        nodes={n.event_id: n for n in doc.nodes}, edges=doc.gold_edges
    )
    # min_size=2: CoNLL scoring excludes singletons on both sides — keeping
    # them inflates B³/CEAFe and makes scores incomparable with the literature.
    return coreference_clusters(gold_graph, min_size=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--path", type=Path, help="override the dataset path in the config")
    parser.add_argument("--output", type=Path, help="write metrics JSON here")
    parser.add_argument(
        "--model",
        type=str,
        help="override relations.extractor_kwargs.model_name",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        help="override relations.extractor_kwargs.adapter_path",
    )
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        help="write per-document predicted edges for offline CPU re-scoring",
    )
    parser.add_argument("--limit-docs", type=int, help="cap documents (sampling / smoke runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model or args.adapter:
        ekw = cfg.setdefault("relations", {}).setdefault("extractor_kwargs", {})
        if args.model:
            ekw["model_name"] = args.model
        if args.adapter:
            ekw["adapter_path"] = args.adapter
    pipeline = RelationPipeline(RelationPipelineConfig.from_dict(cfg))
    loader = LOADERS[cfg["data"]["loader"]]
    path = args.path or Path(cfg["data"]["path"])

    # Optional risk-controlled admission: calibrate a CRC threshold on a held-out
    # fraction of documents, then enforce the gold-FNR bound on the rest. Absent
    # the `admission` config this is a no-op and the loop is the original pass.
    docs = list(loader(path))
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    admission_spec = cfg.get("relations", {}).get("admission") or cfg.get("admission")
    admitter = None
    if admission_spec and admission_spec.get("method", "none") != "none":
        n_cal = max(1, int(len(docs) * float(admission_spec.get("cal_ratio", 0.3))))
        admitter = _calibrate_admission(pipeline, docs[:n_cal], admission_spec)
        docs = docs[n_cal:]  # evaluate on the held-out remainder

    pred_edges, gold_edges = [], []
    pred_clusters, gold_clusters = [], []
    consistency_totals: dict[str, float] = {}
    per_doc_pred: list[dict] = []
    n_docs = 0

    for doc in docs:
        context = ExtractionContext(doc_text={doc.doc_id: doc.doc_text}) if doc.doc_text else None
        graph = pipeline.build_graph(doc.nodes, context)
        if admitter is not None:
            graph = admitter.apply(graph)
        pred_edges.extend(graph.edges)
        if args.dump_predictions is not None:
            per_doc_pred.append(
                {
                    "doc_id": doc.doc_id,
                    "edges": [
                        {
                            "head_id": e.head_id,
                            "tail_id": e.tail_id,
                            "relation_type": e.relation_type.value,
                            "subtype": e.subtype,
                            "directed": e.directed,
                            "confidence": e.confidence,
                        }
                        for e in graph.edges
                    ],
                }
            )
        gold_edges.extend(doc.gold_edges)
        pred_clusters.extend(coreference_clusters(graph, min_size=2))
        gold_clusters.extend(_gold_clusters(doc))
        for key, value in consistency_report(graph).items():
            consistency_totals[key] = consistency_totals.get(key, 0.0) + value
        n_docs += 1
        if n_docs % 25 == 0:
            print(f"[eval] {n_docs}/{len(docs)} docs", flush=True)

    # `relation_prf` is exact-match (kept for backward comparability); the
    # `_temporal_closed` variant closes strict-order temporal predictions before
    # matching, which is the fair number against MAVEN-ERE's transitively-closed
    # temporal gold (see core/eval/relation.py and docs/archive/MIDTERM_HANDOFF.md §2-P1).
    metrics = {
        "n_docs": n_docs,
        "relation_prf": {k: dict(v) for k, v in relation_prf(pred_edges, gold_edges).items()},
        "relation_prf_temporal_closed": {
            k: dict(v)
            for k, v in relation_prf(pred_edges, gold_edges, temporal_closure=True).items()
        },
        "coref_conll": conll_coref_f1(pred_clusters, gold_clusters),
        "consistency_sum": consistency_totals,
        "config": str(args.config),
    }
    if admitter is not None:
        report = admission_report(pred_edges, gold_edges)
        metrics["admission"] = {"threshold": admitter.threshold(), **report}
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.dump_predictions is not None:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_predictions.open("w", encoding="utf-8") as fh:
            for rec in per_doc_pred:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[eval] wrote {len(per_doc_pred)} per-doc predictions to {args.dump_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
