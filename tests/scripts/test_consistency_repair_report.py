"""Offline Phase-B analysis: consume a raw predicted-edge dump, apply the
consistency solver (with trace) then CRC admission, and report violation/cycle
before-after, stratified FNR + admitted-set size, and ECG reconstructability.

The orchestration is under test on a synthetic dump (no checkpoint / GPU): a
gold ECG plus a predicted dump carrying an injected causal cycle, so repair must
fire and every reported axis must be populated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ekg.core.schema import EventNode, EvidenceSpan, RelationEdge, RelationType
from ekg.relations.data.maven_ere import RelationDocument

_REPO = Path(__file__).resolve().parents[2]


def _load_script():
    path = _REPO / "scripts" / "consistency_repair_report.py"
    spec = importlib.util.spec_from_file_location("consistency_repair_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


crr = _load_script()

# ALPHA topology (see test_cgep_build): one 5-node ECG, query edge m2 -CAUSE-> m4.
_TOPOLOGY = [
    ("m1", "m2", "causal", "CAUSE"),
    ("m2", "m3", "causal", "PRECONDITION"),
    ("m2", "m4", "causal", "CAUSE"),
    ("m3", "m5", "subevent", "SUBEVENT_OF"),
    ("m1", "m5", "causal", "CAUSE"),
]


def _document(doc_id: str, triggers: list[str]) -> RelationDocument:
    keys = [f"m{i}" for i in range(1, 6)]
    nodes = [
        EventNode(
            event_id=f"{doc_id}::{k}", event_type=f"T_{t}", doc_id=doc_id, trigger=t,
            trigger_evidence=[EvidenceSpan(doc_id=doc_id, char_start=0, char_end=len(t),
                                           sent_id=i, text=t)],
        )
        for i, (k, t) in enumerate(zip(keys, triggers, strict=True))
    ]
    gold = [
        RelationEdge(head_id=f"{doc_id}::{h}", tail_id=f"{doc_id}::{t}",
                     relation_type=RelationType(rt), subtype=st)
        for h, t, rt, st in _TOPOLOGY
    ]
    return RelationDocument(
        doc_id=doc_id, nodes=nodes, gold_edges=gold,
        doc_text="\n".join(f"sentence {i} mentions {t}" for i, t in enumerate(triggers)),
        representative={f"E{i}": n.event_id for i, n in enumerate(nodes)},
    )


def _dump_record(doc_id: str) -> dict:
    edges = [
        {"head_id": f"{doc_id}::{h}", "tail_id": f"{doc_id}::{t}",
         "relation_type": rt, "subtype": st, "directed": True, "confidence": 0.9}
        for h, t, rt, st in _TOPOLOGY
    ]
    # Injected causal cycle m4 -CAUSE-> m1 (weakest), closing m1->m2->m4->m1.
    edges.append({"head_id": f"{doc_id}::m4", "tail_id": f"{doc_id}::m1",
                  "relation_type": "causal", "subtype": "CAUSE",
                  "directed": True, "confidence": 0.2})
    return {"doc_id": doc_id, "edges": edges}


def test_analyze_reports_full_structure_and_repairs_the_injected_cycle() -> None:
    docs = {
        d.doc_id: d
        for d in (_document("docA", ["attack", "riot", "march", "arrest", "trial"]),
                  _document("docB", ["flood", "evacuate", "rescue", "rebuild", "inquiry"]))
    }
    dump = [_dump_record("docA"), _dump_record("docB")]

    result = crr.analyze(dump, docs, alpha=0.2, cal_ratio=0.5)

    # every reported axis is populated
    assert set(result["consistency"]) == {"raw", "repaired", "repaired_admitted"}
    assert "marginal" in result["admission"] and "by_type" in result["admission"]
    assert set(result["reconstruction"]) == {"raw", "repaired", "repaired_admitted"}
    for stage in result["reconstruction"].values():
        assert "r1_reachability_rate" in stage and "r2_query_prf" in stage
    assert isinstance(result["tau"], float)

    # repair removes the injected causal cycle (before-after violation drop)
    raw_cycles = result["consistency"]["raw"]["causal_cycle_count"]
    repaired_cycles = result["consistency"]["repaired"]["causal_cycle_count"]
    assert raw_cycles >= 1.0
    assert repaired_cycles < raw_cycles
    assert result["repair_trace_totals"]["dropped"] >= 1
    # repaired graph recovers the gold query edge that raw's cycle did not obscure
    assert result["reconstruction"]["repaired"]["r1_reachability_rate"] == 1.0


def test_analyze_is_empty_safe() -> None:
    result = crr.analyze([], {}, alpha=0.1, cal_ratio=0.3)
    assert result["n_docs"] == 0
    assert result["admission"]["admitted_size"] == 0
