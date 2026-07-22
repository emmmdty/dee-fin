"""Regression checks for the midterm event-graph example figure input."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx


def _load_module():
    path = Path(__file__).resolve().parents[2] / "docs" / "midterm" / "make_event_graph_example.py"
    spec = importlib.util.spec_from_file_location("make_event_graph_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_example_subgraph_uses_real_nodes_and_relation_types() -> None:
    mod = _load_module()
    data = mod.load_graph()

    sample = mod.select_example_subgraph(data)

    assert 4 <= len(sample["nodes"]) <= 8
    assert {n["doc_id"] for n in sample["nodes"]} == {mod.DEFAULT_DOC_ID}
    assert {n["subject"] for n in sample["nodes"]} == {"新奥股份"}
    relation_types = {e["relation_type"] for e in sample["edges"]}
    assert {"coreference", "temporal"} <= relation_types


def test_display_graph_separates_events_entities_times_and_values() -> None:
    mod = _load_module()
    sample = mod.select_example_subgraph(mod.load_graph())

    display = mod.build_display_graph(sample)

    node_kinds = {node["kind"] for node in display["nodes"]}
    assert {"event", "entity", "time", "value"} <= node_kinds
    edge_labels = {edge["label"] for edge in display["edges"]}
    assert {"BEFORE", "COREF", "pledger", "pledged_company", "time", "shares"} <= edge_labels


def test_networkx_event_graph_exports_graph_and_neo4j_files(tmp_path) -> None:
    mod = _load_module()
    sample = mod.select_example_subgraph(mod.load_graph())

    graph = mod.build_networkx_event_graph(sample)
    outputs = mod.export_event_graph(graph, tmp_path / "fig9_event_graph_example")

    assert isinstance(graph, nx.MultiDiGraph)
    assert {data["kind"] for _, data in graph.nodes(data=True)} >= {
        "event",
        "entity",
        "time",
        "value",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
    cypher = outputs["cypher"].read_text(encoding="utf-8")
    assert "MERGE (n:GraphNode:EventNode" in cypher
    assert "MERGE (s)-[r:PLEDGER" in cypher
    assert "MERGE (s)-[r:BEFORE" in cypher
