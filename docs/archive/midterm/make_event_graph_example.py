#!/usr/bin/env python
"""Build and render a real event-centric graph example for the midterm defense.

This figure complements the metric/statistical plots: it shows what the
constructed event graph looks like at object level. The full graph has hundreds
of nodes and tens of thousands of closure edges, so the PPT figure intentionally
uses a compact true subgraph. The graph is first built as a NetworkX
MultiDiGraph with typed nodes and edges, then exported/rendered from that graph.

    uv run --with matplotlib python docs/midterm/make_event_graph_example.py
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

HERE = Path(__file__).parent
REPO = HERE.parent.parent
GRAPH = REPO / "data" / "processed" / "event_graph_zh" / "event_graph.json"
FIG = HERE / "figures"
DEFAULT_DOC_ID = "a66e7792d5be36ed81a39f5ee51b8dd4"


def load_graph(path: Path = GRAPH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_index(event_id: str) -> int:
    suffix = event_id.rsplit("::evt", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def select_example_subgraph(
    data: dict,
    *,
    doc_id: str = DEFAULT_DOC_ID,
    max_nodes: int = 8,
) -> dict:
    """Select the fixed, readable real subgraph used in the defense figure."""
    nodes = [
        node for node in data["nodes"].values()
        if node.get("doc_id") == doc_id
    ]
    if not nodes:
        raise ValueError(f"no nodes found for doc_id={doc_id}")
    nodes = sorted(nodes, key=lambda n: _event_index(n["event_id"]))[:max_nodes]
    candidate_ids = {n["event_id"] for n in nodes}
    edges = [
        edge for edge in data.get("edges", [])
        if edge.get("head_id") in candidate_ids
        and edge.get("tail_id") in candidate_ids
        and edge.get("relation_type") in {"coreference", "temporal"}
    ]
    if not edges:
        raise ValueError(f"selected subgraph has no displayable edges: {doc_id}")
    linked_ids = {e["head_id"] for e in edges} | {e["tail_id"] for e in edges}
    nodes = [n for n in nodes if n["event_id"] in linked_ids]
    return {"doc_id": doc_id, "nodes": nodes, "edges": edges}


# 边/图例的英文标签到中文的映射（仅渲染时替换，图数据仍保留英文键，便于测试与导出）。
ROLE_LABEL_CN = {
    "pledged_company": "标的公司",
    "pledger": "质押方",
    "time": "时间",
    "shares": "股份数量",
}
EVENT_REL_CN = {"BEFORE": "先于", "COREF": "共指"}


def _configure_fonts() -> str | None:
    """宋体 + Times New Roman 衬线体；font.family 用真实字体名列表触发逐字回退，
    拉丁/数字走 Times New Roman(无则 Liberation Serif)，中文回退到思源宋体。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    font_family = [
        "Times New Roman", "Liberation Serif",
        "SimSun", "宋体", "Noto Serif CJK SC", "Source Han Serif SC",
        "AR PL SungtiL GB", "Noto Sans CJK SC", "DejaVu Serif",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    family = [name for name in font_family if name in available] or ["DejaVu Serif"]
    cjk = next((name for name in ["SimSun", "宋体", "Noto Serif CJK SC",
                                   "Source Han Serif SC", "Noto Sans CJK SC"]
                if name in available), None)
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
    })
    plt.rcParams["font.family"] = family
    return cjk


def _ordered_events(nodes: list[dict]) -> list[dict]:
    return sorted(
        nodes,
        key=lambda n: (str(n.get("time_anchor") or "9999"), _event_index(n["event_id"])),
    )


def _event_node_id(event_id: str) -> str:
    return f"event:{event_id}"


def _add_typed_node(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    kind: str,
    label: str,
    value: str,
    role: str = "",
    source_event_id: str = "",
) -> None:
    if graph.has_node(node_id):
        return
    graph.add_node(
        node_id,
        kind=kind,
        label=label,
        value=value,
        role=role,
        source_event_id=source_event_id,
    )


def _add_role_edge(
    graph: nx.MultiDiGraph,
    event_id: str,
    target_id: str,
    role: str,
) -> None:
    graph.add_edge(
        _event_node_id(event_id),
        target_id,
        key=f"{role}:{event_id}->{target_id}",
        kind="role",
        relation_type=role,
        label=role,
        role=role,
        directed=True,
        confidence=1.0,
    )


def _assign_layout(graph: nx.MultiDiGraph, ordered_events: list[dict]) -> None:
    event_positions = [
        (0.7, 3.55),
        (3.65, 3.55),
        (6.75, 4.65),
        (6.75, 2.45),
    ]
    if len(ordered_events) > len(event_positions):
        event_positions.extend((9.7, 5.0 - i * 0.95) for i in range(len(ordered_events)))
    event_rank = {event["event_id"]: i for i, event in enumerate(ordered_events)}
    for event, (x, y) in zip(ordered_events, event_positions, strict=False):
        node_id = _event_node_id(event["event_id"])
        graph.nodes[node_id]["x"] = x
        graph.nodes[node_id]["y"] = y

    entity_positions = {
        "pledged_company": (3.65, 6.0),
        "pledger": (3.65, 0.65),
    }
    time_nodes = sorted(
        (node_id for node_id, data in graph.nodes(data=True) if data.get("kind") == "time"),
        key=lambda node_id: graph.nodes[node_id].get("value", ""),
    )
    time_positions = [
        (0.7, 5.15),
        (3.65, 5.15),
        (9.55, 3.55),
        (9.55, 5.25),
    ]
    share_offsets = [(0.0, -1.45), (0.0, -1.45), (2.65, 0.35), (2.65, -0.35)]

    for _node_id, data in graph.nodes(data=True):
        if data.get("kind") == "entity":
            x, y = entity_positions.get(data.get("role", ""), (1.0, 1.0))
            data["x"] = x
            data["y"] = y
        elif data.get("kind") == "value":
            source_id = data.get("source_event_id", "")
            source = graph.nodes[_event_node_id(source_id)]
            rank = event_rank.get(source_id, 0)
            dx, dy = share_offsets[min(rank, len(share_offsets) - 1)]
            data["x"] = source["x"] + dx
            data["y"] = source["y"] + dy

    for i, node_id in enumerate(time_nodes):
        x, y = time_positions[min(i, len(time_positions) - 1)]
        graph.nodes[node_id]["x"] = x
        graph.nodes[node_id]["y"] = y


def build_networkx_event_graph(sample: dict) -> nx.MultiDiGraph:
    """Build a typed event-centric KG subgraph from the selected real events."""
    graph = nx.MultiDiGraph(doc_id=sample["doc_id"], graph_type="event-centric-kg")
    ordered_events = _ordered_events(sample["nodes"])
    event_numbers = {event["event_id"]: i + 1 for i, event in enumerate(ordered_events)}

    for event in ordered_events:
        event_id = event["event_id"]
        event_node = _event_node_id(event_id)
        event_type = event.get("event_type") or "事件"
        graph.add_node(
            event_node,
            kind="event",
            label=f"E{event_numbers[event_id]}\n{event_type}",
            value=event_type,
            event_id=event_id,
            event_type=event_type,
            subject=event.get("subject") or "",
            time_anchor=event.get("time_anchor") or "",
            doc_id=event.get("doc_id") or sample["doc_id"],
            display_order=event_numbers[event_id],
        )

        args = event.get("arguments") or {}
        pledged_company = args.get("质押物所属公司") or event.get("subject") or ""
        if pledged_company:
            node_id = f"entity:pledged_company:{pledged_company}"
            _add_typed_node(
                graph,
                node_id,
                kind="entity",
                label=f"标的公司\n{pledged_company}",
                value=pledged_company,
                role="pledged_company",
            )
            _add_role_edge(graph, event_id, node_id, "pledged_company")

        pledger = args.get("质押方") or ""
        if pledger:
            node_id = f"entity:pledger:{pledger}"
            _add_typed_node(
                graph,
                node_id,
                kind="entity",
                label=f"质押方\n{pledger}",
                value=pledger,
                role="pledger",
            )
            _add_role_edge(graph, event_id, node_id, "pledger")

        event_time = event.get("time_anchor") or args.get("事件时间") or ""
        if event_time:
            node_id = f"time:{event_time}"
            _add_typed_node(
                graph,
                node_id,
                kind="time",
                label=f"时间\n{event_time}",
                value=event_time,
                role="time",
            )
            _add_role_edge(graph, event_id, node_id, "time")

        shares = args.get("质押股票/股份数量") or ""
        if shares:
            node_id = f"value:shares:{event_id}"
            _add_typed_node(
                graph,
                node_id,
                kind="value",
                label=f"股份数量\n{shares}",
                value=shares,
                role="shares",
                source_event_id=event_id,
            )
            _add_role_edge(graph, event_id, node_id, "shares")

    event_ids = {event["event_id"] for event in ordered_events}
    for edge_index, edge in enumerate(sample["edges"]):
        source = edge.get("head_id")
        target = edge.get("tail_id")
        if source not in event_ids or target not in event_ids:
            continue
        relation = edge.get("relation_type", "")
        label = edge.get("subtype") or ("COREF" if relation == "coreference" else relation.upper())
        graph.add_edge(
            _event_node_id(source),
            _event_node_id(target),
            key=f"{label}:{source}->{target}:{edge_index}",
            kind="event_event",
            relation_type=relation,
            subtype=edge.get("subtype") or "",
            label=label,
            directed=bool(edge.get("directed", True)),
            confidence=float(edge.get("confidence") or 0.0),
            evidence_count=len(edge.get("evidence") or []),
            source_relation=edge_index,
        )

    _assign_layout(graph, ordered_events)
    return graph


def build_display_graph(sample: dict) -> dict[str, list[dict]]:
    """Return a serializable graph view used by tests and downstream exports."""
    graph = build_networkx_event_graph(sample)
    nodes = [{"id": node_id, **dict(data)} for node_id, data in graph.nodes(data=True)]
    edges = [
        {"source": source, "target": target, "key": key, **dict(data)}
        for source, target, key, data in graph.edges(keys=True, data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def _cypher_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _cypher_map(data: dict[str, object]) -> str:
    pairs = ", ".join(f"`{key}`: {_cypher_value(value)}" for key, value in sorted(data.items()))
    return f"{{{pairs}}}"


def _neo4j_label(kind: str) -> str:
    return {
        "event": "EventNode",
        "entity": "EntityNode",
        "time": "TimeNode",
        "value": "ValueNode",
    }.get(kind, "GraphNode")


def _neo4j_relation(label: str) -> str:
    return {
        "pledged_company": "PLEDGED_COMPANY",
        "pledger": "PLEDGER",
        "time": "TIME",
        "shares": "SHARES",
        "BEFORE": "BEFORE",
        "COREF": "COREF",
    }.get(label, label.upper().replace("-", "_"))


def _cypher_rows(rows: list[dict[str, object]]) -> list[str]:
    return [f"  {_cypher_map(row)}" for row in rows]


def build_neo4j_cypher(graph: nx.MultiDiGraph) -> str:
    """Create an idempotent Neo4j import script for the display subgraph."""
    lines = [
        "// Import the midterm event-centric KG example.",
        "CREATE CONSTRAINT graph_node_id IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.id IS UNIQUE;",
        "",
    ]
    nodes_by_kind: dict[str, list[dict[str, object]]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        nodes_by_kind[str(data.get("kind", "node"))].append({"id": node_id, **dict(data)})

    for kind in ["event", "entity", "time", "value"]:
        rows = nodes_by_kind.get(kind, [])
        if not rows:
            continue
        lines.extend([
            "UNWIND [",
            ",\n".join(_cypher_rows(rows)),
            f"] AS row\nMERGE (n:GraphNode:{_neo4j_label(kind)} {{id: row.id}})\nSET n += row;",
            "",
        ])

    edges_by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source, target, key, data in graph.edges(keys=True, data=True):
        row = {"source": source, "target": target, "key": key, **dict(data)}
        edges_by_type[_neo4j_relation(str(data.get("label", "")))].append(row)

    for rel_type, rows in sorted(edges_by_type.items()):
        lines.extend([
            "UNWIND [",
            ",\n".join(_cypher_rows(rows)),
            (
                f"] AS row\nMATCH (s:GraphNode {{id: row.source}})\n"
                f"MATCH (t:GraphNode {{id: row.target}})\n"
                f"MERGE (s)-[r:{rel_type} {{key: row.key}}]->(t)\n"
                "SET r += row;"
            ),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def export_event_graph(
    graph: nx.MultiDiGraph,
    stem: Path = FIG / "fig_event_graph_example",
) -> dict[str, Path]:
    """Export the typed graph for graph tools and Neo4j."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    graphml = stem.with_suffix(".graphml")
    json_path = stem.with_suffix(".json")
    cypher = stem.with_suffix(".cypher")
    nx.write_graphml(graph, graphml)
    json_path.write_text(
        json.dumps(json_graph.node_link_data(graph, edges="edges"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cypher.write_text(build_neo4j_cypher(graph), encoding="utf-8")
    return {"graphml": graphml, "json": json_path, "cypher": cypher}


def _short_label(label: str) -> str:
    return "\n".join(
        textwrap.wrap(label, width=12, break_long_words=False, replace_whitespace=False)
    )


def _node_positions(graph: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    return {
        node_id: (float(data.get("x", 0.0)), float(data.get("y", 0.0)))
        for node_id, data in graph.nodes(data=True)
    }


def _visible_role_edge_labels(
    graph: nx.MultiDiGraph,
) -> dict[tuple[str, str, str], str]:
    """Keep the figure readable while every relation remains in the graph data."""
    first_shared_role: set[str] = set()
    labels: dict[tuple[str, str, str], str] = {}
    for source, target, key, data in graph.edges(keys=True, data=True):
        if data.get("kind") != "role":
            continue
        role = str(data.get("label", ""))
        target_kind = graph.nodes[target].get("kind")
        if target_kind == "entity":
            if role in first_shared_role:
                continue
            first_shared_role.add(role)
        labels[(source, target, key)] = ROLE_LABEL_CN.get(role, role)
    return labels


def _event_edge_labels(graph: nx.MultiDiGraph) -> dict[tuple[str, str, str], str]:
    return {
        (source, target, key): EVENT_REL_CN.get(str(data.get("label", "")), str(data.get("label", "")))
        for source, target, key, data in graph.edges(keys=True, data=True)
        if data.get("kind") == "event_event"
    }


def render_event_graph(
    sample: dict,
    output: Path = FIG / "fig_event_graph_example.png",
    *,
    graph: nx.MultiDiGraph | None = None,
) -> None:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    cjk_font = _configure_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = graph or build_networkx_event_graph(sample)
    positions = _node_positions(graph)

    fig, ax = plt.subplots(figsize=(13.4, 7.4), constrained_layout=True)
    node_styles = {
        "event": {"shape": "o", "color": "#d9eafe", "edge": "#1f5f9f", "size": 2700},
        "entity": {"shape": "s", "color": "#ddf2df", "edge": "#2e7d32", "size": 2500},
        "time": {"shape": "D", "color": "#eadcf8", "edge": "#6f45a5", "size": 1800},
        "value": {"shape": "h", "color": "#fff1b8", "edge": "#9a7600", "size": 2100},
    }
    for kind, style in node_styles.items():
        nodelist = [node_id for node_id, data in graph.nodes(data=True) if data.get("kind") == kind]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodelist,
            node_shape=style["shape"],
            node_color=style["color"],
            edgecolors=style["edge"],
            linewidths=1.6,
            node_size=style["size"],
            ax=ax,
        )
    labels = {
        node_id: _short_label(str(data.get("label", "")))
        for node_id, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8.7,
        font_family=cjk_font or "sans-serif",
        font_weight="bold",
        ax=ax,
    )

    role_edges = [
        (source, target, key)
        for source, target, key, data in graph.edges(keys=True, data=True)
        if data.get("kind") == "role"
    ]
    before_edges = [
        (source, target, key)
        for source, target, key, data in graph.edges(keys=True, data=True)
        if data.get("label") == "BEFORE"
    ]
    coref_edges = [
        (source, target, key)
        for source, target, key, data in graph.edges(keys=True, data=True)
        if data.get("label") == "COREF"
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=role_edges,
        edge_color="#8b929d",
        width=1.0,
        alpha=0.72,
        arrowstyle="-|>",
        arrowsize=9,
        min_source_margin=11,
        min_target_margin=11,
        connectionstyle="arc3,rad=0.06",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=before_edges,
        edge_color="#245d9a",
        width=2.1,
        alpha=0.95,
        arrowstyle="-|>",
        arrowsize=15,
        min_source_margin=18,
        min_target_margin=18,
        connectionstyle="arc3,rad=0.14",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=coref_edges,
        edge_color="#c96f16",
        width=2.2,
        alpha=0.95,
        style="dashed",
        arrowstyle="<->",
        arrowsize=14,
        min_source_margin=18,
        min_target_margin=18,
        connectionstyle="arc3,rad=-0.18",
        ax=ax,
    )

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=_visible_role_edge_labels(graph),
        font_size=7.7,
        font_family=cjk_font or "sans-serif",
        font_color="#4d5560",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "#c4c9d0", "alpha": 0.88},
        rotate=False,
        ax=ax,
        connectionstyle="arc3,rad=0.06",
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=_event_edge_labels(graph),
        font_size=8.5,
        font_family=cjk_font or "sans-serif",
        font_color="#173d64",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#9fb9d4", "alpha": 0.92},
        rotate=False,
        ax=ax,
        connectionstyle="arc3,rad=0.14",
    )

    handles = [
        mpatches.Patch(facecolor="#d9eafe", edgecolor="#1f5f9f", label="事件"),
        mpatches.Patch(facecolor="#ddf2df", edgecolor="#2e7d32", label="实体"),
        mpatches.Patch(facecolor="#eadcf8", edgecolor="#6f45a5", label="时间"),
        mpatches.Patch(facecolor="#fff1b8", edgecolor="#9a7600", label="数值"),
        mlines.Line2D([], [], color="#8b929d", linewidth=1.4, label="角色边"),
        mlines.Line2D([], [], color="#245d9a", linewidth=2.1, label="先于（时序）"),
        mlines.Line2D([], [], color="#c96f16", linewidth=2.1, linestyle="--", label="共指"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.58),
        borderaxespad=0.4,
        frameon=True,
        framealpha=0.95,
        prop={"family": cjk_font} if cjk_font else None,
    )
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 1.1, max(xs) + 1.15)
    ax.set_ylim(min(ys) - 0.95, max(ys) + 1.0)
    ax.set_axis_off()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def fig_event_graph_overview(
    output: Path = FIG / "fig_event_graph_overview.png",
    *,
    data: dict | None = None,
) -> dict[str, int]:
    """渲染 event_graph_zh 的整体概览：677 个事件节点按 13 类事件着色，画出证据接地的
    骨架边(共指/时序)；闭包派生的大量传递边不入图，避免成为不可读的"毛线团"。"""
    from collections import Counter

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _configure_fonts()
    data = data or load_graph()
    nodes = data["nodes"]
    id2type = {n["event_id"]: n.get("event_type") or "事件" for n in nodes.values()}

    graph = nx.Graph()
    for event_id, etype in id2type.items():
        graph.add_node(event_id, etype=etype)
    for edge in data.get("edges", []):
        if not edge.get("evidence"):
            continue  # 仅保留证据接地的骨架边，不含闭包派生边
        head, tail = edge.get("head_id"), edge.get("tail_id")
        if head in id2type and tail in id2type and head != tail:
            graph.add_edge(head, tail, rel=edge.get("relation_type", ""))

    backbone = [n for n in graph.nodes if graph.degree(n) > 0]
    sub = graph.subgraph(backbone).copy()
    pos = nx.spring_layout(sub, seed=42, k=0.55, iterations=120)

    type_counts = Counter(id2type[n] for n in sub.nodes)
    types = [t for t, _ in type_counts.most_common()]
    cmap = plt.cm.tab20
    color_of = {t: cmap(i % 20) for i, t in enumerate(types)}

    coref = [(u, v) for u, v, d in sub.edges(data=True) if d.get("rel") == "coreference"]
    temporal = [(u, v) for u, v, d in sub.edges(data=True) if d.get("rel") == "temporal"]

    fig, ax = plt.subplots(figsize=(13.0, 7.6), constrained_layout=True)
    nx.draw_networkx_edges(sub, pos, edgelist=temporal, edge_color="#6f9fd0",
                           width=0.5, alpha=0.40, ax=ax)
    nx.draw_networkx_edges(sub, pos, edgelist=coref, edge_color="#d9822b",
                           width=1.0, alpha=0.75, style="dashed", ax=ax)
    nx.draw_networkx_nodes(
        sub, pos,
        node_color=[color_of[id2type[n]] for n in sub.nodes],
        node_size=60, linewidths=0.3, edgecolors="#3a3a3a", ax=ax,
    )
    handles = [
        mpatches.Patch(facecolor=color_of[t], edgecolor="#3a3a3a", label=t)
        for t in types
    ]
    handles += [
        mlines.Line2D([], [], color="#6f9fd0", linewidth=1.4, label="时序边（证据）"),
        mlines.Line2D([], [], color="#d9822b", linewidth=1.6, linestyle="--", label="共指边（证据）"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
              borderaxespad=0.3, frameon=True, framealpha=0.95, fontsize=10)
    ax.set_axis_off()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "nodes_total": len(id2type),
        "nodes_in_backbone": len(backbone),
        "coref_edges": len(coref),
        "temporal_edges": len(temporal),
    }


def main() -> None:
    data = load_graph()
    sample = select_example_subgraph(data)
    graph = build_networkx_event_graph(sample)
    exports = export_event_graph(graph)
    render_event_graph(sample, graph=graph)
    overview = fig_event_graph_overview(data=data)
    print("event graph overview:", overview, f"-> {FIG / 'fig_event_graph_overview.png'}")
    print(
        "event graph example:",
        f"{graph.number_of_nodes()} typed nodes / {graph.number_of_edges()} typed edges",
        f"-> {FIG / 'fig_event_graph_example.png'}",
        "exports:",
        ", ".join(str(path) for path in exports.values()),
    )


if __name__ == "__main__":
    main()
