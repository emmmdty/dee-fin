#!/usr/bin/env python
"""Mid-term defense figures for the event-extraction stage (ch1, SARGE).

Reads the self-built Chinese financial event graph and writes presentation
figures + a stats table describing what ch1 extracts (event types, structured
arguments, temporal/subject coverage) and how many evidence-grounded heuristic
relations survive the graph builder.

    uv run --with matplotlib --with numpy python docs/midterm/make_ch1_figures.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = Path(__file__).parent
REPO = HERE.parent.parent
GRAPH = REPO / "data" / "processed" / "event_graph_zh" / "event_graph.json"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 宋体 + Times New Roman 衬线体，字号约小四(12pt)。font.family 用真实字体名列表，
# 触发逐字回退：拉丁/数字 Times New Roman(无则 Liberation Serif)，中文回退思源宋体。
_FONT_FAMILY = [
    "Times New Roman", "Liberation Serif",
    "SimSun", "宋体", "Noto Serif CJK SC", "Source Han Serif SC",
    "AR PL SungtiL GB", "Noto Sans CJK SC", "DejaVu Serif",
]
_available = {f.name for f in fm.fontManager.ttflist}
_family = [f for f in _FONT_FAMILY if f in _available] or ["DejaVu Serif"]
_cjk = next((f for f in ["SimSun", "宋体", "Noto Serif CJK SC",
                         "Source Han Serif SC", "Noto Sans CJK SC"] if f in _available), None)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})
plt.rcParams["font.family"] = _family
BAR_C = "#1f77b4"

# English glosses for the bilingual table (figure keeps the Chinese labels).
GLOSS = {
    "股份回购": "Share buyback", "企业收购": "Acquisition", "中标": "Bid won",
    "亏损": "Loss", "高管变动": "Exec change", "解除质押": "Pledge release",
    "股东减持": "Shareholder sell-down", "质押": "Share pledge", "公司上市": "IPO",
    "企业融资": "Financing", "被约谈": "Regulatory inquiry", "企业破产": "Bankruptcy",
    "股东增持": "Shareholder buy-in", "信批违规": "Disclosure violation",
}


def load_stats() -> dict:
    data = json.loads(GRAPH.read_text(encoding="utf-8"))
    nodes = list(data["nodes"].values())
    edges = data.get("edges", [])
    n = len(nodes)
    docs = {v["doc_id"] for v in nodes}
    types = Counter(v["event_type"] for v in nodes)
    nargs = [len(v.get("arguments") or {}) for v in nodes]
    with_time = sum(1 for v in nodes if v.get("time_anchor"))
    with_subject = sum(1 for v in nodes if v.get("subject"))
    subjects = {v.get("subject") for v in nodes if v.get("subject")}
    return {
        "n_events": n,
        "n_docs": len(docs),
        "events_per_doc": n / len(docs),
        "n_types": len(types),
        "types": types,
        "nargs": nargs,
        "mean_args": sum(nargs) / n,
        "time_cov": with_time / n,
        "subject_cov": with_subject / n,
        "n_subjects": len(subjects),
        "n_edges": len(edges),
        "meta": data.get("metadata", {}),
    }


def fig_event_types(s: dict) -> None:
    items = s["types"].most_common()
    labels = [t for t, _ in items]
    counts = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    y = range(len(labels))
    bars = ax.barh(list(y), counts, color=BAR_C, edgecolor="black", linewidth=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # largest on top
    ax.set_xlabel("事件数量")
    ax.set_xlim(0, max(counts) * 1.12)
    for b, c in zip(bars, counts):
        ax.text(b.get_width() + max(counts) * 0.01, b.get_y() + b.get_height() / 2,
                str(c), va="center", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIG / "fig_ch1_event_types.png")
    plt.close(fig)


def fig_arg_richness(s: dict) -> None:
    nargs = s["nargs"]
    hi = max(nargs)
    bins = range(0, hi + 2)
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.hist(nargs, bins=bins, align="left", rwidth=0.85,
            color=BAR_C, edgecolor="black", linewidth=0.6)
    ax.axvline(s["mean_args"], color="#d62728", ls="--", lw=1.8,
               label=f"均值 = {s['mean_args']:.2f}")
    ax.set_xlabel("每个事件的结构化论元数量")
    ax.set_ylabel("事件数量")
    ax.set_xticks(range(0, hi + 1))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig_ch1_arg_richness.png")
    plt.close(fig)


def write_table(s: dict) -> None:
    lines = [
        "## 第一章事件抽取：自建中文金融事件集合",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Events extracted | {s['n_events']} |",
        f"| Source documents | {s['n_docs']} |",
        f"| Events / document | {s['events_per_doc']:.2f} |",
        f"| Distinct event types | {s['n_types']} |",
        f"| Mean arguments / event | {s['mean_args']:.2f} (max {max(s['nargs'])}) |",
        f"| Temporal anchor coverage | {s['time_cov']:.0%} |",
        f"| Subject coverage | {s['subject_cov']:.0%} ({s['n_subjects']} distinct subjects) |",
        f"| 一致性闭包后的图边数量 | {s['n_edges']} |",
        f"| Raw extractor candidate edges | {s['meta'].get('edges_raw', '?')} |",
        f"| Raw candidate edges dropped as ungrounded | "
        f"{s['meta'].get('edges_dropped_ungrounded', '?')} / "
        f"{s['meta'].get('edges_raw', '?')} |",
        "",
        "| Event type | 中文 | Count |",
        "|---|---|---|",
    ]
    for t, c in s["types"].most_common():
        lines.append(f"| {GLOSS.get(t, t)} | {t} | {c} |")
    lines += [
        "",
        f"> 说明：当前关系边来自证据接地的启发式图构建器。图边数量是在一致性求解器补充"
        f"时序传递边之后的结果；原始候选边统计发生在证据过滤和闭包之前。上游 SARGE "
        f"预测当前仍未稳定导出触发词 span，因此触发词证据字段需要后续补齐。",
    ]
    (HERE / "table_ch1_event_extraction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    s = load_stats()
    if not _cjk:
        print("WARNING: no CJK font found; Chinese labels may render as boxes.")
    fig_event_types(s)
    fig_arg_richness(s)
    write_table(s)
    print(f"ch1: {s['n_events']} events / {s['n_docs']} docs / {s['n_types']} types "
          f"/ {s['mean_args']:.2f} args-per-event  (CJK font: {_cjk})")
    print("wrote:")
    for p in ["figures/fig_ch1_event_types.png", "figures/fig_ch1_arg_richness.png",
              "table_ch1_event_extraction.md"]:
        print(" -", p)


if __name__ == "__main__":
    main()
