#!/usr/bin/env python
"""Mid-term defense figures for the relation stage (ch2): SFT vs GRPO-RLVR.

Reads the eval/training JSONs pulled into docs/midterm/data/ and writes
presentation-quality PNGs into docs/midterm/figures/. No GPU needed.

    uv run --with matplotlib --with numpy python docs/midterm/make_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 图中统一用宋体 + Times New Roman 衬线体，字号约小四(12pt)。font.family 必须是
# 真实字体名列表(非"serif")，matplotlib 才会逐字回退：拉丁/数字用 Times New Roman
# (无则 Liberation Serif，与 Times 同度量)，中文回退到思源宋体(Noto Serif CJK SC)。
_FONT_FAMILY = [
    "Times New Roman", "Liberation Serif",
    "SimSun", "宋体", "Noto Serif CJK SC", "Source Han Serif SC",
    "AR PL SungtiL GB", "Noto Sans CJK SC", "DejaVu Serif",
]
_available = {f.name for f in fm.fontManager.ttflist}
_family = [f for f in _FONT_FAMILY if f in _available] or ["DejaVu Serif"]

# Presentation styling: serif fonts, white bg, colourblind-safe pair.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})
plt.rcParams["font.family"] = _family
SFT_C, GRPO_C = "#9aa7b1", "#1f77b4"   # muted grey vs strong blue
ACCENT = "#2f6f73"
ORANGE = "#d9822b"
GREEN = "#2f8f4e"
RED = "#b3433f"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def _round4(value: float) -> float:
    return round(float(value), 4)


def parse_fusion_sweep() -> dict:
    text = (DATA / "fuse_re_gcn.log").read_text(encoding="utf-8")
    match = re.search(r"BEST filtered MRR=([0-9.]+) at ws=([0-9.]+) wc=([0-9.]+)", text)
    if not match:
        raise ValueError("Cannot parse fusion sweep best row")
    best_mrr, ws, wc = match.groups()
    row = re.search(
        rf"ws={re.escape(ws)} wc={re.escape(wc)}\s+MRR=([0-9.]+)\s+H@1=([0-9.]+)\s+H@10=([0-9.]+)",
        text,
    )
    if not row:
        raise ValueError("Cannot parse fusion sweep metrics row")
    mrr, h1, h10 = row.groups()
    return {
        "mrr_tfilt": _round4(float(best_mrr)),
        "hits@1_tfilt": _round4(float(h1)),
        "hits@10_tfilt": _round4(float(h10)),
        "weights": f"ws={ws}, wc={wc}",
    }


def load_ch3_results() -> dict:
    freq = _fc_metrics(load("freq_icews14.json"))
    tgnn = _fc_metrics(load("tgnn_icews14/metrics.json"))
    path = _fc_metrics(load("path_rl_icews14/eval.json"))
    regcn = load("re_gcn_icews14/metrics.json")
    hybrid = load("hybrid_icews14/metrics.json")
    fusion = parse_fusion_sweep()
    return {
        "frequency": {k: _round4(freq[k]) for k in ["mrr", "hits@1", "hits@10"]},
        "temporal_gnn": {k: _round4(tgnn[k]) for k in ["mrr", "hits@1", "hits@10"]},
        # recurrency CPU 基线无独立 metrics.json；来源 RESULTS_SUMMARY_2026-06.md。
        "recurrency": {"mrr": 0.3560, "hits@1": 0.2830, "hits@10": 0.4880},
        "path_rl": {k: _round4(path[k]) for k in ["mrr", "hits@1", "hits@10"]},
        "re_gcn": {
            "mrr_tfilt": _round4(regcn["mrr_tfilt"]),
            "hits@1_tfilt": _round4(regcn["hits@1_tfilt"]),
            "hits@10_tfilt": _round4(regcn["hits@10_tfilt"]),
        },
        "hybrid_single": {
            "mrr_tfilt": _round4(hybrid["mrr_tfilt"]),
            "hits@1_tfilt": _round4(hybrid["hits@1_tfilt"]),
            "hits@10_tfilt": _round4(hybrid["hits@10_tfilt"]),
        },
        "fusion_sweep": fusion,
    }


def load_conformal_results() -> dict:
    data = load("conformal_gnn_icews14.json")
    return {
        name: {
            "coverage": _round4(metrics["conformal_coverage"]),
            "drift_gap": _round4(metrics["coverage_drift_gap"]),
            "set_size": round(float(metrics["conformal_set_size"]), 1),
        }
        for name, metrics in data["calibrators"].items()
    }


def annotate(ax, bars, fmt="{:.3f}", dy=0.005):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=10)


def _box(ax, xy, wh, title, body, face="#edf5f6", edge=ACCENT):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=11, linespacing=1.25)


def _arrow(ax, start, end, color="#4d5d63"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=1.6, color=color))


def fig_closed_loop_overview():
    fig, ax = plt.subplots(figsize=(11.2, 4.6), constrained_layout=True)
    ax.axis("off")
    boxes = [
        ((0.03, 0.45), (0.18, 0.34), "中文金融文本", "公告/新闻\nDuEE-Fin 等"),
        ((0.27, 0.45), (0.18, 0.34), "第一章 SARGE", "schema 接地事件表\n677 事件节点"),
        ((0.51, 0.45), (0.18, 0.34), "第二章关系图", "证据接地 + 一致性\n498 候选边 / 20683 闭包边"),
        ((0.75, 0.45), (0.20, 0.34), "第三章时序推理", "Path-RL / RE-GCN / 风控\n公开基准 MRR 0.411"),
    ]
    for args in boxes:
        _box(ax, *args)
    for x in [0.22, 0.46, 0.70]:
        _arrow(ax, (x, 0.62), (x + 0.04, 0.62))
    ax.text(0.50, 0.24, "数据闭环：文本证据 → 事件节点 → 事件关系边 → 历史路径预测", ha="center", fontsize=14, weight="bold", color=ACCENT)
    ax.text(0.50, 0.13, "逻辑闭环：证据接地 / 一致性 / 路径可信度 / 校准 贯穿抽取、构图、推理与风险控制", ha="center", fontsize=12)
    fig.savefig(FIG / "fig0_closed_loop_overview.png")
    plt.close(fig)


def fig_verifier_spine():
    fig, ax = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
    ax.axis("off")
    stages = [
        ("第一章节点", "证据接地\n论元回填原文证据", GREEN),
        ("第二章关系边", "证据接地 + 一致性\n过滤未接地边 / 修复冲突", ACCENT),
        ("第二章训练", "验证器作为奖励\n格式/接地/一致性/F1", ORANGE),
        ("第三章推理", "路径可信度 + 校准\n证据路径 / 覆盖风险", RED),
    ]
    xs = [0.08, 0.32, 0.56, 0.80]
    for x, (title, body, color) in zip(xs, stages):
        _box(ax, (x, 0.38), (0.16, 0.36), title, body, face="#f7fbfb", edge=color)
    for x in [0.25, 0.49, 0.73]:
        _arrow(ax, (x, 0.56), (x + 0.05, 0.56))
    ax.text(0.50, 0.18, "核心创新口径：验证器不是附属过滤器，而是在训练、推理、风控中承担不同角色", ha="center", fontsize=13, weight="bold")
    fig.savefig(FIG / "fig_verifier_spine.png")
    plt.close(fig)


def fig_framework():
    """总体框架图：数据流水线（三章）+ 贯穿三阶段的验证器主线层。"""
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    stage_face, stage_edge = "#eef4f5", ACCENT
    ver_face, ver_edge = "#fbeede", ORANGE
    centers = [2.4, 6.0, 9.6]
    pw, py, ph = 2.9, 3.35, 1.65
    stages = [
        ("第一章　事件抽取", "中文金融文本 → 带证据事件节点", "schema 约束的事件表生成"),
        ("第二章　关系构图", "事件节点 → 证据接地事件图谱", "共指 · 时序 · 因果 · 子事件"),
        ("第三章　时序推理", "事件图谱 → 预测与风险控制", "路径推理 · 强基线融合 · 校准"),
    ]
    for cx, (title, io, method) in zip(centers, stages):
        ax.add_patch(FancyBboxPatch(
            (cx - pw / 2, py), pw, ph,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.7, edgecolor=stage_edge, facecolor=stage_face,
        ))
        ax.text(cx, py + ph - 0.36, title, ha="center", va="center", fontsize=13, weight="bold", color="#15343a")
        ax.text(cx, py + ph - 0.85, io, ha="center", va="center", fontsize=10.3)
        ax.text(cx, py + 0.32, method, ha="center", va="center", fontsize=9.3, color="#566169")

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#3f4d53"))

    ymid = py + ph / 2
    arrow(centers[0] + pw / 2, centers[1] - pw / 2, ymid)
    arrow(centers[1] + pw / 2, centers[2] - pw / 2, ymid)
    arrow(0.95, centers[0] - pw / 2, ymid)
    arrow(centers[2] + pw / 2, 11.55, ymid)
    ax.text(0.5, ymid, "金融\n文本", ha="center", va="center", fontsize=10)
    ax.text(11.8, ymid, "可信\n决策", ha="center", va="center", fontsize=10)

    bx0, bx1, by, bh = 0.7, 11.3, 0.8, 1.8
    ax.add_patch(FancyBboxPatch(
        (bx0, by), bx1 - bx0, bh,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.7, edgecolor=ver_edge, facecolor=ver_face,
    ))
    ax.text(centers[1], by + bh - 0.32, "验证器：贯穿三阶段的可验证性主线",
            ha="center", va="center", fontsize=12, weight="bold", color="#9a5a16")
    roles = ["证据接地 → 门控", "一致性约束 → 训练奖励", "路径可信度 → 风险控制"]
    for cx, role in zip(centers, roles):
        ax.text(cx, by + 0.55, role, ha="center", va="center", fontsize=10.2, color="#7a4a12")
        ax.plot([cx, cx], [py, by + bh], ls=(0, (4, 3)), lw=1.2, color=ver_edge, alpha=0.8, zorder=0)

    fig.savefig(FIG / "fig_framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_architecture_detail():
    """细节型系统框架图（类 DNN 模块）：五个模块 + 验证器主线（loss/reward/risk），
    每个模块画出输入→算子层→输出的张量流，标注训练目标与张量形状。"""
    fig, ax = plt.subplots(figsize=(15.6, 9.6))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    # ---- 顶部标题与副标题 ----
    ax.text(7.8, 9.20, "Fin-EKG：可验证性约束引导的事件知识图谱构建与时序推理",
            ha="center", va="center", fontsize=15.0, weight="bold", color="#15343a")
    ax.text(7.8, 8.85, "Verifiable Constraint Propagation  ·  Gate (extraction) → Reward (relation) → RiskController (inference)",
            ha="center", va="center", fontsize=10.2, style="italic", color="#5b6b73")

    # ---- 五个模块（沿水平轴分布）----
    mods = [
        # (cx, title, sub,  input_dim,  layer_specs,  output_dim,  accent_color, face, kpi)
        (1.55, "输入层", "中文金融语料",
            "D ∈ ℝ^{n×d}",
            [("Tok", 5), ("Emb", 5), ("Enc", 4), ("Enc", 4), ("Enc", 4)],
            "h ∈ ℝ^{n×d′}",
            "#5b6b73", "#eef2f4", "n ≈ 2K tok / doc"),
        (4.60, "① 抽取层", "SARGE\nQLoRA + Schema Gen.",
            "x_D=[D;S;C_K;P]",
            [("QLoRA", 5), ("SFT", 5), ("Gen", 5), ("Valid", 4), ("Norm", 4)],
            "V={v_i}\n677 节点",
            "#2f7a4b", "#e7f1ee", "ChFinAnn F1 86.0\nSchemaOK 100%"),
        (7.80, "② 构图层", "GRPO-RLVR\nVerifiable Reward",
            "x=(D, V_W)",
            [("Emb", 4), ("Sampler\nG=8", 5), ("Reward\nR(x,y)", 5), ("Consist\nSolver", 5), ("Update\nclip+KL", 5)],
            "E={e_{ij}}\n497→20683",
            "#3b6b8a", "#eef4f5", "CoNLL 0.265→0.771\nP 0.445→0.565"),
        (11.00, "③ 推理层", "Path-RL × RE-GCN\nCounterfactual Faithful.",
            "q=(s_q,r_q,?,τ_q)",
            [("MDP\ns_0=s_q", 5), ("Rollout\nT-step", 5), ("R(ξ)\n+势塑形", 5), ("F(ξ)\n消融", 5), ("Fusion\nws,wc", 5)],
            "rank(o|q)\nMRR 0.411",
            "#9a5a16", "#fbeede", "filter MRR 0.411\nHits@1 0.309"),
        (14.10, "④ 输出层", "Conformal Risk Control\nACI / CRC / Weighted",
            "rank(o*|x)",
            [("nonconf\na_i", 4), ("Quantile\nq_α", 4), ("Set\nC(x)", 4), ("ACI\nα_{t+1}", 4), ("DriftGap", 4)],
            "C(x)={o}\nCov ≥ 1−α",
            "#b3433f", "#f6eaea", "ACI Cov 0.899\nDriftGap 0.20"),
    ]

    box_w, box_h, top = 2.80, 4.90, 8.10
    bot = top - box_h
    for cx, title, sub, inp, layers, out, edge, face, kpi in mods:
        x0 = cx - box_w / 2
        # 模块外框
        ax.add_patch(FancyBboxPatch(
            (x0, bot), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=2.0, edgecolor=edge, facecolor=face))
        # 标题
        ax.text(cx, top - 0.30, title, ha="center", va="center",
                fontsize=14.0, weight="bold", color=edge)
        # 副标题
        ax.text(cx, top - 0.70, sub, ha="center", va="center",
                fontsize=8.6, color="#4a5760")
        # 输入标注
        ax.text(cx, top - 1.18, inp, ha="center", va="center",
                fontsize=9.2, style="italic", color="#2f3b42",
                bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="#cbd1d4", lw=0.7))
        # 内部"算子层"——DNN 风格小圆点
        net_top = top - 1.65
        net_bot = bot + 1.10
        n = len(layers)
        span = net_top - net_bot
        for i, (name, k) in enumerate(layers):
            yc = net_top - (i + 0.5) * (span / n)
            xs = x0 + box_w * np.linspace(0.20, 0.80, k)
            for xi in xs:
                ax.add_patch(mpatches.Circle((xi, yc), 0.075,
                              fc=edge, ec=edge, alpha=0.85, lw=0.4))
            ax.text(x0 + 0.12, yc, name, ha="left", va="center",
                    fontsize=8.0, color="#2f3b42")
        # 输出标注
        ax.text(cx, bot + 0.65, out, ha="center", va="center",
                fontsize=9.2, weight="bold", color=edge,
                bbox=dict(boxstyle="round,pad=0.16", fc="#ffffff", ec=edge, lw=0.9))
        # KPI 角标
        ax.text(x0 + box_w - 0.10, bot + 0.18, kpi, ha="right", va="center",
                fontsize=7.8, color="#5b6b73", style="italic")

    # ---- 模块间正向数据流箭头 ----
    for i in range(len(mods) - 1):
        x_prev = mods[i][0]
        x_next = mods[i + 1][0]
        ax.add_patch(FancyArrowPatch(
            (x_prev + box_w / 2, (top + bot) / 2),
            (x_next - box_w / 2, (top + bot) / 2),
            arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#3f4d53"))

    # ---- 验证器主线：纵向贯穿 + 自上而下虚线回传到各模块 ----
    vy_top = top - 0.10
    vx_left, vx_right = 0.20, 15.40
    ax.plot([vx_left, vx_right], [vy_top, vy_top], color="#9a5a16", lw=2.0, alpha=0.85)
    ax.text((vx_left + vx_right) / 2, vy_top + 0.20,
            "Verifier Spine: 抽取 Gate → 关系 Reward → 推理 RiskController  (loss / reward / risk 反向回传)",
            ha="center", va="bottom", fontsize=9.6, color="#7a4a12", weight="bold")
    # 1→① Gate
    ax.add_patch(FancyArrowPatch(
        (mods[0][0], vy_top), (mods[0][0], top - 1.65),
        arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#c79a4a", linestyle=(0, (5, 3))))
    ax.text(mods[0][0], top - 0.92, "Gate  Valid(ê, S)", ha="center", va="top",
            fontsize=8.4, color="#7a4a12", weight="bold")
    # 2→② Reward
    ax.add_patch(FancyArrowPatch(
        (mods[1][0], vy_top), (mods[1][0], top - 1.65),
        arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#c79a4a", linestyle=(0, (5, 3))))
    ax.text(mods[1][0], top - 0.92, "Reward  R(x,y) = Σ wᵢRᵢ", ha="center", va="top",
            fontsize=8.4, color="#7a4a12", weight="bold")
    # 3→③ Faithfulness
    ax.add_patch(FancyArrowPatch(
        (mods[2][0], vy_top), (mods[2][0], top - 1.65),
        arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#c79a4a", linestyle=(0, (5, 3))))
    ax.text(mods[2][0], top - 0.92, "Faithfulness  F(ξ) = f(·) − f(·\\ξ)", ha="center", va="top",
            fontsize=8.4, color="#7a4a12", weight="bold")
    # 4→④ Risk
    ax.add_patch(FancyArrowPatch(
        (mods[3][0], vy_top), (mods[3][0], top - 1.65),
        arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#c79a4a", linestyle=(0, (5, 3))))
    ax.text(mods[3][0], top - 0.92, "Risk  α_{t+1} = α_t + γ(α − err_t)", ha="center", va="top",
            fontsize=8.4, color="#7a4a12", weight="bold")

    # ---- 底部说明带 ----
    ax.add_patch(FancyBboxPatch(
        (0.20, 0.08), 15.20, 0.65,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0, edgecolor="#cbd1d4", facecolor="#f5f7f8"))
    ax.text(0.35, 0.50, "数据接口：sarfe_extract → SARGE 节点 → GRPO 关系 → 时序图谱 → 预测集合",
            ha="left", va="center", fontsize=9.4, color="#3f4d53", weight="bold")
    ax.text(0.35, 0.24, "数据规约：event_graph_zh 自建数据集 677 节点 / 497 证据边 / 20,683 一致性闭包边    ·    公开基准：ChFinAnn · DuEE-Fin · MAVEN-ERE · ICEWS14",
            ha="left", va="center", fontsize=8.2, color="#5b6b73")

    fig.savefig(FIG / "fig_architecture_detail.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_architecture():
    """分层系统架构图（DNN 风格）：输入→抽取→构图→推理→输出 五层纵向堆叠，
    验证器作为纵向主线贯穿三阶段（门控/奖励/风险控制器）。用于替换报告图2-1。"""
    fig, ax = plt.subplots(figsize=(11.4, 7.6))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    col_x, col_w = 0.55, 7.35
    cx = col_x + col_w / 2
    layer_h, gap, top = 1.06, 0.30, 7.05
    layers = [
        ("输入层", "中文金融文本：上市公司公告 / 财经新闻",
         "分词 · 表面候选 · 事件 schema 提示", "#eef2f4", "#5b6b73"),
        ("抽取层　SARGE", "受 schema 约束的事件表生成",
         "文本 → 677 带证据事件节点（类型·角色·时间·原文证据）", "#e7f1ee", GREEN),
        ("构图层　关系抽取 + 证据接地 + 一致性求解", "共指 · 时序 · 因果 · 子事件 四类关系",
         "事件节点 → 事件图谱（497 证据边 → 20683 一致性闭包边）", "#eef4f5", ACCENT),
        ("推理层　Path-RL × RE-GCN 融合", "图上可回溯路径搜索 + 演化式图卷积强基线",
         "未来查询 → 候选实体排序 + 证据路径（过滤 MRR 0.411）", "#fbeede", ORANGE),
        ("输出层", "保形预测集合 · 风险控制 · 下游选股",
         "带覆盖率保证的预测 → 可信风控 / 投研决策", "#f6eaea", RED),
    ]
    ys = []
    for i, (title, line1, line2, face, edge) in enumerate(layers):
        y = top - i * (layer_h + gap) - layer_h
        ys.append((y, y + layer_h))
        ax.add_patch(FancyBboxPatch(
            (col_x, y), col_w, layer_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.9, edgecolor=edge, facecolor=face))
        ax.text(col_x + 0.20, y + layer_h - 0.27, title, ha="left", va="center",
                fontsize=12.5, weight="bold", color="#15343a")
        ax.text(cx, y + layer_h - 0.63, line1, ha="center", va="center", fontsize=10)
        ax.text(cx, y + 0.27, line2, ha="center", va="center", fontsize=9.2, color="#566169")

    # 层间向下箭头（数据流）
    for upper, lower in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((cx, upper[0]), (cx, lower[1]),
                     arrowstyle="-|>", mutation_scale=20, lw=2.0, color="#3f4d53"))
    ax.text(cx, ys[0][1] + 0.30, "海量非结构化金融文本", ha="center", va="bottom",
            fontsize=9.5, color="#5b6b73")
    ax.text(cx, ys[-1][0] - 0.34, "可信、可追溯、错误率可控的决策信号", ha="center", va="top",
            fontsize=9.5, weight="bold", color="#b3433f")

    # 验证器纵向主线（贯穿三阶段，类似主干/skip 连接）
    vx0, vw = col_x + col_w + 0.55, 2.55
    vy0, vy1 = ys[-1][0], ys[0][1]
    ax.add_patch(FancyBboxPatch(
        (vx0, vy0), vw, vy1 - vy0,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.9, edgecolor="#9a5a16", facecolor="#fff6ea"))
    ax.text(vx0 + vw / 2, vy1 - 0.28, "验证器主线\n可验证性贯穿三阶段", ha="center", va="top",
            fontsize=10.8, weight="bold", color="#9a5a16")
    identities = [
        (1, "证据门控  Gate", "过滤无证据 / schema 非法节点"),
        (2, "可验证奖励  Reward", "接地·一致性·F1 → GRPO 训练信号"),
        (3, "风险控制器  Risk", "保形预测控误覆盖（漂移鲁棒）"),
    ]
    for li, tag, desc in identities:
        ymid = (ys[li][0] + ys[li][1]) / 2
        ax.add_patch(FancyArrowPatch((col_x + col_w, ymid), (vx0, ymid),
                     arrowstyle="-|>", mutation_scale=12, lw=1.3,
                     linestyle=(0, (4, 3)), color="#c79a4a"))
        ax.text(vx0 + vw / 2, ymid + 0.17, tag, ha="center", va="center",
                fontsize=9.6, weight="bold", color="#7a4a12")
        ax.text(vx0 + vw / 2, ymid - 0.20, desc, ha="center", va="center",
                fontsize=7.9, color="#7a4a12")

    fig.savefig(FIG / "fig_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_sarge_baseline_f1() -> dict:
    """读取公开基线 F1（Doc2EDAG/GIT/EPAL/SEELE），来源 docs/chapter1/evidence。"""
    bc = json.loads(
        (HERE.parent / "chapter1" / "evidence" / "baseline_constants.json").read_text(encoding="utf-8")
    )
    out: dict[str, dict[str, float]] = {"ChFinAnn": {}, "DuEE-Fin": {}}
    for row in bc["epal_rows"]:
        out[row["dataset"]][row["method"]] = row["f1"]
    for row in bc["seele_rows"]:
        out[row["dataset"]][row["method"]] = row["f1"]
    return out


def fig_sarge_baselines():
    datasets = ["ChFinAnn", "DuEE-Fin"]
    methods = ["Doc2EDAG", "GIT", "EPAL", "SEELE", "SARGE"]
    base = _load_sarge_baseline_f1()
    # SARGE 主指标（Legacy-FS micro-F1），来源 docs/chapter1/evidence/tables/01_chfinann_main.md、02_dueefin_main.md
    sarge_f1 = {"ChFinAnn": 86.0, "DuEE-Fin": 78.0}
    vals = {
        ds: [base[ds]["Doc2EDAG"], base[ds]["GIT"], base[ds]["EPAL"], base[ds]["SEELE"], sarge_f1[ds]]
        for ds in datasets
    }
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, constrained_layout=True)
    for ax, ds in zip(axes, datasets):
        colors = [SFT_C, SFT_C, "#7fb3d5", "#5dade2", GRPO_C]
        bars = ax.bar(x, vals[ds], color=colors, edgecolor="black", linewidth=0.6)
        annotate(ax, bars, fmt="{:.1f}", dy=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(ds)
        ax.set_ylim(55, 90)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1")
    fig.savefig(FIG / "fig_ch1_sarge_baselines.png")
    plt.close(fig)


def fig_event_graph_funnel():
    """构建与一致性闭包过程：左图为节点/候选/证据边规模(同一量级)，右图单独画
    闭包前后的边数(相差约两个数量级)，避免把三个小条压成贴地的失真比例。"""
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.6, 4.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    left_labels = ["事件节点", "候选关系边", "证据保留边"]
    left_values = [677, 498, 497]
    bl = axl.bar(left_labels, left_values, color=[GREEN, ORANGE, ACCENT],
                 edgecolor="black", linewidth=0.7)
    for b, v in zip(bl, left_values):
        axl.text(b.get_x() + b.get_width() / 2, v + 12, f"{v}", ha="center", fontsize=11)
    axl.set_ylabel("数量")
    axl.set_ylim(0, 760)
    axl.set_title("节点与候选/证据边规模")
    axl.grid(axis="y", alpha=0.25)

    right_labels = ["证据保留边", "一致性闭包边"]
    right_values = [497, 20683]
    br = axr.bar(right_labels, right_values, color=[ACCENT, GRPO_C],
                 edgecolor="black", linewidth=0.7)
    for b, v in zip(br, right_values):
        axr.text(b.get_x() + b.get_width() / 2, v + 350, f"{v}", ha="center", fontsize=11)
    axr.set_ylim(0, 22500)
    axr.set_title("一致性闭包前后")
    axr.grid(axis="y", alpha=0.25)

    fig.savefig(FIG / "fig_event_graph_funnel.png")
    plt.close(fig)


def fig_relation_f1(sft, grpo):
    types = ["coreference", "temporal", "causal", "subevent", "micro"]
    labels = ["Coref", "Temporal", "Causal", "Subevent", "Micro"]
    sft_f1 = [sft["relation_prf"][t]["f1"] for t in types]
    grpo_f1 = [grpo["relation_prf"][t]["f1"] for t in types]
    x = np.arange(len(types)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    b1 = ax.bar(x - w/2, sft_f1, w, label="SFT", color=SFT_C, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, grpo_f1, w, label="GRPO-RLVR", color=GRPO_C, edgecolor="black", linewidth=0.6)
    annotate(ax, b1); annotate(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("F1"); ax.set_ylim(0, max(grpo_f1) * 1.18)
    ax.set_title("Relation Extraction F1 by Type  (MAVEN-ERE valid, 710 docs)")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig1_relation_f1.png"); plt.close(fig)


def fig_coref_conll(sft, grpo):
    keys = ["muc_f1", "b_cubed_f1", "ceafe_f1", "conll_f1"]
    labels = ["MUC", "B³", "CEAFe", "CoNLL 平均"]
    sft_v = [sft["coref_conll"][k] for k in keys]
    grpo_v = [grpo["coref_conll"][k] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    b1 = ax.bar(x - w/2, sft_v, w, label="监督微调", color=SFT_C, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, grpo_v, w, label="GRPO-RLVR", color=GRPO_C, edgecolor="black", linewidth=0.6)
    annotate(ax, b1); annotate(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig_coref_conll.png"); plt.close(fig)


def fig_precision_recall(sft, grpo):
    # The "more AND more accurate" story on the micro level.
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    sft_v = [sft["relation_prf"]["micro"][m] for m in metrics]
    grpo_v = [grpo["relation_prf"]["micro"][m] for m in metrics]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    b1 = ax.bar(x - w/2, sft_v, w, label="SFT", color=SFT_C, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, grpo_v, w, label="GRPO-RLVR", color=GRPO_C, edgecolor="black", linewidth=0.6)
    annotate(ax, b1); annotate(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("score (micro over all relation types)")
    ax.set_ylim(0, max(grpo_v) * 1.2)
    ax.set_title("Micro Precision / Recall / F1  (tp: 797 → 4710)")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig5_micro_prf.png"); plt.close(fig)


def fig_phase_components(p0, p1, p2):
    comps = ["format", "grounding", "consistency", "task_f1", "total"]
    labels = ["Format", "Grounding", "Consistency", "Task-F1", "Total"]
    phases = [("Phase 0 (≤6)", p0, "#a6cee3"),
              ("Phase 1 (≤12)", p1, "#5fa2d6"),
              ("Phase 2 (≤24)", p2, "#1f5fa6")]
    x = np.arange(len(comps)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for i, (name, d, c) in enumerate(phases):
        vals = [d[k] for k in comps]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=c, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean reward (phase-local)"); ax.set_ylim(0, 1.08)
    ax.set_title("Verifiable-Reward Components across Curriculum Phases")
    ax.legend(loc="upper right", ncol=3); ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.02, "Grounding/consistency/format stay high (no reward hacking); "
             "task-F1 falls as the curriculum adds harder, denser windows.",
             ha="center", fontsize=10, style="italic")
    fig.savefig(FIG / "fig4_phase_components.png", bbox_inches="tight"); plt.close(fig)


def fig_reward_curve(curve0, curve12):
    def smooth(v, k=5):
        v = np.asarray(v, float)
        if len(v) < k:
            return v
        return np.convolve(v, np.ones(k) / k, mode="same")
    total = [w["total"] for w in curve0] + [w["total"] for w in curve12]
    tf = [w["task_f1"] for w in curve0] + [w["task_f1"] for w in curve12]
    gr = [w["grounding"] for w in curve0] + [w["grounding"] for w in curve12]
    steps = np.arange(len(total))
    b1 = len(curve0)                       # phase 0 | phase 1
    b2 = len(curve0) + len(curve12) // 2   # phase 1 | phase 2 (phases are equal-length, 300 steps each)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    ax.plot(steps, smooth(total), color="#1f77b4", lw=2.2, label="total reward")
    ax.plot(steps, smooth(gr), color="#2ca02c", lw=1.6, alpha=0.8, label="grounding")
    ax.plot(steps, smooth(tf), color="#d62728", lw=1.6, alpha=0.8, label="task-F1")
    for b in (b1, b2):
        ax.axvline(b, color="grey", ls="--", lw=1)
    ytop = ax.get_ylim()[1]
    for xpos, lab in [(b1 / 2, "Phase 0\n(≤6)"), ((b1 + b2) / 2, "Phase 1\n(≤12)"),
                      ((b2 + len(total)) / 2, "Phase 2\n(≤24)")]:
        ax.text(xpos, ytop * 0.99, lab, fontsize=9, va="top", ha="center", color="grey")
    ax.set_xlabel("training window (64 sampled completions each)")
    ax.set_ylabel("reward (windowed mean)")
    ax.set_title("GRPO-RLVR Reward over Training")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig3_reward_curve.png"); plt.close(fig)


def write_table(sft, grpo):
    types = ["coreference", "temporal", "causal", "subevent", "micro"]
    lines = ["| Relation type | SFT P | SFT R | SFT F1 | GRPO P | GRPO R | GRPO F1 |",
             "|---|---|---|---|---|---|---|"]
    for t in types:
        s, g = sft["relation_prf"][t], grpo["relation_prf"][t]
        s_f1 = f"{s['f1']:.3f}"
        g_f1 = f"{g['f1']:.3f}"
        if g["f1"] > s["f1"]:
            g_f1 = f"**{g_f1}**"
        elif s["f1"] > g["f1"]:
            s_f1 = f"**{s_f1}**"
        lines.append(f"| {t} | {s['precision']:.3f} | {s['recall']:.3f} | {s_f1} "
                     f"| {g['precision']:.3f} | {g['recall']:.3f} | {g_f1} |")
    lines += ["", "| Coref CoNLL | SFT | GRPO |", "|---|---|---|"]
    for k, lab in [("muc_f1", "MUC"), ("b_cubed_f1", "B³"), ("ceafe_f1", "CEAFe"), ("conll_f1", "CoNLL")]:
        lines.append(f"| {lab} | {sft['coref_conll'][k]:.3f} | **{grpo['coref_conll'][k]:.3f}** |")
    (HERE / "table_sft_vs_grpo.md").write_text("\n".join(lines) + "\n")


def fig_temporal_precision(sft, grpo):
    """ch2: temporal *ordering precision* — the comparable temporal signal.

    Recall/F1 are not comparable (MAVEN-ERE temporal gold is a dense transitive
    closure vs a sparse extractor); precision = fraction of asserted orderings
    that agree with the gold order, which is meaningful for both models.
    """
    sp = sft["relation_prf"]["temporal"]["precision"]
    gp = grpo["relation_prf"]["temporal"]["precision"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    bars = ax.bar(["SFT", "GRPO-RLVR"], [sp, gp], width=0.55,
                  color=[SFT_C, GRPO_C], edgecolor="black", linewidth=0.6)
    annotate(ax, bars)
    ax.set_ylabel("时序顺序精度")
    ax.set_ylim(0, max(sp, gp) * 1.25 + 1e-3)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        -0.02,
        "说明：MAVEN-ERE 时序标注包含传递闭包，当前生成式抽取较稀疏，因此以精度作为可比主指标。",
        ha="center",
        fontsize=9,
    )
    fig.savefig(FIG / "fig8_temporal_precision.png", bbox_inches="tight"); plt.close(fig)


def write_temporal_table(sft, grpo):
    """ch2 temporal: P/R/F1 with precision as the headline + closure caveat."""
    lines = ["| model | temporal P | temporal R | temporal F1 |",
             "|---|---|---|---|"]
    for name, d in [("SFT", sft), ("GRPO-RLVR", grpo)]:
        t = d["relation_prf"]["temporal"]
        lines.append(f"| {name} | **{t['precision']:.3f}** | {t['recall']:.3f} | {t['f1']:.3f} |")
    lines += [
        "",
        "> Headline temporal signal is **precision** (ordering accuracy on predicted pairs). "
        "Recall/F1 are not comparable — MAVEN-ERE temporal gold is a transitive closure (dense, "
        "n_gold≈110k) while the extractor is sparse by design. Closure-aware scoring "
        "(`relation_prf(temporal_closure=True)`) is implemented and unit-tested, but is a **no-op** "
        "on the current model: its BEFORE predictions don't chain, so closing them adds 0 edges. "
        "A denser / transitively-structured extractor would benefit.",
    ]
    (HERE / "table_ch2_temporal.md").write_text("\n".join(lines) + "\n")


def _fc_metrics(d):
    """Normalize a forecasting result to {mrr, hits@1, hits@3, hits@10}."""
    return d["metrics"] if "metrics" in d else d


def fig_forecasting_compare(ch3: dict):
    methods = [
        ("频率基线", ch3["frequency"]["mrr"], SFT_C),
        ("时序GNN基线", ch3["temporal_gnn"]["mrr"], "#9ecae1"),
        ("复发基线", ch3["recurrency"]["mrr"], "#74c476"),
        ("Path-RL", ch3["path_rl"]["mrr"], GRPO_C),
        ("RE-GCN", ch3["re_gcn"]["mrr_tfilt"], ORANGE),
        ("融合搜索", ch3["fusion_sweep"]["mrr_tfilt"], RED),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    labels = [m[0] for m in methods]
    vals = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.7)
    annotate(ax, bars, fmt="{:.3f}", dy=0.008)
    # 近年代表方法的同口径参照线（具体文章，见正文表7）。
    ax.axhline(0.420, ls="--", lw=1.3, color="#888888", label="RE-GCN(2021) ≈0.42")
    ax.axhline(0.440, ls="--", lw=1.3, color="#d9822b", label="TiRGN(2022) 0.440")
    ax.axhline(0.459, ls="--", lw=1.3, color="#b3433f", label="L2TKG(2023) 0.459")
    ax.set_ylabel("MRR（时间感知过滤口径）")
    ax.set_ylim(0, 0.50)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig_forecasting_compare.png")
    plt.close(fig)


def fig_pathrl_training(curve):
    ep = curve["epochs"]
    e = [r["epoch"] for r in ep]
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(e, [r["mean_reward"] for r in ep], color="#1f77b4", lw=2, marker="o", ms=3, label="平均奖励")
    ax.plot(e, [r["hit_rate"] for r in ep], color="#d62728", lw=2, marker="s", ms=3, label="命中率")
    ax.set_xlabel("训练轮次"); ax.set_ylabel("指标值")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig_pathrl_training.png"); plt.close(fig)


def fig_conformal_risk(conformal: dict):
    names = ["split", "aci", "weighted", "crc"]
    labels = ["静态划分\n(split)", "自适应\n(ACI)", "近因加权\n(weighted)", "风险控制\n(CRC)"]
    coverage = [conformal[n]["coverage"] for n in names]
    gap = [conformal[n]["drift_gap"] for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    b1 = axes[0].bar(x, coverage, color=[SFT_C, GREEN, ACCENT, ORANGE], edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.9, color=RED, ls="--", lw=1.3, label="目标 0.90")
    annotate(axes[0], b1, fmt="{:.3f}", dy=0.006)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.82, 0.93)
    axes[0].set_ylabel("覆盖率")
    axes[0].set_title("实际覆盖率")
    axes[0].legend(loc="lower right")
    b2 = axes[1].bar(x, gap, color=[SFT_C, GREEN, ACCENT, ORANGE], edgecolor="black", linewidth=0.6)
    annotate(axes[1], b2, fmt="{:.2f}", dy=0.006)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 0.34)
    axes[1].set_ylabel("漂移覆盖差距 ↓")
    axes[1].set_title("漂移覆盖失真")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig_conformal_risk_control.png")
    plt.close(fig)


def fig_roadmap():
    items = [
        ("2026.7-9", "补齐事件抽取触发词/时间证据\n完成第一章写作"),
        ("2026.9-11", "中文关系迁移 + 边质量抽样\nGRPO 消融"),
        ("2026.11-2027.1", "多数据集时序推理\nPath-RL / RE-GCN 消融"),
        ("2027.1-3", "风险控制 + 下游选择性预测\n形成扩展结果"),
        ("2027.3-4", "论文整合、复现说明\n预答辩材料"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = np.linspace(0.08, 0.88, len(items))
    ax.plot([xs[0], xs[-1]], [0.54, 0.54], lw=3, color=ACCENT)
    for i, (x, (date, text)) in enumerate(zip(xs, items)):
        ax.scatter([x], [0.54], s=220, color=GRPO_C if i < 3 else ORANGE, edgecolor="black", zorder=3)
        ax.text(x, 0.75, date, ha="center", fontsize=12, weight="bold")
        ax.text(x, 0.36, text, ha="center", va="top", fontsize=10, linespacing=1.25)
    fig.savefig(FIG / "fig_roadmap.png", bbox_inches="tight")
    plt.close(fig)


def write_fc_table(ch3: dict):
    lines = ["| Method | MRR | Hits@1 | Hits@10 | Note |", "|---|---|---|---|---|"]
    rows = [
        ("Frequency", ch3["frequency"]["mrr"], ch3["frequency"]["hits@1"], ch3["frequency"]["hits@10"], "naive baseline"),
        ("Temporal-GNN", ch3["temporal_gnn"]["mrr"], ch3["temporal_gnn"]["hits@1"], ch3["temporal_gnn"]["hits@10"], "simplified neural baseline"),
        ("Recurrency", ch3["recurrency"]["mrr"], ch3["recurrency"]["hits@1"], ch3["recurrency"]["hits@10"], "CPU copy baseline"),
        ("Path-RL", ch3["path_rl"]["mrr"], ch3["path_rl"]["hits@1"], ch3["path_rl"]["hits@10"], "path policy"),
        ("RE-GCN", ch3["re_gcn"]["mrr_tfilt"], ch3["re_gcn"]["hits@1_tfilt"], ch3["re_gcn"]["hits@10_tfilt"], "时间感知过滤"),
        ("融合搜索", ch3["fusion_sweep"]["mrr_tfilt"], ch3["fusion_sweep"]["hits@1_tfilt"], ch3["fusion_sweep"]["hits@10_tfilt"], ch3["fusion_sweep"]["weights"]),
    ]
    for name, mrr, h1, h10, note in rows:
        bold = "**" if name == "融合搜索" else ""
        lines.append(f"| {name} | {bold}{mrr:.3f}{bold} | {h1:.3f} | {h10:.3f} | {note} |")
    (HERE / "table_forecasting.md").write_text("\n".join(lines) + "\n")


def main():
    sft, grpo = load("eval_sft.json"), load("eval_phase2.json")
    p0 = load("relation_grpo/reward_means_phase0.json")
    p1 = load("relation_grpo/reward_means_phase1.json")
    p2 = load("relation_grpo/reward_means_phase2.json")
    curve0 = load("relation_grpo/reward_curve_phase0.json.bak")
    curve12 = load("relation_grpo/reward_curve.json")

    fig_relation_f1(sft, grpo)
    fig_coref_conll(sft, grpo)
    fig_precision_recall(sft, grpo)
    fig_phase_components(p0, p1, p2)
    fig_reward_curve(curve0, curve12)
    write_table(sft, grpo)
    fig_temporal_precision(sft, grpo)   # ch2 temporal: precision is the comparable signal
    write_temporal_table(sft, grpo)

    fig_framework()
    fig_architecture()
    fig_architecture_detail()
    fig_closed_loop_overview()
    fig_verifier_spine()
    fig_sarge_baselines()
    fig_event_graph_funnel()

    # ── ch3 forecasting (ICEWS14) + risk control ──
    ch3 = load_ch3_results()
    conformal = load_conformal_results()
    pr_b = load("path_rl_icews14/metrics.json")
    fig_forecasting_compare(ch3)
    fig_pathrl_training(pr_b)
    fig_conformal_risk(conformal)
    fig_roadmap()
    write_fc_table(ch3)

    print("wrote figures to", FIG)
    for p in sorted(FIG.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
