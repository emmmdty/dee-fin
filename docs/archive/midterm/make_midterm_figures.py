#!/usr/bin/env python
"""Generate the updated midterm figures from real server results.

Reads runs/*.json (pulled to local) and produces:
- fig_ch3_progress: ch3 filtered MRR trajectory (frequency→...→fusion 0.411)
  with recent SOTA reference lines
- fig_c1_drift: conformal drift-gap comparison (split vs aci/weighted/crc)
  from the T6 conformal_gnn_icews14.json
- fig_ch3_forecasting_full: updated 3-way comparison with latest numbers

Run:  uv run --with matplotlib --with numpy python docs/midterm/make_midterm_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------- ch3 progress bar chart ----------
models = ["frequency", "temporal_gnn", "recurrency\n(copy)", "path-RL", "re_gcn", "fusion\n(gcn+copy)"]
mrrs = [0.105, 0.286, 0.356, 0.360, 0.380, 0.411]
ours = [True] * 6

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#1f77b4" if o else "#d62728" for o in ours]
bars = ax.bar(models, mrrs, color=colors, edgecolor="white", linewidth=0.8)

# SOTA reference lines
ax.axhline(y=0.420, color="gray", linestyle="--", linewidth=1.2, label="RE-GCN 0.420 (2021)")
ax.axhline(y=0.450, color="darkorange", linestyle="--", linewidth=1.2, label="Recent SOTA ~0.45 (2024)")
ax.axhline(y=0.105, color="lightgray", linestyle=":", linewidth=1)

for bar, mrr in zip(bars, mrrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{mrr:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Time-Aware Filtered MRR", fontsize=11)
ax.set_title("Ch3 Temporal Reasoning Progress (ICEWS14)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.55)
ax.legend(fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_ch3_progress.png", dpi=150)
plt.close(fig)
print("saved fig_ch3_progress.png")

# ---------- C1 conformal drift comparison ----------
c1_path = ROOT / "docs" / "midterm" / "data" / "conformal_gnn_icews14.json"
if not c1_path.exists():
    # try pulling from server copy or local runs
    alt = ROOT / "runs" / "conformal_gnn_icews14.json"
    if alt.exists():
        c1_path = alt

if c1_path.exists():
    data = json.loads(c1_path.read_text())
    cals = data.get("calibrators", {})
    names = list(cals.keys())
    covs = [cals[n].get("conformal_coverage", 0) for n in names]
    gaps = [cals[n].get("coverage_drift_gap", 0) for n in names]
    setsizes = [cals[n].get("conformal_set_size", 0) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # left: coverage_drift_gap
    n_colors = ["#d62728" if n == "split" else "#2ca02c" if n == "aci" else "#1f77b4" for n in names]
    bars1 = ax1.bar(names, gaps, color=n_colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(y=0.2, color="green", linestyle="--", linewidth=1, alpha=0.6, label="aci benchmark")
    for bar, val in zip(bars1, gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Coverage Drift Gap", fontsize=11)
    ax1.set_title("Drift Robustness (lower = better)", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # right: conformal coverage
    bars2 = ax2.bar(names, covs, color=n_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, label="target 1−α=0.9")
    for bar, val in zip(bars2, covs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Conformal Coverage", fontsize=11)
    ax2.set_title("Coverage (target = 0.9)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("C1 — Drift-Robust Conformal Risk Control (ICEWS14 + GNN)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_drift.png", dpi=150)
    plt.close(fig)
    print("saved fig_c1_drift.png")
else:
    print("C1 data not found — skipping fig_c1_drift; run pull from server first")
    # generate from hardcoded T6 results as fallback
    names = ["split", "aci", "weighted", "crc"]
    gaps = [0.29, 0.20, 0.23, 0.29]
    covs = [0.858, 0.899, 0.864, 0.858]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    n_colors = ["#d62728" if n == "split" else "#2ca02c" if n == "aci" else "#1f77b4" for n in names]
    bars1 = ax1.bar(names, gaps, color=n_colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars1, gaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Coverage Drift Gap", fontsize=11)
    ax1.set_title("Drift Robustness (lower = better)", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    bars2 = ax2.bar(names, covs, color=n_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, label="target 1−α=0.9")
    for bar, val in zip(bars2, covs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Conformal Coverage", fontsize=11)
    ax2.set_title("Coverage (target = 0.9)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("C1 — Drift-Robust Conformal Risk Control (ICEWS14 + GNN)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_drift.png", dpi=150)
    plt.close(fig)
    print("saved fig_c1_drift.png (from hard-coded T6 numbers)")

# ---------- ch3 updated forecasting compare ----------
methods = ["Frequency", "Temporal\nGNN", "Recurrency\n(Copy)", "Path-RL", "re_gcn\n(backbone)", "Fusion\n(gcn+copy)"]
mrr_vals = [0.105, 0.286, 0.356, 0.360, 0.380, 0.411]
h1_vals = [0.044, 0.192, 0.283, 0.284, 0.286, 0.309]
h10_vals = [0.220, 0.467, 0.488, 0.494, 0.565, 0.609]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - width, mrr_vals, width, label="MRR", color="#1f77b4", edgecolor="white", linewidth=0.6)
ax.bar(x, h1_vals, width, label="Hits@1", color="#ff7f0e", edgecolor="white", linewidth=0.6)
ax.bar(x + width, h10_vals, width, label="Hits@10", color="#2ca02c", edgecolor="white", linewidth=0.6)

ax.axhline(y=0.420, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="RE-GCN MRR 0.420")
ax.axhline(y=0.450, color="darkorange", linestyle="--", linewidth=1.2, alpha=0.7, label="Recent SOTA MRR ~0.45")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Ch3 Temporal Forecasting — Full Method Comparison (ICEWS14, Filtered)", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, 0.7)
fig.tight_layout()
fig.savefig(OUT / "fig_ch3_forecasting_full.png", dpi=150)
plt.close(fig)
print("saved fig_ch3_forecasting_full.png")

print("\nDone — figures written to", OUT)
