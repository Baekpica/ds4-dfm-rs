#!/usr/bin/env python3
"""Render the published Solar measurements; requires matplotlib, no model load."""
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
with (DOCS / "solar-open2-2026-09-07-rounds.csv").open() as f:
    cold = list(csv.DictReader(f))
with (DOCS / "solar-open2-2026-09-07-baseline-sweep.csv").open() as f:
    sweep = list(csv.DictReader(f))
assert len(cold) == 12 and {r["round"] for r in cold} == {"P1"}
assert [int(r["ctx_tokens"]) for r in sweep] == list(range(2048, 65537, 2048))
assert all(int(r["prefill_tokens"]) == 2048 and int(r["gen_tokens"]) == 128 for r in sweep)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Solar Open2 250B MXQ-v1 · DGX Spark / GB10", fontsize=19, y=.98)
fig.text(.5, .935, "Verified P1 improvement · 4,096-token chunks · K-FP8 / V-FP4 · greedy, no speculation",
         ha="center", fontsize=11, color="#475569")
colors = ("#64748b", "#0369a1")
for ax, metric, title in zip(axes[0], ("prefill_tps", "gen_tps"),
                           ("Cold prefill · median of 3", "Decode after cold prefill · median of 3")):
    for offset, variant, label, color in zip((-.18, .18), ("off", "on"),
                                            ("Baseline", "P1 fused MoE"), colors):
        values = []
        for ctx in (8192, 65536):
            subset = [float(r[metric]) for r in cold
                      if int(r["context"]) == ctx and r["variant"] == variant]
            assert len(subset) == 3
            values.append(median(subset))
        bars = ax.bar([offset, 1 + offset], values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:,.2f}" for v in values], padding=5, fontsize=10)
    ax.set(title=title, ylabel="Tokens / second", xticks=[0, 1],
           xticklabels=["8,192 prompt tokens", "65,536 prompt tokens"])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.17)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)

depth = [int(r["ctx_tokens"]) / 1024 for r in sweep]
for ax, metric, title in zip(axes[1], ("prefill_tps", "gen_tps"),
                           ("Baseline only · 2K incremental prefill", "Baseline only · 128-token decode")):
    values = [float(r[metric]) for r in sweep]
    ax.plot(depth, values, color=colors[0], linewidth=2, marker=".", markersize=5)
    ax.set(title=title, xlabel="Session depth (1K = 1,024 tokens)", ylabel="Tokens / second",
           xticks=[2, 8, 16, 32, 48, 64], xlim=(1, 67))
    ax.grid(alpha=.2)
    ax.annotate(f"{values[-1]:,.2f}", (depth[-1], values[-1]),
                xytext=(-10, 12), textcoords="offset points", ha="right", color=colors[0])

fig.text(.5, .035, "Top: independent cold requests, 64 output tokens. Bottom: one warm session, baseline da153be only.\n"
         "No optimized 2K–64K sweep was completed; do not infer its speedup from the baseline curves.",
         ha="center", fontsize=10, color="#475569")
fig.subplots_adjust(top=.86, bottom=.13, hspace=.48, wspace=.22)
fig.savefig(DOCS / "solar-open2-2026-09-07-throughput.png", dpi=160)
