#!/usr/bin/env python3
"""Render the 2026-09-14 clock-capped Solar A/B; matplotlib, no model load."""
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
with (DOCS / "solar-open2-2026-09-14-rounds.csv").open() as f:
    cold = list(csv.DictReader(f))
assert len(cold) == 12 and {r["round"] for r in cold} == {"R1"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
fig.suptitle("Solar Open2 250B MXQ-v1 · DGX Spark / GB10", fontsize=18, y=.98)
fig.text(.5, .91,
         "Clock cap 300–2200 MHz (measured 2190) · 4,096-token chunks · K-FP8 / V-FP4 · greedy, no speculation",
         ha="center", fontsize=10, color="#475569")
colors = ("#64748b", "#0369a1")
for ax, metric, title in zip(axes, ("prefill_tps", "gen_tps"),
                             ("Cold prefill · median of 3",
                              "Decode after cold prefill · median of 3")):
    for offset, variant, label, color in zip(
            (-.18, .18), ("off", "on"),
            ("Pair kernel", "FATTN_WS default"), colors):
        values = []
        for ctx in (8192, 65536):
            subset = [float(r[metric]) for r in cold
                      if int(r["context"]) == ctx and r["variant"] == variant]
            assert len(subset) == 3
            values.append(median(subset))
        bars = ax.bar([offset, 1 + offset], values, width=.34, color=color,
                      label=label)
        ax.bar_label(bars, labels=[f"{v:,.2f}" for v in values],
                     padding=5, fontsize=10)
    ax.set(title=title, ylabel="Tokens / second", xticks=[0, 1],
           xticklabels=["8,192 prompt tokens", "65,536 prompt tokens"])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)

fig.text(.5, .035,
         "Independent cold ds4-bench requests, 64 output tokens. Byte-identical 196,608 frontier logits and 64 IDs.\n"
         "Not HTTP, not the uncapped Sept 7/12 campaigns. Do not mix those numbers onto this graph.",
         ha="center", fontsize=10, color="#475569")
fig.subplots_adjust(top=.80, bottom=.18, wspace=.28)
fig.savefig(DOCS / "solar-open2-2026-09-14-throughput.png", dpi=160)
