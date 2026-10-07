#!/usr/bin/env python3
"""Render the recorded paired medians; no inference or new measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "glm53-spark-2026-10-07.json"
OUTPUT = ROOT / "glm53-spark-paired"
PANELS = [
    ("attention", "8192", "prefill_tps", "Attention · 8K prefill", "+3.08%"),
    ("pool", "8192", "prefill_tps", "Pooled scores · 8K prefill", "+8.91%"),
    ("pool", "8192", "gen_tps", "Pooled scores · 8K decode", "+1.77%"),
    ("shared_q8", "2048", "gen_tps", "Shared Q8 · 2K decode", "+0.47%"),
]


def main():
    data = json.loads(DATA.read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 3.6))
    for ax, (name, ctx, metric, title, gain) in zip(axes, PANELS):
        result = data["rounds"][name]["frontiers"][ctx]
        median = result["median_tps"][metric]
        values = [median["off"], median["on"]]
        ax.bar([0, 1], values, color=["#a3afbd", "#287fbb"], width=0.58)
        for i, value in enumerate(values):
            ax.text(i, value + max(values) * 0.025, f"{value:.2f}", ha="center")
        for arm, row in result["arms"].items():
            x = 0 if arm.startswith("off") else 1
            ax.scatter(x, float(row["metrics"][metric]), s=12, color="#172637",
                       zorder=3)
        ax.set_xticks([0, 1], ["Off", "On"])
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_title(title + "\n" + gain, fontsize=11)
        ax.set_ylabel("tokens/s")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("GLM-5.3 Flash Uncensored · DGX Spark · paired rounds", fontsize=14)
    fig.text(0.5, 0.025,
             "One raw VMM owner / one fresh worker · SSD off · MTP off · 64 greedy output tokens\n"
             "Bars: matched medians; dots: individual arms. Each panel has its own retained baseline.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180)
    svg = OUTPUT.with_suffix(".svg")
    fig.savefig(svg)
    # Keep generated SVG clean for git whitespace checks.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
