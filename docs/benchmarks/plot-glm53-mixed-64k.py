#!/usr/bin/env python3
"""Render the GLM-5.3 Flash Uncensored mixed-quant 2K-64K sweeps.

Fresh ds4-bench-perf processes against one raw VMM weight owner, MTP off:
`base-N.csv` (merged main before the campaign) and `final-N.csv` (after).
usage: uv run --with matplotlib==3.10.7 --python 3.12 plot-glm53-mixed-64k.py DATA_DIR
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(sys.argv[1])
FRONTIERS = list(range(2048, 65537, 2048))
GENERATED = 128
VARIANTS = (("base", "main 8c2068b5", "#687789"), ("final", "this campaign", "#087f8c"))
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Decode"))


def load(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS
    assert all(int(row["prefill_tokens"]) == 2048 and int(row["gen_tokens"]) == GENERATED for row in rows)
    return rows


def summarize():
    report = {"frontiers": FRONTIERS, "generated_tokens": GENERATED, "variants": {}}
    for name, _, _ in VARIANTS:
        runs = [load(path) for path in sorted(DATA.glob(f"{name}-*.csv"))]
        if not runs:
            continue
        variant = {"runs": len(runs)}
        for metric, _ in METRICS:
            values = [[float(row[metric]) for row in run] for run in runs]
            columns = list(zip(*values))
            run_means = [mean(v) for v in values]
            variant[metric] = {"run_means": run_means, "median_run_mean": median(run_means),
                               "median": [median(c) for c in columns],
                               "min": [min(c) for c in columns], "max": [max(c) for c in columns]}
        report["variants"][name] = variant
    return report


def plot(report):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = [f / 1024 for f in FRONTIERS]
    for axis, (metric, title) in zip(axes, METRICS):
        for name, label, color in VARIANTS:
            variant = report["variants"].get(name)
            if not variant:
                continue
            data = variant[metric]
            runs = variant["runs"]
            axis.plot(x, data["median"], color=color, linewidth=2,
                      label=f"{label} ({data['median_run_mean']:.1f} tok/s{'' if runs == 1 else f', {runs} runs'})")
            if runs > 1:
                axis.fill_between(x, data["min"], data["max"], color=color, alpha=0.18, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Context frontier (K tokens)")
        axis.set_ylabel("tok/s")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(frameon=False, loc="lower right" if metric == "prefill_tps" else "lower left")
    figure.suptitle("GLM-5.3 Flash Uncensored mixed quant on one DGX Spark: 2,048-token steps, 128 greedy tokens, MTP off",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(DATA / "throughput.png", dpi=150)


report = summarize()
(DATA / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
plot(report)
for name, variant in report["variants"].items():
    print(name, {metric: round(variant[metric]["median_run_mean"], 2) for metric, _ in METRICS})
