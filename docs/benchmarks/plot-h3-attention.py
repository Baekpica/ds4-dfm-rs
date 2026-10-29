#!/usr/bin/env python3
"""Plot the H3 attention sweep; requires matplotlib, as plot-qwen-ple-fp8.py does.

Data: docs/benchmarks/h3-attention-2026-10-27, three fresh runs per arm, one CSV
per run. Same convention as the rest of this directory: bands are the observed
min-max across the repeats, the line is the per-length median, and the legend
carries the median of the run means.
"""
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).with_name("h3-attention-2026-10-27")
COLORS = {"scalar": "#687789", "tensor": "#087f8c"}
LABELS = {"scalar": "Scalar kernel", "tensor": "Tensor core"}
METRICS = (
    ("tokens_per_second", "Attention throughput, log scale"),
    ("ms_per_call", "Time per call, log scale"),
)
REFERENCE = 18432  # the 768x768, 107-frame denoise length


def load_runs(arm):
    runs = []
    for repeat in range(1, 4):
        with (DATA / f"h3-attention-{arm}-{repeat}.csv").open() as source:
            runs.append(list(csv.DictReader(source)))
    assert all(len(run) == len(runs[0]) for run in runs)
    return runs


def plot():
    runs = {arm: load_runs(arm) for arm in LABELS}
    lengths = [int(row["sequence"]) for row in runs["scalar"][0]]
    summary = {}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    fig.suptitle("MiniMax-H3 attention · scalar vs tensor core", fontsize=18,
                 fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "One DGX Spark / GB10 · 56 heads · head dim 128 · BF16 in, FP32 accumulate · 2026-10-27",
             ha="center", color="#4b5563")
    for ax, (metric, heading) in zip(axes, METRICS):
        summary[metric] = {}
        for arm in LABELS:
            samples = [[float(row[metric]) for row in run] for run in runs[arm]]
            columns = list(zip(*samples))
            run_means = [mean(sample) for sample in samples]
            center = median(run_means)
            summary[metric][arm] = {"run_means": run_means,
                                    "median_run_mean": center}
            ax.fill_between(lengths, [min(c) for c in columns],
                            [max(c) for c in columns], color=COLORS[arm],
                            alpha=0.17, linewidth=0)
            unit = "tok/s" if metric == "tokens_per_second" else "ms"
            ax.plot(lengths, [median(c) for c in columns], color=COLORS[arm],
                    linewidth=2, label=f"{LABELS[arm]} · {center:,.1f} {unit}")
        if metric in ("ms_per_call", "tokens_per_second"):
            ax.set_yscale("log")
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second" if metric == "tokens_per_second" else "Milliseconds")
        ax.grid(alpha=0.2, which="both")
        ax.legend(loc="upper left" if metric == "tokens_per_second" else "lower right",
                  framealpha=0.92)
        ax.margins(y=0.2)
    axes[0].set_title("Attention throughput · tensor core "
                      f"{100 * (summary['tokens_per_second']['tensor']['median_run_mean'] / summary['tokens_per_second']['scalar']['median_run_mean'] - 1):+,.0f}%",
                      loc="left", fontsize=13, pad=10)
    axes[1].set_title(f"Time per call · {summary['ms_per_call']['scalar']['median_run_mean'] / summary['ms_per_call']['tensor']['median_run_mean']:.1f}x shorter on median",
                      loc="left", fontsize=13, pad=10)
    summary["ratio"] = {}
    for metric in ("tokens_per_second", "ms_per_call"):
        summary["ratio"][metric] = (summary[metric]["tensor"]["median_run_mean"] /
                                    summary[metric]["scalar"]["median_run_mean"])
    axes[-1].set_xlabel("Sequence length (tokens)")
    axes[-1].set_xticks(lengths)
    axes[-1].set_xlim(min(lengths), max(lengths))
    fig.text(0.5, 0.055,
             "Curves: per-length medians; bands: observed min–max of 3 fresh runs, 2 calls each.\n"
             "Legend: median of run means, tokens/s counts query rows attended. Same binary; the\n"
             "H3_DIT_ATTENTION_MMA switch is the only difference. Scalar arm's out_fnv held at\n"
             "c6196e81bc4634e0 (4,352) and 5c1e3810028a8fa5 (18,432); the tensor arm rounds P to\n"
             "BF16 and reduces through mma, so it is a benchmark arm and never a gated path.",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.86, bottom=0.175, left=0.1, right=0.97, hspace=0.28)
    target = DATA.parent / "h3-attention-2026-10-27.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    summary["reference_sequence"] = REFERENCE
    summary["scalar_out_fnv_held"] = {
        str(int(row["sequence"])): row["out_fnv"] for row in runs["scalar"][0]}
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {target}")


if __name__ == "__main__":
    plot()
