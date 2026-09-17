#!/usr/bin/env python3
"""Plot the Ling-3.0-flash-VL 2K–64K card sweep, #48 vs #49 vs #50; requires matplotlib."""
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).with_name("ling3-flash-vl-2026-09-17")
OUT = DATA.parent.parent / "ling3-flash-vl-2k-64k-throughput.png"
# Two #48 runs (a third was cut short); three runs each of #49 and #50.
REPEATS = {"base": 2, "new": 3, "r2": 3}
COLORS = {"base": "#687789", "new": "#b4532a", "r2": "#087f8c"}
LABELS = {"base": "8436382 (#48, absorbed MLA)",
          "new": "007f0e4 (#49, expanded MLA)",
          "r2": "3d7078b (#50, BF16 K/V, 64-key tiles)"}
FINAL = "r2"
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Greedy decode"))
FRONTIERS = list(range(2048, 65537, 2048))


def load_runs(tag):
    runs = []
    for repeat in range(1, REPEATS[tag] + 1):
        with (DATA / f"{tag}-{repeat}.csv").open() as source:
            rows = list(csv.DictReader(source))
        assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS
        assert all(int(row["prefill_tokens"]) == 2048 and int(row["gen_tokens"]) == 128
                   for row in rows)
        runs.append(rows)
    return runs


def plot():
    runs = {tag: load_runs(tag) for tag in LABELS}
    summary = {}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Ling-3.0-flash-VL · MQ-Q5-KDA-VIT-BF16", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, "One DGX Spark / GB10 · 2K–64K card protocol · 2026-09-17",
             ha="center", color="#4b5563")
    x = [f // 1024 for f in FRONTIERS]
    for ax, (metric, heading) in zip(axes, METRICS):
        summary[metric] = {}
        for tag in LABELS:
            samples = [[float(row[metric]) for row in run] for run in runs[tag]]
            columns = list(zip(*samples))
            run_means = [mean(sample) for sample in samples]
            center = median(run_means)
            summary[metric][tag] = {
                "run_means": run_means,
                "median_run_mean": center,
                "frontier_median": {str(f): median(c) for f, c in zip(FRONTIERS, columns)},
            }
            if len(samples) > 1:
                ax.fill_between(x, [min(c) for c in columns], [max(c) for c in columns],
                                color=COLORS[tag], alpha=0.17, linewidth=0)
            runs_note = f"{len(samples)} run" + ("s" if len(samples) > 1 else "")
            ax.plot(x, [median(c) for c in columns], color=COLORS[tag], linewidth=2,
                    label=f"{LABELS[tag]} · {center:,.1f} tok/s · {runs_note}")
        change = 100 * (summary[metric][FINAL]["median_run_mean"] /
                        summary[metric]["base"]["median_run_mean"] - 1)
        summary[metric]["change_percent"] = change
        ax.set_title(f"{heading} · #50 vs #48 {change:+.1f}%", loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", framealpha=0.92)
        ax.margins(y=0.2)
    axes[-1].set_xlabel("Context tokens (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.06, "Curves: per-frontier medians; bands: observed min–max over the runs of each binary.\n"
             "Legend: median of run means. 2,048-token incremental prefill + 128 greedy tokens per frontier, one warm session per process.\n"
             "Same artifact, prompt and resident VMM weight owner · no MTP (family has none) · user-managed 300–2200 MHz SM range",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    report = plot()
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for metric, result in report.items():
        print(metric, {tag: round(result[tag]["median_run_mean"], 2) for tag in LABELS},
              f"{result['change_percent']:+.2f}%")
