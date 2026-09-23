#!/usr/bin/env python3
"""Plot three-run MiMo card sweeps; requires matplotlib.

Example: --series baseline='09ce6e6 (#56)' --series final='COMMIT'
Each prefix resolves to PREFIX-{1,2,3}.csv in --data.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


FRONTIERS = list(range(2048, 65537, 2048))
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Greedy decode"))
COLORS = ("#687789", "#087f8c", "#b4532a", "#7657a8")


def load_runs(data, prefix):
    runs, sources = [], []
    for repeat in range(1, 4):
        path = data / f"{prefix}-{repeat}.csv"
        with path.open() as source:
            rows = list(csv.DictReader(source))
        if [int(row["ctx_tokens"]) for row in rows] != FRONTIERS:
            raise ValueError(f"{path}: expected all 32 frontiers from 2K to 64K")
        if any(int(row["prefill_tokens"]) != 2048 or int(row["gen_tokens"]) != 128
               for row in rows):
            raise ValueError(f"{path}: expected 2048 incremental + 128 generated tokens")
        for metric, _ in METRICS:
            if any(not math.isfinite(float(row[metric])) or float(row[metric]) <= 0
                   for row in rows):
                raise ValueError(f"{path}: invalid {metric}")
        runs.append(rows)
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return runs, sources


def summarize(runs):
    result = {}
    for metric, _ in METRICS:
        columns = list(zip(*[[float(row[metric]) for row in run] for run in runs]))
        center = [median(column) for column in columns]
        result[metric] = {
            "median": center,
            "min": [min(column) for column in columns],
            "max": [max(column) for column in columns],
            "retention_64k_over_2k": center[-1] / center[0],
        }
    return result


def plot(series, out, note, metrics=METRICS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    single_panel = len(metrics) == 1
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(11, 6.5 if single_panel else 8),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.suptitle("MiMo-V2.6-Flash-RL · MQ-IQ2-XXS-XS-Q8", fontsize=17,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.91 if single_panel else 0.935,
             "One DGX Spark / GB10 · 2K–64K", ha="center", color="#4b5563")
    x = [frontier // 1024 for frontier in FRONTIERS]
    for ax, (metric, heading) in zip(axes, metrics):
        for index, entry in enumerate(series):
            values = entry["metrics"][metric]
            color = COLORS[index % len(COLORS)]
            ax.fill_between(x, values["min"], values["max"], color=color, alpha=0.16,
                            linewidth=0)
            ax.plot(x, values["median"], color=color, linewidth=2,
                    label=f'{entry["label"]} · 64K: {values["median"][-1]:,.2f} tok/s')
        ax.set_title(heading, loc="left", fontsize=13)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", framealpha=0.92)
        ax.margins(y=0.16)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Context tokens (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.075 if single_panel else 0.06,
             "Curves: per-frontier medians; bands: observed min–max across three fresh processes.\n"
             "2,048-token incremental prefill + 128 greedy tokens per frontier; one warm session per process.\n"
             + note, ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.82 if single_panel else 0.88,
                        bottom=0.25 if single_panel else 0.17,
                        left=0.1, right=0.97, hspace=0.3)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).with_name("mimo2-2026-09-23"))
    parser.add_argument("--series", action="append", required=True, metavar="PREFIX=LABEL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", choices=("both", "prefill", "decode"), default="both")
    parser.add_argument("--note", required=True, help="Verified artifact/MTP/clock conditions")
    args = parser.parse_args()
    series = []
    prefixes = set()
    for value in args.series:
        prefix, separator, label = value.partition("=")
        if not separator or not prefix or not label or prefix in prefixes:
            parser.error("each series must have a unique PREFIX and a nonempty LABEL")
        prefixes.add(prefix)
        runs, sources = load_runs(args.data, prefix)
        series.append({"prefix": prefix, "label": label, "sources": sources,
                       "metrics": summarize(runs)})
    comparison = {}
    if len(series) > 1:
        for metric, _ in METRICS:
            baseline = series[0]["metrics"][metric]
            final = series[-1]["metrics"][metric]
            comparison[metric] = {
                "final_vs_first_gain_pct": [100 * (new / old - 1) for old, new in
                                            zip(baseline["median"], final["median"])],
                "retention_change_percentage_points": 100 * (
                    final["retention_64k_over_2k"] - baseline["retention_64k_over_2k"]),
            }
    metrics = METRICS if args.metric == "both" else (METRICS[0 if args.metric == "prefill" else 1],)
    plot(series, args.out, args.note, metrics)
    args.out.with_suffix(".json").write_text(json.dumps({
        "frontiers": FRONTIERS, "note": args.note, "plotted_metric": args.metric, "series": series,
        "final_vs_first": comparison,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
