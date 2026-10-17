#!/usr/bin/env python3
"""Render the GLM-5.3 Flash VQ 2K-64K sweeps in the Qwen model-card style.

Three fresh ds4-bench processes with two embedded MTP drafts, plus (when
present) one MTP-off process for the plain decode reference. Requires
matplotlib 3.11.
usage: plot-glm53-vq-64k.py [DATA_DIR]   (default: the latest sweep)
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("glm53-vq-spark-64k-2026-10-17")
DATE = DATA.name[-10:]
FRONTIERS = list(range(2048, 65537, 2048))
GENERATED = 128
# Runs present in DATA: name-1.csv, name-2.csv, ...
RUNS = {name: n for name in ("mtp2", "plain") if (n := len(list(DATA.glob(f"{name}-*.csv"))))}
COLORS = {"mtp2": "#087f8c", "plain": "#687789"}
LABELS = {"mtp2": "MTP draft 2", "plain": "MTP off"}
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Decode"))


def load(name, repeat):
    with (DATA / f"{name}-{repeat}.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS
    assert all(int(row["prefill_tokens"]) == 2048 and int(row["gen_tokens"]) == GENERATED for row in rows)
    return rows


def summarize(runs):
    report = {"frontiers": FRONTIERS, "generated_tokens": GENERATED, "variants": {}}
    for name, samples in runs.items():
        variant = {}
        for metric, _ in METRICS:
            values = [[float(row[metric]) for row in run] for run in samples]
            columns = list(zip(*values))
            run_means = [mean(v) for v in values]
            variant[metric] = {
                "run_means": run_means,
                "median_run_mean": median(run_means),
                "median": [median(c) for c in columns],
                "min": [min(c) for c in columns],
                "max": [max(c) for c in columns],
            }
        report["variants"][name] = variant
    return report


def plot(report):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("GLM-5.3 Flash · VQ + Q4_K nonexpert v3", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, f"One DGX Spark / GB10 · 2K–64K card protocol · {DATE}",
             ha="center", color="#4b5563")
    x = [f // 1024 for f in FRONTIERS]
    for ax, (metric, heading) in zip(axes, METRICS):
        for name in RUNS:
            data = report["variants"][name][metric]
            if RUNS[name] > 1:
                ax.fill_between(x, data["min"], data["max"], color=COLORS[name], alpha=0.17, linewidth=0)
            ax.plot(x, data["median"], color=COLORS[name], linewidth=2 if RUNS[name] > 1 else 1.4,
                    label=f"{LABELS[name]} · {data['median_run_mean']:,.1f} tok/s")
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right" if metric == "prefill_tps" else "upper right", framealpha=0.92)
        ax.margins(y=0.2)
        if metric == "prefill_tps":
            # The 2K point is each process's first prefill (first-use
            # allocation); keep it in the data, but scale to the steady curve.
            steady = [v for name in RUNS for v in report["variants"][name][metric]["min"][1:]]
            top = max(v for name in RUNS for v in report["variants"][name][metric]["max"])
            ax.set_ylim(0.9 * min(steady), 1.04 * top)
            first = report["variants"]["mtp2"][metric]["median"][0]
            ax.annotate(f"2K: {first:,.0f} (first prefill, below axis)", xy=(2, 0.9 * min(steady)),
                        xytext=(3, 0.93 * min(steady)), fontsize=9, color="#4b5563")
    axes[-1].set_xlabel("Context tokens (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    plain = " (MTP off: 1 run)" if "plain" in RUNS else ""
    fig.text(0.5, 0.06, f"Curves: per-frontier medians; bands: observed min–max of {RUNS['mtp2']} fresh MTP runs{plain}.\n"
             "Legend: median of run means. 2,048-token incremental prefill + 128 forced greedy tokens per frontier.\n"
             "ds4-bench · embedded MTP layer · forced non-EOS greedy · CUDA 13.4 · 300–2200 MHz",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA / "throughput.png", dpi=180)
    plt.close(fig)


def tables(report):
    lines = ["# GLM-5.3 Flash VQ 2K–64K samples", "",
             "Per-frontier medians (min–max across the three MTP runs). 2,048-token",
             "incremental prefill and 128 forced greedy tokens per frontier.", "",
             "| Frontier | Prefill, MTP 2 | Decode, MTP 2 |" + (" Prefill, MTP off | Decode, MTP off |" if "plain" in RUNS else ""),
             "|---:|---:|---:|" + ("---:|---:|" if "plain" in RUNS else "")]
    mtp, plain = report["variants"]["mtp2"], report["variants"].get("plain")
    for i, frontier in enumerate(FRONTIERS):
        cell = lambda d: f"{d['median'][i]:.1f} ({d['min'][i]:.1f}–{d['max'][i]:.1f})"
        off = f" {plain['prefill_tps']['median'][i]:.1f} | {plain['gen_tps']['median'][i]:.1f} |" if plain else ""
        lines.append(f"| {frontier // 1024}K | {cell(mtp['prefill_tps'])} | {cell(mtp['gen_tps'])} |{off}")
    (DATA / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    runs = {name: [load(name, r) for r in range(1, count + 1)] for name, count in RUNS.items()}
    report = summarize(runs)
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report)
    tables(report)
    for name in RUNS:
        v = report["variants"][name]
        print(name, {m: round(v[m]["median_run_mean"], 2) for m, _ in METRICS})
