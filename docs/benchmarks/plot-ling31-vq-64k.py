#!/usr/bin/env python3
"""Render the Ling-3.1-flash VQ 2K-64K sweeps in the model-card style.

Three fresh ds4-bench processes per variant: MTP off over 2K-64K and two
embedded MTP drafts over 2K-32K (the MTP part, a 64K session and the bench's
host snapshot leave under 2 GiB on a 128 GB Spark). Requires matplotlib.
usage: plot-ling31-vq-64k.py DATA_DIR
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
DATE = DATA.name[-10:]
GENERATED = 128
STEP = 2048
VARIANTS = {"plain": 65536, "mtp2": 32768}
RUNS = {name: n for name in VARIANTS if (n := len(list(DATA.glob(f"{name}-*.csv"))))}
COLORS = {"mtp2": "#087f8c", "plain": "#687789"}
LABELS = {"mtp2": "MTP draft 2 (to 32K)", "plain": "MTP off"}
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Decode"))


def frontiers(name):
    return list(range(STEP, VARIANTS[name] + 1, STEP))


def load(name, repeat):
    with (DATA / f"{name}-{repeat}.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert [int(row["ctx_tokens"]) for row in rows] == frontiers(name)
    assert all(int(row["prefill_tokens"]) == STEP and int(row["gen_tokens"]) == GENERATED for row in rows)
    return rows


def summarize(runs):
    report = {"generated_tokens": GENERATED, "variants": {}}
    for name, samples in runs.items():
        variant = {"frontiers": frontiers(name)}
        for metric, _ in METRICS:
            values = [[float(row[metric]) for row in run] for run in samples]
            columns = list(zip(*values))
            run_means = [mean(v) for v in values]
            shared = [mean(v[: len(frontiers("mtp2"))]) for v in values]
            variant[metric] = {
                "run_means": run_means,
                "median_run_mean": median(run_means),
                # The MTP variant's 2K-32K range, for a like-for-like legend.
                "median_run_mean_to_32k": median(shared),
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
    fig.suptitle("Ling-3.1-flash · VQ experts + Q4_K/FP8 nonexperts", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, f"One DGX Spark / GB10 · 2K–64K card protocol · {DATE}",
             ha="center", color="#4b5563")
    for ax, (metric, heading) in zip(axes, METRICS):
        for name in RUNS:
            data = report["variants"][name]
            x = [f // 1024 for f in data["frontiers"]]
            values = data[metric]
            if RUNS[name] > 1:
                ax.fill_between(x, values["min"], values["max"], color=COLORS[name], alpha=0.17, linewidth=0)
            label = f"{LABELS[name]} · {values['median_run_mean']:,.1f} tok/s"
            if name == "plain" and "mtp2" in RUNS:
                label += f" (2K–32K: {values['median_run_mean_to_32k']:,.1f})"
            ax.plot(x, values["median"], color=COLORS[name], linewidth=2, label=label)
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", framealpha=0.92)
        ax.margins(y=0.2)
        if metric == "prefill_tps":
            # The 2K point is each process's first prefill (first-use
            # allocation); keep it in the data, but scale to the steady curve.
            steady = [v for name in RUNS for v in report["variants"][name][metric]["min"][1:]]
            top = max(v for name in RUNS for v in report["variants"][name][metric]["max"])
            ax.set_ylim(0.9 * min(steady), 1.06 * top)
    axes[-1].set_xlabel("Context tokens (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.06, "Curves: per-frontier medians; bands: observed min–max of three fresh runs per variant.\n"
             "Legend: median of run means. 2,048-token incremental prefill + 128 forced greedy tokens per frontier.\n"
             "ds4-bench · embedded MTP layer · forced non-EOS greedy · CUDA 13.4 · 300–2200 MHz",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA / "throughput.png", dpi=180)
    plt.close(fig)


def tables(report):
    plain, mtp = report["variants"]["plain"], report["variants"].get("mtp2")
    lines = ["# Ling-3.1-flash VQ 2K–64K samples", "",
             "Per-frontier medians (min–max across three fresh runs). 2,048-token",
             "incremental prefill and 128 forced greedy tokens per frontier.", "",
             "| Frontier | Prefill, MTP off | Decode, MTP off | Prefill, MTP 2 | Decode, MTP 2 |",
             "|---:|---:|---:|---:|---:|"]
    cell = lambda d, i: f"{d['median'][i]:.1f} ({d['min'][i]:.1f}–{d['max'][i]:.1f})"
    for i, frontier in enumerate(plain["frontiers"]):
        on = (f" {cell(mtp['prefill_tps'], i)} | {cell(mtp['gen_tps'], i)} |"
              if mtp and i < len(mtp["frontiers"]) else " — | — |")
        lines.append(f"| {frontier // 1024}K | {cell(plain['prefill_tps'], i)} | {cell(plain['gen_tps'], i)} |{on}")
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
