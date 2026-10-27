#!/usr/bin/env python3
"""Render the MiniMax-M3.1-Flash VQ 2K-64K sweep in the model-card style.

Three fresh ds4-bench processes with the DSpark drafter (draft 3); an
optional earlier sweep is drawn dashed. Requires matplotlib 3.11.
usage: plot-minimax-vq-64k.py [DATA_DIR [BEFORE_DIR BEFORE_LABEL]]
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("minimax-vq-spark-64k-2026-10-24-gate")
BEFORE = (Path(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else None
CONDITIONS = DATA / "conditions.json"
CONDITIONS = json.loads(CONDITIONS.read_text()) if CONDITIONS.exists() else {}
DATE = CONDITIONS.get("date", "2026-10-24")
CLOCK = CONDITIONS.get("busy_sm_mhz")
PREFILL_CHUNK = CONDITIONS.get("prefill_chunk", 4096)
FRONTIERS = list(range(2048, 65537, 2048))
GENERATED = 128
NAME, COLOR, LABEL = "dspark3", "#087f8c", "DSpark draft 3"
RUNS = len(list(DATA.glob(f"{NAME}-*.csv")))
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Decode"))


def load(repeat, data=DATA):
    with (data / f"{NAME}-{repeat}.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS
    assert all(int(row["prefill_tokens"]) == 2048 and int(row["gen_tokens"]) == GENERATED for row in rows)
    return rows


def summarize(runs):
    variant = {}
    for metric, _ in METRICS:
        values = [[float(row[metric]) for row in run] for run in runs]
        columns = list(zip(*values))
        run_means = [mean(v) for v in values]
        variant[metric] = {
            "run_means": run_means,
            "median_run_mean": median(run_means),
            "median": [median(c) for c in columns],
            "min": [min(c) for c in columns],
            "max": [max(c) for c in columns],
        }
    return {"frontiers": FRONTIERS, "generated_tokens": GENERATED, "variants": {NAME: variant}}


def plot(report, before=None):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("MiniMax-M3.1-Flash · VQ (13-bit GVQ1, 8-bit residual) + Q4_K", fontsize=18,
                 fontweight="bold", y=0.98)
    clock = f" · SM {CLOCK[0]:,.0f}–{CLOCK[1]:,.0f} MHz" if CLOCK else ""
    fig.text(0.5, 0.93, f"One DGX Spark / GB10 · 2K–64K card protocol · {DATE}{clock}", ha="center", color="#4b5563")
    x = [f // 1024 for f in FRONTIERS]
    for ax, (metric, heading) in zip(axes, METRICS):
        data = report["variants"][NAME][metric]
        if RUNS > 1:
            ax.fill_between(x, data["min"], data["max"], color=COLOR, alpha=0.17, linewidth=0)
        ax.plot(x, data["median"], color=COLOR, linewidth=2,
                label=f"{LABEL} · {data['median_run_mean']:,.1f} tok/s")
        if before:
            old = before[0]["variants"][NAME][metric]
            ax.plot(x, old["median"], color="#687789", linewidth=1.3, linestyle="--",
                    label=f"{before[1]} · {old['median_run_mean']:,.1f} tok/s")
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right" if metric == "prefill_tps" else "upper right", framealpha=0.92)
        ax.margins(y=0.2)
    axes[-1].set_xlabel("Context tokens (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.06, f"Curves: per-frontier medians; bands: observed min–max of {RUNS} fresh runs.\n"
             "Legend: median of run means. 2,048-token incremental prefill + 128 forced greedy tokens per frontier.\n"
             f"524,288-token session: index keys resident, K/V pages in the SSD tier · {PREFILL_CHUNK:,}-row prefill cap\n"
             "ds4-bench · promessi_sposi.txt · DSpark drafter, 3 proposals per step · forced non-EOS greedy · CUDA 13.4",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.19, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA / "throughput.png", dpi=180)
    plt.close(fig)


def tables(report):
    v = report["variants"][NAME]
    lines = ["# MiniMax-M3.1-Flash VQ 2K–64K samples", "",
             f"Per-frontier medians (min–max across the {RUNS} runs), DSpark draft 3. 2,048-token",
             "incremental prefill and 128 forced greedy tokens per frontier.", "",
             "| Frontier | Prefill | Decode |", "|---:|---:|---:|"]
    for i, frontier in enumerate(FRONTIERS):
        cell = lambda d: f"{d['median'][i]:.1f} ({d['min'][i]:.1f}–{d['max'][i]:.1f})"
        lines.append(f"| {frontier // 1024}K | {cell(v['prefill_tps'])} | {cell(v['gen_tps'])} |")
    (DATA / "tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    report = summarize([load(r) for r in range(1, RUNS + 1)])
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    before = None
    if BEFORE:
        count = len(list(BEFORE[0].glob(f"{NAME}-*.csv")))
        before = (summarize([load(r, BEFORE[0]) for r in range(1, count + 1)]), BEFORE[1])
    plot(report, before)
    tables(report)
    v = report["variants"][NAME]
    print(NAME, {m: round(v[m]["median_run_mean"], 2) for m, _ in METRICS})
