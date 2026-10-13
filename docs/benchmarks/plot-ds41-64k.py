#!/usr/bin/env python3
"""Render the three recorded V4.1 sweeps in the Qwen model-card style."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).with_name("ds41-spark-64k-2026-10-13")
FRONTIERS = list(range(2048, 65537, 2048))
REPEATS = 3
GENERATED = 128
COLOR = "#087f8c"
METRICS = (
    ("prefill_tps", "Full prompt prefill"),
    ("gen_tps", "Decode (automatic MTP policy)"),
)


def load_runs():
    runs = []
    for repeat in range(1, REPEATS + 1):
        with (DATA / f"run-{repeat}.csv").open() as source:
            rows = list(csv.DictReader(source))
        assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS
        assert all(int(row["gen_tokens"]) == GENERATED for row in rows)
        assert all(int(row["prefill_cached_tokens"]) == 0 for row in rows)
        for key, _ in METRICS:
            assert all(math.isfinite(float(row[key])) and float(row[key]) > 0 for row in rows)
        runs.append(rows)
    inputs = [[int(row["prefill_tokens"]) for row in run] for run in runs]
    assert all(row == inputs[0] for row in inputs)
    return runs


def summarize(runs):
    report = {"repeats": REPEATS, "frontiers": [], "metrics": {}}
    for key, _ in METRICS:
        run_means = [mean(float(row[key]) for row in run) for run in runs]
        report["metrics"][key] = {
            "run_means": run_means,
            "median_run_mean": median(run_means),
        }
    for index, frontier in enumerate(FRONTIERS):
        row = {"frontier_tokens": frontier,
               "actual_prompt_tokens": int(runs[0][index]["prefill_tokens"]),
               "generated_tokens": GENERATED}
        for key, _ in METRICS:
            values = [float(run[index][key]) for run in runs]
            row[key] = {"runs": values, "median": median(values),
                        "min": min(values), "max": max(values)}
        report["frontiers"].append(row)
    return report


def write_tables(report):
    lines = ["# DeepSeek V4.1 Flash: 2K–64K throughput", "",
             "Tokens/second on one DGX Spark / GB10 at 300–2200 MHz.",
             "Three fresh server processes; all samples retained. Each request has",
             "zero reused tokens and 128 generated tokens. Frontiers come from the",
             "corpus byte calibration; the input column is the actual token count.", "",
             "[Conditions, graph and evidence](README.md).", ""]
    for key, heading in METRICS:
        lines.extend([f"## {heading}", "",
                      "| Frontier | Input tokens | Run 1 | Run 2 | Run 3 | Median | Min–max |",
                      "|---:|---:|---:|---:|---:|---:|---:|"])
        for row in report["frontiers"]:
            value = row[key]
            samples = " | ".join(f"{sample:,.1f}" for sample in value["runs"])
            lines.append(f"| {row['frontier_tokens'] // 1024}K | {row['actual_prompt_tokens']:,} | "
                         f"{samples} | **{value['median']:,.1f}** | "
                         f"{value['min']:,.1f}–{value['max']:,.1f} |")
        summary = report["metrics"][key]
        lines.extend(["", f"Median of the three run means: **{summary['median_run_mean']:,.2f} tok/s**.", ""])
    (DATA / "tables.md").write_text("\n".join(lines))


def plot(report):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("DeepSeek V4.1 Flash · VQ + code GR", fontsize=18,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, "One DGX Spark / GB10 · 2K–64K full-prefill sweep · 2026-10-13",
             ha="center", color="#4b5563")
    x = [row["frontier_tokens"] / 1024 for row in report["frontiers"]]
    for ax, (key, heading) in zip(axes, METRICS):
        values = [row[key] for row in report["frontiers"]]
        center = report["metrics"][key]["median_run_mean"]
        ax.fill_between(x, [row["min"] for row in values], [row["max"] for row in values],
                        color=COLOR, alpha=0.17, linewidth=0)
        ax.plot(x, [row["median"] for row in values], color=COLOR, linewidth=2,
                label=f"VQ + code GR · {center:,.1f} tok/s")
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", framealpha=0.92)
        ax.margins(y=0.2)
    axes[-1].set_xlabel("Context frontier (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.06,
             "Curves: per-frontier medians; bands: min–max of 3 fresh runs; legend: median of run means.\n"
             "Full prompt prefill + 128 greedy tokens · zero KV reuse · actual input counts in the report\n"
             "Original Engram + code GR · automatic MTP · CUDA graphs · 300–2200 MHz\n"
             "DeepSeek V4.1 Flash ported to ds4-dfm-rs by JLFN (PR #80) · YoungAi VQ artifacts",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA / "throughput.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    report = summarize(load_runs())
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_tables(report)
    plot(report)
    print(json.dumps(report["metrics"], indent=2))
