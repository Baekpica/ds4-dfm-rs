#!/usr/bin/env python3
"""Compare the V4.1 indexer score arms over one 2K-64K sweep.

Same presentation as plot-ds41-64k.py (Qwen model-card style), one coloured
line per arm instead of one arm per record.  Data layout:

    <DATA>/<arm>/run-{1..3}.csv   the speed-bench CSV, one frontier per row
    <DATA>/summary.json           medians, min-max and run means per arm
    <DATA>/tables.md              every sample
    <DATA>/throughput.png         the two panels

`python3 docs/benchmarks/plot-ds41-indexer.py`
"""
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).with_name("ds41-indexer-arm-2026-11-08")
FRONTIERS = list(range(2048, 65537, 2048))
REPEATS = 2
GENERATED = 128
# (directory, legend label, colour); the served default is the MXFP4 arm.
ARMS = (
    ("scalar", "scalar score kernel", "#6b7280"),
    ("s8", "s8 mma.sync", "#b45309"),
    ("mxf4", "native MXFP4 (default)", "#087f8c"),
)
METRICS = (
    ("prefill_tps", "Full prompt prefill"),
    ("gen_tps", "Decode (automatic MTP policy)"),
)


def load_runs(arm):
    runs = []
    for repeat in range(1, REPEATS + 1):
        with (DATA / arm / f"run-{repeat}.csv").open() as source:
            rows = list(csv.DictReader(source))
        assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS, arm
        assert all(int(row["gen_tokens"]) == GENERATED for row in rows), arm
        assert all(int(row["prefill_cached_tokens"]) == 0 for row in rows), arm
        for key, _ in METRICS:
            assert all(math.isfinite(float(row[key])) and float(row[key]) > 0 for row in rows), arm
        runs.append(rows)
    inputs = [[int(row["prefill_tokens"]) for row in run] for run in runs]
    assert all(row == inputs[0] for row in inputs), arm
    return runs


def summarize_arm(runs):
    report = {"metrics": {}, "frontiers": []}
    for key, _ in METRICS:
        run_means = [mean(float(row[key]) for row in run) for run in runs]
        report["metrics"][key] = {
            "run_means": run_means,
            "mean_of_run_means": mean(run_means),
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


def summarize(runs_by_arm):
    report = {"repeats": REPEATS, "arms": {}}
    for arm, _, _ in ARMS:
        report["arms"][arm] = summarize_arm(runs_by_arm[arm])
    return report


def write_tables(report):
    lines = ["# DeepSeek V4.1 Flash: indexer score arms, 2K-64K", "",
             "Tokens/second on one DGX Spark / GB10. Two fresh server processes",
             "per arm; every sample retained. Each request has zero reused tokens",
             "and 128 generated tokens. Frontiers come from the corpus byte",
             "calibration; the input column is the actual token count.", "",
             "[Conditions, graph and evidence](README.md).", ""]
    for key, heading in METRICS:
        lines.extend([f"## {heading}", "",
                      "Median per frontier, one column per arm:", "",
                      "| Frontier | Input tokens | scalar | s8 | MXFP4 | MXFP4 vs scalar |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for index in range(len(FRONTIERS)):
            row = report["arms"]["scalar"]["frontiers"][index]
            cells = []
            for arm, _, _ in ARMS:
                cells.append(report["arms"][arm]["frontiers"][index][key]["median"])
            ratio = 100.0 * (cells[2] / cells[0] - 1.0)
            lines.append(f"| {row['frontier_tokens'] // 1024}K | {row['actual_prompt_tokens']:,} | "
                         f"{cells[0]:,.1f} | {cells[1]:,.1f} | {cells[2]:,.1f} | "
                         f"{ratio:+.1f}% |")
        lines.append("")
        for arm, label, _ in ARMS:
            summary = report["arms"][arm]["metrics"][key]
            means = ", ".join(f"{value:,.1f}" for value in summary["run_means"])
            lines.append(f"- {label}: run means {means}; median "
                         f"**{summary['mean_of_run_means']:,.2f} tok/s**.")
        lines.extend(["", "Every sample:", "",
                      "| Frontier | " + " | ".join(f"{label} run 1-3 (min-max)" for _, label, _ in ARMS) + " |",
                      "|---:|" + "---|" * len(ARMS)])
        for index in range(len(FRONTIERS)):
            row = report["arms"]["scalar"]["frontiers"][index]
            cells = []
            for arm, _, _ in ARMS:
                value = report["arms"][arm]["frontiers"][index][key]
                runs = " / ".join(f"{sample:,.1f}" for sample in value["runs"])
                cells.append(f"{runs} ({value['min']:,.1f}-{value['max']:,.1f})")
            lines.append(f"| {row['frontier_tokens'] // 1024}K | " + " | ".join(cells) + " |")
        lines.append("")
    (DATA / "tables.md").write_text("\n".join(lines))


def plot(report):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("DeepSeek V4.1 Flash - indexer score kernel", fontsize=18,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, "One DGX Spark / GB10 - 2K-64K full-prefill sweep - 2026-11-08",
             ha="center", color="#4b5563")
    x = [row["frontier_tokens"] / 1024 for row in report["arms"]["scalar"]["frontiers"]]
    for ax, (key, heading) in zip(axes, METRICS):
        for arm, label, color in ARMS:
            values = report["arms"][arm]["frontiers"]
            center = report["arms"][arm]["metrics"][key]["mean_of_run_means"]
            ax.fill_between(x, [row[key]["min"] for row in values],
                            [row[key]["max"] for row in values],
                            color=color, alpha=0.12, linewidth=0)
            ax.plot(x, [row[key]["median"] for row in values], color=color, linewidth=2,
                    label=f"{label} - {center:,.1f} tok/s")
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", framealpha=0.92)
        ax.margins(y=0.2)
    axes[-1].set_xlabel("Context frontier (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.055,
             "Curves: per-frontier medians; bands: min-max of 2 fresh runs; legend: mean of the run means.\n"
             "Full prompt prefill + 128 greedy tokens - zero KV reuse - actual input counts in the report\n"
             "One binary, arm chosen by DS4_DS41_IDX_MMA - automatic MTP - CUDA graphs\n"
             "Clocks above the documented 300-2200 MHz range: see the record's conditions",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA / "throughput.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    runs_by_arm = {arm: load_runs(arm) for arm, _, _ in ARMS}
    report = summarize(runs_by_arm)
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_tables(report)
    plot(report)
    for arm, label, _ in ARMS:
        print(arm, json.dumps(report["arms"][arm]["metrics"]))
