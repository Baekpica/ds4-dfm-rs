#!/usr/bin/env python3
"""Render recorded V4.1 2K-64K server sweeps in the model-card style.

usage: plot-ds41-64k.py [DATA_DIR]

DATA_DIR (default: the 2026-10-13 record) holds one CSV per fresh server
process: `run-N.csv` for a single variant, or `base-N.csv` and `final-N.csv`
for a before/after record. An optional `plot.json` there overrides the
title, subtitle, variant labels and footer. Requires matplotlib.
"""
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "ds41-spark-64k-2026-10-13"
FRONTIERS = list(range(2048, 65537, 2048))
GENERATED = 128
# File prefix -> default legend label and color, in drawing order.
VARIANTS = (("base", "Before", "#687789"), ("run", "VQ + code GR", "#087f8c"), ("final", "After", "#087f8c"))
METRICS = (
    ("prefill_tps", "Full prompt prefill"),
    ("gen_tps", "Decode (automatic MTP policy)"),
)
DEFAULTS = {
    "title": "DeepSeek V4.1 Flash · VQ + code GR",
    "subtitle": "One DGX Spark / GB10 · 2K–64K full-prefill sweep · 2026-10-13",
    "heading": "DeepSeek V4.1 Flash: 2K–64K throughput",
    "labels": {},
    "footer": [
        "Curves: per-frontier medians; bands: min–max of 3 fresh runs; legend: median of run means.",
        "Full prompt prefill + 128 greedy tokens · zero KV reuse · actual input counts in the report",
        "Original Engram + code GR · automatic MTP · CUDA graphs · 300–2200 MHz",
        "DeepSeek V4.1 Flash ported to ds4-dfm-rs by JLFN (PR #80) · YoungAi VQ artifacts",
    ],
}


def settings():
    path = DATA / "plot.json"
    custom = json.loads(path.read_text()) if path.exists() else {}
    return {**DEFAULTS, **custom}


def load_runs(prefix):
    runs = []
    for path in sorted(DATA.glob(f"{prefix}-*.csv"), key=lambda p: int(p.stem.split("-")[-1])):
        with path.open() as source:
            rows = list(csv.DictReader(source))
        assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS, path
        # A request may stop at the model's own end before the limit.
        assert all(0 < int(row["gen_tokens"]) <= GENERATED for row in rows), path
        assert all(int(row["prefill_cached_tokens"]) == 0 for row in rows), path
        for key, _ in METRICS:
            assert all(math.isfinite(float(row[key])) and float(row[key]) > 0 for row in rows), path
        runs.append(rows)
    inputs = [[int(row["prefill_tokens"]) for row in run] for run in runs]
    assert all(row == inputs[0] for row in inputs), prefix
    return runs


def summarize(runs):
    report = {"repeats": len(runs), "frontiers": [], "metrics": {}}
    for key, _ in METRICS:
        run_means = [mean(float(row[key]) for row in run) for run in runs]
        report["metrics"][key] = {
            "run_means": run_means,
            "median_run_mean": median(run_means),
        }
    for index, frontier in enumerate(FRONTIERS):
        row = {"frontier_tokens": frontier,
               "actual_prompt_tokens": int(runs[0][index]["prefill_tokens"]),
               "generated_tokens": int(runs[0][index]["gen_tokens"])}
        for key, _ in METRICS:
            values = [float(run[index][key]) for run in runs]
            row[key] = {"runs": values, "median": median(values),
                        "min": min(values), "max": max(values)}
        report["frontiers"].append(row)
    return report


def write_tables(reports, config):
    single = list(reports) == ["run"] and reports["run"]["repeats"] == 3
    processes = "Three fresh server processes" if single else "Fresh server processes per variant"
    full = all(row["generated_tokens"] == GENERATED for report in reports.values() for row in report["frontiers"])
    generated = f"{GENERATED} generated tokens" if full else f"at most {GENERATED} generated tokens"
    lines = [f"# {config['heading']}", "",
             "Tokens/second on one DGX Spark / GB10 at 300–2200 MHz.",
             f"{processes}; all samples retained. Each request has",
             f"zero reused tokens and {generated}. Frontiers come from the",
             "corpus byte calibration; the input column is the actual token count.", "",
             "[Conditions, graph and evidence](README.md).", ""]
    for name, report in reports.items():
        label = config["labels"].get(name)
        for key, heading in METRICS:
            runs = report["repeats"]
            header = " | ".join(f"Run {i + 1}" for i in range(runs))
            lines.extend([f"## {heading}{f' · {label}' if label else ''}", "",
                          f"| Frontier | Input tokens | {header} | Median | Min–max |",
                          "|---:|---:|" + "---:|" * (runs + 2)])
            for row in report["frontiers"]:
                value = row[key]
                samples = " | ".join(f"{sample:,.1f}" for sample in value["runs"])
                lines.append(f"| {row['frontier_tokens'] // 1024}K | {row['actual_prompt_tokens']:,} | "
                             f"{samples} | **{value['median']:,.1f}** | "
                             f"{value['min']:,.1f}–{value['max']:,.1f} |")
            summary = report["metrics"][key]
            count = "three run means" if runs == 3 else f"{runs} run mean{'s' if runs > 1 else ''}"
            lines.extend(["", f"Median of the {count}: **{summary['median_run_mean']:,.2f} tok/s**.", ""])
    (DATA / "tables.md").write_text("\n".join(lines))


def plot(reports, config):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(config["title"], fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, config["subtitle"], ha="center", color="#4b5563")
    x = [frontier / 1024 for frontier in FRONTIERS]
    for ax, (key, heading) in zip(axes, METRICS):
        for name, default_label, color in VARIANTS:
            report = reports.get(name)
            if not report:
                continue
            values = [row[key] for row in report["frontiers"]]
            center = report["metrics"][key]["median_run_mean"]
            label = config["labels"].get(name, default_label)
            if report["repeats"] > 1:
                ax.fill_between(x, [row["min"] for row in values], [row["max"] for row in values],
                                color=color, alpha=0.17, linewidth=0)
            ax.plot(x, [row["median"] for row in values], color=color, linewidth=2,
                    label=f"{label} · {center:,.1f} tok/s")
        ax.set_title(heading, loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", framealpha=0.92)
        ax.margins(y=0.2)
    axes[-1].set_xlabel("Context frontier (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.06, "\n".join(config["footer"]),
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA / "throughput.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    config = settings()
    reports = {}
    for name, _, _ in VARIANTS:
        runs = load_runs(name)
        if runs:
            reports[name] = summarize(runs)
    assert reports, f"no run-N.csv, base-N.csv or final-N.csv under {DATA}"
    # The single-variant record keeps its flat summary.
    summary = reports["run"] if list(reports) == ["run"] else reports
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_tables(reports, config)
    plot(reports, config)
    print(json.dumps({name: report["metrics"] for name, report in reports.items()}, indent=2))
