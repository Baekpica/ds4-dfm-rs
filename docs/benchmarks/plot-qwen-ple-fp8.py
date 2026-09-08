#!/usr/bin/env python3
"""Plot the recorded card sweeps; requires matplotlib (see receipt.json)."""
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).with_name("qwen-ple-fp8-2026-09-08")
COLORS = {"bf16": "#687789", "fp8": "#087f8c"}
LABELS = {"bf16": "BF16 PLE", "fp8": "FP8 PLE"}
METRICS = (("prefill_tps", "Incremental prefill"), ("gen_tps", "Decode (automatic MTP policy)"))


def load_runs(variant, storage):
    runs = []
    for repeat in range(1, 4):
        with (DATA / f"{variant}-{storage}-{repeat}.csv").open() as source:
            rows = list(csv.DictReader(source))
        assert [int(row["ctx_tokens"]) for row in rows] == list(range(2048, 65537, 2048))
        assert all(int(row["prefill_tokens"]) == 2048 and int(row["gen_tokens"]) == 128
                   for row in rows)
        runs.append(rows)
    return runs


def plot_variant(variant):
    runs = {storage: load_runs(variant, storage) for storage in LABELS}
    events = json.loads((DATA / "run-events.json").read_text())
    quenched = {storage: sum(bool(events[f"{variant}-{storage}-{repeat}"]["mtp_quench"])
                            for repeat in range(1, 4)) for storage in LABELS}
    summary = {}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    title = "Qwen3.8 Flash Next" + (" Uncensored" if variant == "uncensored" else "")
    fig.suptitle(f"{title} · Q5 + SSD-PLE", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, "One DGX Spark / GB10 · 2K–64K card protocol · 2026-09-08",
             ha="center", color="#4b5563")
    x = list(range(2, 65, 2))
    for ax, (metric, heading) in zip(axes, METRICS):
        summary[metric] = {}
        for storage in LABELS:
            samples = [[float(row[metric]) for row in run] for run in runs[storage]]
            columns = list(zip(*samples))
            run_means = [mean(sample) for sample in samples]
            center = median(run_means)
            summary[metric][storage] = {"run_means": run_means, "median_run_mean": center}
            ax.fill_between(x, [min(c) for c in columns], [max(c) for c in columns],
                            color=COLORS[storage], alpha=0.17, linewidth=0)
            ax.plot(x, [median(c) for c in columns], color=COLORS[storage], linewidth=2,
                    label=f"{LABELS[storage]} · {center:,.1f} tok/s")
        change = 100 * (summary[metric]["fp8"]["median_run_mean"] /
                        summary[metric]["bf16"]["median_run_mean"] - 1)
        summary[metric]["change_percent"] = change
        ax.set_title(f"{heading} · FP8 {change:+.1f}%", loc="left", fontsize=13, pad=10)
        ax.set_ylabel("Tokens / second")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", framealpha=0.92)
        ax.margins(y=0.2)
    axes[-1].set_xlabel("Context tokens (K = 1,024)")
    axes[-1].set_xticks([2, 8, 16, 24, 32, 40, 48, 56, 64])
    axes[-1].set_xlim(2, 64)
    fig.text(0.5, 0.06, "Curves: per-frontier medians; bands: observed min–max of 3 fresh runs.\n"
             "Legend: median of run means. 2,048-token incremental prefill + 128 greedy tokens per frontier.\n"
             "Same binary/main GGUF · 2 GiB PLE cache · 16 workers · MTP draft 2 · aligned-Q8 owner\n"
             f"MTP autoquench: BF16 {quenched['bf16']}/3 runs, FP8 {quenched['fp8']}/3 runs; all runs retained.",
             ha="center", va="center", fontsize=9, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.17, left=0.1, right=0.97, hspace=0.3)
    fig.savefig(DATA.parent.parent / f"qwen38-ple-fp8-{variant}.png", dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    report = {variant: plot_variant(variant) for variant in ("base", "uncensored")}
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
