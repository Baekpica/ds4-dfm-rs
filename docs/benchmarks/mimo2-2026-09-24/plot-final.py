#!/usr/bin/env python3
"""Rebuild the published curve from recorded CSVs and historical medians."""
import importlib.util
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
DATA = HERE / "curve"
ASSET = "mimo2-prefill-2k-64k-20260924"
spec = importlib.util.spec_from_file_location("mimo_plot", HERE.parent / "plot-mimo2.py")
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)

historical = json.loads((DATA / "historical-curve.json").read_text())
receipt = json.loads((DATA / "receipt.json").read_text())
base_runs, base_sources = plotter.load_runs(HERE.parent / "mimo2-2026-09-23", "baseline")
final_runs, final_sources = plotter.load_runs(DATA, "final")
old_metrics = {}
for metric, _ in plotter.METRICS:
    old = historical["metrics"][metric]["cur"]
    center = [old["frontier_median"][str(n)] for n in plotter.FRONTIERS]
    old_metrics[metric] = {"median": center, "retention_64k_over_2k": center[-1] / center[0]}

series = [
    {"label": "Historical PR #56", "sources": base_sources,
     "metrics": plotter.summarize(base_runs)},
    {"label": "Historical f09c1862", "sources": historical["sources"]["cur"],
     "metrics": old_metrics},
    {"label": receipt["source_commit"][:8] + " defaults", "sources": final_sources,
     "metrics": plotter.summarize(final_runs)},
]
note = "MTP/DFlash off · 300–2200 MHz · Historical comparisons are not paired A/B."
plotter.plot(series, DATA / (ASSET + ".png"), note)
summary = {}
for metric, _ in plotter.METRICS:
    new = series[-1]["metrics"][metric]
    old = old_metrics[metric]
    summary[metric] = {
        "at_2k": new["median"][0], "at_64k": new["median"][-1],
        "run_means": [mean(float(row[metric]) for row in run) for run in final_runs],
        "median_run_mean": median(mean(float(row[metric]) for row in run) for run in final_runs),
        "historical_f09c_gain_pct": [100 * (a / b - 1) for a, b in zip(new["median"], old["median"])],
    }
(DATA / (ASSET + ".json")).write_text(json.dumps({
    "frontiers": plotter.FRONTIERS, "note": note, "receipt": receipt,
    "series": series, "summary": summary,
    "historical_band_note": "f09c1862 contains medians only; no band is invented.",
}, indent=2) + "\n")
print(json.dumps(summary, indent=2))
