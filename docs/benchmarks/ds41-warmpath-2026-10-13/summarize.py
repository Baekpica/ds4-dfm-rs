#!/usr/bin/env python3
"""Turn the raw suite output into a per-engine/per-case comparison CSV plus a chart."""
import csv, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open("results-raw.csv")))
by = defaultdict(list)
for r in rows:
    by[(r["engine"], r["case"])].append(r)

cases = ["prose_short", "code_short", "table_short", "prose_long", "table_long"]
engines = ["ds4-dfm-rs-before", "ds4-dfm-rs-after"]  # campaign labels (was youngai, ds4-dfm-rs)

out = open("bench-compare.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["case", "engine", "decode_tps_median", "decode_tps_min", "decode_tps_max",
            "prefill_tps_cold", "ttft_cold_s", "ttft_warm_ms", "e2e_tps_median", "prompt_tokens", "completion_tokens"])
summary = {}
for case in cases:
    for eng in engines:
        rs = by.get((eng, case), [])
        if not rs:
            continue
        dec = [float(r["decode_tps"]) for r in rs if r["decode_tps"]]
        e2e = [float(r["e2e_tps"]) for r in rs if r["e2e_tps"]]
        cold = next((r for r in rs if r["cache"] == "cold"), rs[0])
        warm = [float(r["ttft_s"]) for r in rs if r["cache"] == "warm"]
        m = st.median(dec) if dec else 0.0
        row = [case, eng, f"{m:.2f}", f"{min(dec):.2f}" if dec else "", f"{max(dec):.2f}" if dec else "",
               f"{float(cold['prefill_tps']):.1f}", f"{float(cold['ttft_s']):.3f}",
               f"{st.median(warm)*1000:.1f}" if warm else "", f"{st.median(e2e):.2f}" if e2e else "",
               cold["prompt_tokens"], cold["completion_tokens"]]
        w.writerow(row)
        summary[(eng, case)] = float(m)
out.close()

print("case                 youngai   ds4-dfm-rs   delta%")
for case in cases:
    y = summary.get(("youngai", case)); d = summary.get(("ds4-dfm-rs", case))
    if y and d:
        print(f"{case:20s} {y:8.2f} {d:11.2f}   {(y/d-1)*100:+6.2f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(cases)); wdt = 0.38
colors = {"ds4-dfm-rs-before": "#d62728", "ds4-dfm-rs-after": "#2ca02c"}

ax = axes[0]
for i, eng in enumerate(engines):
    vals = [summary.get((eng, c), 0) for c in cases]
    ax.bar(x + (i - 0.5) * wdt, vals, wdt, label=eng, color=colors[eng])
    for xi, v in zip(x + (i - 0.5) * wdt, vals):
        ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("decode tokens/s"); ax.set_title("Decode throughput (median of 3)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

lg = [c for c in cases if c.endswith("long")]
xl = np.arange(len(lg))
ax = axes[1]
for i, eng in enumerate(engines):
    vals = [float(next(r for r in by[(eng, c)] if r["cache"] == "cold")["prefill_tps"]) for c in lg]
    ax.bar(xl + (i - 0.5) * wdt, vals, wdt, label=eng, color=colors[eng])
    for xi, v in zip(xl + (i - 0.5) * wdt, vals):
        ax.text(xi, v + 10, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(xl); ax.set_xticklabels(lg); ax.set_ylabel("prefill tokens/s")
ax.set_title("Cold prefill (~10.4k tokens)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

ax = axes[2]
for i, eng in enumerate(engines):
    vals = [st.median([float(r["ttft_s"]) for r in by[(eng, c)] if r["cache"] == "warm"]) * 1000 for c in cases]
    ax.bar(x + (i - 0.5) * wdt, vals, wdt, label=eng, color=colors[eng])
    for xi, v in zip(x + (i - 0.5) * wdt, vals):
        ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("ms to first token"); ax.set_title("Warm-prefix TTFT (KV reuse)"); ax.legend(); ax.grid(axis="y", alpha=0.3)

fig.suptitle("DeepSeek V4.1 warm path: asset-open cache off vs on, DGX Spark GB10 - same GGUF, same coding sidecar, temp 0, single stream", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("bench-compare.png", dpi=130)
print("wrote bench-compare.csv and bench-compare.png")
