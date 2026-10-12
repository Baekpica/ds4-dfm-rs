#!/usr/bin/env python3
"""Engine-vs-fork throughput chart for the ds41 campaign on the GB10 Spark.

Default mode (no --engine-csv): the fork series is the frontier sweep CSV and
the engine appears as its MEASURED CELLS only -- the engine's curve has never
been swept, so drawing a line for it would invent data.

When the engine sweep lands (see docs/ds41-speed-bench-2026-10-12.md,
"Open: the engine curve"), pass --engine-csv with a CSV in the same column
shape and both series become lines.

Usage:
  mk_engine_vs_fork_chart.py speed-bench/gb10_spark_ds41_1e7d8f24.csv \
      -o speed-bench/ds41_engine_vs_fork_compare
  mk_engine_vs_fork_chart.py speed-bench/gb10_spark_ds41_1e7d8f24.csv \
      --engine-csv speed-bench/gb10_spark_ds41_engine.csv -o .../compare_full

Engine cells are the only engine measurements on record; every entry names
its source.  Both come from the perf harness (one boot per cell,
/dev/../ds41_perf_sweep.sh); the fork line comes from the speed-bench
harness; on shared cells the two harnesses agree within ~1%.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENGINE_CELLS = [
    # (ctx_tokens, kind, tok_per_s, source)
    (8192, "prefill", 1042.0, "U-A A/B 2026-10-12 (perf harness)"),
    (2048, "decode", 43.78, "r5 same-window A/B 2026-10-12"),
]


def load(path):
    ctx, pref, dec = [], [], []
    for r in csv.DictReader(open(path)):
        ctx.append(int(r["ctx_tokens"]))
        pref.append(float(r["prefill_tps"]))
        dec.append(float(r["gen_tps"]))
    return ctx, pref, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fork_csv", help="the fork build's frontier sweep CSV")
    ap.add_argument("-o", "--output", required=True, help="output base path (writes .png and .svg)")
    ap.add_argument("--engine-csv", help="an engine sweep CSV in the same column shape")
    args = ap.parse_args()

    ctx, pref, dec = load(args.fork_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=110)
    fig.suptitle("ds41 on GB10 Spark -- engine vs fork (fork = r6 build 1e7d8f24, 2026-10-12)",
                 fontsize=12, fontweight="bold")

    ax1.plot(ctx, pref, "-o", ms=3.5, lw=1.8, color="#2563eb",
             label="fork (r6 build) prefill t/s")
    ax2.plot(ctx, dec, "-o", ms=3.5, lw=1.8, color="#dc2626",
             label="fork (r6 build) decode t/s")

    eng_pref = [(c, v) for c, k, v, _ in ENGINE_CELLS if k == "prefill"]
    eng_dec = [(c, v) for c, k, v, _ in ENGINE_CELLS if k == "decode"]

    if args.engine_csv:
        ec, ep, ed = load(args.engine_csv)
        ax1.plot(ec, ep, "--s", ms=3.5, lw=1.8, color="#111827", label="engine prefill t/s")
        ax2.plot(ec, ed, "--s", ms=3.5, lw=1.8, color="#111827", label="engine decode t/s")
        footnote = ("Both series are full frontier sweeps on the same corpus and harness recipe.  "
                    "Decode dips are MTP acceptance variance.")
    else:
        ax1.plot([c for c, _ in eng_pref], [v for _, v in eng_pref], "D", ms=11,
                 color="#111827", label="engine prefill (measured cell only)")
        ax2.plot([c for c, _ in eng_dec], [v for _, v in eng_dec], "D", ms=11,
                 color="#111827", label="engine decode (measured cell only)")
        fp = pref[ctx.index(8192)]
        fd = dec[ctx.index(2048)]
        ax1.annotate(f"engine {1042.0:.0f} t/s\nfork {fp:.0f} t/s at the same frontier",
                     xy=(8192, 1042.0), xytext=(2600, 1075),
                     arrowprops=dict(arrowstyle="->", color="#111827"), fontsize=9)
        ax2.annotate(f"engine {43.78:.2f} t/s\nfork {fd:.1f} t/s at the same cell "
                     f"({(fd / 43.78 - 1) * 100:+.1f}%)",
                     xy=(2048, 43.78), xytext=(7000, 38.5),
                     arrowprops=dict(arrowstyle="->", color="#111827"), fontsize=9)
        footnote = ("Engine curve not swept: only the cells recorded at 2048 (decode, r5 window) and "
                    "8192 (prefill, U-A A/B) exist.  The fork line is the full 22-frontier sweep; "
                    "decode dips are MTP acceptance variance.")

    ax1.set_xlabel("context tokens")
    ax1.set_ylabel("prefill t/s")
    ax1.set_title("Prefill")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="lower right")

    ax2.set_xlabel("context tokens")
    ax2.set_ylabel("decode t/s")
    ax2.set_title("Decode (greedy generation)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="lower left")

    fig.text(0.5, 0.005, footnote, ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(args.output + ".png")
    fig.savefig(args.output + ".svg")
    print(f"wrote {args.output}.png and .svg")


if __name__ == "__main__":
    main()
