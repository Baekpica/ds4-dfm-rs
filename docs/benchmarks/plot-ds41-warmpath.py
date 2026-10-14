#!/usr/bin/env python3
"""Tabulate the warm-path campaign and check the invariants the author's suite asserts.

Reads the three on-box passes of the author's suite (run-1..3.csv, the fifteen
canonical columns) plus the archived pre-change pass (archived-before.csv, the
ds4-dfm-rs arm of the run the author published under /data/YoungAi/bench/), and
writes summary.json and tables.md beside them. The comparison chart and
bench-compare.csv stay the author's own summarize.py output - this script does
not draw, it tabulates and validates.

Invariants checked (fail loud, never silently drop a row):
  * three passes, fifteen rows each, same case order, same prompt_sha per case
    as cases.json (byte identity of the measured prompts across passes);
  * prompt lengths and completion counts identical across the three passes;
  * rep 0 tagged cold and reps 1..2 tagged warm;
  * every reported rate finite and positive.
"""
import csv
import json
from pathlib import Path
from statistics import median

DATA = Path(__file__).with_name("ds41-warmpath-2026-10-13")
PASSES = 3
CASES = ["prose_short", "code_short", "table_short", "prose_long", "table_long"]
COLUMNS = ["engine", "case", "rep", "cache", "prompt_tokens", "completion_tokens",
           "ttft_s", "total_s", "sse_deltas", "prefill_tps", "decode_tps", "e2e_tps",
           "finish", "prompt_sha"]


def read(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert list(rows[0].keys()) == COLUMNS, f"{path.name}: unexpected columns"
    return rows


def load_case_shas():
    return {case["name"]: case["prompt_sha256"] for case in json.load((DATA / "cases.json").open())}


def check(passes, case_shas):
    for index, rows in enumerate(passes, start=1):
        assert len(rows) == len(CASES) * 3, f"run-{index}: {len(rows)} rows, expected 15"
        for position, case in enumerate(CASES):
            window = rows[position * 3:position * 3 + 3]
            assert [row["case"] for row in window] == [case] * 3, f"run-{index}: case order"
            assert [row["rep"] for row in window] == ["0", "1", "2"], f"run-{index}: {case} reps"
            assert [row["cache"] for row in window] == ["cold", "warm", "warm"], f"run-{index}: {case} cache tags"
            assert len({row["prompt_sha"] for row in window}) == 1, f"run-{index}: {case} bytes differ inside a pass"
            assert window[0]["prompt_sha"] == case_shas[case], f"run-{index}: {case} bytes are not the archived ones"
            assert len({row["prompt_tokens"] for row in window}) == 1, f"run-{index}: {case} prompt length varies"
        for row in rows:
            for key in ("ttft_s", "total_s", "e2e_tps"):
                assert float(row[key]) > 0, f"run-{index}: {row['case']} rep {row['rep']} {key}"
    # the same measured regime in every pass
    for case in CASES:
        tokens = {rows[CASES.index(case) * 3]["prompt_tokens"] for rows in passes}
        completion = {rows[CASES.index(case) * 3]["completion_tokens"] for rows in passes}
        assert len(tokens) == 1 and len(completion) == 1, f"{case}: the passes measured different work"


def stats(values):
    return {"median": median(values), "min": min(values), "max": max(values), "samples": values}


def arm(rows):
    """Per-case cold and warm statistics for one pass of the suite."""
    out = {}
    for case in CASES:
        window = [row for row in rows if row["case"] == case]
        cold = next(row for row in window if row["cache"] == "cold")
        warm = [row for row in window if row["cache"] == "warm"]
        out[case] = {
            "prompt_tokens": int(cold["prompt_tokens"]),
            "completion_tokens": int(cold["completion_tokens"]),
            "prompt_sha": cold["prompt_sha"],
            "cold_ttft_s": float(cold["ttft_s"]),
            "cold_prefill_tps": float(cold["prefill_tps"]),
            "warm_ttft_ms": stats([float(row["ttft_s"]) * 1000 for row in warm]),
            "warm_over_cold": stats([float(row["ttft_s"]) / float(cold["ttft_s"]) for row in warm]),
            "decode_tps": stats([float(row["decode_tps"]) for row in window]),
            "e2e_tps": stats([float(row["e2e_tps"]) for row in window]),
        }
    return out


def merge(arms_per_pass):
    """Fold the passes into one report: cold is per pass, warm pools the passes."""
    report = {"passes": PASSES, "cases": [], "arms": {}}
    for label, arms in arms_per_pass.items():
        cases = []
        for case in CASES:
            first = arms[0][case]
            warm = [value for arm_rows in arms for value in arm_rows[case]["warm_ttft_ms"]["samples"]]
            cold = [arm_rows[case]["cold_ttft_s"] for arm_rows in arms]
            cases.append({
                "case": case,
                "prompt_tokens": first["prompt_tokens"],
                "completion_tokens": first["completion_tokens"],
                "prompt_sha": first["prompt_sha"],
                "cold_ttft_s": stats(cold),
                "cold_prefill_tps": stats([arm_rows[case]["cold_prefill_tps"] for arm_rows in arms]),
                "warm_ttft_ms": stats(warm),
                "decode_tps": stats([v for arm_rows in arms for v in arm_rows[case]["decode_tps"]["samples"]]),
                "e2e_tps": stats([v for arm_rows in arms for v in arm_rows[case]["e2e_tps"]["samples"]]),
                "warm_over_cold": stats([value for arm_rows in arms for value in arm_rows[case]["warm_over_cold"]["samples"]]),
            })
        report["arms"][label] = cases
    report["cases"] = CASES
    return report


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def write_tables(report):
    after = {row["case"]: row for row in report["arms"]["ds4-dfm-rs-after"]}
    before = {row["case"]: row for row in report["arms"]["ds4-dfm-rs-before"]}
    lines = [
        "# DeepSeek V4.1 warm-path campaign: the asset-open cache",
        "",
        "One DGX Spark / GB10, the author's five-case HTTP suite, three successive",
        "passes against one server process (this campaign) against the archived",
        "pre-change pass. Every sample is retained; medians carry the min-max band.",
        "",
        "[Conditions and evidence](README.md).",
        "",
        "## Warm-prefix TTFT",
        "",
        "Later requests of a session that reuse the prompt's KV checkpoint.",
        "",
        "| Case | Prompt tokens | Before, ms | After, ms | Before/after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in CASES:
        b, a = before[case], after[case]
        lines.append(
            f"| {case} | {a['prompt_tokens']:,} | {fmt(b['warm_ttft_ms']['median'])} "
            f"({fmt(b['warm_ttft_ms']['min'])}-{fmt(b['warm_ttft_ms']['max'])}) | "
            f"{fmt(a['warm_ttft_ms']['median'])} ({fmt(a['warm_ttft_ms']['min'])}-{fmt(a['warm_ttft_ms']['max'])}) | "
            f"{b['warm_ttft_ms']['median'] / a['warm_ttft_ms']['median']:.1f}x |")
    lines += [
        "",
        "## First request of a case (first-touch TTFT)",
        "",
        "The first request that sends those bytes carries the prefill, so this is the",
        "cold cell, one sample per pass.",
        "",
        "| Case | Prompt tokens | Before, s | After, s | After cold prefill, tok/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in CASES:
        b, a = before[case], after[case]
        lines.append(
            f"| {case} | {a['prompt_tokens']:,} | {b['cold_ttft_s']['median']:.3f} "
            f"({b['cold_ttft_s']['min']:.3f}-{b['cold_ttft_s']['max']:.3f}) | "
            f"{a['cold_ttft_s']['median']:.3f} ({a['cold_ttft_s']['min']:.3f}-{a['cold_ttft_s']['max']:.3f}) | "
            f"{fmt(a['cold_prefill_tps']['median'])} |")
    lines += [
        "",
        "## Decode throughput (automatic embedded MTP)",
        "",
        "| Case | Before, tok/s | After, tok/s |",
        "| --- | ---: | ---: |",
    ]
    for case in CASES:
        b, a = before[case], after[case]
        lines.append(
            f"| {case} | {b['decode_tps']['median']:.2f} "
            f"({b['decode_tps']['min']:.2f}-{b['decode_tps']['max']:.2f}) | "
            f"{a['decode_tps']['median']:.2f} "
            f"({a['decode_tps']['min']:.2f}-{a['decode_tps']['max']:.2f}) |")
    (DATA / "tables.md").write_text("\n".join(lines) + "\n")


def write_summary(report):
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def write_union():
    """Emit results-raw.csv: the union the author's summarize.py compares.

    Row content is copied verbatim from archived-before.csv and run-1..3.csv;
    only the engine label changes, so the two arms can sit in one file. Both
    sources stay beside it, so the relabel is reversible.
    """
    rows = []
    for name, label in (("archived-before.csv", "ds4-dfm-rs-before"),):
        rows += [{**row, "engine": label} for row in read(DATA / name)]
    for index in range(1, PASSES + 1):
        rows += [{**row, "engine": "ds4-dfm-rs-after"} for row in read(DATA / f"run-{index}.csv")]
    with (DATA / "results-raw.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    case_shas = load_case_shas()
    passes = [read(DATA / f"run-{index}.csv") for index in range(1, PASSES + 1)]
    check(passes, case_shas)
    # The archived arm is one pass, so its warm band is the two warm rows of that pass.
    report = merge({"ds4-dfm-rs-before": [arm(read(DATA / "archived-before.csv"))],
                    "ds4-dfm-rs-after": [arm(rows) for rows in passes]})
    write_summary(report)
    write_tables(report)
    print(f"results-raw.csv: {write_union()} rows")
    for case in CASES:
        b = report["arms"]["ds4-dfm-rs-before"][CASES.index(case)]
        a = report["arms"]["ds4-dfm-rs-after"][CASES.index(case)]
        print(f"{case:12s} warm {b['warm_ttft_ms']['median']:7.1f} -> "
              f"{a['warm_ttft_ms']['median']:6.1f} ms   cold {b['cold_ttft_s']['median']:.3f} -> "
              f"{a['cold_ttft_s']['median']:.3f} s   decode {b['decode_tps']['median']:.2f} -> "
              f"{a['decode_tps']['median']:.2f} t/s")
