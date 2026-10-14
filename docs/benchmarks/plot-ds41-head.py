#!/usr/bin/env python3
"""Tabulate the feat/optimize head campaign: the five-case suite and the
2K-64K frontier sweep. The vision surface has its own record and assembler
(plot-ds41-vision.py, ds41-spark-vision-2026-10-14).

Reads the three fresh-process suite passes (suite-run-N.csv) and the frontier
attach passes present on disk (frontier-run-N.csv, one to three), checks the
invariants the harnesses assert, and writes tables.md and summary.json beside
them. Fails loud, never silently drops a row.
"""
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).with_name("ds41-spark-head-2026-10-14")
WARMPATH = Path(__file__).with_name("ds41-warmpath-2026-10-13")
PASSES = 3
CASES = ["prose_short", "code_short", "table_short", "prose_long", "table_long"]
FRONTIERS = list(range(2048, 65537, 2048))
GENERATED = 128
COLOR = "#087f8c"

SUITE_COLUMNS = ["engine", "case", "rep", "cache", "prompt_tokens",
                 "completion_tokens", "ttft_s", "total_s", "sse_deltas",
                 "prefill_tps", "decode_tps", "e2e_tps", "finish", "prompt_sha"]
FRONTIER_COLUMNS = ["ctx_tokens", "prefill_tokens", "prefill_cached_tokens",
                    "prefill_tps", "gen_tokens", "gen_tps", "first_token_sec",
                    "tok_per_step", "wall_s", "clock_mhz"]


def read(path, columns):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert list(rows[0].keys()) == columns, f"{path.name}: unexpected columns"
    return rows


def stats(values):
    return {"median": median(values), "min": min(values), "max": max(values),
            "samples": values}


# ------------------------------------------------------------ the five cases

def load_suite():
    case_shas = {case["name"]: case["prompt_sha256"]
                 for case in json.load((WARMPATH / "cases.json").open())}
    passes = [read(DATA / f"suite-run-{i}.csv", SUITE_COLUMNS)
              for i in range(1, PASSES + 1)]
    for index, rows in enumerate(passes, start=1):
        assert len(rows) == len(CASES) * 3, f"suite-run-{index}: {len(rows)} rows, expected 15"
        for position, case in enumerate(CASES):
            window = rows[position * 3:position * 3 + 3]
            assert [row["case"] for row in window] == [case] * 3, f"suite-run-{index}: case order"
            assert [row["rep"] for row in window] == ["0", "1", "2"], f"suite-run-{index}: {case} reps"
            assert [row["cache"] for row in window] == ["cold", "warm", "warm"], \
                f"suite-run-{index}: {case} cache tags"
            assert len({row["prompt_sha"] for row in window}) == 1, \
                f"suite-run-{index}: {case} bytes differ inside a pass"
            assert window[0]["prompt_sha"] == case_shas[case], \
                f"suite-run-{index}: {case} bytes are not the archived ones"
            assert len({row["prompt_tokens"] for row in window}) == 1, \
                f"suite-run-{index}: {case} prompt length varies"
            assert all(row["finish"] == "length" for row in window), \
                f"suite-run-{index}: {case} finish reason"
        for row in rows:
            for key in ("ttft_s", "total_s", "decode_tps", "e2e_tps"):
                value = float(row[key])
                assert math.isfinite(value) and value > 0, \
                    f"suite-run-{index}: {row['case']} rep {row['rep']} {key}"
    for case in CASES:
        tokens = {rows[CASES.index(case) * 3]["prompt_tokens"] for rows in passes}
        completion = {rows[CASES.index(case) * 3]["completion_tokens"] for rows in passes}
        assert len(tokens) == 1 and len(completion) == 1, \
            f"{case}: the passes measured different work"
    return passes


def suite_report(passes):
    cases = []
    for position, case in enumerate(CASES):
        windows = [rows[position * 3:position * 3 + 3] for rows in passes]
        cold = [window[0] for window in windows]
        warm = [row for window in windows for row in window if row["cache"] == "warm"]
        cases.append({
            "case": case,
            "prompt_tokens": int(cold[0]["prompt_tokens"]),
            "completion_tokens": int(cold[0]["completion_tokens"]),
            "prompt_sha": cold[0]["prompt_sha"],
            "cold_ttft_s": stats([float(row["ttft_s"]) for row in cold]),
            "cold_prefill_tps": stats([float(row["prefill_tps"]) for row in cold]),
            "warm_ttft_ms": stats([float(row["ttft_s"]) * 1000 for row in warm]),
            "decode_tps": stats([float(row["decode_tps"]) for window in windows
                                 for row in window]),
            "e2e_tps": stats([float(row["e2e_tps"]) for window in windows
                              for row in window]),
        })
    return {"passes": PASSES, "cases": cases}


# ---------------------------------------------------------- the frontier set

def load_frontier():
    paths = sorted(DATA.glob("frontier-run-*.csv"))
    assert 1 <= len(paths) <= PASSES, f"{len(paths)} frontier passes on disk"
    runs = [read(path, FRONTIER_COLUMNS) for path in paths]
    for index, rows in enumerate(runs, start=1):
        assert [int(row["ctx_tokens"]) for row in rows] == FRONTIERS, \
            f"frontier-{index}: frontiers"
        assert all(int(row["gen_tokens"]) == GENERATED for row in rows), \
            f"frontier-{index}: generated tokens"
        assert all(int(row["prefill_cached_tokens"]) == 0 for row in rows), \
            f"frontier-{index}: reused prompt tokens"
        for key in ("prefill_tps", "gen_tps"):
            assert all(math.isfinite(float(row[key])) and float(row[key]) > 0
                       for row in rows), f"frontier-{index}: {key}"
    inputs = [[int(row["prefill_tokens"]) for row in run] for run in runs]
    assert all(row == inputs[0] for row in inputs), "frontier: input counts differ across runs"
    return runs


def frontier_report(runs):
    report = {"passes": len(runs), "frontiers": [], "metrics": {}}
    for key in ("prefill_tps", "gen_tps"):
        run_means = [mean(float(row[key]) for row in run) for run in runs]
        report["metrics"][key] = {"run_means": run_means,
                                  "median_run_mean": median(run_means)}
    for index, frontier in enumerate(FRONTIERS):
        row = {"frontier_tokens": frontier,
               "actual_prompt_tokens": int(runs[0][index]["prefill_tokens"]),
               "generated_tokens": GENERATED}
        for key in ("prefill_tps", "gen_tps"):
            values = [float(run[index][key]) for run in runs]
            row[key] = {"runs": values, "median": median(values),
                        "min": min(values), "max": max(values)}
        report["frontiers"].append(row)
    return report


# --------------------------------------------------------------- the outputs

def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def write_tables(suite, frontier):
    lines = [
        "# DeepSeek V4.1 Flash on the feat/optimize head",
        "",
        "One DGX Spark / GB10. The author's five-case suite ran three fresh",
        "server processes; the 2K-64K frontier sweep ran three attach passes",
        "against one unguarded boot (the operator's one-engine protocol; see",
        "README). Every sample is retained; medians carry the min-max band.",
        "",
        "[Conditions and evidence](README.md).",
        "",
        "## Five-case suite",
        "",
        "| Case | Prompt tokens | Decode, tok/s | Cold prefill, tok/s | Warm TTFT, ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in suite["cases"]:
        lines.append(
            f"| {row['case']} | {row['prompt_tokens']:,} | "
            f"{row['decode_tps']['median']:.2f} "
            f"({row['decode_tps']['min']:.2f}-{row['decode_tps']['max']:.2f}) | "
            f"{fmt(row['cold_prefill_tps']['median'])} | "
            f"{fmt(row['warm_ttft_ms']['median'])} "
            f"({fmt(row['warm_ttft_ms']['min'])}-{fmt(row['warm_ttft_ms']['max'])}) |")
    passes = frontier["passes"]
    runs_header = " | ".join(f"Run {i}" for i in range(1, passes + 1))
    rules = "|".join(["---:"] * (2 + passes + 2))
    lines += ["",
              "Cold prefill is the first request of each fresh process (three samples",
              "per case); warm TTFT pools the six repeats. Decode pools all nine",
              "samples. The rates are client-side, prompt tokens over TTFT.",
              "",
              "## 2K-64K frontier sweep",
              "",
              "Full-prompt prefill; every request reports zero reused tokens and 128",
              f"generated tokens. {passes} complete pass(es) of the 32 frontiers;",
              "frontiers come from the corpus byte calibration, and the input column",
              "is the actual token count.",
              ""]
    for key, heading in (("prefill_tps", "Full prompt prefill"),
                         ("gen_tps", "Decode (automatic embedded MTP)")):
        lines += [f"### {heading}", "",
                  f"| Frontier | Input tokens | {runs_header} | Median | Min-max |",
                  f"|{rules}|"]
        for row in frontier["frontiers"]:
            value = row[key]
            samples = " | ".join(f"{sample:,.1f}" for sample in value["runs"])
            lines.append(
                f"| {row['frontier_tokens'] // 1024}K | {row['actual_prompt_tokens']:,} | "
                f"{samples} | {value['median']:,.1f} | "
                f"{value['min']:,.1f}-{value['max']:,.1f} |")
        summary = frontier["metrics"][key]
        lines += ["",
                  f"Median of the three run means: {summary['median_run_mean']:,.2f} tok/s.",
                  ""]
    (DATA / "tables.md").write_text("\n".join(lines) + "\n")


def plot(frontier):
    fig, left = plt.subplots(figsize=(9.5, 5.2))
    right = left.twinx()
    xs = [row["frontier_tokens"] / 1024 for row in frontier["frontiers"]]
    prefill = [row["prefill_tps"]["median"] for row in frontier["frontiers"]]
    prefill_lo = [row["prefill_tps"]["min"] for row in frontier["frontiers"]]
    prefill_hi = [row["prefill_tps"]["max"] for row in frontier["frontiers"]]
    decode = [row["gen_tps"]["median"] for row in frontier["frontiers"]]
    left.plot(xs, prefill, color=COLOR, linewidth=2, label="Prefill")
    left.fill_between(xs, prefill_lo, prefill_hi, color=COLOR, alpha=0.18)
    right.plot(xs, decode, color="#b45309", linewidth=2, label="Decode")
    left.set_xlabel("Prompt tokens (K)")
    left.set_ylabel("Prefill tok/s", color=COLOR)
    right.set_ylabel("Decode tok/s", color="#b45309")
    left.set_ylim(bottom=0)
    right.set_ylim(bottom=0)
    left.set_title("DeepSeek V4.1 Flash on one DGX Spark: prefill and decode")
    fig.tight_layout()
    fig.savefig(DATA / "throughput.png", dpi=180)


def main():
    suite = suite_report(load_suite())
    frontier = frontier_report(load_frontier())
    write_tables(suite, frontier)
    (DATA / "summary.json").write_text(json.dumps(
        {"suite": suite, "frontier": frontier}, indent=2) + "\n")
    plot(frontier)
    print("tables.md, summary.json, throughput.png written")


if __name__ == "__main__":
    main()
