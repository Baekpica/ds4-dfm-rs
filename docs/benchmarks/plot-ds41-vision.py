#!/usr/bin/env python3
"""Tabulate the vision performance record: the ds41 vision surface, one boot.

Reads the three attach passes (vision-run-N.csv, the speed-bench CSV standard
with the vision extras) and the server's own per-request log lines
(tower-lines.txt: the tower seconds and the WxH patch -> token plan, in
request order), then writes tables.md, summary.json and vision-rows.png
beside them. Fails loud, never silently drops a row.
"""
import csv
import json
import math
import re
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).with_name("ds41-spark-vision-2026-10-14")
PASSES = 3
IMAGES = ["review-market-panel.png", "bug.jpeg", "homer.png"]
TEXT_ROWS = 2
GENERATED = 128
HOMER_TOKENS = 59     # homer.png ends early (finish=stop) in every pass
TOWER_LINE = re.compile(r"vision tower: (\d+) image\(s\) ([0-9.]+)s")
PLAN_LINE = re.compile(r"image \d+: (\d+x\d+) patch -> (\d+x\d+) token \(span (\d+)\)")
COLOR = "#087f8c"

COLUMNS = ["ctx_tokens", "prefill_tokens", "prefill_cached_tokens",
           "prefill_tps", "gen_tokens", "gen_tps", "first_token_sec",
           "tok_per_step", "wall_s", "clock_mhz", "kind", "image",
           "patch", "span", "tower_s"]


def read(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert list(rows[0].keys()) == COLUMNS, f"{path.name}: unexpected columns"
    return rows


def stats(values):
    return {"median": median(values), "min": min(values), "max": max(values),
            "samples": values}


# The attach passes cannot capture per-request log tails (the sweep reads
# them only in boot mode), so the tower time and the per-image plan come
# from the server's own log lines, in request order: pass 1's three images
# first, then pass 2's, then pass 3's.
def load_log():
    towers, plans = [], []
    for line in (DATA / "tower-lines.txt").read_text().splitlines():
        tower = TOWER_LINE.search(line)
        if tower:
            assert int(tower.group(1)) == 1, f"multi-image tower line: {line}"
            towers.append(float(tower.group(2)))
            continue
        plan = PLAN_LINE.search(line)
        if plan:
            plans.append({"patch": plan.group(1), "token_grid": plan.group(2),
                          "span": int(plan.group(3))})
    assert len(towers) == PASSES * len(IMAGES), f"{len(towers)} tower lines"
    assert len(plans) == PASSES * len(IMAGES), f"{len(plans)} plan lines"
    assert plans[0:3] == plans[3:6] == plans[6:9], "the image plans differ across passes"
    return towers, plans


def load_passes():
    passes = [read(DATA / f"vision-run-{i}.csv") for i in range(1, PASSES + 1)]
    for index, rows in enumerate(passes, start=1):
        assert len(rows) == len(IMAGES) + TEXT_ROWS, f"vision-run-{index}: rows"
        assert [row["kind"] for row in rows] == ["image"] * 3 + ["text"] * 2, \
            f"vision-run-{index}: row kinds"
        assert [row["image"] for row in rows[:3]] == IMAGES, \
            f"vision-run-{index}: fixture order"
        for row in rows:
            assert int(row["prefill_cached_tokens"]) == 0, \
                f"vision-run-{index}: reused prompt tokens"
            for key in ("prefill_tps", "gen_tps", "first_token_sec"):
                value = float(row[key])
                assert math.isfinite(value) and value > 0, f"vision-run-{index}: {key}"
        generations = [int(row["gen_tokens"]) for row in rows]
        assert generations == [GENERATED, GENERATED, HOMER_TOKENS] + [GENERATED] * 2, \
            f"vision-run-{index}: generated tokens {generations}"
    return passes


def vision_report(passes, towers, plans):
    rows = []
    for index in range(len(passes[0])):
        first = passes[0][index]
        entry = {"tag": first["image"] if first["kind"] == "image"
                 else f"text {int(first['ctx_tokens'])} tok",
                 "kind": first["kind"], "image": first["image"]}
        if first["kind"] == "image":
            position = IMAGES.index(first["image"])
            entry["patch"] = plans[position]["patch"]
            entry["token_grid"] = plans[position]["token_grid"]
            entry["span"] = plans[position]["span"]
            entry["tower_s"] = stats([towers[p * len(IMAGES) + position]
                                      for p in range(PASSES)])
        for key in ("ctx_tokens", "prefill_tps", "gen_tps", "first_token_sec"):
            entry[key] = stats([float(run[index][key]) for run in passes])
        rows.append(entry)
    return {"passes": PASSES, "rows": rows,
            "tower_lines": towers, "plan_lines": plans}


def write_tables(report):
    lines = [
        "# DeepSeek V4.1 Flash vision on the feat/optimize head",
        "",
        "One DGX Spark / GB10, one vision server process, three attach passes.",
        "Image rows send the repo fixtures with one fixed prompt; text rows",
        "bracket the image spans. Tower seconds and the per-image plan come",
        "from the server's own log lines (tower-lines.txt). Every sample is",
        "retained in the run CSVs; medians carry the min-max band.",
        "",
        "[Conditions and evidence](README.md).",
        "",
        "## Rows (median of three passes)",
        "",
        "| Row | Kind | Prompt tokens | Span | Patch -> token | Tower, s | TTFT, ms | Prefill, tok/s | Decode, tok/s |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        if row["kind"] == "image":
            tower = (f"{row['tower_s']['median']:.2f} "
                     f"({row['tower_s']['min']:.2f}-{row['tower_s']['max']:.2f})")
            span = row["span"]
            plan = f"{row['patch']} -> {row['token_grid']}"
        else:
            tower, span, plan = "-", "-", "-"
        lines.append(
            f"| {row['tag']} | {row['kind']} | {row['ctx_tokens']['median']:,.0f} | "
            f"{span} | {plan} | {tower} | "
            f"{row['first_token_sec']['median'] * 1000:,.0f} "
            f"({row['first_token_sec']['min'] * 1000:,.0f}-{row['first_token_sec']['max'] * 1000:,.0f}) | "
            f"{row['prefill_tps']['median']:,.1f} | {row['gen_tps']['median']:,.2f} |")
    lines += ["",
              "Image-row TTFT includes the tower seconds; the prefill rate is the",
              "server's own prefill timer over the whole prompt build, so it also",
              "includes the tower. homer.png ends early (finish=stop, 59 tokens)",
              "in every pass.",
              ""]
    (DATA / "tables.md").write_text("\n".join(lines) + "\n")


def plot(report):
    rows = report["rows"]
    tags = [row["tag"] for row in rows]
    towers = [row["tower_s"]["median"] * 1000 if row["kind"] == "image" else 0.0
              for row in rows]
    ttfts = [row["first_token_sec"]["median"] * 1000 for row in rows]
    residuals = [ttft - tower for ttft, tower in zip(ttfts, towers)]
    lo = [row["first_token_sec"]["min"] * 1000 for row in rows]
    hi = [row["first_token_sec"]["max"] * 1000 for row in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    xs = range(len(rows))
    left.bar(xs, towers, color="#b45309", label="Tower")
    left.bar(xs, residuals, bottom=towers, color=COLOR, label="Rest of TTFT")
    left.errorbar(xs, ttfts,
                  yerr=[[t - l for t, l in zip(ttfts, lo)],
                        [h - t for t, h in zip(ttfts, hi)]],
                  fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
    left.set_xticks(xs)
    left.set_xticklabels(tags, rotation=18, ha="right", fontsize=8)
    left.set_ylabel("First token, ms")
    left.set_title("TTFT (tower share highlighted)")
    left.legend(fontsize=8)

    width = 0.38
    prefill = [row["prefill_tps"]["median"] for row in rows]
    decode = [row["gen_tps"]["median"] for row in rows]
    right.bar([x - width / 2 for x in xs], prefill, width, color=COLOR, label="Prefill")
    right.bar([x + width / 2 for x in xs], decode, width, color="#b45309", label="Decode")
    right.set_xticks(xs)
    right.set_xticklabels(tags, rotation=18, ha="right", fontsize=8)
    right.set_ylabel("tok/s")
    right.set_title("Reported rates")
    right.legend(fontsize=8)

    fig.suptitle("DeepSeek V4.1 Flash vision on one DGX Spark")
    fig.tight_layout()
    fig.savefig(DATA / "vision-rows.png", dpi=180)


def main():
    towers, plans = load_log()
    passes = load_passes()
    report = vision_report(passes, towers, plans)
    write_tables(report)
    (DATA / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report)
    print("tables.md, summary.json, vision-rows.png written")


if __name__ == "__main__":
    main()
