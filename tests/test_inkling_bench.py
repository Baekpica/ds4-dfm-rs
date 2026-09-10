#!/usr/bin/env python3
"""Compare Inkling benchmark prefix replay with independent cold frontiers.

Run under host_memory_guard with the intended weight owner already resident.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


def run(args, name, start, end):
    out = args.out / name
    out.mkdir()
    command = [
        str(args.bench.resolve()), "--cuda", "-m", str(args.model.resolve()),
        "--prompt-file", str(args.prompt_file.resolve()),
        "--ctx-start", str(start), "--ctx-max", str(end), "--step-incr", "65",
        "--ctx-alloc", "138", "--gen-tokens", "8",
        "--dump-frontier-logits-dir", str((out / "proof").resolve()),
    ]
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    (out / "stdout.csv").write_text(result.stdout)
    (out / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, f"{name}: exit {result.returncode}; see {out}"
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    frontiers = [start] if start == end else [start, end]
    assert [int(row["ctx_tokens"]) for row in rows] == frontiers
    previous = 0
    for row, frontier in zip(rows, frontiers):
        assert int(row["prefill_tokens"]) == frontier - previous
        assert int(row["gen_tokens"]) == 8
        assert float(row["prefill_tps"]) > 0 and float(row["gen_tps"]) > 0
        previous = frontier
    return out / "proof"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("./ds4-bench-perf"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True)

    sweep = run(args, "sweep", 64, 129)
    for frontier in [64, 129]:
        cold = run(args, f"cold-{frontier}", frontier, frontier)
        name = f"frontier_{frontier:06}.logits.json"
        actual = json.loads((sweep / name).read_text())
        expected = json.loads((cold / name).read_text())
        assert actual["vocab"] == expected["vocab"] == 200058
        assert len(actual["logits"]) == len(expected["logits"]) == actual["vocab"]
        assert actual["argmax_id"] == expected["argmax_id"]
        assert actual["logits"] == expected["logits"], f"frontier {frontier}: logits differ"
        name = f"tokens-{frontier}.json"
        actual = json.loads((sweep / name).read_text())
        expected = json.loads((cold / name).read_text())
        assert len(actual) == 8 and actual == expected, f"frontier {frontier}: tokens differ"
    print("PASS: Inkling 64/129-token sweep and cold frontiers have exact logits and tokens")


if __name__ == "__main__":
    main()
