#!/usr/bin/env python3
"""Check actual graph/rollback execution and hash the completed gate receipts."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


if len(sys.argv) != 3:
    raise SystemExit("usage: audit.py OUT NATIVE_LOG")
out, log_path = map(lambda value: Path(value).resolve(), sys.argv[1:])
if not (out / "COMPLETE").is_file():
    raise SystemExit("driver did not complete")
log = log_path.read_text(errors="replace")
stats = rows(out / "stats.csv")
goldens = rows(out / "goldens.csv")
comparisons = rows(out / "comparisons.csv")
config = dict(line.split("\t", 1) for line in (out / "config.tsv").read_text().splitlines())
errors = []
if {row["name"] for row in goldens} != {"panel", "bug", "homer"} or len(goldens) != 3 or any(row["status"] != "PASS" for row in goldens):
    errors.append("all three golden fixtures must pass")
default_comparisons = {
    ("cold-p", "full-hit-p", "0"),
    ("cold-p", "warm-suffix-one", "1"),
    ("changed-image-miss", "changed-image-cold", "0"),
    ("follow-text-reencode", "follow-text-reuse", "0"),
    ("append-image-reencode", "append-image-reuse", "0"),
    ("reordered-images-miss", "reordered-images-cold", "0"),
}
completed = {(row["left"], row["right"], row["skip_left_rows"]) for row in comparisons if row["scope"] == "tokens_and_all_logits"}
if not default_comparisons.issubset(completed) or any(row["status"] != "PASS" for row in comparisons):
    errors.append("default exact comparisons are incomplete")
expected_generations = 23
if config["long"] != "off" or config["mtp"] == "fixed5":
    expected_generations += 4
if config["mtp"] == "fixed5":
    expected_generations += 4
if len(stats) != expected_generations or len({row["name"] for row in stats}) != len(stats):
    errors.append("missing or duplicate generation receipts")
segments = dict(re.findall(r"^@@ BEGIN (\S+) [^\n]*\n(.*?)^@@ END \1 [^\n]*$", log, re.M | re.S))
graph_rows = []
for row in stats:
    name = row["name"]
    segment = segments.get(name, "")
    walked = sum(map(int, re.findall(r"\[graph\] walked (\d+) pure-decode steps", segment)))
    batches = sum(map(int, re.findall(r"\[graph\] verify batches walked (\d+) rounds", segment)))
    rollback = re.findall(r"rollback check PASS:.*?keep (\d+)/(\d+)", segment)
    rejected = sum(int(keep) < int(width) for keep, width in rollback)
    failures = []
    if row["status"] != "PASS" or row["requested"] != row["emitted"] or row["reused"] != row["expected_reused"]:
        failures.append("generation/reuse contract")
    if row["nonfinite"] != "0":
        failures.append("nonfinite logits")
    if row["mode"] == "pure" and row["logit_rows"] != row["requested"]:
        failures.append("full logits extent")
    if not segment:
        failures.append("missing native request log")
    if row["graph"] == "captured" and int(row["requested"]) >= 8:
        if row["mode"] == "pure" and walked == 0:
            failures.append("no captured pure steps")
        if row["mode"] == "fixed5" and batches == 0:
            failures.append("no captured verify batches")
    if row["graph"] == "eager" and (walked or batches):
        failures.append("eager request used graphs")
    if row["mode"] == "fixed5" and (not rollback or not rejected):
        failures.append("no checked rejected-draft rollback")
    if "rollback check FAIL" in segment or "capture failed" in segment:
        failures.append("native rollback/capture failure")
    if re.search(r"argmax kernel \d+ != host \d+", segment):
        failures.append("native argmax self-check failed")
    if failures:
        errors.append(f"{name}: {', '.join(failures)}")
    graph_rows.append(dict(name=name, pure_graph_steps=walked, verify_graph_batches=batches,
                           rollback_checks=len(rollback), rejected_checks=rejected,
                           status="FAIL" if failures else "PASS"))
if config["long"] != "off":
    pure_long = [row for row in stats if row["name"].startswith("extended-") and row["mode"] == "pure"]
    if len(pure_long) != 4 or any(row["requested"] != "1024" for row in pure_long):
        errors.append("long gate requires 1024-output eager/captured runs on bare and GR")
if config["mtp"] == "fixed5" and sum(row["mode"] == "fixed5" for row in stats) != 4:
    errors.append("fixed5 must run eager/captured on bare and GR")
with (out / "graph-evidence.csv").open("w") as target:
    writer = csv.DictWriter(target, fieldnames=["name", "pure_graph_steps", "verify_graph_batches", "rollback_checks", "rejected_checks", "status"])
    writer.writeheader()
    writer.writerows(graph_rows)
for line in (out / "SHA256SUMS").read_text().splitlines():
    expected, path = line.split("  ", 1)
    if sha(Path(path)) != expected:
        errors.append(f"changed artifact: {path}")
receipt = dict(passed=not errors, errors=errors, generations=len(stats), golden_fixtures=len(goldens),
               comparisons=len(comparisons), config=config, native_log_sha256=sha(log_path),
               driver_file_hashes={str(path.name): sha(path) for path in out.iterdir() if path.is_file() and path.name not in {"audit.json", "audit.sha256"}})
(out / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
(out / "audit.sha256").write_text(f"{sha(out / 'audit.json')}  {out / 'audit.json'}\n{sha(log_path)}  {log_path}\n")
print(json.dumps({key: receipt[key] for key in ["passed", "errors", "generations", "golden_fixtures", "comparisons"]}, indent=2))
raise SystemExit(bool(errors))
