#!/usr/bin/env python3
"""Build the model-free stream fixture with the actual engine-close body.

The C fixture mocks unrelated engine cleanup while preserving real expert I/O,
reader cancellation, stream destruction and weights-table invalidation. Source
extraction keeps shutdown-order regressions tied to ds4.c instead of duplicating
its call order in a hand-written fake. No model or GPU is loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["ds4.c", "ds4_glm53_stream.inc", "ds4_gpu.h", "tests/test_glm53_stream.c"]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def close_body(source):
    match = re.search(r"^void ds4_engine_close\(ds4_engine \*e\) \{", source, re.M)
    if not match:
        raise ValueError("engine-close signature changed; update the fixture boundary")
    # Ignore braces in strings and comments while finding the exact function.
    tokens = re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|[{}]',
                         source[match.start():], re.S)
    depth = 0
    for token in tokens:
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if not depth:
                return source[match.start():match.start() + token.end()] + "\n"
    raise ValueError("unterminated engine-close function")


def run(command, output, name, env=None):
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    record = dict(command=command, exit_code=result.returncode,
                  seconds=time.monotonic() - start)
    (output / f"{name}.stdout").write_text(result.stdout)
    (output / f"{name}.stderr").write_text(result.stderr)
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(case=name, **record)))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="fresh evidence directory")
    parser.add_argument("cases", nargs="*", default=["ctor", "close"])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    for name in SOURCES:
        saved = output / "source" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((ROOT / name).read_bytes())
    body = close_body((ROOT / "ds4.c").read_text())
    header = output / "engine-close.inc"
    header.write_text(body)
    binary = output / "test-stream"
    flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] \
        if args.sanitize else ["-O2"]
    command = shlex.split(args.cc) + flags + ["-D_GNU_SOURCE", "-std=gnu11", "-I.",
        f'-DGLM53_CLOSE_FIXTURE="{header}"', "-ffunction-sections", "-fdata-sections",
        "-o", str(binary), "tests/test_glm53_stream.c", "-Wl,--gc-sections", "-lm", "-pthread"]
    built = run(command, output, "build")
    after = hashes()
    receipt = dict(sources=before, unchanged=before == after,
                  close_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  scope="actual engine-close body; model-free CPU I/O/tensor boundary")
    (output / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if before != after:
        raise ValueError("source changed during fixture compilation")
    if built:
        return 1
    env = os.environ.copy()
    if args.sanitize:
        env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1")
    failed = False
    for case in args.cases:
        failed |= run([str(binary), "prefetch", case], output, case, env) != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
