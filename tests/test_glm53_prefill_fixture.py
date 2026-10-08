#!/usr/bin/env python3
"""Compile only the actual GLM Prefill coordinator with bounded mocks.

Checks failure cleanup and absolute chunk frontiers. Kernel arithmetic, actual
KV/cache bytes, model loading and bank admission are outside this fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import os

from test_glm53_close_fixture import ROOT, run


def body(source, name="glm53_graph_prefill"):
    match = re.search(r"^(?:static )?(?:bool|int|void|uint32_t|uint64_t) " + re.escape(name) + r"\(",
                      source, re.M)
    if not match:
        raise ValueError("Prefill signature changed; update the fixture boundary")
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
    raise ValueError("unterminated Prefill function")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="fresh evidence directory")
    parser.add_argument("cases", nargs="*", default=["tail", "window", "bind", "alignment", "caps"])
    parser.add_argument("--source", type=Path, default=ROOT / "ds4_glm53_graph.inc")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    fixture = ROOT / "tests/test_glm53_prefill.c"
    before = {str(path): digest(path) for path in [source, fixture]}
    graph = source.read_text()
    code = "\n".join(body(graph, name) for name in (
        "glm53_graph_row_cap", "glm53_window_cap", "glm53_graph_prefill"))
    header = output / "prefill.inc"
    header.write_text(code)
    (output / "graph-source.inc").write_bytes(source.read_bytes())
    (output / "test-source.c").write_bytes(fixture.read_bytes())
    binary = output / "test-prefill"
    command = shlex.split(args.cc) + ["-O2", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
        f'-DGLM53_PREFILL_FIXTURE="{header}"', "-o", str(binary), str(fixture)]
    built = run(command, output, "build")
    after = {str(path): digest(path) for path in [source, fixture]}
    receipt = dict(sources=before, unchanged=before == after,
                  prefill_sha256=hashlib.sha256(code.encode()).hexdigest(),
                  scope="actual Prefill coordinator only; arithmetic/kernel/bank mocks")
    (output / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if before != after:
        raise ValueError("source changed during fixture compilation")
    if built:
        return 1
    failed = False
    for case in args.cases:
        failed |= run([str(binary), case], output, case) != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
