#!/usr/bin/env python3
"""Compile actual GLM bank admission with controlled allocator failures.

No model or GPU is loaded. The fixture preserves the production retry and
publication code; numerical execution and GPU allocation are mocked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

from test_glm53_close_fixture import ROOT, run
from test_glm53_prefill_fixture import body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="*", default=[
        "retry", "two", "one", "disabled", "small", "strict", "fail"])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--window-policy", type=Path,
                        help="Rust allocation-policy receipt for a 2-to-1 bank retry")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["ds4_glm53_graph.inc", "ds4_glm53_batch.inc", "tests/test_glm53_fit.c"]
    contents = {name: (ROOT / name).read_bytes() for name in names}
    code = "\n".join(body(contents[names[0]].decode(), name) for name in (
        "glm53_graph_row_cap", "glm53_window_cap", "glm53_prefill_cap"))
    code += body(contents[names[1]].decode(), "glm53_batch_ctx_create")
    header = output / "fit.inc"
    header.write_text(code)
    for name, raw in contents.items():
        saved = output / "source" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(raw)
    binary = output / "test-fit"
    flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] \
        if args.sanitize else ["-O2"]
    command = shlex.split(args.cc) + flags + ["-std=gnu11", "-Wall", "-Wextra", "-Werror",
        f'-DGLM53_FIT_FIXTURE="{header}"', "-o", str(binary), str(ROOT / names[2])]
    built = run(command, output, "build")
    receipt = dict(sources={name: hashlib.sha256(raw).hexdigest()
                           for name, raw in contents.items()},
                   unchanged=all((ROOT / name).read_bytes() == raw
                                 for name, raw in contents.items()),
                   coordinator_sha256=hashlib.sha256(code.encode()).hexdigest(),
                   scope="actual bank admission/retry; graph/GPU allocation mocks")
    (output / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["unchanged"]:
        raise ValueError("source changed during fixture compilation")
    if built:
        return 1
    env = os.environ.copy()
    if args.window_policy:
        env["DS4_GLM53_PREFILL_WINDOW"] = args.window_policy.read_text().strip()
        args.cases.append("policy-retry")
    if args.sanitize:
        env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1")
    failed = False
    for case in args.cases:
        failed |= run([str(binary), case], output, case, env) != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
