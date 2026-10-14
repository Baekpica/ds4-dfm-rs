#!/usr/bin/env python3
"""Run the actual ds41 close/bias paths with bounded host tensors."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function_body(source, name, *, optional=False):
    match = re.search(r"^.*\b" + re.escape(name) + r"\([^;]*?\) \{", source, re.M)
    if not match:
        if optional:
            return ""
        raise ValueError(f"missing function {name}")
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
    raise ValueError(f"unterminated function {name}")


def main():
    engine = (ROOT / "ds4.c").read_text()
    vision = (ROOT / "ds4_ds41_vision.inc").read_text()
    body = function_body(vision, "v41_vis_free_list")
    # Optional only so the pre-fix regression executes the original close path.
    body += function_body(vision, "v41_vision_cleanup", optional=True)
    body += function_body(vision, "v41_vis_bias_vl_upload")
    body += function_body(engine, "ds4_engine_close")
    with tempfile.TemporaryDirectory(prefix="ds41-model-lifetime-") as temp:
        output = Path(temp)
        header = output / "close.inc"
        header.write_text(body)
        binary = output / "lifetime"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-D_GNU_SOURCE", "-O2", "-I.",
            f'-DDS41_CLOSE_FIXTURE="{header}"',
            "tests/test_ds41_model_lifetime.c", "-o", str(binary)]
        subprocess.run(command, cwd=ROOT, check=True)
        return subprocess.run([str(binary)], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
