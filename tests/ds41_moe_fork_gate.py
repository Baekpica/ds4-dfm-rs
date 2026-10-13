#!/usr/bin/env python3
"""Execute the exact native MoE driver with failing launchers, without a GPU."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def extract_function(source: str, name: str) -> tuple[str, int]:
    # Mask strings/comments to count code braces while retaining source offsets.
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
    masked = tokens.sub(lambda match: " " * len(match.group()), source)
    start = re.search(rf"(?:static\s+)?(?:bool|int) {re.escape(name)}\s*\(", masked)
    if start is None:
        raise RuntimeError(f"missing native function: {name}")
    opening = masked.index("{", start.start())
    depth = 0
    for pos in range(opening, len(masked)):
        if masked[pos] == "{":
            depth += 1
        elif masked[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start.start() : pos + 1], source[: start.start()].count("\n") + 1
    raise RuntimeError(f"unterminated native function: {name}")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source_path = root / "ds4_ds41_forward.inc"
    function, line = extract_function(source_path.read_text(), "v41_moe")
    with tempfile.TemporaryDirectory(prefix="ds41-moe-fork-") as directory:
        temp = Path(directory)
        # Diagnostics identify the original native source, not a rewritten copy.
        header = temp / "ds41_moe_under_test.inc"
        header.write_text(f'#line {line} "{source_path}"\n{function}\n')
        binary = temp / "test_ds41_moe_fork"
        compiler = shlex.split(os.environ.get("CC", "cc"))
        subprocess.run(
            compiler
            + [
                "-std=c99",
                "-O0",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-I",
                str(temp),
                str(root / "tests/test_ds41_moe_fork.c"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        return subprocess.run([str(binary)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
