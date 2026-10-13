#!/usr/bin/env python3
"""Exercise the live harness's actual consumed-history comparison."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from ds41_generate_boundary_gate import function


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source = root / "tests/test_ds41_generate.cu"
    text = source.read_text()
    body, line = function(text, "hist_vs_golden")
    block_start = text.index("if (rows_ref) {")
    block_end = text.index("/* No initial prompt fill:", block_start)
    block = text[block_start:block_end].rstrip()
    block_line = text[:block_start].count("\n") + 1
    with tempfile.TemporaryDirectory(prefix="ds41-generate-history-") as directory:
        temp = Path(directory)
        (temp / "ds41_history_under_test.inc").write_text(
            f'#line {line} "{source}"\n{body}\n'
            + "static int load_refs(gen_feed &g, const std::vector<int> &gids, const char *rows_ref) {\n"
            + f'#line {block_line} "{source}"\n{block}\nreturn 0;\n}}\n'
        )
        for name, count in [("full", 3), ("short", 2)]:
            (temp / f"{name}.erows_L00.txt").write_text("1 2\n" * count)
        binary = temp / "test_ds41_generate_history"
        subprocess.run(
            shlex.split(os.environ.get("CXX", "c++"))
            + ["-std=c++17", "-O0", "-Wall", "-Wextra", "-I", str(temp),
               str(root / "tests/test_ds41_generate_history.cpp"), "-o", str(binary)],
            check=True,
        )
        return subprocess.run([str(binary), str(temp / "full"), str(temp / "short")], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
