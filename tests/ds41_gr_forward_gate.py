#!/usr/bin/env python3
"""Check native GR upload/clear forwarding without a GPU or model."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from ds41_moe_fork_gate import extract_function


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source_path = root / "ds4_ds41_forward.inc"
    function, line = extract_function(source_path.read_text(), "ds4_v41_zchain_gr_set")
    with tempfile.TemporaryDirectory(prefix="ds41-gr-forward-") as directory:
        temp = Path(directory)
        (temp / "ds41_gr_under_test.inc").write_text(
            f'#line {line} "{source_path}"\n{function}\n'
        )
        binary = temp / "test_ds41_gr_forward"
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + [
                "-std=c99",
                "-O0",
                "-Wall",
                "-Wextra",
                "-I",
                str(temp),
                str(root / "tests/test_ds41_gr_forward.c"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        return subprocess.run([str(binary)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
