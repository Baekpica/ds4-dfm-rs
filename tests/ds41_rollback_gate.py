#!/usr/bin/env python3
"""Execute native rollback/readback against bounded tensors and corrupt restores."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from ds41_generate_boundary_gate import function


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source_path = root / "ds4_ds41_forward.inc"
    source = source_path.read_text()
    names = ["v41_restore_match", "v41_rollback_check", "v41_spec_rollback"]
    with tempfile.TemporaryDirectory(prefix="ds41-rollback-") as directory:
        temp = Path(directory)
        pieces = []
        for name in names:
            if re.search(rf"\b{re.escape(name)}\s*\(", source) is None:
                continue
            body, line = function(source, name)
            pieces.append(f'#line {line} "{source_path}"\n{body}\n')
        (temp / "ds41_rollback_under_test.inc").write_text("\n".join(pieces))
        binary = temp / "test_ds41_rollback"
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc"))
            + ["-std=c99", "-D_POSIX_C_SOURCE=200809L", "-O0", "-Wall", "-Wextra", "-I", str(temp),
               str(root / "tests/test_ds41_rollback.c"), "-o", str(binary)],
            check=True,
        )
        return subprocess.run([str(binary)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
