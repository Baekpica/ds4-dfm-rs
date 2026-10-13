#!/usr/bin/env python3
"""Check the real VQ promotion's Linux cache release without a GPU."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ds4_cuda.cu").read_text()
ranges = [
    ("struct cuda_model_range {", "static "),
    ("static void cuda_model_discard_source_pages(", "static uint64_t cuda_round_down("),
    ("static int cuda_unit_materialize_copy(", "/* memgov D1b-2: the LAZY pass"),
]
parts = []
for start, end in ranges:
    lo = source.index(start)
    hi = source.index("};", lo) + 2 if start.startswith("struct ") else source.index(end, lo)
    parts.append(source[lo:hi])
with tempfile.TemporaryDirectory(prefix="ds4-vq-source-release-") as tmp:
    directory = Path(tmp)
    for name, text in zip(("range.inc", "release.inc", "materialize.inc"), parts):
        (directory / name).write_text(text)
    binary = directory / "gate"
    subprocess.run([
        "c++", "-std=c++17", "-O0", "-Wall", "-Wextra", "-Wno-unused-function",
        "-Wno-missing-field-initializers", "-I", str(directory),
        str(root / "tests/test_vq_source_release.cpp"), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], cwd=root, check=True)
