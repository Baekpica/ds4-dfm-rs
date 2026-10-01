#!/usr/bin/env python3
"""Compile every IQuest Metal stub and require unsupported calls to fail.

This checks the backend API surface on Linux, not a complete Metal link.
"""
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
STUB = "ds4_iquest_stub.inc"


def main():
    metal = (ROOT / "ds4_metal.m").read_text()
    assert f'#include "{STUB}"' in metal, "Metal must include IQuest fail-closed stubs"

    # Derive calls from the production ABI so an added entrypoint cannot hide
    # behind a hand-maintained list of already-covered symbols.
    header = (ROOT / "ds4_gpu.h").read_text()
    functions = re.findall(r"int\s+(ds4_gpu_iquest_\w+)\s*\(([^;]+)\);", header)
    assert functions, "IQuest GPU declarations are missing"
    calls = []
    for name, params in functions:
        args = [] if params.strip() == "void" else ["0"] * len(params.split(","))
        calls.append(f"    if ({name}({', '.join(args)}) != 0) {{ return 1; }}")
    source = '\n'.join([
        '#include "ds4_gpu.h"',
        f'#include "{STUB}"',
        "int main(void) {",
        *calls,
        "    return 0;",
        "}",
    ])
    with tempfile.TemporaryDirectory(prefix="iquest-metal-stubs-") as work:
        work = Path(work)
        code, binary = work / "check.c", work / "check"
        code.write_text(source)
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-I", str(ROOT), str(code), "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)
    print(f"IQuest Metal stub API: {len(functions)} entrypoints fail closed")


if __name__ == "__main__":
    main()
