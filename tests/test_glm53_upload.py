#!/usr/bin/env python3
"""Compile the production range transaction with CPU mocks, or emit CUDA bodies."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    """Find one definition; ignore braces in comments and quoted literals."""
    marker = name + "("
    start = source.index(marker)
    while True:
        brace = source.index("{", start)
        semi = source.index(";", start)
        if brace < semi:
            break
        start = source.index(marker, semi + 1)
    start = source.rfind("\n", 0, start) + 1
    brace = source.index("{", start)
    depth, state, i = 0, "code", brace
    while i < len(source):
        char, following = source[i], source[i:i + 2]
        if state == "line":
            if char == "\n":
                state = "code"
        elif state == "comment":
            if following == "*/":
                state, i = "code", i + 1
        elif state in ('"', "'"):
            if char == "\\":
                i += 1
            elif char == state:
                state = "code"
        elif following == "//":
            state, i = "line", i + 1
        elif following == "/*":
            state, i = "comment", i + 1
        elif char in ('"', "'"):
            state = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                line = source[:start].count("\n") + 1
                return f'#line {line} "ds4_cuda.cu"\n' + source[start:i + 1] + "\n"
        i += 1
    raise ValueError("unterminated function: " + name)


def emit(directory):
    source = (ROOT / "ds4_cuda.cu").read_text()
    directory.mkdir(parents=True, exist_ok=True)
    bodies = {
        "glm53_upload_range.inc": ["cuda_model_range_populate_device_copy"],
        "glm53_upload_stage.inc": ["cuda_align_ptr", "cuda_model_stage_pool_release",
                                    "cuda_model_stage_pool_alloc", "cuda_stage_copy_to_dev"],
    }
    for filename, names in bodies.items():
        path = directory / filename
        path.write_text("\n".join(function(source, name) for name in names))
        print(f"{path}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print("production_sha256=" + hashlib.sha256(source.encode()).hexdigest(), flush=True)


def mock(directory):
    binary = directory / "upload-mock"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall",
                    "-Wextra", "-Werror", "-Wno-unused-function",
                    "-Wno-missing-field-initializers", "-I", str(directory),
                    str(ROOT / "tests/test_glm53_upload.cpp"), "-o", str(binary)], check=True)
    baseline = subprocess.run([str(binary), "transaction"])
    if baseline.returncode:
        return baseline.returncode
    return subprocess.run([str(binary), "replacement"]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, help="emit exact source bodies; do not compile/run")
    args = parser.parse_args()
    if args.emit:
        emit(args.emit)
        return 0
    with tempfile.TemporaryDirectory(prefix="ds4-upload-") as directory:
        path = Path(directory)
        emit(path)
        return mock(path)


if __name__ == "__main__":
    raise SystemExit(main())
