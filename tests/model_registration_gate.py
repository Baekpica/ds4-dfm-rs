#!/usr/bin/env python3
"""Exercise the actual registration entry with model-free CUDA stubs."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ds4_cuda.cu").read_text()
start = source.index('extern "C" int ds4_gpu_set_model_map(')
end = source.index('extern "C" void ds4_gpu_unregister_model_map(', start)
with tempfile.TemporaryDirectory(prefix="ds4-map-registration-") as tmp:
    directory = Path(tmp)
    (directory / "model_registration.inc").write_text(source[start:end])
    binary = directory / "gate"
    subprocess.run([
        "c++", "-std=c++17", "-O0", "-Wall", "-Wextra", "-Wno-unused-function",
        "-I", str(directory), str(root / "tests/test_model_registration.cpp"),
        "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
