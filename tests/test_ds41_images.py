#!/usr/bin/env python3
"""Exercise native image admission without a model or GPU."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from test_ds41_model_lifetime import function_body

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "ds4_ds41_vision.inc").read_text()
    body = function_body(source, "ds4_v41_images_build")
    body += function_body(source, "ds4_v41_images_free")
    body += function_body(source, "v41_vision_run")
    with tempfile.TemporaryDirectory(prefix="ds41-images-") as temp:
        directory = Path(temp)
        fixture = directory / "images.inc"
        fixture.write_text(body)
        binary = directory / "images"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-D_GNU_SOURCE", "-O2", "-I.",
            "-ffunction-sections", "-fdata-sections",
            f'-DDS41_IMAGES_FIXTURE="{fixture}"',
            "tests/test_ds41_images.c", "-Wl,--gc-sections", "-o", str(binary)]
        subprocess.run(command, cwd=ROOT, check=True)
        return subprocess.run([str(binary)], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
