#!/usr/bin/env python3
"""Exercise actual vision transfer/lifetime code with small host tensors."""
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from test_ds41_model_lifetime import function_body

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "ds4_ds41_vision.inc").read_text()
    stage = function_body(source, "v41_vis_stage")
    old_api = stage.split("{", 1)[0].count(",") == 2
    body = "\n".join(re.findall(
        r"^#define V41_VIS_(?:RMS_EPS|STAGE_BYTES)\b[^\n]*", source, re.M)) + "\n"
    for name in ("v41_vision_model", "v41_vis_stage", "v41_vis_grow",
                 "v41_vis_free_list", "v41_vis_proj", "v41_vis_residual",
                 "v41_vis_run_images", "ds4_v41_images_bytes"):
        body += function_body(source, name)
    # Keep the original call graph executable for a behavioral pre-fix failure.
    if old_api:
        body += """
static bool test_stage(const ds4_model *vm, const ds4_tensor *t,
                       ds4_gpu_tensor **slot, void *stage) {
    (void)stage;
    return v41_vis_stage(vm, t, slot);
}
"""
    else:
        body += "#define test_stage v41_vis_stage\n"
    print(f"vision source sha256={hashlib.sha256(source.encode()).hexdigest()} "
          f"stage_api={'original-3arg' if old_api else 'staged-4arg'}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ds41-vision-stage-") as temp:
        output = Path(temp)
        header = output / "vision.inc"
        header.write_text(body)
        binary = output / "stage"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-D_GNU_SOURCE", "-O2", "-I.",
            f'-DDS41_VISION_STAGE_FIXTURE="{header}"',
            "tests/test_ds41_vision_stage.c", "-o", str(binary)]
        subprocess.run(command, cwd=ROOT, check=True)
        return subprocess.run([str(binary)], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
