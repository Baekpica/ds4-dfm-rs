#!/usr/bin/env python3
"""Compare per-stage hidden states: engine dump vs reference dump.

  python compare_hidden.py REFDIR OURDIR

Prints one line per stage: cosine, relative RMS (||d|| / ||ref||) and the max
absolute difference. The first stage whose cosine leaves 0.9999 is the first
diverging layer; later stages inherit it.
"""

import sys
from pathlib import Path

import numpy as np


def read(path):
    return np.fromfile(path, dtype=np.float32)


def main():
    ref_dir, our_dir = Path(sys.argv[1]), Path(sys.argv[2])
    names = ["embed"]
    for il in range(36):
        names += [f"norm_attn{il}", f"norm_ffn{il}", f"layer{il + 1}"]
    names.append("final")
    for name in names:
        ref_path, our_path = ref_dir / f"{name}.f32", our_dir / f"{name}.f32"
        if not our_path.exists() and (our_dir / f"stage.{name}.f32").exists():
            # The engine's dump is <prefix>.<tag>.f32.
            our_path = our_dir / f"stage.{name}.f32"
        if not ref_path.exists() or not our_path.exists():
            print(f"{name:12s} missing ({ref_path.exists()}, {our_path.exists()})")
            continue
        ref, ours = read(ref_path), read(our_path)
        if ref.shape != ours.shape:
            print(f"{name:12s} shape mismatch {ref.shape} vs {ours.shape}")
            continue
        rn = float(np.linalg.norm(ref))
        on = float(np.linalg.norm(ours))
        cos = float(ref @ ours / (rn * on)) if rn and on else 0.0
        rel = float(np.linalg.norm(ref - ours) / rn) if rn else 0.0
        mx = float(np.max(np.abs(ref - ours)))
        print(f"{name:12s} cos {cos:.6f} relRMS {rel:.3e} max {mx:.3e} "
              f"norm {rn:.3f}/{on:.3f}")


if __name__ == "__main__":
    main()
