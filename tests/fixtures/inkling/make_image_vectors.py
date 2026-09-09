#!/usr/bin/env python3
"""Pinned TorchvisionBackend fused normalization, including raw -1 padding."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]) * 255
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]) * 255
    pixels = torch.arange(-1, 256, dtype=torch.float32)[:, None]
    values = (pixels - mean) / std
    bits = [[struct.unpack("<I", struct.pack("<f", v))[0] for v in row] for row in values.tolist()]
    header = json.dumps({
        "torch": torch.__version__,
        "source_revision": "cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55",
        "scope": "RGB values -1..255, FP32 fused rescale/CLIP normalization",
    }, indent=2)
    raw = header[:-2] + ',\n  "bits": [\n    '
    raw += ',\n    '.join(json.dumps(row) for row in bits) + '\n  ]\n}\n'
    args.output.write_text(raw)
    print(hashlib.sha256(raw.encode()).hexdigest())


if __name__ == "__main__":
    main()
