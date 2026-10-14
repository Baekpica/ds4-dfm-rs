#!/usr/bin/env python3
"""Create two deterministic, equal-geometry PNG inputs without model work."""
import hashlib
import json
from pathlib import Path
import struct
import zlib

HERE = Path(__file__).resolve().parent
OUT = HERE / "inputs"
EDGE = 64


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def png(first, second):
    raw = bytearray()
    for y in range(EDGE):
        raw.append(0)
        for x in range(EDGE):
            raw.extend(first if (x // 8 + y // 8) % 2 == 0 else second)
    header = struct.pack(">IIBBBBB", EDGE, EDGE, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


OUT.mkdir(parents=True, exist_ok=True)
images = {
    "image-a.png": png((240, 24, 24), (255, 225, 40)),
    "image-b.png": png((24, 70, 240), (24, 220, 100)),
}
receipt = {}
for name, data in images.items():
    path = OUT / name
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f"refusing to replace changed fixture {path}")
    path.write_bytes(data)
    receipt[name] = {"width": EDGE, "height": EDGE, "sha256": hashlib.sha256(data).hexdigest()}
(OUT / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"model_or_GPU_execution": False, "fixtures": receipt}, indent=2))
