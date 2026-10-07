#!/usr/bin/env python3
"""Metadata-only canonical I/O plan; reuses an existing artifact hash receipt."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from test_glm53_route_fixture import BLOCK, integer, string, value

LAYERS = (3, 4, 5, 43, 44, 17)
EXPERTS = (0, 143, 287)
PARTS = ("gate", "up", "down")
MODEL_SHA = "7f6f96df758b5d651561c2f06ffdd0d1075a24f10654320e24e7d66c48db6017"
MODEL_BYTES = 93920031264
MAGIC = b"GLMIO01\0"


def identity(stat):
    return dict(bytes=stat.st_size, inode=stat.st_ino, device=stat.st_dev,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def prepare(model, receipt, output):
    pin = json.loads(receipt.read_text())["files"]["main"]
    if pin["sha256"] != MODEL_SHA or pin["bytes"] != MODEL_BYTES:
        raise ValueError("unexpected artifact hash receipt")
    wanted = {f"blk.{layer}.ffn_{part}_exps.weight"
              for layer in LAYERS for part in PARTS}
    catalog = {}
    with model.open("rb") as source:
        before = os.fstat(source.fileno())
        if identity(before) != {key: pin[key] for key in identity(before)}:
            raise ValueError("artifact stat differs from prior hash receipt")
        if source.read(4) != b"GGUF" or integer(source, "I") not in (2, 3):
            raise ValueError("expected GGUF v2/v3")
        tensors, metadata = integer(source, "Q"), integer(source, "Q")
        architecture, alignment = None, 32
        for _ in range(metadata):
            key, kind = string(source), integer(source, "I")
            data = value(source, kind)
            if key == "general.architecture":
                architecture = data
            elif key == "general.alignment":
                alignment = data
        if architecture != "glm5-next" or alignment != 32:
            raise ValueError("unexpected artifact architecture/alignment")
        for _ in range(tensors):
            name, ndim = string(source), integer(source, "I")
            dims = [integer(source, "Q") for _ in range(ndim)]
            kind, offset = integer(source, "I"), integer(source, "Q")
            if name in wanted:
                catalog[name] = dict(name=name, dims=dims, type=kind,
                                     relative_offset=offset)
        base = (source.tell() + alignment - 1) // alignment * alignment
        units, gate_max, down_max, gate_align, down_align = [], 0, 0, 1, 1
        for layer in LAYERS:
            edge = layer != 17
            for expert in EXPERTS:
                for part, name in enumerate(PARTS):
                    tensor = catalog[f"blk.{layer}.ffn_{name}_exps.weight"]
                    dims = [4096, 2048, 288] if part < 2 else [2048, 4096, 288]
                    kind = (17 if part < 2 else 10) if edge else (16 if part < 2 else 17)
                    if tensor["dims"] != dims or tensor["type"] != kind:
                        raise ValueError("unexpected routed shape/recipe")
                    size = dims[0] // 256 * dims[1] * BLOCK[kind]
                    start = base + tensor["relative_offset"]
                    total = size * dims[2]
                    if start > before.st_size or total > before.st_size - start:
                        raise ValueError("canonical tensor lies outside artifact")
                    units.append(dict(layer=layer, expert=expert, part=part, type=kind,
                                      offset=start + expert * size, bytes=size,
                                      name=tensor["name"], dims=dims))
                    if part < 2:
                        gate_max = max(gate_max, size)
                        gate_align = math.lcm(gate_align, BLOCK[kind])
                    else:
                        down_max = max(down_max, size)
                        down_align = math.lcm(down_align, BLOCK[kind])
        if identity(os.fstat(source.fileno())) != identity(before):
            raise ValueError("artifact changed while reading metadata")
    strides = tuple((size + align - 1) // align * align
                    for size, align in ((gate_max, gate_align), (down_max, down_align)))
    output.mkdir(parents=True, exist_ok=True)
    # The binary carries immutable stat identity and exact canonical offsets.
    header = struct.pack("<7Q3I", before.st_size, before.st_dev, before.st_ino,
                         before.st_mtime_ns, before.st_ctime_ns, *strides,
                         len(units), 8, max(gate_max, down_max))
    plan = MAGIC + header + b"".join(struct.pack("<4I2Q", u["layer"], u["expert"],
           u["part"], u["type"], u["offset"], u["bytes"]) for u in units)
    (output / "plan.bin").write_bytes(plan)
    manifest = dict(schema="glm53-stream-io-v1", model=str(model.resolve()),
                    model_sha256=MODEL_SHA, source_stat=identity(before),
                    hash_receipt=str(receipt.resolve()),
                    hash_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    gate_stride=strides[0], down_stride=strides[1], slots=8,
                    staging_bytes=max(gate_max, down_max), units=units,
                    bytes_per_cycle=sum(u["bytes"] for u in units),
                    plan_sha256=hashlib.sha256(plan).hexdigest(),
                    scope="canonical fixed-route hot-page I/O; no math or error-contract parity")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("bytes_per_cycle", "gate_stride", "down_stride", "staging_bytes")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.receipt, args.output)


if __name__ == "__main__":
    main()
