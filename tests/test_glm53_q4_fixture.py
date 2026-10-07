#!/usr/bin/env python3
"""Extract only the actual layer17 Q4_K query tensor; no model allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from test_glm53_route_fixture import integer, string, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    name = "blk.17.kda_q.weight"
    with args.model.open("rb") as source:
        before = os.fstat(source.fileno())
        if source.read(4) != b"GGUF" or integer(source, "I") not in (2, 3):
            raise ValueError("expected GGUF v2/v3")
        tensors, metadata = integer(source, "Q"), integer(source, "Q")
        architecture, alignment = None, 32
        for _ in range(metadata):
            key = string(source)
            data = value(source, integer(source, "I"))
            if key == "general.architecture":
                architecture = data
            elif key == "general.alignment":
                alignment = data
        if architecture != "glm5-next" or alignment != 32:
            raise ValueError("unexpected artifact architecture/alignment")
        found = None
        for _ in range(tensors):
            tensor = string(source)
            dims = [integer(source, "Q") for _ in range(integer(source, "I"))]
            kind, offset = integer(source, "I"), integer(source, "Q")
            if tensor == name:
                found = dict(name=name, dims=dims, type=kind, relative_offset=offset)
        if not found or found["dims"] != [4096, 8192] or found["type"] != 12:
            raise ValueError("unexpected layer17 query shape/type")
        base = (source.tell() + alignment - 1) // alignment * alignment
        size = 4096 // 256 * 8192 * 144
        offset = base + found["relative_offset"]
        if offset > before.st_size or size > before.st_size - offset:
            raise ValueError("tensor outside source")
        raw = os.pread(source.fileno(), size, offset)
        if len(raw) != size:
            raise ValueError("truncated query tensor")
        after = os.fstat(source.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("source changed during extraction")
    output = args.directory / "q4.bin"
    output.write_bytes(raw)
    found.update(absolute_offset=offset, bytes=size, sha256=hashlib.sha256(raw).hexdigest())
    receipt = dict(source=str(args.model.resolve()), source_stat=identity(before),
                   tensor=found, scope="actual layer17 query only; full SHA not repeated")
    (args.directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
