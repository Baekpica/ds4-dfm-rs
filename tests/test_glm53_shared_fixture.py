#!/usr/bin/env python3
"""Extract the actual layer17 Q8 shared gate/up, never the full model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from test_glm53_route_fixture import integer, string, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    names = ("blk.17.ffn_gate_shexp.weight", "blk.17.ffn_up_shexp.weight")
    found = {}
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
            raise ValueError("unexpected architecture/alignment")
        for _ in range(tensors):
            name = string(source)
            dims = [integer(source, "Q") for _ in range(integer(source, "I"))]
            kind, offset = integer(source, "I"), integer(source, "Q")
            if name in names:
                found[name] = dict(name=name, dims=dims, type=kind, relative_offset=offset)
        base = (source.tell() + alignment - 1) // alignment * alignment
        size = 4096 // 32 * 2048 * 34
        raw = bytearray()
        for name in names:
            tensor = found.get(name)
            if not tensor or tensor["dims"] != [4096, 2048] or tensor["type"] != 8:
                raise ValueError("unexpected shared tensor shape/type")
            offset = base + tensor["relative_offset"]
            if offset > before.st_size or size > before.st_size - offset:
                raise ValueError("tensor outside source")
            data = os.pread(source.fileno(), size, offset)
            if len(data) != size:
                raise ValueError("truncated tensor")
            tensor.update(absolute_offset=offset, bytes=size, sha256=hashlib.sha256(data).hexdigest())
            raw.extend(data)
        after = os.fstat(source.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("source changed during extraction")
    output = args.directory / "shared.bin"
    output.write_bytes(raw)
    receipt = dict(source=str(args.model.resolve()), source_stat=identity(before),
                   tensors=[found[name] for name in names], bytes=len(raw),
                   sha256=hashlib.sha256(raw).hexdigest(),
                   scope="Actual layer17 shared gate/up only; full SHA not repeated")
    (args.directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
