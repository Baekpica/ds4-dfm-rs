#!/usr/bin/env python3
"""Extract only canonical layer-3 Q2_K down weights for the bounded target."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from test_glm53_route_fixture import integer, string, value


def prepare(path, directory):
    wanted = "blk.3.ffn_down_exps.weight"
    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        if file.read(4) != b"GGUF" or integer(file, "I") not in (2, 3):
            raise ValueError("expected GGUF v2/v3")
        tensors, metadata = integer(file, "Q"), integer(file, "Q")
        alignment, architecture = 32, None
        for _ in range(metadata):
            key, kind = string(file), integer(file, "I")
            data = value(file, kind)
            if key == "general.alignment":
                alignment = data
            elif key == "general.architecture":
                architecture = data
        if architecture != "glm5-next" or alignment != 32:
            raise ValueError("unexpected architecture/alignment")
        record = None
        for _ in range(tensors):
            name, ndim = string(file), integer(file, "I")
            dims = [integer(file, "Q") for _ in range(ndim)]
            kind, offset = integer(file, "I"), integer(file, "Q")
            if name == wanted:
                record = dict(name=name, dims=dims, type=kind, relative_offset=offset)
        if record is None or record["dims"] != [2048, 4096, 288] or record["type"] != 10:
            raise ValueError("unexpected layer-3 down recipe/shape")
        start = (file.tell() + alignment - 1) // alignment * alignment
        unit = (2048 // 256) * 4096 * 84
        offset = start + record["relative_offset"]
        size = unit * 288
        if offset > before.st_size or size > before.st_size - offset:
            raise ValueError("tensor outside file")
        directory.mkdir(parents=True, exist_ok=False)
        digest = hashlib.sha256()
        with (directory / "q2.bin").open("xb") as dest:
            for expert in range(288):
                data = os.pread(file.fileno(), unit, offset + expert * unit)
                if len(data) != unit:
                    raise ValueError("truncated expert")
                dest.write(data)
                digest.update(data)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        after = os.fstat(file.fileno())
        if identity(before) != identity(after):
            raise ValueError("source changed during extraction")
    record.update(absolute_offset=offset, expert_bytes=unit, tensor_bytes=size)
    receipt = dict(source=str(path.resolve()), source_stat_before=identity(before),
                   source_stat_after=identity(after), tensor=record,
                   fixture_sha256=digest.hexdigest(), fixture_bytes=size,
                   scope="canonical original GGUF bytes; full source SHA not re-read")
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    prepare(args.model, args.directory)
