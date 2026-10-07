#!/usr/bin/env python3
"""Emit exact GLM kernels/helper; optionally pread eight artifact experts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from test_glm53_upload import function

ROOT = Path(__file__).resolve().parents[1]
IDS = [287, 256, 270, 3, 128, 255, 1, 281]
BLOCK = {16: 66, 17: 74, 10: 84}


def emit(directory):
    cuda = (ROOT / "ds4_cuda.cu").read_text()
    repack = (ROOT / "cuda/mmq/ds4_repack.cu").read_text()
    start = cuda.index("struct ds4_gpu_tensor {")
    end = cuda.index("};", start) + 2
    bodies = [cuda[start:end], "enum class GlmMoELayout { Raw, IQ2SoA, Q2SoA };"]
    for name in ("repack_iq2_xxs_aligned_kernel", "repack_q2_k_aligned_kernel"):
        bodies.append(function(repack, name).replace('"ds4_cuda.cu"', '"cuda/mmq/ds4_repack.cu"'))
    for name in ("moe_mmq_swiglu_weighted_clamp_kernel", "moe_sum_kernel", "glm53_moe_mixed"):
        bodies.append(function(cuda, name))
    mmq = (ROOT / "cuda/mmq/mmq.cuh").read_text()
    start = mmq.index("struct block_q8_1_mmq {")
    end = mmq.index("// this struct is used for fp4", start)
    bodies.append(mmq[start:end])
    quantize = (ROOT / "cuda/mmq/quantize.cuh").read_text()
    start = quantize.index("void quantize_mmq_q8_1_cuda(")
    bodies.append(quantize[start:quantize.index(";", start) + 1])
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "glm53_route_prod.inc"
    output.write_text("\n\n".join(bodies) + "\n")
    print(f"{output}: sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


def integer(file, code):
    return struct.unpack("<" + code, file.read(struct.calcsize(code)))[0]


def string(file):
    size = integer(file, "Q")
    return file.read(size).decode()


def value(file, kind):
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    if kind in formats:
        return integer(file, formats[kind])
    if kind == 8:
        return string(file)
    if kind == 9:
        element, count = integer(file, "I"), integer(file, "Q")
        # Tokenizer arrays are metadata only; keep no full token table.
        for _ in range(count):
            value(file, element)
        return None
    raise ValueError("unknown metadata type " + str(kind))


def prepare(path, layer, directory):
    names = [f"blk.{layer}.ffn_{part}_exps.weight" for part in ("gate", "up", "down")]
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
            raise ValueError("unexpected artifact architecture/alignment")
        catalog = {}
        for _ in range(tensors):
            name, ndim = string(file), integer(file, "I")
            dims = [integer(file, "Q") for _ in range(ndim)]
            kind, offset = integer(file, "I"), integer(file, "Q")
            if name in names:
                catalog[name] = {"name": name, "dims": dims, "type": kind, "relative_offset": offset}
        data_start = (file.tell() + alignment - 1) // alignment * alignment
        records = [catalog[name] for name in names]
        edge = layer in (3, 4, 5, 43, 44, 45)
        expected_types = [17, 17, 10] if edge else [16, 16, 17]
        expected_dims = [[4096, 2048, 288], [4096, 2048, 288], [2048, 4096, 288]]
        if [r["type"] for r in records] != expected_types or [r["dims"] for r in records] != expected_dims:
            raise ValueError("unexpected GLM recipe/shape")
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / "experts.bin"
        copied = 0
        with output.open("xb") as dest:
            for record in records:
                rows, columns, _ = record["dims"]
                unit = (rows // 256) * columns * BLOCK[record["type"]]
                absolute = data_start + record["relative_offset"]
                record.update(absolute_offset=absolute, expert_bytes=unit, fixture_offset=copied)
                for expert in IDS:
                    offset = absolute + expert * unit
                    if offset > before.st_size or unit > before.st_size - offset:
                        raise ValueError("tensor outside file")
                    raw = os.pread(file.fileno(), unit, offset)
                    if len(raw) != unit:
                        raise ValueError("truncated expert")
                    dest.write(raw)
                    copied += unit
        after = os.fstat(file.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("source changed during extraction")
        receipt = {"source": str(path.resolve()), "source_stat": identity(before),
                   "layer": layer, "experts": IDS, "tensors": records,
                   "fixture_bytes": copied, "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                   "scope": "selected original GGUF bytes; full source SHA not re-read"}
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, choices=[3, 6, 45])
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    emit(args.emit)
    if args.model:
        if args.layer is None or args.fixture is None:
            parser.error("--model needs --layer and --fixture")
        prepare(args.model, args.layer, args.fixture)


if __name__ == "__main__":
    main()
