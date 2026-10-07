#!/usr/bin/env python3
"""Extract the real dense Q8 dispatcher and bounded block0 artifact weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from test_glm53_route_fixture import ROOT, integer, string, value
from test_glm53_upload import function

SHAPES = {
    "blk.0.kda_f_b.weight": [128, 8192],
    "blk.0.kda_g_b.weight": [128, 8192],
    "blk.0.kda_output.weight": [8192, 4096],
    "blk.0.ffn_gate.weight": [4096, 12288],
    "blk.0.ffn_up.weight": [4096, 12288],
    "blk.0.ffn_down.weight": [12288, 4096],
}


def emit(directory):
    cuda = (ROOT / "ds4_cuda.cu").read_text()
    repack = (ROOT / "cuda/mmq/ds4_repack.cu").read_text()
    types = []
    for name in ("ds4_gpu_tensor", "dense_graph_key", "dense_graph_entry"):
        start = cuda.index("struct " + name + " {")
        types.append(cuda[start:cuda.index("};", start) + 2])
    bodies = [function(repack, "repack_q8_0_aligned_kernel").replace(
        '"ds4_cuda.cu"', '"cuda/mmq/ds4_repack.cu"')]
    for name in ("f32_to_f16_kernel", "warp_sum_f32", "load_i8x4_i32_aligned",
                 "load_i8x4_i32_unaligned", "dot_i8x32_dp4a", "dot_i8_block",
                 "quantize_q8_0_f32_kernel", "matmul_q8_0_preq_kernel",
                 "matmul_q8_0_preq_warp8_kernel", "matmul_q8_0_preq_n2_warp8_kernel"):
        bodies.append(function(cuda, name))
    bodies.append("template <uint32_t group_width>\n" +
                  function(cuda, "matmul_q8_0_preq_batch_warp8_kernel"))
    bodies.append(function(cuda, "cuda_matmul_q8_0_tensor_labeled_impl"))
    directory.mkdir(parents=True, exist_ok=True)
    for name, parts in (("glm53_dense_types.inc", types), ("glm53_dense_prod.inc", bodies)):
        output = directory / name
        output.write_text("\n\n".join(parts) + "\n")
        print(f"{output}: sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


def prepare(path, directory):
    with path.open("rb") as source:
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
        catalog = {}
        for _ in range(tensors):
            name = string(source)
            dims = [integer(source, "Q") for _ in range(integer(source, "I"))]
            kind, offset = integer(source, "I"), integer(source, "Q")
            if name in SHAPES:
                catalog[name] = dict(name=name, dims=dims, type=kind, relative_offset=offset)
        base = (source.tell() + alignment - 1) // alignment * alignment
        directory.mkdir(parents=True, exist_ok=False)
        output, records = directory / "weights.bin", []
        with output.open("xb") as dest:
            dest.write(b"GLMDQ8\x00\x01" + struct.pack("<I", len(SHAPES)))
            for name, shape in SHAPES.items():
                record = catalog[name]
                if record["dims"] != shape or record["type"] != 8:
                    raise ValueError("unexpected dense shape/type")
                columns, rows = shape
                size = columns // 32 * rows * 34
                offset = base + record["relative_offset"]
                if offset > before.st_size or size > before.st_size - offset:
                    raise ValueError("tensor outside file")
                encoded = name.encode()
                dest.write(struct.pack("<IIIIQ", len(encoded), columns, rows, 8, size) + encoded)
                record.update(absolute_offset=offset, tensor_bytes=size, fixture_offset=dest.tell())
                raw = os.pread(source.fileno(), size, offset)
                if len(raw) != size:
                    raise ValueError("truncated dense tensor")
                dest.write(raw)
                records.append(record)
        after = os.fstat(source.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("source changed")
        receipt = dict(source=str(path.resolve()), source_stat=identity(before), tensors=records,
                       fixture_bytes=output.stat().st_size,
                       fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                       scope="six actual block0 Q8 tensors; full model SHA not re-read")
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    emit(args.emit)
    if args.model:
        if not args.fixture:
            parser.error("--model needs --fixture")
        prepare(args.model, args.fixture)


if __name__ == "__main__":
    main()
