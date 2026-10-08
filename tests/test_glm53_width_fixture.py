#!/usr/bin/env python3
"""Extract layer-0 projection bytes and recorded embeddings, without a model load.

Pass the mixed GGUF, a GLM session payload, and a fresh fixture directory.
Compile test_glm53_width.cu with that directory on the include path. Its
arguments are weights.bin, input.f32, width (1/128/2048), and output prefix.
DS4_WIDTH_FIXED_NORM may name a width-1 .norm.f32 capture for decomposition.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_glm53_route_fixture import integer, string, value

F32, Q8_0, Q4_K, BF16 = 0, 8, 12, 30
HIDDEN, HEAD_INPUT, HC, MIX, BASE_ROWS = 4096, 8192, 4, 24, 128
VOCAB, ALIGNMENT = 154880, 32
Q4_BLOCK, Q4_BYTES, Q8_BLOCK, Q8_BYTES = 256, 144, 32, 34
PAYLOAD_MAGIC, PAYLOAD_VERSION = 0x34565344, 3
PAYLOAD_HEADER = struct.Struct("<13I")
PAYLOAD_TOKENS = struct.Struct(f"<{BASE_ROWS}I")

SPECS = {
    "hc_attn_fn": ([HIDDEN * HC, MIX], BF16, HIDDEN * HC * MIX * 2),
    "hc_attn_base": ([MIX], F32, MIX * 4),
    "hc_attn_scale": ([3], F32, 3 * 4),
    "attn_norm": ([HIDDEN], F32, HIDDEN * 4),
    "kda_q": ([HIDDEN, HEAD_INPUT], Q4_K, HIDDEN // Q4_BLOCK * Q4_BYTES * HEAD_INPUT),
    "kda_k": ([HIDDEN, HEAD_INPUT], Q4_K, HIDDEN // Q4_BLOCK * Q4_BYTES * HEAD_INPUT),
    "kda_v": ([HIDDEN, HEAD_INPUT], Q8_0, HIDDEN // Q8_BLOCK * Q8_BYTES * HEAD_INPUT),
}


def identity(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.payload.open("rb") as payload:
        header = PAYLOAD_HEADER.unpack(payload.read(PAYLOAD_HEADER.size))
        if header[0] != PAYLOAD_MAGIC or header[1] != PAYLOAD_VERSION or header[7] < BASE_ROWS:
            raise ValueError("expected recorded GLM payload with at least 128 tokens")
        tokens = PAYLOAD_TOKENS.unpack(payload.read(PAYLOAD_TOKENS.size))
        if max(tokens) >= VOCAB:
            raise ValueError("prompt token exceeds vocabulary")
    records = {}
    with args.model.open("rb") as source:
        before = os.fstat(source.fileno())
        if source.read(4) != b"GGUF" or integer(source, "I") not in (2, 3):
            raise ValueError("expected GGUF")
        count, meta = integer(source, "Q"), integer(source, "Q")
        alignment, arch = ALIGNMENT, None
        for _ in range(meta):
            key, kind = string(source), integer(source, "I")
            data = value(source, kind)
            if key == "general.alignment":
                alignment = data
            elif key == "general.architecture":
                arch = data
        if alignment != ALIGNMENT or arch != "glm5-next":
            raise ValueError("unexpected GLM architecture or alignment")
        embeddings = None
        for _ in range(count):
            name = string(source)
            dims = [integer(source, "Q") for _ in range(integer(source, "I"))]
            kind, offset = integer(source, "I"), integer(source, "Q")
            record = dict(name=name, dims=dims, type=kind, relative_offset=offset)
            if name == "token_embd.weight":
                embeddings = record
            for key in SPECS:
                if name == f"blk.0.{key}.weight":
                    records[key] = record
        start = (source.tell() + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        cursor = 0
        with (args.output / "weights.bin").open("xb") as weights:
            for key, (dims, kind, size) in SPECS.items():
                record = records[key]
                if record["dims"] != dims or record["type"] != kind:
                    raise ValueError("unexpected recipe: " + key)
                offset = start + record["relative_offset"]
                raw = os.pread(source.fileno(), size, offset)
                if len(raw) != size:
                    raise ValueError("truncated " + key)
                record.update(offset=cursor, source_offset=offset, bytes=size,
                              sha256=hashlib.sha256(raw).hexdigest())
                weights.write(raw)
                cursor += size
                padding = (-cursor) % ALIGNMENT
                weights.write(bytes(padding))
                cursor += padding
        if not embeddings or embeddings["dims"] != [HIDDEN, VOCAB] or embeddings["type"] != Q8_0:
            raise ValueError("unexpected embeddings")
        embedding_bytes = HIDDEN // Q8_BLOCK * Q8_BYTES
        with (args.output / "input.f32").open("xb") as dest:
            for token in tokens:
                raw = os.pread(source.fileno(), embedding_bytes,
                    start + embeddings["relative_offset"] + token * embedding_bytes)
                if len(raw) != embedding_bytes:
                    raise ValueError("truncated embedding")
                row = []
                for block in range(HIDDEN // Q8_BLOCK):
                    offset = block * Q8_BYTES
                    scale = struct.unpack_from("<e", raw, offset)[0]
                    codes = struct.unpack_from(f"<{Q8_BLOCK}b", raw, offset + 2)
                    row.extend(scale * code for code in codes)
                dest.write(struct.pack(f"<{HIDDEN}f", *row))
        if identity(before) != identity(os.fstat(source.fileno())):
            raise ValueError("source changed")
    includes = "\n".join(f"static constexpr uint64_t W_{key.upper()} = UINT64_C({r['offset']});"
        for key, r in records.items())
    (args.output / "weights.h").write_text(includes + "\n")
    receipt = dict(model=str(args.model.resolve()), source_stat=identity(before),
        payload=str(args.payload.resolve()), payload_sha256=hashlib.sha256(args.payload.read_bytes()).hexdigest(),
        tensors=records, tokens=tokens, weight_bytes=cursor,
        input_sha256=hashlib.sha256((args.output / "input.f32").read_bytes()).hexdigest(),
        scope="actual first 128 prompt embedding rows and layer-0 projection weights; no engine/model load")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), weight_bytes=cursor, tokens=len(tokens))))


if __name__ == "__main__":
    main()
