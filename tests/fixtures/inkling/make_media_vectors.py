#!/usr/bin/env python3
"""Extract only MQ85GB media weights; generate independent CPU references."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct

import torch
from torch.nn import functional as F

ALIGN = 4096
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
NAMES = [f"model.visual.layers.linear_{i}.weight" for i in range(4)]
NAMES += [f"model.visual.layers.norm_{i}.weight" for i in range(3)]
NAMES += ["model.visual.final_norm.weight", "model.audio.encoder.weight",
          "model.audio.final_norm.weight"]
IMAGE_ROWS, AUDIO_ROWS = 3, 5


def number(fp, fmt):
    return struct.unpack(fmt, fp.read(struct.calcsize(fmt)))[0]


def string(fp):
    return fp.read(number(fp, "<Q")).decode()


def skip(fp, kind):
    if kind in SIZES:
        fp.seek(SIZES[kind], 1)
    elif kind == 8:
        fp.seek(number(fp, "<Q"), 1)
    elif kind == 9:
        sub, count = number(fp, "<I"), number(fp, "<Q")
        if sub in SIZES:
            fp.seek(count * SIZES[sub], 1)
        else:
            for _ in range(count):
                skip(fp, sub)
    else:
        raise ValueError(f"unknown GGUF metadata type {kind}")


def extract(model_dir):
    tensors, evidence = {}, {}
    for path in sorted(model_dir.glob("*.gguf")):
        with path.open("rb") as fp:
            assert fp.read(4) == b"GGUF" and number(fp, "<I") == 3
            count, meta = number(fp, "<Q"), number(fp, "<Q")
            alignment = 32
            for _ in range(meta):
                key, kind = string(fp), number(fp, "<I")
                if key == "general.alignment":
                    assert kind == 4
                    alignment = number(fp, "<I")
                else:
                    skip(fp, kind)
            selected = []
            for _ in range(count):
                name, rank = string(fp), number(fp, "<I")
                dims = [number(fp, "<Q") for _ in range(rank)]
                kind, offset = number(fp, "<I"), number(fp, "<Q")
                if name in NAMES:
                    assert kind == 30 and name not in tensors
                    selected.append((name, dims, offset))
            base = (fp.tell() + alignment - 1) // alignment * alignment
            for name, dims, offset in selected:
                fp.seek(base + offset)
                raw = fp.read(math.prod(dims) * 2)
                assert len(raw) == math.prod(dims) * 2
                tensors[name] = (dims, raw)
                evidence[name] = {"shard": path.name, "offset": base + offset,
                                  "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    assert set(tensors) == set(NAMES)
    return tensors, evidence


def bf(value):
    return value.to(torch.bfloat16).float()


def norm(value, weight):
    # SGLang CUDA RMSNorm: FP32 multiply by weight BEFORE the BF16 store.
    return bf(value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * weight)


def write_tensor(path, value, integer=False):
    values = array.array("i" if integer else "f", value.flatten().tolist())
    assert values.itemsize == 4
    path.write_bytes(values.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tensors, evidence = extract(args.model_dir)
    weights = []
    with (args.output / "weights.bin").open("wb") as fp:
        fp.write(b"INKMEDIA" + struct.pack("<II", 1, len(NAMES)))
        fp.truncate(ALIGN)
        offset = ALIGN
        for i, name in enumerate(NAMES):
            dims, raw = tensors[name]
            fp.seek(16 + i * 32)
            fp.write(struct.pack("<QQQQ", offset, dims[0], dims[1] if len(dims) > 1 else 1, len(raw)))
            fp.seek(offset)
            fp.write(raw)
            offset = (fp.tell() + ALIGN - 1) // ALIGN * ALIGN
            weight = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
            weights.append(weight.reshape(tuple(reversed(dims))))
        fp.truncate(offset)

    x = ((torch.arange(IMAGE_ROWS * 2 * 40 * 40 * 3) * 37 % 1021) - 510).float() / 255
    x = x.reshape(IMAGE_ROWS, 2, 40, 40, 3)
    write_tensor(args.output / "pixels.f32", x)
    x = bf(x)
    for i, (tf, hf) in enumerate([(1, 5), (1, 2), (1, 4), (2, 1)]):
        b, t, h, w, c = x.shape
        x = x.reshape(b, t // tf, tf, h // hf, hf, w // hf, hf, c)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(b, t // tf, h // hf, w // hf, -1)
        # FP64 GEMM oracle isolates equation/rounding from CUDA reduction order.
        x = bf(F.linear(x.double(), weights[i].double()))
        x = norm(x, weights[4 + i])
        if i < 3:
            x = bf(F.gelu(x))
        write_tensor(args.output / f"image-stage-{i}.f32", x)

    ids = (torch.arange(AUDIO_ROWS * 80).reshape(AUDIO_ROWS, 80) * 7 + 3) % 16
    write_tensor(args.output / "audio.i32", ids, integer=True)
    embedded = weights[8][ids + torch.arange(80) * 16]
    x = bf(embedded.double().sum(-2))
    write_tensor(args.output / "audio-sum.f32", x)
    write_tensor(args.output / "audio-output.f32", norm(x, weights[9]))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
             if p.suffix in (".bin", ".f32", ".i32")}
    (args.output / "reference.json").write_text(json.dumps({
        "torch": torch.__version__, "image_rows": IMAGE_ROWS, "audio_rows": AUDIO_ROWS,
        "source_revision": "8cc5877b44d343f88b92086aa1fb72897950f06a",
        "sglang_revision": "03d06a764e4a83268eefd1bafc676418f7269c89",
        "scope": "SGLang CUDA BF16 equations; CPU FP64 matmul/sum oracle; no preprocessing or full-model parity",
        "weights": evidence, "files": files,
    }, indent=2) + "\n")
    print(f"Generated 3 image patches and 5 audio frames in {args.output}")


if __name__ == "__main__":
    main()
