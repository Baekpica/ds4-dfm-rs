#!/usr/bin/env python3
"""CPU equation oracle for all eight actual BF16 MTP blocks, without base weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from make_media_vectors import bf, norm, number, skip, string, write_tensor

ROWS, HIDDEN, DEPTHS = 7, 4096, 8


def tensor_index(path):
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
        index = {}
        for _ in range(count):
            name, rank = string(fp), number(fp, "<I")
            dims = [number(fp, "<Q") for _ in range(rank)]
            kind, offset = number(fp, "<I"), number(fp, "<Q")
            assert kind == 30 and name not in index
            index[name] = (tuple(reversed(dims)), offset)
        base = (fp.tell() + alignment - 1) // alignment * alignment
    assert len(index) == 160
    return {name: (dims, base + offset) for name, (dims, offset) in index.items()}


def convolution(x, weight):
    # SGLang residual sconv accumulates in FP32 and rounds the sum once.
    value = torch.zeros_like(x)
    for tap in range(4):
        delay = 3 - tap
        if delay < len(x):
            value[delay:] += x[:len(x) - delay] * weight[:, 0, tap]
    return bf(value + x)


def block(hidden, embeddings, w):
    def linear(x, name):
        return bf(F.linear(x.double(), w[name].double()))

    x = linear(torch.cat((norm(hidden, w["hidden_norm.weight"]),
                          norm(embeddings, w["embed_norm.weight"])), -1), "input_proj.weight")
    p = "transformer_block."
    a = p + "attn."
    normalized = norm(x, w[p + "attn_norm.weight"])
    q = norm(linear(normalized, a + "wq_du.weight").reshape(ROWS, 32, 128), w[a + "q_norm.weight"])
    k = convolution(linear(normalized, a + "wk_dv.weight"), w[a + "k_sconv.weight"])
    k = norm(k.reshape(ROWS, 8, 128), w[a + "k_norm.weight"])
    v = convolution(linear(normalized, a + "wv_dv.weight"), w[a + "v_sconv.weight"]).reshape(ROWS, 8, 128)
    r = linear(normalized, a + "wr_du.weight").reshape(ROWS, 32, 16)
    relative = bf(r.double() @ w[a + "rel_logits_proj.proj"].double())
    # Seven positions are below the global tau floor and both local extents.
    # Match SGLang attention's FP32 softmax/value accumulation, BF16 output.
    heads = []
    for row in range(ROWS):
        keys = k[:row + 1].repeat_interleave(4, 1)
        values = v[:row + 1].repeat_interleave(4, 1)
        score = torch.einsum("hd,khd->hk", q[row].double(), keys.double()) / 128
        score += relative[row, :, torch.arange(row, -1, -1)].double()
        prob = torch.softmax(score.float(), -1)
        heads.append(bf(torch.einsum("hk,khd->hd", prob.double(), values.double())))
    attention = linear(torch.stack(heads).reshape(ROWS, HIDDEN), a + "wo_ud.weight")
    x = bf(x + convolution(attention, w[p + "attn_sconv.weight"]))
    normalized = norm(x, w[p + "mlp_norm.weight"])
    pairs = linear(normalized, p + "mlp.w13_dn.weight")
    middle = bf(F.silu(pairs[:, ::2]) * pairs[:, 1::2])
    mlp = bf(linear(middle, p + "mlp.w2_md.weight") * w[p + "mlp.global_scale"])
    return bf(x + convolution(mlp, w[p + "mlp_sconv.weight"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    index = tensor_index(args.sidecar)
    grid = torch.arange(ROWS * HIDDEN).reshape(ROWS, HIDDEN)
    hidden = bf(((grid * 37 % 1021) - 510).float() / 255)
    embeddings = bf(((grid * 29 % 509) - 254).float() / 127)
    write_tensor(args.output / "hidden.f32", hidden)
    write_tensor(args.output / "embeddings.f32", embeddings)
    evidence = {}
    with args.sidecar.open("rb") as fp:
        for depth in range(DEPTHS):
            prefix = f"model.mtp.layers.{depth}."
            weights = {}
            for name, (dims, offset) in index.items():
                if not name.startswith(prefix):
                    continue
                fp.seek(offset)
                raw = fp.read(math.prod(dims) * 2)
                assert len(raw) == math.prod(dims) * 2
                evidence[name] = {"offset": offset, "sha256": hashlib.sha256(raw).hexdigest()}
                weights[name.removeprefix(prefix)] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().reshape(dims)
            hidden = block(hidden, embeddings, weights)
            write_tensor(args.output / f"depth-{depth}.f32", hidden)
            print(f"Generated depth {depth}", flush=True)
    names = ["hidden.f32", "embeddings.f32"] + [f"depth-{i}.f32" for i in range(DEPTHS)]
    files = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in names}
    (args.output / "reference.json").write_text(json.dumps({
        "torch": torch.__version__, "rows": ROWS,
        "sglang_revision": "03d06a764e4a83268eefd1bafc676418f7269c89",
        "mtp_revision": "01e829c5acf2c9aa8026f5b8157d980e1c20730c",
        "scope": "BF16 MTP block equations, CPU FP64 matmul; supplied main-normalized embeddings and hidden; no shared embed/head or speculation",
        "weights": evidence, "files": files,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
