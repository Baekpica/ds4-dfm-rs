#!/usr/bin/env python3
"""CPU equation oracle for the three actual Step Q8 predictor blocks.

Use gguf-py from the pinned StepFun llama.cpp checkout. Equations follow the
handoff's pinned vLLM step3p5_mtp.py; GGUF norms already include the +1 offset.
This isolates predictor arithmetic from target trajectories and speculation.
"""
import argparse
import hashlib
import json

from pathlib import Path

import gguf
import numpy as np
import torch
from torch.nn import functional as F

ROWS, HIDDEN, DEPTHS, FIRST_LAYER = 7, 4096, 3, 45
HEADS, KV_HEADS, HEAD = 96, 8, 128
EPSILON = 1e-5


def norm(x, weight):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPSILON) * weight


def rope(x):
    freq = torch.pow(10000.0, -torch.arange(HEAD // 2, dtype=torch.float64) * 2 / HEAD).float()
    angle = torch.arange(ROWS).float()[:, None] * freq
    co, si = angle.cos()[:, None], angle.sin()[:, None]
    a, b = x.chunk(2, -1)
    return torch.cat((a * co - b * si, b * co + a * si), -1)


def q8_input(x):
    # Pinned GGUF MMVQ Q8_1: independent 32-value activation blocks,
    # roundf (ties away from zero), and an F16 stored scale. The Q8_0
    # weight dot does not use Q8_1's sum field.
    blocks = x.reshape(-1, 32)
    scale = blocks.abs().amax(-1, keepdim=True) / 127
    units = blocks / torch.where(scale == 0, 1, scale)
    quant = units.sign() * torch.floor(units.abs() + 0.5)
    return (quant * scale.half().float()).reshape_as(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--activation", choices=("fp32", "q8_1"), default="fp32")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    reader = gguf.GGUFReader(str(args.sidecar))
    tensors = {t.name: t for t in reader.tensors}
    assert len(tensors) == 55
    weights = {}

    def load(name):
        t = tensors[name]
        assert t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q8_0)
        weights[name] = hashlib.sha256(t.data).hexdigest()
        array = gguf.dequantize(t.data, t.tensor_type).reshape(tuple(reversed(t.shape)))
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

    def linear(x, name):
        # Independent FP64 accumulation isolates the selected activation
        # contract from CUDA reduction order.
        weight = load(name)
        if args.activation == "q8_1":
            x = q8_input(x)
        return F.linear(x.double(), weight.double()).float()

    def save(name, tensor):
        tensor.detach().numpy().astype('<f4').tofile(args.output / name)

    grid = torch.arange(ROWS * HIDDEN).reshape(ROWS, HIDDEN)
    hidden = ((grid * 37 % 1021) - 510).float() / 255
    embedding = ((grid * 29 % 509) - 254).float() / 127
    save("hidden.f32", hidden)
    save("embeddings.f32", embedding)
    for depth in range(DEPTHS):
        prefix = f"blk.{FIRST_LAYER + depth}."
        def trace(name, tensor):
            save(f"{name}-{FIRST_LAYER + depth}.bin", tensor)

        x = linear(torch.cat((norm(embedding, load(prefix + "nextn.enorm.weight")),
                              norm(hidden, load(prefix + "nextn.hnorm.weight"))), -1),
                   prefix + "nextn.eh_proj.weight")
        trace("attn_norm_in", x)
        a = norm(x, load(prefix + "attn_norm.weight"))
        trace("attn_norm", a)
        q = rope(norm(linear(a, prefix + "attn_q.weight").reshape(ROWS, HEADS, HEAD),
                      load(prefix + "attn_q_norm.weight")))
        k = rope(norm(linear(a, prefix + "attn_k.weight").reshape(ROWS, KV_HEADS, HEAD),
                      load(prefix + "attn_k_norm.weight")))
        v = linear(a, prefix + "attn_v.weight").reshape(ROWS, KV_HEADS, HEAD)
        trace("Qcur_pos", q)
        trace("Kcur_pos", k)
        trace("Vcur", v)
        k, v = k.half().float(), v.half().float()
        heads = []
        for row in range(ROWS):
            keys = k[:row + 1].repeat_interleave(HEADS // KV_HEADS, 1)
            values = v[:row + 1].repeat_interleave(HEADS // KV_HEADS, 1)
            score = torch.einsum('hd,khd->hk', q[row].double(), keys.double()) / HEAD**0.5
            prob = torch.softmax(score.float(), -1)
            heads.append(torch.einsum('hk,khd->hd', prob.double(), values.double()).float())
        gate = linear(a, prefix + "attn_gate.weight").sigmoid()
        trace("attn_out", torch.stack(heads))
        attention = (torch.stack(heads) * gate[:, :, None]).reshape(ROWS, HEADS * HEAD)
        trace("attn_gated", attention)
        projection = linear(attention, prefix + "attn_output.weight")
        trace("attn_proj", projection)
        x = x + projection
        trace("ffn_inp", x)
        a = norm(x, load(prefix + "ffn_norm.weight"))
        trace("ffn_norm", a)
        middle = F.silu(linear(a, prefix + "ffn_gate.weight")) * linear(a, prefix + "ffn_up.weight")
        ffn = linear(middle, prefix + "ffn_down.weight")
        trace("ffn_out", ffn)
        hidden = x + ffn
        save(f"depth-{depth}.f32", hidden)
        logits = linear(norm(hidden[-1:], load(prefix + "nextn.shared_head_norm.weight")),
                        prefix + "nextn.shared_head_head.weight")
        save(f"logits-{depth}.f32", logits)
        print(f"depth {depth}: argmax {logits.argmax().item()}", flush=True)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob('*.f32'))}
    (args.output / "reference.json").write_text(json.dumps({
        "scope": "Q8 weights dequantized, CPU FP64 linear/attention accumulation, F16 KV; synthetic inputs",
        "activation": args.activation,
        "source": "Step handoff vLLM 0.24.0 model_executor/models/step3p5_mtp.py",
        "source_sha256": "9290cc54b17188817026a58a7fec3f8b385e5c18c12fc0388f5ee6a87570368d",
        "gguf_revision": "0b69336d2fd2adfdef9c66e425f7778196c31482",
        "torch": torch.__version__, "rows": ROWS, "weights": weights, "files": files,
    }, indent=2) + '\n')


if __name__ == "__main__":
    main()
