"""Reference logits with the port's rotary pair counts, for the P6b bisect.

The port's graph rotates 64 pairs on sliding layers and 128 on full layers
(ds4_gemma4_plan.h G4_ROT/G4_GLOBAL_ROT). The reference rotates 128 pairs on
sliding layers (rope_type default, head_dim 256) and 64 on full layers
(proportional, partial_rotary_factor 0.25 of 512) -- see
modeling_rope_utils.py:193-265. This script runs the reference as shipped and
then with the port's inverse-frequency tables, so the two logits dumps can be
compared against the port.

Usage:
  python ref_rope_variant.py --model DIR --ids-file FILE \
      --out-true FILE.npy --out-port FILE.npy
"""

import argparse

import numpy as np
import torch
from transformers import Gemma4ForConditionalGeneration


def read_ids(path):
    with open(path, "r", encoding="utf-8") as fp:
        return [int(part) for part in fp.read().strip().split(",") if part.strip()]


def find_rope_modules(module):
    return [
        child
        for child in module.modules()
        if any(name.endswith("_inv_freq") for name, _ in child.named_buffers())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--out-true", required=True)
    parser.add_argument("--out-port", required=True)
    args = parser.parse_args()

    ids = read_ids(args.ids_file)
    torch.set_grad_enabled(False)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.eval()
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")

    def forward():
        out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        return out.logits[0, -1].to(torch.float32).cpu().numpy()

    true_logits = forward()
    np.save(args.out_true, true_logits)

    modules = find_rope_modules(model)
    if not modules:
        raise SystemExit("no rotary module with *_inv_freq buffers")
    for rope in modules:
        for name, buf in list(rope.named_buffers()):
            if name.endswith("_inv_freq"):
                print(f"{type(rope).__name__}.{name}: shape={tuple(buf.shape)} "
                      f"nonzero={int((buf != 0).sum())} head={buf[:3].tolist()}")

    port_logits = None
    for rope in modules:
        for layer_type, pairs in (("sliding_attention", 64), ("full_attention", 128)):
            name = f"{layer_type}_inv_freq"
            if name not in dict(rope.named_buffers()):
                continue
            ref = dict(rope.named_buffers())[name]
            dim = ref.shape[0] * 2
            theta = 10000.0 if layer_type == "sliding_attention" else 1000000.0
            table = np.zeros(ref.shape[0], dtype=np.float32)
            for i in range(min(pairs, ref.shape[0])):
                table[i] = 1.0 / (theta ** (2.0 * i / dim))
            ref.copy_(torch.from_numpy(table).to(ref.device))
            print(f"patched {name}: nonzero={int((ref != 0).sum())}")
    port_logits = forward()
    np.save(args.out_port, port_logits)

    for tag, logits in (("true", true_logits), ("port-rope", port_logits)):
        top = np.argsort(-logits)[:5]
        print(f"{tag}: argmax={int(top[0])} top5=" +
              " ".join(f"{int(i)}:{logits[i]:.2f}" for i in top))


if __name__ == "__main__":
    main()
