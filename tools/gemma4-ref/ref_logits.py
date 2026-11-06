"""Reference last-position logits for the Gemma-4 26B A4B gate.

Runs the pinned Hugging Face implementation (transformers commit
cc1451df3b5b218295bc16803569256c53c9d6e2, modeling_gemma4.py sha256
257dc8a7dc36c6d331351e9f0d8f0903bbe69749d2b9327c5cc9942bec0deafe) on the
checkpoint and writes the next-token logits as float32 .npy. The caller
supplies token ids, so no tokenizer or chat template takes part: the same ids
go to the port through its rendered prompt.

Usage:
  python ref_logits.py --model DIR \
      --ids-file FILE --out FILE.npy [--ids-file FILE --out FILE.npy ...]

where each ids file holds the ids comma separated on one line.
"""

import argparse

import numpy as np
import torch
from transformers import Gemma4ForConditionalGeneration


def read_ids(path):
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read().strip()
    return [int(part) for part in text.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids-file", action="append", required=True)
    parser.add_argument("--out", action="append", required=True)
    parser.add_argument("--attn", default="sdpa")
    args = parser.parse_args()
    if len(args.ids_file) != len(args.out):
        parser.error("--ids-file and --out must pair up")

    torch.set_grad_enabled(False)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=args.attn,
    )
    model.eval()

    for ids_path, out_path in zip(args.ids_file, args.out):
        ids = read_ids(ids_path)
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
        np.save(out_path, logits)
        top = np.argsort(-logits)[:10]
        finite = int(np.isfinite(logits).sum())
        print(
            f"{out_path}: tokens={len(ids)} vocab={logits.shape[0]} "
            f"finite={finite} argmax={int(top[0])} logit={float(logits[top[0]]):.4f}"
        )
        print("  top10:", " ".join(f"{int(i)}:{float(logits[i]):.3f}" for i in top))


if __name__ == "__main__":
    main()
