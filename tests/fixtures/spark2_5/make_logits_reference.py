#!/usr/bin/env python3
"""Regenerate the Spark-X2.6 forward fixture from the checkpoint.

The graph's gate compares top-k ids and logits against this file, so it must
come from the reference implementation, never from this tree's Rust/native
code:

  python make_logits_reference.py <hf-dir> \
      logits-reference.json

The environment is recorded in provenance.json. The model is loaded in F32 on
the CPU: the checkpoint stores BF16, widening it is exact, and F32 keeps the
4B model inside a 31 GiB host. The last position's logits are the fixture.

The prompt is fixed ASCII; its token ids come from the checkpoint's own
tokenizer, so the fixture pins what the graph must consume.
"""

import hashlib
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"
TOP_K = 64


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    model_dir = sys.argv[1]
    out_path = sys.argv[2]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = tokenizer(PROMPT, add_special_tokens=True)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long), use_cache=False)
    logits = out.logits[0, -1].float()
    values, indices = torch.topk(logits, TOP_K)
    fixture = {
        "prompt": PROMPT,
        "token_ids": ids,
        "positions": list(range(len(ids))),
        "dtype": "float32",
        "device": "cpu",
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "config_sha256": sha256(f"{model_dir}/config.json"),
        "modeling_sha256": sha256(f"{model_dir}/modeling_spark.py"),
        "argmax": int(indices[0]),
        "top_k": [
            {"id": int(i), "logit": float(v)}
            for i, v in zip(indices.tolist(), values.tolist())
        ],
        "logit_sum": float(logits.sum()),
        "logit_abs_sum": float(logits.abs().sum()),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_path}: {len(ids)} tokens, argmax {fixture['argmax']}")


if __name__ == "__main__":
    main()
