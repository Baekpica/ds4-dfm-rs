#!/usr/bin/env python3
"""Write the checkpoint's per-stage hidden states for the parity walk.

  python make_hidden_reference.py <hf-dir> OUTDIR

Each file is the LAST row of the stage as float32, named like the engine's
DS4_SPARK2_5_DUMP_HIDDEN output:

  embed.f32       input embedding (model.embedding)
  layer1..36.f32  the residual stream after each decoder layer
  final.f32       model.norm output rounded to BF16, then widened
                  (the head's operand; modeling_spark.py:404-410)

The token ids are the same five raw ids the logits fixture pins.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROMPT = "The capital of France is"


def main():
    model_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((Path(__file__).parent / "logits-reference.json").read_text())
    ids = fixture["token_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    captured = {}

    def capture(name):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor[0, -1].float().numpy()

        return hook

    model.model.embedding.register_forward_hook(capture("embed"))
    for il, layer in enumerate(model.model.layers):
        layer.register_forward_hook(capture(f"layer{il + 1}"))
        layer.input_layernorm.register_forward_hook(capture(f"norm_attn{il}"))
        layer.post_attention_layernorm.register_forward_hook(capture(f"norm_ffn{il}"))
    model.model.norm.register_forward_hook(capture("final_raw"))

    with torch.no_grad():
        model(torch.tensor([ids], dtype=torch.long), use_cache=False)

    if len(captured) != 2 + 36 * 3:
        raise SystemExit(f"expected 110 captures, got {len(captured)}")
    for name, value in captured.items():
        if name == "final_raw":
            # The head's operand: the norm output cast to the embedding dtype
            # (modeling_spark.py:409-412), then widened for the comparison.
            wide = torch.from_numpy(value).to(torch.bfloat16).float().numpy()
            wide.tofile(out_dir / "final.f32")
            continue
        value.tofile(out_dir / f"{name}.f32")
    print(f"wrote {len(captured)} stages to {out_dir}")


if __name__ == "__main__":
    main()
