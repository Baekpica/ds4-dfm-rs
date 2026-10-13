#!/usr/bin/env python3
"""Vision-tower golden fixtures for the ds41 port (fixture generator only).

Ported from the engine's tools/vision_golden.py (YoungAi 846c929c5) and
translated to English; the logic is unchanged.  It runs the official
inference/{image_processor,vision}.py code from an original HF directory on
the BF16 original weights for one image, and writes the preprocessing result
and the aligner rows in the VGLD format, for ./ds4 --image <img>
--vision-golden <golden> to compare segment by segment (the engine's
src/core/core_v41_vision_gate.c; the port's reader is
crates/ds4-core/src/v41_golden.rs).

Fixture generation only, outside any numeric chain: Python is allowed for
external collection and golden generation, never for shipped logic.

    python3 tools/vision_golden.py --hf <hf dir> --image <img> --out <golden.bin>

--hf points at the original HF directory (config.json, inference/, and the
model-0000N-of-000M.safetensors shards that the vision./aligner. tensors live
in).  --dtype f32 runs the same official code all the way in f32 (no bf16
rounding): compared against the bf16 golden it measures the official code's
own bf16 noise, which is the yardstick the gate's thresholds come from
(core_v41_vision_gate.c:39-41).

File format (little endian):

    "VGLD" u32 ver=1 | u32 w h n_vit_h n_vit_w n_llm_h n_llm_w pp E best_w best_h
    u8 rgb[best_h][best_w][3] (after the PIL pad)
    f32 patches[n_vit][pp] | f32 embeds[n_llm][E]
"""
import argparse
import json
import os
import struct
import sys
import types

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open


def read_golden(path):
    """Read one VGLD golden (the reader side of the format above)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"VGLD"
        h = struct.unpack("<12I", f.read(48))
        ver, w, h0, vh, vw, lh, lw, pp, E, bw, bh, _ = h
        rgb = np.frombuffer(f.read(bw * bh * 3), dtype=np.uint8)
        pat = np.frombuffer(f.read(vh * vw * pp * 4), dtype="<f4").reshape(vh * vw, pp)
        emb = np.frombuffer(f.read(lh * lw * E * 4), dtype="<f4").reshape(lh * lw, E)
    return rgb, pat, emb


def compare(a, b):
    """Compare two goldens' aligner rows with the gate's own measure
    (core_v41_vision_gate.c:24-38: min row cosine / max absolute
    difference, plus the mean)."""
    _, pa, ea = read_golden(a)
    _, pb, eb = read_golden(b)
    assert ea.shape == eb.shape, (ea.shape, eb.shape)
    cos = (ea * eb).sum(1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-30)
    print(f"[compare] {a} vs {b}: patch max diff {np.abs(pa - pb).max():.5f}; "
          f"embeds {ea.shape} row cos min {cos.min():.6f} (row {cos.argmin()}) "
          f"mean {cos.mean():.6f} median {np.median(cos):.6f}; "
          f"max abs diff {np.abs(ea - eb).max():.5f}; |mean| {np.abs(ea).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", help="original HF directory (config.json / inference/ / model-0000*.safetensors)")
    ap.add_argument("--image")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A.bin", "B.bin"),
                    help="only compare two goldens' aligner rows, no model run")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f32"],
                    help="f32 = the same official code all the way in f32 (no bf16 rounding): "
                         "compared against the bf16 golden it shows how large the official "
                         "code's own bf16 noise is; a deviation inside that band is correct")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if not (a.hf and a.image and a.out):
        ap.error("--hf/--image/--out are all required (or use --compare alone)")

    # The official preprocessing (the same functions the engine mirrors).
    sys.path.insert(0, os.path.join(a.hf, "inference"))
    import image_processor as ip
    from vision import Aligner, ViT

    cfg = json.load(open(os.path.join(a.hf, "config.json")))
    vc = cfg["vision_config"]
    dim = cfg["text_config"]["hidden_size"]  # the top-level config is the HF two-part text/vision form
    args = types.SimpleNamespace(
        dim=dim, vision_n_layers=vc["num_hidden_layers"], vision_dim=vc["hidden_size"],
        vision_n_heads=vc["num_attention_heads"], vision_inter_dim=vc["intermediate_size"],
        vision_patch_size=vc["patch_size"], vision_rope_theta=float(vc["rope_theta"]),
        vision_downsample_ratio=vc["downsample_ratio"], vision_max_n_token=vc["max_image_tokens"],
        vision_min_pixels=vc["min_pixels"], vision_max_wh_ratio=vc.get("max_wh_ratio"),
        image_token_id=cfg["image_token_id"], vision_enabled=True)

    # Preprocessing through the official function; the padded pixels are
    # recomputed with the same few lines load_image uses (same operations,
    # same image).
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"url": a.image}, args)
    with Image.open(a.image) as src:
        img = src.convert("RGB")
    w0, h0 = img.width, img.height
    lh, lw, best_h, best_w = ip.plan_image_grid(img.width, img.height, args)
    assert (lh, lw) == (n_llm_h, n_llm_w)
    padded = ImageOps.pad(img, (best_w, best_h), color=(127, 127, 127))
    rgb = np.asarray(padded, dtype=np.uint8)
    assert rgb.shape == (best_h, best_w, 3)

    # Weights: vision.* / aligner.* live in the shards the
    # model.safetensors.index.json points at (the BF16 originals).
    idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    need = {k: v for k, v in idx.items() if k.startswith(("vision.", "aligner."))}
    sd = {}
    for shard in sorted(set(need.values())):
        with safe_open(os.path.join(a.hf, shard), framework="pt", device="cpu") as f:
            for k, v in need.items():
                if v == shard:
                    sd[k] = f.get_tensor(k)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(dev)  # the official get_vision_cos_sin builds its frequency table on
                                   # the default device; the official generate.py sets it the same way
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    torch.set_default_dtype(dt)
    vit, al = ViT(args).to(dev), Aligner(args).to(dev)
    vit.load_state_dict({k[len("vision."):]: v.to(dt) for k, v in sd.items() if k.startswith("vision.")}, strict=True)
    al.load_state_dict({k[len("aligner."):]: v.to(dt) for k, v in sd.items() if k.startswith("aligner.")}, strict=True)
    with torch.inference_mode():
        emb = al(vit(patches.to(dev).to(dt), n_vit_h, n_vit_w), n_vit_h, n_vit_w)
    emb = emb.float().cpu().numpy()
    pat = patches.float().cpu().numpy().reshape(n_vit_h * n_vit_w, -1)
    assert emb.shape == (n_llm_h * n_llm_w, args.dim), emb.shape

    with open(a.out, "wb") as f:
        f.write(b"VGLD")
        f.write(struct.pack("<12I", 1, w0, h0, n_vit_h, n_vit_w, n_llm_h, n_llm_w,
                            pat.shape[1], args.dim, best_w, best_h, 0))
        f.write(rgb.tobytes())
        f.write(pat.astype("<f4").tobytes())
        f.write(emb.astype("<f4").tobytes())
    print(f"[golden] {a.image} {w0}x{h0} -> pad {best_w}x{best_h}, patch {n_vit_h}x{n_vit_w}, "
          f"token {n_llm_h}x{n_llm_w} (span {n_llm_h * (n_llm_w + 1) + 2}); "
          f"embeds {emb.shape} |mean| {np.abs(emb).mean():.4f} -> {a.out}")


if __name__ == "__main__":
    main()
