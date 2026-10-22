#!/usr/bin/env python3
"""Re-train one layer's shared codebook from the original FP4 weights with the
YoungAi recipe (pool sampling + 8 unweighted Lloyd iterations + f16->E4M3) and
compare it with the codebook stored in the base GGUF.

    CUDA_VISIBLE_DEVICES=7 python3 -I tests/replicate_youngai_codebook.py [L | mtpT] [iters]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ds41_vq_format as F  # noqa: E402


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "20"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mtp = int(arg[3:]) if arg.startswith("mtp") else None
    L = None if mtp is not None else int(arg)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    lay = F.read_vq_layer(L, mtp=mtp)
    orig = F.OriginalWeights()
    ps, ntr = F.youngai_pool_stride(lay.nexp, lay.shapes, lay.K)
    print(f"{lay.name}: nexp {lay.nexp} K {lay.K}; pool stride {ps}, {ntr}/expert -> {ntr * lay.nexp} vectors", flush=True)
    T = torch.tensor(F.FP4_TABLE, device=dev)
    pool = torch.empty(ntr * lay.nexp, lay.dim, dtype=torch.float32, device=dev)
    pre = f"mtp.{mtp}" if mtp is not None else f"layers.{L}"
    for e in range(lay.nexp):
        parts = []
        for p in F.PROJS:
            w = torch.from_numpy(orig.raw(f"{pre}.ffn.experts.{e}.{p}.weight").view(np.uint8)).to(dev)
            s = torch.from_numpy(F.e8m0_to_f32(orig.raw(f"{pre}.ffn.experts.{e}.{p}.scale"))).to(dev)
            rows = w.shape[0]
            W = torch.stack([T[(w & 0xF).long()], T[(w >> 4).long()]], -1).reshape(rows, -1)
            W = (W.reshape(rows, -1, 32) * s[:, :, None]).reshape(rows, -1)
            g = torch.from_numpy(lay.gains[p][e].astype(np.float32)).to(dev)   # == recomputed (verified)
            parts.append((W / g[:, None]).reshape(-1, lay.dim))
        V = torch.cat(parts, 0)
        pool[e * ntr:(e + 1) * ntr] = V[::ps][:ntr]
    print(f"  pool built {time.time() - t0:.0f}s", flush=True)
    C = F.youngai_lloyd(pool, lay.K, iters)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    cb = F.codebook_to_e4m3_like_youngai(C.cpu().numpy())
    same_b = float((cb == lay.cb_bytes).mean())
    same_w = float((cb == lay.cb_bytes).all(1).mean())
    print(f"  Lloyd {iters} it done {time.time() - t0:.0f}s: E4M3 bytes identical {same_b:.6f}, "
          f"codewords identical {same_w:.6f}", flush=True)
    if same_w < 1:
        bad = np.nonzero(~(cb == lay.cb_bytes).all(1))[0][:5]
        for k in bad:
            print(f"   k={k}: ours {F.e4m3_decode(cb[k])} stored {lay.cb[k]} f32 {C[k].cpu().numpy()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
