#!/usr/bin/env python3
"""Validation of ds41_vq_format against the published base GGUF and the
original FP4 checkpoint.  Run as a script (python3 -I is fine):

    CUDA_VISIBLE_DEVICES=7 python3 -I tests/test_ds41_vq_format.py [--quick]

Checks
  1. GGUF table over the 40 parts (and, if present, equality with the
     assembled copy); 43 vqblob tensors with the predicted sizes.
  2. read -> pack round trip is byte-identical: layers 0, 13, 14, 39, mtp.0.
  3. dequant vs original FP4 for sampled experts of layers 0, 20 (and 39):
     true cosine, rel. MSE = sse/energy and YoungAi's printed
     "cos" = sqrt(1 - sse/energy) (v41_quantize.c:279).
  4. Replication of YoungAi numerics on those experts: f16 row gains
     recomputed from the originals == stored gains; nearest-codeword
     re-assignment (unweighted L2 against the E4M3 book, lowest index on
     ties) == stored indices.
Exit status 0 iff all hard checks pass.
"""
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ds41_vq_format as F  # noqa: E402

ASSEMBLED = "/work/ivq/base/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
QUICK = "--quick" in sys.argv
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def dev():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def test_table(g):
    print("[1] GGUF table", flush=True)
    check(len(g.tensors) == 1000 and len(g.kv) == 73, f"1000 tensors / 73 KV (got {len(g.tensors)} / {len(g.kv)})")
    check(g.size == F.BASE_SIZE, f"virtual size {g.size}")
    vq = g.vq_tensors()
    check(len(vq) == 43, f"43 vqblob tensors (got {len(vq)})")
    exp = {}
    for L in range(40):
        exp[F.vq_tensor_name(L)] = 2_767_968_272 if L < 14 else 2_555_598_864
    for T in range(3):
        exp[F.vq_tensor_name(mtp=T)] = 851_888_144
    check({t.name: t.nbytes for t in vq} == exp, "vqblob names and sizes as predicted by blob_layout")
    if os.path.exists(ASSEMBLED) and os.path.getsize(ASSEMBLED) == F.BASE_SIZE:
        a = F.GGUFFile(ASSEMBLED)
        same = [(t.name, t.ggml_type, t.ne, t.abs_offset) for t in a.tensors] == \
               [(t.name, t.ggml_type, t.ne, t.abs_offset) for t in g.tensors]
        check(same, "assembled copy has the identical tensor table")
        t = g.tensor(F.vq_tensor_name(20))
        ok = all(np.array_equal(g.vf.pread(o, 4096), a.vf.pread(o, 4096))
                 for o in (t.abs_offset, t.abs_offset + t.nbytes - 4096, 2_838_915_986 * 24 - 2048))
        check(ok, "parts reader == assembled file at sample offsets (incl. a part boundary)")
        a.close()


def test_roundtrip(g, names):
    print("[2] read -> pack round trip", flush=True)
    out = {}
    for name in names:
        t0 = time.time()
        raw = g.read(name)
        lay = F.decode_vq_blob(raw, name)
        t1 = time.time()
        packed = F.pack_vq_layer(lay)
        t2 = time.time()
        eq = packed.size == raw.size and np.array_equal(packed, raw)
        h = hashlib.sha256(raw.tobytes()).hexdigest()[:16]
        check(eq, f"{name}: L={lay.L} nexp={lay.nexp} K={lay.K} bits={lay.bits} {raw.size} B "
                  f"byte-identical (sha256 {h}..., decode {t1 - t0:.1f}s pack {t2 - t1:.1f}s)")
        if name.startswith("mtp."):      # keep only the small one (a layer's int32 indices = 6.8 GB)
            out[name] = lay
        del raw, packed, lay
    return out


def assign_check(V, C, stored, device):
    """Nearest codeword with YoungAi's exact distance (fp32, d sequential, no FMA,
    strict < => lowest index wins) vs stored indices.  Returns (match, tie_ok)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    Vt = torch.from_numpy(V).to(device)
    Ct = torch.from_numpy(C).to(device)
    St = torch.from_numpy(stored.astype(np.int64)).to(device)
    cn = (Ct * Ct).sum(1)
    match = tie = 0
    n = Vt.shape[0]
    bs = 16384
    for i in range(0, n, bs):
        v = Vt[i:i + bs]
        d2 = cn[None, :] - 2.0 * (v @ Ct.T)
        cand = torch.topk(d2, 6, dim=1, largest=False).indices                 # [b, 6]
        cand = torch.cat([cand, St[i:i + bs, None]], 1)                         # + stored
        e = v[:, None, :] - Ct[cand]                                            # [b, 7, 8]
        dist = e[..., 0] * e[..., 0]
        for d in range(1, 8):
            dist = dist + e[..., d] * e[..., d]
        dmin = dist.min(1, keepdim=True).values
        # lowest index among exact minima
        big = torch.iinfo(torch.int64).max
        best = torch.where(dist == dmin, cand, torch.full_like(cand, big)).min(1).values
        match += int((best == St[i:i + bs]).sum())
        tie += int((dist[:, -1] == dmin[:, 0]).sum())
    return match / n, tie / n


def test_vs_original(layers_experts, g):
    print("[3/4] dequant vs original FP4 + YoungAi numerics", flush=True)
    orig = F.OriginalWeights()
    device = dev()
    print(f"  device {device}; {len(orig.ent)} original tensors indexed", flush=True)
    for L, experts, expect_cos in layers_experts:
        name = F.vq_tensor_name(L)
        if not orig.has(f"layers.{L}.ffn.experts.{experts[-1]}.w2.weight"):
            print(f"  SKIP layer {L}: original shard not downloaded", flush=True)
            continue
        lay = F.read_vq_layer(L, g)
        agg = {"sse": 0.0, "en": 0.0}
        g_match = g_tot = 0
        a_match, a_tie, a_n = 0.0, 0.0, 0
        for e in experts:
            for p in F.PROJS:
                W = orig.expert_fp4(L, e, p)
                Wh = F.dequant(lay, e, p, device).double().cpu().numpy()
                Wd = W.astype(np.float64)
                sse = float(((Wd - Wh) ** 2).sum())
                en = float((Wd ** 2).sum())
                cos = float((Wd * Wh).sum() / np.sqrt(en * (Wh ** 2).sum()))
                agg["sse"] += sse
                agg["en"] += en
                agg.setdefault(p, [0.0, 0.0])
                agg[p][0] += sse
                agg[p][1] += en
                agg.setdefault("cos_" + p, []).append(cos)
                # gains
                gy = F.youngai_row_gain(W)
                gs = lay.gains[p][e]
                g_match += int((gy.view(np.uint16) == gs.view(np.uint16)).sum())
                g_tot += gs.size
                # assignment (one expert per layer, all three matrices; quick mode: w1 only)
                if e == experts[0] and (not QUICK or p == "w1"):
                    V = (W / gs.astype(np.float32)[:, None]).astype(np.float32).reshape(-1, 8)
                    m, t = assign_check(V, lay.cb, lay.idx[p][e].reshape(-1), device)
                    a_match += m * V.shape[0]
                    a_tie += t * V.shape[0]
                    a_n += V.shape[0]
        rel = agg["sse"] / agg["en"]
        cos_y = float(np.sqrt(1 - rel))
        per = ", ".join(f"{p}: rel {agg[p][0] / agg[p][1]:.4f} cos {np.mean(agg['cos_' + p]):.4f}" for p in F.PROJS)
        print(f"  L{L} K={lay.K} experts {experts}: {per}", flush=True)
        check(abs(cos_y - expect_cos) < 0.006,
              f"L{L}: rel.MSE {rel:.5f}, YoungAi cos sqrt(1-rel) {cos_y:.4f} (reported ~{expect_cos}), "
              f"true cos mean {np.mean([c for p in F.PROJS for c in agg['cos_' + p]]):.4f}")
        check(g_match == g_tot, f"L{L}: recomputed f16 RMS gains == stored for {g_match}/{g_tot} rows")
        check(a_match / a_n > 0.999,
              f"L{L}: nearest-E4M3-codeword re-assignment == stored for {a_match / a_n:.6f} of {a_n} vectors "
              f"(stored idx at exact min distance: {a_tie / a_n:.6f})")
        del lay


def test_write_blob(g, lay):
    """write_blob into a sparse file of the base's size (no 113 GB copy), read back."""
    print("[5] write_blob at the tensor's absolute offset", flush=True)
    path = "/work/ivq/_sparse_write_test.gguf"
    try:
        with open(path, "wb") as f:
            f.truncate(g.size)
        blob = F.pack_vq_layer(lay)
        off = F.write_blob(path, lay.name, blob, g, fsync=False, check_header=False)
        t = g.tensor(lay.name)
        fd = os.open(path, os.O_RDONLY)
        back = np.frombuffer(os.pread(fd, blob.size, off), np.uint8)
        before = os.pread(fd, 32, off - 32)
        after = os.pread(fd, 32, off + blob.size)
        os.close(fd)
        check(off == t.abs_offset and np.array_equal(back, blob) and not any(before) and not any(after),
              f"{lay.name}: wrote {blob.size} B at {off}, read back identical, neighbours untouched")
        F.write_blob(path, lay.name, blob, g, fsync=False)   # header check now passes
        try:
            F.write_blob(path, lay.name, blob[:-1], g, fsync=False)
            check(False, "size mismatch rejected")
        except ValueError:
            check(True, "size mismatch rejected")
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    t0 = time.time()
    g = F.open_base()   # the 40 parts as one virtual file
    test_table(g)
    names = [F.vq_tensor_name(L) for L in ((0, 39) if QUICK else (0, 13, 14, 39))] + [F.vq_tensor_name(mtp=0)]
    rt = test_roundtrip(g, names)
    test_write_blob(g, rt[F.vq_tensor_name(mtp=0)])
    del rt
    ex = [0, 383] if QUICK else [0, 1, 77, 128, 200, 255, 300, 383]
    plan = [(0, ex, 0.925), (20, ex, 0.911)] + ([] if QUICK else [(39, ex[:4], 0.910)])
    test_vs_original(plan, g)
    print(f"done in {time.time() - t0:.0f}s: {'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
