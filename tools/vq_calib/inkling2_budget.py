#!/usr/bin/env python3
"""Inkling-2-Small VQ: exact weight bytes per recipe and the DGX Spark (GB10) residency budget.

Weight bytes follow the container layouts ds4-dfm-rs reads (crates/ds4-core/src/glm_vq.rs for
GVQ1 v1 12/13-bit and GVR1 residuals; tools/vq_calib/axk2_vq_format.py for GVQ v2 9/10-bit;
GGML Q4_K/Q6_K/Q8_0 blocks). Runtime terms (scratch, vision transient) are labelled estimates
carried over from the MiniMax / A.X-K2 / GLM VQ budgets in docs/.
"""
import argparse
import json
import math

GIB = 2**30
H, F, E, L, TOPK = 4608, 2304, 384, 48, 8
VOCAB = 201024
GLOBAL = [5, 11, 17, 23, 29, 35, 41, 47]
LOCAL = [l for l in range(L) if l not in GLOBAL]
HEADS, KVH, HD, DREL = 48, 8, 128, 16
WINDOW = 1024


def align(n, a=128):
    return (n + a - 1) // a * a


def gvq1(rows, cols, bits):            # GVQ1 v1: 12-bit main stream (+ 1-bit plane at 13)
    assert cols % 64 == 0 and bits in (12, 13)
    elems = rows * cols
    main = align(128 + 2 * rows)
    end = main + elems // 16 * 3
    if bits == 13:
        end = align(end) + elems // 64
    return align(end)


def gvr8(rows, cols):                  # GVR1 residual: one byte per vector
    return align(128 + rows * cols // 8)


def gvq2(rows, cols, bits):            # GVQ v2 byte-major 9/10-bit
    assert cols % 64 == 0 and bits in (9, 10)
    n = rows * cols // 8
    main = align(128 + 2 * rows)
    plane = align(main + n)
    return align(plane + n * (bits - 8) // 8)


def q4k(r, c): return r * c // 256 * 144
def q6k(r, c): return r * c // 256 * 210
def q8_0(r, c): return r * c // 32 * 34
def bf16(*s): return 2 * math.prod(s)


MATS = [(F, H), (F, H), (H, F)]        # gate, up, down per expert


def expert_layer(bits, res=False):
    """One MoE layer: 384 x (gate, up, down) + its book(s)."""
    if bits in (9, 10):
        per = sum(gvq2(r, c, bits) for r, c in MATS)
        books = 3 * (1 << bits) * 8    # split3: gate/up/down books (A.X-K2)
    else:
        per = sum(gvq1(r, c, bits) for r, c in MATS)
        books = (1 << bits) * 8        # one shared E4M3 book per layer (GLM/MiniMax/DS41)
        if res:
            per += sum(gvr8(r, c) for r, c in MATS)
            books += 256 * 8
    return E * per + books


def trunk(head='q4k'):
    attn = 0
    for l in range(L):
        attn += q4k(HEADS * HD, H) + q4k(KVH * HD, H) * 2 + q4k(HEADS * DREL, H) \
            + q4k(HEADS * HD, H) + q4k(H, HEADS * HD)             # q, k, v, r, gate, o
        attn += bf16(HEADS, H)                                      # sink_proj
        attn += bf16(HEADS, HD) + bf16(HEADS, DREL) + bf16(DREL, 1024)  # q/rel clip scales, rel profiles
        if l in GLOBAL:
            attn += 2 * (bf16(32, H) + bf16(4, 1024))               # dynamic k/v conv: weight_proj, static
            attn += 3 * bf16(H)                                     # q/k/v input norms
        else:
            attn += 2 * bf16(1024, 4) + bf16(H)                     # k/v sconv, attn_norm
        attn += 2 * bf16(H, 4) + bf16(H)                            # attn_sconv, mlp_sconv, mlp_norm
    router = L * (bf16(E, H) + 4 * E)
    attn_res = 4 * (H * (47 + 2 * 8) + H * 48 + H)                  # F32 queries (3 per global layer)
    headb = {'q4k': q4k, 'q6k': q6k, 'q8_0': q8_0}[head](VOCAB, H)
    norms = 2 * bf16(H)                                             # final norm, embed_norm
    return dict(attention_q4k=attn, router_bf16=router, attn_res_f32=attn_res, head=headb, norms=norms)


def drafter(kind='q8_0'):
    qf = {'q8_0': q8_0, 'q4k': q4k}[kind]
    per = qf(HEADS * HD, H) + 4 * qf(KVH * HD, H) + qf(H, HEADS * HD) \
        + 2 * qf(18432, H) + qf(H, 18432) + 4 * bf16(H) + 4 * bf16(HD)
    markov = bf16(VOCAB, 256) + bf16(768, 9728) + bf16(H, 256) + bf16(H) + bf16(768) + bf16(H)
    return 5 * per + markov + bf16(6, 8) + 2 * bf16(H)


VISION = 2 * 327_700_000               # BF16 tower + merger (census: 0.3277B params)
EMBED = bf16(VOCAB, H)                 # BF16 part, mapped from the SSD (embssd), not resident


def kv_per_token(kind):
    """Global-attention K+V bytes per token over the 8 global layers (+ drafter global layer)."""
    vec = {'bf16': 2 * HD, 'fp8': HD + 4, 'nvfp4': 72}[kind]     # per 128-value head vector
    return len(GLOBAL) * 2 * KVH * vec, 2 * KVH * vec              # main, drafter global layer


def local_ring(chunk):
    return len(LOCAL) * (WINDOW + chunk) * 2 * KVH * HD * 2 + 4 * WINDOW * 2 * KVH * HD * 2


def recipe_bytes(spec):
    """spec: list of (bits, residual) per layer."""
    return sum(expert_layer(b, r) for b, r in spec)


RECIPES = {
    'r13+8 all (MiniMax tier)': [(13, True)] * L,
    'r13 all': [(13, False)] * L,
    'r12 + r13 on 40-47 (12/13 mix)': [(13, False) if l >= 40 else (12, False) for l in range(L)],
    'r12 all': [(12, False)] * L,
    'r10 + r12 on 40-47': [(12, False) if l >= 40 else (10, False) for l in range(L)],
    'r10 all': [(10, False)] * L,
    'r9 + r10 on 40-47': [(10, False) if l >= 40 else (9, False) for l in range(L)],
    'r9 all': [(9, False)] * L,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan-gib', type=float, default=117.58, help='A.X-K2 --check-config device total (docs/axk2-vq.md)')
    p.add_argument('--headroom-gib', type=float, default=6.0, help='quote floor: host floor 4 + burst 2 (serving-contract.md)')
    p.add_argument('--scratch-gib', type=float, default=2.5, help='estimate: 2,048-row graph (A.X-K2 1.81, MiniMax 1.86) + attn-res block states')
    p.add_argument('--vision-transient-gib', type=float, default=0.5, help='estimate: MiniMax still-image transient 0.42-0.90')
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--head', default='q6k')
    p.add_argument('--json')
    a = p.parse_args()
    t = trunk(a.head)
    trunk_b = sum(t.values())
    d = drafter('q8_0')
    fixed = trunk_b + d + VISION + local_ring(a.chunk)
    runtime = (a.scratch_gib + a.vision_transient_gib + a.headroom_gib) * GIB
    out = {'trunk_gib': {k: v / GIB for k, v in t.items()}, 'trunk_total_gib': trunk_b / GIB,
           'drafter_q8_0_gib': d / GIB, 'vision_bf16_gib': VISION / GIB, 'embedding_bf16_ssd_gib': EMBED / GIB,
           'local_ring_gib': local_ring(a.chunk) / GIB,
           'expert_layer_gib': {f'{b}{"+8" if r else ""}': expert_layer(b, r) / GIB
                                for b, r in [(9, 0), (10, 0), (12, 0), (13, 0), (13, 1)]},
           'kv_bytes_per_token': {k: kv_per_token(k) for k in ('bf16', 'fp8', 'nvfp4')}, 'recipes': {}}
    params = L * E * 3 * F * H
    for name, spec in RECIPES.items():
        ex = recipe_bytes(spec)
        left = a.plan_gib * GIB - ex - fixed - runtime
        row = {'experts_gib': ex / GIB, 'bpw': ex * 8 / params, 'resident_weights_gib': (ex + trunk_b) / GIB,
               'left_for_kv_gib': left / GIB}
        for kind in ('bf16', 'fp8', 'nvfp4'):
            main, dr = kv_per_token(kind)
            row[f'max_resident_ctx_{kind}'] = max(0, int(left // (main + dr)))
        main, dr = kv_per_token('fp8')
        row['spare_at_1M_fp8_gib'] = (left - (1 << 20) * (main + dr)) / GIB
        main, dr = kv_per_token('nvfp4')
        row['spare_at_1M_nvfp4_gib'] = (left - (1 << 20) * (main + dr)) / GIB
        out['recipes'][name] = row
    out['assumptions'] = vars(a)
    if a.json:
        open(a.json, 'w').write(json.dumps(out, indent=2) + '\n')
    print(f"trunk {trunk_b/GIB:.2f} GiB (head {a.head}), drafter Q8_0 {d/GIB:.2f}, vision {VISION/GIB:.2f}, "
          f"local ring {local_ring(a.chunk)/GIB:.2f}, embedding on SSD {EMBED/GIB:.2f}")
    print('per-layer GiB:', {k: round(v, 3) for k, v in out['expert_layer_gib'].items()})
    print('KV bytes/token (main 8 global layers, drafter global layer):', out['kv_bytes_per_token'])
    print(f"{'recipe':34s} {'experts':>8s} {'bpw':>6s} {'resident':>9s} {'KV room':>8s} "
          f"{'ctx BF16':>9s} {'ctx FP8':>9s} {'ctx NVFP4':>10s} {'1M FP8':>8s} {'1M NVFP4':>9s}")
    for name, r in out['recipes'].items():
        print(f"{name:34s} {r['experts_gib']:8.2f} {r['bpw']:6.3f} {r['resident_weights_gib']:9.2f} "
              f"{r['left_for_kv_gib']:8.2f} {r['max_resident_ctx_bf16']:9d} {r['max_resident_ctx_fp8']:9d} "
              f"{r['max_resident_ctx_nvfp4']:10d} {r['spare_at_1M_fp8_gib']:8.2f} {r['spare_at_1M_nvfp4_gib']:9.2f}")


if __name__ == '__main__':
    main()
