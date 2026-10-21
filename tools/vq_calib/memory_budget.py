#!/usr/bin/env python3
"""A.X-K2 VQ: weight bytes and the 256K-context residency budget on one DGX Spark.

Every weight byte count is computed from the layouts the converter writes
(GVQ v2 byte-major 9/10-bit matrices, GGML Q4_K / Q8_0 blocks, source bytes;
docs/DESIGN.md), not estimated. Runtime terms (scratch, floor, margin) are
labelled estimates carried over from the Beam / GLM-5.3 VQ budgets.

Usage: memory_budget.py [--ctx 262144] [--r10-layers 1-7] [--json out.json]
"""
import argparse
import json

GIB = 2**30
H, F_MOE, F_DENSE, VOCAB = 7168, 2048, 18432, 163840
LAYERS, DENSE, EXPERTS, TOPK = 61, 1, 256, 8
MOE_LAYERS = list(range(DENSE, LAYERS))
Q_LORA, KV_LORA, ROPE, HEADS, QK_NOPE, V_HEAD = 1536, 512, 64, 64, 128, 128
IDX_HEADS, IDX_DIM = 64, 128
GN_RANK = 16
NATIVE_CTX = 262_144
# KV row per token per layer (FlashMLA FP8 layout): 512 E4M3 latent + 4 F32
# scales (one per 128) + 64 BF16 rope dims; indexer key: 128 E4M3 + 1 F32 scale.
MLA_ROW_FP8, IDX_ROW_FP8 = 512 + 4 * 4 + 64 * 2, 128 + 4
MLA_ROW_BF16, IDX_ROW_BF16 = (512 + 64) * 2, 128 * 2


def align(n, a=128):
    return (n + a - 1) // a * a


def gvq2_bytes(rows, cols, bits):
    """GVQ v2: 128 B header, F16 gains, low-8 stream (1 B/index), (bits-8)-bit plane."""
    assert cols % 64 == 0 and bits in (9, 10)
    n = rows * cols // 8
    main = align(128 + 2 * rows)
    plane = align(main + n)
    return align(plane + n * (bits - 8) // 8)


def q4k(rows, cols):
    assert cols % 256 == 0
    return rows * cols // 256 * 144


def q8_0(rows, cols):
    assert cols % 32 == 0
    return rows * cols // 32 * 34


def expert_layer_bytes(bits):
    per_expert = 2 * gvq2_bytes(F_MOE, H, bits) + gvq2_bytes(H, F_MOE, bits)
    return EXPERTS * per_expert + (2**bits) * 8          # + one E4M3 [K, 8] codebook


def nonexpert():
    q_out = HEADS * (QK_NOPE + ROPE + V_HEAD)            # q_b_proj = fused q + output gate: 20480
    attn = LAYERS * (q4k(Q_LORA, H)                      # q_a_proj
                     + q4k(q_out, 2 * Q_LORA)            # q_b_proj [20480, 3072]
                     + q4k(KV_LORA + ROPE, H)            # kv_a_proj_with_mqa [576, 7168]
                     + q4k(HEADS * (QK_NOPE + V_HEAD), KV_LORA)   # kv_b_proj [16384, 512]
                     + q4k(H, HEADS * V_HEAD))           # o_proj [7168, 8192]
    indexer = LAYERS * (q8_0(IDX_HEADS * IDX_DIM, Q_LORA)        # wq_b [8192, 1536]
                        + q8_0(IDX_DIM, H)                       # wk [128, 7168]
                        + IDX_HEADS * H * 2                      # weights_proj BF16
                        + 2 * IDX_DIM * 2)                       # k_norm weight + bias BF16
    shared = len(MOE_LAYERS) * (2 * q4k(F_MOE, H) + q4k(H, F_MOE))
    dense = DENSE * (2 * q4k(F_DENSE, H) + q4k(H, F_DENSE))
    head = q4k(VOCAB, H)
    router = len(MOE_LAYERS) * (EXPERTS * H * 2 + EXPERTS * 4)   # BF16 weight + F32 bias (source)
    gated_norms = (LAYERS + len(MOE_LAYERS)) * (GN_RANK * H * 2 * 2 + H * 2)  # W_down, W_up, norm (BF16)
    plain_norms = DENSE * H * 2 + LAYERS * (Q_LORA + KV_LORA) * 2 + H * 2      # dense post-attn, q_a/kv_a norms, final
    return dict(attention_q4k=attn, indexer_q8_0_bf16=indexer, shared_experts_q4k=shared,
                dense_ffn_q4k=dense, output_head_q4k=head, router_bf16_bias_f32=router,
                norms_bf16=gated_norms + plain_norms)


def weights(r10):
    experts = sum(expert_layer_bytes(10 if l in r10 else 9) for l in MOE_LAYERS)
    out = nonexpert()
    fixed = sum(out.values())
    out.update(routed_experts=experts, fixed_nonexpert=fixed, resident=experts + fixed,
               embedding_bf16_ssd=VOCAB * H * 2)
    return out


def kv(ctx, kind='fp8'):
    mla, idx = (MLA_ROW_FP8, IDX_ROW_FP8) if kind == 'fp8' else (MLA_ROW_BF16, IDX_ROW_BF16)
    return dict(mla_latent=LAYERS * ctx * mla, indexer_keys=LAYERS * ctx * idx,
                total=LAYERS * ctx * (mla + idx))


def max_r10(ctx, a):
    """Largest bottom-first count of 10-bit MoE layers whose resident weights fit."""
    budget = a.available_gib - a.floor_gib - a.scratch_gib - kv(ctx)['total'] / GIB - a.margin_gib
    best = None
    for n in range(len(MOE_LAYERS) + 1):
        if weights(set(MOE_LAYERS[:n]))['resident'] / GIB <= budget:
            best = n
    return best, budget


def parse_layers(spec):
    if not spec:
        return set()
    lo, _, hi = spec.partition('-')
    return set(range(int(lo), int(hi or lo) + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ctx', type=int, default=NATIVE_CTX)
    p.add_argument('--r10-layers', default='1-7')
    p.add_argument('--available-gib', type=float, default=118.3,
                   help='GB10 MemAvailable used by ds4-server --check-config (GLM-5.3 VQ quote)')
    p.add_argument('--floor-gib', type=float, default=4.0)
    p.add_argument('--scratch-gib', type=float, default=3.0,
                   help='estimate: Beam 2.4 GiB scaled to hidden 7168 and 64-head MLA + indexer workspaces')
    p.add_argument('--margin-gib', type=float, default=1.5)
    p.add_argument('--json')
    a = p.parse_args()
    r10 = parse_layers(a.r10_layers)
    w = weights(r10)
    out = {'ctx': a.ctx, 'r10_layers': sorted(r10),
           'r9_layers': [l for l in MOE_LAYERS if l not in r10],
           'weights_bytes': w, 'weights_gib': {k: v / GIB for k, v in w.items()},
           'expert_layer_gib': {'9': expert_layer_bytes(9) / GIB, '10': expert_layer_bytes(10) / GIB},
           'per_r10_layer_gib': (expert_layer_bytes(10) - expert_layer_bytes(9)) / GIB}
    for kind in ('fp8', 'bf16'):
        k = kv(a.ctx, kind)
        budget = a.available_gib - a.floor_gib - a.scratch_gib - k['total'] / GIB - a.margin_gib
        out[f'kv_{kind}'] = {'mla_latent_gib': k['mla_latent'] / GIB, 'indexer_keys_gib': k['indexer_keys'] / GIB,
                             'total_gib': k['total'] / GIB, 'weight_budget_gib': budget,
                             'spare_gib': budget - w['resident'] / GIB,
                             'quote_gib': w['resident'] / GIB + k['total'] / GIB + a.scratch_gib + a.floor_gib}
    table = []
    for ctx in (131_072, 196_608, 262_144, 524_288):
        n, budget = max_r10(ctx, a)
        row = {'ctx': ctx, 'kv_fp8_gib': kv(ctx)['total'] / GIB, 'weight_budget_gib': budget}
        if n is None:
            # Even all-9-bit does not fit: report the all-9-bit overshoot.
            row.update(r10_layers=None, resident_all_r9_gib=weights(set())['resident'] / GIB)
        else:
            res = weights(set(MOE_LAYERS[:n]))['resident'] / GIB
            row.update(r10_layers=n, r9_layers=len(MOE_LAYERS) - n, resident_gib=res, spare_gib=budget - res,
                       routed_expert_bpw=(weights(set(MOE_LAYERS[:n]))['routed_experts'] * 8)
                       / (len(MOE_LAYERS) * EXPERTS * 3 * F_MOE * H))
        table.append(row)
    out['context_tradeoff'] = table
    params_routed = len(MOE_LAYERS) * EXPERTS * 3 * F_MOE * H
    out['bpw'] = {'routed_experts': w['routed_experts'] * 8 / params_routed,
                  'resident_over_all_params_wo_embedding': w['resident'] * 8 / (691.7e9 - VOCAB * H),
                  'routed_expert_params': params_routed}
    out['assumptions'] = {'available_gib': a.available_gib, 'floor_gib': a.floor_gib,
                          'scratch_gib_estimate': a.scratch_gib, 'margin_gib': a.margin_gib,
                          'kv_rows': {'mla_fp8_bytes': MLA_ROW_FP8, 'indexer_fp8_bytes': IDX_ROW_FP8,
                                      'mla_bf16_bytes': MLA_ROW_BF16, 'indexer_bf16_bytes': IDX_ROW_BF16}}
    text = json.dumps(out, indent=2)
    if a.json:
        open(a.json, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
