#ifndef DS4_K2H_H
#define DS4_K2H_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* K2-Horizon-375B-A23B hVQ device runtime. Weights stay in the GVQ pointer
 * plan (Rust validated every part); weight arguments are GVQ record indices.
 * Arithmetic contract: tools/vq_calib/k2h_ref.py (transformers
 * K2HorizonDecoderLayer to BF16 rounding). F32 activation rows hold
 * BF16-rounded values unless named otherwise.
 *
 *   embed ─▶ [ rms ─▶ q k v ─▶ RoPE ─▶ NVFP4 K/V ─▶ GQA ─▶ o ─▶ +h
 *              ─▶ rms ─▶ dense SwiGLU (0-2) | shared + 8 of 192 experts (3-60) ─▶ +h ] x61
 *         ─▶ rms ─▶ head
 *
 * Q8_0 projections (attention, dense, shared, head), 13-bit GVQ1 experts on
 * GLM's decode and MiniMax's warp-specialized prefill kernels. */
typedef struct ds4_k2h_moe ds4_k2h_moe;

enum {
    DS4_K2H_HIDDEN = 6144, DS4_K2H_HEADS = 48, DS4_K2H_KV_HEADS = 8, DS4_K2H_HEAD = 128,
    DS4_K2H_GROUP = DS4_K2H_HEADS / DS4_K2H_KV_HEADS,
    DS4_K2H_Q = DS4_K2H_HEADS * DS4_K2H_HEAD, DS4_K2H_KV = DS4_K2H_KV_HEADS * DS4_K2H_HEAD,
    /* RoPE rotates pairs (j, j + 64) for j < 32 (k2h_ref.apply_partial_rope:
     * the split/interleave round trip of the first 64 interleaved dims). */
    DS4_K2H_ROPE_PAIRS = 32,
    DS4_K2H_EXPERTS = 192, DS4_K2H_USED = 8, DS4_K2H_FFN = 1792, DS4_K2H_DENSE_FF = 16384,
    DS4_K2H_VOCAB = 250624,
    /* Q8_0: 32 weights per 34-byte block (F16 scale, 32 int8). */
    DS4_K2H_Q8_BLOCK = 32, DS4_K2H_Q8_BYTES = 34,
    DS4_K2H_EMBED_ROW = DS4_K2H_HIDDEN / DS4_K2H_Q8_BLOCK * DS4_K2H_Q8_BYTES,
};

/* NVFP4 K/V (the release's contract, MiniMax VQ's encoder): per 16 values an
 * E4M3 scale max(amax / 6, 2^-9) and E2M1 codes, global scale 1, in pages of
 * one (128-position block, KV head): K data (128 x 64 B), K scales
 * (128 x 8 B), V data, V scales; a block's eight pages are contiguous.
 * K is stored after RoPE. */
enum {
    DS4_K2H_KV_BLOCK = 128, DS4_K2H_FP4_DATA = DS4_K2H_HEAD / 2, DS4_K2H_FP4_ROW = DS4_K2H_FP4_DATA + 8,
    DS4_K2H_PAGE = 2 * DS4_K2H_KV_BLOCK * DS4_K2H_FP4_ROW,
    DS4_K2H_BLOCK_BYTES = DS4_K2H_KV_HEADS * DS4_K2H_PAGE,
    /* Decode attention splits one (row, KV head)'s keys into at most this
     * many partials of DS4_K2H_GROUP x (128 + max + sum) floats. */
    DS4_K2H_SPLITS = 64,
};

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode items and the
 * per-layer device routing tables). */
uint64_t ds4_gpu_k2h_moe_bytes(uint32_t rows);
ds4_k2h_moe *ds4_gpu_k2h_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_k2h_moe_free(ds4_k2h_moe *moe);
/* out = BF16(sum + shared) with sum the BF16 running sum, in ascending
 * expert order, of BF16(BF16(down_e) * weight_e) (the reference's BF16
 * index_add_). `matrices`: the layer's gate[192], up[192], down[192] GVQ
 * indices; `weights` carry the routed scale; `shared`: the shared expert's
 * BF16-rounded rows. */
int ds4_gpu_k2h_moe(ds4_k2h_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared,
    const uint32_t *matrices, uint32_t rows);

/* logits = BF16(router . x) (the F32 router record holds BF16 values), then
 * the top-8 of sigmoid(logits) + bias (selection only; ties to the lower
 * expert) and weights = BF16(2.5 * s / sum of the chosen sigmoids). */
int ds4_gpu_k2h_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *logits, const ds4_gpu_tensor *input, uint32_t router, uint32_t bias, uint32_t rows);

/* out = W x for a Q8_0 record over F32 rows (d * q exact in F32). One row
 * streams on the GEMV, wider inputs take MMQ (input quantized to Q8_1).
 * K2H_Q8_ROUND rounds every output to BF16, as the reference's BF16 GEMM. */
typedef enum { DS4_K2H_Q8_F32, DS4_K2H_Q8_ROUND } ds4_k2h_q8_out;
int ds4_gpu_k2h_q8(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight, const ds4_gpu_tensor *input,
    uint32_t rows, ds4_k2h_q8_out mode);

/* With `branch`: cur = BF16(cur + BF16(branch)). Then norm = BF16(cur *
 * rsqrt(mean cur^2 + 1e-6) * w) over 6,144-wide rows (F32 record `weight`). */
int ds4_gpu_k2h_norm(ds4_gvq_ctx *ctx, ds4_gpu_tensor *norm, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
    uint32_t weight, uint32_t rows);

/* Per-position rows of `rope` (rows x 64 F32): BF16 cos then sin of the 32
 * angles F32(pos) * inv_freq_j, inv_freq_j = 1 / 1e7^(2j / 64) in F32. */
int ds4_gpu_k2h_rope(ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0);

/* The attention input's tails for rows at pos..pos+rows-1: q (48 heads) and
 * k (8) rounded to BF16 and rotated (each product and sum rounded to BF16,
 * as the reference's BF16 tensors), v rounded; then K and V as NVFP4 into
 * the layer's pages (block pos / 128, KV head h). */
int ds4_gpu_k2h_qkv(ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, ds4_gpu_tensor *pages,
    const ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos);

/* Causal GQA attention of rows at pos.. over the pages' keys [0, p]: 48
 * query heads, KV head h / 6, scores q . k * 128^-1/2 in F32, an F32 online
 * softmax, BF16 probabilities times BF16 V, out = BF16(acc / l). Prefill
 * rows run on BF16 tensor cores; decode rows split their keys into partials
 * (`partials`: ds4_gpu_k2h_attn_bytes()). */
uint64_t ds4_gpu_k2h_attn_bytes(void);
int ds4_gpu_k2h_attn(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *pages,
    ds4_gpu_tensor *partials, uint32_t rows, uint32_t pos);

/* out = BF16(BF16(silu(BF16(g))) * BF16(u)) of [rows][width] gate/up rows. */
int ds4_gpu_k2h_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t rows,
    uint32_t width);

/* cur = BF16(F32(d) * q) of `rows` host-gathered Q8_0 embedding rows. */
int ds4_gpu_k2h_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_q8, uint32_t rows);

#ifdef __cplusplus
}
#endif
#endif
