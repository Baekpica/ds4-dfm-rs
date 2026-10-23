#ifndef DS4_MINIMAX_H
#define DS4_MINIMAX_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MiniMax-M3.1-Flash VQ device runtime. Weights stay in the GVQ pointer plan
 * (Rust validated every part); this adds what GLM's, Ling's and Beam's
 * adapters do not cover: 128 x (6144 -> 3072 -> 6144) routed experts on a
 * K8192 (13-bit) layer book, with GLM's 8-bit residual stage on twenty
 * layers, top-4 of the sigmoid router plus one shared expert, the clamped
 * SwiGLU-OAI activation, Gemma RMSNorm (x * rsqrt(mean x^2 + eps) * (1 + w)),
 * per-head q/k/index norms with partial NeoX RoPE (64 of 128 dims) and MSA:
 * a block indexer over NVFP4 index keys that picks 16 blocks of 128 tokens
 * per KV head, then attention over NVFP4 K/V pages of those blocks only.
 * Weight arguments are GVQ record indices. Arithmetic contract: the
 * handoff's tools/minimax_ref.py (matches transformers 5.20.2
 * MiniMaxM3VLDecoderLayer to BF16 rounding). */
typedef struct ds4_minimax_moe ds4_minimax_moe;

enum { DS4_MINIMAX_EXPERTS = 128, DS4_MINIMAX_USED = 4, DS4_MINIMAX_HIDDEN = 6144, DS4_MINIMAX_FFN = 3072,
       DS4_MINIMAX_HEADS = 64, DS4_MINIMAX_KV_HEADS = 4, DS4_MINIMAX_HEAD = 128, DS4_MINIMAX_ROT = 64,
       DS4_MINIMAX_INDEX = 128, DS4_MINIMAX_BLOCK = 128, DS4_MINIMAX_TOPK = 16,
       DS4_MINIMAX_MOE_LAYERS = 57, DS4_MINIMAX_VOCAB = 200064,
       /* One NVFP4 vector of 128 values: 64 bytes of E2M1 pairs (even element
        * in the low nibble), then eight E4M3 scales, one per 16 values. */
       DS4_MINIMAX_FP4_DATA = DS4_MINIMAX_HEAD / 2, DS4_MINIMAX_FP4_ROW = DS4_MINIMAX_FP4_DATA + DS4_MINIMAX_HEAD / 16,
       /* A K/V page holds one (block, KV head): K data, K scales, V data, V
        * scales for its 128 tokens (18,432 bytes; 73,728 per block). */
       DS4_MINIMAX_KV_PAGE = 2 * DS4_MINIMAX_BLOCK * DS4_MINIMAX_FP4_ROW,
       DS4_MINIMAX_KV_BLOCK = DS4_MINIMAX_KV_HEADS * DS4_MINIMAX_KV_PAGE,
       /* Decode attention partials: one per (KV head, selected block). */
       DS4_MINIMAX_SPLITS = DS4_MINIMAX_KV_HEADS * DS4_MINIMAX_TOPK };

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode items and the
 * per-layer device routing tables). */
uint64_t ds4_gpu_minimax_moe_bytes(uint32_t rows);
ds4_minimax_moe *ds4_gpu_minimax_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_minimax_moe_free(ds4_minimax_moe *moe);
/* out = round(sum over the selected experts e of weights[e] * expert_e(input)
 * + shared), the sum in F32 in ascending expert order (the reference's F32
 * index_add, then one rounding). `matrices` holds the layer's w1[128],
 * w3[128], w2[128] GVQ indices; `weights` already carry the routed scale. */
int ds4_gpu_minimax_moe(ds4_minimax_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared,
    const uint32_t *matrices, uint32_t rows);

/* F32 router logits of the 6144-wide rows (never TF32), then the top-4 ids
 * by sigmoid + correction bias and weights = the unbiased sigmoids over
 * their sum, times 2.0. `router` and `bias` are the layer's
 * block_sparse_moe.gate weight and e_score_correction_bias records. */
int ds4_gpu_minimax_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *logits, const ds4_gpu_tensor *input, uint32_t router, uint32_t bias, uint32_t rows);

/* out = gemma_rms(x) over 6144-wide rows with a BF16 weight record. */
int ds4_gpu_minimax_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    uint32_t weight, uint32_t rows);

/* cur = round(cur + branch) (the reference's BF16 residual add), then, when
 * `next` is a record, norm = gemma_rms(cur) with that weight. */
int ds4_gpu_minimax_residual(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
    ds4_gpu_tensor *norm, uint32_t next, uint32_t rows);

/* SwiGLU-OAI of [rows][width] gate and up rows into `out`:
 * min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, -7, 7) + 1), in F32,
 * rounded once. */
int ds4_gpu_minimax_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
    uint32_t rows, uint32_t width);

/* Per-position rows of `rope` (rows x 64 F32): cos then sin of the 32 RoPE
 * angles pos * 1e7^(-2j/64), evaluated in double and rounded once. */
int ds4_gpu_minimax_rope(ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0);

/* Per-head Gemma RMSNorm of `heads` x 128 rows in place, rounded once, then
 * NeoX RoPE on dims [0, 64) (pairs i, i + 32) with the row's cos/sin,
 * rounded once; dims [64, 128) pass. */
int ds4_gpu_minimax_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t heads,
    uint32_t rows, const ds4_gpu_tensor *rope);

/* NVFP4 stores of rows at positions pos..pos+rows-1 (the vLLM encoder: per
 * 16 values the E4M3 scale of max(amax / 6, 2^-9), E2M1 codes rounded to
 * nearest even, global scale 1). K/V go to the layer's pages (block
 * pos / 128, KV head h); index keys to `pool` rows of 72 bytes. */
int ds4_gpu_minimax_kv_store(ds4_gpu_tensor *pages, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
    uint32_t rows, uint32_t pos);
int ds4_gpu_minimax_index_store(ds4_gpu_tensor *pool, const ds4_gpu_tensor *key, uint32_t rows, uint32_t pos);

/* MSA block selection for rows at pos..: per row and KV head h the block
 * score is the max over the block's visible tokens of
 * index_q[h] . index_key; the row's own block always wins; the top 16
 * blocks (ties to the lower block) or every visible block when there are at
 * most 16. `blocks` receives rows x 4 x 16 ids, -1 past the selection.
 * `scores` holds rows x 4 x ceil((pos + rows) / 128) F32 block scores for
 * the rows that need them (`score_rows` rows at a time). */
int ds4_gpu_minimax_select(ds4_gpu_tensor *blocks, ds4_gpu_tensor *scores, uint32_t score_rows,
    const ds4_gpu_tensor *query, const ds4_gpu_tensor *pool, uint32_t rows, uint32_t pos);

/* Attention of rows at pos.. over their selected blocks: 64 query heads,
 * KV head h = head / 16, causal inside the row's own block, softmax scale
 * 128^-0.5 in F32, output rounded once. One row splits its blocks into
 * DS4_MINIMAX_SPLITS partials (`partials`: SPLITS x 16 x 130 floats). */
int ds4_gpu_minimax_attn(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *pages,
    const ds4_gpu_tensor *blocks, ds4_gpu_tensor *partials, uint32_t rows, uint32_t pos);

/* logits = -inf on the head rows past the tokenizer (`tokens`). */
int ds4_gpu_minimax_logits(ds4_gpu_tensor *logits, uint32_t rows, uint32_t tokens);

/* BF16 embedding rows (host-gathered) -> the F32 hidden rows. */
int ds4_gpu_minimax_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows);

#ifdef __cplusplus
}
#endif
#endif
