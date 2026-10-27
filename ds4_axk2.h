#ifndef DS4_AXK2_H
#define DS4_AXK2_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A.X-K2 VQ device runtime. Weights stay in the GVQ pointer plan (Rust
 * validated every part); this adds what the GLM/Ling/Beam adapters do not
 * cover: 256 x (7168 -> 2048 -> 7168) routed experts on GVQ1 v2 books
 * (K512 9-bit, K1024 10-bit, byte-major indices; gate, up and down books),
 * the gated RMSNorm, MLA with the fused query/output-gate projection over
 * FP8 latent rows, the DSA indexer (64 x 128, top-2,048) over FP8 keys and
 * the absorbed attention through the Q4_K kv_b. Weight arguments are GVQ
 * record indices. Arithmetic contract: the handoff's tools/axk2_ref.py
 * (transformers models/axk2 with the -inf router mask). F32 rows hold
 * BF16-rounded values unless stated. */
typedef struct ds4_axk2_moe ds4_axk2_moe;

enum { DS4_AXK2_EXPERTS = 256, DS4_AXK2_USED = 8, DS4_AXK2_HIDDEN = 7168, DS4_AXK2_FFN = 2048,
       DS4_AXK2_HEADS = 64, DS4_AXK2_LATENT = 512, DS4_AXK2_NOPE = 128, DS4_AXK2_ROPE = 64,
       DS4_AXK2_KEY = 192, DS4_AXK2_VALUE = 128, DS4_AXK2_QB = DS4_AXK2_HEADS * (DS4_AXK2_KEY + DS4_AXK2_VALUE),
       DS4_AXK2_Q_RANK = 1536, DS4_AXK2_KV_ROW = DS4_AXK2_LATENT + DS4_AXK2_ROPE,
       DS4_AXK2_INDEX_HEADS = 64, DS4_AXK2_INDEX_DIM = 128, DS4_AXK2_TOPK = 2048,
       DS4_AXK2_GATE_RANK = 16, DS4_AXK2_MOE_LAYERS = 60, DS4_AXK2_VOCAB = 163840,
       // Latent rows and index keys keep one int8 power-of-two exponent per
       // row after the E4M3 bytes, the region rounded up to 32 rows.
       DS4_AXK2_EXP_ALIGN = 32 };

/* Cache bytes of `cap` rows: E4M3 latent rows (512 B) or index keys
 * (128 B) with their exponents; k_pe rows are 64 BF16. BF16 index keys
 * (DS4_AXK2_INDEX_KEYS=bf16, a diagnostic of the FP8 rounding) take 256 B. */
typedef enum { DS4_AXK2_KEYS_FP8, DS4_AXK2_KEYS_BF16 } ds4_axk2_keys;
static inline uint64_t ds4_axk2_exp_bytes(uint32_t cap) {
    return ((uint64_t)cap + DS4_AXK2_EXP_ALIGN - 1) / DS4_AXK2_EXP_ALIGN * DS4_AXK2_EXP_ALIGN;
}
static inline uint64_t ds4_axk2_latent_bytes(uint32_t cap) {
    return (uint64_t)cap * DS4_AXK2_LATENT + ds4_axk2_exp_bytes(cap);
}
static inline uint64_t ds4_axk2_index_bytes(uint32_t cap, ds4_axk2_keys keys) {
    return keys == DS4_AXK2_KEYS_BF16 ? (uint64_t)cap * DS4_AXK2_INDEX_DIM * 2
                                      : (uint64_t)cap * DS4_AXK2_INDEX_DIM + ds4_axk2_exp_bytes(cap);
}

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode items and
 * the per-layer device routing tables). */
uint64_t ds4_gpu_axk2_moe_bytes(uint32_t rows);
ds4_axk2_moe *ds4_gpu_axk2_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_axk2_moe_free(ds4_axk2_moe *moe);
/* `matrices` holds the layer's gate[256], up[256], down[256] GVQ indices.
 * out = bf16(bf16(F32 sum of weight x expert row, ascending expert) +
 * shared); `weights` already carry the 2.5 routed scale. */
int ds4_gpu_axk2_moe(ds4_axk2_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows,
    const ds4_gpu_tensor *shared);

/* Group-limited sigmoid router over F32 logits [rows][256]: 8 groups of
 * 32, the best 4 by their two best biased scores, the others masked to
 * -inf, top-8 by biased score, weights = unbiased sigmoid normalized x 2.5.
 * `bias` is the layer's F32 e_score_correction_bias record. */
int ds4_gpu_axk2_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *logits, uint32_t bias, uint32_t rows);

/* cur = bf16(cur + branch) when `branch`, then out = norm(cur): RMSNorm
 * (weight x the BF16 normalized row, rounded) and, when `down` is a record,
 * the gate bf16(y * sigmoid(W_up silu(W_down y))) of rank 16. `mid` holds
 * rows x 17 floats of decode scratch. */
int ds4_gpu_axk2_norm(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
    uint32_t norm, uint32_t down, uint32_t up, ds4_gpu_tensor *mid, uint32_t rows);

/* rows x 32 (cos, sin) pairs of positions pos0... for the 32 YaRN inverse
 * frequencies in `inv_freq` (F32 [32]): cos/sin of the F32 angle, rounded
 * to BF16 as the source's cast (attention factor 1). */
int ds4_gpu_axk2_rope(ds4_gpu_tensor *rope, const ds4_gpu_tensor *inv_freq, uint32_t rows, uint32_t pos0);

/* q_b rows [rows][64][192 + 128] -> q [rows][64][192] with the interleaved
 * MLA RoPE on dims 128..191 (de-interleaved output, as transformers'
 * apply_rotary_pos_emb_interleave) and gate [rows][64][128]. */
int ds4_gpu_axk2_q_split(ds4_gpu_tensor *q, ds4_gpu_tensor *gate, const ds4_gpu_tensor *qb,
    const ds4_gpu_tensor *rope, uint32_t rows);

/* kv_a rows [rows][576]: kv_a_layernorm over the 512 latent values into the
 * FP8 latent rows of slots pos0.., the 64 rope values rotated (interleaved)
 * into k_pe (BF16). */
int ds4_gpu_axk2_kv_store(ds4_gvq_ctx *ctx, ds4_gpu_tensor *latent, ds4_gpu_tensor *k_pe,
    const ds4_gpu_tensor *kv, uint32_t norm, const ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0,
    uint32_t cap);

/* Indexer key rows wk(x) [rows][128]: LayerNorm (weight, bias), split-half
 * RoPE on dims 0..63, BF16, then stored in slots pos0.. as E4M3 with a row
 * exponent (or BF16). */
int ds4_gpu_axk2_index_store(ds4_gvq_ctx *ctx, ds4_gpu_tensor *keys, const ds4_gpu_tensor *k,
    uint32_t norm, uint32_t bias, const ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0, uint32_t cap,
    ds4_axk2_keys format);
/* Indexer queries wq_b(q_r) [rows][64][128] in place: split-half RoPE on
 * dims 0..63, BF16. */
int ds4_gpu_axk2_index_q(ds4_gpu_tensor *q, const ds4_gpu_tensor *rope, uint32_t rows);
/* scores[r][t] for keys t <= pos0 + r (-inf above, up to `keys`):
 * sum_h w[r][h] relu(q[r][h] . k[t] / sqrt(128)); w = weights_proj(x)
 * x 64^-0.5 in F32. `stride` floats per score row. */
int ds4_gpu_axk2_index_scores(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *w,
    const ds4_gpu_tensor *cache, uint32_t rows, uint32_t pos0, uint32_t keys, uint32_t stride, uint32_t cap,
    ds4_axk2_keys format);
/* selected[r] = the 2,048 largest of scores[r][0..keys) (keys >= 2,048),
 * ties lowest index first, in no particular order. */
int ds4_gpu_axk2_topk(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores, uint32_t keys, uint32_t rows);

/* q_abs [rows][64][512] F32 = q_nope [rows][64][:128] x W_UK (the k rows of
 * the Q4_K kv_b). Rows above the direct kernel's width run cuBLAS on kv_b's
 * BF16 restore in `scratch` (ds4_gpu_axk2_absorb_bytes): RESTORE writes it,
 * REUSE takes the one a previous slice of the same layer left there. */
typedef enum { DS4_AXK2_KV_B_RESTORE, DS4_AXK2_KV_B_REUSE } ds4_axk2_kv_b;
uint64_t ds4_gpu_axk2_absorb_bytes(uint32_t rows);
int ds4_gpu_axk2_absorb(ds4_gvq_ctx *ctx, ds4_gpu_tensor *q_abs, const ds4_gpu_tensor *q, uint32_t kv_b,
    uint32_t rows, ds4_gpu_tensor *scratch, ds4_axk2_kv_b restore);
/* Absorbed attention of rows at pos0.. over FP8 latent rows and k_pe:
 * every causal key, or the selected [rows][2048] (keys above a row's
 * position skipped), s = (q_abs . c + q_pe . k_pe) x scale. Decode rows
 * split their keys over CTAs through `partials`
 * (ds4_gpu_axk2_flash_bytes). */
uint64_t ds4_gpu_axk2_flash_bytes(void);
int ds4_gpu_axk2_flash(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *partials, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *q_abs, const ds4_gpu_tensor *latent, const ds4_gpu_tensor *k_pe,
    const ds4_gpu_tensor *selected, uint32_t rows, uint32_t pos0, float scale, uint32_t cap);
/* heads [rows][64][128] = bf16(bf16(latent_out x W_UV^T) x sigmoid(gate)),
 * W_UV the v rows of kv_b (same scratch rule as the absorb). */
int ds4_gpu_axk2_value(ds4_gvq_ctx *ctx, ds4_gpu_tensor *heads, const ds4_gpu_tensor *latent_out, uint32_t kv_b,
    const ds4_gpu_tensor *gate, uint32_t rows, ds4_gpu_tensor *scratch);

/* The two Q8_0 indexer projections: F32 rows of `outs` outputs. */
int ds4_gpu_axk2_q8(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight, const ds4_gpu_tensor *input,
    uint32_t rows);

/* BF16 embedding rows (host-gathered) -> the F32 hidden rows. */
int ds4_gpu_axk2_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows);

#ifdef __cplusplus
}
#endif
#endif
