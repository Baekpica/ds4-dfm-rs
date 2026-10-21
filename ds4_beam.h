#ifndef DS4_BEAM_H
#define DS4_BEAM_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Beam-1.0 VQ device runtime. Weights stay in the GVQ pointer plan (Rust
 * validated every part); this adds what GLM's and Ling's adapters do not
 * cover: 256 x (5120 -> 2560 -> 5120) routed experts on K4096 (12-bit) or
 * K8192 (13-bit) layer books, the group-limited sigmoid router, Beam's
 * RMSNorm (weight times the F32 normalized row, one rounding), the
 * depth-scaled sandwich residual, q/k norm with RoPE or the NoPE query
 * scale, the per-element output gate and the soft-capped logits. Weight
 * arguments are GVQ record indices. Arithmetic contract: transformers
 * 5.20.2 models/beam (the handoff's tools/eval/beam_ref.py). */
typedef struct ds4_beam_moe ds4_beam_moe;

enum { DS4_BEAM_EXPERTS = 256, DS4_BEAM_USED = 8, DS4_BEAM_HIDDEN = 5120, DS4_BEAM_FFN = 2560,
       DS4_BEAM_HEADS = 48, DS4_BEAM_KV_HEADS = 8, DS4_BEAM_HEAD = 128, DS4_BEAM_MOE_LAYERS = 49,
       DS4_BEAM_VOCAB = 151936 };

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode/verify items
 * and the per-layer device routing tables). */
uint64_t ds4_gpu_beam_moe_bytes(uint32_t rows);
ds4_beam_moe *ds4_gpu_beam_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_beam_moe_free(ds4_beam_moe *moe);
/* `matrices` holds the layer's gate[256], up[256], down[256] GVQ indices.
 * `out` receives the routed BF16 sum, before the 2.5 routed scale. */
int ds4_gpu_beam_moe(ds4_beam_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows);

/* F32 router logits of the 5120-wide rows (never TF32), then the top-8 ids
 * and normalized weights. `router` and `bias` are the layer's mlp.gate
 * weight and e_score_correction_bias records. */
int ds4_gpu_beam_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *logits, const ds4_gpu_tensor *input, uint32_t router, uint32_t bias, uint32_t rows);

/* out = rms(x) * weight over 5120-wide rows (BF16 weight record). */
int ds4_gpu_beam_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    uint32_t weight, uint32_t rows);

/* After sublayer `index` (attention 2(L+1), FFN 2(L+1)+1):
 *   cur = rms(scale * branch) * post / sqrt(index) + cur * sqrt((index - 1) / index)
 * and, when `next` is a record, norm = rms(cur) * next. */
int ds4_gpu_beam_residual(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cur, const ds4_gpu_tensor *branch,
    uint32_t post, ds4_gpu_tensor *norm, uint32_t next, uint32_t rows, uint32_t index, float scale);

/* Per-position rows of `rope` (rows x 129 F32) for positions pos0...:
 * cos and sin of the 64 RoPE angles, then the full layers' query scale
 * (1 + (pos + 1) / 2048) ** 0.1. Written once per forward. */
int ds4_gpu_beam_rope(ds4_gpu_tensor *rope, uint32_t rows, uint32_t pos0);

/* Per-head RMSNorm of `heads` x 128 rows in place, then rotate_half RoPE
 * (theta 10,000, whole head) or the query scale from `rope`, or nothing
 * (full-layer keys). */
typedef enum { DS4_BEAM_QK_ROPE, DS4_BEAM_QK_SCALE, DS4_BEAM_QK_NORM } ds4_beam_qk;
int ds4_gpu_beam_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t heads,
    uint32_t rows, const ds4_gpu_tensor *rope, ds4_beam_qk mode);

/* Decode attention of the query row over the stored positions (window 0)
 * or the last `window`, rows at slot pos % kv_cap: FP8 (Solar's E4M3 K |
 * E4M3 V | f16 scales, fp8 != 0) or f16 [K | V]. `partials` holds up to
 * `max_splits` x 48 x 130 floats. */
int ds4_gpu_beam_attn_decode(ds4_gpu_tensor *heads, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv,
    ds4_gpu_tensor *partials, uint32_t max_splits, uint32_t kv_cap, uint32_t pos, uint32_t window, int fp8);

/* heads *= sigmoid(gate + bias[head]) per element (bias: BF16 [48] record). */
int ds4_gpu_beam_gate(ds4_gvq_ctx *ctx, ds4_gpu_tensor *heads, const ds4_gpu_tensor *gate,
    uint32_t bias, uint32_t rows);

/* logits = 30 tanh(logits / 30), then -1e30 on the rows past the tokenizer. */
int ds4_gpu_beam_logits(ds4_gpu_tensor *logits, uint32_t rows, uint32_t tokens);

/* BF16 embedding rows (host-gathered) -> the F32 hidden rows. */
int ds4_gpu_beam_embed(ds4_gpu_tensor *cur, const ds4_gpu_tensor *rows_bf16, uint32_t rows);

/* DFlash2 drafter (z-lab dflash model.py), F32 rows throughout: 40 query
 * heads over 8 KV heads, a 4,096-key window both ways and a block of at
 * most 16 rows. Context rings hold f16 [K | V] rows (the K-EXAONE layout)
 * at slot = position % DS4_BEAM_DF_CAP. */
enum { DS4_BEAM_DF_HEADS = 40, DS4_BEAM_DF_BLOCK = 16, DS4_BEAM_DF_WINDOW = 4096,
       DS4_BEAM_DF_CAP = DS4_BEAM_DF_WINDOW + DS4_BEAM_DF_BLOCK, DS4_BEAM_DF_GROUP = 16,
       DS4_BEAM_DF_DYN = 2 * 2 * DS4_BEAM_HIDDEN / DS4_BEAM_DF_GROUP };

/* GroupedDynamicCausalConv over the block rows, one stage:
 *   out[l][c] = sum_o (base[stage][o][c] + dyn[l][stage][o][c / 16]) * x[l - o][c]
 * for taps o = 0, 1 with x[-1] = 0; dyn rows are the kernel_projection rows
 * (stage, tap, group), base the F32 [2][2][5120] kernel. `out` must not
 * alias `x`. */
int ds4_gpu_beam_df_conv(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *dyn,
    const void *map, uint64_t size, uint64_t base, uint32_t stage, uint32_t rows);

/* Block attention: row r (position start + r) attends every ring key in
 * [max(first, start + r - 4095), start + rows), the context below the block
 * and all block rows, softmax scale 1/sqrt(128). */
int ds4_gpu_beam_df_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv,
    uint32_t first, uint32_t start, uint32_t rows);

#ifdef __cplusplus
}
#endif
#endif
