#ifndef DS4_INKLING2_H
#define DS4_INKLING2_H

#include <stdint.h>

#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Inkling-2-Small VQ device runtime. Weights stay in the GVQ pointer plan
 * (Rust validated every part); weight arguments are GVQ record indices.
 * Arithmetic contract: tools/vq_calib/inkling2_ref.py (transformers
 * 109a7d19 modeling_inkling2.py, eager attention). F32 activation rows hold
 * BF16-rounded values unless named otherwise; every kernel rounds an input
 * where the source has a BF16 tensor (prefill projections leave F32 sums).
 *
 *   embed ─▶ [ attn-res ─▶ norm(s) ─▶ q k v r gate sink ─▶ conv ─▶ clip
 *              ─▶ attention (+rel bias, sink, gate) ─▶ o ─▶ conv ─▶ +h
 *              ─▶ attn-res ─▶ norm ─▶ router ─▶ 8 of 384 experts ─▶ conv ─▶ +h ] x48
 *         ─▶ attn-res ─▶ norm ─▶ head ─▶ 30 tanh(z / 30)
 *
 * Block attention residuals read the inputs of layers 0, 6, ..., 42 of the
 * same token ("block states", BF16) beside the running partial state. */
typedef struct ds4_ink2_moe ds4_ink2_moe;

enum {
    DS4_INK2_HIDDEN = 4608, DS4_INK2_HEADS = 48, DS4_INK2_KV_HEADS = 8, DS4_INK2_HEAD = 128,
    DS4_INK2_GROUP = DS4_INK2_HEADS / DS4_INK2_KV_HEADS,
    DS4_INK2_Q = DS4_INK2_HEADS * DS4_INK2_HEAD, DS4_INK2_KV = DS4_INK2_KV_HEADS * DS4_INK2_HEAD,
    DS4_INK2_REL = 16, DS4_INK2_EXTENT = 1024, DS4_INK2_WINDOW = 1024,
    DS4_INK2_TAPS = 4, DS4_INK2_HISTORY = DS4_INK2_TAPS - 1,
    DS4_INK2_EXPERTS = 384, DS4_INK2_USED = 8, DS4_INK2_FFN = 2304,
    DS4_INK2_LAYERS = 48, DS4_INK2_BLOCK = 6, DS4_INK2_BLOCKS = DS4_INK2_LAYERS / DS4_INK2_BLOCK,
    DS4_INK2_VOCAB = 201024, DS4_INK2_TOKENS = 200058,
    /* Decode and verify rows: every row is computed as a lone decode row. */
    DS4_INK2_SPLIT_ROWS = 8,
    /* Decode attention splits the keys of one (row, KV head) into at most
     * this many partials of DS4_INK2_GROUP x (128 + max + sum) floats. */
    DS4_INK2_SPLITS = 64,
};

/* K/V formats. BF16 rows: K then V, 1,024 values each, per position.
 * NVFP4 (full layers): per 16 values an E4M3 scale max(amax / 6, 2^-9) and
 * E2M1 codes (the MiniMax VQ encoder, global scale 1), in pages of one
 * (128-position block, KV head): K data (128 x 64 B), K scales (128 x 8 B),
 * V data, V scales; a block's eight pages are contiguous. */
enum {
    DS4_INK2_KV_BF16 = 0, DS4_INK2_KV_NVFP4 = 1,
    DS4_INK2_KV_BLOCK = 128, DS4_INK2_FP4_DATA = DS4_INK2_HEAD / 2, DS4_INK2_FP4_ROW = DS4_INK2_FP4_DATA + 8,
    DS4_INK2_PAGE = 2 * DS4_INK2_KV_BLOCK * DS4_INK2_FP4_ROW,
    DS4_INK2_BLOCK_BYTES = DS4_INK2_KV_HEADS * DS4_INK2_PAGE,
};

/* Routed-expert scratch for `rows` (prefill FP8 planes, decode items and the
 * per-layer device routing tables). */
uint64_t ds4_gpu_ink2_moe_bytes(uint32_t rows);
ds4_ink2_moe *ds4_gpu_ink2_moe_create(ds4_gvq_ctx *gvq, uint32_t rows);
void ds4_gpu_ink2_moe_free(ds4_ink2_moe *moe);
/* out = round(sum over the selected experts e of weights[e] * expert_e(input)),
 * the sum in F32 in ascending expert order (the reference's F32 index_add,
 * then one rounding). `matrices` holds the layer's gate[384], up[384],
 * down[384] GVQ indices; `weights` already carry the routed scale. */
int ds4_gpu_ink2_moe(ds4_ink2_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t rows);

/* Top-8 of sigmoid(logits) + bias (selection only; ties to the lower expert),
 * weights = the chosen sigmoids over their sum + 1e-20, times 8. `logits`:
 * F32 [rows][384]; `bias`: the layer's F32 mlp.gate.bias record. */
int ds4_gpu_ink2_route(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *logits, uint32_t bias, uint32_t rows);

/* Block attention residual of `rows` tokens: the stack of `nblocks` block
 * states (BF16 rows; block b of row r at (b * cap + r) * 4608) and the
 * partial state (F32 rows, values as they are), each normalized by its
 * unweighted RMS; query j of the F32 record `query` ([nq][4608]) attends
 * over them with scale 4608^-1/2 (F32 softmax); out[j] = rms(attention_j)
 * with BF16 weight norms[j], rounded to BF16. `query` DS4_GVQ_NO_LINK skips
 * the attention: out[0] = rms(partial) with norms[0]. */
int ds4_gpu_ink2_attn_res(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *out, const ds4_gpu_tensor *blocks,
    uint32_t nblocks, uint32_t cap, const ds4_gpu_tensor *partial, uint32_t query, uint32_t nq,
    const uint32_t *norms, uint32_t rows);

/* out = rms(embedding rows, embed_norm) rounded to BF16 (BF16 input rows). */
int ds4_gpu_ink2_embed(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *rows_bf16, uint32_t norm,
    uint32_t rows);

/* Block state `slot` of rows 0..rows-1 = x (F32 rows of BF16 values). */
int ds4_gpu_ink2_push(ds4_gpu_tensor *blocks, uint32_t slot, uint32_t cap, const ds4_gpu_tensor *x, uint32_t rows);

/* K and V of rows at pos..: the 4-tap causal convolution of the projections
 * kin/vin (1,024 wide) over their 3-row histories, then K = L2 clip
 * (22.627) and V = unweighted RMS per head, rounded to BF16 and stored at
 * cache row (pos + r) % cap (sliding) or pos + r (full). Sliding layers
 * (`dyn_k` NULL): taps are the layer's [1024][1][4] weights (taps[0] K,
 * taps[1] V), y = (conv + x) / sqrt(5). Full layers: taps = static
 * [4][1024] (tap = backward distance) + the per-row, per-KV-head dynamic
 * taps dyn_k/dyn_v ([rows][8 x 4], BF16 values), y = (conv + x) / 3. The
 * histories then hold the last three rows. */
int ds4_gpu_ink2_kv(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cache, uint32_t cap, uint32_t format,
    const ds4_gpu_tensor *kin, const ds4_gpu_tensor *vin, ds4_gpu_tensor *hist_k, ds4_gpu_tensor *hist_v,
    const uint32_t *taps, const ds4_gpu_tensor *dyn_k, const ds4_gpu_tensor *dyn_v, uint32_t rows, uint32_t pos);

/* In place: q = scaled L2 clip (scale [48][128], 22.627) and r = scaled L2
 * clip (scale [48][16], 8), rounded to BF16; full layers (`global`) then
 * multiply both by 1 + 0.1 ln(max((pos + 1) / 128000, 1)), rounded to BF16.
 * rel[row][h][d] = BF16(r[h] . proj[:, d]) for d < 1024 (BF16 rows). */
int ds4_gpu_ink2_qr(ds4_gvq_ctx *ctx, ds4_gpu_tensor *q, ds4_gpu_tensor *r, ds4_gpu_tensor *rel,
    uint32_t q_scale, uint32_t r_scale, uint32_t proj, uint32_t rows, uint32_t pos, int global);

/* Attention of rows at pos.. over cache rows [max(0, p - window + 1), p]
 * (window 0: every row from 0), 48 query heads, KV head h / 6: scores
 * BF16(BF16(q . k) / sqrt(128)) + rel[d] for d < 1024, the row's sink logit
 * per head in the softmax denominator, then out = BF16(BF16(attention) *
 * sigmoid(gate)) (F32 rows). `partials`: decode scratch of
 * ds4_gpu_ink2_attn_bytes(). */
uint64_t ds4_gpu_ink2_attn_bytes(void);
int ds4_gpu_ink2_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *rel,
    const ds4_gpu_tensor *sinks, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *cache, uint32_t cap,
    uint32_t window, uint32_t format, ds4_gpu_tensor *partials, uint32_t rows, uint32_t pos);

/* out = y + residual (F32, not rounded) or, with `round_out`, BF16(y +
 * residual); y = BF16((conv4(x) + x) / sqrt(5)) of the 4,608-wide rows x
 * over their 3-row history (taps: the [4608][1][4] record), which then holds
 * the last three rows. residual NULL: out = y. */
int ds4_gpu_ink2_conv(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x, ds4_gpu_tensor *hist,
    uint32_t taps, const ds4_gpu_tensor *residual, uint32_t rows, int round_out);

/* logits = 30 tanh(z / 30) over the 200,058 tokenizer rows, -inf past them. */
int ds4_gpu_ink2_logits(ds4_gpu_tensor *logits, uint32_t rows);

#ifdef __cplusplus
}
#endif
#endif
