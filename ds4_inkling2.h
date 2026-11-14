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
/* Where a full layer's NVFP4 blocks live under the SSD tier: blocks [0,
 * resident) in the layer's store (block b at b x DS4_INK2_BLOCK_BYTES), the
 * frontier blocks [window_block0, window_block0 + window_blocks) in `window`
 * (pinned host memory the device reads in place). NULL: all in the store. */
typedef struct {
    ds4_gpu_tensor *window;
    uint32_t resident, window_block0, window_blocks;
} ds4_ink2_tier;

int ds4_gpu_ink2_kv(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cache, uint32_t cap, uint32_t format,
    const ds4_gpu_tensor *kin, const ds4_gpu_tensor *vin, ds4_gpu_tensor *hist_k, ds4_gpu_tensor *hist_v,
    const uint32_t *taps, const ds4_gpu_tensor *dyn_k, const ds4_gpu_tensor *dyn_v, uint32_t rows, uint32_t pos,
    const ds4_ink2_tier *tier);

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

/* A full layer's NVFP4 attention over the key segment [seg_lo, seg_hi)
 * (multiples of 128 except a last segment's end), its blocks in `store`
 * from block0 on. `mode` DS4_INK2_SEG_FIRST starts the per-(row, head)
 * state (m, l, acc[128], the sink as m = sink, l = 1), NEXT continues it,
 * LAST continues it and writes `out` as ds4_gpu_ink2_attn does. `state`:
 * ds4_gpu_ink2_state_bytes(rows); decode rows also use `partials`. */
enum { DS4_INK2_SEG_FIRST = 1, DS4_INK2_SEG_NEXT = 2, DS4_INK2_SEG_LAST = 3 };
uint64_t ds4_gpu_ink2_state_bytes(uint32_t rows);
int ds4_gpu_ink2_attn_seg(ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *rel, const ds4_gpu_tensor *sinks, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *store,
    uint32_t seg_lo, uint32_t seg_hi, uint32_t block0, ds4_gpu_tensor *partials, uint32_t rows, uint32_t pos,
    int mode);

/* Two stream marks for the SSD tier's buffers: mark records an event on the
 * current stream; mark_wait blocks the host until that event has run. */
int ds4_gpu_ink2_mark(uint32_t slot);
int ds4_gpu_ink2_mark_wait(uint32_t slot);

/* out = y + residual (F32, not rounded) or, with `round_out`, BF16(y +
 * residual); y = BF16((conv4(x) + x) / sqrt(5)) of the 4,608-wide rows x
 * over their 3-row history (taps: the [4608][1][4] record), which then holds
 * the last three rows. residual NULL: out = y. */
int ds4_gpu_ink2_conv(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *x, ds4_gpu_tensor *hist,
    uint32_t taps, const ds4_gpu_tensor *residual, uint32_t rows, int round_out);

/* logits = 30 tanh(z / 30) over the 200,058 tokenizer rows, -inf past them. */
int ds4_gpu_ink2_logits(ds4_gpu_tensor *logits, uint32_t rows);

/* DSpark drafter (the release's `drafter` part; TokenSpeed inkling2_mtp.py):
 * target taps (the partial state entering layers 5, 11, ..., 41 and the
 * normalized output), per-stage tap mixtures softmax(fuse_weight), five
 * pre-norm layers (0, 1, 3, 4 sliding over 1,031 keys, 2 global) over an
 * 8-row non-causal block [anchor, mask x 7] and BF16 context rings, the
 * target head on rows 1..7 and the rnn_hidden head. */
#define DS4_INK2_DRAFT_THETA 1000000.0f
enum {
    DS4_INK2_DRAFT_LAYERS = 5, DS4_INK2_DRAFT_BLOCK = 8, DS4_INK2_DRAFT_TAPS = 8, DS4_INK2_DRAFT_FF = 18432,
    DS4_INK2_DRAFT_WINDOW = 1024 + DS4_INK2_DRAFT_BLOCK - 1, DS4_INK2_RANK = 256,
    DS4_INK2_RNN_IN = 2 * DS4_INK2_RANK + 2 * DS4_INK2_HIDDEN,
};
int ds4_gpu_ink2_draft_tap(ds4_gpu_tensor *taps, const ds4_gpu_tensor *x, uint32_t tap, uint32_t rows);
/* out = norm(BF16(sum_j alpha[stage][j] tap_j)) of tap rows first.. ;
 * `alpha`: F32 [6][8]. */
int ds4_gpu_ink2_draft_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *taps,
    const ds4_gpu_tensor *alpha, uint32_t stage, uint32_t norm, uint32_t first, uint32_t rows);
/* norm_out = norm(h + residual) from the F32 sum, residual = BF16(h + residual). */
int ds4_gpu_ink2_draft_add_norm(ds4_gvq_ctx *ctx, ds4_gpu_tensor *norm_out, ds4_gpu_tensor *residual,
    const ds4_gpu_tensor *h, uint32_t norm, uint32_t rows);
/* Per head: BF16 RMS norm (weight `norm`), then NeoX RoPE at pos0 + row. */
int ds4_gpu_ink2_draft_qk(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t norm, uint32_t heads, uint32_t rows,
    uint32_t pos0);
int ds4_gpu_ink2_draft_ring(ds4_gpu_tensor *ring_k, ds4_gpu_tensor *ring_v, const ds4_gpu_tensor *k,
    const ds4_gpu_tensor *v, uint32_t rows, uint32_t pos, uint32_t ring);
/* Rows pos0.. attend ring positions [max(first, end - ring, window bound), end)
 * with end = pos0 + rows (non-causal over the block); `partials`: the
 * ds4_gpu_ink2_attn_bytes() scratch. */
int ds4_gpu_ink2_draft_attn(ds4_gpu_tensor *out, ds4_gpu_tensor *partials, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *ring_k, const ds4_gpu_tensor *ring_v, uint32_t ring, uint32_t rows, uint32_t pos0,
    uint32_t first, uint32_t window);
int ds4_gpu_ink2_draft_silu(ds4_gpu_tensor *out, const ds4_gpu_tensor *g, const ds4_gpu_tensor *u, uint64_t count);
int ds4_gpu_ink2_draft_mask(ds4_gvq_ctx *ctx, ds4_gpu_tensor *h, uint32_t mask, uint32_t rows);
int ds4_gpu_ink2_draft_markov_in(ds4_gpu_tensor *z, const ds4_gpu_tensor *hidden, const ds4_gpu_tensor *f_rnn,
    uint32_t steps);
/* Step `step` of the rnn_hidden head: logits of proposal row `step` from the
 * F32 base logits rows, the recurrence state and ids[step]. */
int ds4_gpu_ink2_draft_markov(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, ds4_gpu_tensor *state,
    ds4_gpu_tensor *latent, ds4_gpu_tensor *z, const ds4_gpu_tensor *invariant, const ds4_gpu_tensor *base,
    const ds4_gpu_tensor *fold_w, const ds4_gpu_tensor *fold_b, const ds4_gpu_tensor *ids, uint32_t proj,
    uint32_t bias, uint32_t w1, uint32_t step);
int ds4_gpu_ink2_draft_fold_x(ds4_gvq_ctx *ctx, ds4_gpu_tensor *x, uint32_t weight, uint32_t bias);
int ds4_gpu_ink2_draft_fold_pack(ds4_gpu_tensor *fold_w, ds4_gpu_tensor *fold_b, const ds4_gpu_tensor *head_rows);
int ds4_gpu_ink2_draft_copy(ds4_gpu_tensor *dst, uint64_t dst_off, const ds4_gpu_tensor *src, uint64_t count);
/* hist = the last three rows of [hist; x[0..rows)] (`width`-wide F32 rows). */
int ds4_gpu_ink2_history(ds4_gpu_tensor *hist, const ds4_gpu_tensor *x, uint32_t width, uint32_t rows);

/* Vision tower (ds4_inkling2_vision.inc): an image of rows x cols language
 * tokens (40 x 40 pixels each) is 2 rows x 2 cols ViT patches of 20 x 20;
 * the tower's sequence is [CLS, patches in 2 x 2 block order]. */
enum {
    DS4_INK2_VIT_DIM = 1024, DS4_INK2_VIT_HEADS = 16, DS4_INK2_VIT_HEAD = 64, DS4_INK2_VIT_PATCH = 1200,
    DS4_INK2_VIT_GRID = 32, DS4_INK2_IMAGE_VALUES = 40 * 40 * 3,
};
/* patches[4 t + 2 mh + mw] = BF16(pixels[t] at rows 20 mh.., columns 20 mw..),
 * (y, x, channel) order: the patch embedding's input. */
int ds4_gpu_ink2_vit_patches(ds4_gpu_tensor *patches, const ds4_gpu_tensor *pixels, uint32_t tokens);
/* Patches p0..p0 + count - 1 (emb holds their rows): x[1 + i] = BF16(emb[i - p0]
 * + BF16(the 32 x 32 `table` bilinearly resized to the 2 rows x 2 cols patch
 * grid at patch i)); from patch 0 also x[0] = BF16(cls[0] + cls[1]). */
int ds4_gpu_ink2_vit_embed(ds4_gpu_tensor *x, const ds4_gpu_tensor *emb, const ds4_gpu_tensor *table,
    const ds4_gpu_tensor *cls, uint32_t rows, uint32_t cols, uint32_t p0, uint32_t count);
/* Rows r0..r0 + m - 1 of packed F32 [q | k | v] (qkv holds those m rows):
 * axial 2-D RoPE (theta 1e4, pairwise) of q and k, patches at (row + 1,
 * col + 1), CLS at (0, 0); q, k and v land as BF16 in qkv16's rows r0... */
int ds4_gpu_ink2_vit_rope(ds4_gpu_tensor *qkv16, const ds4_gpu_tensor *qkv, uint32_t rows, uint32_t cols,
    uint32_t r0, uint32_t m);
/* Query rows of one dense-attention tile over n rows (V4.1's S/P budget). */
uint32_t ds4_gpu_ink2_vit_attn_rows(uint32_t n);
/* Dense attention (scale 1/8, 16 heads of 64) of query rows q0..q0 + m - 1
 * over all n rows of BF16 [q | k | v]: o [m][1024] F32, BF16 values. */
int ds4_gpu_ink2_vit_attn(ds4_gpu_tensor *o, const ds4_gpu_tensor *qkv16, uint32_t n, uint32_t q0, uint32_t m);
/* Layer scale: x = BF16(x + BF16(scale * y)) over `n` rows of `dim`. */
int ds4_gpu_ink2_vit_scale_add(ds4_gpu_tensor *x, const ds4_gpu_tensor *y, const ds4_gpu_tensor *scale, uint32_t n,
    uint32_t dim);

#ifdef __cplusplus
}
#endif
#endif
