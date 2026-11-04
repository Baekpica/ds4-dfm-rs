#ifndef DS4_GEMMA4_VQ_H
#define DS4_GEMMA4_VQ_H

#include <stdint.h>

#include "ds4_gemma4_plan.h"
#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma-4-26B-A4B VQ device runtime: Gemma's own entry points over the
 * model-agnostic GVQ pointer plan (Rust validated every part). ds4_glmvq.h
 * is the shared VQ runtime's type and API header (it carries no GLM model
 * code); the Gemma-specific kernels are cuda/gemma_vq_decode.cuh,
 * cuda/gemma4_attn.cuh, cuda/gemma4_attn_fp8.cuh and
 * cuda/gemma4_primitives.cuh. This file is the one Gemma optimizes against,
 * so a change here never edits a sibling family's header. */

typedef struct ds4_g4_moe ds4_g4_moe;

/* Routed-expert scratch for `chunk` tokens (the engine's prefill chunk). */
uint64_t ds4_gpu_g4_moe_bytes(uint32_t chunk);
ds4_g4_moe *ds4_gpu_g4_moe_create(ds4_gvq_ctx *gvq, uint32_t chunk);
void ds4_gpu_g4_moe_free(ds4_g4_moe *moe);
/* `matrices` holds gate[128], up[128], down[128] GVQ indices. The routed sum
 * accumulates into `out` in bf16, one rounding per expert, as the reference's
 * index_add_ does. */
int ds4_gpu_g4_moe_run(ds4_g4_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
    const uint32_t *matrices, uint32_t rows);
/* The same run with the route ids already on the host (the SSD stage reads
 * them once); `routes` is rows * G4_USED expert indices. */
int ds4_gpu_g4_moe_run_ids(ds4_g4_moe *moe, ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *weights, const uint32_t *matrices,
    const int32_t *routes, uint32_t rows);
/* Cache-miss fallback (DS4_GEMMA4_MOE_MISS=cpu): one expert's down output
 * computed on the host, and the ordered add of that contribution. */
int ds4_gpu_g4_cpu_moe_expert(ds4_gvq_ctx *ctx, const uint32_t *matrices,
    uint32_t expert, const float *x, float *out);
int ds4_gpu_g4_moe_add_one(ds4_gpu_tensor *out, const ds4_gpu_tensor *d,
    float weight, uint32_t count);
/* One-token persistent layer-book decode (cuda/gemma4_moe_persist.cuh): the
 * gate/up and down passes each stage the layer's book in shared memory and
 * run every routed expert in one launch, and the route order is sorted from
 * the device's `selected` row, so no route is read back per layer.
 * `matrices` is the layer's gate[128], up[128], down[128] table and `layer`
 * selects its cached device copy. Returns 0 for a geometry or a layer the
 * table cannot describe; the caller keeps the per-expert loop. */
int ds4_gpu_g4_moe_run_persist(ds4_g4_moe *moe, ds4_gpu_tensor *out,
    const ds4_gpu_tensor *input, const ds4_gpu_tensor *selected,
    const ds4_gpu_tensor *weights, const uint32_t *matrices, uint32_t layer);

/* One-token Q4_K decode (cuda/gemma4_q4_decode.cuh): `count` matrices sharing
 * one input, laid out matrix by matrix in one launch with the weights staged
 * in shared, and the dense gate/up pair fused with the gelu product. Both
 * return 0 for any geometry the Gemma kernels do not take; the shared
 * dispatcher stays the fallback. */
int ds4_gpu_g4_q4_many(ds4_gvq_ctx *gvq, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows);
int ds4_gpu_g4_q4_glu(ds4_gvq_ctx *gvq, ds4_gpu_tensor *out, uint32_t gate,
    uint32_t up, const ds4_gpu_tensor *input, uint32_t rows);

/* Gemma-only SSD rebind: point the plan's host source entry for `index` at a
 * device slot holding a byte copy of the VQ1 record. `header` is the first
 * 128 bytes of that record (the slot copy and the header share the record's
 * own layout). `res_index`/`res_slot`/`res_header` describe the linked GVR8
 * residual record, or DS4_GVQ_NO_LINK when the artifact has no residual
 * stage. The book and residual book stay resident. */
int ds4_gpu_g4_stream_rebind(ds4_gvq_model *model, uint32_t index,
    const ds4_gpu_tensor *slot, uint64_t offset, uint64_t bytes, const void *header,
    uint32_t res_index, const ds4_gpu_tensor *res_slot, uint64_t res_offset,
    uint64_t res_bytes, const void *res_header);

/* KV store into bf16 slots; `ring` 0 means slot = absolute position. */
int ds4_gpu_g4_kv_store(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
    const ds4_gpu_tensor *positions, uint32_t rows, uint32_t width, uint32_t ring,
    uint32_t capacity);

/* Full-layer KV in E4M3: K codes, V codes, then one f16 scale per KV head
 * for each (the row the Solar FP8 store writes, ds4_cuda.cu:44407-44480).
 * `width` is the KV width (kv_heads * head_dim); `ring` 0 means slot =
 * absolute position. The option is off unless DS4_GEMMA4_FULL_KV=fp8; the
 * 25 sliding layers keep bf16 rows either way. */
int ds4_gpu_g4_kv_store_fp8(ds4_gpu_tensor *dst, const ds4_gpu_tensor *k,
    const ds4_gpu_tensor *v, const ds4_gpu_tensor *positions, uint32_t rows,
    uint32_t width, uint32_t kv_heads, uint32_t head_dim, uint32_t ring,
    uint32_t capacity);

/* Attention over the combined FP8 full-layer rows; the same query layout
 * and causal window as ds4_gpu_g4_attn, with the per-head K/V scales applied
 * to the score and the probability (ds4_beam_gpu.cuh:795-880). */
int ds4_gpu_g4_attn_fp8(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
    const ds4_gpu_tensor *kv, const ds4_gpu_tensor *positions, uint32_t rows,
    uint32_t first, uint32_t heads, uint32_t kv_heads, uint32_t dim,
    uint32_t window, uint32_t ring, uint32_t keys, uint64_t row_bytes);

/* Decode-only grouped-KV attention: one CTA per KV head, its query heads
 * sharing one shared-memory K/V tile (cuda/gemma4_attn_fast.cuh). The same
 * arithmetic as ds4_gpu_g4_attn, so the oracle stays the reference. */
int ds4_gpu_g4_attn_fast(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
    const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
    const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
    uint32_t ring, uint32_t keys);

/* Prefill grouped-KV attention: one CTA per (row tile, KV head), the tile
 * staged once for the block's warps (cuda/gemma4_attn_tile.cuh). The same
 * arithmetic as ds4_gpu_g4_attn, so the oracle stays the reference. */
int ds4_gpu_g4_attn_tile(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
    const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
    const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
    uint32_t ring, uint32_t keys);

/* Tensor-core prefill attention (cuda/gemma4_attn_mma.cuh): the sliding
 * <256, 2> and full <512, 8> specializations, rows >= 16 and keys == 0.
 * A parity-gated path: the QK dot, the 32-key softmax grouping and the
 * bf16 P differ from the tile kernel's arithmetic. */
int ds4_gpu_g4_attn_mma(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
    const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
    const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
    uint32_t ring, uint32_t keys);

/* Split-K decode attention (cuda/gemma4_attn_split.cuh): one row, the key
 * range divided across CTAs and joined by a combine kernel. The per-key
 * arithmetic is the fast kernel's; the softmax combine order differs. */
int ds4_gpu_g4_attn_split(ds4_gpu_tensor *out, const ds4_gpu_tensor *query,
    const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
    const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window,
    uint32_t ring, uint32_t keys);

/* Primitives as tensor calls (cuda/gemma4_primitives.cuh owns the kernels). */
int ds4_gpu_g4_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
    uint32_t weight, uint32_t rows, uint32_t width, float eps);
int ds4_gpu_g4_router_input(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
    uint32_t scale, uint32_t rows, uint32_t width, float eps, float root);
int ds4_gpu_g4_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions, const ds4_gpu_tensor *inv,
    uint32_t rows, uint32_t heads, uint32_t dim, uint32_t pairs);
/* The same norm and rope in one block per (row, head): the row is staged once
 * and rotated in place, with the two kernels' arithmetic unchanged. */
int ds4_gpu_g4_rms_rope(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
    uint32_t weight, const ds4_gpu_tensor *positions, const ds4_gpu_tensor *inv,
    uint32_t rows, uint32_t heads, uint32_t dim, uint32_t pairs, float eps);
int ds4_gpu_g4_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *logits, uint32_t per_expert, uint32_t rows);
/* The router projection through cuda/gemma4_router.cuh for one token; 0 for
 * any other shape or record, so the shared linear stays the fallback. */
int ds4_gpu_g4_router_logits(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, uint32_t index,
    const ds4_gpu_tensor *input, uint32_t rows);
int ds4_gpu_g4_gelu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
    uint64_t count);
int ds4_gpu_g4_embed_scale(ds4_gpu_tensor *x, uint64_t count);
int ds4_gpu_g4_softcap(ds4_gpu_tensor *logits, uint64_t count);
/* The tied BF16 vocabulary head through cuda/gemma4_head.cuh, softcap fused;
 * 0 when the record is not a BF16 table of the expected width. */
int ds4_gpu_g4_head(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, uint32_t index,
    const ds4_gpu_tensor *input, float cap);
/* The verify head: rows (1..G4_HEAD_ROWS) final-norm rows against the tied
 * BF16 table in one pass, row t at logits[t * vocab + row], softcap fused;
 * 0 when the record is not a BF16 table of the expected width. */
int ds4_gpu_g4_head_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *logits, uint32_t index,
    const ds4_gpu_tensor *input, uint32_t rows, float cap);
int ds4_gpu_g4_layer_scale(ds4_gpu_tensor *x, uint64_t count, float scalar);
int ds4_gpu_g4_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *query, const ds4_gpu_tensor *key,
    const ds4_gpu_tensor *value, const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window, uint32_t ring,
    uint32_t keys);

#ifdef __cplusplus
}
#endif

#endif
