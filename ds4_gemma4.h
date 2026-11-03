#ifndef DS4_GEMMA4_H
#define DS4_GEMMA4_H

#include <stdint.h>

#include "ds4_gemma4_plan.h"
#include "ds4_glmvq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gemma-4-26B-A4B VQ device runtime. The weights stay in the GVQ pointer
 * plan (Rust validated every part); this adds what GLM's adapter does not
 * cover: two attention geometries with scale 1.0, the proportional rotary of
 * the full layers, tanh-GELU experts with a float32-weight top-8 sum, the
 * tanh-30 logit cap and the block's layer_scalar. */

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

/* Primitives as tensor calls (cuda/gemma4_primitives.cuh owns the kernels). */
int ds4_gpu_g4_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
    uint32_t weight, uint32_t rows, uint32_t width, float eps);
int ds4_gpu_g4_router_input(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, const ds4_gpu_tensor *in,
    uint32_t scale, uint32_t rows, uint32_t width, float eps, float root);
int ds4_gpu_g4_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *positions, const ds4_gpu_tensor *inv,
    uint32_t rows, uint32_t heads, uint32_t dim, uint32_t pairs);
int ds4_gpu_g4_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *logits, uint32_t per_expert, uint32_t rows);
int ds4_gpu_g4_gelu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
    uint64_t count);
int ds4_gpu_g4_embed_scale(ds4_gpu_tensor *x, uint64_t count);
int ds4_gpu_g4_softcap(ds4_gpu_tensor *logits, uint64_t count);
int ds4_gpu_g4_layer_scale(ds4_gpu_tensor *x, uint64_t count, float scalar);
int ds4_gpu_g4_attn(ds4_gpu_tensor *out, const ds4_gpu_tensor *query, const ds4_gpu_tensor *key,
    const ds4_gpu_tensor *value, const ds4_gpu_tensor *positions, uint32_t rows, uint32_t first,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t window, uint32_t ring,
    uint32_t keys);

/* Sliding layers keep window + the widest block one forward writes (a ring):
 * that block may overwrite the slots it writes, and window + block keeps those
 * slots older than the oldest key the first row of the block still reads. The
 * block is the prefill chunk or one draft round's verify block, whichever is
 * wider (gemma4_token_rows); with a ring of only window + chunk, a verify
 * block wider than the chunk would clobber keys its own first rows read. A
 * full layer keeps the whole context. A context shorter than the ring is
 * clamped to ctx, the largest position the ring can ever hold, so the
 * allocator and the host quote (serving_host) price the same number. */
static inline uint32_t ds4_gemma4_kv_rows(unsigned layer, unsigned ctx, unsigned chunk) {
    if (gemma4_is_full(layer)) { return ctx; }
    const uint32_t rows = G4_WINDOW + gemma4_token_rows(chunk);
    return rows < ctx ? rows : ctx;
}

static inline uint64_t ds4_gemma4_kv_bytes(unsigned ctx, unsigned chunk) {
    uint64_t total = 0;
    for (unsigned il = 0; il < G4_LAYERS; il++) {
        const uint64_t row = (uint64_t)gemma4_kv_heads(il) * gemma4_head_dim(il) *
            2u * sizeof(uint16_t);
        total += row * ds4_gemma4_kv_rows(il, ctx, chunk);
    }
    return total;
}

#ifdef __cplusplus
}
#endif
#endif
