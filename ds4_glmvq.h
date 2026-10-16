#ifndef DS4_GLMVQ_H
#define DS4_GLMVQ_H

#include <stdint.h>

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_gvq_model ds4_gvq_model;
typedef struct ds4_gvq_ctx ds4_gvq_ctx;

enum { DS4_GVQ_NO_LINK = UINT32_MAX, DS4_GVQ_CTX_MAX = 8192 };
typedef enum { DS4_GVQ_BF16, DS4_GVQ_F32 } ds4_gvq_math;
typedef enum { DS4_GVQ_RMS, DS4_GVQ_GATED_RMS, DS4_GVQ_PLAIN_RMS } ds4_gvq_norm;

/* Native-only pointer plan. Rust has validated payloads and links; this
 * table names source intervals and never exposes device handles to hosts. */
typedef struct {
    uint64_t offset, bytes;
    uint32_t type, rows, cols, scale, book;
} ds4_gvq_record;

ds4_gvq_model *ds4_gpu_gvq_bind(const void *map, uint64_t size,
    const ds4_gvq_record *records, uint32_t count);
void ds4_gpu_gvq_unbind(ds4_gvq_model *model);
uint64_t ds4_gpu_gvq_workspace(uint32_t rows);
uint64_t ds4_gpu_gvq_linear_bytes(uint32_t rows, uint32_t cols,
    uint32_t tokens, uint32_t type, ds4_gvq_math math);
ds4_gvq_ctx *ds4_gpu_gvq_create(const ds4_gvq_model *model, uint64_t scratch, uint32_t rows);
void ds4_gpu_gvq_free(ds4_gvq_ctx *ctx);
int ds4_gpu_gvq_linear(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_math math);
int ds4_gpu_gvq_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
    const ds4_gpu_tensor *input, const ds4_gpu_tensor *gate, uint32_t weight,
    uint32_t rows, uint32_t width, float eps, ds4_gvq_norm kind);
/* With `norm`, also writes the ordinary RMS norm of `out` (bit-identical
 * to a separate ds4_gpu_gvq_rms). */
int ds4_gpu_gvq_split(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *split,
    const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual,
    uint32_t scale, uint32_t base, uint32_t rows, uint32_t width,
    uint32_t iterations, float eps, ds4_gpu_tensor *norm, uint32_t norm_weight, float norm_eps);
/* mHC RMS + F32 mapping dots for 1-4 rows; 0 when the shape needs the fallback. */
int ds4_gpu_gvq_hc_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mix,
    const ds4_gpu_tensor *residual, uint32_t weight, uint32_t width, float eps, uint32_t rows);
int ds4_gpu_gvq_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *branch,
    const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split, uint32_t rows, uint32_t width);
int ds4_gpu_gvq_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint64_t count, float limit);
int ds4_gpu_gvq_round(ds4_gpu_tensor *values, uint64_t count);
int ds4_gpu_gvq_control(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight, uint64_t count);
int ds4_gpu_gvq_unpack(ds4_gpu_tensor *out, const ds4_gpu_tensor *cache, uint64_t count);
int ds4_gpu_gvq_store(ds4_gpu_tensor *cache, const ds4_gpu_tensor *input,
    uint32_t rows, uint32_t first, uint32_t capacity, uint32_t width);
int ds4_gpu_gvq_positions(ds4_gpu_tensor *out, uint32_t first, uint32_t rows);
int ds4_gpu_gvq_router(ds4_gvq_ctx *ctx, ds4_gpu_tensor *selected,
    ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, uint32_t bias,
    uint32_t rows, float scale);
uint64_t ds4_gpu_gvq_index_bytes(uint32_t rows, uint32_t pools);
int ds4_gpu_gvq_pool(ds4_gvq_ctx *ctx, ds4_gpu_tensor *cache,
    ds4_gpu_tensor *tail_k, ds4_gpu_tensor *tail_gate,
    const ds4_gpu_tensor *raw, const ds4_gpu_tensor *gate,
    uint32_t norm, uint32_t bias, uint32_t ape,
    uint32_t first, uint32_t rows, uint32_t capacity);
int ds4_gpu_gvq_index(ds4_gvq_ctx *ctx, ds4_gpu_tensor *scores,
    const ds4_gpu_tensor *query, const ds4_gpu_tensor *weights,
    const ds4_gpu_tensor *cache, uint32_t pools, uint32_t rows, uint32_t first);
uint64_t ds4_gpu_gvq_attn_bytes(uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim);
int ds4_gpu_gvq_attn(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
    const ds4_gpu_tensor *query, const ds4_gpu_tensor *kv,
    const ds4_gpu_tensor *positions, const ds4_gpu_tensor *selected,
    uint32_t stride, uint32_t rows, uint32_t keys, uint32_t heads, uint32_t dim);
/* Expand latent KV and attend over its causal prefix. Temporaries retain
 * their F32 capacity for the format/alignment fallback. */
int ds4_gpu_gvq_dsa(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
    const ds4_gpu_tensor *query, const ds4_gpu_tensor *cache, uint32_t weight,
    ds4_gpu_tensor *latent, ds4_gpu_tensor *kv, ds4_gpu_tensor *positions,
    uint32_t first, uint32_t rows, uint32_t heads, uint32_t dim,
    const ds4_gpu_tensor *selected, uint32_t stride);
int ds4_gpu_gvq_moe(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
    ds4_gpu_tensor *gathered, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
    ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
    const uint32_t *matrices, uint32_t rows, float clamp);
/* Chunked prefill planes for `rows` tokens; 0 when rows use the recurrence. */
uint64_t ds4_gpu_gvq_kda_bytes(uint32_t rows, uint32_t heads);
/* `snap` (optional) receives the state and tails after each row but the
 * last, one `stride`-byte slot per row, laid out like the state pool. */
int ds4_gpu_gvq_kda(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *state,
    ds4_gpu_tensor *q_tail, ds4_gpu_tensor *k_tail, ds4_gpu_tensor *v_tail,
    const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v,
    const ds4_gpu_tensor *forget, const ds4_gpu_tensor *beta,
    const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
    const ds4_gpu_tensor *v_weight, const ds4_gpu_tensor *a_log,
    const ds4_gpu_tensor *dt_bias, uint32_t rows, uint32_t heads, float lower_bound,
    ds4_gpu_tensor *snap, uint64_t stride);

#ifdef __cplusplus
}
#endif
#endif
