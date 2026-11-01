#ifndef DS4_GLMVQ_H
#define DS4_GLMVQ_H

#include <stdint.h>

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_gvq_model ds4_gvq_model;
typedef struct ds4_gvq_ctx ds4_gvq_ctx;

/* The source model's native context (max_position_embeddings 1,048,576):
 * a VQ artifact changes weights, not the context contract. Flash DSA
 * attention and fused index scores keep no per-key scratch; latent, pool and
 * draft caches grow linearly with the session context. */
enum { DS4_GVQ_NO_LINK = UINT32_MAX, DS4_GVQ_CTX_MAX = 1048576 };
/* TF32: F32 math on tensor cores for operands that hold BF16 values, which
 * TF32 represents exactly: products stay exact, accumulation order moves. */
typedef enum { DS4_GVQ_BF16, DS4_GVQ_F32, DS4_GVQ_TF32 } ds4_gvq_math;
typedef enum { DS4_GVQ_RMS, DS4_GVQ_GATED_RMS, DS4_GVQ_PLAIN_RMS } ds4_gvq_norm;
/* Activation rows of BF16-rounded values: F32, or BF16 packed into the
 * first half of the same buffer (mHC residuals from the fused prefill
 * expand, KDA gate projections). */
typedef enum { DS4_GVQ_ROWS_F32, DS4_GVQ_ROWS_BF16 } ds4_gvq_rows;

/* Native-only pointer plan. Rust has validated payloads and links; this
 * table names source intervals and never exposes device handles to hosts. */
/* `res` links a GVQ1 matrix to its optional GVR8 residual record (container
 * v2); the residual record's `book` is its K=256 residual book. */
typedef struct {
    uint64_t offset, bytes;
    uint32_t type, rows, cols, scale, book, res;
} ds4_gvq_record;

ds4_gvq_model *ds4_gpu_gvq_bind(const void *map, uint64_t size,
    const ds4_gvq_record *records, uint32_t count);
void ds4_gpu_gvq_unbind(ds4_gvq_model *model);
uint64_t ds4_gpu_gvq_workspace(uint32_t rows);
uint64_t ds4_gpu_gvq_linear_bytes(uint32_t rows, uint32_t cols,
    uint32_t tokens, uint32_t type, ds4_gvq_math math);
ds4_gvq_ctx *ds4_gpu_gvq_create(const ds4_gvq_model *model, uint64_t scratch, uint32_t rows);
/* Linear, KDA, norm and copy kernels only: no GLM routed-expert scratch
 * (Ling-3.1 runs its own expert runtime). */
uint64_t ds4_gpu_gvq_workspace_linear(uint32_t rows);
ds4_gvq_ctx *ds4_gpu_gvq_create_linear(const ds4_gvq_model *model, uint64_t scratch, uint32_t rows);
void ds4_gpu_gvq_free(ds4_gvq_ctx *ctx);
/* Up to four matrices over one input; prefill shares its Q8_1/BF16 copy. */
int ds4_gpu_gvq_linear_many(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_math math);
/* Up to eight tokens through up to four Q4_K matrices of one input in one
 * launch (BF16-rounded F32 outputs), each token as a one-token call; 0
 * (outputs untouched) for other weights. */
/* ds4_gpu_gvq_q4_decode_many only where every output equals its own
 * ds4_gpu_gvq_linear bit for bit (same STAGE kernel and split); 0 otherwise. */
int ds4_gpu_gvq_q4_decode_group(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t tokens);
int ds4_gpu_gvq_q4_decode_many(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t tokens);
/* Whether q4p (prefill Q4_K on FP16 tensor cores) runs these matrices at
 * `rows`; then ds4_gpu_gvq_q4p_prepared runs them over an input the caller
 * prepared as q4p's own pass would (FP16 rows in the (0, 2, 1, 3) quad order,
 * a 2^(e-6) scale per row). 0 when q4p does not take them. */
int ds4_gpu_gvq_q4p_takes(ds4_gvq_ctx *ctx, const uint32_t *weights, uint32_t count, uint32_t rows);
int ds4_gpu_gvq_q4p_prepared(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs, const uint32_t *weights,
    uint32_t count, const ds4_gpu_tensor *xh, const ds4_gpu_tensor *xscale, uint32_t rows);
/* ds4_gpu_gvq_q4p_prepared of one matrix into BF16 rows (each the F32 row
 * value rounded to nearest). */
int ds4_gpu_gvq_q4p_prepared_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *xh, const ds4_gpu_tensor *xscale, uint32_t rows);
/* Up to eight tokens of one Q4_K matrix on the one-token SIMT kernels (no
 * MMA verify path); 0 for other types and widths. */
int ds4_gpu_gvq_q4_decode_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t index,
    const ds4_gpu_tensor *input, uint32_t rows);
/* Prefill Q4_K matrices over one input as BF16 rows; 0 (outputs untouched)
 * when the rows or weights keep F32 outputs. */
int ds4_gpu_gvq_linear_many_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows);
/* Prefill FP8 matrices over one input as BF16 rows (the BF16 GEMM results
 * the F32 path widens); 0 (outputs untouched) at other widths or weights. */
int ds4_gpu_gvq_fp8_many_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows);
int ds4_gpu_gvq_linear(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_math math);
/* Whether ds4_gpu_gvq_linear_rows reads BF16 input rows of `weight` at
 * these rows (prefill Q4_K), so their producer can store BF16 rows. */
int ds4_gpu_gvq_bf16_input(ds4_gvq_ctx *ctx, uint32_t weight, uint32_t rows);
/* ds4_gpu_gvq_linear (BF16 math) of F32 or BF16 input rows (BF16 rows:
 * prefill Q4_K through q4p, or BF16 weights above decode widths). */
int ds4_gpu_gvq_linear_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *input, ds4_gvq_rows in_rows, uint32_t rows);
/* ds4_gpu_gvq_linear_rows into BF16 output rows (each the F32 value rounded
 * to nearest); 0 when q4p does not take the weight at `rows`. */
int ds4_gpu_gvq_linear_rows_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *input, ds4_gvq_rows in_rows, uint32_t rows);
/* A BF16 weight times BF16 input rows into F32 rows (F32 accumulation, no
 * output rounding); 0 for other weights and decode widths. */
int ds4_gpu_gvq_linear_rows_f32(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *input, uint32_t rows);
/* Prefill Q4_K matrices over BF16 input rows into F32 or BF16 rows; 0 when
 * q4p does not take every weight at `rows`. */
int ds4_gpu_gvq_linear_many_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *const *outs,
    const uint32_t *weights, uint32_t count, const ds4_gpu_tensor *input, uint32_t rows, ds4_gvq_rows out_rows);
/* Q4_K gate and up of one input with the clamped SwiGLU in one pass, at
 * decode and verify widths; 0 when the width or weights need the
 * separate projections. */
int ds4_gpu_gvq_gate_up(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mid, uint32_t gate, uint32_t up,
    const ds4_gpu_tensor *input, uint32_t rows, float limit);
/* BF16-weight prefill projection leaving BF16 rows in `out`; 0 when the
 * shape takes ds4_gpu_gvq_linear instead. */
int ds4_gpu_gvq_linear_bf16(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, uint32_t weight,
    const ds4_gpu_tensor *input, uint32_t rows);
int ds4_gpu_gvq_rms(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
    const ds4_gpu_tensor *input, const ds4_gpu_tensor *gate, uint32_t weight,
    uint32_t rows, uint32_t width, float eps, ds4_gvq_norm kind);
/* With `norm`, also writes the ordinary RMS norm of `out` (bit-identical
 * to a separate ds4_gpu_gvq_rms). */
int ds4_gpu_gvq_split(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *split,
    const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual, ds4_gvq_rows format,
    uint32_t scale, uint32_t base, uint32_t rows, uint32_t width,
    uint32_t iterations, float eps, ds4_gpu_tensor *norm, uint32_t norm_weight, float norm_eps);
/* mHC RMS + F32 mapping dots; 0 when the shape needs the fallback. */
int ds4_gpu_gvq_hc_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *mix,
    const ds4_gpu_tensor *residual, uint32_t weight, uint32_t width, float eps, uint32_t rows);
/* Prefill expand fused with the next mHC mix, writing BF16 rows; 0 when
 * rows or shape need the separate kernels. */
int ds4_gpu_gvq_expand_mix(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *mix,
    const ds4_gpu_tensor *branch, const ds4_gpu_tensor *residual, ds4_gvq_rows format,
    const ds4_gpu_tensor *split, uint32_t weight, uint32_t rows, uint32_t width, float eps);
int ds4_gpu_gvq_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *branch,
    const ds4_gpu_tensor *residual, ds4_gvq_rows format, const ds4_gpu_tensor *split,
    uint32_t rows, uint32_t width);
int ds4_gpu_gvq_swiglu(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *up, uint32_t rows, uint32_t width, float limit, ds4_gvq_rows out_rows);
int ds4_gpu_gvq_round(ds4_gpu_tensor *values, uint64_t count);
/* Stream-ordered device copy launched like the `rows`-wide kernels around
 * it; a driver memcpy between dependent launches stalled MTP prefill. */
int ds4_gpu_gvq_copy(ds4_gpu_tensor *dst, uint64_t dst_offset, const ds4_gpu_tensor *src,
    uint64_t src_offset, uint64_t bytes, uint32_t rows);
/* Rows of [left | right], each `width` floats: the MTP eh_proj input. */
int ds4_gpu_gvq_concat(ds4_gpu_tensor *out, const ds4_gpu_tensor *left,
    const ds4_gpu_tensor *right, uint32_t rows, uint32_t width);
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
/* Flash attention scratch for `rows` query rows. */
uint64_t ds4_gpu_gvq_flash_bytes(uint32_t rows);
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
/* ds4_gpu_gvq_dsa with BF16 query or output rows, which only the prefill
 * flash path takes (0 otherwise). */
int ds4_gpu_gvq_dsa_rows(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gvq_rows out_rows,
    const ds4_gpu_tensor *query, ds4_gvq_rows query_rows, const ds4_gpu_tensor *cache, uint32_t weight,
    ds4_gpu_tensor *latent, ds4_gpu_tensor *kv, ds4_gpu_tensor *positions,
    uint32_t first, uint32_t rows, uint32_t heads, uint32_t dim,
    const ds4_gpu_tensor *selected, uint32_t stride);
int ds4_gpu_gvq_moe(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out,
    ds4_gpu_tensor *gathered, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
    ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
    const uint32_t *matrices, uint32_t rows, float clamp, const ds4_gpu_tensor *shared);
/* Chunked prefill planes for `rows` tokens; 0 when rows use the recurrence. */
uint64_t ds4_gpu_gvq_kda_bytes(uint32_t rows, uint32_t heads);
/* The tensor-core path's chunk records alone (ds4_gpu_gvq_kda_gated). */
uint64_t ds4_gpu_gvq_kda_mma_bytes(uint32_t rows, uint32_t heads);
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
/* Whether ds4_gpu_gvq_kda_gated takes these prefill rows (not short
 * appends or DS4_GLMVQ_KDA_MMA=0). */
int ds4_gpu_gvq_kda_fused(ds4_gvq_ctx *ctx, uint32_t rows);
/* Prefill rows: KDA and the gated output norm in one call, writing the
 * normed heads to `out`. Returns -1 when the rows take the recurrence or
 * the separate norm. */
int ds4_gpu_gvq_kda_gated(ds4_gvq_ctx *ctx, ds4_gpu_tensor *out, ds4_gpu_tensor *state,
    ds4_gpu_tensor *q_tail, ds4_gpu_tensor *k_tail, ds4_gpu_tensor *v_tail,
    const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, ds4_gvq_rows qkv_rows,
    const ds4_gpu_tensor *forget, ds4_gvq_rows forget_rows, const ds4_gpu_tensor *beta,
    const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
    const ds4_gpu_tensor *v_weight, const ds4_gpu_tensor *a_log,
    const ds4_gpu_tensor *dt_bias, const ds4_gpu_tensor *gate, ds4_gvq_rows gate_rows, uint32_t norm,
    uint32_t rows, uint32_t heads, float lower_bound, float eps, ds4_gvq_rows out_rows);

#ifdef __cplusplus
}
#endif
#endif
